from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.data.preprocess import run_preprocessing
from src.explainability.reason_codes import describe_feature_value, to_readable_feature_name
from src.models.train_catboost import prepare_catboost_inputs
from src.utils.config import SETTINGS


# Only features that are realistically actionable in a performance-improvement setting.
DEFAULT_ACTIONABLE_FEATURES = [
    "EmpEnvironmentSatisfaction",
    "EmpLastSalaryHikePercent",
    "EmpWorkLifeBalance",
    "EmpJobInvolvement",
    "TrainingTimesLastYear",
]

# For upward counterfactuals, these features are assumed to improve when increased.
MONOTONIC_INCREASE_FEATURES = {
    "EmpEnvironmentSatisfaction",
    "EmpLastSalaryHikePercent",
    "EmpWorkLifeBalance",
    "EmpJobInvolvement",
    "TrainingTimesLastYear",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, pd.Series):
        return {str(k): to_jsonable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    return obj


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)

def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_catboost_model(catboost_dir: Path) -> CatBoostClassifier:
    model_path = catboost_dir / "catboost_model.cbm"
    if not model_path.exists():
        raise FileNotFoundError(
            f"CatBoost model not found: {model_path}\n"
            f"Please train the CatBoost model first."
        )

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def pick_sample_index(
    X_test_raw: pd.DataFrame,
    sample_index: Optional[int] = None,
    sample_position: Optional[int] = None,
) -> int:
    if sample_index is not None:
        if sample_index not in X_test_raw.index:
            raise ValueError(
                f"sample_index={sample_index} not found in test set index."
            )
        return int(sample_index)

    if sample_position is not None:
        if sample_position < 0 or sample_position >= len(X_test_raw):
            raise ValueError(
                f"sample_position={sample_position} out of range. "
                f"Valid range: 0 to {len(X_test_raw) - 1}"
            )
        return int(X_test_raw.index[sample_position])

    return int(X_test_raw.index[0])


def build_single_pool(
    sample_df: pd.DataFrame,
    target_series: Optional[pd.Series],
    drop_sensitive: bool,
) -> Tuple[pd.DataFrame, Pool]:
    """
    sample_df must already contain only model input columns
    """
    y_dummy = target_series if target_series is not None else pd.Series([0], index=sample_df.index)

    X_cb, _, feature_names, cat_feature_indices = prepare_catboost_inputs(
        X_train=sample_df,
        X_test=sample_df.copy(),
        drop_sensitive=drop_sensitive,
    )

    pool = Pool(
        data=X_cb,
        label=y_dummy,
        cat_features=cat_feature_indices,
        feature_names=feature_names,
    )
    return X_cb, pool


def predict_single_sample(
    model: CatBoostClassifier,
    sample_df: pd.DataFrame,
    drop_sensitive: bool,
) -> Dict[str, Any]:
    _, pool = build_single_pool(
        sample_df=sample_df,
        target_series=None,
        drop_sensitive=drop_sensitive,
    )

    pred_class = int(model.predict(pool).flatten()[0])
    proba = model.predict_proba(pool)[0]
    class_labels = [int(c) for c in model.classes_]
    proba_map = {int(c): float(p) for c, p in zip(class_labels, proba)}

    return {
        "predicted_class": pred_class,
        "probabilities": proba_map,
    }


def get_feature_scale_stats(
    X_train_raw: pd.DataFrame,
    actionable_features: List[str],
) -> Dict[str, float]:
    """
    Used for normalized change cost.
    """
    stats: Dict[str, float] = {}

    for feature in actionable_features:
        if feature not in X_train_raw.columns:
            continue

        series = pd.to_numeric(X_train_raw[feature], errors="coerce").dropna()
        if len(series) == 0:
            stats[feature] = 1.0
            continue

        std = float(series.std())
        rng = float(series.max() - series.min())

        scale = std if std > 0 else rng
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        stats[feature] = scale

    return stats


def is_change_allowed_for_upward_move(
    feature: str,
    old_value: Any,
    new_value: Any,
) -> bool:
    """
    For upward moves, monotonic-increase features should not decrease.
    """
    if feature not in MONOTONIC_INCREASE_FEATURES:
        return True

    try:
        old_num = float(old_value)
        new_num = float(new_value)
        return new_num >= old_num
    except Exception:
        return old_value == new_value


def build_modified_sample(
    original_row: pd.Series,
    prototype_row: pd.Series,
    actionable_features: List[str],
    target_higher_than_current: bool,
    max_features_changed: int,
) -> Optional[Tuple[pd.Series, List[Dict[str, Any]]]]:
    modified = original_row.copy()
    changes: List[Dict[str, Any]] = []

    for feature in actionable_features:
        if feature not in original_row.index or feature not in prototype_row.index:
            continue

        old_value = original_row[feature]
        new_value = prototype_row[feature]

        values_different = str(old_value) != str(new_value)

        if not values_different:
            continue

        if target_higher_than_current:
            if not is_change_allowed_for_upward_move(feature, old_value, new_value):
                continue

        modified[feature] = new_value

        changes.append(
            {
                "feature": feature,
                "old_value": old_value,
                "new_value": new_value,
            }
        )

    if len(changes) == 0:
        return None

    if len(changes) > max_features_changed:
        return None

    return modified, changes


def compute_change_cost(
    changes: List[Dict[str, Any]],
    scale_stats: Dict[str, float],
) -> float:
    total_cost = 0.0

    for change in changes:
        feature = change["feature"]
        old_value = change["old_value"]
        new_value = change["new_value"]

        try:
            old_num = float(old_value)
            new_num = float(new_value)
            scale = scale_stats.get(feature, 1.0)
            cost = abs(new_num - old_num) / max(scale, 1e-8)
        except Exception:
            cost = 1.0 if str(old_value) != str(new_value) else 0.0

        total_cost += float(cost)

    # Small penalty for number of changes
    total_cost += 0.15 * len(changes)

    return float(total_cost)


def convert_changes_to_actions(
    changes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []

    for change in changes:
        feature = change["feature"]
        old_value = change["old_value"]
        new_value = change["new_value"]

        feature_label = to_readable_feature_name(feature)
        old_desc = describe_feature_value(feature, old_value)
        new_desc = describe_feature_value(feature, new_value)

        sentence = (
            f"{feature_label.capitalize()} değişkenini {old_desc} düzeyinden {new_desc} düzeyine taşımak önerilmektedir."
        )

        actions.append(
            {
                "feature": feature,
                "feature_label_tr": feature_label,
                "old_value": old_value,
                "new_value": new_value,
                "old_value_description": old_desc,
                "new_value_description": new_desc,
                "action_sentence_tr": sentence,
            }
        )

    return actions


def generate_counterfactual_candidates(
    model: CatBoostClassifier,
    original_sample_df: pd.DataFrame,
    original_prediction: int,
    desired_class: int,
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    drop_sensitive: bool,
    actionable_features: List[str],
    max_features_changed: int = 3,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    target_higher_than_current = desired_class > original_prediction

    scale_stats = get_feature_scale_stats(X_train_raw, actionable_features)

    candidate_pool = X_train_raw.loc[y_train[y_train == desired_class].index].copy()

    candidates: List[Dict[str, Any]] = []

    original_row = original_sample_df.iloc[0]

    for candidate_index, prototype_row in candidate_pool.iterrows():
        built = build_modified_sample(
            original_row=original_row,
            prototype_row=prototype_row,
            actionable_features=actionable_features,
            target_higher_than_current=target_higher_than_current,
            max_features_changed=max_features_changed,
        )

        if built is None:
            continue

        modified_row, changes = built
        modified_df = pd.DataFrame([modified_row], columns=original_sample_df.columns)

        pred = predict_single_sample(
            model=model,
            sample_df=modified_df,
            drop_sensitive=drop_sensitive,
        )
        predicted_class = int(pred["predicted_class"])
        desired_prob = float(pred["probabilities"].get(desired_class, 0.0))

        if predicted_class != desired_class:
            continue

        cost = compute_change_cost(changes, scale_stats=scale_stats)
        actions = convert_changes_to_actions(changes)

        candidates.append(
            {
                "prototype_index": int(candidate_index),
                "predicted_class_after_change": predicted_class,
                "desired_class_probability": desired_prob,
                "num_changes": len(changes),
                "cost": cost,
                "changes": changes,
                "actions_tr": actions,
            }
        )

    candidates = sorted(
        candidates,
        key=lambda x: (x["cost"], -x["desired_class_probability"], x["num_changes"]),
    )

    return candidates[:top_k]


def build_counterfactual_summary(
    sample_index: int,
    true_class: int,
    predicted_class: int,
    desired_class: int,
    original_probabilities: Dict[int, float],
    candidates: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []

    lines.append(f"Seçilen çalışanın indeks değeri: {sample_index}.")
    lines.append(f"Gerçek sınıf: {true_class}.")
    lines.append(f"Model tahmini: {predicted_class}.")
    lines.append(f"Hedef karşı-olgusal sınıf: {desired_class}.")
    lines.append("Başlangıç olasılıkları:")
    for cls, prob in sorted(original_probabilities.items()):
        lines.append(f"- Sınıf {cls}: {prob:.4f}")

    lines.append("")

    if not candidates:
        lines.append("Belirlenen kısıtlar altında geçerli bir counterfactual öneri bulunamadı.")
        return "\n".join(lines)

    best = candidates[0]
    lines.append("En iyi counterfactual öneri:")
    lines.append(
        f"- Toplam değişiklik sayısı: {best['num_changes']}"
    )
    lines.append(
        f"- Hedef sınıf olasılığı: {best['desired_class_probability']:.4f}"
    )
    lines.append(
        f"- Maliyet skoru: {best['cost']:.4f}"
    )
    lines.append("Önerilen aksiyonlar:")
    for action in best["actions_tr"]:
        lines.append(f"- {action['action_sentence_tr']}")

    return "\n".join(lines)


def run_counterfactual_search(
    sample_index: Optional[int] = None,
    sample_position: Optional[int] = None,
    desired_class: Optional[int] = None,
    max_features_changed: int = 3,
    top_k: int = 5,
    actionable_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    actionable_features = actionable_features or DEFAULT_ACTIONABLE_FEATURES

    catboost_dir = SETTINGS.artifacts_dir / "catboost"
    cf_dir = SETTINGS.reports_dir / "xai" / "counterfactuals"
    ensure_dir(cf_dir)

    run_summary_path = catboost_dir / "run_summary.json"
    if not run_summary_path.exists():
        raise FileNotFoundError(
            f"CatBoost run summary not found: {run_summary_path}\n"
            f"Please train CatBoost first."
        )

    run_summary = load_json(run_summary_path)
    model = load_catboost_model(catboost_dir)

    prep = run_preprocessing(
        test_size=run_summary.get("test_size", 0.20),
        random_state=run_summary.get("random_state", 42),
        drop_sensitive=run_summary.get("drop_sensitive", False),
    )

    X_train_raw = prep["X_train_raw"].copy()
    X_test_raw = prep["X_test_raw"].copy()
    y_train = prep["y_train"].astype(int).copy()
    y_test = prep["y_test"].astype(int).copy()

    selected_index = pick_sample_index(
        X_test_raw=X_test_raw,
        sample_index=sample_index,
        sample_position=sample_position,
    )

    sample_df = X_test_raw.loc[[selected_index]].copy()

    drop_sensitive = run_summary.get("drop_sensitive", False)
    original_pred = predict_single_sample(
        model=model,
        sample_df=sample_df,
        drop_sensitive=drop_sensitive,
    )

    predicted_class = int(original_pred["predicted_class"])
    original_probabilities = original_pred["probabilities"]
    true_class = int(y_test.loc[selected_index])

    all_classes = sorted([int(c) for c in model.classes_])

    if desired_class is None:
        higher_classes = [c for c in all_classes if c > predicted_class]
        if len(higher_classes) == 0:
            raise ValueError(
                f"Sample {selected_index} is already predicted as the highest class ({predicted_class}). "
                f"Please provide --desired-class explicitly if needed."
            )
        desired_class = higher_classes[0]

    if desired_class not in all_classes:
        raise ValueError(
            f"desired_class={desired_class} is not in model classes: {all_classes}"
        )

    valid_actionable_features = [
        f for f in actionable_features if f in sample_df.columns
    ]
    if len(valid_actionable_features) == 0:
        raise ValueError("No valid actionable features found for the current model input.")

    candidates = generate_counterfactual_candidates(
        model=model,
        original_sample_df=sample_df,
        original_prediction=predicted_class,
        desired_class=desired_class,
        X_train_raw=X_train_raw,
        y_train=y_train,
        drop_sensitive=drop_sensitive,
        actionable_features=valid_actionable_features,
        max_features_changed=max_features_changed,
        top_k=top_k,
    )

    summary_text = build_counterfactual_summary(
        sample_index=selected_index,
        true_class=true_class,
        predicted_class=predicted_class,
        desired_class=desired_class,
        original_probabilities=original_probabilities,
        candidates=candidates,
    )

    sample_cf_dir = cf_dir / f"sample_{selected_index}"
    ensure_dir(sample_cf_dir)

    # Save tabular candidate summary
    candidate_rows: List[Dict[str, Any]] = []
    for rank, cand in enumerate(candidates, start=1):
        candidate_rows.append(
            {
                "rank": rank,
                "prototype_index": cand["prototype_index"],
                "predicted_class_after_change": cand["predicted_class_after_change"],
                "desired_class_probability": cand["desired_class_probability"],
                "num_changes": cand["num_changes"],
                "cost": cand["cost"],
                "changed_features": ", ".join([c["feature"] for c in cand["changes"]]),
            }
        )

    candidates_df = pd.DataFrame(candidate_rows)
    candidates_csv_path = sample_cf_dir / f"counterfactual_candidates_target_{desired_class}.csv"
    candidates_df.to_csv(candidates_csv_path, index=False)

    output = {
        "sample_index": int(selected_index),
        "true_class": int(true_class),
        "predicted_class": int(predicted_class),
        "desired_class": int(desired_class),
        "original_probabilities": {str(k): float(v) for k, v in original_probabilities.items()},
        "drop_sensitive": bool(drop_sensitive),
        "actionable_features": valid_actionable_features,
        "max_features_changed": int(max_features_changed),
        "top_k": int(top_k),
        "n_valid_counterfactuals_found": int(len(candidates)),
        "candidates": candidates,
        "summary_text_tr": summary_text,
        "output_files": {
            "candidates_csv": str(candidates_csv_path),
        },
    }

    json_path = sample_cf_dir / f"counterfactuals_target_{desired_class}.json"
    txt_path = sample_cf_dir / f"counterfactuals_target_{desired_class}.txt"

    save_json(output, json_path)
    save_text(summary_text, txt_path)

    print("\n=== COUNTERFACTUAL SEARCH COMPLETE ===")
    print(f"Sample index: {selected_index}")
    print(f"True class: {true_class}")
    print(f"Predicted class: {predicted_class}")
    print(f"Desired class: {desired_class}")
    print(f"Valid counterfactuals found: {len(candidates)}")
    print("\nSummary:\n")
    print(summary_text)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate counterfactual suggestions for a test sample.")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Original pandas index of the selected test sample.",
    )
    parser.add_argument(
        "--sample-position",
        type=int,
        default=None,
        help="0-based row position in X_test.",
    )
    parser.add_argument(
        "--desired-class",
        type=int,
        default=None,
        help="Desired target class. Default: next higher class than the current prediction.",
    )
    parser.add_argument(
        "--max-features-changed",
        type=int,
        default=3,
        help="Maximum number of actionable features allowed to change.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top counterfactual candidates to keep.",
    )
    parser.add_argument(
        "--actionable-features",
        type=str,
        default="EmpEnvironmentSatisfaction,EmpLastSalaryHikePercent,EmpWorkLifeBalance,EmpJobInvolvement,TrainingTimesLastYear",
        help="Comma-separated actionable features.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    actionable_features = [item.strip() for item in args.actionable_features.split(",") if item.strip()]

    run_counterfactual_search(
        sample_index=args.sample_index,
        sample_position=args.sample_position,
        desired_class=args.desired_class,
        max_features_changed=args.max_features_changed,
        top_k=args.top_k,
        actionable_features=actionable_features,
    )