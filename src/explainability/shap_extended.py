from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.data.preprocess import run_preprocessing
from src.models.train_catboost import prepare_catboost_inputs
from src.utils.config import SETTINGS


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_")


def load_model_and_summary() -> Tuple[CatBoostClassifier, Dict[str, Any]]:
    model_path = SETTINGS.artifacts_dir / "catboost" / "catboost_model.cbm"
    summary_path = SETTINGS.artifacts_dir / "catboost" / "run_summary.json"

    if not model_path.exists():
        raise FileNotFoundError(f"CatBoost model not found: {model_path}")

    if not summary_path.exists():
        raise FileNotFoundError(f"CatBoost run summary not found: {summary_path}")

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    return model, summary


def normalize_catboost_shap(
    raw_shap: Any,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(raw_shap)

    if arr.ndim == 2 and arr.shape == (n_samples, n_features + 1):
        shap_values = arr[:, :n_features][:, np.newaxis, :]
        base_values = arr[:, n_features][:, np.newaxis]
        return shap_values, base_values

    if arr.ndim == 3:
        if arr.shape == (n_samples, n_classes, n_features + 1):
            shap_values = arr[:, :, :n_features]
            base_values = arr[:, :, n_features]
            return shap_values, base_values

        if arr.shape == (n_classes, n_samples, n_features + 1):
            arr = np.transpose(arr, (1, 0, 2))
            shap_values = arr[:, :, :n_features]
            base_values = arr[:, :, n_features]
            return shap_values, base_values

        if arr.shape == (n_samples, n_features + 1, n_classes):
            arr = np.transpose(arr, (0, 2, 1))
            shap_values = arr[:, :, :n_features]
            base_values = arr[:, :, n_features]
            return shap_values, base_values

    raise ValueError(f"Unexpected CatBoost SHAP shape: {arr.shape}")


def compute_shap_outputs(sample_size: Optional[int] = None):
    model, summary = load_model_and_summary()

    prep = run_preprocessing(
        test_size=summary.get("test_size", 0.20),
        random_state=summary.get("random_state", 42),
        drop_sensitive=summary.get("drop_sensitive", False),
    )

    X = prep["X_test_raw"].copy()
    y = prep["y_test"].astype(int).copy()

    if sample_size is not None and sample_size < len(X):
        selected_index = X.sample(sample_size, random_state=summary.get("random_state", 42)).index
        X = X.loc[selected_index].copy()
        y = y.loc[selected_index].copy()

    X_cb, _, feature_names, cat_feature_indices = prepare_catboost_inputs(
        X_train=X,
        X_test=X.copy(),
        drop_sensitive=summary.get("drop_sensitive", False),
    )

    pool = Pool(
        data=X_cb,
        label=y,
        cat_features=cat_feature_indices,
        feature_names=feature_names,
    )

    raw_shap = model.get_feature_importance(data=pool, type="ShapValues")
    class_labels = [int(c) for c in model.classes_]

    shap_values, base_values = normalize_catboost_shap(
        raw_shap=raw_shap,
        n_samples=len(X_cb),
        n_features=len(feature_names),
        n_classes=len(class_labels),
    )

    pred = pd.Series(
        model.predict(pool).flatten().astype(int),
        index=X.index,
        name="predicted_class",
    )

    proba = pd.DataFrame(
        model.predict_proba(pool),
        columns=class_labels,
        index=X.index,
    )

    return X, y, pred, proba, shap_values, base_values, feature_names, class_labels


def plot_summary_class(
    shap_values: np.ndarray,
    feature_names: List[str],
    class_labels: List[int],
    class_label: int,
    output_path: Path,
    top_k: int,
) -> None:
    class_idx = class_labels.index(class_label)
    values = shap_values[:, class_idx, :]

    mean_abs = np.mean(np.abs(values), axis=0)
    selected_idx = np.argsort(mean_abs)[-top_k:]

    fig, ax = plt.subplots(figsize=(9, 7))
    rng = np.random.default_rng(42)

    for pos, feature_idx in enumerate(selected_idx):
        shap_col = values[:, feature_idx]
        jitter = rng.normal(0, 0.06, size=len(shap_col))
        ax.scatter(
            shap_col,
            np.full(len(shap_col), pos) + jitter,
            s=18,
            alpha=0.65,
        )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(selected_idx)))
    ax.set_yticklabels([feature_names[i] for i in selected_idx])
    ax.set_xlabel("SHAP value")
    ax.set_title(f"SHAP Summary | Class {class_label}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_local_shap_table(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: List[str],
    class_labels: List[int],
    sample_index: int,
    explained_class: int,
) -> pd.DataFrame:
    row_position = X.index.tolist().index(sample_index)
    class_idx = class_labels.index(explained_class)

    local_df = pd.DataFrame({
        "feature": feature_names,
        "feature_value": [X.loc[sample_index, f] for f in feature_names],
        "shap_value": shap_values[row_position, class_idx, :],
    })

    local_df["abs_shap_value"] = local_df["shap_value"].abs()
    local_df["direction"] = np.where(local_df["shap_value"] >= 0, "positive", "negative")
    local_df = local_df.sort_values("abs_shap_value", ascending=False).reset_index(drop=True)

    return local_df


def plot_local_explanation(
    local_df: pd.DataFrame,
    output_path: Path,
    title: str,
    top_k: int = 12,
) -> None:
    data = local_df.head(top_k).iloc[::-1].copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(data["feature"], data["shap_value"])
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("SHAP contribution")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_dependence(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: List[str],
    class_labels: List[int],
    class_label: int,
    feature: str,
    output_path: Path,
) -> None:
    if feature not in feature_names:
        return

    class_idx = class_labels.index(class_label)
    feature_idx = feature_names.index(feature)

    x_values = X[feature]
    y_values = shap_values[:, class_idx, feature_idx]

    fig, ax = plt.subplots(figsize=(7, 5))

    if pd.api.types.is_numeric_dtype(x_values):
        ax.scatter(x_values, y_values, alpha=0.7)
    else:
        ax.scatter(x_values.astype(str), y_values, alpha=0.7)
        ax.tick_params(axis="x", rotation=30)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP value")
    ax.set_title(f"SHAP dependence | {feature} | class {class_label}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def select_representative_cases(
    y: pd.Series,
    pred: pd.Series,
    proba: pd.DataFrame,
    class_labels: List[int],
) -> List[Tuple[str, int, int, int, float]]:
    cases: List[Tuple[str, int, int, int, float]] = []

    for class_label in class_labels:
        candidates = proba[pred == class_label]
        if len(candidates) > 0:
            idx = int(candidates[class_label].idxmax())
            cases.append((
                f"high_conf_class_{class_label}",
                idx,
                int(pred.loc[idx]),
                int(y.loc[idx]),
                float(proba.loc[idx, class_label]),
            ))

    uncertainty = proba.max(axis=1)
    uncertain_idx = int(uncertainty.idxmin())
    cases.append((
        "most_uncertain",
        uncertain_idx,
        int(pred.loc[uncertain_idx]),
        int(y.loc[uncertain_idx]),
        float(uncertainty.loc[uncertain_idx]),
    ))

    wrong = pred[pred != y]
    if len(wrong) > 0:
        wrong_idx = int(wrong.index[0])
        cases.append((
            "misclassified_example",
            wrong_idx,
            int(pred.loc[wrong_idx]),
            int(y.loc[wrong_idx]),
            float(proba.loc[wrong_idx].max()),
        ))

    return cases


def get_top_global_features(top_n: int = 5) -> List[str]:
    global_path = SETTINGS.reports_dir / "xai" / "global_shap_importance.csv"

    if global_path.exists():
        return pd.read_csv(global_path)["feature"].head(top_n).tolist()

    return [
        "EmpEnvironmentSatisfaction",
        "EmpLastSalaryHikePercent",
        "YearsSinceLastPromotion",
        "EmpDepartment",
        "ExperienceYearsInCurrentRole",
    ]


def run_shap_extended(sample_size: Optional[int] = None, top_k: int = 15) -> None:
    output_dir = SETTINGS.reports_dir / "xai" / "extended"
    ensure_dir(output_dir)

    X, y, pred, proba, shap_values, base_values, feature_names, class_labels = compute_shap_outputs(
        sample_size=sample_size
    )

    for class_label in class_labels:
        plot_summary_class(
            shap_values=shap_values,
            feature_names=feature_names,
            class_labels=class_labels,
            class_label=class_label,
            output_path=output_dir / f"summary_class_{class_label}.png",
            top_k=top_k,
        )

    cases = select_representative_cases(
        y=y,
        pred=pred,
        proba=proba,
        class_labels=class_labels,
    )

    case_rows = []

    for case_name, sample_index, predicted_class, true_class, confidence in cases:
        local_df = build_local_shap_table(
            X=X,
            shap_values=shap_values,
            feature_names=feature_names,
            class_labels=class_labels,
            sample_index=sample_index,
            explained_class=predicted_class,
        )

        local_df.to_csv(output_dir / f"{case_name}_local_shap_table.csv", index=False)

        plot_local_explanation(
            local_df=local_df,
            output_path=output_dir / f"{case_name}_local_explanation.png",
            title=f"{case_name} | pred={predicted_class} true={true_class} prob={confidence:.3f}",
        )

        case_rows.append({
            "case": case_name,
            "sample_index": sample_index,
            "predicted_class": predicted_class,
            "true_class": true_class,
            "confidence": confidence,
        })

    if not any(row["case"] == "misclassified_example" for row in case_rows):
        (output_dir / "misclassified_example_not_found.txt").write_text(
            "No misclassified sample was found in the selected SHAP evaluation set.",
            encoding="utf-8",
        )

    top_features = [f for f in get_top_global_features(top_n=5) if f in feature_names]

    for class_label in class_labels:
        for feature in top_features:
            plot_dependence(
                X=X,
                shap_values=shap_values,
                feature_names=feature_names,
                class_labels=class_labels,
                class_label=class_label,
                feature=feature,
                output_path=output_dir / f"dependence_class_{class_label}_{slugify(feature)}.png",
            )

    pd.DataFrame(case_rows).to_csv(output_dir / "representative_cases.csv", index=False)

    save_json(
        {
            "sample_size": sample_size,
            "top_k": top_k,
            "class_labels": class_labels,
            "n_samples": int(len(X)),
            "n_features": int(len(feature_names)),
            "top_dependence_features": top_features,
            "output_dir": str(output_dir),
        },
        output_dir / "shap_extended_metadata.json",
    )

    print("\n=== EXTENDED SHAP ANALYSIS COMPLETE ===")
    print(f"Outputs: {output_dir}")
    print("Created:")
    for item in sorted(output_dir.glob("*")):
        print(f"  - {item.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended SHAP analysis for employee performance model.")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_shap_extended(sample_size=args.sample_size, top_k=args.top_k)