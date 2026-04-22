from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_catboost_model(catboost_dir: Path) -> CatBoostClassifier:
    model_path = catboost_dir / "catboost_model.cbm"
    if not model_path.exists():
        raise FileNotFoundError(
            f"CatBoost model not found: {model_path}\n"
            f"Train CatBoost first."
        )

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def rebuild_split_from_run_summary(run_summary: Dict[str, Any]) -> Dict[str, Any]:
    return run_preprocessing(
        test_size=run_summary.get("test_size", 0.20),
        random_state=run_summary.get("random_state", 42),
        drop_sensitive=run_summary.get("drop_sensitive", False),
    )


def load_shap_artifacts(xai_dir: Path, class_label: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    shap_path = xai_dir / f"test_shap_values_class_{class_label}.csv"
    base_path = xai_dir / f"test_base_values_class_{class_label}.csv"

    if not shap_path.exists() or not base_path.exists():
        raise FileNotFoundError(
            f"SHAP artifacts for class {class_label} not found.\n"
            f"Please run: python -m src.explainability.shap_global"
        )

    shap_df = pd.read_csv(shap_path, index_col=0)
    base_df = pd.read_csv(base_path)

    shap_df.index = shap_df.index.astype(int)
    if "index" in base_df.columns:
        base_df["index"] = base_df["index"].astype(int)

    return shap_df, base_df


def build_test_pool(
    X_test_raw: pd.DataFrame,
    y_test: pd.Series,
    drop_sensitive: bool,
) -> tuple[pd.DataFrame, Pool]:
    X_test_cb, _, feature_names, cat_feature_indices = prepare_catboost_inputs(
        X_train=X_test_raw,
        X_test=X_test_raw.copy(),
        drop_sensitive=drop_sensitive,
    )

    test_pool = Pool(
        data=X_test_cb,
        label=y_test,
        cat_features=cat_feature_indices,
        feature_names=feature_names,
    )
    return X_test_cb, test_pool


def pick_sample_index(
    X_test_raw: pd.DataFrame,
    sample_index: Optional[int] = None,
    sample_position: Optional[int] = None,
) -> int:
    if sample_index is not None:
        if sample_index not in X_test_raw.index:
            raise ValueError(
                f"sample_index={sample_index} not found in X_test index.\n"
                f"Available index range example: {list(X_test_raw.index[:10])}"
            )
        return int(sample_index)

    if sample_position is not None:
        if sample_position < 0 or sample_position >= len(X_test_raw):
            raise ValueError(
                f"sample_position={sample_position} out of range. "
                f"Valid range: 0 to {len(X_test_raw) - 1}"
            )
        return int(X_test_raw.index[sample_position])

    # default: first test row
    return int(X_test_raw.index[0])


def create_local_contribution_table(
    raw_feature_values: pd.Series,
    shap_row: pd.Series,
) -> pd.DataFrame:
    df = pd.DataFrame({
        "feature": shap_row.index,
        "feature_value": [raw_feature_values.get(col) for col in shap_row.index],
        "shap_value": shap_row.values,
    })

    df["abs_shap_value"] = df["shap_value"].abs()
    df["direction"] = df["shap_value"].apply(lambda x: "positive" if x >= 0 else "negative")
    df = df.sort_values(by="abs_shap_value", ascending=False).reset_index(drop=True)
    return df


def save_local_barplot(
    local_df: pd.DataFrame,
    output_path: Path,
    title: str,
    top_k: int = 12,
) -> None:
    plot_df = local_df.head(top_k).copy().iloc[::-1]

    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in plot_df["shap_value"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["feature"], plot_df["shap_value"], color=colors)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("SHAP contribution")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_natural_language_summary(
    local_df: pd.DataFrame,
    predicted_class: int,
    true_class: int,
    predicted_probability: float,
) -> Dict[str, Any]:
    top_positive = (
        local_df[local_df["shap_value"] > 0]
        .head(5)[["feature", "feature_value", "shap_value"]]
        .to_dict(orient="records")
    )
    top_negative = (
        local_df[local_df["shap_value"] < 0]
        .head(5)[["feature", "feature_value", "shap_value"]]
        .to_dict(orient="records")
    )

    summary_text = (
        f"Model bu çalışanı {predicted_class} sınıfında tahmin etti. "
        f"Gerçek sınıf: {true_class}. "
        f"Tahmin olasılığı yaklaşık {predicted_probability:.4f}. "
        f"Pozitif katkı yapan değişkenler ve negatif katkı yapan değişkenler ayrı listelendi."
    )

    return {
        "summary_text": summary_text,
        "predicted_class": int(predicted_class),
        "true_class": int(true_class),
        "predicted_probability": float(predicted_probability),
        "top_positive_drivers": top_positive,
        "top_negative_drivers": top_negative,
    }


def run_local_shap_analysis(
    sample_index: Optional[int] = None,
    sample_position: Optional[int] = None,
    class_label: Optional[int] = None,
) -> Dict[str, Any]:
    catboost_dir = SETTINGS.artifacts_dir / "catboost"
    xai_dir = SETTINGS.reports_dir / "xai"
    local_dir = xai_dir / "local"
    ensure_dir(local_dir)

    run_summary = load_json(catboost_dir / "run_summary.json")
    model = load_catboost_model(catboost_dir)

    prep = rebuild_split_from_run_summary(run_summary)
    X_test_raw = prep["X_test_raw"]
    y_test = prep["y_test"]
    drop_sensitive = run_summary.get("drop_sensitive", False)

    selected_index = pick_sample_index(
        X_test_raw=X_test_raw,
        sample_index=sample_index,
        sample_position=sample_position,
    )

    X_test_cb, test_pool = build_test_pool(
        X_test_raw=X_test_raw,
        y_test=y_test,
        drop_sensitive=drop_sensitive,
    )

    pred_labels = pd.Series(
        model.predict(test_pool).flatten().astype(int),
        index=X_test_cb.index,
        name="prediction",
    )
    pred_proba = pd.DataFrame(
        model.predict_proba(test_pool),
        columns=[int(c) for c in model.classes_],
        index=X_test_cb.index,
    )

    predicted_class = int(pred_labels.loc[selected_index])
    true_class = int(y_test.loc[selected_index])

    target_class = int(class_label) if class_label is not None else predicted_class

    shap_df, base_df = load_shap_artifacts(xai_dir=xai_dir, class_label=target_class)

    if selected_index not in shap_df.index:
        raise ValueError(
            f"Selected index {selected_index} not found in SHAP artifacts.\n"
            f"Run shap_global again after the latest CatBoost training."
        )

    shap_row = shap_df.loc[selected_index]
    raw_feature_values = X_test_raw.loc[selected_index]
    local_df = create_local_contribution_table(
        raw_feature_values=raw_feature_values,
        shap_row=shap_row,
    )

    base_row = base_df[base_df["index"] == selected_index]
    base_value = float(base_row["base_value"].iloc[0]) if len(base_row) > 0 else None

    predicted_probability = float(pred_proba.loc[selected_index, predicted_class])

    sample_dir = local_dir / f"sample_{selected_index}"
    ensure_dir(sample_dir)

    local_df.to_csv(sample_dir / f"local_explanation_class_{target_class}.csv", index=False)

    save_local_barplot(
        local_df=local_df,
        output_path=sample_dir / f"local_explanation_class_{target_class}.png",
        title=f"Local SHAP Explanation | sample={selected_index} | class={target_class}",
        top_k=12,
    )

    summary = build_natural_language_summary(
        local_df=local_df,
        predicted_class=predicted_class,
        true_class=true_class,
        predicted_probability=predicted_probability,
    )

    metadata = {
        "sample_index": int(selected_index),
        "sample_position": int(list(X_test_raw.index).index(selected_index)),
        "true_class": int(true_class),
        "predicted_class": int(predicted_class),
        "explained_class": int(target_class),
        "predicted_probability_for_predicted_class": float(predicted_probability),
        "base_value_for_explained_class": base_value,
        "drop_sensitive": bool(drop_sensitive),
        "top_10_local_features": local_df.head(10).to_dict(orient="records"),
        "summary": summary,
    }
    save_json(metadata, sample_dir / f"local_explanation_class_{target_class}.json")

    print("\n=== LOCAL SHAP ANALYSIS COMPLETE ===")
    print(f"Sample index: {selected_index}")
    print(f"True class: {true_class}")
    print(f"Predicted class: {predicted_class}")
    print(f"Explained class: {target_class}")
    print(f"Predicted probability: {predicted_probability:.4f}")
    print("\nTop 10 local contributions:")
    print(local_df.head(10).to_string(index=False))

    return {
        "local_df": local_df,
        "metadata": metadata,
        "sample_dir": sample_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local SHAP analysis for one test sample.")
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
        help="0-based row position within X_test.",
    )
    parser.add_argument(
        "--class-label",
        type=int,
        default=None,
        help="Optional class to explain. Default: predicted class.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_local_shap_analysis(
        sample_index=args.sample_index,
        sample_position=args.sample_position,
        class_label=args.class_label,
    )