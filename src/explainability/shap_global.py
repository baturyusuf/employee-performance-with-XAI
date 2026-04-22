from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def load_catboost_run_summary(catboost_dir: Path) -> Dict[str, Any]:
    summary_path = catboost_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"CatBoost run summary not found: {summary_path}\n"
            f"Train the CatBoost model first."
        )

    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_catboost_model(catboost_dir: Path) -> CatBoostClassifier:
    model_path = catboost_dir / "catboost_model.cbm"
    if not model_path.exists():
        raise FileNotFoundError(
            f"CatBoost model file not found: {model_path}\n"
            f"Train the CatBoost model first."
        )

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def rebuild_training_split_from_summary(run_summary: Dict[str, Any]) -> Dict[str, Any]:
    return run_preprocessing(
        test_size=run_summary.get("test_size", 0.20),
        random_state=run_summary.get("random_state", 42),
        drop_sensitive=run_summary.get("drop_sensitive", False),
    )


def build_pool_from_split(
    X_raw: pd.DataFrame,
    y: pd.Series,
    drop_sensitive: bool,
) -> Tuple[pd.DataFrame, Pool, List[str], List[int]]:
    X_cb, _, feature_names, cat_feature_indices = prepare_catboost_inputs(
        X_train=X_raw,
        X_test=X_raw.copy(),
        drop_sensitive=drop_sensitive,
    )

    pool = Pool(
        data=X_cb,
        label=y,
        cat_features=cat_feature_indices,
        feature_names=feature_names,
    )
    return X_cb, pool, feature_names, cat_feature_indices


def normalize_catboost_shap_output(
    raw_shap: np.ndarray,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        shap_values: shape (n_samples, n_classes, n_features)
        base_values: shape (n_samples, n_classes)

    CatBoost multiclass SHAP output can vary by version. This function normalizes it.
    """
    arr = np.array(raw_shap)

    # Binary/regression style: (n_samples, n_features + 1)
    if arr.ndim == 2:
        if arr.shape[0] == n_samples and arr.shape[1] == n_features + 1:
            shap_values = arr[:, :n_features][:, np.newaxis, :]
            base_values = arr[:, n_features][:, np.newaxis]
            return shap_values, base_values

    # Multiclass style: various possible axis orders
    if arr.ndim == 3:
        # Expected modern shape: (n_samples, n_classes, n_features + 1)
        if arr.shape == (n_samples, n_classes, n_features + 1):
            shap_values = arr[:, :, :n_features]
            base_values = arr[:, :, n_features]
            return shap_values, base_values

        # Possible shape: (n_classes, n_samples, n_features + 1)
        if arr.shape == (n_classes, n_samples, n_features + 1):
            arr = np.transpose(arr, (1, 0, 2))
            shap_values = arr[:, :, :n_features]
            base_values = arr[:, :, n_features]
            return shap_values, base_values

        # Possible shape: (n_samples, n_features + 1, n_classes)
        if arr.shape == (n_samples, n_features + 1, n_classes):
            arr = np.transpose(arr, (0, 2, 1))
            shap_values = arr[:, :, :n_features]
            base_values = arr[:, :, n_features]
            return shap_values, base_values

    raise ValueError(
        "Unexpected SHAP output shape from CatBoost.\n"
        f"Got shape: {arr.shape}\n"
        f"Expected something compatible with "
        f"(n_samples={n_samples}, n_classes={n_classes}, n_features+1={n_features + 1})."
    )


def compute_global_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    class_labels: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    shap_values shape: (n_samples, n_classes, n_features)
    """
    mean_abs_overall = np.mean(np.abs(shap_values), axis=(0, 1))
    overall_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_overall,
    }).sort_values(by="mean_abs_shap", ascending=False)

    per_class_rows = []
    for class_idx, class_label in enumerate(class_labels):
        class_mean_abs = np.mean(np.abs(shap_values[:, class_idx, :]), axis=0)
        for feature, value in zip(feature_names, class_mean_abs):
            per_class_rows.append({
                "class_label": int(class_label),
                "feature": feature,
                "mean_abs_shap": float(value),
            })

    per_class_df = pd.DataFrame(per_class_rows).sort_values(
        by=["class_label", "mean_abs_shap"],
        ascending=[True, False],
    )

    return overall_df, per_class_df


def save_global_barplot(
    importance_df: pd.DataFrame,
    output_path: Path,
    title: str,
    top_k: int = 20,
) -> None:
    top_df = importance_df.head(top_k).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_df["feature"], top_df["mean_abs_shap"])
    ax.set_title(title)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_classwise_barplots(
    per_class_df: pd.DataFrame,
    output_dir: Path,
    top_k: int = 15,
) -> None:
    class_labels = sorted(per_class_df["class_label"].unique().tolist())

    for class_label in class_labels:
        class_df = (
            per_class_df[per_class_df["class_label"] == class_label]
            .sort_values(by="mean_abs_shap", ascending=False)
            .head(top_k)
            .iloc[::-1]
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(class_df["feature"], class_df["mean_abs_shap"])
        ax.set_title(f"Top Features for Class {class_label} (Mean |SHAP|)")
        ax.set_xlabel("Mean |SHAP|")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        fig.savefig(output_dir / f"class_{class_label}_top_shap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_shap_matrices(
    shap_values: np.ndarray,
    base_values: np.ndarray,
    feature_names: List[str],
    class_labels: List[int],
    sample_index: pd.Index,
    output_dir: Path,
) -> None:
    """
    Saves per-class SHAP matrices for later local explanation scripts.
    """
    for class_idx, class_label in enumerate(class_labels):
        shap_df = pd.DataFrame(
            shap_values[:, class_idx, :],
            index=sample_index,
            columns=feature_names,
        )
        shap_df.to_csv(output_dir / f"test_shap_values_class_{class_label}.csv", index=True)

        base_df = pd.DataFrame(
            {
                "index": sample_index,
                "base_value": base_values[:, class_idx],
            }
        )
        base_df.to_csv(output_dir / f"test_base_values_class_{class_label}.csv", index=False)


def run_global_shap_analysis(
    sample_size: int | None = None,
) -> Dict[str, Any]:
    catboost_dir = SETTINGS.artifacts_dir / "catboost"
    xai_dir = SETTINGS.reports_dir / "xai"
    ensure_dir(xai_dir)

    run_summary = load_catboost_run_summary(catboost_dir)
    model = load_catboost_model(catboost_dir)

    prep = rebuild_training_split_from_summary(run_summary)

    X_test_raw = prep["X_test_raw"]
    y_test = prep["y_test"]

    drop_sensitive = run_summary.get("drop_sensitive", False)

    # Optional subsampling for speed
    if sample_size is not None and sample_size < len(X_test_raw):
        sampled_idx = X_test_raw.sample(n=sample_size, random_state=run_summary.get("random_state", 42)).index
        X_test_raw = X_test_raw.loc[sampled_idx].copy()
        y_test = y_test.loc[sampled_idx].copy()

    X_test_cb, test_pool, feature_names, cat_feature_indices = build_pool_from_split(
        X_raw=X_test_raw,
        y=y_test,
        drop_sensitive=drop_sensitive,
    )

    raw_shap = model.get_feature_importance(
        data=test_pool,
        type="ShapValues",
    )

    class_labels = sorted(model.classes_.tolist())
    shap_values, base_values = normalize_catboost_shap_output(
        raw_shap=raw_shap,
        n_samples=len(X_test_cb),
        n_features=len(feature_names),
        n_classes=len(class_labels),
    )

    overall_df, per_class_df = compute_global_importance(
        shap_values=shap_values,
        feature_names=feature_names,
        class_labels=class_labels,
    )

    overall_csv = xai_dir / "global_shap_importance.csv"
    per_class_csv = xai_dir / "global_shap_importance_by_class.csv"

    overall_df.to_csv(overall_csv, index=False)
    per_class_df.to_csv(per_class_csv, index=False)

    save_global_barplot(
        importance_df=overall_df,
        output_path=xai_dir / "global_shap_top20.png",
        title="Global SHAP Importance (Overall)",
        top_k=20,
    )

    save_classwise_barplots(
        per_class_df=per_class_df,
        output_dir=xai_dir,
        top_k=15,
    )

    save_shap_matrices(
        shap_values=shap_values,
        base_values=base_values,
        feature_names=feature_names,
        class_labels=class_labels,
        sample_index=X_test_cb.index,
        output_dir=xai_dir,
    )

    metadata = {
        "model_path": str(catboost_dir / "catboost_model.cbm"),
        "source_run_summary": str(catboost_dir / "run_summary.json"),
        "n_samples_used": int(len(X_test_cb)),
        "n_features": int(len(feature_names)),
        "n_classes": int(len(class_labels)),
        "class_labels": [int(c) for c in class_labels],
        "drop_sensitive": bool(drop_sensitive),
        "categorical_feature_names": [feature_names[i] for i in cat_feature_indices],
        "top_10_features_overall": overall_df.head(10).to_dict(orient="records"),
        "outputs": {
            "overall_csv": str(overall_csv),
            "per_class_csv": str(per_class_csv),
            "overall_plot": str(xai_dir / "global_shap_top20.png"),
        },
    }
    save_json(metadata, xai_dir / "global_shap_metadata.json")

    print("\n=== GLOBAL SHAP ANALYSIS COMPLETE ===")
    print(f"Samples used: {len(X_test_cb)}")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: {class_labels}")
    print("\nTop 10 global features:")
    print(overall_df.head(10).to_string(index=False))

    return {
        "overall_importance": overall_df,
        "per_class_importance": per_class_df,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global SHAP analysis for CatBoost model.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of test samples to use for SHAP computation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_global_shap_analysis(sample_size=args.sample_size)