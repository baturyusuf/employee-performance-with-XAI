from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.data.preprocess import (
    load_validated_or_raw_data,
    run_preprocessing,
    split_features_and_target,
)
from src.models.train_catboost import prepare_catboost_inputs
from src.utils.config import SETTINGS


DEFAULT_AUDIT_ATTRIBUTES = [
    "Gender",
    "MaritalStatus",
    "EmpDepartment",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").lower()


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


def rebuild_model_test_split(run_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rebuild the exact model input split using the same preprocessing params
    that were used during CatBoost training.
    """
    return run_preprocessing(
        test_size=run_summary.get("test_size", 0.20),
        random_state=run_summary.get("random_state", 42),
        drop_sensitive=run_summary.get("drop_sensitive", False),
    )


def rebuild_full_test_split(run_summary: Dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Rebuild the exact same test split from the FULL validated data
    (including sensitive columns), so fairness attributes can be audited
    even when the model was trained with drop_sensitive=True.
    """
    df_full = load_validated_or_raw_data()

    X_full, y_full = split_features_and_target(df_full, drop_sensitive=False)

    _, X_test_full, _, y_test_full = train_test_split(
        X_full,
        y_full,
        test_size=run_summary.get("test_size", 0.20),
        random_state=run_summary.get("random_state", 42),
        stratify=y_full,
    )

    return X_test_full, y_test_full


def build_model_test_pool(
    X_test_raw_model: pd.DataFrame,
    y_test: pd.Series,
    drop_sensitive: bool,
) -> tuple[pd.DataFrame, Pool]:
    X_test_cb, _, feature_names, cat_feature_indices = prepare_catboost_inputs(
        X_train=X_test_raw_model,
        X_test=X_test_raw_model.copy(),
        drop_sensitive=drop_sensitive,
    )

    test_pool = Pool(
        data=X_test_cb,
        label=y_test,
        cat_features=cat_feature_indices,
        feature_names=feature_names,
    )

    return X_test_cb, test_pool


def compute_group_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Optional[float]]:
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    metrics: Dict[str, Optional[float]] = {
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "weighted_f1": safe_float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": safe_float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": safe_float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_recall": safe_float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_recall": safe_float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "ordinal_mae": safe_float(mean_absolute_error(y_true, y_pred)),
    }

    unique_true = sorted(pd.Series(y_true).unique().tolist())
    unique_all = sorted(set(pd.Series(y_true).unique().tolist()) | set(pd.Series(y_pred).unique().tolist()))

    if len(unique_true) >= 2:
        metrics["balanced_accuracy"] = safe_float(balanced_accuracy_score(y_true, y_pred))
    else:
        metrics["balanced_accuracy"] = None

    if len(unique_all) >= 2:
        metrics["quadratic_weighted_kappa"] = safe_float(
            cohen_kappa_score(y_true, y_pred, weights="quadratic")
        )
    else:
        metrics["quadratic_weighted_kappa"] = None

    return metrics


def compute_disparity_summary(group_df: pd.DataFrame, metric_names: List[str]) -> Dict[str, Optional[float]]:
    summary: Dict[str, Optional[float]] = {}

    for metric in metric_names:
        if metric not in group_df.columns:
            summary[f"{metric}_max_gap"] = None
            continue

        values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
        if len(values) == 0:
            summary[f"{metric}_max_gap"] = None
        else:
            summary[f"{metric}_max_gap"] = float(values.max() - values.min())

    return summary


def save_metric_barplot(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    output_path: Path,
    title: str,
) -> None:
    plot_df = df.copy()
    plot_df = plot_df.sort_values(by=metric_col, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_df[group_col].astype(str), plot_df[metric_col])
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(metric_col)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_group_report(
    audit_df: pd.DataFrame,
    attribute: str,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if attribute not in audit_df.columns:
        raise ValueError(f"Attribute '{attribute}' not found in audit dataframe.")

    temp_df = audit_df.copy()
    temp_df[attribute] = temp_df[attribute].astype("string").fillna("__MISSING__")

    rows: List[Dict[str, Any]] = []

    for group_value, group_data in temp_df.groupby(attribute, dropna=False):
        y_true = group_data["y_true"].astype(int)
        y_pred = group_data["y_pred"].astype(int)

        metrics = compute_group_metrics(y_true, y_pred)

        row = {
            "attribute": attribute,
            "group_value": str(group_value),
            "n_samples": int(len(group_data)),
            "sample_share": float(len(group_data) / len(temp_df)),
            "n_true_classes": int(y_true.nunique()),
            "n_pred_classes": int(y_pred.nunique()),
            "true_class_distribution": json.dumps(y_true.value_counts().sort_index().to_dict(), ensure_ascii=False),
            "pred_class_distribution": json.dumps(y_pred.value_counts().sort_index().to_dict(), ensure_ascii=False),
        }
        row.update(metrics)
        rows.append(row)

    group_report_df = pd.DataFrame(rows).sort_values(by="n_samples", ascending=False).reset_index(drop=True)

    disparity = compute_disparity_summary(
        group_df=group_report_df,
        metric_names=[
            "accuracy",
            "weighted_f1",
            "macro_f1",
            "macro_recall",
            "balanced_accuracy",
            "ordinal_mae",
            "quadratic_weighted_kappa",
        ],
    )

    summary = {
        "attribute": attribute,
        "n_groups": int(group_report_df["group_value"].nunique()),
        "groups": group_report_df["group_value"].tolist(),
        "disparity": disparity,
    }

    return group_report_df, summary


def run_fairness_audit(
    attributes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    attributes = attributes or DEFAULT_AUDIT_ATTRIBUTES

    catboost_dir = SETTINGS.artifacts_dir / "catboost"
    fairness_dir = SETTINGS.reports_dir / "fairness"
    ensure_dir(fairness_dir)

    run_summary_path = catboost_dir / "run_summary.json"
    if not run_summary_path.exists():
        raise FileNotFoundError(
            f"CatBoost run summary not found: {run_summary_path}\n"
            f"Please train CatBoost first."
        )

    run_summary = load_json(run_summary_path)
    model = load_catboost_model(catboost_dir)

    # Rebuild model-facing split
    prep_model = rebuild_model_test_split(run_summary)
    X_test_raw_model = prep_model["X_test_raw"]
    y_test_model = prep_model["y_test"].astype(int)

    # Rebuild full-data split for auditing sensitive columns
    X_test_full, y_test_full = rebuild_full_test_split(run_summary)
    y_test_full = y_test_full.astype(int)

    # Defensive alignment check
    if not X_test_raw_model.index.equals(X_test_full.index):
        raise ValueError(
            "Test split indices do not match between model split and full-data split. "
            "Please check split reproducibility."
        )

    if not y_test_model.index.equals(y_test_full.index):
        raise ValueError(
            "Target indices do not match between model split and full-data split."
        )

    drop_sensitive = run_summary.get("drop_sensitive", False)

    X_test_cb, test_pool = build_model_test_pool(
        X_test_raw_model=X_test_raw_model,
        y_test=y_test_model,
        drop_sensitive=drop_sensitive,
    )

    y_pred = pd.Series(
        model.predict(test_pool).flatten().astype(int),
        index=X_test_cb.index,
        name="y_pred",
    )

    overall_metrics = compute_group_metrics(y_test_model, y_pred)

    audit_df = X_test_full.copy()
    audit_df["y_true"] = y_test_model
    audit_df["y_pred"] = y_pred
    audit_df["is_correct"] = (audit_df["y_true"] == audit_df["y_pred"]).astype(int)

    available_attributes = [attr for attr in attributes if attr in audit_df.columns]
    missing_attributes = [attr for attr in attributes if attr not in audit_df.columns]

    attribute_summaries: List[Dict[str, Any]] = []

    for attribute in available_attributes:
        group_report_df, summary = build_group_report(audit_df=audit_df, attribute=attribute)

        slug = slugify(attribute)
        csv_path = fairness_dir / f"{slug}_group_metrics.csv"
        group_report_df.to_csv(csv_path, index=False)

        # Charts
        if "accuracy" in group_report_df.columns:
            save_metric_barplot(
                df=group_report_df,
                group_col="group_value",
                metric_col="accuracy",
                output_path=fairness_dir / f"{slug}_accuracy.png",
                title=f"{attribute} - Accuracy by Group",
            )

        if "weighted_f1" in group_report_df.columns:
            save_metric_barplot(
                df=group_report_df,
                group_col="group_value",
                metric_col="weighted_f1",
                output_path=fairness_dir / f"{slug}_weighted_f1.png",
                title=f"{attribute} - Weighted F1 by Group",
            )

        summary["csv_path"] = str(csv_path)
        attribute_summaries.append(summary)

    report = {
        "model_name": run_summary.get("model_name"),
        "drop_sensitive": run_summary.get("drop_sensitive"),
        "audited_attributes_requested": attributes,
        "audited_attributes_available": available_attributes,
        "audited_attributes_missing": missing_attributes,
        "test_rows": int(len(audit_df)),
        "overall_test_metrics": overall_metrics,
        "attribute_summaries": attribute_summaries,
    }

    save_json(report, fairness_dir / "fairness_report_summary.json")

    print("\n=== FAIRNESS AUDIT COMPLETE ===")
    print(f"Model name: {run_summary.get('model_name')}")
    print(f"Drop sensitive during training: {run_summary.get('drop_sensitive')}")
    print(f"Audited attributes: {available_attributes}")
    print("\nOverall test metrics:")
    for k, v in overall_metrics.items():
        print(f"  {k}: {v}")

    print("\nAttribute disparity summary:")
    for summary in attribute_summaries:
        print(f"\n- {summary['attribute']}")
        for metric_name, gap in summary["disparity"].items():
            print(f"    {metric_name}: {gap}")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fairness audit report for CatBoost model.")
    parser.add_argument(
        "--attributes",
        type=str,
        default="Gender,MaritalStatus,EmpDepartment",
        help="Comma-separated list of attributes to audit. Example: Gender,MaritalStatus,EmpDepartment",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    attributes = [item.strip() for item in args.attributes.split(",") if item.strip()]
    run_fairness_audit(attributes=attributes)