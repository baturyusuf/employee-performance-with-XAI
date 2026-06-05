from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.data.preprocess import load_validated_or_raw_data, split_features_and_target
from src.experiments.leakage_safe_cv import fit_predict_model
from src.features.feature_sets import apply_feature_set, build_feature_columns
from src.utils.config import SETTINGS
from src.utils.config_loader import load_config
from src.utils.experiment_registry import append_registry_row, collect_package_versions, get_git_commit, utc_now_iso


DEFAULT_FEATURE_SETS = [
    "no_salary_hike_no_attrition",
    "no_salary_hike_no_attrition_no_department",
    "no_salary_hike_no_attrition_no_department_no_job_role",
]
DEFAULT_OUTPUT_DIR = SETTINGS.reports_dir / "fairness" / "feature_set_sensitivity"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def load_fairness_attributes() -> List[str]:
    config = load_config("fairness")
    attrs = config.get("fairness", {}).get("audit_attributes", [])
    if not isinstance(attrs, list):
        raise ValueError("configs/fairness.yaml must define fairness.audit_attributes as a list.")
    return [str(attr) for attr in attrs]


def parse_csv_arg(value: str, default: List[str]) -> List[str]:
    if value == "default":
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


def generate_oof_predictions(
    feature_sets: List[str],
    model_name: str,
    n_splits: int,
    random_state: int,
    drop_sensitive: bool,
) -> pd.DataFrame:
    df = load_validated_or_raw_data()
    X_raw, y = split_features_and_target(df, drop_sensitive=drop_sensitive)
    y = y.astype(int)
    labels = sorted(y.unique().tolist())
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for feature_set in feature_sets:
        X = apply_feature_set(X_raw.copy(), feature_set)
        selected_features = build_feature_columns(X_raw.columns, feature_set)
        removed_features = [col for col in X_raw.columns if col not in selected_features]

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()

            pred, proba = fit_predict_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                random_state=random_state,
            )

            for row_pos, sample_index in enumerate(X_test.index):
                row = {
                    "feature_set": feature_set,
                    "model": model_name,
                    "fold": fold,
                    "sample_index": int(sample_index),
                    "y_true": int(y_test.loc[sample_index]),
                    "y_pred": int(pred[row_pos]),
                    "n_features": int(X.shape[1]),
                    "removed_features": ", ".join(removed_features),
                }
                for class_idx, label in enumerate(labels):
                    row[f"prob_class_{label}"] = float(proba[row_pos, class_idx])
                rows.append(row)

            print(
                f"[fairness-cv] feature_set={feature_set} | model={model_name} | "
                f"fold={fold} | n_test={len(test_idx)}"
            )

    return pd.DataFrame(rows)


def safe_divide(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return float(num / den)


def compute_group_metrics(
    predictions: pd.DataFrame,
    full_df: pd.DataFrame,
    attributes: List[str],
    labels: List[int],
    min_support: int,
) -> pd.DataFrame:
    rows = []
    indexed_full = full_df.copy()

    for (feature_set, model), pred_subset in predictions.groupby(["feature_set", "model"]):
        pred_subset = pred_subset.copy()
        pred_subset["sample_index"] = pred_subset["sample_index"].astype(int)

        for attribute in attributes:
            if attribute not in indexed_full.columns:
                continue

            attr_values = indexed_full.loc[pred_subset["sample_index"], attribute].astype("string").fillna("__MISSING__").values
            attr_subset = pred_subset.assign(attribute_value=attr_values)

            for group_value, group in attr_subset.groupby("attribute_value", dropna=False):
                y_true = group["y_true"].astype(int).to_numpy()
                y_pred = group["y_pred"].astype(int).to_numpy()
                n_samples = int(len(group))
                support_warning = n_samples < min_support
                group_accuracy = float(accuracy_score(y_true, y_pred))
                group_macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

                for label in labels:
                    true_pos = int(np.sum((y_true == label) & (y_pred == label)))
                    false_pos = int(np.sum((y_true != label) & (y_pred == label)))
                    false_neg = int(np.sum((y_true == label) & (y_pred != label)))
                    true_neg = int(np.sum((y_true != label) & (y_pred != label)))
                    actual_support = int(np.sum(y_true == label))
                    predicted_support = int(np.sum(y_pred == label))

                    rows.append(
                        {
                            "feature_set": feature_set,
                            "model": model,
                            "attribute": attribute,
                            "group_value": str(group_value),
                            "n_samples": n_samples,
                            "support_warning": bool(support_warning),
                            "class_label": int(label),
                            "accuracy": group_accuracy,
                            "macro_f1": group_macro_f1,
                            "actual_class_support": actual_support,
                            "predicted_class_support": predicted_support,
                            "positive_prediction_rate": safe_divide(predicted_support, n_samples),
                            "true_positive_rate": safe_divide(true_pos, true_pos + false_neg),
                            "false_positive_rate": safe_divide(false_pos, false_pos + true_neg),
                            "precision": safe_divide(true_pos, true_pos + false_pos),
                            "mean_predicted_probability": float(group[f"prob_class_{label}"].mean()),
                        }
                    )

    return pd.DataFrame(rows)


def disparity_rows_for_metric(
    group_df: pd.DataFrame,
    metric: str,
    min_support: int,
    class_specific: bool,
) -> List[Dict[str, Any]]:
    rows = []
    filtered = group_df[group_df["n_samples"] >= min_support].copy()
    if filtered.empty:
        return rows

    group_cols = ["feature_set", "model", "attribute"]
    if class_specific:
        group_cols.append("class_label")
    else:
        filtered = filtered.drop_duplicates(["feature_set", "model", "attribute", "group_value"])

    for keys, subset in filtered.groupby(group_cols):
        values = pd.to_numeric(subset[metric], errors="coerce").dropna()
        if values.empty:
            continue
        metric_subset = subset.loc[values.index]
        min_idx = values.idxmin()
        max_idx = values.idxmax()
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        rows.append(
            {
                **key_map,
                "metric": metric,
                "n_groups_included": int(subset.loc[values.index, "group_value"].nunique()),
                "min_group_value": str(metric_subset.loc[min_idx, "group_value"]),
                "min_group_support": int(metric_subset.loc[min_idx, "n_samples"]),
                "min": float(values.min()),
                "max_group_value": str(metric_subset.loc[max_idx, "group_value"]),
                "max_group_support": int(metric_subset.loc[max_idx, "n_samples"]),
                "max": float(values.max()),
                "max_gap": float(values.max() - values.min()),
                "min_group_support_threshold": int(min_support),
            }
        )
    return rows


def compute_disparity_summary(group_df: pd.DataFrame, min_support: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for metric in ["accuracy", "macro_f1"]:
        rows.extend(disparity_rows_for_metric(group_df, metric, min_support, class_specific=False))

    for metric in [
        "positive_prediction_rate",
        "true_positive_rate",
        "false_positive_rate",
        "precision",
        "mean_predicted_probability",
    ]:
        rows.extend(disparity_rows_for_metric(group_df, metric, min_support, class_specific=True))

    return pd.DataFrame(rows).sort_values(["feature_set", "attribute", "metric", "max_gap"], ascending=[True, True, True, False])


def compute_small_group_warnings(full_df: pd.DataFrame, attributes: List[str], min_support: int) -> pd.DataFrame:
    rows = []
    for attribute in attributes:
        if attribute not in full_df.columns:
            continue
        counts = full_df[attribute].astype("string").fillna("__MISSING__").value_counts()
        for group_value, count in counts.items():
            if int(count) < min_support:
                rows.append(
                    {
                        "attribute": attribute,
                        "group_value": str(group_value),
                        "n_samples": int(count),
                        "warning": f"n_samples < {min_support}",
                    }
                )
    return pd.DataFrame(rows)


def interpretation_text(disparity_df: pd.DataFrame, feature_sets: List[str], model_name: str, min_support: int) -> str:
    lines = [
        "# Fairness Feature-Set Sensitivity",
        "",
        f"Model used for controlled comparison: `{model_name}`",
        f"Minimum group support threshold: {min_support}",
        "",
        "Feature sets audited:",
        "",
    ]
    lines.extend([f"- `{feature_set}`" for feature_set in feature_sets])
    lines.extend(
        [
            "",
            "## Largest Support-Filtered Gaps",
            "",
        ]
    )

    if disparity_df.empty:
        lines.append("No support-filtered disparity rows were available.")
    else:
        top = disparity_df.sort_values("max_gap", ascending=False).head(12)
        for row in top.itertuples(index=False):
            class_part = "" if not hasattr(row, "class_label") or pd.isna(row.class_label) else f", class={row.class_label}"
            group_part = ""
            if hasattr(row, "min_group_value") and hasattr(row, "max_group_value"):
                group_part = f", min_group={row.min_group_value}, max_group={row.max_group_value}"
            lines.append(
                f"- feature_set={row.feature_set}, attribute={row.attribute}, metric={row.metric}{class_part}: "
                f"max_gap={row.max_gap:.4f}, groups={row.n_groups_included}{group_part}"
            )

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- These are audit metrics, not proof of discrimination or proof of fairness.",
            "- Gaps below the support threshold are excluded from disparity summaries but retained in small-group warnings.",
            "- Feature-set changes should be interpreted jointly with performance, calibration, proxy risk, SHAP stability, and actionability.",
            "- Removing `EmpDepartment` and `EmpJobRole` can reduce direct/proxy organisational information but may also reduce model utility.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_fairness_sensitivity(
    feature_sets: List[str] = DEFAULT_FEATURE_SETS,
    model_name: str = "xgboost",
    n_splits: int = 10,
    random_state: int = 42,
    min_support: int = 30,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_registry: bool = True,
) -> Dict[str, Path]:
    ensure_dir(output_dir)

    full_df = load_validated_or_raw_data()
    labels = sorted(full_df[SETTINGS.target_col].astype(int).unique().tolist())
    attributes = load_fairness_attributes()

    predictions = generate_oof_predictions(
        feature_sets=feature_sets,
        model_name=model_name,
        n_splits=n_splits,
        random_state=random_state,
        drop_sensitive=True,
    )
    group_df = compute_group_metrics(predictions, full_df, attributes, labels, min_support)
    disparity_df = compute_disparity_summary(group_df, min_support)
    warnings_df = compute_small_group_warnings(full_df, attributes, min_support)

    outputs = {
        "oof_predictions": output_dir / "fairness_oof_predictions.csv",
        "group_metrics": output_dir / "fairness_group_metrics.csv",
        "disparity_summary": output_dir / "fairness_disparity_summary.csv",
        "small_group_warnings": output_dir / "small_group_warnings.csv",
        "interpretation": output_dir / "fairness_feature_set_sensitivity_interpretation.md",
        "metadata": output_dir / "metadata.json",
    }

    predictions.to_csv(outputs["oof_predictions"], index=False)
    group_df.to_csv(outputs["group_metrics"], index=False)
    disparity_df.to_csv(outputs["disparity_summary"], index=False)
    warnings_df.to_csv(outputs["small_group_warnings"], index=False)
    outputs["interpretation"].write_text(
        interpretation_text(disparity_df, feature_sets, model_name, min_support),
        encoding="utf-8",
    )
    save_json(
        {
            "feature_sets": feature_sets,
            "model": model_name,
            "n_splits": n_splits,
            "random_state": random_state,
            "min_support": min_support,
            "audit_attributes": attributes,
            "git_commit_if_available": get_git_commit(),
            "package_versions": collect_package_versions(["numpy", "pandas", "scikit-learn", "xgboost"]),
            "outputs": {name: str(path) for name, path in outputs.items() if name != "metadata"},
        },
        outputs["metadata"],
    )

    if write_registry:
        append_registry_row(
            {
                "run_id": f"fairness_feature_set_sensitivity_{utc_now_iso()}",
                "date_time": utc_now_iso(),
                "git_commit_if_available": get_git_commit(),
                "script": "python -m src.experiments.fairness_sensitivity",
                "config": "configs/feature_sets.yaml; configs/fairness.yaml",
                "feature_set": "; ".join(feature_sets),
                "model": model_name,
                "seed": random_state,
                "cv_strategy": f"StratifiedKFold(n_splits={n_splits}, shuffle=True)",
                "primary_metrics": "support-filtered subgroup gaps",
                "output_dir": str(output_dir.relative_to(SETTINGS.project_root)) if output_dir.is_relative_to(SETTINGS.project_root) else str(output_dir),
                "notes": "OOF fairness sensitivity across feature sets using a fixed model to isolate feature-set effects.",
                "decision_status": "candidate",
            }
        )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CV-based fairness sensitivity across feature sets.")
    parser.add_argument("--feature-sets", default="default")
    parser.add_argument("--model", default="xgboost", choices=["catboost", "lightgbm", "xgboost"])
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_feature_sets = parse_csv_arg(args.feature_sets, DEFAULT_FEATURE_SETS)
    paths = run_fairness_sensitivity(
        feature_sets=selected_feature_sets,
        model_name=args.model,
        n_splits=args.n_splits,
        random_state=args.random_state,
        min_support=args.min_support,
        output_dir=args.output_dir,
        write_registry=not args.no_registry,
    )
    print("Fairness sensitivity outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
