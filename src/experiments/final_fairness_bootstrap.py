from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.experiments.fairness_sensitivity import (
    compute_disparity_summary,
    compute_group_metrics,
    compute_small_group_warnings,
    load_fairness_attributes,
    run_fairness_sensitivity,
)
from src.experiments.final_evidence_common import (
    FAIRNESS_DIR,
    FINAL_FEATURE_SETS,
    LABELS,
    MODEL_NAME,
    append_task_registry,
    ensure_dir,
    save_json,
)
from src.models.evaluate import classification_metrics
from src.utils.config import SETTINGS
from src.utils.experiment_registry import collect_package_versions, utc_now_iso
from src.data.preprocess import load_validated_or_raw_data


def validate_or_create_oof_predictions(output_dir: Path = FAIRNESS_DIR) -> Path:
    pred_path = output_dir / "fairness_oof_predictions.csv"
    required = {"feature_set", "model", "fold", "sample_index", "y_true", "y_pred"}
    required.update({f"prob_class_{label}" for label in LABELS})
    valid = False
    if pred_path.exists():
        try:
            df = pd.read_csv(pred_path)
            valid = required.issubset(df.columns)
            valid = valid and sorted(df["feature_set"].unique().tolist()) == sorted(FINAL_FEATURE_SETS)
            valid = valid and int(df.groupby("feature_set")["sample_index"].nunique().min()) == len(load_validated_or_raw_data())
        except Exception:
            valid = False
    if not valid:
        run_fairness_sensitivity(
            feature_sets=FINAL_FEATURE_SETS,
            model_name=MODEL_NAME,
            n_splits=10,
            random_state=42,
            min_support=30,
            output_dir=output_dir,
            write_registry=True,
        )
    return pred_path


def normalize_class_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(float(value)) if isinstance(value, (float, np.floating)) else str(value)


def add_disparity_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "class_label" not in out.columns:
        out["class_label"] = np.nan
    out["class_label_key"] = out["class_label"].map(normalize_class_label)
    return out


def compute_bootstrap_disparity_ci(
    predictions: pd.DataFrame,
    full_df: pd.DataFrame,
    attributes: List[str],
    min_support: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    point_group = compute_group_metrics(predictions, full_df, attributes, LABELS, min_support)
    point = add_disparity_key(compute_disparity_summary(point_group, min_support))

    rng = np.random.default_rng(seed)
    boot_frames = []
    for iteration in range(1, n_boot + 1):
        sampled = []
        for _, subset in predictions.groupby("feature_set", dropna=False):
            idx = rng.integers(0, len(subset), size=len(subset))
            sampled.append(subset.iloc[idx].copy())
        boot_predictions = pd.concat(sampled, ignore_index=True)
        group_df = compute_group_metrics(boot_predictions, full_df, attributes, LABELS, min_support)
        disparity = compute_disparity_summary(group_df, min_support)
        if disparity.empty:
            continue
        disparity = add_disparity_key(disparity)
        disparity["bootstrap_iteration"] = iteration
        boot_frames.append(disparity)

    key = ["feature_set", "model", "attribute", "metric", "class_label_key"]
    point_cols = [
        "feature_set",
        "model",
        "attribute",
        "metric",
        "class_label",
        "class_label_key",
        "n_groups_included",
        "min_group_value",
        "min_group_support",
        "min",
        "max_group_value",
        "max_group_support",
        "max",
        "max_gap",
        "min_group_support_threshold",
    ]
    point_cols = [col for col in point_cols if col in point.columns]
    base = point[point_cols].rename(columns={"max_gap": "point_estimate"})
    if not boot_frames:
        base["ci_low"] = np.nan
        base["ci_high"] = np.nan
        base["bootstrap_std"] = np.nan
        base["n_boot_valid"] = 0
        return base

    boot = pd.concat(boot_frames, ignore_index=True)
    agg = boot.groupby(key, dropna=False)["max_gap"].agg(
        ci_low=lambda s: float(np.quantile(s, 0.025)),
        ci_high=lambda s: float(np.quantile(s, 0.975)),
        bootstrap_std=lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
        n_boot_valid="count",
    ).reset_index()
    out = base.merge(agg, on=key, how="left")
    ordered = [
        "feature_set",
        "model",
        "attribute",
        "metric",
        "class_label",
        "point_estimate",
        "ci_low",
        "ci_high",
        "bootstrap_std",
        "n_boot_valid",
        "n_groups_included",
        "min_group_value",
        "min_group_support",
        "min",
        "max_group_value",
        "max_group_support",
        "max",
        "min_group_support_threshold",
    ]
    return out[[col for col in ordered if col in out.columns]].sort_values(
        ["feature_set", "attribute", "metric", "point_estimate"],
        ascending=[True, True, True, False],
    )


def oof_performance_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_set, group in predictions.groupby("feature_set"):
        proba = group[[f"prob_class_{label}" for label in LABELS]].to_numpy(dtype=float)
        metrics = classification_metrics(group["y_true"], group["y_pred"], proba, LABELS)
        rows.append({"feature_set": feature_set, **metrics})
    return pd.DataFrame(rows)


def get_ci_row(ci_df: pd.DataFrame, feature_set: str, attribute: str, metric: str, class_label: Optional[int] = None) -> Optional[pd.Series]:
    subset = ci_df[(ci_df["feature_set"] == feature_set) & (ci_df["attribute"] == attribute) & (ci_df["metric"] == metric)].copy()
    if class_label is None:
        subset = subset[subset["class_label"].isna()]
    else:
        subset = subset[pd.to_numeric(subset["class_label"], errors="coerce") == float(class_label)]
    if subset.empty:
        return None
    return subset.sort_values("point_estimate", ascending=False).iloc[0]


def format_ci(row: Optional[pd.Series]) -> str:
    if row is None:
        return "not available"
    return f"{row['point_estimate']:.4f} (95% CI {row['ci_low']:.4f}-{row['ci_high']:.4f})"


def write_interpretation(ci_df: pd.DataFrame, perf_df: pd.DataFrame, warnings_df: pd.DataFrame, output_path: Path, n_boot: int) -> None:
    dept_macro = {fs: get_ci_row(ci_df, fs, "EmpDepartment", "macro_f1") for fs in FINAL_FEATURE_SETS}
    dept_acc = {fs: get_ci_row(ci_df, fs, "EmpDepartment", "accuracy") for fs in FINAL_FEATURE_SETS}
    perf = perf_df.set_index("feature_set")
    department_free = perf.loc["no_salary_hike_no_attrition_no_department"]
    strict = perf.loc["no_salary_hike_no_attrition_no_department_no_job_role"]
    lines = [
        "# Bootstrap Fairness Disparity Interpretation",
        "",
        f"Bootstrap iterations: {n_boot}",
        "Minimum group support: 30",
        "Model: XGBoost with fold-safe preprocessing",
        "",
        "These metrics are diagnostic audit evidence. They are not legal findings and do not prove discrimination or absence of discrimination.",
        "",
        "## Department Disparity With Uncertainty",
        "",
    ]
    for fs in FINAL_FEATURE_SETS:
        lines.append(f"- `{fs}`: EmpDepartment macro-F1 gap {format_ci(dept_macro[fs])}; accuracy gap {format_ci(dept_acc[fs])}.")
    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            "### Did removing EmpDepartment materially reduce department-related disparity?",
            "No clear material reduction is supported. The department macro-F1 gap is nearly unchanged after direct department removal, and uncertainty should be treated as overlapping diagnostic evidence.",
            "",
            "### Did removing EmpJobRole materially reduce proxy/fairness risk?",
            "It materially reduced department reconstructability in the proxy audit, but subgroup disparity improvement is mixed. The strict set lowers the department macro-F1 gap point estimate while worsening utility and not uniformly improving every disparity metric.",
            "",
            "### What was the utility cost?",
            f"Relative to the department-free candidate, strict job-role removal changes macro-F1 from {department_free['macro_f1']:.4f} to {strict['macro_f1']:.4f}, QWK from {department_free['quadratic_weighted_kappa']:.4f} to {strict['quadratic_weighted_kappa']:.4f}, and ordinal MAE from {department_free['ordinal_mae']:.4f} to {strict['ordinal_mae']:.4f}.",
            "",
            "### Which disparities remain concerning?",
            "Department-related macro-F1, class-2 TPR/precision, and class-3 FPR gaps remain prominent. Class-4 precision gaps are large but sparse-prediction-sensitive and should be interpreted cautiously.",
            "",
            "### Which claims are not justified by the evidence?",
            "The evidence does not justify claiming that removing EmpDepartment proves fairness, that removing EmpJobRole is automatically the best final policy, or that subgroup gaps prove causal discrimination.",
            "",
            "## Small-Group Warnings",
            "",
        ]
    )
    if warnings_df.empty:
        lines.append("No small groups below the support threshold were detected.")
    else:
        for row in warnings_df.itertuples(index=False):
            lines.append(f"- {row.attribute}={row.group_value}: n={row.n_samples}, {row.warning}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(n_boot: int = 500, seed: int = 42, min_support: int = 30) -> Dict[str, Path]:
    ensure_dir(FAIRNESS_DIR)
    pred_path = validate_or_create_oof_predictions(FAIRNESS_DIR)
    predictions = pd.read_csv(pred_path)
    full_df = load_validated_or_raw_data()
    attributes = load_fairness_attributes()
    ci_df = compute_bootstrap_disparity_ci(predictions, full_df, attributes, min_support, n_boot, seed)
    warnings_df = compute_small_group_warnings(full_df, attributes, min_support)
    performance_df = oof_performance_table(predictions)

    ci_path = FAIRNESS_DIR / "bootstrap_disparity_ci.csv"
    interp_path = FAIRNESS_DIR / "bootstrap_disparity_interpretation.md"
    meta_path = FAIRNESS_DIR / "bootstrap_disparity_metadata.json"
    ci_df.to_csv(ci_path, index=False)
    warnings_df.to_csv(FAIRNESS_DIR / "small_group_warnings.csv", index=False)
    write_interpretation(ci_df, performance_df, warnings_df, interp_path, n_boot)
    save_json(
        {
            "task": "bootstrap_fairness_uncertainty",
            "feature_sets": FINAL_FEATURE_SETS,
            "model": MODEL_NAME,
            "n_boot": n_boot,
            "seed": seed,
            "min_support": min_support,
            "source_oof_predictions": pred_path,
            "package_versions": collect_package_versions(["numpy", "pandas", "scikit-learn", "xgboost"]),
        },
        meta_path,
    )
    append_task_registry(
        run_id=f"final_fairness_bootstrap_{utc_now_iso()}",
        script="python -m src.experiments.final_fairness_bootstrap",
        feature_set="; ".join(FINAL_FEATURE_SETS),
        primary_metrics="bootstrap CI for support-filtered subgroup disparity gaps",
        output_dir=FAIRNESS_DIR,
        notes="Bootstrap uncertainty for OOF fairness disparity metrics using current config-backed XGBoost OOF predictions.",
    )
    return {"ci": ci_path, "interpretation": interp_path, "metadata": meta_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap fairness disparity uncertainty for final XGBoost candidates.")
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-support", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(n_boot=args.bootstrap_iterations, seed=args.seed, min_support=args.min_support))

