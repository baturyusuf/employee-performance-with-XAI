from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


DEFAULT_BASE_CV = SETTINGS.reports_dir / "leakage_safe_cv_config" / "summary_metrics.csv"
DEFAULT_JOB_ROLE_FREE_CV = SETTINGS.reports_dir / "leakage_safe_cv_job_role_sensitivity" / "summary_metrics.csv"
DEFAULT_PROXY_WITH_JOB_ROLE = SETTINGS.reports_dir / "fairness" / "proxy_with_job_role" / "proxy_analysis_department_cv_summary.csv"
DEFAULT_PROXY_NO_JOB_ROLE = SETTINGS.reports_dir / "fairness" / "proxy_no_job_role" / "proxy_analysis_department_cv_summary.csv"
DEFAULT_OUTPUT_DIR = SETTINGS.reports_dir / "fairness"


def read_metric(summary_path: Path, metric: str) -> float:
    df = pd.read_csv(summary_path)
    row = df[df["metric"] == metric]
    if row.empty:
        raise ValueError(f"Metric {metric} not found in {summary_path}")
    return float(row.iloc[0]["mean"])


def best_model_row(summary_path: Path, feature_set: str | None = None) -> pd.Series:
    df = pd.read_csv(summary_path)
    if feature_set is not None:
        df = df[df["feature_set"] == feature_set]
    if df.empty:
        raise ValueError(f"No rows available in {summary_path} for feature_set={feature_set}")
    return df.sort_values("macro_f1_mean", ascending=False).iloc[0]


def build_comparison(
    base_cv: Path,
    job_role_free_cv: Path,
    proxy_with_job_role: Path,
    proxy_no_job_role: Path,
) -> pd.DataFrame:
    base_row = best_model_row(base_cv, "no_salary_hike_no_attrition_no_department")
    strict_row = best_model_row(job_role_free_cv, "no_salary_hike_no_attrition_no_department_no_job_role")

    with_proxy_macro_f1 = read_metric(proxy_with_job_role, "macro_f1")
    no_proxy_macro_f1 = read_metric(proxy_no_job_role, "macro_f1")
    with_proxy_bal_acc = read_metric(proxy_with_job_role, "balanced_accuracy")
    no_proxy_bal_acc = read_metric(proxy_no_job_role, "balanced_accuracy")

    rows = [
        {
            "comparison": "best_department_free_with_job_role",
            "feature_set": base_row["feature_set"],
            "best_model": base_row["model"],
            "performance_macro_f1": float(base_row["macro_f1_mean"]),
            "performance_qwk": float(base_row["quadratic_weighted_kappa_mean"]),
            "performance_ordinal_mae": float(base_row["ordinal_mae_mean"]),
            "department_proxy_macro_f1": with_proxy_macro_f1,
            "department_proxy_balanced_accuracy": with_proxy_bal_acc,
        },
        {
            "comparison": "best_department_free_without_job_role",
            "feature_set": strict_row["feature_set"],
            "best_model": strict_row["model"],
            "performance_macro_f1": float(strict_row["macro_f1_mean"]),
            "performance_qwk": float(strict_row["quadratic_weighted_kappa_mean"]),
            "performance_ordinal_mae": float(strict_row["ordinal_mae_mean"]),
            "department_proxy_macro_f1": no_proxy_macro_f1,
            "department_proxy_balanced_accuracy": no_proxy_bal_acc,
        },
    ]
    out = pd.DataFrame(rows)
    baseline = out.iloc[0]
    strict = out.iloc[1]
    out["macro_f1_delta_vs_with_job_role"] = out["performance_macro_f1"] - float(baseline["performance_macro_f1"])
    out["qwk_delta_vs_with_job_role"] = out["performance_qwk"] - float(baseline["performance_qwk"])
    out["proxy_macro_f1_delta_vs_with_job_role"] = out["department_proxy_macro_f1"] - float(baseline["department_proxy_macro_f1"])
    out["proxy_macro_f1_relative_reduction"] = [
        0.0,
        (float(baseline["department_proxy_macro_f1"]) - float(strict["department_proxy_macro_f1"]))
        / abs(float(baseline["department_proxy_macro_f1"])),
    ]
    return out


def interpretation_text(comparison_df: pd.DataFrame) -> str:
    base = comparison_df.iloc[0]
    strict = comparison_df.iloc[1]
    perf_drop = float(base["performance_macro_f1"]) - float(strict["performance_macro_f1"])
    qwk_drop = float(base["performance_qwk"]) - float(strict["performance_qwk"])
    proxy_drop = float(base["department_proxy_macro_f1"]) - float(strict["department_proxy_macro_f1"])
    proxy_relative = float(strict["proxy_macro_f1_relative_reduction"])

    return f"""# Job-Role Proxy Sensitivity Comparison

## Summary

Removing `EmpJobRole` sharply reduces department reconstructability, but it also reduces employee-performance predictive performance.

## Performance comparison

- With `EmpJobRole`: best model `{base["best_model"]}`, macro-F1={float(base["performance_macro_f1"]):.4f}, QWK={float(base["performance_qwk"]):.4f}.
- Without `EmpJobRole`: best model `{strict["best_model"]}`, macro-F1={float(strict["performance_macro_f1"]):.4f}, QWK={float(strict["performance_qwk"]):.4f}.
- Macro-F1 drop from removing job role: {perf_drop:.4f}.
- QWK drop from removing job role: {qwk_drop:.4f}.

## Proxy-risk comparison

- Department proxy macro-F1 with `EmpJobRole`: {float(base["department_proxy_macro_f1"]):.4f}.
- Department proxy macro-F1 without `EmpJobRole`: {float(strict["department_proxy_macro_f1"]):.4f}.
- Absolute proxy macro-F1 reduction: {proxy_drop:.4f}.
- Relative proxy macro-F1 reduction: {proxy_relative:.1%}.

## Interpretation

`EmpJobRole` is the dominant department proxy. Removing it makes `EmpDepartment` much harder to reconstruct, but the performance cost is non-trivial. This supports keeping the job-role-free model as a strict fairness/proxy sensitivity baseline, not automatically as the final model.

## Decision implication

Do not claim that removing `EmpDepartment` alone removes organisational proxy risk. If `EmpJobRole` remains in a final model, reason codes, model cards, and G-XAIR scoring should include explicit organisational proxy warnings.
"""


def run_job_role_sensitivity_report(
    base_cv: Path = DEFAULT_BASE_CV,
    job_role_free_cv: Path = DEFAULT_JOB_ROLE_FREE_CV,
    proxy_with_job_role: Path = DEFAULT_PROXY_WITH_JOB_ROLE,
    proxy_no_job_role: Path = DEFAULT_PROXY_NO_JOB_ROLE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_registry: bool = True,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df = build_comparison(base_cv, job_role_free_cv, proxy_with_job_role, proxy_no_job_role)

    outputs = {
        "comparison": output_dir / "job_role_sensitivity_comparison.csv",
        "interpretation": output_dir / "job_role_sensitivity_interpretation.md",
    }
    comparison_df.to_csv(outputs["comparison"], index=False)
    outputs["interpretation"].write_text(interpretation_text(comparison_df), encoding="utf-8")

    if write_registry:
        append_registry_row(
            {
                "run_id": f"job_role_sensitivity_report_{utc_now_iso()}",
                "date_time": utc_now_iso(),
                "git_commit_if_available": get_git_commit(),
                "script": "python -m src.experiments.job_role_sensitivity",
                "config": "configs/feature_sets.yaml; reports/leakage_safe_cv_config; reports/leakage_safe_cv_job_role_sensitivity; reports/fairness/proxy_*",
                "feature_set": "no_salary_hike_no_attrition_no_department; no_salary_hike_no_attrition_no_department_no_job_role",
                "model": "summary_report",
                "seed": "source_reports",
                "cv_strategy": "source_reports",
                "primary_metrics": comparison_df.to_dict(orient="records"),
                "output_dir": str(output_dir.relative_to(SETTINGS.project_root)) if output_dir.is_relative_to(SETTINGS.project_root) else str(output_dir),
                "notes": "Joined performance and department-proxy evidence for EmpJobRole sensitivity.",
                "decision_status": "needs_review",
            }
        )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare performance/proxy tradeoff for job-role-free sensitivity.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = run_job_role_sensitivity_report(output_dir=args.output_dir, write_registry=not args.no_registry)
    print("Job-role sensitivity outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
