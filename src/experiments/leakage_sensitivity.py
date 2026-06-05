from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from src.models.evaluate import leakage_sensitivity_index
from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


DEFAULT_SOURCE_DIR = SETTINGS.reports_dir / "robustness" / "leakage_ablation"
DEFAULT_OUTPUT_DIR = SETTINGS.reports_dir / "leakage"
PRIMARY_METRICS = ["macro_f1", "quadratic_weighted_kappa"]


def metric_mean_column(metric: str) -> str:
    return f"{metric}_mean"


def compute_lsi_table(
    summary_df: pd.DataFrame,
    full_feature_name: str = "all_features",
    comparison_feature_sets: Iterable[str] = (
        "no_salary_hike",
        "no_salary_hike_no_attrition",
        "no_salary_hike_no_attrition_no_department",
    ),
    metrics: Iterable[str] = PRIMARY_METRICS,
) -> pd.DataFrame:
    rows = []
    comparison_set = list(comparison_feature_sets)

    for model_name, model_df in summary_df.groupby("model"):
        full_rows = model_df[model_df["feature_set"] == full_feature_name]
        if full_rows.empty:
            continue

        full_row = full_rows.iloc[0]

        for feature_set in comparison_set:
            safe_rows = model_df[model_df["feature_set"] == feature_set]
            if safe_rows.empty:
                continue
            safe_row = safe_rows.iloc[0]

            for metric in metrics:
                col = metric_mean_column(metric)
                if col not in summary_df.columns:
                    continue
                full_score = float(full_row[col])
                safe_score = float(safe_row[col])
                lsi = leakage_sensitivity_index(full_score, safe_score)
                rows.append(
                    {
                        "model": model_name,
                        "metric": metric,
                        "full_feature_set": full_feature_name,
                        "comparison_feature_set": feature_set,
                        "full_feature_score": full_score,
                        "comparison_score": safe_score,
                        "absolute_drop": lsi["absolute_drop"],
                        "lsi": lsi["lsi"],
                        "interpretation": "higher_lsi_means_stronger_dependence_on_leakage_risk_features",
                    }
                )

    return pd.DataFrame(rows).sort_values(["metric", "comparison_feature_set", "model"])


def plot_performance_drop(lsi_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = lsi_df[lsi_df["metric"].isin(PRIMARY_METRICS)].copy()
    if plot_df.empty:
        return

    labels = [
        f"{row.model}\n{row.comparison_feature_set}\n{row.metric}"
        for row in plot_df.itertuples(index=False)
    ]
    values = plot_df["absolute_drop"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 6))
    ax.bar(range(len(values)), values)
    ax.set_ylabel("Absolute performance drop")
    ax.set_title("Leakage Sensitivity: Full-Feature Upper Bound vs Reduced Feature Sets")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def interpretation_text(lsi_df: pd.DataFrame, source_summary: Path) -> str:
    try:
        source_display = source_summary.relative_to(SETTINGS.project_root)
    except ValueError:
        source_display = source_summary

    primary = lsi_df[
        (lsi_df["metric"] == "macro_f1")
        & (lsi_df["comparison_feature_set"] == "no_salary_hike_no_attrition")
    ].sort_values("lsi", ascending=False)

    lines = [
        "# Leakage Ablation Interpretation",
        "",
        "This report derives Leakage Sensitivity Index (LSI) values from existing cross-validation ablation outputs.",
        "",
        f"Source summary: `{source_display}`",
        "",
        "Full-feature results are upper-bound/leakage-warning baselines, not deployable HR decision-support models.",
        "",
        "LSI is computed as:",
        "",
        "`LSI(metric) = (score_full_feature - score_leakage_safe) / abs(score_full_feature)`",
        "",
        "A higher LSI indicates stronger model dependence on leakage-risk or outcome-proximal features. This is evidence of sensitivity to leakage-risk variables, not proof of causal leakage.",
        "",
        "## Primary macro-F1 LSI vs no_salary_hike_no_attrition",
        "",
    ]

    if primary.empty:
        lines.append("No matching rows were available for the primary leakage-safe comparison.")
    else:
        for row in primary.itertuples(index=False):
            lines.append(
                f"- {row.model}: full={row.full_feature_score:.4f}, "
                f"safe={row.comparison_score:.4f}, "
                f"absolute_drop={row.absolute_drop:.4f}, LSI={row.lsi:.4f}"
            )

    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "`EmpLastSalaryHikePercent` and `Attrition` must remain excluded from primary decision-support feature sets. The full-feature benchmark should be presented as a cautionary upper bound showing how apparent performance can inflate when outcome-proximal variables are available.",
            "",
            "## Required follow-up",
            "",
            "- Regenerate this report from a config-backed leakage ablation script.",
            "- Add synthetic leakage stress tests.",
            "- Compare explanation shift between full-feature and leakage-safe models.",
            "- Do not select a final model until calibration, fairness, SHAP stability, and actionability evidence are reviewed.",
        ]
    )

    return "\n".join(lines) + "\n"


def run_leakage_sensitivity_report(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_registry: bool = True,
) -> dict[str, Path]:
    source_summary = source_dir / "leakage_ablation_summary.csv"
    source_folds = source_dir / "leakage_ablation_fold_metrics.csv"

    if not source_summary.exists():
        raise FileNotFoundError(f"Required source summary not found: {source_summary}")
    if not source_folds.exists():
        raise FileNotFoundError(f"Required source fold metrics not found: {source_folds}")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(source_summary)
    lsi_df = compute_lsi_table(summary_df)

    outputs = {
        "fold_metrics": output_dir / "leakage_ablation_fold_metrics.csv",
        "summary": output_dir / "leakage_ablation_summary.csv",
        "lsi": output_dir / "leakage_sensitivity_index.csv",
        "interpretation": output_dir / "leakage_ablation_interpretation.md",
        "figure": figures_dir / "leakage_performance_drop.png",
    }

    shutil.copyfile(source_folds, outputs["fold_metrics"])
    shutil.copyfile(source_summary, outputs["summary"])
    lsi_df.to_csv(outputs["lsi"], index=False)
    outputs["interpretation"].write_text(interpretation_text(lsi_df, source_summary), encoding="utf-8")
    plot_performance_drop(lsi_df, outputs["figure"])

    if write_registry:
        append_registry_row(
            {
                "run_id": "leakage_sensitivity_existing_ablation_20260605",
                "date_time": utc_now_iso(),
                "git_commit_if_available": get_git_commit(),
                "script": "python -m src.experiments.leakage_sensitivity",
                "config": "reports/robustness/leakage_ablation/leakage_ablation_summary.csv",
                "feature_set": "multiple",
                "model": "catboost; lightgbm; xgboost",
                "seed": "source_report",
                "cv_strategy": "source_report",
                "primary_metrics": "macro_f1_lsi; quadratic_weighted_kappa_lsi",
                "output_dir": str(output_dir.relative_to(SETTINGS.project_root)),
                "notes": "Derived LSI report from existing leakage ablation outputs. No model training was run.",
                "decision_status": "candidate",
            }
        )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Leakage Sensitivity Index reports from existing ablation outputs.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = run_leakage_sensitivity_report(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        write_registry=not args.no_registry,
    )
    print("Leakage sensitivity outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
