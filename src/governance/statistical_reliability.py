from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.core.reporting import markdown_table, write_markdown
from src.core.run_registry import RunRegistryEntry, append_run_entry
from src.utils.config import SETTINGS
from src.utils.experiment_registry import utc_now_iso


OUTPUT_DIR = SETTINGS.reports_dir / "statistical_reliability"


def run(output_dir: Path = OUTPUT_DIR) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    performance_path = write_performance_ci(output_dir / "performance_ci.csv")
    fairness_path = write_fairness_ci(output_dir / "fairness_disparity_ci.csv")
    llm_path = write_llm_guardrail_ci(output_dir / "llm_guardrail_ci.csv")
    summary_path = output_dir / "uncertainty_summary.md"
    write_uncertainty_summary(performance_path, fairness_path, llm_path, summary_path)
    append_run_entry(
        RunRegistryEntry(
            run_id=f"statistical_reliability_{utc_now_iso()}",
            command="python -m src.governance.statistical_reliability",
            dataset="multiple",
            model="xgboost; llm_agent_eval; chatbot",
            output_files=[str(performance_path), str(fairness_path), str(llm_path), str(summary_path)],
        )
    )
    return {
        "performance_ci": performance_path,
        "fairness_disparity_ci": fairness_path,
        "llm_guardrail_ci": llm_path,
        "summary": summary_path,
    }


def write_performance_ci(path: Path) -> Path:
    source = SETTINGS.reports_dir / "leakage_safe_cv_config" / "fold_metrics.csv"
    rows: List[Dict[str, Any]] = []
    if source.exists():
        df = pd.read_csv(source)
        subset = df[(df["feature_set"] == "no_salary_hike_no_attrition_no_department") & (df["model"] == "xgboost")]
        for metric in ["macro_f1", "balanced_accuracy", "quadratic_weighted_kappa", "ordinal_mae", "nll_log_loss", "multiclass_brier", "ece_confidence"]:
            if metric not in subset.columns:
                continue
            values = subset[metric].dropna().astype(float)
            rows.append(ci_row("inx_primary", metric, values, str(source), "fold_level_normal_approximation"))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_fairness_ci(path: Path) -> Path:
    source = SETTINGS.reports_dir / "fairness" / "feature_set_sensitivity" / "bootstrap_disparity_ci.csv"
    if source.exists():
        df = pd.read_csv(source)
        subset = df[df["feature_set"] == "no_salary_hike_no_attrition_no_department"].copy()
        cols = [
            "feature_set",
            "attribute",
            "metric",
            "class_label",
            "point_estimate",
            "ci_low",
            "ci_high",
            "bootstrap_std",
            "n_boot_valid",
            "min_group_support_threshold",
        ]
        subset = subset[[col for col in cols if col in subset.columns]]
        subset["source_file"] = str(source)
        subset.to_csv(path, index=False)
    else:
        pd.DataFrame(
            [{"status": "evidence_missing", "source_file": str(source), "notes": "Fairness bootstrap CI file not found."}]
        ).to_csv(path, index=False)
    return path


def write_llm_guardrail_ci(path: Path) -> Path:
    rows: List[Dict[str, Any]] = []
    faith_path = SETTINGS.reports_dir / "llm_explanations" / "faithfulness_eval.csv"
    guard_path = SETTINGS.reports_dir / "chatbot_eval" / "guardrail_evaluation.csv"
    eval_mode = read_llm_eval_mode()
    if faith_path.exists():
        faith = pd.read_csv(faith_path)
        rows.append(with_eval_mode(binomial_ci_row("faithfulness_pass_rate", int(faith["faithfulness_pass"].sum()), len(faith), str(faith_path)), eval_mode))
        unsupported_fail = ((faith["unsupported_metric_count"] + faith["unsupported_feature_count"]) == 0).sum()
        rows.append(with_eval_mode(binomial_ci_row("no_unsupported_claim_rate", int(unsupported_fail), len(faith), str(faith_path)), eval_mode))
    else:
        rows.append(with_eval_mode({"metric": "faithfulness_pass_rate", "status": "evidence_missing", "source_file": str(faith_path)}, eval_mode))
    if guard_path.exists():
        guard = pd.read_csv(guard_path)
        unsafe = guard[guard["expected_behavior"] == "refuse_with_safe_alternative"]
        safe = guard[guard["expected_behavior"] == "answer_with_governance_warnings"]
        if not unsafe.empty:
            rows.append(with_eval_mode(binomial_ci_row("unsafe_refusal_rate", int(unsafe["refused"].sum()), len(unsafe), str(guard_path)), eval_mode))
        if not safe.empty:
            rows.append(with_eval_mode(binomial_ci_row("safe_answer_pass_rate", int(safe["pass"].sum()), len(safe), str(guard_path)), eval_mode))
    else:
        rows.append(with_eval_mode({"metric": "unsafe_refusal_rate", "status": "evidence_missing", "source_file": str(guard_path)}, eval_mode))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def read_llm_eval_mode() -> Dict[str, Any]:
    summary_path = SETTINGS.reports_dir / "llm_explanations" / "llm_agent_eval_summary.csv"
    if not summary_path.exists():
        return {"run_mode": "unknown", "real_llm_used": ""}
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return {"run_mode": "unknown", "real_llm_used": ""}
    row = summary.iloc[-1].to_dict()
    return {
        "run_mode": row.get("run_mode", "unknown"),
        "real_llm_used": row.get("real_llm_used", ""),
    }


def with_eval_mode(row: Dict[str, Any], eval_mode: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(row)
    enriched["run_mode"] = eval_mode.get("run_mode", "unknown")
    enriched["real_llm_used"] = eval_mode.get("real_llm_used", "")
    return enriched


def ci_row(dataset: str, metric: str, values: pd.Series, source_file: str, method: str) -> Dict[str, Any]:
    n = int(values.shape[0])
    if n < 5:
        return {
            "dataset": dataset,
            "metric": metric,
            "n": n,
            "mean": float(values.mean()) if n else "",
            "std": float(values.std(ddof=1)) if n > 1 else "",
            "ci_low": "",
            "ci_high": "",
            "method": "insufficient_sample_size",
            "source_file": source_file,
        }
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    half_width = 1.96 * std / math.sqrt(n)
    return {
        "dataset": dataset,
        "metric": metric,
        "n": n,
        "mean": mean,
        "std": std,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "method": method,
        "source_file": source_file,
    }


def binomial_ci_row(metric: str, successes: int, n: int, source_file: str) -> Dict[str, Any]:
    if n < 5:
        return {
            "metric": metric,
            "successes": successes,
            "n": n,
            "rate": float(successes / n) if n else "",
            "ci_low": "",
            "ci_high": "",
            "method": "insufficient_sample_size",
            "source_file": source_file,
        }
    z = 1.96
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return {
        "metric": metric,
        "successes": successes,
        "n": n,
        "rate": phat,
        "ci_low": max(0.0, center - half),
        "ci_high": min(1.0, center + half),
        "method": "wilson_95_ci",
        "source_file": source_file,
    }


def write_uncertainty_summary(performance_path: Path, fairness_path: Path, llm_path: Path, summary_path: Path) -> None:
    perf = pd.read_csv(performance_path) if performance_path.exists() else pd.DataFrame()
    fair = pd.read_csv(fairness_path) if fairness_path.exists() else pd.DataFrame()
    llm = pd.read_csv(llm_path) if llm_path.exists() else pd.DataFrame()
    real_llm_values = set(str(value).lower() for value in llm.get("real_llm_used", pd.Series(dtype=str)).dropna().unique())
    run_modes = set(str(value).lower() for value in llm.get("run_mode", pd.Series(dtype=str)).dropna().unique())
    dry_run_note = ""
    if "false" in real_llm_values or "dry_run" in run_modes or "offline" in run_modes:
        dry_run_note = (
            "The current LLM faithfulness interval is computed from dry-run/offline-stub outputs. "
            "It validates the pipeline and deterministic compliance checks, but it is not manuscript-grade real LLM evidence."
        )
    lines = [
        "# Statistical Reliability and Uncertainty Summary",
        "",
        "This report uses available fold-level, bootstrap, and binomial evidence. It does not impute missing uncertainty estimates.",
        "",
        dry_run_note,
        "",
        "## Performance CI",
        "",
        *markdown_table(perf.head(20)),
        "",
        "## Fairness Disparity CI",
        "",
        *markdown_table(fair.head(20)),
        "",
        "## LLM / Guardrail CI",
        "",
        *markdown_table(llm),
        "",
        "## Limitations",
        "",
        "- Fold-level CIs use a simple normal approximation over folds and should be interpreted as descriptive uncertainty.",
        "- Fairness CIs use existing bootstrap outputs where available.",
        "- LLM and chatbot CIs are binomial technical-evaluation intervals, not human-study estimates.",
        "- Small samples are marked insufficient rather than forced into misleading intervals.",
    ]
    write_markdown(summary_path, lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate statistical reliability summaries.")
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    print(run())
