from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.governance.external_claims import (
    external_allowed_claim,
    external_claim_boundary,
    validate_configured_claim,
)
from src.llm.usage_logger import USAGE_LOG_PATH
from src.models.task_schema import (
    BINARY_ATTRITION_TRANSFER,
    BINARY_TURNOVER_TRANSFER,
    ORDINAL_MULTICLASS_PERFORMANCE,
    RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS,
    apply_metric_applicability,
    get_task_schema,
)
from src.utils.config import SETTINGS
from src.utils.config_loader import load_config
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


REPORT_ROOT = SETTINGS.reports_dir / "external_validation"
MANUSCRIPT_DIR = SETTINGS.reports_dir / "manuscript_assets"
GOVERNANCE_DIR = SETTINGS.reports_dir / "governance_reports"


DATASET_RUNS = [
    {
        "key": "inx_primary",
        "display": "INX primary model",
        "path": None,
        "role": "internal primary research benchmark",
        "task_type": ORDINAL_MULTICLASS_PERFORMANCE,
        "target_kind": "primary",
        "target": "PerformanceRating 2/3/4",
        "policy": "no_salary_hike_no_attrition_no_department",
    },
    {
        "key": "hrdataset_v14",
        "display": "HRDataset_v14",
        "path": REPORT_ROOT / "hrdataset_v14",
        "role": external_allowed_claim("hrdataset_v14"),
        "task_type": ORDINAL_MULTICLASS_PERFORMANCE,
        "target_kind": "primary",
        "target": "PerformanceScore mapped to 2/3/4",
        "policy": "department_free",
    },
    {
        "key": "ibm_hr_analytics",
        "display": "IBM HR Analytics performance",
        "path": REPORT_ROOT / "ibm_hr_analytics",
        "role": external_allowed_claim("ibm_hr_analytics"),
        "task_type": RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS,
        "target_kind": "primary",
        "target": "PerformanceRating restricted to 3/4",
        "policy": "department_free",
    },
    {
        "key": "ibm_hr_analytics_attrition",
        "display": "IBM HR Analytics attrition",
        "path": REPORT_ROOT / "ibm_hr_analytics_attrition",
        "role": external_allowed_claim("ibm_hr_analytics", "attrition"),
        "task_type": BINARY_ATTRITION_TRANSFER,
        "target_kind": "attrition",
        "target": "Attrition mapped No/Yes to 0/1",
        "policy": "department_free",
    },
    {
        "key": "employee_turnover",
        "display": "Employee Turnover",
        "path": REPORT_ROOT / "employee_turnover",
        "role": external_allowed_claim("employee_turnover"),
        "task_type": BINARY_TURNOVER_TRANSFER,
        "target_kind": "primary",
        "target": "left 0/1",
        "policy": "without_last_evaluation",
    },
]


def run() -> Dict[str, Path]:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    GOVERNANCE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    tables = build_tables(write_usage=True)
    summary_path = REPORT_ROOT / "external_validation_summary.md"
    manuscript_path = MANUSCRIPT_DIR / "external_validation_tables.md"
    governance_path = GOVERNANCE_DIR / "external_validation_governance_summary.md"
    roles_path = REPORT_ROOT / "external_dataset_roles.csv"

    summary_path.write_text(summary_markdown(tables), encoding="utf-8")
    manuscript_path.write_text(manuscript_markdown(tables), encoding="utf-8")
    governance_path.write_text(governance_markdown(tables), encoding="utf-8")
    write_external_dataset_roles(roles_path)

    append_registry_row(
        {
            "run_id": f"external_validation_reports_{utc_now_iso()}",
            "date_time": utc_now_iso(),
            "git_commit_if_available": get_git_commit(),
            "script": "python -m src.governance.external_validation_reports",
            "config": "generated external validation artifacts",
            "feature_set": "external policies; see reports",
            "model": "xgboost; OpenAI governed explanation and agents",
            "seed": "42; LLM temperature 0",
            "cv_strategy": "artifact aggregation",
            "primary_metrics": "external replication and robustness summary tables",
            "output_dir": "reports/external_validation; reports/manuscript_assets; reports/governance_reports",
            "notes": "Generated integrated external replication, robustness, manuscript, and governance summaries.",
            "decision_status": "candidate",
        }
    )
    return {
        "summary": summary_path,
        "manuscript": manuscript_path,
        "governance": governance_path,
        "external_dataset_roles": roles_path,
    }


def write_external_dataset_roles(output_path: Path) -> Path:
    config = load_config("external_validation")
    rows = []
    for row in config.get("external_validation", {}).get("datasets", []):
        dataset_name = str(row.get("dataset_name", ""))
        configured_claim = str(row.get("role_in_manuscript", ""))
        validate_configured_claim(dataset_name, "primary", configured_claim)
        task_schema = get_task_schema(str(row.get("task_type", "")))
        expected_run = next((item for item in DATASET_RUNS if item["key"] == dataset_name), None)
        if expected_run is None or task_schema.name != expected_run["task_type"]:
            expected = expected_run["task_type"] if expected_run is not None else "registered dataset"
            raise ValueError(
                f"External task schema mismatch for {dataset_name}: expected {expected}, observed {task_schema.name}."
            )
        transported = bool(row.get("locked_inx_model_transported", False))
        boundary = external_claim_boundary(dataset_name)
        if transported != boundary.transported_locked_inx_model:
            raise ValueError(
                f"Locked-model transport status mismatch for {dataset_name}: "
                f"expected {boundary.transported_locked_inx_model}, observed {transported}."
            )
        rows.append(
            {
                "dataset_name": dataset_name,
                "source_path": row.get("source_path", ""),
                "task_type": task_schema.name,
                "task_comparison_group": task_schema.comparison_group,
                "target_variable": row.get("target_variable", ""),
                "target_semantics": row.get("target_semantics", ""),
                "comparable_to_inx_performance": bool(row.get("comparable_to_inx_performance", False)),
                "locked_inx_model_transported": transported,
                "role_in_manuscript": configured_claim,
                "limitations": row.get("limitations", ""),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def build_tables(*, write_usage: bool = False) -> Dict[str, pd.DataFrame]:
    role_rows = []
    target_rows = []
    performance_rows = []
    fairness_rows = []
    calibration_rows = []
    actionability_rows = []
    llm_rows = []

    for run_info in DATASET_RUNS:
        task_schema = get_task_schema(run_info["task_type"])
        role_rows.append(
            {
                "dataset": run_info["display"],
                "role": run_info["role"],
                "task_type": task_schema.name,
                "task_comparison_group": task_schema.comparison_group,
                "target_definition": run_info["target"],
                "default_policy": run_info["policy"],
                "allowed_claim": allowed_claim_for(run_info["key"]),
            }
        )
        target_rows.append(target_row(run_info))
        performance_rows.append(performance_row(run_info))
        fairness_rows.append(fairness_row(run_info))
        calibration_rows.append(calibration_row(run_info))
        actionability_rows.append(actionability_row(run_info))
        llm_rows.append(llm_row(run_info))

    overlap_path = REPORT_ROOT / "hrdataset_v14" / "cross_dataset_inx_to_hrdataset" / "feature_overlap.csv"
    overlap = pd.read_csv(overlap_path) if overlap_path.exists() else pd.DataFrame()
    common = overlap[overlap["common"] == True]["feature"].tolist() if not overlap.empty else []
    overlap_table = pd.DataFrame(
        [
            {
                "comparison": "INX primary -> HRDataset_v14 department-free",
                "common_safe_feature_count": len(common),
                "common_features": "; ".join(common),
                "status": "infeasible/too limited" if len(common) < 5 else "completed",
            }
        ]
    )

    limitations = pd.DataFrame(
        [
            {
                "area": "Dataset provenance",
                "limitation": "External CSVs are retrieved from public mirrors and documented in dataset cards; source authenticity should be checked before publication submission.",
                "allowed_claim": "Reproducible public-mirror robustness evidence, not audited data provenance.",
            },
            {
                "area": "HRDataset_v14",
                "limitation": "Small external sample with mapped performance labels and public cross-sectional data.",
                "allowed_claim": "Independent external performance-target replication.",
            },
            {
                "area": "Cross-dataset validation",
                "limitation": "Only three common department-free safe features were available.",
                "allowed_claim": "Locked-INX-model transport is infeasible/too limited under the three-safe-feature overlap.",
            },
            {
                "area": "IBM performance",
                "limitation": "PerformanceRating contains only classes 3 and 4.",
                "allowed_claim": "Restricted-target performance robustness.",
            },
            {
                "area": "Employee turnover",
                "limitation": "Target is attrition/turnover, not performance.",
                "allowed_claim": "Related HR task-transfer robustness only.",
            },
            {
                "area": "LLM/agents",
                "limitation": "Expanded real OpenAI technical evaluation currently prioritizes INX and HRDataset_v14; IBM and turnover related-task LLM regeneration are second-stage robustness work.",
                "allowed_claim": "Technical LLM-agent governance evidence on structured ML/XAI artifacts for the stated dataset scope.",
            },
            {
                "area": "Stub/dry-run outputs",
                "limitation": "offline_stub_llm and run_mode=dry_run outputs are retained for reproducibility and pipeline tests only.",
                "allowed_claim": "Not manuscript-grade real LLM evidence; excluded from final evidence claims.",
            },
        ]
    )

    llm_df = pd.DataFrame(llm_rows)
    usage = build_external_usage_table(llm_df)
    if write_usage:
        usage.to_csv(REPORT_ROOT / "external_llm_usage_summary.csv", index=False)
    expanded_llm = expanded_llm_summary_table()
    performance = pd.DataFrame(performance_rows)

    return {
        "roles": pd.DataFrame(role_rows),
        "targets": pd.DataFrame(target_rows),
        "overlap": overlap_table,
        # Retained as a machine-readable superset only. Renderers below must use
        # the task-separated tables so binary/restricted tasks cannot appear as
        # directly equivalent to the primary three-class ordinal task.
        "performance": performance,
        "performance_ordinal": performance[
            performance["task_comparison_group"] == "three_class_ordinal_performance"
        ].reset_index(drop=True),
        "performance_restricted": performance[
            performance["task_comparison_group"] == "restricted_target_performance_robustness"
        ].reset_index(drop=True),
        "performance_binary_transfer": performance[
            performance["task_comparison_group"] == "related_binary_task_transfer"
        ].reset_index(drop=True),
        "fairness": pd.DataFrame(fairness_rows),
        "calibration": pd.DataFrame(calibration_rows),
        "actionability": pd.DataFrame(actionability_rows),
        "llm": llm_df,
        "expanded_llm": expanded_llm,
        "usage": usage,
        "limitations": limitations,
    }


def target_row(run_info: Dict[str, Any]) -> Dict[str, Any]:
    path = run_info["path"]
    if path is None:
        return {
            "dataset": run_info["display"],
            "raw_target": "PerformanceRating",
            "canonical_target": "PerformanceRating",
            "mapping": "2/3/4 unchanged",
            "class_distribution": "see INX reports/model card",
        }
    metadata_path = path / "metadata.json"
    metadata = _read_json(metadata_path)
    target_dist_path = path / "target_distribution.csv"
    dist = pd.read_csv(target_dist_path) if target_dist_path.exists() else pd.DataFrame()
    dist_text = "; ".join(f"{r.target_value}:{r.count}" for r in dist.itertuples(index=False)) if not dist.empty else ""
    return {
        "dataset": run_info["display"],
        "raw_target": metadata.get("target_raw_column", ""),
        "canonical_target": metadata.get("target_column", ""),
        "mapping": run_info["target"],
        "class_distribution": dist_text,
    }


def performance_row(run_info: Dict[str, Any]) -> Dict[str, Any]:
    schema = get_task_schema(run_info["task_type"])
    if run_info["path"] is None:
        dashboard = pd.read_csv(SETTINGS.reports_dir / "model_selection" / "final_candidate_dashboard.csv")
        row = dashboard[dashboard["feature_set"] == run_info["policy"]].iloc[0]
        raw_metrics = {
            "macro_f1": row["macro_f1"],
            "balanced_accuracy": row["balanced_accuracy"],
            "quadratic_weighted_kappa": row["qwk"],
            "ordinal_mae": row["ordinal_mae"],
            "severe_error_rate": row["severe_error_rate"],
            "nll_log_loss": row["log_loss"],
            "multiclass_brier": row["multiclass_brier"],
            "binary_brier": None,
            "roc_auc": None,
            "average_precision": None,
            "ece_confidence": row["ece"],
        }
    else:
        metrics = pd.read_csv(run_info["path"] / "performance_metrics.csv")
        row = metrics[metrics["policy"] == run_info["policy"]].iloc[0]
        observed_task = get_task_schema(str(row.get("task_type", schema.name))).name
        if observed_task != schema.name:
            raise ValueError(
                f"Task schema mismatch for {run_info['key']}: expected {schema.name}, observed {observed_task}."
            )
        raw_metrics = {
            "macro_f1": row.get("macro_f1"),
            "balanced_accuracy": row.get("balanced_accuracy"),
            "quadratic_weighted_kappa": row.get("quadratic_weighted_kappa"),
            "ordinal_mae": row.get("ordinal_mae"),
            "severe_error_rate": row.get("severe_error_rate"),
            "nll_log_loss": row.get("nll_log_loss"),
            "multiclass_brier": row.get("multiclass_brier"),
            "binary_brier": row.get("binary_brier"),
            "roc_auc": row.get("roc_auc"),
            "average_precision": row.get("average_precision"),
            "ece_confidence": row.get("ece_confidence"),
        }
    task_metrics = apply_metric_applicability(raw_metrics, schema.name)
    return {
        "dataset": run_info["display"],
        "policy": run_info["policy"],
        "task_type": schema.name,
        "task_comparison_group": schema.comparison_group,
        "metric_applicability": schema.applicability_note,
        "macro_f1": task_metrics.get("macro_f1"),
        "balanced_accuracy": task_metrics.get("balanced_accuracy"),
        "qwk": task_metrics.get("quadratic_weighted_kappa"),
        "ordinal_mae": task_metrics.get("ordinal_mae"),
        "severe_error_rate": task_metrics.get("severe_error_rate"),
        "log_loss": task_metrics.get("nll_log_loss"),
        "brier": task_metrics.get("multiclass_brier"),
        "binary_brier": task_metrics.get("binary_brier"),
        "roc_auc": task_metrics.get("roc_auc"),
        "average_precision": task_metrics.get("average_precision"),
        "ece": task_metrics.get("ece_confidence"),
    }


def fairness_row(run_info: Dict[str, Any]) -> Dict[str, Any]:
    if run_info["path"] is None:
        dashboard = pd.read_csv(SETTINGS.reports_dir / "model_selection" / "final_candidate_dashboard.csv")
        row = dashboard[dashboard["feature_set"] == run_info["policy"]].iloc[0]
        return {
            "dataset": run_info["display"],
            "policy": run_info["policy"],
            "largest_disparity_gap": row["empdepartment_macro_f1_gap"],
            "largest_disparity_attribute": "EmpDepartment macro-F1",
            "department_proxy_macro_f1": row["department_proxy_macro_f1"],
            "fairness_claim": "audit evidence only; no fairness guarantee",
        }
    disparity_path = run_info["path"] / "fairness_proxy" / "fairness_disparity_summary.csv"
    proxy_path = run_info["path"] / "department_proxy_reconstruction.csv"
    disparity = pd.read_csv(disparity_path) if disparity_path.exists() else pd.DataFrame()
    proxy = pd.read_csv(proxy_path) if proxy_path.exists() else pd.DataFrame()
    subset = disparity[disparity["feature_set"] == run_info["policy"]] if not disparity.empty and "feature_set" in disparity.columns else pd.DataFrame()
    if not subset.empty and "max_gap" in subset.columns:
        top = subset.sort_values("max_gap", ascending=False).iloc[0]
        gap = top["max_gap"]
        attr = f"{top['attribute']} {top['metric']}"
    else:
        gap = ""
        attr = "not available"
    proxy_subset = proxy[proxy["policy"] == run_info["policy"]] if not proxy.empty and "policy" in proxy.columns else pd.DataFrame()
    proxy_f1 = proxy_subset.iloc[0].get("macro_f1", "") if not proxy_subset.empty else ""
    return {
        "dataset": run_info["display"],
        "policy": run_info["policy"],
        "largest_disparity_gap": gap,
        "largest_disparity_attribute": attr,
        "department_proxy_macro_f1": proxy_f1,
        "fairness_claim": "audit evidence only; no fairness guarantee",
    }


def calibration_row(run_info: Dict[str, Any]) -> Dict[str, Any]:
    row = performance_row(run_info)
    return {
        "dataset": row["dataset"],
        "policy": row["policy"],
        "task_type": row["task_type"],
        "cross_task_comparability": "descriptive within task only",
        "log_loss": row["log_loss"],
        "multiclass_brier": row["brier"],
        "binary_brier": row["binary_brier"],
        "ece": row["ece"],
        "interpretation": "probability confidence requires calibration caution",
    }


def actionability_row(run_info: Dict[str, Any]) -> Dict[str, Any]:
    if run_info["path"] is None:
        dashboard = pd.read_csv(SETTINGS.reports_dir / "model_selection" / "final_candidate_dashboard.csv")
        row = dashboard[dashboard["feature_set"] == run_info["policy"]].iloc[0]
        return {
            "dataset": run_info["display"],
            "policy": run_info["policy"],
            "employee_controllable_share": row["employee_only_validity"],
            "manager_or_org_share": row["organization_allowed_validity"],
            "status": "counterfactual validity from INX final evidence",
        }
    path = run_info["path"] / "actionability_summary.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    subset = df[df["policy"] == run_info["policy"]] if not df.empty else pd.DataFrame()
    if subset.empty:
        return {"dataset": run_info["display"], "policy": run_info["policy"], "employee_controllable_share": "", "manager_or_org_share": "", "status": "not available"}
    row = subset.iloc[0]
    return {
        "dataset": run_info["display"],
        "policy": run_info["policy"],
        "employee_controllable_share": row.get("employee_controllable_share", ""),
        "manager_or_org_share": row.get("manager_or_organisation_share", ""),
        "status": row.get("actionability_status", ""),
    }


def llm_row(run_info: Dict[str, Any]) -> Dict[str, Any]:
    if run_info["path"] is None:
        path = SETTINGS.reports_dir / "llm_explanations" / "real_llm_eval_summary.csv"
    else:
        path = run_info["path"] / "llm_agent_governance" / "external_llm_agent_summary.csv"
    if not path.exists():
        return {
            "dataset": run_info["display"],
            "case_ids": "",
            "n_cases": 0,
            "faithfulness_pass_rate": "",
            "unsupported_claim_rate": "",
            "forbidden_claim_rate": "",
            "missing_warning_rate": "",
            "agent_success_rate": "",
            "notes": "not run",
        }
    row = pd.read_csv(path).iloc[0]
    return {
        "dataset": run_info["display"],
        "case_ids": row.get("case_ids", ""),
        "n_cases": row.get("n_cases", ""),
        "faithfulness_pass_rate": row.get("faithfulness_pass_rate", ""),
        "unsupported_claim_rate": row.get("unsupported_claim_rate", ""),
        "forbidden_claim_rate": row.get("forbidden_claim_rate", ""),
        "missing_warning_rate": row.get("missing_warning_rate", ""),
        "agent_success_rate": row.get("agent_success_rate", ""),
        "notes": "real OpenAI + OpenAI Agents SDK" if int(row.get("n_cases", 0)) else "not run",
    }


def build_external_usage_table(llm_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["dataset", "logged_usage_rows", "input_tokens", "output_tokens", "total_tokens", "estimated_cost_usd"]
    if not USAGE_LOG_PATH.exists() or llm_df.empty:
        return pd.DataFrame(columns=columns)

    usage = pd.read_csv(USAGE_LOG_PATH)
    rows = []
    for row in llm_df.itertuples(index=False):
        case_ids = [part for part in str(getattr(row, "case_ids", "")).split(";") if part]
        if not case_ids:
            rows.append(
                {
                    "dataset": row.dataset,
                    "logged_usage_rows": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                }
            )
            continue
        subset = usage[usage["case_id"].astype(str).isin(case_ids)].copy()
        rows.append(
            {
                "dataset": row.dataset,
                "logged_usage_rows": int(len(subset)),
                "input_tokens": int(pd.to_numeric(subset.get("input_tokens", 0), errors="coerce").fillna(0).sum()),
                "output_tokens": int(pd.to_numeric(subset.get("output_tokens", 0), errors="coerce").fillna(0).sum()),
                "total_tokens": int(pd.to_numeric(subset.get("total_tokens", 0), errors="coerce").fillna(0).sum()),
                "estimated_cost_usd": float(pd.to_numeric(subset.get("estimated_cost_usd", 0), errors="coerce").fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def expanded_llm_summary_table() -> pd.DataFrame:
    summary_path = SETTINGS.reports_dir / "llm_explanations" / "llm_agent_eval_summary.csv"
    manifest_path = SETTINGS.reports_dir / "llm_explanations" / "eval_case_manifest.csv"
    columns = [
        "dataset",
        "n_cases",
        "run_mode",
        "real_llm_used",
        "llm_model",
        "faithfulness_pass_rate",
        "unsupported_claim_rate",
        "forbidden_claim_rate",
        "missing_warning_rate",
        "parsing_success_rate",
        "agent_compliance_pass_rate",
        "notes",
    ]
    if not summary_path.exists() or not manifest_path.exists():
        return pd.DataFrame(columns=columns)
    summary = pd.read_csv(summary_path)
    manifest = pd.read_csv(manifest_path)
    if summary.empty or manifest.empty:
        return pd.DataFrame(columns=columns)
    row = summary.iloc[0]
    out = []
    for dataset, subset in manifest.groupby("dataset_name"):
        out.append(
            {
                "dataset": dataset,
                "n_cases": int(len(subset)),
                "run_mode": row.get("run_mode", ""),
                "real_llm_used": row.get("real_llm_used", ""),
                "llm_model": row.get("llm_model_used", ""),
                "faithfulness_pass_rate": row.get("faithfulness_pass_rate", ""),
                "unsupported_claim_rate": row.get("unsupported_claim_rate", ""),
                "forbidden_claim_rate": row.get("forbidden_claim_rate", ""),
                "missing_warning_rate": row.get("missing_warning_rate", ""),
                "parsing_success_rate": row.get("parsing_success_rate", ""),
                "agent_compliance_pass_rate": row.get("agent_compliance_pass_rate", ""),
                "notes": "expanded real priority scope; IBM/attrition/turnover LLM regeneration deferred"
                if str(row.get("real_llm_used", "")).lower() in {"true", "1"}
                else "dry-run/stub; not manuscript-grade real LLM evidence",
            }
        )
    return pd.DataFrame(out, columns=columns)


def allowed_claim_for(key: str) -> str:
    if key == "inx_primary":
        return "internal benchmark only"
    lookup = {
        "hrdataset_v14": ("hrdataset_v14", "primary"),
        "ibm_hr_analytics": ("ibm_hr_analytics", "primary"),
        "ibm_hr_analytics_attrition": ("ibm_hr_analytics", "attrition"),
        "employee_turnover": ("employee_turnover", "primary"),
    }
    try:
        dataset_name, target_kind = lookup[key]
    except KeyError as exc:
        raise ValueError(f"No allowed external claim registered for '{key}'.") from exc
    return external_allowed_claim(dataset_name, target_kind)


def summary_markdown(tables: Dict[str, pd.DataFrame]) -> str:
    lines = [
        "# External Replication and Robustness Evidence Summary",
        "",
        "This report extends the INX HR XAI evidence with independent external performance-target replication, restricted-target robustness, and related-task transfer. Claims are intentionally conservative: research-grade decision support only, no autonomous HR decisions, no causal SHAP claims, no fairness guarantee, and no deployment readiness claim.",
        "",
        "## Dataset Roles",
        "",
        markdown_table(tables["roles"]),
        "",
        "## Target Mapping",
        "",
        markdown_table(tables["targets"]),
        "",
        "## Feature Overlap",
        "",
        markdown_table(tables["overlap"]),
        "",
        "## Three-Class Ordinal Performance Evidence",
        "",
        "HRDataset_v14 uses an independently trained dataset-specific model; this table is not evidence of locked-INX-model transport.",
        "",
        markdown_table(_round_table(tables["performance_ordinal"])),
        "",
        "## Restricted-Target Performance Robustness",
        "",
        "The 3/4-only target is reported separately because its metric support is not equivalent to the primary 2/3/4 task.",
        "",
        markdown_table(_round_table(tables["performance_restricted"])),
        "",
        "## Related Binary Task Transfer",
        "",
        "Attrition and turnover are different targets. Ordinal and severe-error metrics are N/A by task contract.",
        "",
        markdown_table(_round_table(tables["performance_binary_transfer"])),
        "",
        "## Fairness / Proxy Evidence by Dataset",
        "",
        markdown_table(_round_table(tables["fairness"])),
        "",
        "## Calibration Evidence by Non-Comparable Task",
        "",
        markdown_table(_round_table(tables["calibration"])),
        "",
        "## Actionability Evidence by Dataset",
        "",
        markdown_table(_round_table(tables["actionability"])),
        "",
        "## LLM / Agent Governance Evidence by Dataset",
        "",
        markdown_table(_round_table(tables["llm"])),
        "",
        "## Expanded Real LLM-Agent Evaluation: Current Priority Scope",
        "",
        "This table is read from the current config-driven real run. It prioritizes INX and HRDataset_v14; related-task datasets do not validate employee-performance models.",
        "",
        "Stub/dry-run LLM outputs are not manuscript-grade real LLM evidence and are excluded from this expanded evidence scope.",
        "",
        markdown_table(_round_table(tables["expanded_llm"])),
        "",
        "## External LLM Usage / Cost Summary",
        "",
        "Usage is cumulative for the listed case IDs and includes remediation reruns; billing dashboards remain the source of truth.",
        "",
        markdown_table(_round_table(tables["usage"])),
        "",
        "## Explicit Limitations",
        "",
        markdown_table(tables["limitations"]),
        "",
    ]
    return "\n".join(lines)


def manuscript_markdown(tables: Dict[str, pd.DataFrame]) -> str:
    sections = [
        ("Table 1: Dataset Roles and Target Definitions", tables["roles"]),
        ("Table 2: External Schema Mapping Summary", tables["targets"]),
        ("Table 3a: Three-Class Ordinal Performance Evidence", _round_table(tables["performance_ordinal"])),
        ("Table 3b: Restricted-Target Performance Robustness", _round_table(tables["performance_restricted"])),
        ("Table 3c: Related Binary Task Transfer", _round_table(tables["performance_binary_transfer"])),
        ("Table 4: Leakage/Proxy/Fairness Findings", _round_table(tables["fairness"])),
        ("Table 5a: Expanded Real LLM-Agent Results for Priority Scope", _round_table(tables["expanded_llm"])),
        ("Table 5: LLM-Agent Governance Results Across Datasets", _round_table(tables["llm"])),
        ("Table 5b: External LLM Usage and Estimated Cost", _round_table(tables["usage"])),
        ("Table 6: Limitations and Allowed Claims", tables["limitations"]),
    ]
    lines = [
        "# External Replication and Robustness Manuscript Tables",
        "",
        "Metric tables are separated by task schema; values must not be compared across the three sections as if they evaluated the same target.",
        "",
    ]
    for title, table in sections:
        lines.extend([f"## {title}", "", markdown_table(table), ""])
    return "\n".join(lines)


def governance_markdown(tables: Dict[str, pd.DataFrame]) -> str:
    return "\n".join(
        [
            "# External Replication and Robustness Governance Summary",
            "",
            "## What Can Be Claimed",
            "",
            "- HRDataset_v14 provides independent external performance-target replication using an independently trained dataset-specific model.",
            "- IBM HR Analytics provides restricted-target performance robustness for classes 3 and 4.",
            "- IBM attrition and Employee Turnover provide related HR task-transfer evidence only; they do not validate employee-performance models.",
            "- Real OpenAI governed explanations now include an expanded 80-case priority-scope batch for INX and HRDataset_v14.",
            "- Earlier small real OpenAI and OpenAI Agents SDK audits exist for HRDataset_v14, IBM performance robustness, and Employee Turnover, but the related-task datasets remain robustness evidence only.",
            "",
            "## What Cannot Be Claimed",
            "",
            "- No autonomous hiring, firing, promotion, compensation, discipline, or individual employment decision capability.",
            "- No causal interpretation of SHAP or counterfactuals.",
            "- No proof of fairness from removing sensitive or group variables.",
            "- IBM and turnover results cannot support employee-performance model-validation claims.",
            "- No deployment readiness without independent data provenance review, human validation, legal review, and organisation-specific governance.",
            "",
            "## Remaining Deployment Blockers",
            "",
            "- External data provenance and licensing should be independently verified before publication.",
            "- Cross-dataset INX-to-HRDataset feature overlap is too weak for a defensible transportability result.",
            "- Subgroup results remain sample-size and support-threshold sensitive.",
            "- LLM-agent evaluation is automated technical evidence, not human-subject validation.",
            "- Stub/dry-run LLM outputs are not manuscript-grade real LLM evidence and are excluded from final evidence claims.",
            "",
            "## Q3 Manuscript Positioning",
            "",
            "The project can be positioned as a leakage-aware, fairness/proxy-audited, calibration-aware, actionability-constrained, SHAP/XAI, LLM-assisted governance framework with independent external replication and related-task robustness. It should not be positioned as a deployable HR decision system.",
            "",
            "## Supporting Tables",
            "",
            "### Three-Class Ordinal Performance Evidence",
            "",
            markdown_table(_round_table(tables["performance_ordinal"])),
            "",
            "### Restricted-Target Performance Robustness",
            "",
            markdown_table(_round_table(tables["performance_restricted"])),
            "",
            "### Related Binary Task Transfer",
            "",
            markdown_table(_round_table(tables["performance_binary_transfer"])),
            "",
            markdown_table(_round_table(tables["expanded_llm"])),
            "",
            markdown_table(_round_table(tables["llm"])),
            "",
            "## External LLM Usage / Cost Summary",
            "",
            "Usage is cumulative for the listed case IDs and includes remediation reruns; billing dashboards remain the source of truth.",
            "",
            markdown_table(_round_table(tables["usage"])),
            "",
        ]
    )


def _round_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(6)
    return out


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(_escape_cell(col) for col in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        values = [_escape_cell(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _escape_cell(value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print(run())
