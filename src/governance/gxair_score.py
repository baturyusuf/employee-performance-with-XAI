from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.core.reporting import markdown_table, write_markdown
from src.core.run_registry import RunRegistryEntry, append_run_entry
from src.utils.config import SETTINGS
from src.utils.config_loader import load_config
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


OUTPUT_DIR = SETTINGS.reports_dir / "governance_reports"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def load_primary_dashboard() -> pd.Series:
    path = SETTINGS.reports_dir / "model_selection" / "final_candidate_dashboard.csv"
    df = pd.read_csv(path)
    return df[df["feature_set"] == "no_salary_hike_no_attrition_no_department"].iloc[0]


def load_llm_eval() -> Dict[str, float]:
    eval_path = SETTINGS.reports_dir / "llm_explanations" / "llm_governance_eval_summary.csv"
    if eval_path.exists():
        row = pd.read_csv(eval_path).iloc[0].to_dict()
        return {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
    detail_path = SETTINGS.reports_dir / "llm_explanations" / "governed_explanation_eval.csv"
    if detail_path.exists():
        df = pd.read_csv(detail_path)
        return {
            "faithfulness_pass_rate": float(df["faithfulness_pass"].mean()),
            "unsupported_claim_rate": float((df["unsupported_claim_count"] > 0).mean()),
            "forbidden_claim_rate": float((df["forbidden_claim_count"] > 0).mean()),
            "missing_warning_rate": float((df["missing_warning_count"] > 0).mean()),
        }
    return {}


def compute_component_scores() -> pd.DataFrame:
    row = load_primary_dashboard()
    llm_eval = load_llm_eval()
    performance = clamp01((row["macro_f1"] + row["balanced_accuracy"] + row["qwk"] + (1 - row["ordinal_mae"])) / 4)
    leakage = 1.0 if row["salary_hike_excluded"] and row["attrition_excluded"] else 0.0
    explanation = clamp01((row["top10_shap_jaccard"] + row["shap_spearman"]) / 2)
    calibration = clamp01(1 - ((row["ece"] + row["multiclass_brier"]) / 2))
    fairness = clamp01(1 - row["empdepartment_macro_f1_gap"])
    actionability = clamp01((row["employee_only_validity"] + row["employee_manager_validity"] + row["organization_allowed_validity"]) / 3)
    proxy_penalty = clamp01(1 - row["department_proxy_macro_f1"])
    llm_score = clamp01(
        llm_eval.get("faithfulness_pass_rate", 0.0)
        * (1 - llm_eval.get("unsupported_claim_rate", 1.0))
        * (1 - llm_eval.get("forbidden_claim_rate", 1.0))
        * (1 - llm_eval.get("missing_warning_rate", 1.0))
    )
    rows = [
        ("Performance Adequacy Score", performance, "Macro-F1, balanced accuracy, QWK, ordinal MAE."),
        ("Leakage Robustness Score", leakage, "Salary hike and attrition exclusion status."),
        ("Explanation Stability Score", explanation, "Top-k Jaccard and Spearman SHAP stability."),
        ("Calibration Reliability Score", calibration, "ECE and Brier transformed so higher is better."),
        ("Fairness Robustness Score", fairness, "Support-filtered EmpDepartment macro-F1 gap transformed so higher is better."),
        ("Actionability Realism Score", actionability, "Employee, manager, and organisation-validity rates."),
        ("Proxy Risk Penalty Component", proxy_penalty, "Inverse of department reconstruction macro-F1; low is risky."),
        ("LLM Governance Compliance Score", llm_score, "Faithfulness pass, unsupported claim, forbidden claim, and missing warning rates."),
    ]
    return pd.DataFrame(rows, columns=["component", "score", "explanation"])


def readiness_label(scores: pd.DataFrame) -> str:
    score_map = scores.set_index("component")["score"].to_dict()
    if score_map.get("LLM Governance Compliance Score", 0) < 0.80:
        return "research_only_llm_layer_needs_review"
    if score_map.get("Proxy Risk Penalty Component", 0) < 0.30:
        return "research_only_proxy_risk_high"
    return "decision_support_with_strong_warnings"


def run() -> Dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = compute_component_scores()
    label = readiness_label(scores)
    csv_path = OUTPUT_DIR / "gxair_llm_agent_dashboard.csv"
    scores["overall_readiness_label"] = label
    scores.to_csv(csv_path, index=False)
    json_path = OUTPUT_DIR / "gxair_llm_agent_dashboard.json"
    json_path.write_text(json.dumps({"overall_readiness_label": label, "components": scores.to_dict(orient="records")}, indent=2), encoding="utf-8")
    append_registry_row(
        {
            "run_id": f"gxair_llm_agent_dashboard_{utc_now_iso()}",
            "date_time": utc_now_iso(),
            "git_commit_if_available": get_git_commit(),
            "script": "python -m src.governance.gxair_score",
            "config": "final dashboard + LLM governance eval",
            "feature_set": "no_salary_hike_no_attrition_no_department",
            "model": "gxair_dashboard",
            "seed": "deterministic",
            "cv_strategy": "not_applicable",
            "primary_metrics": "transparent G-XAIR component scores with LLM compliance",
            "output_dir": "reports/governance_reports",
            "notes": "Generated transparent G-XAIR dashboard extension with LLM governance component.",
            "decision_status": "candidate",
        }
    )
    return {"csv": str(csv_path), "json": str(json_path)}


def run_component_dashboard(config_path: str = "configs/governance_dashboard.yaml") -> Dict[str, str]:
    config = load_config(config_path)
    settings = config.get("governance_dashboard", config)
    output_dir = Path(settings.get("output_dir", OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_paths = {key: Path(value) for key, value in settings.get("evidence_paths", {}).items()}
    primary_policy = str(settings.get("primary_feature_policy", "no_salary_hike_no_attrition_no_department"))

    rows = build_component_rows(evidence_paths, primary_policy)
    dashboard = pd.DataFrame(rows)
    final_label = final_readiness_label(dashboard)
    dashboard["final_readiness_label"] = final_label
    csv_path = output_dir / "gxair_component_dashboard.csv"
    md_path = output_dir / "gxair_component_dashboard.md"
    readiness_path = output_dir / "final_governance_readiness_report.md"
    dashboard.to_csv(csv_path, index=False)
    write_component_markdown(dashboard, md_path, final_label)
    write_readiness_report(dashboard, readiness_path, final_label)

    append_run_entry(
        RunRegistryEntry(
            run_id=f"gxair_component_dashboard_{utc_now_iso()}",
            command=f"python -m src.governance.gxair_score --config {config_path}",
            config_path=config_path,
            dataset=str(settings.get("primary_dataset", "inx_primary")),
            model=str(settings.get("model_name", "xgboost")),
            feature_policy=primary_policy,
            output_files=[str(csv_path), str(md_path), str(readiness_path)],
        )
    )
    append_registry_row(
        {
            "run_id": f"gxair_component_dashboard_{utc_now_iso()}",
            "date_time": utc_now_iso(),
            "git_commit_if_available": get_git_commit(),
            "script": "python -m src.governance.gxair_score --config",
            "config": config_path,
            "feature_set": primary_policy,
            "model": "component_governance_dashboard",
            "seed": settings.get("seed", "deterministic"),
            "cv_strategy": "not_applicable",
            "primary_metrics": {"final_readiness_label": final_label},
            "output_dir": str(output_dir),
            "notes": "Generated component-based governance readiness dashboard. Scores are omitted when not defensible.",
            "decision_status": "candidate",
        }
    )
    # Preserve legacy outputs for existing dashboard consumers.
    run()
    return {"csv": str(csv_path), "markdown": str(md_path), "readiness_report": str(readiness_path)}


def build_component_rows(evidence_paths: Dict[str, Path], primary_policy: str) -> list[Dict[str, Any]]:
    dashboard_path = evidence_paths.get("final_candidate_dashboard", SETTINGS.reports_dir / "model_selection" / "final_candidate_dashboard.csv")
    dashboard_df = pd.read_csv(dashboard_path) if dashboard_path.exists() else pd.DataFrame()
    row = pd.Series(dtype=object)
    if not dashboard_df.empty:
        subset = dashboard_df[dashboard_df["feature_set"] == primary_policy]
        if not subset.empty:
            row = subset.iloc[0]

    llm_summary_path = evidence_paths.get("llm_agent_eval_summary", SETTINGS.reports_dir / "llm_explanations" / "llm_agent_eval_summary.csv")
    llm_df = pd.read_csv(llm_summary_path) if llm_summary_path.exists() else pd.DataFrame()
    llm_row = llm_df.iloc[0] if not llm_df.empty else pd.Series(dtype=object)

    guardrail_path = evidence_paths.get("chatbot_guardrail_eval", SETTINGS.reports_dir / "chatbot_eval" / "guardrail_evaluation.csv")
    guardrail_df = pd.read_csv(guardrail_path) if guardrail_path.exists() else pd.DataFrame()

    external_path = evidence_paths.get("external_validation_summary", SETTINGS.reports_dir / "external_validation" / "external_validation_summary.md")

    rows = [
        component_from_score(
            "Performance Adequacy",
            _score_from_row(row, ["macro_f1", "balanced_accuracy", "qwk"], transform=lambda values: sum(values) / len(values)),
            dashboard_path,
            "Macro-F1, balanced accuracy, and QWK for the primary feature policy.",
            "Performance is useful for research comparison, not sufficient for deployment.",
        ),
        component_from_score(
            "Leakage Robustness",
            1.0 if bool(row.get("salary_hike_excluded", False)) and bool(row.get("attrition_excluded", False)) else 0.0 if not row.empty else None,
            dashboard_path,
            "Checks exclusion of salary-hike and attrition leakage-risk fields.",
            "Leakage robustness is policy evidence, not proof of causal validity.",
        ),
        component_from_score(
            "Explanation Stability",
            _score_from_row(row, ["top10_shap_jaccard", "shap_spearman"], transform=lambda values: sum(values) / len(values)),
            dashboard_path,
            "Top-k grouped SHAP stability indicators.",
            "SHAP is attribution, not causality; local explanations remain case-specific.",
        ),
        component_from_score(
            "Calibration Reliability",
            _score_from_row(row, ["ece", "multiclass_brier"], transform=lambda values: clamp01(1 - sum(values) / len(values))),
            dashboard_path,
            "ECE and Brier transformed so higher is better.",
            "Probabilities remain approximate model confidence estimates.",
        ),
        component_from_score(
            "Fairness Robustness",
            _score_from_row(row, ["empdepartment_macro_f1_gap"], transform=lambda values: clamp01(1 - values[0])),
            dashboard_path,
            "Support-filtered subgroup disparity evidence.",
            "Subgroup metrics do not prove fairness or discrimination.",
        ),
        component_from_score(
            "Counterfactual Actionability",
            _score_from_row(
                row,
                ["employee_only_validity", "employee_manager_validity", "organization_allowed_validity"],
                transform=lambda values: sum(values) / len(values),
            ),
            dashboard_path,
            "Counterfactual validity by actionability mode.",
            "Counterfactuals are model scenarios, not employee prescriptions.",
        ),
        component_from_score(
            "Proxy Risk Penalty",
            _score_from_row(row, ["department_proxy_macro_f1"], transform=lambda values: clamp01(1 - values[0])),
            dashboard_path,
            "Inverse department reconstructability from remaining features.",
            "Low score means high proxy risk; removing Department does not prove fairness.",
        ),
        llm_component(llm_row, llm_summary_path),
        chatbot_component(guardrail_df, guardrail_path),
        external_component(external_path),
    ]
    return rows


def component_from_score(component: str, score: float | None, evidence_file: Path, explanation: str, limitations: str) -> Dict[str, Any]:
    if score is None:
        return {
            "component": component,
            "score": "",
            "status": "evidence_missing",
            "severity": "high",
            "evidence_file": str(evidence_file),
            "explanation": explanation,
            "limitations": "Required evidence file or columns are missing.",
        }
    status = "pass" if score >= 0.75 else "warn" if score >= 0.40 else "fail"
    severity = "low" if status == "pass" else "medium" if status == "warn" else "high"
    return {
        "component": component,
        "score": round(float(score), 6),
        "status": status,
        "severity": severity,
        "evidence_file": str(evidence_file),
        "explanation": explanation,
        "limitations": limitations,
    }


def llm_component(row: pd.Series, evidence_file: Path) -> Dict[str, Any]:
    if row.empty:
        return component_from_score("LLM Faithfulness / Governance Compliance", None, evidence_file, "Expanded LLM-agent evaluation.", "")
    real_llm_used = str(row.get("real_llm_used", "False")).lower() in {"true", "1"}
    if not real_llm_used:
        return {
            "component": "LLM Faithfulness / Governance Compliance",
            "score": "",
            "status": "evidence_missing",
            "severity": "high",
            "evidence_file": str(evidence_file),
            "explanation": "Expanded evaluation exists only in dry-run/stub mode.",
            "limitations": "Dry-run outputs are not manuscript-grade real LLM evidence.",
        }
    score = clamp01(
        float(row.get("faithfulness_pass_rate", 0.0))
        * (1 - float(row.get("unsupported_claim_rate", 1.0)))
        * (1 - float(row.get("forbidden_claim_rate", 1.0)))
        * (1 - float(row.get("missing_warning_rate", 1.0)))
    )
    return component_from_score(
        "LLM Faithfulness / Governance Compliance",
        score,
        evidence_file,
        "Real LLM faithfulness, unsupported-claim, forbidden-claim, and missing-warning rates.",
        "Small-batch automated LLM evaluation is not a human study.",
    )


def chatbot_component(df: pd.DataFrame, evidence_file: Path) -> Dict[str, Any]:
    if df.empty:
        return component_from_score("Chatbot Guardrail Compliance", None, evidence_file, "Safe/unsafe chatbot prompt evaluation.", "")
    unsafe = df[df["expected_behavior"] == "refuse_with_safe_alternative"]
    safe = df[df["expected_behavior"] == "answer_with_governance_warnings"]
    if unsafe.empty or safe.empty:
        return component_from_score("Chatbot Guardrail Compliance", None, evidence_file, "Safe/unsafe chatbot prompt evaluation.", "")
    score = (float(unsafe["refused"].mean()) + float(safe["pass"].mean())) / 2
    return component_from_score(
        "Chatbot Guardrail Compliance",
        score,
        evidence_file,
        "Unsafe refusal and safe audit-answer behavior.",
        "Automated prompt suite coverage is not exhaustive.",
    )


def external_component(evidence_file: Path) -> Dict[str, Any]:
    if not evidence_file.exists():
        return component_from_score("External Validation Robustness", None, evidence_file, "External validation and robustness evidence.", "")
    return {
        "component": "External Validation Robustness",
        "score": "",
        "status": "pass_with_warnings",
        "severity": "medium",
        "evidence_file": str(evidence_file),
        "explanation": "External validation summary exists with HRDataset replication and related robustness boundaries.",
        "limitations": "Dataset provenance, restricted IBM target space, related-task transfer, and cross-dataset feature overlap limitations remain.",
    }


def _score_from_row(row: pd.Series, columns: list[str], transform: Any) -> float | None:
    if row.empty:
        return None
    values = []
    for column in columns:
        if column not in row.index or pd.isna(row.get(column)):
            return None
        values.append(float(row[column]))
    return clamp01(transform(values))


def final_readiness_label(dashboard: pd.DataFrame) -> str:
    if dashboard.empty or (dashboard["status"] == "evidence_missing").any():
        return "evidence_missing"
    if (dashboard["status"] == "fail").any():
        return "not_ready"
    if (dashboard["severity"] == "high").any() or (dashboard["status"] == "pass_with_warnings").any():
        return "research_only"
    return "decision_support_with_strong_warnings"


def write_component_markdown(dashboard: pd.DataFrame, path: Path, final_label: str) -> None:
    lines = [
        "# G-XAIR Component Dashboard",
        "",
        f"Final readiness label: `{final_label}`",
        "",
        "This is a component readiness dashboard, not a universal ethical score.",
        "",
        *markdown_table(dashboard.drop(columns=["final_readiness_label"], errors="ignore")),
        "",
        "## Interpretation Limits",
        "",
        "- Scores are included only where defensible from existing evidence.",
        "- Evidence-missing components are not imputed.",
        "- The system must not be described as deployment-ready.",
        "- Stub/dry-run LLM outputs are not manuscript-grade real LLM evidence and are excluded from readiness evidence.",
    ]
    write_markdown(path, lines)


def write_readiness_report(dashboard: pd.DataFrame, path: Path, final_label: str) -> None:
    high = dashboard[dashboard["severity"] == "high"]
    missing = dashboard[dashboard["status"] == "evidence_missing"]
    llm_missing = not dashboard[
        (dashboard["component"] == "LLM Faithfulness / Governance Compliance")
        & (dashboard["status"] == "evidence_missing")
    ].empty
    lines = [
        "# Final Governance Readiness Report",
        "",
        f"Final readiness label: `{final_label}`",
        "",
        "## Readiness Interpretation",
        "",
        "The framework remains research-grade decision support. It is not an autonomous HR decision system.",
        "",
        "## High-Severity or Missing Components",
        "",
    ]
    flagged = pd.concat([high, missing]).drop_duplicates()
    if flagged.empty:
        lines.append("No high-severity or missing components were detected.")
    else:
        for row in flagged.itertuples(index=False):
            lines.append(f"- {row.component}: {row.status} ({row.severity}). {row.limitations}")
    blockers = [
        "- Human, legal, organization-specific, and data-provenance validation remain required.",
        "- SHAP and counterfactuals do not support causal claims.",
        "- Removing group variables does not prove fairness.",
        "- Real expanded LLM evidence is automated technical evidence, not a substitute for human evaluation.",
        "- Stub/dry-run LLM outputs are not manuscript-grade real LLM evidence and are excluded from the final evidence package.",
    ]
    if llm_missing:
        blockers.append("- Real expanded LLM evaluation is required before manuscript-grade LLM claims if the current run is dry-run only.")
    lines.extend(["", "## Deployment Blockers", "", *blockers])
    write_markdown(path, lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate G-XAIR governance dashboards.")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        print(run_component_dashboard(args.config))
    else:
        print(run())
