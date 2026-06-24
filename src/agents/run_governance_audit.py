from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.agents.calibration_agent import CalibrationAuditAgent
from src.agents.counterfactual_actionability_agent import CounterfactualActionabilityAgent
from src.agents.explanation_agent import ExplanationComplianceAgent
from src.agents.fairness_proxy_agent import FairnessProxyAuditAgent
from src.agents.leakage_agent import LeakageAuditAgent
from src.agents.shap_stability_agent import ShapStabilityAuditAgent
from src.agents.supervisor_agent import SupervisorGovernanceAgent
from src.core.io_utils import read_jsonl, write_jsonl
from src.core.reporting import markdown_table, write_markdown
from src.core.run_registry import RunRegistryEntry, append_run_entry
from src.llm.evidence_schema import CompleteCaseEvidence
from src.llm.governed_explainer import GovernedExplainer
from src.utils.config_loader import load_config
from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


OUTPUT_DIR = SETTINGS.reports_dir / "agent_audits"


def deterministic_agent_findings(evidence_dict: Dict[str, Any], explanation: Dict[str, Any]) -> tuple[List[Any], Dict[str, Any]]:
    agents = [
        LeakageAuditAgent(),
        FairnessProxyAuditAgent(),
        CalibrationAuditAgent(),
        ShapStabilityAuditAgent(),
        CounterfactualActionabilityAgent(),
    ]
    findings = [agent.audit(evidence_dict) for agent in agents]
    findings.append(ExplanationComplianceAgent().audit_explanation(explanation, evidence_dict))
    supervisor = SupervisorGovernanceAgent().aggregate(findings)
    return findings, supervisor


def run_batch_from_records(
    records: List[Dict[str, Any]],
    *,
    run_id: str,
    output_dir: Path = OUTPUT_DIR,
    command: str = "python -m src.agents.run_governance_audit --config configs/llm_agent_eval.yaml",
    config_path: str = "",
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for record in records:
        evidence = record.get("structured_evidence") or {}
        explanation = record.get("raw_response") if isinstance(record.get("raw_response"), dict) else record
        findings, supervisor = deterministic_agent_findings(evidence, explanation)
        dataset_name = str(record.get("dataset_name") or evidence.get("prediction", {}).get("dataset_name", "unknown"))
        case_id = str(record.get("case_id") or evidence.get("prediction", {}).get("case_id", "unknown"))
        for finding in findings:
            payload = {
                "run_id": run_id,
                "dataset_name": dataset_name,
                "case_id": case_id,
                **finding.to_dict(),
                "supervisor_overall_status": supervisor["overall_status"],
            }
            jsonl_rows.append(payload)
            csv_rows.append(
                {
                    "run_id": run_id,
                    "dataset_name": dataset_name,
                    "case_id": case_id,
                    "agent_name": finding.agent_name,
                    "status": finding.status,
                    "risk_level": finding.risk_level,
                    "summary": finding.summary,
                    "required_warnings": "; ".join(finding.required_warnings),
                    "supervisor_overall_status": supervisor["overall_status"],
                }
            )
        jsonl_rows.append(
            {
                "run_id": run_id,
                "dataset_name": dataset_name,
                "case_id": case_id,
                "agent_name": "SupervisorGovernanceAgent",
                "status": supervisor["overall_status"],
                "risk_level": _supervisor_risk(supervisor["overall_status"]),
                "summary": supervisor["readiness_summary"],
                "required_warnings": supervisor["critical_warnings"],
                "details": supervisor,
                "supervisor_overall_status": supervisor["overall_status"],
            }
        )
        csv_rows.append(
            {
                "run_id": run_id,
                "dataset_name": dataset_name,
                "case_id": case_id,
                "agent_name": "SupervisorGovernanceAgent",
                "status": supervisor["overall_status"],
                "risk_level": _supervisor_risk(supervisor["overall_status"]),
                "summary": supervisor["readiness_summary"],
                "required_warnings": "; ".join(supervisor["critical_warnings"]),
                "supervisor_overall_status": supervisor["overall_status"],
            }
        )

    jsonl_path = output_dir / "agent_audit_results.jsonl"
    csv_path = output_dir / "agent_audit_results.csv"
    md_path = output_dir / "multi_agent_governance_audit.md"
    write_jsonl(jsonl_path, jsonl_rows)
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, index=False)
    write_batch_markdown(df, md_path)
    append_run_entry(
        RunRegistryEntry(
            run_id=run_id,
            command=command,
            config_path=config_path,
            dataset="multiple",
            model="deterministic_governance_agents",
            feature_policy="multiple",
            seed="deterministic",
            output_files=[str(jsonl_path), str(csv_path), str(md_path)],
        )
    )
    return {"jsonl": jsonl_path, "csv": csv_path, "markdown": md_path}


def write_batch_markdown(df: pd.DataFrame, path: Path) -> None:
    grouped = (
        df.groupby(["agent_name", "status", "risk_level"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["agent_name", "status"])
    )
    readiness = (
        df[df["agent_name"] == "SupervisorGovernanceAgent"]["status"]
        .value_counts()
        .rename_axis("readiness_status")
        .reset_index(name="count")
    )
    lines = [
        "# Multi-Agent Governance Audit",
        "",
        "This batch audit is deterministic and evidence-based. Agents audit structured ML/XAI/LLM evidence; they do not make HR decisions.",
        "",
        "## Readiness Distribution",
        "",
        *markdown_table(readiness),
        "",
        "## Agent Status Summary",
        "",
        *markdown_table(grouped),
        "",
        "## Limitations",
        "",
        "- Agent outputs are governance diagnostics, not legal determinations.",
        "- High-risk or warning statuses must be reported as limitations, not hidden.",
        "- The system remains research-grade decision support only.",
    ]
    write_markdown(path, lines)


def run_batch_from_config(config_path: str) -> Dict[str, Path]:
    config = load_config(config_path)
    settings = config.get("llm_agent_eval", config)
    output_dir = Path(settings.get("agent_output_dir", "reports/agent_audits"))
    run_id = str(settings.get("last_run_id") or f"agent_audit_{utc_now_iso()}")
    explanation_path = Path(settings.get("governed_explanations_path", "reports/llm_explanations/governed_explanations.jsonl"))
    records = read_jsonl(explanation_path)
    if not records:
        raise FileNotFoundError(f"No governed explanation records found at {explanation_path}")
    return run_batch_from_records(records, run_id=run_id, output_dir=output_dir, config_path=config_path)


def _supervisor_risk(status: str) -> str:
    if status in {"not_ready", "evidence_missing"}:
        return "high"
    if status == "research_only":
        return "high"
    return "medium"


def run(case_id: str | None = None) -> Dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence = CompleteCaseEvidence.from_reports(case_id=case_id)
    evidence_dict = evidence.to_dict()
    explanation = GovernedExplainer().generate(evidence)
    findings, supervisor = deterministic_agent_findings(evidence_dict, explanation)

    audit_json = {
        "case_id": evidence.prediction.case_id,
        "explanation": explanation,
        "supervisor_audit": supervisor,
    }
    json_path = OUTPUT_DIR / "multi_agent_governance_audit.json"
    json_path.write_text(json.dumps(audit_json, indent=2, sort_keys=True), encoding="utf-8")

    md_path = OUTPUT_DIR / "multi_agent_governance_audit.md"
    lines: List[str] = [
        "# Multi-Agent Governance Audit",
        "",
        "## Architecture",
        "XGBoost predicts -> XAI explains -> audit modules evaluate reliability -> deterministic LLM-style layer interprets structured evidence -> agents audit the explanation -> chatbot exposes governed answers.",
        "",
        "## Agent Roles",
        "- LeakageAuditAgent: checks leakage-risk feature exclusions and LSI context.",
        "- FairnessProxyAuditAgent: checks subgroup gaps, low support, and proxy warnings.",
        "- CalibrationAuditAgent: checks probability-quality risk.",
        "- ShapStabilityAuditAgent: checks explanation stability and non-causal warning.",
        "- CounterfactualActionabilityAgent: separates technical validity from practical actionability.",
        "- ExplanationComplianceAgent: checks forbidden claims, unsupported claims, and missing warnings.",
        "- SupervisorGovernanceAgent: aggregates readiness status and prohibited uses.",
        "",
        "## Final Readiness Status",
        f"`{supervisor['overall_status']}`",
        "",
        "## Major Warnings",
    ]
    for warning in supervisor["critical_warnings"]:
        lines.append(f"- {warning}")
    lines.extend(["", "## Agent Findings"])
    for name, finding in supervisor["agent_findings"].items():
        lines.append(f"### {name}")
        lines.append(f"- Status: {finding['status']}")
        lines.append(f"- Risk level: {finding['risk_level']}")
        lines.append(f"- Summary: {finding['summary']}")
        lines.append("")
    lines.extend(
        [
            "## Limitations",
            "- The agents audit structured evidence; they do not retrain or validate the predictive model externally.",
            "- Findings are research governance diagnostics, not legal determinations.",
            "- The system must not be used for autonomous HR decisions.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_registry_row(
        {
            "run_id": f"multi_agent_governance_audit_{utc_now_iso()}",
            "date_time": utc_now_iso(),
            "git_commit_if_available": get_git_commit(),
            "script": "python -m src.agents.run_governance_audit",
            "config": "structured evidence loaded from final candidate reports",
            "feature_set": "no_salary_hike_no_attrition_no_department",
            "model": "governance_agents",
            "seed": "deterministic",
            "cv_strategy": "not_applicable",
            "primary_metrics": "agent risk levels; supervisor readiness status",
            "output_dir": "reports/agent_audits",
            "notes": "Generated multi-agent governance audit from structured final-candidate evidence.",
            "decision_status": "candidate",
        }
    )
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-agent governance audit.")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        print(run_batch_from_config(args.config))
    else:
        print(run(case_id=args.case_id))
