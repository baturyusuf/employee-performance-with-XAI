from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.agents.calibration_agent import CalibrationAuditAgent
from src.agents.counterfactual_actionability_agent import CounterfactualActionabilityAgent
from src.agents.explanation_agent import ExplanationComplianceAgent
from src.agents.fairness_proxy_agent import FairnessProxyAuditAgent
from src.agents.leakage_agent import LeakageAuditAgent
from src.agents.shap_stability_agent import ShapStabilityAuditAgent
from src.agents.supervisor_agent import SupervisorGovernanceAgent
from src.llm.evidence_schema import CompleteCaseEvidence
from src.llm.governed_explainer import GovernedExplainer
from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


OUTPUT_DIR = SETTINGS.reports_dir / "agent_audits"


def run(case_id: str | None = None) -> Dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence = CompleteCaseEvidence.from_reports(case_id=case_id)
    evidence_dict = evidence.to_dict()
    explanation = GovernedExplainer().generate(evidence)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(case_id=args.case_id))

