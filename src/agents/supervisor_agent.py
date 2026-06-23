from __future__ import annotations

from typing import Any, Dict, List

from src.agents.base_agent import AgentFinding


class SupervisorGovernanceAgent:
    agent_name = "SupervisorGovernanceAgent"

    def aggregate(self, findings: List[AgentFinding]) -> Dict[str, Any]:
        finding_dict = {finding.agent_name: finding.to_dict() for finding in findings}
        risk_levels = [finding.risk_level for finding in findings]
        statuses = [finding.status for finding in findings]
        critical_warnings = []
        for finding in findings:
            if finding.risk_level == "high" or finding.status in {"fail", "needs_evidence"}:
                critical_warnings.extend(finding.required_warnings)
        if any(status == "needs_evidence" for status in statuses):
            overall = "evidence_missing"
        elif any(status == "fail" for status in statuses):
            overall = "not_ready"
        elif "high" in risk_levels:
            overall = "research_only"
        else:
            overall = "decision_support_with_strong_warnings"
        return {
            "overall_status": overall,
            "readiness_summary": (
                "The system is suitable for research governance demonstration only. "
                "It is not an autonomous HR decision system."
            ),
            "agent_findings": finding_dict,
            "critical_warnings": sorted(set(critical_warnings)),
            "required_human_review": True,
            "prohibited_uses": [
                "hiring decisions",
                "firing decisions",
                "promotion decisions",
                "compensation decisions",
                "disciplinary decisions",
                "autonomous employee evaluation",
            ],
            "recommended_research_actions": [
                "Use final-candidate evidence in manuscript tables.",
                "Report proxy risk and actionability limitations explicitly.",
                "Perform external validation before operational claims.",
            ],
        }

