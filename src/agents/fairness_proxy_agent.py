from __future__ import annotations

from typing import Any, Dict, List

from src.agents.base_agent import AgentFinding, BaseGovernanceAgent


class FairnessProxyAuditAgent(BaseGovernanceAgent):
    agent_name = "FairnessProxyAuditAgent"

    def audit(self, evidence: Dict[str, Any]) -> AgentFinding:
        fairness = evidence.get("fairness") or {}
        gaps = fairness.get("disparity_gaps", {})
        dept_gap = gaps.get("EmpDepartment_macro_f1_gap")
        low_support = fairness.get("low_support_warnings", [])
        proxy_warnings: List[str] = fairness.get("proxy_risk_warnings", [])
        warnings = [
            "Subgroup gaps require further investigation; they do not prove discrimination.",
            "Department removal reduces direct use but does not eliminate proxy risk.",
        ] + proxy_warnings
        if not fairness:
            return AgentFinding(self.agent_name, "needs_evidence", "high", "Fairness evidence is unavailable.", warnings)
        risk = "high" if dept_gap is not None and float(dept_gap) >= 0.20 else "medium"
        summary = (
            f"EmpDepartment macro-F1 gap is {dept_gap}; proxy warnings are present. "
            "The model must not be described as fair or unbiased."
        )
        return AgentFinding(
            agent_name=self.agent_name,
            status="pass_with_warnings",
            risk_level=risk,
            summary=summary,
            required_warnings=warnings,
            details={"department_macro_f1_gap": dept_gap, "low_support_warnings": low_support},
        )

