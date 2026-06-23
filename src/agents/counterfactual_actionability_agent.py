from __future__ import annotations

from typing import Any, Dict

from src.agents.base_agent import AgentFinding, BaseGovernanceAgent


class CounterfactualActionabilityAgent(BaseGovernanceAgent):
    agent_name = "CounterfactualActionabilityAgent"

    def audit(self, evidence: Dict[str, Any]) -> AgentFinding:
        cf = evidence.get("counterfactual") or {}
        if not cf:
            return AgentFinding(self.agent_name, "needs_evidence", "medium", "Counterfactual evidence is unavailable.", ["Do not provide employee prescriptions."])
        validity = cf.get("validity")
        label = cf.get("actionability_label", "unavailable")
        if label == "not_employee_actionable" or validity == 0 or validity == 0.0:
            risk = "high"
            status = "pass_with_warnings"
            summary = "Employee-only counterfactual validity is zero or unavailable; recourse is not employee-actionable."
        else:
            risk = "medium"
            status = "pass_with_warnings"
            summary = f"Counterfactual actionability label: {label}; validity={validity}."
        return AgentFinding(
            agent_name=self.agent_name,
            status=status,
            risk_level=risk,
            summary=summary,
            required_warnings=[
                "Counterfactuals are model-level scenarios, not employee instructions.",
                "Do not say the employee should change a feature.",
            ],
            details=cf,
        )

