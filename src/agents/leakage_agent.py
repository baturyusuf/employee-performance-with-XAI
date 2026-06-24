from __future__ import annotations

from typing import Any, Dict

from src.agents.base_agent import AgentFinding, BaseGovernanceAgent


class LeakageAuditAgent(BaseGovernanceAgent):
    agent_name = "LeakageAuditAgent"

    def audit(self, evidence: Dict[str, Any]) -> AgentFinding:
        leakage = evidence.get("leakage") or {}
        prediction = evidence.get("prediction") or {}
        excluded = set(leakage.get("excluded_leakage_features", []))
        is_external = str(prediction.get("leakage_safe_status", "")).startswith("external_")
        required = set() if is_external else {"EmpLastSalaryHikePercent", "Attrition"}
        missing = sorted(required - excluded)
        lsi = leakage.get("leakage_sensitivity_index")
        warnings = [leakage.get("leakage_warning", "Full-feature models must be diagnostic upper-bound only.")]
        if missing:
            status = "fail"
            risk = "high"
            summary = f"Final feature policy is missing leakage exclusions: {missing}."
        elif is_external and not excluded:
            status = "needs_evidence"
            risk = "high"
            summary = "External leakage evidence does not list mapped leakage exclusions."
        elif leakage:
            status = "pass"
            risk = "medium" if lsi is None else "high" if float(lsi) > 0.25 else "medium"
            if is_external:
                summary = "Mapped external leakage-risk fields are excluded for the selected research policy."
            else:
                summary = "Leakage-risk features are excluded from the final candidate; full-feature results remain diagnostic only."
        else:
            status = "needs_evidence"
            risk = "high"
            summary = "Leakage evidence is unavailable."
        return AgentFinding(
            agent_name=self.agent_name,
            status=status,
            risk_level=risk,
            summary=summary,
            required_warnings=warnings,
            details={"missing_required_exclusions": missing, "leakage_sensitivity_index": lsi},
        )
