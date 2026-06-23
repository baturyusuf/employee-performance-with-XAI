from __future__ import annotations

from typing import Any, Dict

from src.agents.base_agent import AgentFinding, BaseGovernanceAgent


class ShapStabilityAuditAgent(BaseGovernanceAgent):
    agent_name = "ShapStabilityAuditAgent"

    def audit(self, evidence: Dict[str, Any]) -> AgentFinding:
        shap = evidence.get("shap") or {}
        summary = shap.get("shap_stability_summary", {})
        jaccard = summary.get("top10_jaccard")
        spearman = summary.get("spearman")
        if not shap:
            return AgentFinding(self.agent_name, "needs_evidence", "high", "SHAP evidence is unavailable.", ["SHAP stability evidence is missing."])
        risk = "low"
        if jaccard is None or spearman is None:
            risk = "high"
        elif float(jaccard) < 0.60 or float(spearman) < 0.70:
            risk = "medium"
        return AgentFinding(
            agent_name=self.agent_name,
            status="pass_with_warnings",
            risk_level=risk,
            summary=f"Grouped SHAP stability: top-10 Jaccard={jaccard}, Spearman={spearman}. SHAP is attribution, not causality.",
            required_warnings=["SHAP is attribution, not causality.", "Discuss only stable global feature groups."],
            details=summary,
        )

