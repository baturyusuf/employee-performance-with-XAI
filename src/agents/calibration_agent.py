from __future__ import annotations

from typing import Any, Dict

from src.agents.base_agent import AgentFinding, BaseGovernanceAgent


class CalibrationAuditAgent(BaseGovernanceAgent):
    agent_name = "CalibrationAuditAgent"

    def audit(self, evidence: Dict[str, Any]) -> AgentFinding:
        calibration = evidence.get("calibration") or {}
        ece = calibration.get("expected_calibration_error")
        brier = calibration.get("brier_score")
        if not calibration:
            return AgentFinding(self.agent_name, "needs_evidence", "high", "Calibration evidence is unavailable.", ["Do not present probabilities as reliable confidence."])
        if ece is None:
            risk = "high"
        elif float(ece) > 0.10:
            risk = "high"
        elif float(ece) > 0.05:
            risk = "medium"
        else:
            risk = "low"
        return AgentFinding(
            agent_name=self.agent_name,
            status="pass_with_warnings",
            risk_level=risk,
            summary=f"Probability quality is approximate: ECE={ece}, Brier={brier}. Use probability bands and calibration warnings.",
            required_warnings=["Probability estimates are approximate and may be imperfectly calibrated."],
            details=calibration,
        )

