from __future__ import annotations

from typing import Any, Dict

from src.agents.base_agent import AgentFinding, BaseGovernanceAgent
from src.llm.faithfulness_checker import check_faithfulness


class ExplanationComplianceAgent(BaseGovernanceAgent):
    agent_name = "ExplanationComplianceAgent"

    def audit_explanation(self, explanation: Dict[str, Any], evidence: Dict[str, Any]) -> AgentFinding:
        result = check_faithfulness(explanation, evidence).to_dict()
        status = "pass" if result["faithfulness_pass"] else "fail"
        risk = "low" if result["score"] >= 90 else "medium" if result["score"] >= 75 else "high"
        missing = result["missing_warnings"]
        forbidden = result["forbidden_claims"]
        unsupported = result["unsupported_claims"]
        return AgentFinding(
            agent_name=self.agent_name,
            status=status,
            risk_level=risk,
            summary=f"Explanation compliance score={result['score']}; forbidden={len(forbidden)}, unsupported={len(unsupported)}, missing_warnings={len(missing)}.",
            required_warnings=missing,
            details=result,
        )

    def audit(self, evidence: Dict[str, Any]) -> AgentFinding:
        return AgentFinding(self.agent_name, "needs_explanation", "medium", "Call audit_explanation with generated explanation.", [])

