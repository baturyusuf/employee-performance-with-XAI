from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentFinding:
    agent_name: str
    status: str
    risk_level: str
    summary: str
    required_warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "status": self.status,
            "risk_level": self.risk_level,
            "summary": self.summary,
            "required_warnings": self.required_warnings,
            "details": self.details,
        }


class BaseGovernanceAgent:
    agent_name = "BaseGovernanceAgent"

    def audit(self, evidence: Dict[str, Any]) -> AgentFinding:
        raise NotImplementedError

