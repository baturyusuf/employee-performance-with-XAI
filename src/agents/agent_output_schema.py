from __future__ import annotations

from typing import Any, Dict


AGENT_AUDIT_SYNTHESIS_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "agent_name": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["pass", "pass_with_warnings", "fail", "needs_evidence"],
        },
        "risk_level": {"type": "string", "enum": ["low", "medium", "high", "unknown"]},
        "governed_summary": {"type": "string"},
        "evidence_used": {
            "type": "array",
            "items": {"type": "string"},
        },
        "required_warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
        "unsupported_claims_detected": {
            "type": "array",
            "items": {"type": "string"},
        },
        "requires_human_review": {"type": "boolean"},
        "recommended_research_actions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "agent_name",
        "status",
        "risk_level",
        "governed_summary",
        "evidence_used",
        "required_warnings",
        "unsupported_claims_detected",
        "requires_human_review",
        "recommended_research_actions",
    ],
    "additionalProperties": False,
}


AGENT_AUDIT_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "hr_xai_agent_audit_synthesis",
        "strict": True,
        "schema": AGENT_AUDIT_SYNTHESIS_JSON_SCHEMA,
    },
}


class AgentOutputSchemaError(ValueError):
    """Raised when an LLM agent output violates the expected audit synthesis contract."""


def validate_agent_audit_synthesis(payload: Dict[str, Any]) -> None:
    required = AGENT_AUDIT_SYNTHESIS_JSON_SCHEMA["required"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise AgentOutputSchemaError(f"Agent audit synthesis missing required keys: {missing}")

    string_fields = ["agent_name", "status", "risk_level", "governed_summary"]
    for field in string_fields:
        if not isinstance(payload[field], str):
            raise AgentOutputSchemaError(f"{field} must be a string")

    list_fields = [
        "evidence_used",
        "required_warnings",
        "unsupported_claims_detected",
        "recommended_research_actions",
    ]
    for field in list_fields:
        if not isinstance(payload[field], list):
            raise AgentOutputSchemaError(f"{field} must be a list")
        if not all(isinstance(item, str) for item in payload[field]):
            raise AgentOutputSchemaError(f"{field} must contain only strings")

    if not isinstance(payload["requires_human_review"], bool):
        raise AgentOutputSchemaError("requires_human_review must be a boolean")
