from __future__ import annotations

from typing import Any, Dict, List


GOVERNED_EXPLANATION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "short_explanation": {"type": "string"},
        "detailed_explanation": {"type": "string"},
        "warnings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "leakage",
                            "fairness",
                            "proxy",
                            "calibration",
                            "actionability",
                            "causality",
                            "deployment",
                            "evidence",
                            "validation",
                            "other",
                        ],
                    },
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "message": {"type": "string"},
                },
                "required": ["type", "severity", "message"],
                "additionalProperties": False,
            },
        },
        "unsupported_claims_detected": {
            "type": "array",
            "items": {"type": "string"},
        },
        "requires_human_review": {"type": "boolean"},
    },
    "required": [
        "case_id",
        "short_explanation",
        "detailed_explanation",
        "warnings",
        "unsupported_claims_detected",
        "requires_human_review",
    ],
    "additionalProperties": False,
}


GOVERNED_EXPLANATION_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "governed_hr_xai_explanation",
        "strict": True,
        "schema": GOVERNED_EXPLANATION_JSON_SCHEMA,
    },
}


class OutputSchemaError(ValueError):
    """Raised when an LLM output does not match the expected governed explanation contract."""


def validate_governed_explanation_payload(payload: Dict[str, Any]) -> None:
    required = GOVERNED_EXPLANATION_JSON_SCHEMA["required"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise OutputSchemaError(f"Governed explanation missing required keys: {missing}")

    if not isinstance(payload["case_id"], str):
        raise OutputSchemaError("case_id must be a string")
    if not isinstance(payload["short_explanation"], str):
        raise OutputSchemaError("short_explanation must be a string")
    if not isinstance(payload["detailed_explanation"], str):
        raise OutputSchemaError("detailed_explanation must be a string")
    if not isinstance(payload["warnings"], list):
        raise OutputSchemaError("warnings must be a list")
    if not isinstance(payload["unsupported_claims_detected"], list):
        raise OutputSchemaError("unsupported_claims_detected must be a list")
    if not isinstance(payload["requires_human_review"], bool):
        raise OutputSchemaError("requires_human_review must be a boolean")

    allowed_types = set(
        GOVERNED_EXPLANATION_JSON_SCHEMA["properties"]["warnings"]["items"]["properties"]["type"]["enum"]
    )
    allowed_severities = set(
        GOVERNED_EXPLANATION_JSON_SCHEMA["properties"]["warnings"]["items"]["properties"]["severity"]["enum"]
    )
    for warning in payload["warnings"]:
        _validate_warning(warning, allowed_types, allowed_severities)


def _validate_warning(warning: Any, allowed_types: set[str], allowed_severities: set[str]) -> None:
    if not isinstance(warning, dict):
        raise OutputSchemaError("Each warning must be an object")
    for key in ["type", "severity", "message"]:
        if key not in warning:
            raise OutputSchemaError(f"Warning missing required key: {key}")
    extra = set(warning) - {"type", "severity", "message"}
    if extra:
        raise OutputSchemaError(f"Warning contains unsupported keys: {sorted(extra)}")
    if warning["type"] not in allowed_types:
        raise OutputSchemaError(f"Unsupported warning type: {warning['type']}")
    if warning["severity"] not in allowed_severities:
        raise OutputSchemaError(f"Unsupported warning severity: {warning['severity']}")
    if not isinstance(warning["message"], str):
        raise OutputSchemaError("Warning message must be a string")


def warning_types(payload: Dict[str, Any]) -> List[str]:
    return [str(item.get("type")) for item in payload.get("warnings", []) if isinstance(item, dict)]
