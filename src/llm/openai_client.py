from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from src.llm.llm_client import LLMClient
from src.llm.output_schema import (
    GOVERNED_EXPLANATION_RESPONSE_FORMAT,
    validate_governed_explanation_payload,
)
from src.llm.usage_logger import append_llm_usage
from src.utils.experiment_registry import utc_now_iso


class OpenAIClientConfigurationError(RuntimeError):
    """Raised when the OpenAI client cannot be configured for a real API call."""


class OpenAIClientAPIError(RuntimeError):
    """Raised when the OpenAI API returns a runtime error."""


class OpenAIChatStructuredClient(LLMClient):
    """OpenAI-backed LLM client using strict JSON Schema structured outputs."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise OpenAIClientConfigurationError(
                "The OpenAI SDK is not installed. Run: pip install -r requirements.txt"
            ) from exc

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise OpenAIClientConfigurationError(
                "OPENAI_API_KEY is not set. Set it before using HR_XAI_LLM_PROVIDER=openai."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=resolved_key)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.generate_structured_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            evidence=evidence,
            response_format=GOVERNED_EXPLANATION_RESPONSE_FORMAT,
            validator=validate_governed_explanation_payload,
        )

    def generate_structured_json(
        self,
        system_prompt: str,
        user_prompt: str,
        evidence: Dict[str, Any],
        response_format: Dict[str, Any],
        validator: Any,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"{user_prompt}\n\n"
                    "Use only this JSON evidence. Do not infer missing data.\n"
                    f"EVIDENCE_JSON:\n{json.dumps(evidence, ensure_ascii=False, sort_keys=True)}"
                ),
            },
        ]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )
        except Exception as exc:
            error_type = exc.__class__.__name__
            message = str(exc)
            if "insufficient_quota" in message or error_type == "RateLimitError":
                raise OpenAIClientAPIError(
                    "OpenAI API quota/billing is not available for this key. "
                    "Check API billing, project limits, and model access in the OpenAI dashboard."
                ) from exc
            if error_type in {"AuthenticationError", "PermissionDeniedError"}:
                raise OpenAIClientAPIError(
                    "OpenAI API authentication or permission failed. Check OPENAI_API_KEY and project/model access."
                ) from exc
            raise OpenAIClientAPIError(f"OpenAI API call failed ({error_type}): {message}") from exc
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("OpenAI response ended due to token limit; increase HR_XAI_LLM_MAX_TOKENS.")
        message = choice.message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise RuntimeError(f"OpenAI model refused the request: {refusal}")
        if not message.content:
            raise RuntimeError("OpenAI response did not contain JSON content.")

        payload = json.loads(message.content)
        validator(payload)
        backend = "openai_chat_completions"
        payload["_llm_backend"] = backend
        payload["_llm_model"] = self.model
        usage = _usage_to_dict(getattr(response, "usage", None))
        payload["_llm_usage"] = usage
        operation = _response_format_name(response_format)
        case_id = str((evidence.get("prediction") or {}).get("case_id") or payload.get("case_id") or "unknown")
        append_llm_usage(
            run_id=f"{operation}_{case_id}_{utc_now_iso()}",
            case_id=case_id,
            operation=operation,
            provider=backend,
            model=self.model,
            usage=usage,
            notes="OpenAI structured output call.",
        )
        return payload


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return dict(usage)
    return {
        key: getattr(usage, key)
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
        if hasattr(usage, key)
    }


def _response_format_name(response_format: Dict[str, Any]) -> str:
    try:
        return str(response_format.get("json_schema", {}).get("name") or response_format.get("type") or "openai_call")
    except Exception:
        return "openai_call"
