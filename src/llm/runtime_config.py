from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


ProviderName = Literal["auto", "offline", "openai"]


@dataclass(frozen=True)
class LLMRuntimeConfig:
    provider: ProviderName = "auto"
    model: str = "gpt-5.4-mini"
    temperature: float = 0.0
    max_tokens: int = 1200
    require_real_llm: bool = False

    @classmethod
    def from_env(cls) -> "LLMRuntimeConfig":
        _load_dotenv_if_available()
        provider = os.getenv("HR_XAI_LLM_PROVIDER", "auto").strip().lower()
        if provider not in {"auto", "offline", "openai"}:
            raise ValueError(
                "HR_XAI_LLM_PROVIDER must be one of: auto, offline, openai"
            )
        model = os.getenv("OPENAI_MODEL") or os.getenv("HR_XAI_OPENAI_MODEL") or "gpt-5.4-mini"
        temperature = _float_env("HR_XAI_LLM_TEMPERATURE", default=0.0)
        max_tokens = _int_env("HR_XAI_LLM_MAX_TOKENS", default=1200)
        require_real_llm = os.getenv("HR_XAI_REQUIRE_REAL_LLM", "0").strip() in {"1", "true", "yes"}
        return cls(
            provider=provider,  # type: ignore[arg-type]
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            require_real_llm=require_real_llm,
        )


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float") from exc


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
