from __future__ import annotations

import os
from typing import Optional

from src.llm.llm_client import LLMClient
from src.llm.offline_stub_llm import OfflineStubLLM
from src.llm.openai_client import OpenAIChatStructuredClient, OpenAIClientConfigurationError
from src.llm.runtime_config import LLMRuntimeConfig


def build_llm_client(config: Optional[LLMRuntimeConfig] = None) -> LLMClient:
    resolved = config or LLMRuntimeConfig.from_env()

    if resolved.provider == "offline":
        if resolved.require_real_llm:
            raise OpenAIClientConfigurationError(
                "HR_XAI_REQUIRE_REAL_LLM is enabled, but provider is offline."
            )
        return OfflineStubLLM()

    if resolved.provider == "openai":
        return OpenAIChatStructuredClient(
            model=resolved.model,
            temperature=resolved.temperature,
            max_tokens=resolved.max_tokens,
        )

    if resolved.provider == "auto":
        if os.getenv("OPENAI_API_KEY"):
            try:
                return OpenAIChatStructuredClient(
                    model=resolved.model,
                    temperature=resolved.temperature,
                    max_tokens=resolved.max_tokens,
                )
            except OpenAIClientConfigurationError:
                if resolved.require_real_llm:
                    raise
        if resolved.require_real_llm:
            raise OpenAIClientConfigurationError(
                "Real LLM is required, but OpenAI SDK/API key is not configured."
            )
        return OfflineStubLLM()

    raise ValueError(f"Unsupported LLM provider: {resolved.provider}")


def active_provider_label(config: Optional[LLMRuntimeConfig] = None) -> str:
    resolved = config or LLMRuntimeConfig.from_env()
    if resolved.provider == "auto":
        return "openai" if os.getenv("OPENAI_API_KEY") else "offline"
    return resolved.provider
