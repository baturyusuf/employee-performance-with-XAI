from __future__ import annotations

import importlib.util
import json
import os
from typing import Any, Dict

from src.llm.runtime_config import LLMRuntimeConfig


def check_setup() -> Dict[str, Any]:
    config = LLMRuntimeConfig.from_env()
    openai_installed = importlib.util.find_spec("openai") is not None
    agents_installed = importlib.util.find_spec("agents") is not None
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    provider_ready = (
        config.provider == "offline"
        or (config.provider in {"auto", "openai"} and openai_installed and api_key_present)
    )
    return {
        "provider": config.provider,
        "resolved_model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "require_real_llm": config.require_real_llm,
        "openai_sdk_installed": openai_installed,
        "openai_agents_sdk_installed": agents_installed,
        "openai_api_key_present": api_key_present,
        "provider_ready": provider_ready,
        "required_user_setup": _required_setup(openai_installed, api_key_present),
    }


def _required_setup(openai_installed: bool, api_key_present: bool) -> list[str]:
    items = []
    if not openai_installed:
        items.append("Run: .\\myenv\\Scripts\\pip.exe install -r requirements.txt")
    if not api_key_present:
        items.append("Set OPENAI_API_KEY in the shell or .env before real LLM runs.")
    return items


if __name__ == "__main__":
    print(json.dumps(check_setup(), indent=2, sort_keys=True))
