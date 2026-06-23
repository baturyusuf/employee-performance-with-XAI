from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMClient(ABC):
    """Minimal interface for optional LLM backends."""

    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

