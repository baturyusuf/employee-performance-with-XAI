from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.llm.check_llm_setup import check_setup
from src.llm.client_factory import build_llm_client
from src.llm.offline_stub_llm import OfflineStubLLM
from src.llm.output_schema import validate_governed_explanation_payload
from src.llm.runtime_config import LLMRuntimeConfig


class LLMRuntimeConfigTests(unittest.TestCase):
    def test_auto_without_api_key_uses_offline_fallback(self) -> None:
        with patch.dict(os.environ, {"HR_XAI_LLM_PROVIDER": "auto"}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            client = build_llm_client(LLMRuntimeConfig(provider="auto", require_real_llm=False))
        self.assertIsInstance(client, OfflineStubLLM)

    def test_require_real_llm_blocks_offline_provider(self) -> None:
        config = LLMRuntimeConfig(provider="offline", require_real_llm=True)
        with self.assertRaises(RuntimeError):
            build_llm_client(config)

    def test_setup_reports_missing_api_key(self) -> None:
        with patch.dict(os.environ, {"HR_XAI_LLM_PROVIDER": "openai"}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            result = check_setup()
        self.assertFalse(result["openai_api_key_present"])
        self.assertIn("required_user_setup", result)

    def test_governed_explanation_schema_validation(self) -> None:
        payload = {
            "case_id": "case-1",
            "short_explanation": "short",
            "detailed_explanation": "detailed",
            "warnings": [
                {"type": "deployment", "severity": "high", "message": "Human review required."}
            ],
            "unsupported_claims_detected": [],
            "requires_human_review": True,
        }
        validate_governed_explanation_payload(payload)


if __name__ == "__main__":
    unittest.main()
