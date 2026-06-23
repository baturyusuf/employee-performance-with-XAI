from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.agents.openai_agents_runtime import assert_agents_sdk_runtime_available
from src.llm.openai_client import OpenAIClientConfigurationError


class OpenAIAgentsRuntimeTests(unittest.TestCase):
    def test_agents_sdk_runtime_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(OpenAIClientConfigurationError):
                assert_agents_sdk_runtime_available()


if __name__ == "__main__":
    unittest.main()
