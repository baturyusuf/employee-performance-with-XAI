from __future__ import annotations

import unittest

from src.llm.evidence_schema import CompleteCaseEvidence
from src.llm.governed_explainer import GovernedExplainer
from src.llm.offline_stub_llm import OfflineStubLLM
from src.llm.runtime_config import LLMRuntimeConfig


class GovernedExplainerTests(unittest.TestCase):
    def test_offline_explainer_uses_complete_case_evidence(self) -> None:
        evidence = CompleteCaseEvidence.from_reports()
        explainer = GovernedExplainer(
            llm_client=OfflineStubLLM(),
            runtime_config=LLMRuntimeConfig(provider="offline", model="offline_stub_llm"),
        )
        output = explainer.generate(evidence)
        self.assertIn("short_explanation", output)
        self.assertTrue(output["requires_human_review"])
        self.assertTrue(output["faithfulness_check"]["faithfulness_pass"])
        warning_text = " ".join(w["message"] for w in output["warnings"])
        self.assertIn("SHAP is attribution", warning_text)
        self.assertIn("not for autonomous hr decisions", warning_text.lower())


if __name__ == "__main__":
    unittest.main()
