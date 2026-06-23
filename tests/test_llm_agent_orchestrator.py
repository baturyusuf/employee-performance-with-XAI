from __future__ import annotations

import unittest

from src.agents.llm_agent_orchestrator import LLMAgentGovernanceOrchestrator
from src.llm.evidence_schema import CompleteCaseEvidence
from src.llm.runtime_config import LLMRuntimeConfig


class LLMAgentOrchestratorTests(unittest.TestCase):
    def test_orchestrator_runs_with_offline_synthesis(self) -> None:
        evidence = CompleteCaseEvidence.from_reports(case_id=None)
        orchestrator = LLMAgentGovernanceOrchestrator(
            runtime_config=LLMRuntimeConfig(provider="offline")
        )
        result = orchestrator.run(evidence)
        self.assertIn("llm_agent_syntheses", result)
        self.assertGreaterEqual(len(result["llm_agent_syntheses"]), 6)
        self.assertEqual(result["supervisor_audit"]["overall_status"], "research_only")


if __name__ == "__main__":
    unittest.main()
