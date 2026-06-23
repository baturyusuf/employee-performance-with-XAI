from __future__ import annotations

import unittest

from src.llm.interpret_real_llm_evaluation import (
    assess_agent_warning_consistency,
    assess_consistency,
    assess_high_is_good,
    assess_low_is_good,
    build_interpretation,
)


class RealLLMEvalInterpretationTests(unittest.TestCase):
    def test_metric_assessments_match_expected_thresholds(self) -> None:
        self.assertEqual(assess_high_is_good(1.0, strict=True), "clean")
        self.assertEqual(assess_high_is_good(0.9, strict=True), "acceptable_with_monitoring")
        self.assertEqual(assess_low_is_good(0.0), "clean")
        self.assertEqual(assess_low_is_good(0.2), "requires_fix")

    def test_warning_consistency_has_middle_interpretation_band(self) -> None:
        self.assertEqual(
            assess_consistency(0.679861),
            "moderate_consistency_expected_for_case_specific_warnings",
        )
        self.assertEqual(assess_consistency(0.4), "low_consistency_requires_fix")

    def test_agent_warning_consistency_flags_low_agents(self) -> None:
        result = assess_agent_warning_consistency(
            {
                "LeakageAuditAgent": 1.0,
                "CounterfactualActionabilityAgent": 0.4666,
                "ExplanationComplianceAgent": 0.5,
            }
        )
        self.assertIn("CounterfactualActionabilityAgent", result)
        self.assertIn("ExplanationComplianceAgent", result)

    def test_build_interpretation_keeps_llm_as_interpretation_layer(self) -> None:
        summary = {
            "faithfulness_pass_rate": 1.0,
            "unsupported_claim_rate": 0.0,
            "forbidden_claim_rate": 0.0,
            "missing_warning_rate": 0.0,
            "agent_success_rate": 1.0,
            "warning_consistency_rate": 0.679861,
            "unsafe_prompt_refusal_rate": 1.0,
            "warning_consistency_by_agent": {"LeakageAuditAgent": 1.0},
        }
        interpretation = build_interpretation(summary)
        self.assertIn("structured evidence", interpretation["faithfulness_pass_rate"]["claim_supported"])
        self.assertIn("does not prove deployment safety", interpretation["faithfulness_pass_rate"]["claim_not_supported"])


if __name__ == "__main__":
    unittest.main()
