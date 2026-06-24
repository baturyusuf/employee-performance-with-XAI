from __future__ import annotations

import unittest

import pandas as pd

from src.llm.run_llm_agent_evaluation import select_risk_aware_cases, validate_explanation_payload


class LLMAgentEvaluationRunnerTests(unittest.TestCase):
    def test_risk_aware_sampling_includes_misclassification_and_classes(self) -> None:
        predictions = pd.DataFrame(
            [
                {"sample_index": 1, "y_true": 2, "y_pred": 2, "confidence": 0.95, "correct": True},
                {"sample_index": 2, "y_true": 3, "y_pred": 3, "confidence": 0.55, "correct": True},
                {"sample_index": 3, "y_true": 4, "y_pred": 3, "confidence": 0.92, "correct": False},
                {"sample_index": 4, "y_true": 4, "y_pred": 4, "confidence": 0.88, "correct": True},
            ]
        )
        selected = select_risk_aware_cases(
            predictions,
            sample_size=4,
            seed=42,
            representative=pd.DataFrame(),
            representative_case_col="case",
            extra_available_cases=set(),
        )
        reasons = " ".join(row["sampling_reason"] for row in selected)
        self.assertIn("misclassification", reasons)
        self.assertIn("class_2", reasons)

    def test_explanation_payload_validation_detects_missing_keys(self) -> None:
        ok, error = validate_explanation_payload({"short_explanation": "x"})
        self.assertFalse(ok)
        self.assertIn("missing", error)


if __name__ == "__main__":
    unittest.main()
