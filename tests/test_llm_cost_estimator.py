from __future__ import annotations

import unittest

from src.llm.cost_estimator import estimate_cost


class LLMCostEstimatorTests(unittest.TestCase):
    def test_cost_estimate_matches_expected_formula(self) -> None:
        estimate = estimate_cost(
            model="gpt-5.4-mini",
            n_cases=5,
            calls_per_case=8,
            avg_input_tokens=8000,
            avg_output_tokens=600,
        )
        self.assertEqual(estimate["total_calls"], 40)
        self.assertAlmostEqual(estimate["estimated_total_cost_usd"], 0.348)

    def test_unknown_model_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_cost(
                model="unknown-model",
                n_cases=1,
                calls_per_case=1,
                avg_input_tokens=100,
                avg_output_tokens=100,
            )


if __name__ == "__main__":
    unittest.main()
