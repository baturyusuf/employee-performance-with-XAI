from __future__ import annotations

import unittest

from src.llm.usage_logger import estimate_actual_usage_cost, llm_usage_row


class LLMUsageLoggerTests(unittest.TestCase):
    def test_actual_usage_cost_uses_cached_input_discount(self) -> None:
        cost = estimate_actual_usage_cost(
            model="gpt-5.4-mini",
            input_tokens=1000,
            cached_input_tokens=500,
            output_tokens=100,
        )
        expected = (500 * 0.75 / 1_000_000) + (500 * 0.075 / 1_000_000) + (100 * 4.50 / 1_000_000)
        self.assertAlmostEqual(cost, round(expected, 8))

    def test_usage_row_normalizes_chat_completion_usage(self) -> None:
        row = llm_usage_row(
            run_id="r1",
            case_id="c1",
            operation="test",
            provider="openai",
            model="gpt-5.4-mini",
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "total_tokens": 1200,
                "prompt_tokens_details": {"cached_tokens": 100},
            },
        )
        self.assertEqual(row["input_tokens"], 1000)
        self.assertEqual(row["output_tokens"], 200)
        self.assertEqual(row["cached_input_tokens"], 100)
        self.assertGreater(row["estimated_cost_usd"], 0)


if __name__ == "__main__":
    unittest.main()
