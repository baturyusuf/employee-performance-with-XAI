from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.chatbot.guardrails import check_question
from src.chatbot.prompt_suite import load_prompt_suite
from src.chatbot.run_guardrail_eval import build_category_summary, run, wilson_interval
from src.utils.config_loader import PROJECT_ROOT, load_config


SUITE_PATH = PROJECT_ROOT / "configs" / "chatbot_guardrail_prompt_suite_v2.yaml"
EVAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "chatbot_guardrail_eval.yaml"


class GuardrailSuiteContractTests(unittest.TestCase):
    def test_suite_retains_minimums_and_required_categories(self) -> None:
        suite = load_prompt_suite(SUITE_PATH)
        settings = load_config(EVAL_CONFIG_PATH)["chatbot_guardrail_eval"]

        self.assertGreaterEqual(len(suite.prompts_of_type("unsafe")), 50)
        self.assertGreaterEqual(len(suite.prompts_of_type("safe")), 25)
        self.assertGreaterEqual(len(suite.prompts_of_type("unsafe")), settings["min_unsafe_prompts"])
        self.assertGreaterEqual(len(suite.prompts_of_type("safe")), settings["min_safe_prompts"])
        self.assertTrue(set(settings["required_categories"]).issubset(suite.categories))
        self.assertEqual("2.0.0", suite.suite_version)

    def test_prompt_ids_are_unique_and_expected_behavior_matches_type(self) -> None:
        suite = load_prompt_suite(SUITE_PATH)
        ids = [case.prompt_id for case in suite.prompts]
        self.assertEqual(len(ids), len(set(ids)))
        for case in suite.prompts:
            expected = (
                "refuse_with_safe_alternative"
                if case.prompt_type == "unsafe"
                else "answer_with_governance_warnings"
            )
            self.assertEqual(expected, case.expected_behavior, case.prompt_id)

    def test_every_versioned_prompt_routes_to_expected_side_of_boundary(self) -> None:
        suite = load_prompt_suite(SUITE_PATH)
        failures = []
        for case in suite.prompts:
            allowed = check_question(case.prompt).allowed
            expected_allowed = case.prompt_type == "safe"
            if allowed != expected_allowed:
                failures.append(case.prompt_id)
        self.assertEqual([], failures)

    def test_category_summary_contains_wilson_intervals(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "suite_id": "test_suite",
                    "suite_version": "1.0.0",
                    "prompt_type": "unsafe",
                    "category": "category_a",
                    "expected_behavior": "refuse_with_safe_alternative",
                    "pass": True,
                }
                for _ in range(3)
            ]
        )
        summary = build_category_summary(frame)
        row = summary.iloc[0]
        self.assertEqual(3, row["n_prompts"])
        self.assertEqual(3, row["n_passed"])
        self.assertEqual(1.0, row["pass_rate"])
        self.assertLess(row["wilson_ci_low"], 1.0)
        self.assertEqual(1.0, row["wilson_ci_high"])
        self.assertEqual("wilson_score", row["interval_method"])

    def test_wilson_interval_validates_counts(self) -> None:
        low, high = wilson_interval(50, 50)
        self.assertLess(low, 1.0)
        self.assertEqual(1.0, high)
        with self.assertRaises(ValueError):
            wilson_interval(6, 5)

    def test_runner_uses_versioned_source_and_writes_category_results(self) -> None:
        settings = load_config(EVAL_CONFIG_PATH)["chatbot_guardrail_eval"]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "outputs"
            config_path = Path(directory) / "guardrail_test_config.yaml"
            config_path.write_text(
                json.dumps(
                    {
                        "chatbot_guardrail_eval": {
                            "seed": 42,
                            "run_id_prefix": "guardrail_test",
                            "suite_source_path": str(SUITE_PATH),
                            "output_dir": str(output_dir),
                            "engine": "deterministic_report_backed_chatbot",
                            "min_unsafe_prompts": settings["min_unsafe_prompts"],
                            "min_safe_prompts": settings["min_safe_prompts"],
                            "required_categories": settings["required_categories"],
                            "wilson_confidence_level": 0.95,
                            "register_run": False,
                        }
                    }
                ),
                encoding="utf-8",
            )

            outputs = run(str(config_path))
            self.assertEqual(
                {
                    "unsafe_prompt_suite",
                    "safe_prompt_suite",
                    "evaluation",
                    "category_summary",
                    "summary",
                },
                set(outputs),
            )
            for path in outputs.values():
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0, path)

            evaluation = pd.read_csv(outputs["evaluation"])
            category_summary = pd.read_csv(outputs["category_summary"])
            self.assertEqual(114, len(evaluation))
            self.assertTrue(evaluation["pass"].all())
            self.assertEqual({"2.0.0"}, set(evaluation["suite_version"]))
            self.assertTrue(set(settings["required_categories"]).issubset(set(category_summary["category"])))
            self.assertTrue(
                {"n_prompts", "n_passed", "pass_rate", "wilson_ci_low", "wilson_ci_high"}.issubset(
                    category_summary.columns
                )
            )

            summary_text = outputs["summary"].read_text(encoding="utf-8").lower()
            self.assertIn("not an exhaustive", summary_text)
            self.assertIn("not proof of zero failure probability", summary_text)
            self.assertIn("not be described as comprehensive safety validation", summary_text)


if __name__ == "__main__":
    unittest.main()
