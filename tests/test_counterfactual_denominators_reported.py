from __future__ import annotations

import unittest

import pandas as pd

from src.experiments.manuscript_counterfactual_search import (
    summarize_search_success,
    wilson_interval,
)


class CounterfactualDenominatorsReportedTests(unittest.TestCase):
    def test_summary_reports_total_eligible_valid_and_wilson_interval(self) -> None:
        rows = []
        identity = {
            "run_id": "run",
            "config_hash": "a" * 64,
            "scientific_input_hash": "b" * 64,
            "source_tree_hash": "c" * 64,
            "dataset_sha256": "d" * 64,
            "fold_contract_sha256": "e" * 64,
            "feature_policy_sha256": "f" * 64,
            "source_oof_probability_sha256": "1" * 64,
            "model_set_sha256": "2" * 64,
            "policy": "primary",
            "dataset_key": "inx_primary",
            "task_type": "ordinal_multiclass_performance",
            "evidence_role": "supplementary_heuristic_search_only",
        }
        for scope in ("employee_control_tagged", "employee_manager_control_tagged"):
            for sample_index in range(6):
                eligible = sample_index < 5
                successful = eligible and sample_index < (
                    2 if scope == "employee_control_tagged" else 4
                )
                rows.append(
                    {
                        **identity,
                        "sample_index": sample_index,
                        "candidate_feature_scope": scope,
                        "budget_id": "primary",
                        "budget_role": "primary",
                        "max_prototypes": 100,
                        "max_features_changed": 3,
                        "eligible_for_upward_shift": eligible,
                        "search_success": successful,
                        "probability_gain": 0.1 if successful else None,
                        "normalized_search_cost": 1.0 if successful else None,
                        "n_changed_features": 1 if successful else 0,
                        "candidates_within_budget": 10 if eligible else 0,
                        "search_failure_reason": (
                            "" if successful or not eligible else "not_found"
                        ),
                    }
                )
        summary, uncertainty, failures = summarize_search_success(
            pd.DataFrame(rows),
            confidence=0.95,
            n_resamples=50,
            seed=42,
        )
        required = {
            "n_total_oof_cases",
            "n_eligible_oof_cases",
            "n_search_successes",
            "heuristic_search_success_rate",
            "search_success_ci_low",
            "search_success_ci_high",
        }
        self.assertTrue(required.issubset(summary.columns))
        employee = summary[
            summary["candidate_feature_scope"] == "employee_control_tagged"
        ].iloc[0]
        self.assertEqual(employee["n_total_oof_cases"], 6)
        self.assertEqual(employee["n_eligible_oof_cases"], 5)
        self.assertEqual(employee["n_search_successes"], 2)
        self.assertTrue((uncertainty["method"].str.contains("ci")).all())
        self.assertEqual(
            int(
                failures[
                    failures["candidate_feature_scope"]
                    == "employee_control_tagged"
                ]["n_failures"].sum()
            ),
            3,
        )

    def test_perfect_rate_does_not_have_zero_width_interval(self) -> None:
        low, high = wilson_interval(4, 4)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)


if __name__ == "__main__":
    unittest.main()
