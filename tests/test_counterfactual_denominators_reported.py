from __future__ import annotations

import unittest

import pandas as pd

from src.experiments.manuscript_counterfactual_actionability import summarize_actionability, wilson_interval


class CounterfactualDenominatorsReportedTests(unittest.TestCase):
    def test_summary_reports_total_eligible_valid_and_wilson_interval(self) -> None:
        rows = []
        for mode in ("employee_only", "organization_allowed"):
            for sample_index in range(6):
                eligible = sample_index < 5
                valid = eligible and sample_index < (2 if mode == "employee_only" else 4)
                rows.append(
                    {
                        "run_id": "run",
                        "config_hash": "hash",
                        "policy": "primary",
                        "sample_index": sample_index,
                        "intervention_mode": mode,
                        "eligible_for_upward_shift": eligible,
                        "valid": valid,
                        "probability_gain": 0.1 if valid else None,
                        "cost": 1.0 if valid else None,
                        "num_changed_features": 1 if valid else 0,
                        "failure_reason": "" if valid or not eligible else "not_found",
                    }
                )
        summary, uncertainty, failures = summarize_actionability(
            pd.DataFrame(rows),
            confidence=0.95,
            n_resamples=50,
            seed=42,
        )
        required = {
            "n_total_oof_cases",
            "n_eligible_oof_cases",
            "n_valid_counterfactuals",
            "validity_rate",
            "validity_ci_low",
            "validity_ci_high",
        }
        self.assertTrue(required.issubset(summary.columns))
        employee = summary[summary["intervention_mode"] == "employee_only"].iloc[0]
        self.assertEqual(employee["n_total_oof_cases"], 6)
        self.assertEqual(employee["n_eligible_oof_cases"], 5)
        self.assertEqual(employee["n_valid_counterfactuals"], 2)
        self.assertTrue((uncertainty["method"].str.contains("ci")).all())
        self.assertEqual(int(failures[failures["intervention_mode"] == "employee_only"]["n_failures"].sum()), 3)

    def test_perfect_rate_does_not_have_zero_width_interval(self) -> None:
        low, high = wilson_interval(4, 4)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)


if __name__ == "__main__":
    unittest.main()
