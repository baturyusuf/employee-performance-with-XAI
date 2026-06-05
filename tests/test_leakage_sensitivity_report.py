from __future__ import annotations

import unittest

import pandas as pd

from src.experiments.leakage_sensitivity import compute_lsi_table


class LeakageSensitivityReportTests(unittest.TestCase):
    def test_compute_lsi_table_from_summary(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "feature_set": "all_features",
                    "model": "xgboost",
                    "macro_f1_mean": 0.90,
                    "quadratic_weighted_kappa_mean": 0.80,
                },
                {
                    "feature_set": "no_salary_hike_no_attrition",
                    "model": "xgboost",
                    "macro_f1_mean": 0.60,
                    "quadratic_weighted_kappa_mean": 0.64,
                },
            ]
        )

        lsi = compute_lsi_table(
            summary,
            comparison_feature_sets=["no_salary_hike_no_attrition"],
            metrics=["macro_f1", "quadratic_weighted_kappa"],
        )

        self.assertEqual(len(lsi), 2)
        macro_row = lsi[lsi["metric"] == "macro_f1"].iloc[0]
        self.assertAlmostEqual(macro_row["absolute_drop"], 0.30)
        self.assertAlmostEqual(macro_row["lsi"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
