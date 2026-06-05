from __future__ import annotations

import unittest

import pandas as pd

from src.experiments.fairness_sensitivity import compute_disparity_summary


class FairnessSupportFilterTests(unittest.TestCase):
    def test_disparity_summary_excludes_small_groups(self) -> None:
        group_df = pd.DataFrame(
            [
                {
                    "feature_set": "fs",
                    "model": "xgboost",
                    "attribute": "Gender",
                    "group_value": "A",
                    "n_samples": 40,
                    "class_label": 2,
                    "accuracy": 0.8,
                    "macro_f1": 0.7,
                    "positive_prediction_rate": 0.2,
                    "true_positive_rate": 0.5,
                    "false_positive_rate": 0.1,
                    "precision": 0.6,
                    "mean_predicted_probability": 0.3,
                },
                {
                    "feature_set": "fs",
                    "model": "xgboost",
                    "attribute": "Gender",
                    "group_value": "B",
                    "n_samples": 35,
                    "class_label": 2,
                    "accuracy": 0.6,
                    "macro_f1": 0.5,
                    "positive_prediction_rate": 0.4,
                    "true_positive_rate": 0.7,
                    "false_positive_rate": 0.2,
                    "precision": 0.5,
                    "mean_predicted_probability": 0.5,
                },
                {
                    "feature_set": "fs",
                    "model": "xgboost",
                    "attribute": "Gender",
                    "group_value": "small",
                    "n_samples": 5,
                    "class_label": 2,
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "positive_prediction_rate": 1.0,
                    "true_positive_rate": 1.0,
                    "false_positive_rate": 1.0,
                    "precision": 0.0,
                    "mean_predicted_probability": 1.0,
                },
            ]
        )

        summary = compute_disparity_summary(group_df, min_support=30)
        acc_row = summary[summary["metric"] == "accuracy"].iloc[0]
        self.assertEqual(acc_row["n_groups_included"], 2)
        self.assertEqual(acc_row["min_group_value"], "B")
        self.assertEqual(acc_row["max_group_value"], "A")
        self.assertEqual(acc_row["min_group_support"], 35)
        self.assertEqual(acc_row["max_group_support"], 40)
        self.assertAlmostEqual(acc_row["max_gap"], 0.2)


if __name__ == "__main__":
    unittest.main()
