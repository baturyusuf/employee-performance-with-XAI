from __future__ import annotations

import unittest

import numpy as np

from src.experiments.final_evidence_common import (
    bootstrap_mean_ci,
    calibrate_probabilities,
    dashboard_required_columns,
)
from src.experiments.final_shap_stability import group_transformed_importance, jaccard_at_k


class FinalEvidencePackageTests(unittest.TestCase):
    def test_bootstrap_mean_ci_bounds_mean(self) -> None:
        low, high = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], n_boot=200, seed=7)
        self.assertLessEqual(low, 2.5)
        self.assertGreaterEqual(high, 2.5)

    def test_calibrated_probabilities_are_normalized(self) -> None:
        calib_proba = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.2, 0.2, 0.6],
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.3, 0.6],
            ]
        )
        calib_y = [2, 3, 4, 2, 3, 4]
        test_proba = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
        out = calibrate_probabilities(calib_proba, calib_y, test_proba, method="sigmoid")
        np.testing.assert_allclose(out.sum(axis=1), np.ones(out.shape[0]))
        self.assertEqual(out.shape, test_proba.shape)

    def test_group_transformed_importance_sums_one_hot_family(self) -> None:
        importance = np.array([1.0, 0.2, 0.3, 0.5])
        grouped = group_transformed_importance(
            importance,
            ["num_feature", "cat_feature"],
            {"num_feature": [0], "cat_feature": [1, 2, 3]},
        )
        self.assertAlmostEqual(grouped["num_feature"], 1.0)
        self.assertAlmostEqual(grouped["cat_feature"], 1.0)

    def test_jaccard_at_k(self) -> None:
        self.assertAlmostEqual(jaccard_at_k(["a", "b", "c"], ["b", "a", "d"], 2), 1.0)
        self.assertAlmostEqual(jaccard_at_k(["a", "b", "c"], ["c", "d", "e"], 2), 0.0)

    def test_dashboard_schema_contains_required_decision_columns(self) -> None:
        cols = dashboard_required_columns()
        for required in [
            "feature_set",
            "macro_f1",
            "qwk",
            "ece",
            "department_proxy_macro_f1",
            "top10_shap_jaccard",
            "employee_only_validity",
            "recommendation_category",
        ]:
            self.assertIn(required, cols)


if __name__ == "__main__":
    unittest.main()

