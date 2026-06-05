from __future__ import annotations

import unittest

import numpy as np

from src.models.evaluate import (
    adjacent_accuracy,
    classification_metrics,
    expected_calibration_error,
    multiclass_brier,
    severe_error_rate,
)


class MetricsTests(unittest.TestCase):
    def test_ordinal_error_metrics(self) -> None:
        y_true = [2, 2, 3, 4]
        y_pred = [2, 4, 3, 3]

        self.assertEqual(severe_error_rate(y_true, y_pred), 0.25)
        self.assertEqual(adjacent_accuracy(y_true, y_pred), 0.75)

    def test_multiclass_brier_perfect_probabilities(self) -> None:
        y_true = [2, 3, 4]
        y_proba = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.assertEqual(multiclass_brier(y_true, y_proba, labels=[2, 3, 4]), 0.0)

    def test_ece_perfect_confidence(self) -> None:
        y_true = [2, 3, 4]
        y_pred = [2, 3, 4]
        y_proba = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.assertEqual(expected_calibration_error(y_true, y_pred, y_proba, n_bins=10), 0.0)

    def test_classification_metrics_contains_required_keys(self) -> None:
        y_true = [2, 2, 3, 4]
        y_pred = [2, 3, 3, 4]
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
        ])
        metrics = classification_metrics(y_true, y_pred, y_proba, labels=[2, 3, 4])

        for key in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "quadratic_weighted_kappa",
            "ordinal_mae",
            "nll_log_loss",
            "multiclass_brier",
            "ece_confidence",
        ]:
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()
