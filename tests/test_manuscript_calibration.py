from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.experiments.manuscript_calibration import (
    calibration_bin_rows,
    select_calibration_method,
    summarize_fold_metrics,
)


class ManuscriptCalibrationTests(unittest.TestCase):
    def test_calibration_bins_cover_each_case_once_per_class(self) -> None:
        labels = [2, 3, 4]
        probability = np.asarray([[0.0, 0.4, 0.6], [1.0, 0.0, 0.0], [0.2, 0.7, 0.1]])
        rows = calibration_bin_rows(
            [4, 2, 3],
            probability,
            labels,
            run_id="run",
            config_hash="hash",
            method="raw",
            n_bins=10,
        )
        frame = pd.DataFrame(rows)
        self.assertEqual(frame.groupby("class_label")["n_samples"].sum().to_dict(), {2: 3, 3: 3, 4: 3})
        self.assertEqual(set(frame["run_id"]), {"run"})

    def test_method_selection_uses_predeclared_aggregate_rank(self) -> None:
        summary = pd.DataFrame(
            [
                {"method": "raw", "nll_log_loss_mean": 0.4, "multiclass_brier_mean": 0.3, "ece_confidence_mean": 0.08},
                {"method": "sigmoid", "nll_log_loss_mean": 0.35, "multiclass_brier_mean": 0.25, "ece_confidence_mean": 0.06},
                {"method": "isotonic", "nll_log_loss_mean": 0.5, "multiclass_brier_mean": 0.28, "ece_confidence_mean": 0.04},
            ]
        )
        selected, ranked = select_calibration_method(
            summary,
            ["nll_log_loss", "multiclass_brier", "ece_confidence"],
        )
        self.assertEqual(selected, "sigmoid")
        self.assertEqual(int(ranked["selected"].sum()), 1)

    def test_fold_summary_reports_uncertainty(self) -> None:
        rows = []
        for method in ("raw", "sigmoid", "isotonic"):
            for fold in range(1, 6):
                row = {"run_id": "r", "config_hash": "h", "method": method, "fold": fold}
                for metric in (
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "quadratic_weighted_kappa",
                    "ordinal_mae",
                    "severe_error_rate",
                    "nll_log_loss",
                    "multiclass_brier",
                    "ece_confidence",
                ):
                    row[metric] = 0.1 * fold
                rows.append(row)
        summary = summarize_fold_metrics(pd.DataFrame(rows))
        self.assertIn("nll_log_loss_ci_low", summary.columns)
        self.assertTrue((summary["n_folds"] == 5).all())


if __name__ == "__main__":
    unittest.main()
