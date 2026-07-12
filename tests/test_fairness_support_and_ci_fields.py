from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.experiments.manuscript_fairness_proxy import (
    compute_group_metric_rows,
    proxy_predictor_frames,
    summarize_disparities_with_bootstrap,
)
from src.utils.config_loader import load_config


class FairnessSupportAndConfidenceIntervalTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
        records = []
        data_rows = []
        labels = [2, 3, 4]
        for index in range(36):
            group = "small" if index == 35 else ("A" if index < 18 else "B")
            y_true = labels[index % 3]
            y_pred = y_true if group == "A" else labels[(index + 1) % 3]
            probabilities = {label: 0.1 for label in labels}
            probabilities[y_pred] = 0.8
            records.append(
                {
                    "run_id": "test_run",
                    "config_hash": "a" * 64,
                    "policy": "primary",
                    "fold": index % 3 + 1,
                    "sample_index": index,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    **{f"prob_class_{label}": probabilities[label] for label in labels},
                }
            )
            data_rows.append({"Gender": group})
        return pd.DataFrame(records), pd.DataFrame(data_rows)

    def test_manuscript_rows_report_support_ci_and_valid_bootstrap_counts(self) -> None:
        predictions, data = self._fixture()
        group_metrics = compute_group_metric_rows(
            predictions,
            data,
            labels=[2, 3, 4],
            attributes=["Gender"],
            transforms={},
            sensitive_attributes={"Gender"},
            minimum_group_support=10,
            minimum_class_denominator=3,
        )
        disparity = summarize_disparities_with_bootstrap(
            group_metrics,
            predictions,
            data,
            labels=[2, 3, 4],
            attributes=["Gender"],
            transforms={},
            sensitive_attributes={"Gender"},
            minimum_group_support=10,
            minimum_class_denominator=3,
            n_bootstrap=80,
            confidence_level=0.95,
            seed=42,
            minimum_valid_fraction=0.5,
            wide_interval_threshold=1.0,
        )
        required = {
            "attribute",
            "metric",
            "class_label",
            "gap",
            "ci_low",
            "ci_high",
            "minimum_subgroup_support",
            "minimum_metric_denominator",
            "valid_bootstrap_samples",
            "interpretation_category",
            "estimate_status",
            "limitations",
            "run_id",
            "config_hash",
        }
        self.assertTrue(required.issubset(disparity.columns))
        accuracy = disparity[disparity["metric"] == "accuracy"].iloc[0]
        self.assertEqual(accuracy["n_groups_total"], 3)
        self.assertEqual(accuracy["n_groups_included"], 2)
        self.assertGreaterEqual(accuracy["minimum_subgroup_support"], 10)
        self.assertGreater(accuracy["valid_bootstrap_samples"], 0)
        self.assertTrue(np.isfinite(accuracy["ci_low"]))
        self.assertTrue(np.isfinite(accuracy["ci_high"]))
        self.assertEqual(
            accuracy["interpretation_category"], "protected_or_sensitive_descriptive_audit"
        )
        self.assertIn("do not establish", accuracy["limitations"])

    def test_class_specific_denominators_are_not_replaced_by_group_size(self) -> None:
        predictions, data = self._fixture()
        group_metrics = compute_group_metric_rows(
            predictions,
            data,
            labels=[2, 3, 4],
            attributes=["Gender"],
            transforms={},
            sensitive_attributes={"Gender"},
            minimum_group_support=10,
            minimum_class_denominator=10,
        )
        tpr = group_metrics[group_metrics["metric"] == "true_positive_rate"]
        supported_groups = tpr[tpr["group_support_eligible"]]
        self.assertTrue((supported_groups["metric_denominator"] < supported_groups["n_samples"]).all())
        self.assertFalse(supported_groups["eligible_for_gap"].any())

    def test_proxy_target_is_removed_from_every_policy_predictor_frame(self) -> None:
        settings = load_config("configs/manuscript_final.yaml")["manuscript_final"]
        columns = [
            "Age",
            "Gender",
            "MaritalStatus",
            "EmpDepartment",
            "EmpJobRole",
            "EmpLastSalaryHikePercent",
            "Attrition",
            "EmpNumber",
            "PerformanceRating",
            "EmpJobSatisfaction",
        ]
        frame = pd.DataFrame([{column: 1 for column in columns}])
        predictors = proxy_predictor_frames(frame, settings)
        self.assertEqual(set(predictors), {
            "no_salary_hike_no_attrition",
            "no_salary_hike_no_attrition_no_department",
            "no_salary_hike_no_attrition_no_department_no_job_role",
        })
        for predictor_frame, _, _ in predictors.values():
            self.assertNotIn("EmpDepartment", predictor_frame.columns)
            self.assertNotIn("PerformanceRating", predictor_frame.columns)


if __name__ == "__main__":
    unittest.main()
