from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.experiments.manuscript_fairness_proxy import (
    FairnessProxyError,
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
                    "outer_fold": index % 3 + 1,
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

    def test_missing_configured_audit_attribute_fails_closed(self) -> None:
        predictions, data = self._fixture()

        with self.assertRaisesRegex(FairnessProxyError, "audit attribute.*MissingAudit"):
            compute_group_metric_rows(
                predictions,
                data,
                labels=[2, 3, 4],
                attributes=["Gender", "MissingAudit"],
                transforms={},
                sensitive_attributes={"Gender"},
                minimum_group_support=10,
                minimum_class_denominator=3,
            )

    def test_audit_attributes_join_by_sample_identity_not_row_position(self) -> None:
        predictions, data = self._fixture()
        expected = compute_group_metric_rows(
            predictions,
            data,
            labels=[2, 3, 4],
            attributes=["Gender"],
            transforms={},
            sensitive_attributes={"Gender"},
            minimum_group_support=10,
            minimum_class_denominator=3,
        )
        remap = {index: 1000 + index * 3 for index in range(len(data))}
        remapped_predictions = predictions.copy()
        remapped_predictions["sample_index"] = remapped_predictions["sample_index"].map(remap)
        remapped_data = data.copy()
        remapped_data.index = [remap[index] for index in range(len(data))]
        remapped_data = remapped_data.sample(frac=1.0, random_state=91)

        observed = compute_group_metric_rows(
            remapped_predictions.sample(frac=1.0, random_state=17),
            remapped_data,
            labels=[2, 3, 4],
            attributes=["Gender"],
            transforms={},
            sensitive_attributes={"Gender"},
            minimum_group_support=10,
            minimum_class_denominator=3,
        )

        columns = [
            "attribute",
            "group_value",
            "metric",
            "class_label",
            "metric_value",
            "metric_denominator",
            "eligible_for_gap",
        ]
        sort_by = ["attribute", "group_value", "metric", "class_label"]
        expected = expected[columns].sort_values(sort_by, na_position="first").reset_index(drop=True)
        observed = observed[columns].sort_values(sort_by, na_position="first").reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, observed, check_exact=True)

    def test_bootstrap_freezes_full_oof_group_eligibility(self) -> None:
        labels = [2, 3, 4]
        records = []
        data_rows = []
        for index in range(60):
            y_true = labels[index % len(labels)]
            group = "A" if index < 30 else "B"
            records.append(
                {
                    "run_id": "test_run",
                    "config_hash": "a" * 64,
                    "policy": "primary",
                    "outer_fold": index % 10 + 1,
                    "sample_index": index,
                    "y_true": y_true,
                    "y_pred": y_true if group == "A" else labels[(index + 1) % 3],
                    **{
                        f"prob_class_{label}": 0.8 if label == y_true else 0.1
                        for label in labels
                    },
                }
            )
            data_rows.append({"Gender": group})
        predictions = pd.DataFrame(records)
        data = pd.DataFrame(data_rows)
        group_metrics = compute_group_metric_rows(
            predictions,
            data,
            labels=labels,
            attributes=["Gender"],
            transforms={},
            sensitive_attributes={"Gender"},
            minimum_group_support=30,
            minimum_class_denominator=3,
        )

        disparity = summarize_disparities_with_bootstrap(
            group_metrics,
            predictions,
            data,
            labels=labels,
            attributes=["Gender"],
            transforms={},
            sensitive_attributes={"Gender"},
            minimum_group_support=30,
            minimum_class_denominator=3,
            n_bootstrap=200,
            confidence_level=0.95,
            seed=42,
            minimum_valid_fraction=0.8,
            wide_interval_threshold=1.0,
        )

        accuracy = disparity[disparity["metric"] == "accuracy"].iloc[0]
        self.assertEqual(accuracy["n_groups_included"], 2)
        self.assertEqual(accuracy["valid_bootstrap_samples"], 200)
        self.assertEqual(accuracy["valid_bootstrap_fraction"], 1.0)

    def test_wide_or_unstable_interval_is_not_headline_eligible(self) -> None:
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
            wide_interval_threshold=0.0,
        )

        self.assertIn("headline_eligible", disparity.columns)
        wide_or_unstable = disparity[
            disparity["estimate_status"].isin(
                {
                    "support_sufficient_but_interval_wide",
                    "unstable_insufficient_valid_bootstrap_replicates",
                    "insufficient_subgroup_or_metric_support",
                }
            )
        ]
        self.assertFalse(wide_or_unstable.empty)
        self.assertFalse(wide_or_unstable["headline_eligible"].astype(bool).any())

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
            self.assertNotIn("EmpNumber", predictor_frame.columns)

        including, including_columns, including_removed = predictors[
            "no_salary_hike_no_attrition"
        ]
        department_free, department_free_columns, department_free_removed = predictors[
            "no_salary_hike_no_attrition_no_department"
        ]
        strict, strict_columns, strict_removed = predictors[
            "no_salary_hike_no_attrition_no_department_no_job_role"
        ]
        self.assertEqual(including_columns, department_free_columns)
        pd.testing.assert_frame_equal(including, department_free)
        self.assertTrue(including_removed)
        self.assertFalse(department_free_removed)
        self.assertFalse(strict_removed)
        self.assertIn("EmpJobRole", including_columns)
        self.assertNotIn("EmpJobRole", strict_columns)


if __name__ == "__main__":
    unittest.main()
