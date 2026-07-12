from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.experiments.manuscript_shap_evidence import (
    ShapEvidenceError,
    assert_feature_names_allowed,
    reorder_grouped_shap_to_feature_order,
    select_representative_cases,
    shap_stability_pairwise,
    validate_shap_artifacts,
)


class ShapOutputsMatchPrimaryPolicyTests(unittest.TestCase):
    def test_forbidden_feature_is_rejected(self) -> None:
        with self.assertRaises(ShapEvidenceError):
            assert_feature_names_allowed(["Signal", "Age"], ["Age", "Gender"])

    def test_artifact_validation_requires_run_and_config_identity(self) -> None:
        valid = pd.DataFrame(
            {"run_id": ["r"], "config_hash": ["h"], "feature": ["Signal"]}
        )
        result = validate_shap_artifacts(
            global_tables=[valid],
            rankings=valid,
            local_values=valid,
            forbidden_features=["Age"],
            run_id="r",
            config_hash="h",
        )
        self.assertEqual(result["status"], "passed")
        invalid = valid.assign(config_hash="other")
        with self.assertRaises(ShapEvidenceError):
            validate_shap_artifacts(
                global_tables=[invalid],
                rankings=valid,
                local_values=valid,
                forbidden_features=["Age"],
                run_id="r",
                config_hash="h",
            )

    def test_representative_cases_cover_required_strata(self) -> None:
        frame = pd.DataFrame(
            [
                {"sample_index": 1, "fold": 1, "y_true": 2, "y_pred": 2, "confidence": 0.9},
                {"sample_index": 2, "fold": 1, "y_true": 3, "y_pred": 3, "confidence": 0.55},
                {"sample_index": 3, "fold": 2, "y_true": 4, "y_pred": 3, "confidence": 0.95},
                {"sample_index": 4, "fold": 2, "y_true": 3, "y_pred": 2, "confidence": 0.4},
                {"sample_index": 5, "fold": 3, "y_true": 4, "y_pred": 4, "confidence": 0.8},
                {"sample_index": 6, "fold": 3, "y_true": 3, "y_pred": 3, "confidence": 0.85},
            ]
        )
        frame["run_id"] = "r"
        frame["config_hash"] = "h"
        frame["policy"] = "p"
        selected = select_representative_cases(frame, [2, 3, 4])
        types = set(selected["case_type"])
        self.assertIn("correct_high_confidence", types)
        self.assertIn("incorrect_high_confidence", types)
        self.assertTrue(any(value.startswith("minority_class_") for value in types))
        self.assertEqual(set(selected["run_id"]), {"r"})
        self.assertEqual(set(selected["config_hash"]), {"h"})

    def test_pairwise_stability_has_all_fold_pairs_and_top_k(self) -> None:
        rows = []
        orders = {1: ["a", "b", "c"], 2: ["a", "c", "b"], 3: ["b", "a", "c"]}
        for fold, order in orders.items():
            for rank, feature in enumerate(order, start=1):
                rows.append(
                    {"run_id": "r", "config_hash": "h", "policy": "p", "fold": fold, "feature": feature, "rank": rank}
                )
        result = shap_stability_pairwise(pd.DataFrame(rows), [2])
        self.assertEqual(len(result), 3)
        self.assertEqual(set(result["top_k"]), {2})

    def test_grouped_preprocessor_order_is_explicitly_aligned_to_raw_features(self) -> None:
        grouped = np.asarray([[[10.0, 20.0, 30.0]]])
        aligned = reorder_grouped_shap_to_feature_order(
            grouped,
            ["numeric_a", "numeric_b", "categorical_a"],
            ["numeric_a", "categorical_a", "numeric_b"],
        )
        np.testing.assert_array_equal(aligned, np.asarray([[[10.0, 30.0, 20.0]]]))
        with self.assertRaisesRegex(ShapEvidenceError, "missing"):
            reorder_grouped_shap_to_feature_order(
                grouped,
                ["numeric_a", "numeric_b", "unexpected"],
                ["numeric_a", "categorical_a", "numeric_b"],
            )


if __name__ == "__main__":
    unittest.main()
