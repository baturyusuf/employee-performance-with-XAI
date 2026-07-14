from __future__ import annotations

import ast
import inspect
import unittest

import numpy as np
import pandas as pd

import src.experiments.manuscript_shap_evidence as shap_evidence_module
from src.experiments.manuscript_shap_evidence import (
    ShapEvidenceError,
    assert_feature_names_allowed,
    reorder_grouped_shap_to_feature_order,
    select_representative_cases,
    shap_stability_pairwise,
    summarize_stability,
    validate_shap_artifacts,
)
from src.utils.config_loader import load_config


IDENTITY = {
    "run_id": "r",
    "config_hash": "c" * 64,
    "scientific_input_hash": "s" * 64,
    "fold_contract_hash": "f" * 64,
    "policy": "p",
    "model": "xgboost",
    "model_set_sha256": "m" * 64,
}


def _with_identity(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for field, value in IDENTITY.items():
        result[field] = value
    return result


class ShapOutputsMatchPrimaryPolicyTests(unittest.TestCase):
    def test_canonical_config_freezes_exact_model_no_refit_shap_contract(self) -> None:
        settings = load_config("configs/manuscript_final.yaml")["manuscript_final"]["shap"]
        self.assertEqual(
            settings["model_source"],
            "model_benchmarks.persisted_selected_xgboost_outer_fold_pipelines",
        )
        self.assertFalse(settings["model_refit_in_shap_stage"])
        self.assertTrue(settings["oof_prediction_replay_required"])
        self.assertEqual(
            settings["stability"]["uncertainty_type"],
            "descriptive_dependent_fold_pairs",
        )
        self.assertFalse(settings["stability"]["confidence_interval_applicable"])

    def test_scientific_shap_stage_contains_no_refit_or_splitter_path(self) -> None:
        source = inspect.getsource(shap_evidence_module)
        tree = ast.parse(source)
        fit_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"
        ]
        self.assertEqual(fit_calls, [])
        self.assertNotIn("StratifiedKFold", source)
        self.assertNotIn("_fit_pipeline", source)

    def test_forbidden_feature_is_rejected(self) -> None:
        with self.assertRaises(ShapEvidenceError):
            assert_feature_names_allowed(["Signal", "Age"], ["Age", "Gender"])

    def test_artifact_validation_requires_run_and_config_identity(self) -> None:
        valid = _with_identity(pd.DataFrame({"feature": ["Signal"]}))
        result = validate_shap_artifacts(
            global_tables=[valid],
            rankings=valid,
            local_values=valid,
            forbidden_features=["Age"],
            identity=IDENTITY,
        )
        self.assertEqual(result["status"], "passed")
        invalid = valid.assign(config_hash="other")
        with self.assertRaises(ShapEvidenceError):
            validate_shap_artifacts(
                global_tables=[invalid],
                rankings=valid,
                local_values=valid,
                forbidden_features=["Age"],
                identity=IDENTITY,
            )

    def test_representative_cases_cover_required_strata(self) -> None:
        frame = pd.DataFrame(
            [
                {"sample_index": 1, "outer_fold": 1, "y_true": 2, "y_pred": 2, "confidence": 0.9},
                {"sample_index": 2, "outer_fold": 1, "y_true": 3, "y_pred": 3, "confidence": 0.55},
                {"sample_index": 3, "outer_fold": 2, "y_true": 4, "y_pred": 3, "confidence": 0.95},
                {"sample_index": 4, "outer_fold": 2, "y_true": 3, "y_pred": 2, "confidence": 0.4},
                {"sample_index": 5, "outer_fold": 3, "y_true": 4, "y_pred": 4, "confidence": 0.8},
                {"sample_index": 6, "outer_fold": 3, "y_true": 3, "y_pred": 3, "confidence": 0.85},
            ]
        )
        frame = _with_identity(frame)
        frame["model_sha256"] = frame["outer_fold"].map(lambda value: str(value) * 64)
        frame["selected_candidate_index"] = 0
        selected = select_representative_cases(frame, [2, 3, 4])
        types = set(selected["case_type"])
        self.assertIn("correct_high_confidence", types)
        self.assertIn("incorrect_high_confidence", types)
        self.assertTrue(any(value.startswith("minority_class_") for value in types))
        self.assertEqual(set(selected["run_id"]), {"r"})
        self.assertEqual(set(selected["config_hash"]), {"c" * 64})
        self.assertIn("model_set_sha256", selected.columns)

    def test_pairwise_stability_has_all_fold_pairs_and_top_k(self) -> None:
        rows = []
        orders = {
            fold: (["a", "b", "c"] if fold % 2 else ["a", "c", "b"])
            for fold in range(1, 11)
        }
        for fold, order in orders.items():
            for rank, feature in enumerate(order, start=1):
                rows.append(
                    {
                        "outer_fold": fold,
                        "model_sha256": str(fold).zfill(2) * 32,
                        "selected_candidate_index": 0,
                        "feature": feature,
                        "rank": rank,
                        **IDENTITY,
                    }
                )
        result = shap_stability_pairwise(pd.DataFrame(rows), [2])
        self.assertEqual(len(result), 45)
        self.assertEqual(set(result["top_k"]), {2})
        summary = summarize_stability(result)
        self.assertEqual(summary.loc[0, "n_fold_pairs"], 45)
        self.assertFalse(bool(summary.loc[0, "confidence_interval_applicable"]))
        self.assertEqual(
            summary.loc[0, "uncertainty_type"],
            "descriptive_dependent_fold_pairs",
        )
        self.assertFalse(any("ci_" in column for column in summary.columns))

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
