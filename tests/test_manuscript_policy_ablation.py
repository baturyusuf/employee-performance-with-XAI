from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.experiments.manuscript_policy_ablation import (
    PolicyAblationError,
    _holm_adjust,
    exact_policy_frame,
    leakage_sensitivity_indices,
    policy_pairwise_tests,
    summarize_policies,
)


class ManuscriptPolicyAblationTests(unittest.TestCase):
    def test_exact_policy_frame_requires_explicit_id_and_target(self) -> None:
        frame = pd.DataFrame({"EmpNumber": ["E1"], "Age": [30], "PerformanceRating": [3]})
        with self.assertRaises(PolicyAblationError):
            exact_policy_frame(
                frame,
                "bad",
                {"excluded_features": ["PerformanceRating"]},
                target_column="PerformanceRating",
                id_column="EmpNumber",
            )

    def test_exact_policy_frame_applies_only_declared_definition(self) -> None:
        frame = pd.DataFrame(
            {
                "EmpNumber": ["E1"],
                "Age": [30],
                "Gender": ["F"],
                "Signal": [1.0],
                "PerformanceRating": [3],
            }
        )
        result, excluded = exact_policy_frame(
            frame,
            "primary",
            {"excluded_features": ["EmpNumber", "PerformanceRating", "Age", "Gender"]},
            target_column="PerformanceRating",
            id_column="EmpNumber",
        )
        self.assertEqual(list(result.columns), ["Signal"])
        self.assertEqual(excluded, ["EmpNumber", "PerformanceRating", "Age", "Gender"])

    def test_holm_adjustment_is_monotone_in_sorted_p_values(self) -> None:
        raw = [0.01, 0.04, 0.02]
        adjusted = _holm_adjust(raw)
        ordered = sorted(zip(raw, adjusted))
        self.assertEqual([value for _, value in ordered], sorted(value for _, value in ordered))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))

    def test_summary_pairwise_and_sensitivity_keep_run_contract(self) -> None:
        rows = []
        for policy, offset in (("full_feature_upper_bound", 0.0), ("reduced", -0.05)):
            for fold in range(1, 6):
                rows.append(
                    {
                        "run_id": "run-1",
                        "config_hash": "abc",
                        "policy": policy,
                        "role": "diagnostic" if policy.startswith("full") else "candidate",
                        "audit_only": False,
                        "fold": fold,
                        "n_features": 5 if policy.startswith("full") else 4,
                        "excluded_features": "EmpNumber;PerformanceRating",
                        "accuracy": 0.8 + offset,
                        "balanced_accuracy": 0.7 + offset,
                        "macro_f1": 0.6 + offset + fold / 1000,
                        "weighted_f1": 0.75 + offset,
                        "quadratic_weighted_kappa": 0.5 + offset + fold / 1000,
                        "ordinal_mae": 0.2 - offset,
                        "severe_error_rate": 0.01,
                        "nll_log_loss": 0.4 - offset,
                        "multiclass_brier": 0.3 - offset,
                        "ece_confidence": 0.05,
                    }
                )
        frame = pd.DataFrame(rows)
        summary = summarize_policies(frame)
        pairs = policy_pairwise_tests(frame, alpha=0.05)
        sensitivity = leakage_sensitivity_indices(frame)
        self.assertEqual(set(summary["run_id"]), {"run-1"})
        self.assertEqual(set(pairs["config_hash"]), {"abc"})
        reduced_macro = sensitivity[
            (sensitivity["policy"] == "reduced") & (sensitivity["metric"] == "macro_f1")
        ].iloc[0]
        self.assertGreater(reduced_macro["index_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
