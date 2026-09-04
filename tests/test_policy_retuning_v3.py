from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.policy_retuning_v3 import (
    FIXED_ESTIMAND,
    RETUNED_ESTIMAND,
    PolicyRetuningV3Error,
    evaluate_policy_retuning_v3,
    preflight_policy_retuning_v3,
    selected_candidate_frequency_v3,
    summarize_policy_oof_v3,
)
from src.experiments.shared_folds import generate_shared_folds
from src.governance.policy_retuning_contract_v3 import POLICY_IDS, POLICY_NAMES


class _FastClassifier:
    def fit(self, X, y):
        self.classes_ = np.asarray([2, 3, 4])
        return self

    def predict(self, X):
        return np.asarray(X["signal"], dtype=int)

    def predict_proba(self, X):
        predicted = self.predict(X)
        probability = np.full((len(predicted), 3), 0.01, dtype=float)
        for row, label in enumerate(predicted):
            probability[row, int(label) - 2] = 0.98
        return probability


def _contract() -> dict:
    return json.loads(Path("configs/policy_retuning_v3.json").read_text(encoding="utf-8"))


def _synthetic_inputs():
    target_array = np.resize(np.asarray([2, 3, 4]), 90)
    source = pd.DataFrame(
        {
            "EmpNumber": np.arange(10_000, 10_090),
            "PerformanceRating": target_array,
        }
    )
    target = pd.Series(target_array)
    feature_frames = {
        policy_id: pd.DataFrame(
            {
                "signal": target_array,
                "noise": np.linspace(-1.0, 1.0, len(target_array)),
            }
        )
        for policy_id in POLICY_IDS
    }
    folds = generate_shared_folds(
        source,
        target_column="PerformanceRating",
        id_column="EmpNumber",
        run_id="synthetic_policy_retuning",
        config_hash="a" * 64,
        scientific_input_hash="b" * 64,
        dataset_key="inx_primary",
        dataset_sha256="c" * 64,
        outer_splits=10,
        inner_splits=5,
        seed=42,
        inner_seed=43,
    )
    schedule_rows = []
    for outer_fold in range(1, 11):
        schedule_rows.append(
            {
                "outer_fold": outer_fold,
                "model": "xgboost",
                "selected_candidate_index": 0,
                "selected_candidate_parameters_json": "{}",
                "fixed_parameters_json": "{}",
                "outer_test_used_for_selection": False,
            }
        )
    schedule = pd.DataFrame(schedule_rows)
    probability = _FastClassifier().fit(None, None).predict_proba(feature_frames["P0"])
    aliases = [
        "full_feature_upper_bound",
        "no_salary_hike_no_attrition_sensitive_retaining_audit",
        "no_salary_hike_no_attrition",
        "no_salary_hike_no_attrition_no_department",
    ]
    fixed_rows = []
    outer = folds.outer_assignments.set_index("sample_index")
    for alias in aliases:
        for sample_index in range(len(source)):
            fixed_rows.append(
                {
                    "system_id": alias,
                    "sample_index": sample_index,
                    "outer_fold": int(outer.loc[sample_index, "outer_fold"]),
                    "y_true": int(target.iloc[sample_index]),
                    "y_pred": int(target.iloc[sample_index]),
                    "selected_candidate_index": 0,
                    "prob_class_2": probability[sample_index, 0],
                    "prob_class_3": probability[sample_index, 1],
                    "prob_class_4": probability[sample_index, 2],
                }
            )
    policy_rows = pd.DataFrame(
        [
            {
                "policy_id": policy_id,
                "policy_order": order,
                "policy_name": policy_name,
                "policy_role": "synthetic",
                "n_features": 2,
                "retained_features_json": '["signal","noise"]',
                "excluded_features_json": "[]",
            }
            for order, (policy_id, policy_name) in enumerate(zip(POLICY_IDS, POLICY_NAMES))
        ]
    )
    model_definition = {"fixed_params": {}, "candidates": [{"candidate": index} for index in range(8)]}
    return (
        source,
        feature_frames,
        target,
        folds,
        model_definition,
        schedule,
        pd.DataFrame(fixed_rows),
        policy_rows,
    )


def test_mocked_full_policy_comparison_has_exact_estimand_grids(monkeypatch) -> None:
    (
        source,
        feature_frames,
        target,
        folds,
        model_definition,
        schedule,
        fixed_source,
        policy_rows,
    ) = _synthetic_inputs()
    fit_count = 0

    def _fit(estimator, X, y, **kwargs):
        nonlocal fit_count
        fit_count += 1
        return estimator.fit(X, y)

    monkeypatch.setattr(
        "src.experiments.policy_retuning_v3._pipeline",
        lambda *args, **kwargs: _FastClassifier(),
    )
    monkeypatch.setattr("src.experiments.policy_retuning_v3._fit_or_fail", _fit)
    monkeypatch.setattr(
        "src.experiments.policy_retuning_v3._selection_scores",
        lambda *args, **kwargs: (0.5, 0.25),
    )
    result = evaluate_policy_retuning_v3(
        source,
        feature_frames,
        target,
        folds,
        model_definition,
        schedule,
        fixed_source,
        _contract(),
        policy_rows,
        run_id="synthetic_policy_v3",
        policy_contract_sha256="d" * 64,
        scientific_input_sha256="e" * 64,
    )
    assert result.evidence_status == "complete_two_estimand_exactly_once_oof"
    assert fit_count == 2480
    assert len(result.candidate_search_results) == 6 * 10 * 8
    assert len(result.selected_hyperparameters) == 6 * 10
    assert len(result.selected_candidate_frequency) == 6
    assert len(result.fixed_oof_predictions) == 6 * 90
    assert len(result.retuned_oof_predictions) == 6 * 90
    assert len(result.combined_oof_predictions) == 2 * 6 * 90
    assert len(result.fold_metrics) == 2 * 6 * 10
    assert len(result.aggregate_metrics) == 2 * 6 * 16
    assert len(result.metric_comparison) == 6 * 16
    assert len(result.headline_policy_comparison) == 6
    assert not result.candidate_search_results["outer_test_used_for_selection"].any()
    assert result.combined_oof_predictions.groupby(["estimand", "policy_id"])[
        "sample_index"
    ].nunique().eq(90).all()
    assert set(result.combined_oof_predictions["estimand"]) == {
        FIXED_ESTIMAND,
        RETUNED_ESTIMAND,
    }
    assert result.metric_comparison["raw_difference_retuned_minus_fixed"].abs().le(1e-14).all()
    subset_fold, _, subset_comparison, subset_headline = summarize_policy_oof_v3(
        result.combined_oof_predictions[
            result.combined_oof_predictions["policy_id"] == "P3"
        ],
        policy_feature_contract=policy_rows[policy_rows["policy_id"] == "P3"],
        total_sample_count=len(source),
    )
    assert set(subset_comparison["policy_id"]) == {"P3"}
    assert subset_headline["policy_id"].tolist() == ["P3"]
    assert subset_fold["n_train"].eq(81).all()


def test_summarizer_rejects_a_missing_estimand() -> None:
    combined = pd.DataFrame(
        [
            {
                "run_id": "x",
                "policy_contract_sha256": "a" * 64,
                "scientific_input_sha256": "b" * 64,
                "fold_contract_hash": "c" * 64,
                "estimand": FIXED_ESTIMAND,
                "policy_id": "P0",
                "policy_name": "INFORMATION_RICH_DIAGNOSTIC",
                "n_features": 1,
                "model": "xgboost",
                "sample_index": 0,
                "outer_fold": 1,
                "y_true": 2,
                "y_pred": 2,
                "prob_class_2": 1.0,
                "prob_class_3": 0.0,
                "prob_class_4": 0.0,
            }
        ]
    )
    policy_rows = pd.DataFrame(
        [{"policy_id": "P0", "policy_name": "INFORMATION_RICH_DIAGNOSTIC", "n_features": 1}]
    )
    with pytest.raises(PolicyRetuningV3Error, match="estimand set"):
        summarize_policy_oof_v3(combined, policy_feature_contract=policy_rows)


def test_candidate_frequency_rejects_outer_test_selection() -> None:
    selected = pd.DataFrame(
        [
            {
                "policy_id": "P0",
                "policy_name": "INFORMATION_RICH_DIAGNOSTIC",
                "outer_fold": 1,
                "selected_candidate_index": 0,
                "selected_candidate_parameters_json": "{}",
                "outer_test_used_for_selection": True,
            }
        ]
    )
    with pytest.raises(PolicyRetuningV3Error, match="Outer test entered"):
        selected_candidate_frequency_v3(selected)


@pytest.mark.skipif(
    not Path("data/raw/inx_employee_performance.csv").is_file(),
    reason="ignored local INX input is unavailable",
)
def test_real_policy_preflight_validates_sources_without_fit(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.experiments.policy_retuning_v3._fit_or_fail",
        lambda *args, **kwargs: pytest.fail("preflight attempted a model fit"),
    )
    receipt = preflight_policy_retuning_v3()
    assert receipt["status"] == "passed"
    assert receipt["sample_count"] == 1200
    assert receipt["target_support"] == {"2": 194, "3": 874, "4": 132}
    assert receipt["policy_feature_counts"] == {
        "P0": 26,
        "P1": 24,
        "P2": 21,
        "P3": 20,
        "P4": 13,
        "P5": 6,
    }
    assert receipt["outer_splits"] == 10
    assert receipt["inner_splits"] == 5
    assert receipt["candidate_count"] == 8
    assert receipt["reusable_fixed_oof_rows"] == 4800
    assert receipt["planned_new_estimator_fit_calls"] == 2480
    assert receipt["model_fit_count"] == 0
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0
