from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.repeated_nested_cv_v3 import (
    ALL_MODEL_NAMES,
    TUNED_MODEL_NAMES,
    RepeatedNestedCVError,
    evaluate_repeated_nested_cv_v3,
    preflight_repeated_nested_cv_v3,
    selected_candidate_frequency_v3,
    summarize_repeated_metrics_v3,
)
from src.governance.repeated_nested_cv_contract_v3 import PRIORITY_METRICS


def _contract() -> dict:
    return json.loads(Path("configs/repeated_nested_cv_v3.json").read_text(encoding="utf-8"))


class _FastClassifier:
    def __init__(self, model_index: int) -> None:
        self.model_index = model_index

    def fit(self, X, y):
        self.classes_ = np.asarray([2, 3, 4])
        return self

    def predict(self, X):
        predicted = np.asarray(X["signal"], dtype=int).copy()
        if self.model_index == 1:
            predicted[predicted == 2] = 3
        elif self.model_index == 2:
            predicted[predicted == 4] = 3
        elif self.model_index == 3:
            predicted[:] = 3
        elif self.model_index == 4:
            predicted[predicted == 2] = 4
        elif self.model_index == 5:
            predicted[:] = 2
        return predicted

    def predict_proba(self, X):
        predicted = self.predict(X)
        probabilities = np.full((len(predicted), 3), 0.01, dtype=float)
        for row, value in enumerate(predicted):
            probabilities[row, int(value) - 2] = 0.98
        return probabilities


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    target = np.resize(np.asarray([2, 3, 4]), 75)
    source = pd.DataFrame(
        {
            "EmpNumber": np.arange(10_000, 10_075),
            "PerformanceRating": target,
        }
    )
    features = pd.DataFrame(
        {
            "signal": target,
            "noise": np.linspace(-1.0, 1.0, len(target)),
        }
    )
    return source, features, pd.Series(target)


def test_mocked_full_five_repetition_evaluation_has_exact_grids(monkeypatch) -> None:
    source, features, target = _synthetic_inputs()
    definitions = {
        name: {"fixed_params": {}, "candidates": [{}]} for name in TUNED_MODEL_NAMES
    }
    model_positions = {name: index for index, name in enumerate(TUNED_MODEL_NAMES)}
    monkeypatch.setattr(
        "src.experiments.repeated_nested_cv_v3._model_definitions",
        lambda *args, **kwargs: definitions,
    )
    monkeypatch.setattr(
        "src.experiments.repeated_nested_cv_v3._pipeline",
        lambda model_name, *args, **kwargs: _FastClassifier(model_positions[model_name]),
    )
    monkeypatch.setattr(
        "src.experiments.repeated_nested_cv_v3._selection_scores",
        lambda *args, **kwargs: (0.5, 0.25),
    )
    monkeypatch.setattr(
        "src.experiments.repeated_nested_cv_v3._fit_or_fail",
        lambda estimator, X, y, **kwargs: estimator.fit(X, y),
    )
    result = evaluate_repeated_nested_cv_v3(
        source,
        features,
        target,
        _contract(),
        {},
        {},
        run_id="synthetic_repeated_v3",
        repeated_contract_sha256="a" * 64,
        scientific_input_sha256="b" * 64,
        dataset_sha256="c" * 64,
    )

    assert result.evidence_status == "complete_five_repetition_exactly_once_oof"
    assert len(result.fold_contracts) == 5
    assert len(
        {
            record["outer_assignment_semantic_sha256"]
            for record in result.fold_contracts
        }
    ) == 5
    assert len(result.candidate_search_results) == 5 * 5 * 6
    assert len(result.selected_hyperparameters) == 5 * 5 * 9
    assert len(result.fold_metrics) == 5 * 5 * 9
    assert len(result.oof_predictions) == 5 * 9 * 75
    assert len(result.repetition_metrics) == 5 * 9 * 16
    assert len(result.variability_summary) == 9 * 4
    assert len(result.rank_by_repetition) == 5 * 6 * 4
    assert len(result.model_rank_summary) == 6 * 4
    assert len(result.ordering_stability) == 4
    assert result.oof_predictions.groupby(["repetition", "model"])[
        "sample_index"
    ].nunique().eq(75).all()
    assert not result.candidate_search_results["outer_test_used_for_selection"].any()


def _synthetic_repetition_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model_offsets = {name: index * 0.02 for index, name in enumerate(ALL_MODEL_NAMES)}
    for repetition in range(1, 6):
        for model_name in ALL_MODEL_NAMES:
            for metric in PRIORITY_METRICS:
                base = model_offsets[model_name] + repetition * 0.001
                value = 1.0 - base if metric == "ordinal_mae" else base
                rows.append(
                    {
                        "repetition": repetition,
                        "model_name": model_name,
                        "metric": metric,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def test_variability_and_ordering_summaries_are_descriptive_and_complete() -> None:
    variability, ranks, rank_summary, stability = summarize_repeated_metrics_v3(
        _synthetic_repetition_metrics()
    )
    assert len(variability) == 9 * 4
    assert variability["repetition_count"].eq(5).all()
    assert variability["range_interpretation"].eq(
        "empirical_repetition_range_not_confidence_interval"
    ).all()
    assert len(ranks) == 5 * 6 * 4
    assert len(rank_summary) == 6 * 4
    assert len(stability) == 4
    assert stability["repetition_pair_count"].eq(10).all()
    assert stability["mean_pairwise_rank_spearman"].eq(1.0).all()
    assert stability["modal_winner_frequency"].eq(1.0).all()


def test_repetition_summary_rejects_a_missing_model_metric_cell() -> None:
    frame = _synthetic_repetition_metrics().iloc[:-1].copy()
    with pytest.raises(RepeatedNestedCVError, match="grid is incomplete"):
        summarize_repeated_metrics_v3(frame)


def test_candidate_frequency_uses_only_training_selected_records() -> None:
    rows = []
    for repetition in range(1, 6):
        for outer_fold in range(1, 6):
            for model_name in TUNED_MODEL_NAMES:
                selected_index = (repetition + outer_fold) % 2
                rows.append(
                    {
                        "repetition": repetition,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "selection_performed": True,
                        "selected_candidate_index": selected_index,
                        "selected_candidate_parameters_json": json.dumps(
                            {"candidate": selected_index}
                        ),
                        "outer_test_used_for_selection": False,
                    }
                )
    frequency = selected_candidate_frequency_v3(pd.DataFrame(rows))
    assert frequency.groupby("model_name")["selection_count"].sum().eq(25).all()
    assert frequency.groupby("model_name")["selection_frequency"].sum().eq(1.0).all()


def test_candidate_frequency_rejects_outer_test_selection() -> None:
    frame = pd.DataFrame(
        [
            {
                "repetition": 1,
                "outer_fold": 1,
                "model": "xgboost",
                "selection_performed": True,
                "selected_candidate_index": 0,
                "selected_candidate_parameters_json": "{}",
                "outer_test_used_for_selection": True,
            }
        ]
    )
    with pytest.raises(RepeatedNestedCVError, match="Outer test entered"):
        selected_candidate_frequency_v3(frame)


@pytest.mark.skipif(
    not Path("data/raw/inx_employee_performance.csv").is_file(),
    reason="ignored local INX input is unavailable",
)
def test_real_preflight_generates_five_distinct_fold_assignments_without_fit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.experiments.repeated_nested_cv_v3._fit_or_fail",
        lambda *args, **kwargs: pytest.fail("preflight attempted a model fit"),
    )
    receipt = preflight_repeated_nested_cv_v3()
    assert receipt["status"] == "passed"
    assert receipt["sample_count"] == 1200
    assert receipt["feature_count"] == 20
    assert receipt["repetitions"] == 5
    assert receipt["distinct_outer_assignment_count"] == 5
    assert len(set(receipt["fold_contract_hashes"])) == 5
    assert receipt["planned_estimator_fit_calls"] == 5_725
    assert receipt["model_fit_count"] == 0
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0
