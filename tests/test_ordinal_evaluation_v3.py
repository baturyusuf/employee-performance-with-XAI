from __future__ import annotations

import numpy as np
import pytest

from src.models.ordinal_evaluation_v3 import (
    OrdinalEvaluationContractError,
    ordinal_evaluation_bundle_v3,
    ranked_probability_score,
    two_level_reversal_rate,
)


LABELS = [2, 3, 4]


def test_ranked_probability_score_is_zero_for_perfect_probabilities() -> None:
    target = [2, 3, 4]
    probabilities = np.eye(3)
    assert ranked_probability_score(target, probabilities, labels=LABELS) == 0.0


def test_ranked_probability_score_is_one_for_opposite_extreme_prediction() -> None:
    assert ranked_probability_score([2], np.asarray([[0.0, 0.0, 1.0]]), labels=LABELS) == 1.0
    assert ranked_probability_score([4], np.asarray([[1.0, 0.0, 0.0]]), labels=LABELS) == 1.0


def test_two_level_reversal_uses_class_positions_not_numeric_distance() -> None:
    target = [10, 10, 20, 30]
    predicted = [30, 20, 30, 10]
    assert two_level_reversal_rate(target, predicted, labels=[10, 20, 30]) == 0.5


def test_bundle_returns_renamed_metric_per_class_rows_and_full_confusion_grid() -> None:
    target = [2, 2, 3, 3, 4, 4]
    predicted = [2, 4, 3, 2, 4, 2]
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.2, 0.7],
            [0.7, 0.2, 0.1],
        ]
    )
    bundle = ordinal_evaluation_bundle_v3(
        target,
        predicted,
        probabilities,
        labels=LABELS,
        dataset_key="inx_primary",
        model_name="example",
    )

    metrics = bundle["aggregate_metrics"]
    assert "severe_error_rate" not in metrics
    assert len(metrics) == 16
    assert all(value is not None for value in metrics.values())
    assert metrics["two_level_reversal_rate"] == pytest.approx(2 / 6)
    assert 0.0 <= metrics["ranked_probability_score"] <= 1.0
    assert len(bundle["per_class_metrics"]) == 3
    assert sum(row["support"] for row in bundle["per_class_metrics"]) == 6
    assert len(bundle["confusion_matrix"]) == 9
    assert sum(row["count"] for row in bundle["confusion_matrix"]) == 6
    assert bundle["two_level_reversal_definition"].endswith("greater_than_or_equal_to_two")


@pytest.mark.parametrize(
    "target,predicted,probabilities,message",
    [
        ([2, 3], [2], np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), "equally sized"),
        ([2], [2], np.asarray([[0.5, 0.5]]), "Probability shape"),
        ([2], [2], np.asarray([[0.5, 0.4, 0.0]]), "sum to one"),
        ([2], [9], np.asarray([[1.0, 0.0, 0.0]]), "declared ordered labels"),
    ],
)
def test_bundle_fails_closed_on_invalid_evidence_inputs(
    target,
    predicted,
    probabilities,
    message,
) -> None:
    with pytest.raises(OrdinalEvaluationContractError, match=message):
        ordinal_evaluation_bundle_v3(
            target,
            predicted,
            probabilities,
            labels=LABELS,
            dataset_key="inx_primary",
            model_name="example",
        )


def test_rps_rejects_undeclared_target_and_non_simplex_probabilities() -> None:
    with pytest.raises(OrdinalEvaluationContractError, match="undeclared"):
        ranked_probability_score([9], np.asarray([[1.0, 0.0, 0.0]]), labels=LABELS)
    with pytest.raises(OrdinalEvaluationContractError, match="sum to one"):
        ranked_probability_score([2], np.asarray([[0.7, 0.2, 0.0]]), labels=LABELS)
