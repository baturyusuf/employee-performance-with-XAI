from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluate import classification_metrics
from src.models.task_schema import (
    BINARY_ATTRITION_TRANSFER,
    BINARY_TURNOVER_TRANSFER,
    NOMINAL_MULTICLASS_PROXY_DIAGNOSTIC,
    ORDINAL_METRICS,
    ORDINAL_MULTICLASS_PERFORMANCE,
    RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS,
    KNOWN_METRICS,
    METRIC_DEFINITIONS,
    canonical_task_type,
    get_task_schema,
    metric_schema_hash,
    metric_schema_records,
)


def test_legacy_task_names_resolve_to_explicit_canonical_schema() -> None:
    assert canonical_task_type("restricted_ordinal_performance") == RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS
    assert canonical_task_type("binary_attrition") == BINARY_ATTRITION_TRANSFER
    assert canonical_task_type("binary_turnover") == BINARY_TURNOVER_TRANSFER


@pytest.mark.parametrize("task_type", [BINARY_ATTRITION_TRANSFER, BINARY_TURNOVER_TRANSFER])
def test_binary_transfer_metrics_mark_ordinal_values_na(task_type: str) -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    y_proba = np.array(
        [
            [0.9, 0.1],
            [0.4, 0.6],
            [0.7, 0.3],
            [0.2, 0.8],
        ]
    )

    metrics = classification_metrics(y_true, y_pred, y_proba, labels=[0, 1], task_type=task_type)

    assert metrics["macro_f1"] is not None
    assert metrics["nll_log_loss"] is not None
    assert metrics["binary_brier"] is not None
    assert metrics["roc_auc"] is not None
    assert metrics["average_precision"] is not None
    assert metrics["multiclass_brier"] is None
    for metric in ORDINAL_METRICS:
        assert metrics[metric] is None, f"{task_type}/{metric} must be N/A, not zero"


def test_restricted_target_metrics_are_not_presented_as_three_class_ordinal_metrics() -> None:
    metrics = classification_metrics(
        [3, 3, 4, 4],
        [3, 4, 3, 4],
        np.array([[0.9, 0.1], [0.4, 0.6], [0.7, 0.3], [0.2, 0.8]]),
        labels=[3, 4],
        task_type=RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS,
    )

    assert metrics["macro_f1"] is not None
    assert metrics["nll_log_loss"] is not None
    assert metrics["ece_confidence"] is not None
    assert metrics["binary_brier"] is not None
    assert metrics["roc_auc"] is not None
    assert metrics["average_precision"] is not None
    assert metrics["multiclass_brier"] is None
    for metric in ORDINAL_METRICS:
        assert metrics[metric] is None


def test_nominal_proxy_task_is_not_labelled_as_performance_or_ordinal() -> None:
    schema = get_task_schema(NOMINAL_MULTICLASS_PROXY_DIAGNOSTIC)

    assert schema.comparison_group == "nominal_proxy_risk_diagnostic"
    assert schema.is_metric_applicable("macro_f1")
    assert not schema.ordinal_metrics_comparable
    assert not schema.is_metric_applicable("quadratic_weighted_kappa")
    assert "not an employee-performance task" in schema.applicability_note


def test_three_class_ordinal_task_retains_ordinal_and_multiclass_probability_metrics() -> None:
    metrics = classification_metrics(
        [2, 2, 3, 4],
        [2, 4, 3, 4],
        np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.2, 0.7],
                [0.1, 0.8, 0.1],
                [0.1, 0.2, 0.7],
            ]
        ),
        labels=[2, 3, 4],
        task_type=ORDINAL_MULTICLASS_PERFORMANCE,
    )

    assert metrics["severe_error_rate"] == pytest.approx(0.25)
    assert metrics["ordinal_mae"] is not None
    assert metrics["quadratic_weighted_kappa"] is not None
    assert metrics["multiclass_brier"] is not None
    assert metrics["binary_brier"] is None


def test_unknown_task_type_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unknown task type"):
        get_task_schema("generic_classification")


def test_metric_registry_is_complete_machine_readable_and_deterministic() -> None:
    records = metric_schema_records()
    task_names = {
        ORDINAL_MULTICLASS_PERFORMANCE,
        RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS,
        NOMINAL_MULTICLASS_PROXY_DIAGNOSTIC,
        BINARY_ATTRITION_TRANSFER,
        BINARY_TURNOVER_TRANSFER,
    }

    assert set(METRIC_DEFINITIONS) == KNOWN_METRICS
    assert len(records) == len(task_names) * len(KNOWN_METRICS)
    assert {(row["task_type"], row["metric"]) for row in records} == {
        (task, metric) for task in task_names for metric in KNOWN_METRICS
    }
    assert len(metric_schema_hash()) == 64
    assert metric_schema_hash() == metric_schema_hash()
    macro = next(
        row
        for row in records
        if row["task_type"] == ORDINAL_MULTICLASS_PERFORMANCE and row["metric"] == "macro_f1"
    )
    assert macro["selection_role"] == "primary"
    assert macro["aggregation"] == "all_exactly_once_out_of_fold_samples"
    assert macro["uncertainty_resamples"] == 5000
    assert macro["better_direction"] == "higher"
    severe_binary = next(
        row
        for row in records
        if row["task_type"] == BINARY_ATTRITION_TRANSFER
        and row["metric"] == "severe_error_rate"
    )
    assert severe_binary["applicable"] is False
    assert severe_binary["not_applicable_reason"]
