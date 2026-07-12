from __future__ import annotations

import math

import pandas as pd
import pytest

from src.models.oof_bootstrap import (
    BootstrapProtocol,
    METRIC_DEFINITIONS,
    OOFBootstrapError,
    compute_paired_oof_bootstrap,
    metric_definition,
    validate_aligned_oof_predictions,
    validate_metric_value,
)


LABELS = [2, 3, 4]
METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "quadratic_weighted_kappa",
    "ordinal_mae",
    "severe_error_rate",
    "nll_log_loss",
    "multiclass_brier",
    "ece_confidence",
]


def _perfect_system(system_id: str = "perfect") -> pd.DataFrame:
    rows = []
    sample_index = 0
    for outer_fold in (1, 2):
        for label in LABELS:
            for _ in range(2):
                rows.append(
                    {
                        "system_id": system_id,
                        "sample_index": sample_index,
                        "outer_fold": outer_fold,
                        "y_true": label,
                        "y_pred": label,
                        "prob_class_2": float(label == 2),
                        "prob_class_3": float(label == 3),
                        "prob_class_4": float(label == 4),
                    }
                )
                sample_index += 1
    return pd.DataFrame(rows)


def test_metric_registry_has_explicit_scientific_directions_and_domains() -> None:
    assert metric_definition("macro_f1").better_direction == "higher"
    assert (metric_definition("macro_f1").lower_bound, metric_definition("macro_f1").upper_bound) == (0.0, 1.0)
    assert metric_definition("quadratic_weighted_kappa").lower_bound == -1.0
    assert metric_definition("quadratic_weighted_kappa").upper_bound == 1.0
    assert metric_definition("ordinal_mae").better_direction == "lower"
    assert metric_definition("ordinal_mae").upper_bound == 2.0
    assert metric_definition("severe_error_rate").better_direction == "lower"
    assert metric_definition("severe_error_rate").upper_bound == 1.0
    assert metric_definition("nll_log_loss").better_direction == "lower"
    assert math.isinf(metric_definition("nll_log_loss").upper_bound)
    assert metric_definition("multiclass_brier").better_direction == "lower"
    assert metric_definition("multiclass_brier").upper_bound == 2.0
    with pytest.raises(TypeError):
        METRIC_DEFINITIONS["macro_f1"] = metric_definition("ordinal_mae")  # type: ignore[index]


def test_perfect_oof_bootstrap_intervals_remain_in_registered_domains() -> None:
    result = compute_paired_oof_bootstrap(
        _perfect_system(),
        labels=LABELS,
        task_type="ordinal_multiclass_performance",
        metrics=METRICS,
        protocol=BootstrapProtocol(n_resamples=30, seed=7),
    )

    assert len(result.metric_intervals) == len(METRICS)
    assert set(result.metric_intervals["n_valid"]) == {30}
    for row in result.metric_intervals.itertuples(index=False):
        definition = metric_definition(row.metric)
        assert row.point_estimate >= definition.lower_bound
        assert row.ci_low >= definition.lower_bound
        if math.isfinite(definition.upper_bound):
            assert row.point_estimate <= definition.upper_bound
            assert row.ci_high <= definition.upper_bound
    by_metric = result.metric_intervals.set_index("metric")
    assert by_metric.loc["macro_f1", "point_estimate"] == pytest.approx(1.0)
    assert by_metric.loc["macro_f1", "ci_low"] == pytest.approx(1.0)
    assert by_metric.loc["ordinal_mae", "ci_high"] == pytest.approx(0.0)
    assert by_metric.loc["severe_error_rate", "ci_high"] == pytest.approx(0.0)
    assert by_metric.loc["multiclass_brier", "ci_high"] == pytest.approx(0.0)


def test_domain_validation_fails_instead_of_clipping_or_dropping() -> None:
    assert validate_metric_value("quadratic_weighted_kappa", -1.0) == -1.0
    assert validate_metric_value("ordinal_mae", 2.0) == 2.0
    assert validate_metric_value("multiclass_brier", 2.0) == 2.0
    with pytest.raises(OOFBootstrapError, match="upper bound"):
        validate_metric_value("macro_f1", 1.01)
    with pytest.raises(OOFBootstrapError, match="upper bound"):
        validate_metric_value("multiclass_brier", 2.01)
    with pytest.raises(OOFBootstrapError, match="non-finite"):
        validate_metric_value("nll_log_loss", float("nan"))


def test_oof_alignment_requires_exactly_once_coverage_and_valid_probabilities() -> None:
    first = _perfect_system("first")
    second = _perfect_system("second")
    valid = pd.concat([first, second], ignore_index=True)
    receipt = validate_aligned_oof_predictions(
        valid,
        labels=LABELS,
        task_type="ordinal_multiclass_performance",
        metrics=["macro_f1", "multiclass_brier"],
    )
    assert receipt["n_samples"] == len(first)
    assert receipt["n_systems"] == 2

    duplicate = pd.concat([valid, first.iloc[[0]]], ignore_index=True)
    with pytest.raises(OOFBootstrapError, match="duplicate OOF sample"):
        validate_aligned_oof_predictions(
            duplicate,
            labels=LABELS,
            task_type="ordinal_multiclass_performance",
            metrics=["macro_f1"],
        )

    missing = valid.drop(second.index[-1] + len(first)).reset_index(drop=True)
    with pytest.raises(OOFBootstrapError, match="coverage differs"):
        validate_aligned_oof_predictions(
            missing,
            labels=LABELS,
            task_type="ordinal_multiclass_performance",
            metrics=["macro_f1"],
        )

    truth_mismatch = valid.copy()
    truth_mismatch.loc[len(first), "y_true"] = 3
    with pytest.raises(OOFBootstrapError, match="y_true differs"):
        validate_aligned_oof_predictions(
            truth_mismatch,
            labels=LABELS,
            task_type="ordinal_multiclass_performance",
            metrics=["macro_f1"],
        )

    fractional_label = valid.copy()
    fractional_label["y_pred"] = fractional_label["y_pred"].astype(float)
    fractional_label.loc[0, "y_pred"] = 2.5
    with pytest.raises(OOFBootstrapError, match="non-integer labels"):
        validate_aligned_oof_predictions(
            fractional_label,
            labels=LABELS,
            task_type="ordinal_multiclass_performance",
            metrics=["macro_f1"],
        )

    invalid_probability = valid.copy()
    invalid_probability.loc[0, ["prob_class_2", "prob_class_3", "prob_class_4"]] = [0.8, 0.3, -0.1]
    with pytest.raises(OOFBootstrapError, match=r"outside \[0,1\]"):
        validate_aligned_oof_predictions(
            invalid_probability,
            labels=LABELS,
            task_type="ordinal_multiclass_performance",
            metrics=["multiclass_brier"],
        )


def test_protocol_rejects_one_resample_because_bootstrap_std_requires_two() -> None:
    with pytest.raises(OOFBootstrapError, match="at least two"):
        BootstrapProtocol(n_resamples=1)
