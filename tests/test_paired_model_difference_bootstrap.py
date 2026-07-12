from __future__ import annotations

import pandas as pd
import pytest

from src.models.oof_bootstrap import (
    BootstrapProtocol,
    ComparisonSpec,
    compute_paired_oof_bootstrap,
)


LABELS = [2, 3, 4]


def _system(system_id: str, *, degraded: bool = False) -> pd.DataFrame:
    rows = []
    sample_index = 0
    for outer_fold in (1, 2):
        for label in LABELS:
            for _ in range(3):
                predicted = ({2: 4, 3: 2, 4: 2}[label] if degraded else label)
                probabilities = {value: 0.1 for value in LABELS}
                probabilities[predicted] = 0.8
                if not degraded:
                    probabilities = {value: float(value == predicted) for value in LABELS}
                rows.append(
                    {
                        "system_id": system_id,
                        "sample_index": sample_index,
                        "outer_fold": outer_fold,
                        "y_true": label,
                        "y_pred": predicted,
                        **{f"prob_class_{value}": probabilities[value] for value in LABELS},
                    }
                )
                sample_index += 1
    return pd.DataFrame(rows)


def _predictions() -> pd.DataFrame:
    return pd.concat(
        [
            _system("reference"),
            _system("identical"),
            _system("degraded", degraded=True),
        ],
        ignore_index=True,
    )


def test_identical_systems_have_exact_zero_paired_difference_distribution() -> None:
    result = compute_paired_oof_bootstrap(
        _predictions(),
        labels=LABELS,
        task_type="ordinal_multiclass_performance",
        metrics=["macro_f1", "ordinal_mae", "severe_error_rate"],
        comparisons=[ComparisonSpec("identical_pair", "reference", "identical")],
        protocol=BootstrapProtocol(n_resamples=40, seed=13),
    )
    rows = result.paired_differences
    assert set(rows["raw_difference_a_minus_b"]) == {0.0}
    assert set(rows["raw_difference_ci_low"]) == {0.0}
    assert set(rows["raw_difference_ci_high"]) == {0.0}
    assert set(rows["improvement_oriented_difference"]) == {0.0}
    assert set(rows["improvement_ci_low"]) == {0.0}
    assert set(rows["improvement_ci_high"]) == {0.0}
    assert set(rows["n_valid"]) == {40}


def test_lower_is_better_metrics_are_oriented_without_changing_raw_difference() -> None:
    result = compute_paired_oof_bootstrap(
        _predictions(),
        labels=LABELS,
        task_type="ordinal_multiclass_performance",
        metrics=["macro_f1", "ordinal_mae", "severe_error_rate"],
        comparisons=[ComparisonSpec("reference_vs_degraded", "reference", "degraded")],
        protocol=BootstrapProtocol(n_resamples=40, seed=21),
    )
    rows = result.paired_differences.set_index("metric")
    assert rows.loc["macro_f1", "better_direction"] == "higher"
    assert rows.loc["macro_f1", "raw_difference_a_minus_b"] > 0
    assert rows.loc["macro_f1", "improvement_ci_low"] > 0

    for metric in ("ordinal_mae", "severe_error_rate"):
        assert rows.loc[metric, "better_direction"] == "lower"
        assert rows.loc[metric, "raw_difference_a_minus_b"] < 0
        assert rows.loc[metric, "improvement_oriented_difference"] > 0
        assert rows.loc[metric, "improvement_ci_low"] > 0


def test_full_paired_result_is_invariant_to_input_row_order() -> None:
    predictions = _predictions()
    kwargs = {
        "labels": LABELS,
        "task_type": "ordinal_multiclass_performance",
        "metrics": ["macro_f1", "ordinal_mae"],
        "comparisons": [ComparisonSpec("reference_vs_degraded", "reference", "degraded")],
        "protocol": BootstrapProtocol(n_resamples=30, seed=5),
    }
    first = compute_paired_oof_bootstrap(predictions, **kwargs)
    shuffled = compute_paired_oof_bootstrap(
        predictions.sample(frac=1.0, random_state=444).reset_index(drop=True),
        **kwargs,
    )

    assert first.resample_plan.resample_hash == shuffled.resample_plan.resample_hash
    pd.testing.assert_frame_equal(
        first.metric_intervals.sort_values(["system_id", "metric"]).reset_index(drop=True),
        shuffled.metric_intervals.sort_values(["system_id", "metric"]).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        first.paired_differences.sort_values(["comparison_id", "metric"]).reset_index(drop=True),
        shuffled.paired_differences.sort_values(["comparison_id", "metric"]).reset_index(drop=True),
    )
