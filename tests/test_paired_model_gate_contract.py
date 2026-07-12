from __future__ import annotations

import pandas as pd
import pytest

from src.models.oof_bootstrap import (
    BootstrapProtocol,
    ComparisonSpec,
    OOFBootstrapError,
    compute_paired_oof_bootstrap,
)


LABELS = [2, 3, 4]


def _system(system_id: str, *, degraded: bool) -> pd.DataFrame:
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


def _better_baseline_predictions() -> pd.DataFrame:
    return pd.concat(
        [_system("baseline", degraded=False), _system("xgboost", degraded=True)],
        ignore_index=True,
    )


def test_primary_gate_requires_an_explicit_metric() -> None:
    with pytest.raises(OOFBootstrapError, match="explicitly supplied primary_metric"):
        compute_paired_oof_bootstrap(
            _better_baseline_predictions(),
            labels=LABELS,
            task_type="ordinal_multiclass_performance",
            metrics=["macro_f1"],
            comparisons=[ComparisonSpec("baseline_vs_xgboost", "baseline", "xgboost", primary_gate=True)],
            protocol=BootstrapProtocol(n_resamples=20, seed=1),
        )


def test_gate_triggers_only_for_the_explicit_primary_metric_with_ci_strictly_above_zero() -> None:
    result = compute_paired_oof_bootstrap(
        _better_baseline_predictions(),
        labels=LABELS,
        task_type="ordinal_multiclass_performance",
        metrics=["macro_f1", "ordinal_mae", "severe_error_rate"],
        comparisons=[ComparisonSpec("baseline_vs_xgboost", "baseline", "xgboost", primary_gate=True)],
        primary_metric="macro_f1",
        protocol=BootstrapProtocol(n_resamples=30, seed=8),
    )
    rows = result.paired_differences.set_index("metric")
    assert rows.loc["macro_f1", "improvement_ci_low"] > 0
    assert bool(rows.loc["macro_f1", "gate_eligible"]) is True
    assert bool(rows.loc["macro_f1", "gate_triggered"]) is True
    for secondary in ("ordinal_mae", "severe_error_rate"):
        assert rows.loc[secondary, "improvement_ci_low"] > 0
        assert bool(rows.loc[secondary, "gate_eligible"]) is False
        assert bool(rows.loc[secondary, "gate_triggered"]) is False


def test_lower_is_better_primary_gate_uses_improvement_orientation() -> None:
    result = compute_paired_oof_bootstrap(
        _better_baseline_predictions(),
        labels=LABELS,
        task_type="ordinal_multiclass_performance",
        metrics=["severe_error_rate"],
        comparisons=[ComparisonSpec("baseline_vs_xgboost", "baseline", "xgboost", primary_gate=True)],
        primary_metric="severe_error_rate",
        protocol=BootstrapProtocol(n_resamples=24, seed=4),
    )
    row = result.paired_differences.iloc[0]
    assert row["better_direction"] == "lower"
    assert row["raw_difference_a_minus_b"] < 0
    assert row["improvement_ci_low"] > 0
    assert bool(row["gate_triggered"]) is True


def test_ci_touching_zero_does_not_trigger_gate() -> None:
    predictions = pd.concat(
        [_system("baseline", degraded=False), _system("xgboost", degraded=False)],
        ignore_index=True,
    )
    result = compute_paired_oof_bootstrap(
        predictions,
        labels=LABELS,
        task_type="ordinal_multiclass_performance",
        metrics=["macro_f1"],
        comparisons=[ComparisonSpec("baseline_vs_xgboost", "baseline", "xgboost", primary_gate=True)],
        primary_metric="macro_f1",
        protocol=BootstrapProtocol(n_resamples=20, seed=6),
    )
    row = result.paired_differences.iloc[0]
    assert row["improvement_ci_low"] == pytest.approx(0.0)
    assert bool(row["gate_eligible"]) is True
    assert bool(row["gate_triggered"]) is False


def test_primary_metric_must_be_part_of_the_bootstrap_metric_set() -> None:
    with pytest.raises(OOFBootstrapError, match="included in metrics"):
        compute_paired_oof_bootstrap(
            _better_baseline_predictions(),
            labels=LABELS,
            task_type="ordinal_multiclass_performance",
            metrics=["macro_f1"],
            comparisons=[ComparisonSpec("baseline_vs_xgboost", "baseline", "xgboost", primary_gate=True)],
            primary_metric="quadratic_weighted_kappa",
            protocol=BootstrapProtocol(n_resamples=20, seed=2),
        )
