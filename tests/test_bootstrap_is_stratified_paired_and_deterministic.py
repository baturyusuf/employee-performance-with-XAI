from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.oof_bootstrap import BootstrapProtocol, generate_stratified_resample_indices


def _base_samples() -> pd.DataFrame:
    rows = []
    sample_index = 0
    for outer_fold in (1, 2):
        for label in (2, 3, 4):
            for _ in range(3):
                rows.append(
                    {
                        "sample_index": sample_index,
                        "outer_fold": outer_fold,
                        "y_true": label,
                    }
                )
                sample_index += 1
    return pd.DataFrame(rows)


def test_stratified_resamples_preserve_joint_fold_target_counts() -> None:
    base = _base_samples()
    protocol = BootstrapProtocol(n_resamples=40, seed=17)
    plan = generate_stratified_resample_indices(base, protocol)
    ordered = base.sort_values("sample_index").reset_index(drop=True)
    expected = ordered.groupby(["outer_fold", "y_true"]).size().sort_index()

    assert plan.indices.shape == (40, len(base))
    assert plan.indices.flags.writeable is False
    for positions in plan.indices:
        observed = ordered.iloc[positions].groupby(["outer_fold", "y_true"]).size().sort_index()
        pd.testing.assert_series_equal(observed, expected)


def test_resample_hash_is_seed_deterministic_and_shuffled_input_invariant() -> None:
    base = _base_samples()
    protocol = BootstrapProtocol(n_resamples=32, seed=91)
    first = generate_stratified_resample_indices(base, protocol)
    shuffled = generate_stratified_resample_indices(
        base.sample(frac=1.0, random_state=123).reset_index(drop=True),
        protocol,
    )
    repeated = generate_stratified_resample_indices(base, protocol)
    different_seed = generate_stratified_resample_indices(
        base,
        BootstrapProtocol(n_resamples=32, seed=92),
    )

    assert first.sorted_sample_ids == shuffled.sorted_sample_ids == repeated.sorted_sample_ids
    assert np.array_equal(first.indices, shuffled.indices)
    assert np.array_equal(first.indices, repeated.indices)
    assert first.resample_hash == shuffled.resample_hash == repeated.resample_hash
    assert not np.array_equal(first.indices, different_seed.indices)
    assert first.resample_hash != different_seed.resample_hash


def test_protocol_defaults_are_the_predeclared_manuscript_contract() -> None:
    protocol = BootstrapProtocol()
    assert protocol.n_resamples == 5000
    assert protocol.confidence_level == 0.95
    assert protocol.seed == 42
    assert protocol.strata_columns == ("outer_fold", "y_true")
    assert protocol.method == "paired_stratified_percentile"
    assert protocol.quantile_method == "linear"
