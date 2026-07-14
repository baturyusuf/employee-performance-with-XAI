from __future__ import annotations

import pandas as pd
import pytest

from src.experiments.shared_folds import (
    SharedFoldArtifacts,
    SharedFoldContractError,
    generate_inner_assignments,
    generate_shared_folds,
    validate_shared_folds,
)


def _artifacts():
    frame = pd.DataFrame(
        {
            "EmpNumber": [f"E{index:03d}" for index in range(90)],
            "feature": [index % 13 for index in range(90)],
            "PerformanceRating": [2 + index % 3 for index in range(90)],
        },
        index=range(90),
    )
    return generate_shared_folds(
        frame,
        target_column="PerformanceRating",
        id_column="EmpNumber",
        run_id="inner-isolation-test",
        config_hash="a" * 64,
        scientific_input_hash="b" * 64,
        dataset_key="inx_primary",
        dataset_sha256="c" * 64,
        outer_splits=10,
        inner_splits=5,
        seed=73,
        inner_seed=74,
    )


def test_every_inner_search_population_is_exactly_the_outer_training_partition() -> None:
    artifacts = _artifacts()
    outer = artifacts.outer_assignments
    inner = artifacts.inner_assignments
    all_samples = set(outer["sample_index"])

    for outer_fold in range(1, 11):
        outer_test = set(outer.loc[outer["outer_fold"] == outer_fold, "sample_index"])
        scoped = inner[inner["outer_fold"] == outer_fold]
        inner_population = set(scoped["sample_index"])
        assert inner_population == all_samples - outer_test
        assert inner_population.isdisjoint(outer_test)
        assert scoped["sample_index"].is_unique
        assert set(scoped["inner_fold"]) == {1, 2, 3, 4, 5}


def test_outer_test_target_changes_cannot_change_that_outer_folds_inner_assignment() -> None:
    artifacts = _artifacts()
    outer = artifacts.outer_assignments.copy()
    baseline = generate_inner_assignments(
        outer,
        inner_splits=5,
        inner_seed=74,
        fold_contract_hash=str(artifacts.contract["fold_contract_hash"]),
    )
    held_out = outer["outer_fold"] == 1
    outer.loc[held_out, "y_true"] = outer.loc[held_out, "y_true"].map({2: 3, 3: 4, 4: 2})
    changed = generate_inner_assignments(
        outer,
        inner_splits=5,
        inner_seed=74,
        fold_contract_hash=str(artifacts.contract["fold_contract_hash"]),
    )

    baseline_fold = baseline[baseline["outer_fold"] == 1].reset_index(drop=True)
    changed_fold = changed[changed["outer_fold"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline_fold, changed_fold)


def test_validation_rejects_outer_test_sample_injected_into_inner_search() -> None:
    artifacts = _artifacts()
    outer = artifacts.outer_assignments
    inner = artifacts.inner_assignments.copy()
    outer_test_row = outer[outer["outer_fold"] == 1].iloc[0]
    injected = inner[inner["outer_fold"] == 1].iloc[0].copy()
    injected["sample_index"] = outer_test_row["sample_index"]
    injected["sample_key_sha256"] = outer_test_row["sample_key_sha256"]
    injected["y_true"] = outer_test_row["y_true"]
    inner = pd.concat([inner, injected.to_frame().T], ignore_index=True)
    tampered = SharedFoldArtifacts(outer, inner, artifacts.contract)

    with pytest.raises(SharedFoldContractError, match="hash mismatch|strict outer-training"):
        validate_shared_folds(tampered)


def test_inner_seeds_are_deterministic_and_outer_fold_specific() -> None:
    artifacts = _artifacts()
    seed_by_outer = artifacts.inner_assignments.groupby("outer_fold")["inner_seed"].unique()

    assert seed_by_outer.map(len).eq(1).all()
    assert len({int(values[0]) for values in seed_by_outer}) == 10
