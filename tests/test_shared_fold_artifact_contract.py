from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from src.experiments.shared_folds import (
    INNER_COLUMNS,
    OUTER_COLUMNS,
    SharedFoldArtifacts,
    SharedFoldContractError,
    generate_shared_folds,
    read_shared_folds,
    validate_shared_folds,
    write_shared_folds,
)


def _frame() -> pd.DataFrame:
    rows = []
    for offset in range(90):
        rows.append(
            {
                "EmpNumber": f"E{offset:04d}",
                "signal": float(offset % 11),
                "category": f"group_{offset % 4}",
                "PerformanceRating": 2 + (offset % 3),
            }
        )
    return pd.DataFrame(rows, index=pd.Index(range(1000, 1090), name="source_position"))


def _generate(frame: pd.DataFrame | None = None):
    return generate_shared_folds(
        _frame() if frame is None else frame,
        target_column="PerformanceRating",
        id_column="EmpNumber",
        run_id="shared-fold-test",
        config_hash="a" * 64,
        scientific_input_hash="b" * 64,
        dataset_key="inx_primary",
        dataset_sha256="c" * 64,
        outer_splits=10,
        inner_splits=3,
        seed=42,
        inner_seed=43,
    )


def test_generated_artifacts_have_complete_schema_and_nested_membership() -> None:
    artifacts = _generate()
    outer = artifacts.outer_assignments
    inner = artifacts.inner_assignments
    contract = artifacts.contract

    assert tuple(outer.columns) == OUTER_COLUMNS
    assert tuple(inner.columns) == INNER_COLUMNS
    assert len(outer) == 90
    assert outer["sample_index"].is_unique
    assert outer["sample_key_sha256"].is_unique
    assert set(outer["outer_fold"]) == set(range(1, 11))
    assert len(inner) == 90 * 9
    assert inner.groupby("outer_fold")["sample_index"].nunique().eq(81).all()
    assert inner.groupby(["outer_fold", "sample_index"]).size().eq(1).all()
    assert set(inner["inner_fold"]) == {1, 2, 3}
    assert contract["identifier_used_for_model"] is False
    assert contract["target_used_for_model"] is False
    assert contract["model_feature_exclusions"] == ["EmpNumber", "PerformanceRating"]
    assert contract["target_distribution"] == {"2": 30, "3": 30, "4": 30}
    assert contract["outer_seed"] == 42
    assert contract["inner_seed"] == 43
    assert all(contract["invariants"].values())
    validate_shared_folds(artifacts, source_frame=_frame())


def test_generation_is_deterministic_and_in_memory_row_order_invariant() -> None:
    frame = _frame()
    first = _generate(frame)
    shuffled = frame.sample(frac=1.0, random_state=901)
    second = _generate(shuffled)

    pd.testing.assert_frame_equal(first.outer_assignments, second.outer_assignments)
    pd.testing.assert_frame_equal(first.inner_assignments, second.inner_assignments)
    assert first.contract == second.contract


def test_write_read_round_trip_verifies_exact_file_hashes(tmp_path: Path) -> None:
    generated = _generate()
    paths = write_shared_folds(generated, tmp_path / "shared_folds")
    loaded = read_shared_folds(tmp_path / "shared_folds")

    assert set(paths) == {"outer_assignments", "inner_assignments", "contract"}
    pd.testing.assert_frame_equal(generated.outer_assignments, loaded.outer_assignments)
    pd.testing.assert_frame_equal(generated.inner_assignments, loaded.inner_assignments)
    assert generated.contract == loaded.contract


def test_read_rejects_tampered_assignment_bytes(tmp_path: Path) -> None:
    paths = write_shared_folds(_generate(), tmp_path / "shared_folds")
    outer = pd.read_csv(paths["outer_assignments"])
    outer.loc[0, "outer_fold"] = 10 if int(outer.loc[0, "outer_fold"]) != 10 else 9
    outer.to_csv(paths["outer_assignments"], index=False)

    with pytest.raises(SharedFoldContractError, match="hash mismatch|not class-stratified"):
        read_shared_folds(tmp_path / "shared_folds")


@pytest.mark.parametrize("defect", ["duplicate_identifier", "duplicate_sample_position", "insufficient_class"])
def test_generation_fails_closed_on_unstable_or_unstratifiable_samples(defect: str) -> None:
    frame = _frame()
    if defect == "duplicate_identifier":
        frame.iloc[1, frame.columns.get_loc("EmpNumber")] = frame.iloc[0]["EmpNumber"]
    elif defect == "duplicate_sample_position":
        frame.index = [1000, 1000, *range(1002, 1090)]
    else:
        frame = frame[frame["PerformanceRating"] != 4].copy()
        rare = frame.iloc[[0]].copy()
        rare["EmpNumber"] = "RARE"
        rare["PerformanceRating"] = 4
        rare.index = [9999]
        frame = pd.concat([frame, rare])

    with pytest.raises(SharedFoldContractError):
        _generate(frame)


@pytest.mark.parametrize("field", ["EmpNumber", "PerformanceRating"])
def test_validation_binds_source_identifier_and_target(field: str) -> None:
    frame = _frame()
    artifacts = _generate(frame)
    changed = frame.copy()
    if field == "EmpNumber":
        changed.loc[1000, field] = "DIFFERENT-ID"
    else:
        changed.loc[1000, field] = 4 if int(changed.loc[1000, field]) != 4 else 3

    with pytest.raises(SharedFoldContractError, match="identifier|target"):
        validate_shared_folds(artifacts, source_frame=changed)


def test_validation_rejects_contract_or_assignment_identity_tampering() -> None:
    artifacts = _generate()
    changed_contract = copy.deepcopy(dict(artifacts.contract))
    changed_contract["dataset_sha256"] = "d" * 64
    tampered = SharedFoldArtifacts(
        artifacts.outer_assignments.copy(),
        artifacts.inner_assignments.copy(),
        changed_contract,
    )

    with pytest.raises(SharedFoldContractError, match="identity mismatch|contract hash"):
        validate_shared_folds(tampered)
