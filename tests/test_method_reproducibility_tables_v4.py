from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.governance.method_reproducibility_tables_v4 import (
    EXPECTED_RETAINED_COUNTS,
    validate_method_reproducibility_tables_v4,
    write_method_reproducibility_tables_v4,
)


OUTPUT = Path("reports/research_log/major_revision_round2/METHOD_REPRODUCIBILITY_TABLES")


def test_tracked_method_package_is_exactly_reproducible() -> None:
    receipt = validate_method_reproducibility_tables_v4(OUTPUT)
    assert receipt["status"] == "passed"
    assert receipt["file_count"] == 9
    assert receipt["feature_count"] == 28
    assert receipt["trained_model_count"] == 6
    assert receipt["design_row_count"] == 8


def test_table_s1_preserves_all_policy_decisions_and_counts() -> None:
    table = pd.read_csv(OUTPUT / "TABLE_S1_FEATURE_AVAILABILITY.csv")
    assert table["feature_name"].is_unique
    assert len(table) == 28
    assert {policy: int(table[policy].eq("Included").sum()) for policy in EXPECTED_RETAINED_COUNTS} == EXPECTED_RETAINED_COUNTS
    assert table["timestamp_caveat"].str.contains("No row-level feature timestamp").all()
    assert table.loc[table["feature_name"] == "PerformanceRating", list(EXPECTED_RETAINED_COUNTS)].eq("Excluded").all(axis=None)


def test_table_s2_contains_complete_six_model_registries() -> None:
    table = pd.read_csv(OUTPUT / "TABLE_S2_PREPROCESSING_HYPERPARAMETER_SEARCH.csv")
    models = table.loc[table["section"] == "trained_model"]
    assert len(models) == 6
    assert models["candidate_count"].astype(int).tolist() == [6, 8, 8, 8, 6, 8]
    for cell in models["fixed_parameters"]:
        assert isinstance(json.loads(cell), dict)
    for cell, count in zip(models["candidate_grid"], models["candidate_count"]):
        assert len(json.loads(cell)) == int(count)
    regimes = table.loc[table["section"] == "selection_regime"].set_index("item_id")
    assert regimes.loc["macro_f1", "selection_primary"] == "macro_f1"
    assert regimes.loc["qwk", "selection_primary"] == "quadratic_weighted_kappa"
    assert regimes["tie_tolerance"].eq(0.001).all()


def test_table_s3_preserves_exact_seed_schedules_and_boundaries() -> None:
    table = pd.read_csv(OUTPUT / "TABLE_S3_CV_SEED_CONTRACT.csv").set_index("analysis")
    assert table.loc["Canonical INX six-model benchmark", "seed_schedule"] == "outer=42;inner=43;model=44"
    assert table.loc["INX repeated nested-CV sensitivity", "seed_schedule"].startswith("r1:1042/1043/1044")
    assert table.loc["HRDataset target-formulation/CV sensitivity", "seed_schedule"].startswith("r1:4201/4301/4401/4501/4601")
    assert "fit-free restriction to 309" in table.loc["HR target-alias matched-sample sensitivity", "reuse_or_refit_boundary"]
    assert table["outer_test_role"].str.contains("evaluation_only").all()


def test_methods_notes_are_self_contained_for_ordinal_models() -> None:
    notes = (OUTPUT / "METHODS_NOTES.md").read_text(encoding="utf-8")
    for phrase in (
        "proportional-odds",
        "P(Y > 2)",
        "P(Y > 3)",
        "pool-adjacent-violators (PAVA)",
        "one-hot encoding with unknown categories ignored",
        "inclusive 0.001",
        "does not add an MAE- or RPS-selected regime",
    ):
        assert phrase in notes


def test_method_writer_is_deterministic(tmp_path: Path) -> None:
    first = write_method_reproducibility_tables_v4(tmp_path / "first")
    second = write_method_reproducibility_tables_v4(tmp_path / "second")
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
