from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.governance.subgroup_manuscript_summary_v4 import (
    build_subgroup_manuscript_summary_v4,
    validate_subgroup_manuscript_summary_v4,
    write_subgroup_manuscript_summary_v4,
)


OUTPUT = Path("reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv")


def test_builder_extracts_complete_p3_threshold_30_grid() -> None:
    summary = build_subgroup_manuscript_summary_v4()
    assert len(summary) == 18
    assert summary["attribute"].nunique() == 6
    assert summary["metric"].nunique() == 3
    assert set(summary["policy_id"]) == {"P3"}
    assert set(summary["support_threshold"]) == {30}
    assert set(summary["gap_status"]) == {"estimable_descriptive_gap"}
    assert summary["source_gap_row"].nunique() == 18
    assert summary["source_interval_row"].nunique() == 18


def test_department_rows_are_not_suppressed() -> None:
    summary = build_subgroup_manuscript_summary_v4()
    department = summary.loc[summary["attribute"] == "Department"].set_index("metric")
    assert department.loc["macro_f1", "gap_max_minus_min"] == pytest.approx(0.21790672381805382, abs=1e-15)
    assert department.loc["quadratic_weighted_kappa", "gap_max_minus_min"] == pytest.approx(0.43882398286949276, abs=1e-15)
    assert department.loc["ordinal_mae", "gap_max_minus_min"] == pytest.approx(0.11928317033184468, abs=1e-15)
    assert department.loc["quadratic_weighted_kappa", "eligible_group_count"] == 3
    assert department.loc["quadratic_weighted_kappa", "declared_group_count"] == 6


def test_tracked_summary_replays_sources_exactly() -> None:
    receipt = validate_subgroup_manuscript_summary_v4(OUTPUT)
    assert receipt["status"] == "passed"
    observed = pd.read_csv(OUTPUT)
    assert observed["model_training_variability_included"].eq(False).all()


def test_writer_is_deterministic(tmp_path: Path) -> None:
    first = write_subgroup_manuscript_summary_v4(tmp_path / "first.csv")
    second = write_subgroup_manuscript_summary_v4(tmp_path / "second.csv")
    assert first.read_bytes() == second.read_bytes()
