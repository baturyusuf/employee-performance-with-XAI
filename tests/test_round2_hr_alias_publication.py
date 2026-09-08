from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.governance.hr_target_alias_compact_export_v4 import (
    validate_hr_target_alias_compact_v4,
)


ROOT = Path("reports/research_log/major_revision_round2")
PACKAGE = ROOT / "hr_target_alias_sensitivity"


def test_round2_hr_alias_compact_package_is_closed_and_valid() -> None:
    receipt = validate_hr_target_alias_compact_v4(PACKAGE)
    assert receipt["file_count"] == 9
    assert receipt["row_identities_published"] is False


def test_hr_mapping_report_replays_matched_and_sample_removal_effects() -> None:
    report = (ROOT / "HR_MAPPING_SENSITIVITY_REPORT.md").read_text(encoding="utf-8")
    deltas = pd.read_csv(PACKAGE / "comparison_deltas.csv")
    matched = deltas.loc[
        (deltas["comparison_id"] == "matched_refit_effect")
        & (deltas["metric"].isin(["macro_f1", "quadratic_weighted_kappa", "ranked_probability_score"]))
    ]
    removal = deltas.loc[
        (deltas["comparison_id"] == "sample_removal_effect")
        & (deltas["metric"].isin(["macro_f1", "quadratic_weighted_kappa", "ranked_probability_score"]))
    ]
    for value in pd.concat([matched, removal])["right_minus_left"]:
        assert f"{value:+.6f}" in report
    assert "identical 309 rows" in report
    assert "measures only removal of two evaluation rows" in report
    assert "must not be attributed to retraining" in report


def test_hr_mapping_report_preserves_mapping_and_cv_boundaries() -> None:
    report = (ROOT / "HR_MAPPING_SENSITIVITY_REPORT.md").read_text(encoding="utf-8")
    assert "31/243/37" in report
    assert "13/18/243/37" in report
    assert "11 of 14" in report
    assert "Three estimates fell outside" in report
    assert "different target estimands" in report
    assert "not an equivalence test" in report
