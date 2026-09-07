from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.governance.data_quality_compact_export_v3 import (
    CSV_ROW_COUNTS,
    DEFAULT_OUTPUT,
    EXPECTED_EXPORT_FILES,
    DataQualityCompactExportV3Error,
    export_data_quality_compact_v3,
    validate_data_quality_compact_v3,
)
from src.governance.data_quality_run_validator_v3 import DEFAULT_DATA_QUALITY_RUN


@pytest.fixture(scope="module")
def temporary_export(tmp_path_factory):
    if not DEFAULT_DATA_QUALITY_RUN.is_dir():
        pytest.skip("local aggregate Phase 3B run is intentionally ignored")
    output = tmp_path_factory.mktemp("phase3b_compact") / "package"
    receipt = export_data_quality_compact_v3(output_dir=output)
    return output, receipt


def test_temporary_export_is_complete_aggregate_only_and_bounded(temporary_export) -> None:
    output, receipt = temporary_export
    assert receipt["status"] == "passed"
    assert receipt["file_count"] == 13
    assert receipt["row_counts"] == CSV_ROW_COUNTS
    assert receipt["raw_data_included"] is False
    assert receipt["row_level_values_or_identifiers_included"] is False
    assert receipt["source_rows_repaired"] is False
    assert receipt["supplementary_datasets_represented_as_core"] is False
    assert receipt["construct_validity_established"] is False
    assert receipt["external_generalization_established"] is False
    assert {path.name for path in output.iterdir()} == EXPECTED_EXPORT_FILES


def test_two_compact_exports_are_byte_deterministic(temporary_export, tmp_path: Path) -> None:
    first, first_receipt = temporary_export
    second = tmp_path / "second"
    second_receipt = export_data_quality_compact_v3(output_dir=second)
    assert second_receipt["manifest_sha256"] == first_receipt["manifest_sha256"]
    for name in EXPECTED_EXPORT_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_compact_validator_rejects_report_tampering(temporary_export, tmp_path: Path) -> None:
    source, _ = temporary_export
    copied = tmp_path / "tampered"
    shutil.copytree(source, copied)
    report = copied / "DATA_QUALITY_REPORT.md"
    report.write_bytes(report.read_bytes() + b"\n")
    with pytest.raises(DataQualityCompactExportV3Error, match="(size|hash) drifted"):
        validate_data_quality_compact_v3(copied)


def test_report_contains_required_construct_and_leakage_limits(temporary_export) -> None:
    output, _ = temporary_export
    report = (output / "DATA_QUALITY_REPORT.md").read_text(encoding="utf-8")
    assert "true employee capability" in report
    assert "objective productivity" in report
    assert "future potential" in report
    assert "does not prove that indirect proxy leakage is absent" in report
    assert "No source values were corrected" in report


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="tracked compact Phase 3B package is not published yet")
def test_tracked_compact_package_validates() -> None:
    receipt = validate_data_quality_compact_v3(DEFAULT_OUTPUT)
    assert receipt["status"] == "passed"
