from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.governance.hrdataset_sensitivity_compact_export_v3 import (
    CSV_ROW_COUNTS,
    DEFAULT_OUTPUT,
    EXPECTED_EXPORT_FILES,
    HRDatasetSensitivityCompactExportV3Error,
    export_hrdataset_sensitivity_compact_v3,
    validate_hrdataset_sensitivity_compact_v3,
)
from src.governance.hrdataset_sensitivity_run_validator_v3 import (
    DEFAULT_HRDATASET_SENSITIVITY_RUN,
)


@pytest.fixture(scope="module")
def temporary_export(tmp_path_factory):
    if not DEFAULT_HRDATASET_SENSITIVITY_RUN.is_dir():
        pytest.skip("local row-level Phase 3A run is intentionally ignored")
    output = tmp_path_factory.mktemp("phase3a_compact") / "package"
    receipt = export_hrdataset_sensitivity_compact_v3(output_dir=output)
    return output, receipt


def test_temporary_compact_export_is_complete_and_safe(temporary_export) -> None:
    output, receipt = temporary_export
    assert receipt["status"] == "passed"
    assert receipt["file_count"] == 14
    assert receipt["row_counts"] == CSV_ROW_COUNTS
    assert receipt["employee_level_rows_included"] is False
    assert receipt["calibration_training_rows_included"] is False
    assert receipt["fold_level_rows_included"] is False
    assert receipt["target_equivalence_claim_allowed"] is False
    assert receipt["external_validation_claim_allowed"] is False
    assert receipt["locked_model_transport_claim_allowed"] is False
    assert {path.name for path in output.iterdir()} == EXPECTED_EXPORT_FILES


def test_two_compact_exports_are_byte_deterministic(temporary_export, tmp_path: Path) -> None:
    first, first_receipt = temporary_export
    second = tmp_path / "second"
    second_receipt = export_hrdataset_sensitivity_compact_v3(output_dir=second)
    assert second_receipt["manifest_sha256"] == first_receipt["manifest_sha256"]
    for name in EXPECTED_EXPORT_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_compact_validator_rejects_readme_tampering(temporary_export, tmp_path: Path) -> None:
    source, _ = temporary_export
    copied = tmp_path / "tampered"
    shutil.copytree(source, copied)
    path = copied / "README.md"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(HRDatasetSensitivityCompactExportV3Error, match="(size|hash) drifted"):
        validate_hrdataset_sensitivity_compact_v3(copied)


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="tracked compact Phase 3A package is not published yet")
def test_tracked_compact_package_validates() -> None:
    receipt = validate_hrdataset_sensitivity_compact_v3(DEFAULT_OUTPUT)
    assert receipt["status"] == "passed"
