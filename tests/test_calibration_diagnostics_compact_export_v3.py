from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from src.governance.calibration_diagnostics_compact_export_v3 import (
    DEFAULT_OUTPUT,
    EXPECTED_EXPORT_FILES,
    V3CalibrationDiagnosticsCompactExportError,
    export_calibration_diagnostics_compact_v3,
    validate_calibration_diagnostics_compact_v3,
)
from src.governance.calibration_diagnostics_run_validator_v3 import (
    DEFAULT_CALIBRATION_DIAGNOSTICS_RUN,
)


pytestmark = pytest.mark.skipif(
    not DEFAULT_CALIBRATION_DIAGNOSTICS_RUN.is_dir(),
    reason="ignored local complete Phase 2B run is unavailable",
)


def test_temporary_phase2b_export_is_safe_and_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    receipt = export_calibration_diagnostics_compact_v3(output_dir=first)
    export_calibration_diagnostics_compact_v3(output_dir=second)
    assert receipt["status"] == "passed"
    assert receipt["file_count"] == 13
    assert receipt["row_counts"] == {
        "calibration_metric_summary.csv": 2,
        "classwise_calibration_metrics.csv": 6,
        "cumulative_calibration_metrics.csv": 4,
        "extended_reliability_bins.csv": 120,
        "method_comparison.csv": 6,
    }
    assert set(path.name for path in first.iterdir()) == EXPECTED_EXPORT_FILES
    assert all(
        (first / name).read_bytes() == (second / name).read_bytes()
        for name in EXPECTED_EXPORT_FILES
    )
    assert receipt["employee_level_rows_included"] is False


def test_phase2b_compact_validator_rejects_tampering(tmp_path: Path) -> None:
    package = tmp_path / "package"
    export_calibration_diagnostics_compact_v3(output_dir=package)
    with (package / "README.md").open("ab") as stream:
        stream.write(b"tampered\n")
    with pytest.raises(V3CalibrationDiagnosticsCompactExportError, match="size drifted"):
        validate_calibration_diagnostics_compact_v3(package)


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="tracked compact Phase 2B package is absent")
def test_tracked_phase2b_package_validates() -> None:
    receipt = validate_calibration_diagnostics_compact_v3()
    assert receipt["status"] == "passed"
    assert receipt["employee_level_rows_included"] is False


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="tracked compact Phase 2B package is absent")
def test_tracked_phase2b_package_is_byte_preserved() -> None:
    completed = subprocess.run(
        ["git", "check-attr", "text", "--", str(DEFAULT_OUTPUT / "calibration_metric_summary.csv")],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.stdout.rstrip().endswith(": text: unset")


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="tracked compact Phase 2B package is absent")
def test_tracked_phase2b_package_has_closed_world(tmp_path: Path) -> None:
    copied = tmp_path / "package"
    shutil.copytree(DEFAULT_OUTPUT, copied)
    (copied / "calibration_predictions.csv").write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(V3CalibrationDiagnosticsCompactExportError, match="closed-world inventory"):
        validate_calibration_diagnostics_compact_v3(copied)
