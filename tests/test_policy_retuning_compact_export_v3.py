from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from src.governance.policy_retuning_compact_export_v3 import (
    DEFAULT_OUTPUT,
    EXPECTED_EXPORT_FILES,
    V3PolicyRetuningCompactExportError,
    export_policy_retuning_compact_v3,
    validate_policy_retuning_compact_v3,
)
from src.governance.policy_retuning_run_validator_v3 import DEFAULT_POLICY_RETUNING_RUN


pytestmark = pytest.mark.skipif(
    not DEFAULT_POLICY_RETUNING_RUN.is_dir(),
    reason="ignored local complete Phase 1D run is unavailable",
)


def test_temporary_policy_retuning_export_is_safe_and_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    receipt = export_policy_retuning_compact_v3(output_dir=first)
    export_policy_retuning_compact_v3(output_dir=second)
    assert receipt["status"] == "passed"
    assert receipt["file_count"] == 7
    assert receipt["row_counts"] == {
        "aggregate_metrics.csv": 192,
        "metric_comparison.csv": 96,
        "headline_policy_comparison.csv": 6,
        "selected_candidate_frequency.csv": 21,
    }
    assert set(path.name for path in first.iterdir()) == EXPECTED_EXPORT_FILES
    assert all((first / name).read_bytes() == (second / name).read_bytes() for name in EXPECTED_EXPORT_FILES)
    assert receipt["employee_level_rows_included"] is False


def test_compact_validator_rejects_tampering(tmp_path: Path) -> None:
    package = tmp_path / "package"
    export_policy_retuning_compact_v3(output_dir=package)
    with (package / "README.md").open("ab") as stream:
        stream.write(b"tampered\n")
    with pytest.raises(V3PolicyRetuningCompactExportError, match="Compact size drifted"):
        validate_policy_retuning_compact_v3(package)


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="tracked compact Phase 1D package is absent")
def test_tracked_policy_retuning_package_validates() -> None:
    receipt = validate_policy_retuning_compact_v3()
    assert receipt["status"] == "passed"
    assert receipt["employee_level_rows_included"] is False


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="tracked compact Phase 1D package is absent")
def test_tracked_policy_retuning_package_is_byte_preserved() -> None:
    completed = subprocess.run(
        ["git", "check-attr", "text", "--", str(DEFAULT_OUTPUT / "aggregate_metrics.csv")],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.stdout.rstrip().endswith(": text: unset")


@pytest.mark.skipif(not DEFAULT_OUTPUT.is_dir(), reason="tracked compact Phase 1D package is absent")
def test_tracked_policy_retuning_package_has_closed_world(tmp_path: Path) -> None:
    copied = tmp_path / "package"
    shutil.copytree(DEFAULT_OUTPUT, copied)
    (copied / "fixed_oof_predictions.csv").write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(V3PolicyRetuningCompactExportError, match="closed-world inventory"):
        validate_policy_retuning_compact_v3(copied)
