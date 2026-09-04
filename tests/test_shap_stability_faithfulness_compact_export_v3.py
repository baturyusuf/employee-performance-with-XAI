from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from src.governance.shap_stability_faithfulness_compact_export_v3 import (
    DEFAULT_OUTPUT,
    EXPECTED_EXPORT_FILES,
    V3ShapStabilityFaithfulnessCompactExportError,
    export_shap_stability_faithfulness_compact_v3,
    validate_shap_stability_faithfulness_compact_v3,
)
from src.governance.shap_stability_faithfulness_run_validator_v3 import (
    DEFAULT_SHAP_STABILITY_FAITHFULNESS_RUN,
)


pytestmark = pytest.mark.skipif(
    not DEFAULT_SHAP_STABILITY_FAITHFULNESS_RUN.is_dir(),
    reason="ignored local complete Phase 2A run is unavailable",
)


def test_temporary_phase2a_export_is_safe_and_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    receipt = export_shap_stability_faithfulness_compact_v3(output_dir=first)
    export_shap_stability_faithfulness_compact_v3(output_dir=second)
    assert receipt["status"] == "passed"
    assert receipt["file_count"] == 8
    assert receipt["row_counts"] == {
        "stability_summary.csv": 9,
        "faithfulness_summary.csv": 63,
        "faithfulness_contrasts.csv": 6,
        "deletion_auc_summary.csv": 21,
    }
    assert set(path.name for path in first.iterdir()) == EXPECTED_EXPORT_FILES
    assert all(
        (first / name).read_bytes() == (second / name).read_bytes()
        for name in EXPECTED_EXPORT_FILES
    )
    assert receipt["per_sample_rows_included"] is False


def test_phase2a_compact_validator_rejects_tampering(tmp_path: Path) -> None:
    package = tmp_path / "package"
    export_shap_stability_faithfulness_compact_v3(output_dir=package)
    with (package / "README.md").open("ab") as stream:
        stream.write(b"tampered\n")
    with pytest.raises(
        V3ShapStabilityFaithfulnessCompactExportError,
        match="Compact size drifted",
    ):
        validate_shap_stability_faithfulness_compact_v3(package)


@pytest.mark.skipif(
    not DEFAULT_OUTPUT.is_dir(), reason="tracked compact Phase 2A package is absent"
)
def test_tracked_phase2a_package_validates() -> None:
    receipt = validate_shap_stability_faithfulness_compact_v3()
    assert receipt["status"] == "passed"
    assert receipt["per_sample_rows_included"] is False


@pytest.mark.skipif(
    not DEFAULT_OUTPUT.is_dir(), reason="tracked compact Phase 2A package is absent"
)
def test_tracked_phase2a_package_is_byte_preserved() -> None:
    completed = subprocess.run(
        ["git", "check-attr", "text", "--", str(DEFAULT_OUTPUT / "stability_summary.csv")],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.stdout.rstrip().endswith(": text: unset")


@pytest.mark.skipif(
    not DEFAULT_OUTPUT.is_dir(), reason="tracked compact Phase 2A package is absent"
)
def test_tracked_phase2a_package_has_closed_world(tmp_path: Path) -> None:
    copied = tmp_path / "package"
    shutil.copytree(DEFAULT_OUTPUT, copied)
    (copied / "faithfulness_sample_results.csv").write_text(
        "forbidden\n", encoding="utf-8"
    )
    with pytest.raises(
        V3ShapStabilityFaithfulnessCompactExportError,
        match="closed-world inventory",
    ):
        validate_shap_stability_faithfulness_compact_v3(copied)
