from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.governance.calibration_diagnostics_run_validator_v3 import (
    DEFAULT_CALIBRATION_DIAGNOSTICS_RUN,
    V3CalibrationDiagnosticsRunValidationError,
    validate_calibration_diagnostics_run_v3,
)


pytestmark = pytest.mark.skipif(
    not DEFAULT_CALIBRATION_DIAGNOSTICS_RUN.is_dir(),
    reason="ignored local complete Phase 2B run is unavailable",
)


def test_complete_phase2b_run_passes_independent_validation() -> None:
    receipt = validate_calibration_diagnostics_run_v3()
    assert receipt["status"] == "passed"
    assert receipt["generation_commit"] == "21d1aecb6e61511e95aee498ab81c54fe6e5a6ab"
    assert receipt["file_count"] == 11
    assert receipt["reliability_bin_row_count"] == 120
    assert receipt["empty_reliability_bin_count"] == 24
    assert receipt["sigmoid_metrics"]["nll_log_loss"] < receipt["raw_metrics"]["nll_log_loss"]
    assert receipt["sigmoid_metrics"]["top_label_ece"] > receipt["raw_metrics"]["top_label_ece"]
    assert receipt["new_model_fit_calls"] == receipt["new_calibrator_fit_calls"] == 0
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0


def test_phase2b_validator_rejects_unexpected_file(tmp_path: Path) -> None:
    copied = tmp_path / DEFAULT_CALIBRATION_DIAGNOSTICS_RUN.parent.name
    copied = copied / DEFAULT_CALIBRATION_DIAGNOSTICS_RUN.name
    shutil.copytree(DEFAULT_CALIBRATION_DIAGNOSTICS_RUN, copied)
    (copied / "unexpected.txt").write_text("not allowed\n", encoding="utf-8")
    with pytest.raises(V3CalibrationDiagnosticsRunValidationError, match="closed-world inventory"):
        validate_calibration_diagnostics_run_v3(copied)


def test_phase2b_validator_rejects_metric_tampering(tmp_path: Path) -> None:
    copied = tmp_path / DEFAULT_CALIBRATION_DIAGNOSTICS_RUN.parent.name
    copied = copied / DEFAULT_CALIBRATION_DIAGNOSTICS_RUN.name
    shutil.copytree(DEFAULT_CALIBRATION_DIAGNOSTICS_RUN, copied)
    with (copied / "calibration_metric_summary.csv").open("ab") as stream:
        stream.write(b"tampered\n")
    with pytest.raises(V3CalibrationDiagnosticsRunValidationError, match="output hash drifted"):
        validate_calibration_diagnostics_run_v3(copied)
