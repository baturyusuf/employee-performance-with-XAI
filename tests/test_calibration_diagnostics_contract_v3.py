from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.governance.calibration_diagnostics_contract_v3 import (
    CalibrationDiagnosticsContractV3Error,
    DEFAULT_CALIBRATION_DIAGNOSTICS_CONTRACT,
    validate_calibration_diagnostics_contract_v3,
)


def _contract() -> dict:
    return json.loads(DEFAULT_CALIBRATION_DIAGNOSTICS_CONTRACT.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_calibration_diagnostics_contract_validates_exact_sources() -> None:
    receipt = validate_calibration_diagnostics_contract_v3()
    assert receipt["status"] == "passed"
    assert receipt["sample_count"] == 1200
    assert receipt["methods"] == ["raw", "sigmoid"]
    assert receipt["classwise_targets"] == 3
    assert receipt["cumulative_targets"] == 2
    assert receipt["planned_new_model_fit_calls"] == 0
    assert receipt["planned_new_calibrator_fit_calls"] == 0
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("source_calibration", "method_selection_performed"), True, "Source calibration"),
        (("reliability", "bin_count"), 15, "Reliability"),
        (("calibration_regression", "confidence_interval_applicable"), True, "Calibration regression"),
        (("comparison", "all_metrics_improved_claim_allowed"), True, "Universal calibration improvement"),
        (("publication", "publish_oof_prediction_rows"), True, "OOF-row publication"),
    ],
)
def test_calibration_diagnostics_contract_rejects_claim_or_method_drift(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    payload = _contract()
    payload[path[0]][path[1]] = value
    with pytest.raises(CalibrationDiagnosticsContractV3Error, match=message):
        validate_calibration_diagnostics_contract_v3(_write(tmp_path, payload))
