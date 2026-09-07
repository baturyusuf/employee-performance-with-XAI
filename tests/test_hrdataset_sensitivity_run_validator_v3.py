from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.governance.hrdataset_sensitivity_run_validator_v3 import (
    HRDatasetSensitivityRunValidationV3Error,
    validate_hrdataset_sensitivity_run_v3,
)


RUN_DIR = Path(
    "reports/major_revision_v3_runs/"
    "phase3a_v3_20260907T141213Z_f6e6a0a/hrdataset_sensitivity"
)


@pytest.fixture(scope="module")
def validation_receipt():
    if not RUN_DIR.is_dir():
        pytest.skip("local row-level Phase 3A run is intentionally ignored")
    return validate_hrdataset_sensitivity_run_v3(RUN_DIR)


def test_independent_validator_recomputes_complete_phase3a_run(validation_receipt) -> None:
    assert validation_receipt["status"] == "passed"
    assert validation_receipt["generation_commit"] == (
        "f6e6a0af7f36d7d69f028df07b6c0842c25ebc56"
    )
    assert validation_receipt["file_count"] == 18
    assert validation_receipt["distinct_outer_assignment_count"] == 10
    assert validation_receipt["candidate_search_row_count"] == 400
    assert validation_receipt["oof_prediction_row_count"] == 15_550
    assert validation_receipt["calibration_training_oof_row_count"] == 12_440
    assert validation_receipt["calibrator_parameter_row_count"] == 175
    assert validation_receipt["fold_metric_row_count"] == 250
    assert validation_receipt["repetition_metric_row_count"] == 500
    assert validation_receipt["per_class_metric_row_count"] == 175
    assert validation_receipt["confusion_matrix_row_count"] == 625
    assert validation_receipt["maximum_sigmoid_probability_replay_error"] == 0.0
    assert validation_receipt["cross_formulation_metric_difference_computed"] is False
    assert validation_receipt["locked_model_transport"] is False
    assert validation_receipt["network_calls"] == validation_receipt["paid_api_calls"] == 0


def test_validator_does_not_import_phase3a_runner_calculations() -> None:
    source = Path(
        "src/governance/hrdataset_sensitivity_run_validator_v3.py"
    ).read_text(encoding="utf-8")
    assert "from src.experiments.hrdataset_sensitivity_v3" not in source
    assert "import hrdataset_sensitivity_v3" not in source


@pytest.mark.skipif(not RUN_DIR.is_dir(), reason="local row-level Phase 3A run is intentionally ignored")
def test_validator_rejects_tampered_scientific_output(tmp_path: Path) -> None:
    copied = tmp_path / "hrdataset_sensitivity"
    shutil.copytree(RUN_DIR, copied)
    path = copied / "calibration_training_oof.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(HRDatasetSensitivityRunValidationV3Error, match="Output byte hash drifted"):
        validate_hrdataset_sensitivity_run_v3(copied)


@pytest.mark.skipif(not RUN_DIR.is_dir(), reason="local row-level Phase 3A run is intentionally ignored")
def test_validator_rejects_extra_file_before_recomputation(tmp_path: Path) -> None:
    copied = tmp_path / "hrdataset_sensitivity"
    shutil.copytree(RUN_DIR, copied)
    (copied / "unexpected.csv").write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(HRDatasetSensitivityRunValidationV3Error, match="closed-world inventory"):
        validate_hrdataset_sensitivity_run_v3(copied)
