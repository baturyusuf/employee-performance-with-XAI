from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.governance.repeated_nested_cv_run_validator_v3 import (
    V3RepeatedNestedCVRunValidationError,
    validate_repeated_nested_cv_run_v3,
)


RUN_DIR = Path(
    "reports/major_revision_v3_runs/"
    "phase1c_v3_20260903T215015Z_78649c4/repeated_nested_cv"
)


@pytest.mark.skipif(not RUN_DIR.is_dir(), reason="local row-level v3 run is intentionally ignored")
def test_completed_local_phase1c_run_passes_independent_recomputation() -> None:
    receipt = validate_repeated_nested_cv_run_v3(RUN_DIR)
    assert receipt["status"] == "passed"
    assert receipt["generation_commit"] == "78649c426e69fb5270f9d027b11ba6ba87d71a41"
    assert receipt["distinct_outer_assignment_count"] == 5
    assert receipt["oof_prediction_row_count"] == 54_000
    assert receipt["candidate_search_row_count"] == 1_100
    assert receipt["fold_metric_row_count"] == 225
    assert receipt["repetition_metric_row_count"] == 720
    assert receipt["best_mean_macro_f1_model"] == "xgboost"
    assert receipt["best_mean_balanced_accuracy_model"] == "cumulative_threshold_xgboost"
    assert receipt["best_mean_qwk_model"] == "random_forest"
    assert receipt["best_mean_ordinal_mae_model"] == "random_forest"


@pytest.mark.skipif(not RUN_DIR.is_dir(), reason="local row-level v3 run is intentionally ignored")
def test_validator_rejects_an_unexpected_file_before_reading_evidence(tmp_path: Path) -> None:
    copied = tmp_path / "repeated_nested_cv"
    shutil.copytree(RUN_DIR, copied)
    (copied / "unexpected.txt").write_text("not allowed\n", encoding="utf-8")
    with pytest.raises(V3RepeatedNestedCVRunValidationError, match="closed-world inventory"):
        validate_repeated_nested_cv_run_v3(copied)
