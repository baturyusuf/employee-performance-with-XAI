from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.governance.ordinal_benchmark_run_validator_v3 import (
    V3OrdinalRunValidationError,
    validate_ordinal_benchmark_run_v3,
)


RUN_DIR = Path(
    "reports/major_revision_v3_runs/"
    "phase1b_v3_20260903T130912Z_dc5cb8b/ordinal_benchmark"
)


@pytest.mark.skipif(not RUN_DIR.is_dir(), reason="local row-level v3 run is intentionally ignored")
def test_completed_local_phase1b_run_passes_independent_recomputation() -> None:
    receipt = validate_ordinal_benchmark_run_v3(RUN_DIR)
    assert receipt["status"] == "passed"
    assert receipt["generation_commit"] == "dc5cb8b96b096bb2efc6c242403b7e51f870a01b"
    assert receipt["model_count"] == 9
    assert receipt["combined_oof_row_count"] == 10_800
    assert receipt["candidate_search_row_count"] == 140
    assert receipt["aggregate_metric_row_count"] == 144
    assert receipt["per_class_row_count"] == 27
    assert receipt["confusion_row_count"] == 81
    assert receipt["best_macro_f1_model"] == "cumulative_threshold_xgboost"
    assert receipt["best_qwk_model"] == "random_forest"
    assert receipt["best_rps_model"] == "lightgbm"


@pytest.mark.skipif(not RUN_DIR.is_dir(), reason="local row-level v3 run is intentionally ignored")
def test_validator_rejects_an_unexpected_file_before_reading_evidence(tmp_path: Path) -> None:
    copied = tmp_path / "ordinal_benchmark"
    shutil.copytree(RUN_DIR, copied)
    (copied / "unexpected.txt").write_text("not allowed\n", encoding="utf-8")
    with pytest.raises(V3OrdinalRunValidationError, match="closed-world inventory"):
        validate_ordinal_benchmark_run_v3(copied)
