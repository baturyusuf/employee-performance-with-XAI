from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.governance.policy_retuning_run_validator_v3 import (
    DEFAULT_POLICY_RETUNING_RUN,
    V3PolicyRetuningRunValidationError,
    validate_policy_retuning_run_v3,
)


pytestmark = pytest.mark.skipif(
    not DEFAULT_POLICY_RETUNING_RUN.is_dir(),
    reason="ignored local complete Phase 1D run is unavailable",
)


def test_complete_policy_retuning_run_passes_independent_validation() -> None:
    receipt = validate_policy_retuning_run_v3()
    assert receipt["status"] == "passed"
    assert receipt["generation_commit"] == "823c84866b461266c75f3224527f679a86ab670e"
    assert receipt["file_count"] == 12
    assert receipt["candidate_search_row_count"] == 480
    assert receipt["selected_hyperparameter_row_count"] == 60
    assert receipt["combined_oof_row_count"] == 14400
    assert receipt["aggregate_metric_row_count"] == 192
    assert receipt["p3_maximum_probability_replay_error"] == 0.0
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0


def test_policy_retuning_validator_rejects_unexpected_file(tmp_path: Path) -> None:
    copied = tmp_path / DEFAULT_POLICY_RETUNING_RUN.parent.name / DEFAULT_POLICY_RETUNING_RUN.name
    shutil.copytree(DEFAULT_POLICY_RETUNING_RUN, copied)
    (copied / "unexpected.txt").write_text("not allowed\n", encoding="utf-8")
    with pytest.raises(V3PolicyRetuningRunValidationError, match="closed-world inventory"):
        validate_policy_retuning_run_v3(copied)
