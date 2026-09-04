from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.governance.shap_stability_faithfulness_run_validator_v3 import (
    DEFAULT_SHAP_STABILITY_FAITHFULNESS_RUN,
    V3ShapStabilityFaithfulnessRunValidationError,
    validate_shap_stability_faithfulness_run_v3,
)


pytestmark = pytest.mark.skipif(
    not DEFAULT_SHAP_STABILITY_FAITHFULNESS_RUN.is_dir(),
    reason="ignored local complete Phase 2A run is unavailable",
)


def test_complete_phase2a_run_passes_independent_validation() -> None:
    receipt = validate_shap_stability_faithfulness_run_v3()
    assert receipt["status"] == "passed"
    assert receipt["generation_commit"] == (
        "6e52de76f7e486985cbc2b32a53b2554c1c6f6c1"
    )
    assert receipt["file_count"] == 14
    assert receipt["stability_pairwise_row_count"] == 75
    assert receipt["faithfulness_sample_row_count"] == 75600
    assert receipt["maximum_raw_margin_additivity_error"] <= 1e-5
    assert receipt["seed_stability_top5_jaccard_mean"] == 1.0
    assert receipt["resample_stability_top5_jaccard_mean"] == 1.0
    assert receipt["guided_minus_random_mean_deletion_auc"] > 0.0
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0


def test_phase2a_validator_rejects_unexpected_file(tmp_path: Path) -> None:
    copied = tmp_path / DEFAULT_SHAP_STABILITY_FAITHFULNESS_RUN.parent.name
    copied = copied / DEFAULT_SHAP_STABILITY_FAITHFULNESS_RUN.name
    shutil.copytree(DEFAULT_SHAP_STABILITY_FAITHFULNESS_RUN, copied)
    (copied / "unexpected.txt").write_text("not allowed\n", encoding="utf-8")
    with pytest.raises(
        V3ShapStabilityFaithfulnessRunValidationError,
        match="closed-world inventory",
    ):
        validate_shap_stability_faithfulness_run_v3(copied)
