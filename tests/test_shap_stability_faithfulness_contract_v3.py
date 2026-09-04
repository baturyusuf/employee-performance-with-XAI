from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.governance.shap_stability_faithfulness_contract_v3 import (
    DEFAULT_SHAP_STABILITY_FAITHFULNESS_CONTRACT,
    ShapStabilityFaithfulnessContractError,
    validate_shap_stability_faithfulness_contract_v3,
)


def _contract() -> dict:
    return json.loads(
        DEFAULT_SHAP_STABILITY_FAITHFULNESS_CONTRACT.read_text(encoding="utf-8")
    )


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def test_default_phase2a_contract_is_exact_and_fit_free() -> None:
    receipt = validate_shap_stability_faithfulness_contract_v3()
    assert receipt["status"] == "passed"
    assert receipt["sample_count"] == 1200
    assert receipt["feature_count"] == 20
    assert receipt["seed_stability_run_count_including_reference"] == 6
    assert receipt["resampling_run_count"] == 5
    assert receipt["top_k_values"] == [5, 10, 15]
    assert receipt["deletion_feature_counts"] == [1, 3, 5]
    assert receipt["random_baseline_repetitions"] == 20
    assert receipt["planned_new_estimator_fit_calls"] == 100
    assert receipt["local_canonical_sources"]["validated"] is True
    assert receipt["model_fit_count"] == 0
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda c: c["grouped_shap_implementation"].update({"model_output": "probability"}), "Grouped SHAP implementation"),
        (lambda c: c["seed_stability"].update({"only_intended_change": "seed_and_candidate"}), "Seed stability"),
        (lambda c: c["resampling_stability"].update({"selection_repeated": True}), "Resampling stability"),
        (lambda c: c["stability_evaluation"].update({"confidence_interval_applicable": True}), "Stability evaluation"),
        (lambda c: c["faithfulness"].update({"human_usefulness_claim_allowed": True}), "Faithfulness"),
        (lambda c: c["computational_scope"].update({"total_new_estimator_fit_calls": 99}), "Computational scope"),
        (lambda c: c["publication"].update({"publish_local_shap_rows": True}), "publish_local_shap_rows"),
    ],
)
def test_phase2a_contract_rejects_scientific_or_publication_drift(
    tmp_path: Path, mutator, message: str
) -> None:
    value = _contract()
    mutator(value)
    with pytest.raises(ShapStabilityFaithfulnessContractError, match=message):
        validate_shap_stability_faithfulness_contract_v3(_write(tmp_path, value))


def test_phase2a_contract_rejects_unexpected_top_level_field(tmp_path: Path) -> None:
    value = _contract()
    value["result_selected_seed"] = 1044
    with pytest.raises(ShapStabilityFaithfulnessContractError, match="top-level inventory"):
        validate_shap_stability_faithfulness_contract_v3(_write(tmp_path, value))
