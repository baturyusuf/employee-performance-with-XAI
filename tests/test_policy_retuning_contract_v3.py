from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.governance.policy_retuning_contract_v3 import (
    DEFAULT_POLICY_RETUNING_CONTRACT_PATH,
    POLICY_FEATURE_COUNTS,
    POLICY_IDS,
    PolicyRetuningContractError,
    validate_policy_retuning_contract_v3,
)


def _contract() -> dict:
    return json.loads(DEFAULT_POLICY_RETUNING_CONTRACT_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "policy_retuning_v3.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_default_policy_retuning_contract_is_exact_and_fit_free() -> None:
    receipt = validate_policy_retuning_contract_v3()
    assert receipt["status"] == "passed"
    assert receipt["policy_feature_counts"] == dict(zip(POLICY_IDS, POLICY_FEATURE_COUNTS))
    assert receipt["candidate_count"] == 8
    assert receipt["planned_new_estimator_fit_calls"] == 2480
    assert receipt["reused_fixed_policy_count"] == 4
    assert receipt["new_fixed_policy_count"] == 2
    assert receipt["retuned_policy_count"] == 6
    assert receipt["model_fit_count"] == 0
    assert receipt["network_calls"] == 0
    assert receipt["paid_api_calls"] == 0


def test_local_canonical_sources_and_crosswalk_validate_when_available() -> None:
    receipt = validate_policy_retuning_contract_v3()
    local = receipt["local_canonical_sources"]
    if local["available"]:
        assert local["validated"] is True
        assert local["exact_fixed_policy_crosswalk_count"] == 4
        assert local["sample_count"] == 1200


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda c: c["information_contract"].update({"primary_policy_id": "P2"}), "Primary policy"),
        (lambda c: c["fixed_hyperparameter_estimand"].update({"independent_policy_tuning": True}), "Fixed-hyperparameter estimand"),
        (lambda c: c["independently_retuned_estimand"].update({"outer_test_used_for_selection": True}), "Retuned estimand"),
        (lambda c: c["computational_scope"].update({"planned_new_estimator_fit_calls": 2479}), "Computational scope"),
        (lambda c: c["evaluation"].update({"inferential_claim_from_point_difference_allowed": True}), "Point differences"),
        (lambda c: c["publication"].update({"publish_employee_level_oof_rows": True}), "publish_employee_level_oof_rows"),
    ],
)
def test_contract_rejects_estimand_or_claim_drift(tmp_path: Path, mutator, message: str) -> None:
    payload = _contract()
    mutator(payload)
    with pytest.raises(PolicyRetuningContractError, match=message):
        validate_policy_retuning_contract_v3(_write(tmp_path, payload))


def test_contract_rejects_unexpected_top_level_field(tmp_path: Path) -> None:
    payload = _contract()
    payload["result_selected_policy"] = "P0"
    with pytest.raises(PolicyRetuningContractError, match="top-level inventory"):
        validate_policy_retuning_contract_v3(_write(tmp_path, payload))
