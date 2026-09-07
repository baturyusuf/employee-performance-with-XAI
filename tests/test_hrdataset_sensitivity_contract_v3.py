from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.governance.hrdataset_sensitivity_contract_v3 import (
    HRDatasetSensitivityContractV3Error,
    validate_hrdataset_sensitivity_contract_v3,
)


CONTRACT = Path("configs/hrdataset_sensitivity_v3.json")


def _payload() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_real_contract_binds_two_targets_exact_features_and_v2_reference() -> None:
    receipt = validate_hrdataset_sensitivity_contract_v3()
    assert receipt["status"] == "passed"
    assert receipt["sample_count"] == 311
    assert receipt["feature_count"] == 7
    assert receipt["formulation_count"] == 2
    assert receipt["mapped_support"] == {
        "primary_three_class": {2: 31, 3: 243, 4: 37},
        "raw_order_four_class": {1: 13, 2: 18, 3: 243, 4: 37},
    }
    assert receipt["planned_xgboost_fit_calls"] == 2300
    assert receipt["planned_baseline_fit_calls"] == 150
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["target_formulations"][0]["mapping"].update({"PIP": 3}), "primary_three_class.mapping drifted"),
        (lambda value: value["target_formulations"][1]["mapping"].update({"PIP": 2}), "raw_order_four_class.mapping drifted"),
        (lambda value: value["design"].update({"outer_splits": 10}), "design.outer_splits drifted"),
        (lambda value: value["model_protocol"].update({"outer_test_used_for_selection": True}), "outer_test_used_for_selection drifted"),
        (lambda value: value["evaluation"].update({"cross_formulation_metric_difference_allowed": True}), "must remain prohibited"),
        (lambda value: value["publication"].update({"publish_employee_level_oof": True}), "publication is prohibited"),
    ],
)
def test_contract_rejects_scientific_drift(tmp_path: Path, mutation, match: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(HRDatasetSensitivityContractV3Error, match=match):
        validate_hrdataset_sensitivity_contract_v3(_write(tmp_path, payload))


def test_contract_rejects_bound_source_hash_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["source_contracts"]["model_grid"]["sha256"] = "0" * 64
    with pytest.raises(HRDatasetSensitivityContractV3Error, match="Bound source hash drifted"):
        validate_hrdataset_sensitivity_contract_v3(_write(tmp_path, payload))
