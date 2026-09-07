from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.governance.data_quality_contract_v3 import (
    DataQualityContractV3Error,
    validate_data_quality_contract_v3,
)


CONTRACT = Path("configs/data_quality_v3.json")


def _payload() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_real_contract_binds_exact_core_scope_and_offline_aggregate_audit() -> None:
    receipt = validate_data_quality_contract_v3()
    assert receipt["status"] == "passed"
    assert receipt["dataset_keys"] == ["inx_primary", "hrdataset_v14"]
    assert receipt["row_counts"] == {"inx_primary": 1200, "hrdataset_v14": 311}
    assert receipt["raw_column_counts"] == {"inx_primary": 28, "hrdataset_v14": 36}
    assert receipt["cleaned_column_counts"] == {"inx_primary": 28, "hrdataset_v14": 39}
    assert receipt["primary_feature_counts"] == {"inx_primary": 20, "hrdataset_v14": 7}
    assert receipt["model_fit_calls"] == 0
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["scope"].update({"dataset_keys": ["inx_primary"]}), "Core dataset order or identity drifted"),
        (lambda value: value["audit_policy"].update({"near_constant_nonmissing_mode_share_threshold": 0.95}), "near_constant_nonmissing_mode_share_threshold drifted"),
        (lambda value: value["audit_policy"].update({"raw_values_or_row_identifiers_may_be_published": True}), "raw_values_or_row_identifiers_may_be_published drifted"),
        (lambda value: value["claim_boundary"].update({"external_generalization_established": True}), "Every data-quality claim boundary"),
    ],
)
def test_contract_rejects_scope_policy_and_claim_drift(tmp_path: Path, mutation, match: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(DataQualityContractV3Error, match=match):
        validate_data_quality_contract_v3(_write(tmp_path, payload))


def test_contract_rejects_bound_source_hash_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["source_contracts"]["canonical_config"]["sha256"] = "0" * 64
    with pytest.raises(DataQualityContractV3Error, match="Bound source hash drifted"):
        validate_data_quality_contract_v3(_write(tmp_path, payload))
