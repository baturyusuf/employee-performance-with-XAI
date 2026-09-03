from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.governance.feature_availability_contract import (
    DEFAULT_ACQUISITION_PATH,
    DEFAULT_CONTRACT_PATH,
    EXPECTED_POLICIES,
    EXPECTED_RISK_TYPES,
    FeatureAvailabilityContractError,
    main,
    render_feature_availability_markdown,
    validate_feature_availability_contract,
)


def _contract() -> dict:
    return json.loads(DEFAULT_CONTRACT_PATH.read_text(encoding="utf-8"))


def _write_contract(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_real_contract_covers_exact_pinned_inx_schema_and_taxonomy() -> None:
    receipt = validate_feature_availability_contract()
    assert receipt["status"] == "passed"
    assert receipt["feature_count"] == 28
    assert receipt["policy_count"] == 6
    assert receipt["timestamp_verified_feature_count"] == 0
    assert receipt["policy_feature_counts"] == {
        "P0": 26,
        "P1": 24,
        "P2": 21,
        "P3": 20,
        "P4": 13,
        "P5": 6,
    }
    assert set(receipt["risk_type_counts"]) == EXPECTED_RISK_TYPES
    assert all(value > 0 for value in receipt["risk_type_counts"].values())


def test_policy_identity_is_exact_and_exclusions_are_nested() -> None:
    contract = _contract()
    assert tuple((row["policy_id"], row["name"]) for row in contract["policies"]) == EXPECTED_POLICIES
    exclusions = [set(row["excluded_features"]) for row in contract["policies"]]
    assert all(left < right for left, right in zip(exclusions, exclusions[1:]))
    features = {row["feature_name"]: row for row in contract["features"]}
    assert {
        name for name, row in features.items() if row["risk_type"] == "timing_uncertain"
    } <= exclusions[4]
    assert {
        name for name, row in features.items() if row["risk_type"] == "organizational_proxy"
    } <= exclusions[5]


def test_contract_never_claims_verified_availability_or_leakage_elimination() -> None:
    contract = _contract()
    assert contract["prediction_scenario"]["evidence_status"] == (
        "conceptual_estimand_only_no_observed_feature_or_decision_timestamps"
    )
    assert all(
        row["availability_confidence"] != "high" for row in contract["features"]
    )
    assert all(
        "confirmed" not in row["availability_at_prediction_time"].lower()
        for row in contract["features"]
    )
    limitation_text = " ".join(contract["global_limitations"]).lower()
    assert "no policy supports the claim that all leakage was eliminated" in limitation_text


def test_markdown_rendering_contains_all_features_and_boundaries() -> None:
    rendered = render_feature_availability_markdown()
    contract = _contract()
    assert rendered.startswith("# Feature Availability and Governance Contract — v3")
    for row in contract["features"]:
        assert f"| {row['feature_name']} |" in rendered
    for policy_id, policy_name in EXPECTED_POLICIES:
        assert f"| {policy_id} | {policy_name} |" in rendered
    assert "not verified prospective deployment evidence" in rendered


def test_missing_feature_fails_closed(tmp_path: Path) -> None:
    contract = _contract()
    contract["features"].pop()
    path = _write_contract(tmp_path, contract)
    with pytest.raises(FeatureAvailabilityContractError, match="columns/order differ"):
        validate_feature_availability_contract(path, DEFAULT_ACQUISITION_PATH)


def test_high_availability_confidence_fails_closed(tmp_path: Path) -> None:
    contract = _contract()
    contract["features"][1]["availability_confidence"] = "high"
    path = _write_contract(tmp_path, contract)
    with pytest.raises(FeatureAvailabilityContractError, match="cannot support high/confirmed"):
        validate_feature_availability_contract(path, DEFAULT_ACQUISITION_PATH)


def test_non_nested_policy_fails_closed(tmp_path: Path) -> None:
    contract = _contract()
    contract["policies"][4]["excluded_features"].remove("Gender")
    path = _write_contract(tmp_path, contract)
    with pytest.raises(FeatureAvailabilityContractError, match="preceding policy"):
        validate_feature_availability_contract(path, DEFAULT_ACQUISITION_PATH)


def test_strict_proxy_policy_must_remove_every_declared_proxy(tmp_path: Path) -> None:
    contract = _contract()
    contract["policies"][5]["excluded_features"].remove("EmpHourlyRate")
    path = _write_contract(tmp_path, contract)
    with pytest.raises(FeatureAvailabilityContractError, match="organisational proxy"):
        validate_feature_availability_contract(path, DEFAULT_ACQUISITION_PATH)


def test_policy_cannot_silently_add_an_unscoped_exclusion(tmp_path: Path) -> None:
    contract = _contract()
    contract["policies"][1]["excluded_features"].append("Age")
    path = _write_contract(tmp_path, contract)
    with pytest.raises(FeatureAvailabilityContractError, match="exact declared exclusion set"):
        validate_feature_availability_contract(path, DEFAULT_ACQUISITION_PATH)


def test_primary_status_must_match_p3_membership(tmp_path: Path) -> None:
    contract = _contract()
    contract["features"][1]["primary_policy_status"] = "retained_sensitive"
    path = _write_contract(tmp_path, contract)
    with pytest.raises(FeatureAvailabilityContractError, match="excluded by P3"):
        validate_feature_availability_contract(path, DEFAULT_ACQUISITION_PATH)


def test_cli_writes_validated_markdown_atomically(tmp_path: Path) -> None:
    output = tmp_path / "contract.md"
    assert main(["--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == render_feature_availability_markdown()
    assert not (tmp_path / ".contract.md.tmp").exists()


def test_semantically_equivalent_json_has_stable_semantic_hash(tmp_path: Path) -> None:
    contract = _contract()
    first = validate_feature_availability_contract()
    reordered = copy.deepcopy(contract)
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(reordered, indent=4, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    second = validate_feature_availability_contract(path, DEFAULT_ACQUISITION_PATH)
    assert first["contract_sha256"] != second["contract_sha256"]
    assert first["contract_semantic_sha256"] == second["contract_semantic_sha256"]
