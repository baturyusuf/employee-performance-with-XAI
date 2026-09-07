from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.governance.release_readiness_v3 import (
    EXPECTED_OUTPUTS,
    ReleaseReadinessV3Error,
    export_release_readiness_package_v3,
    validate_release_readiness_contract_v3,
    validate_release_readiness_package_v3,
)


CONTRACT = Path("configs/release_readiness_v3.json")


def _payload() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _write_contract(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_real_contract_freezes_fail_closed_release_readiness() -> None:
    receipt = validate_release_readiness_contract_v3()
    assert receipt["status"] == "passed"
    assert receipt["dataset_count"] == 4
    assert receipt["artifact_component_count"] == 10
    assert receipt["artifact_file_count"] == 203
    assert receipt["artifact_size_bytes"] == 13_056_035
    assert receipt["declaration_count"] == 8
    assert receipt["open_blocker_count"] == 11
    assert receipt["historical_raw_path_count"] == 5
    assert receipt["raw_redistribution_allowed_count"] == 0
    assert receipt["github_release_count"] == receipt["remote_tag_count"] == 0
    assert receipt["final_commit_recorded"] is receipt["doi_recorded"] is False
    assert receipt["paid_api_calls"] == 0


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["controls"].update({"public_release_authorized": True}), "Unsafe Phase 4B control"),
        (lambda value: value["datasets"][3].update({"redistribution_status": "allowed"}), "raw redistribution must remain blocked"),
        (lambda value: value["datasets"][3]["license"].update({"status": "verified_unrestricted"}), "license is overclaimed"),
        (lambda value: value["release_plan"].update({"final_commit": "0" * 40}), "Uncreated release identity"),
        (lambda value: value["artifact_inventory"][0].update({"manifest_sha256": "0" * 64}), "Artifact manifest hash drifted"),
        (lambda value: value["blockers"][0].update({"status": "closed"}), "silently closed"),
        (lambda value: value["repository_snapshot"]["historical_raw_dataset_paths"].pop(), "Historical raw-data path register drifted"),
    ],
)
def test_contract_rejects_authority_rights_release_and_blocker_drift(tmp_path: Path, mutation, match: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(ReleaseReadinessV3Error, match=match):
        validate_release_readiness_contract_v3(_write_contract(tmp_path, payload))


def test_dataset_rights_and_declarations_remain_explicitly_unresolved() -> None:
    payload = _payload()
    assert all(item["redistribution_status"].startswith("blocked_") for item in payload["datasets"])
    assert payload["datasets"][0]["license"]["identifier"] == "IABAC-all-rights-reserved"
    assert payload["datasets"][1]["license"]["identifier"] == "CC-BY-NC-ND-4.0"
    assert payload["datasets"][2]["license"]["identifier"] == "ODbL-1.0-and-DbCL-1.0"
    assert payload["datasets"][3]["license"]["identifier"] is None
    declaration_statuses = {item["field"]: item["status"] for item in payload["declarations"]}
    assert declaration_statuses["Institutional Review Board Statement"] == "pending_institutional_determination"
    assert declaration_statuses["Informed Consent Statement"] == "pending_institutional_determination"
    assert declaration_statuses["Data Availability Statement"] == "draft_ready_with_limits"
    assert declaration_statuses["Code Availability Statement"] == "draft_ready_with_limits"


def test_export_is_closed_world_and_exactly_contract_derived(tmp_path: Path) -> None:
    output = tmp_path / "package"
    receipt = export_release_readiness_package_v3(
        CONTRACT,
        output,
        generation_commit="test",
        require_clean_git=False,
    )
    assert receipt["status"] == "passed"
    assert receipt["file_count"] == 12
    assert {path.name for path in output.iterdir()} == EXPECTED_OUTPUTS
    assert receipt["dataset_count"] == 4
    assert receipt["artifact_component_count"] == 10
    assert receipt["declaration_count"] == 8
    assert receipt["open_blocker_count"] == 11
    with (output / "DATA_PROVENANCE_LICENSE_REGISTER.csv").open(encoding="utf-8", newline="") as stream:
        datasets = list(csv.DictReader(stream))
    assert datasets[0]["redistribution_status"] == "blocked_pending_written_permission"
    assert datasets[3]["license_status"] == "unverified"
    candidate = json.loads((output / "release_candidate_manifest.json").read_text(encoding="utf-8"))
    assert candidate["final_commit"] is None
    assert candidate["release_authorized"] is False
    assert candidate["raw_dataset_included"] is False


def test_two_temporary_exports_are_byte_identical(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    export_release_readiness_package_v3(CONTRACT, first, generation_commit="test", require_clean_git=False)
    export_release_readiness_package_v3(CONTRACT, second, generation_commit="test", require_clean_git=False)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }


def test_package_validator_rejects_rehashed_boundary_tampering(tmp_path: Path) -> None:
    output = tmp_path / "package"
    export_release_readiness_package_v3(CONTRACT, output, generation_commit="test", require_clean_git=False)
    path = output / "PROVENANCE_LICENSE_REPORT.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "No raw dataset is approved for redistribution",
            "Raw dataset redistribution is approved",
        ),
        encoding="utf-8",
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        if record["path"] == path.name:
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            record["size_bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ReleaseReadinessV3Error, match="No raw dataset is approved"):
        validate_release_readiness_package_v3(output, contract_path=CONTRACT)


def test_release_plan_never_uses_mutable_branch_as_final_identity() -> None:
    payload = _payload()
    release = payload["release_plan"]
    assert release["final_commit"] is None
    assert release["status"] == "blocked_preparation_only"
    assert payload["repository_snapshot"]["active_branch"] not in release["proposed_tag"]
    assert "the current mutable branch is the final reproducibility reference" in payload["prohibited_assertions"]
