from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.governance.claim_matrix_v3 import (
    EXPECTED_OUTPUTS,
    ClaimMatrixV3Error,
    export_claim_matrix_package_v3,
    validate_claim_matrix_contract_v3,
    validate_claim_matrix_package_v3,
)


CONTRACT = Path("configs/claim_matrix_v3.json")


def _payload() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _write_contract(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "claim_matrix.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_contract_validates_all_components_and_exact_claims() -> None:
    receipt = validate_claim_matrix_contract_v3(CONTRACT)
    assert receipt["status"] == "passed"
    assert receipt["claim_count"] == 45
    assert receipt["numerical_claim_count"] == 32
    assert receipt["narrative_claim_count"] == 13
    assert receipt["source_file_count"] == 27
    assert receipt["component_count"] == 12
    assert receipt["approval_status"] == "approved"
    assert receipt["manuscript_editing_authorized"] is True
    assert receipt["paid_api_calls"] == 0


def test_numerical_claims_have_unique_row_selectors_and_deterministic_display() -> None:
    payload = _payload()
    numerical = [claim for claim in payload["claims"] if claim["claim_type"] == "numerical"]
    assert all(claim["source_path"].endswith(".csv") for claim in numerical)
    assert all(claim["source_selector"] for claim in numerical)
    assert all(claim["source_column"] and claim["exact_value"] for claim in numerical)
    assert all(claim["display_value"] in claim["proposed_claim"] for claim in numerical)


def test_approved_digest_preserves_frozen_claim_rows_and_boundaries() -> None:
    payload = _payload()
    assert payload["status"] == "approved_for_phase5b"
    assert payload["user_approval"]["status"] == "approved"
    assert payload["user_approval"]["approved_claim_set_sha256"] == "1664b188df14135d3b3de8642de3d080c25a3fbadc440615c2bd04b2f14ddabe"
    assert all(claim["approval_status"] == "pending_user_approval" for claim in payload["claims"])
    assert all(claim["mandatory_qualifier"] for claim in payload["claims"])
    assert all(claim["prohibited_overclaim"] for claim in payload["claims"])
    assert len(payload["global_prohibited_assertions"]) == 16


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda payload: payload["claims"][-1].__setitem__("exact_value", "7"),
            "exact value drifted",
        ),
        (
            lambda payload: payload["claims"][0].__setitem__("source_sha256", "0" * 64),
            "source SHA-256 drifted",
        ),
        (
            lambda payload: payload["claims"][0].__setitem__("evidence_anchor", "absent narrative anchor value"),
            "evidence anchor drifted",
        ),
        (
            lambda payload: payload["user_approval"].__setitem__("approved_claim_set_sha256", "0" * 64),
            "Approved claim-set digest does not match",
        ),
        (
            lambda payload: payload["controls"].__setitem__("manuscript_editing_authorized", False),
            "Manuscript authorization does not match",
        ),
        (
            lambda payload: payload["user_approval"].__setitem__("status", "pending"),
            "Contract and approval states disagree",
        ),
    ],
)
def test_contract_rejects_evidence_or_approval_drift(tmp_path: Path, mutation, match: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(ClaimMatrixV3Error, match=match):
        validate_claim_matrix_contract_v3(_write_contract(tmp_path, payload))


def test_export_is_closed_world_and_records_digest_approval(tmp_path: Path) -> None:
    output = tmp_path / "package"
    receipt = export_claim_matrix_package_v3(
        CONTRACT,
        output,
        generation_commit="test",
        require_clean_git=False,
    )
    assert receipt["status"] == "passed_approved_for_phase5b"
    assert receipt["file_count"] == 9
    assert {path.name for path in output.iterdir()} == EXPECTED_OUTPUTS
    approval = json.loads((output / "approval_record.json").read_text(encoding="utf-8"))
    assert approval["decision"] == "approved"
    assert approval["approved_by"] == "user"
    assert approval["approved_claim_set_sha256"] == "1664b188df14135d3b3de8642de3d080c25a3fbadc440615c2bd04b2f14ddabe"
    assert approval["manuscript_editing_authorized"] is True
    matrix = (output / "CLAIM_MATRIX.csv").read_text(encoding="utf-8")
    assert "manuscript/mdpi_information/main.md" not in matrix
    assert "data/raw/" not in matrix


def test_temporary_exports_are_byte_identical(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    export_claim_matrix_package_v3(CONTRACT, first, generation_commit="test", require_clean_git=False)
    export_claim_matrix_package_v3(CONTRACT, second, generation_commit="test", require_clean_git=False)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }


def test_replace_existing_republishes_the_same_approved_bytes(tmp_path: Path) -> None:
    output = tmp_path / "package"
    export_claim_matrix_package_v3(CONTRACT, output, generation_commit="test", require_clean_git=False)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    receipt = export_claim_matrix_package_v3(
        CONTRACT,
        output,
        generation_commit="test",
        require_clean_git=False,
        replace_existing=True,
    )
    assert receipt["status"] == "passed_approved_for_phase5b"
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_package_validator_rejects_rehashed_sentence_tampering(tmp_path: Path) -> None:
    output = tmp_path / "package"
    export_claim_matrix_package_v3(CONTRACT, output, generation_commit="test", require_clean_git=False)
    path = output / "CLAIM_MATRIX.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "not an observed prospective prediction exercise",
            "a proven prospective prediction exercise",
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
    with pytest.raises(ClaimMatrixV3Error, match="manifest content drifted|derived file drifted"):
        validate_claim_matrix_package_v3(output, contract_path=CONTRACT)
