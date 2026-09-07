from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.governance.literature_contract_v3 import (
    EXPECTED_OUTPUTS,
    LiteratureContractV3Error,
    POSITIONING_FIELDS,
    export_literature_package_v3,
    validate_literature_contract_v3,
    validate_literature_package_v3,
)


CONTRACT = Path("configs/literature_positioning_v3.json")


def _payload() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _write_contract(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "literature.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_real_contract_freezes_source_verified_bounded_positioning_set() -> None:
    receipt = validate_literature_contract_v3()
    assert receipt["status"] == "passed"
    assert receipt["study_count"] == 25
    assert receipt["exact_inx_study_count"] == 4
    assert receipt["exact_hrdataset_v14_study_count"] == 1
    assert receipt["publisher_asserted_unresolved_doi_count"] == 1
    assert receipt["registered_doi_count"] + receipt["no_assigned_doi_count"] + 1 == 25
    assert receipt["paid_api_calls"] == 0
    assert receipt["flag_counts"]["exact_fold_explanations"].get("yes", 0) == 0
    assert receipt["flag_counts"]["artifact_provenance"].get("yes", 0) == 0


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["studies"].pop(), "25 studies"),
        (lambda value: value["studies"][1]["metadata_sources"].pop(0), "lacks Crossref verification"),
        (lambda value: value["studies"][4]["flags"].update({"exact_fold_explanations": "yes"}), "Prior exact-fold explanation count drifted"),
        (lambda value: value["novelty_boundary"].update({"bounded_claim": "This is universally first."}), "not bounded"),
    ],
)
def test_contract_rejects_scope_doi_matrix_and_novelty_drift(tmp_path: Path, mutation, match: str) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(LiteratureContractV3Error, match=match):
        validate_literature_contract_v3(_write_contract(tmp_path, payload))


def test_contract_has_complete_operational_field_order() -> None:
    payload = _payload()
    assert tuple(payload["positioning_field_definitions"]) == POSITIONING_FIELDS
    assert all(tuple(study["flags"]) == POSITIONING_FIELDS for study in payload["studies"])


def test_export_is_closed_world_and_exactly_contract_derived(tmp_path: Path) -> None:
    output = tmp_path / "package"
    receipt = export_literature_package_v3(
        CONTRACT,
        output,
        generation_commit="test",
        require_clean_git=False,
    )
    assert receipt["status"] == "passed"
    assert receipt["file_count"] == 9
    assert {path.name for path in output.iterdir()} == EXPECTED_OUTPUTS
    assert receipt["study_count"] == receipt["positioning_row_count"] == 25
    with (output / "positioning_table.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[3]["dataset_match"] == "exact_inx"
    assert rows[4]["dataset_match"] == "exact_hrdataset_v14"
    assert rows[15]["shap_stability"] == "yes"
    assert rows[18]["nested_cv"] == "yes"


def test_package_validator_rejects_rehashed_content_tampering(tmp_path: Path) -> None:
    output = tmp_path / "package"
    export_literature_package_v3(
        CONTRACT,
        output,
        generation_commit="test",
        require_clean_git=False,
    )
    path = output / "NOVELTY_POSITIONING.md"
    path.write_text(path.read_text(encoding="utf-8").replace("PerfScoreID", "TargetAlias"), encoding="utf-8")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        if record["path"] == path.name:
            import hashlib

            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            record["size_bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(LiteratureContractV3Error, match="PerfScoreID"):
        validate_literature_package_v3(output, contract_path=CONTRACT)
