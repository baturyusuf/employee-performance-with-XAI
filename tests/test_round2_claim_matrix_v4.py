from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from src.governance.round2_claim_matrix_v4 import (
    CONFIG,
    DISPOSITIONS,
    EXPECTED_OUTPUTS,
    OUTPUT_DIR,
    Round2ClaimMatrixError,
    validate_round2_claim_matrix,
)


def _rows(name: str) -> list[dict[str, str]]:
    with (OUTPUT_DIR / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_round2_claim_package_is_approved_and_complete() -> None:
    receipt = validate_round2_claim_matrix()
    assert receipt["status"] == "passed_approved_for_rewrite"
    assert receipt["claim_count"] == 129
    assert receipt["active_claim_count"] == 117
    assert receipt["numerical_claim_count"] == 99
    assert receipt["narrative_claim_count"] == 30
    assert receipt["source_file_count"] == 40
    assert receipt["disposition_counts"] == {
        "modified": 5,
        "new": 79,
        "prohibited": 5,
        "retained": 38,
        "superseded": 2,
    }
    assert receipt["manuscript_editing_authorized"] is True
    assert receipt["bibliography_editing_authorized"] is True
    assert receipt["reviewer_response_editing_authorized"] is True
    assert receipt["paid_api_calls"] == 0
    assert {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()} == EXPECTED_OUTPUTS


def test_every_historical_and_round2_item_has_an_explicit_disposition() -> None:
    rows = _rows("ROUND2_CLAIM_MATRIX.csv")
    assert {row["disposition"] for row in rows} == DISPOSITIONS
    historical = [row for row in rows if row["prior_claim_id"]]
    assert len(historical) == 45
    assert {row["claim_id"] for row in historical} == {row["prior_claim_id"] for row in historical}
    assert all(row["active_for_rewrite"] == "False" for row in rows if row["disposition"] in {"modified", "superseded", "prohibited"})
    assert all(row["approval_status"] == "approved" for row in rows)


def test_selection_sensitivity_remains_separated_not_binary() -> None:
    rows = {row["claim_id"]: row for row in _rows("ROUND2_CLAIM_MATRIX.csv")}
    assert rows["R2E001"]["display_value"] == "6"
    assert rows["R2E002"]["display_value"] == "9"
    assert rows["R2E003"]["display_value"] == "37"
    assert "separate leader changes" in rows["R2N001"]["proposed_claim"]
    assert rows["R2P001"]["disposition"] == "prohibited"
    assert rows["R2P001"]["active_for_rewrite"] == "False"


def test_hr_primary_matched_comparison_and_sample_removal_are_distinct() -> None:
    rows = {row["claim_id"]: row for row in _rows("ROUND2_CLAIM_MATRIX.csv")}
    matched = [rows[f"R2E{claim_id:03d}"] for claim_id in range(218, 225)]
    removal = [rows[f"R2E{claim_id:03d}"] for claim_id in range(225, 229)]
    assert all('"comparison_id":"matched_refit_effect"' in row["source_selector"] for row in matched)
    assert all('"comparison_id":"sample_removal_effect"' in row["source_selector"] for row in removal)
    assert rows["R2E218"]["display_value"] == "0.004210"
    assert rows["R2E219"]["display_value"] == "-0.000339"
    assert rows["R2E225"]["display_value"] == "0.005357"
    assert rows["R2P002"]["disposition"] == "prohibited"


def test_digest_specific_approval_authorizes_only_rewrite_scope() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    digest = (OUTPUT_DIR / "ROUND2_CLAIM_DIGEST.txt").read_text(encoding="utf-8").strip()
    approval = json.loads((OUTPUT_DIR / "approval_record.json").read_text(encoding="utf-8"))
    assert len(digest) == 64
    assert config["status"] == "approved_for_rewrite"
    assert config["user_approval"]["status"] == "approved"
    assert approval["claim_set_sha256"] == digest
    assert approval["approved_claim_set_sha256"] == digest
    assert approval["manuscript_editing_authorized"] is True
    assert config["controls"]["release_authorized"] is False
    assert config["controls"]["tag_creation_authorized"] is False
    assert config["controls"]["doi_minting_authorized"] is False
    assert config["controls"]["raw_data_publication_authorized"] is False


def test_closed_world_package_rejects_tampering(tmp_path: Path) -> None:
    copy = tmp_path / "claim_package"
    shutil.copytree(OUTPUT_DIR, copy)
    (copy / "ROUND2_CLAIM_DIGEST.txt").write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(Round2ClaimMatrixError, match="derived file drifted"):
        validate_round2_claim_matrix(copy)
