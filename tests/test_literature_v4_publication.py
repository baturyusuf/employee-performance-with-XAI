from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.governance.literature_v4_validator import (
    FIXED_NOVELTY,
    LiteratureV4ValidationError,
    validate_literature_v4,
)


PACKAGE = Path("reports/research_log/major_revision_round2/LITERATURE_V4")


def test_literature_v4_package_is_closed_and_valid() -> None:
    receipt = validate_literature_v4(PACKAGE)
    assert receipt["status"] == "passed"
    assert receipt["reference_count"] == 9
    assert receipt["coverage_count"] == 9
    assert receipt["file_count"] == 7
    assert receipt["paid_api_calls"] == 0


def test_every_core_method_has_one_verified_source() -> None:
    references = pd.read_csv(PACKAGE / "CORE_METHOD_REFERENCES.csv", keep_default_na=False)
    assert references["coverage"].is_unique
    assert references["verification_basis"].str.len().gt(40).all()
    assert references["manuscript_use"].str.len().gt(20).all()
    assert references["boundary"].str.len().gt(20).all()
    assert references.loc[references["citation_key"] == "ke2017lightgbm", "doi"].item() == ""


def test_bibliography_is_preapproval_only_and_novelty_is_bounded() -> None:
    readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
    novelty = (PACKAGE / "NOVELTY_BOUNDARY.md").read_text(encoding="utf-8")
    provenance = json.loads((PACKAGE / "provenance_receipt.json").read_text(encoding="utf-8"))
    assert FIXED_NOVELTY in novelty
    assert "not a new exhaustive or systematic review" in readme
    assert provenance["manuscript_bibliography_modified"] is False
    assert provenance["manuscript_modified"] is False


def test_manifest_rejects_payload_tampering(tmp_path: Path) -> None:
    target = tmp_path / "literature"
    target.mkdir()
    for source in PACKAGE.iterdir():
        (target / source.name).write_bytes(source.read_bytes())
    (target / "NOVELTY_BOUNDARY.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(LiteratureV4ValidationError, match="manifest"):
        validate_literature_v4(target)
