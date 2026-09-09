from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.governance import round2_manuscript_v4 as revision


ROOT = Path(__file__).resolve().parents[1]


def _sources() -> tuple[str, str, str]:
    markdown = (ROOT / "manuscript/mdpi_information/main.md").read_text(encoding="utf-8")
    tex = (ROOT / "manuscript/mdpi_information/main.tex").read_text(encoding="utf-8")
    bibliography = (ROOT / "manuscript/mdpi_information/references.bib").read_text(encoding="utf-8")
    return markdown, tex, bibliography


def test_round2_manuscript_matches_approved_boundary() -> None:
    markdown, tex, bibliography = _sources()
    report = revision.validate_manuscript(markdown, tex, bibliography)

    assert report["claim_set_sha256"] == revision.CLAIM_DIGEST
    assert report["active_claim_count"] == 117
    assert report["active_numerical_claim_count"] == 99
    assert report["citation_count"] == 34
    assert report["figure_count"] == 7
    assert report["table_count"] == 11
    assert report["visible_claim_id_count"] == 0
    assert report["release_authorized"] is False
    assert report["submission_ready"] is False


def test_round2_generated_latex_is_exact() -> None:
    markdown, tex, _ = _sources()
    assert revision.generate_tex(markdown) == tex
    assert tex.count(r"\begin{figure}") == 7
    assert tex.count(r"\begin{table}") == 11
    assert "round2_manuscript_v4.py" in tex.splitlines()[0]
    assert "@@" not in tex


def test_round2_package_revalidates_closed_world() -> None:
    receipt = revision.validate_package()
    assert receipt["package_file_count"] == 11
    assert receipt["paid_api_calls"] == 0
    assert receipt["status"] == "scientifically_revised_author_actions_open"

    package = ROOT / "reports/research_log/major_revision_round2/round2_manuscript"
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) == 10
    assert manifest["claim_set_sha256"] == revision.CLAIM_DIGEST


def test_supplementary_ledger_contains_every_active_claim() -> None:
    ledger = ROOT / "reports/research_log/major_revision_round2/SUPPLEMENTARY_EVIDENCE_LEDGER_ROUND2.csv"
    with ledger.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 117
    assert sum(row["claim_type"] == "numerical" for row in rows) == 99
    assert all(row["active_for_rewrite"] == "True" for row in rows)
    assert all(row["approval_status"] == "approved" for row in rows)
    assert all(row["source_sha256"] for row in rows)


def test_visible_claim_code_is_rejected() -> None:
    markdown, _, bibliography = _sources()
    changed = markdown.replace("The INX analysis", "The INX analysis (R2N009)", 1)
    with pytest.raises(revision.Round2ManuscriptError, match="Visible claim IDs"):
        revision.validate_manuscript(changed, revision.generate_tex(changed), bibliography)


def test_extreme_class_boundary_is_required() -> None:
    markdown, _, bibliography = _sources()
    changed = markdown.replace(
        "Aggregate QWK or MAE improvement therefore does not guarantee class-4 success",
        "Aggregate metrics require inspection",
        1,
    )
    with pytest.raises(revision.Round2ManuscriptError, match="Required Round 2 statements"):
        revision.validate_manuscript(changed, revision.generate_tex(changed), bibliography)


def test_unverified_round2_citation_is_rejected() -> None:
    markdown, _, bibliography = _sources()
    changed = markdown.replace("[@mccullagh1980regression]", "[@unverified2026; @mccullagh1980regression]", 1)
    with pytest.raises(revision.Round2ManuscriptError, match="Every verified bibliography"):
        revision.validate_manuscript(changed, revision.generate_tex(changed), bibliography)
