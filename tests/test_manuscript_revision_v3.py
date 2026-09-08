from __future__ import annotations

import json
import csv
import re
from pathlib import Path

import pytest

from src.governance import manuscript_revision_v3 as revision


ROOT = Path(__file__).resolve().parents[1]


def test_current_manuscript_matches_approved_claim_boundary() -> None:
    claims = revision.load_approved_claims()
    markdown = (ROOT / "manuscript/mdpi_information/main.md").read_text(encoding="utf-8")
    tex = revision.markdown_to_tex(markdown)

    report = revision.validate_manuscript(markdown, tex, claims)

    assert report["claim_digest"] == revision.CLAIM_DIGEST
    assert report["approved_claim_count"] == 45
    assert report["numeric_claim_ids_traced"] == 32
    assert report["verified_citation_count_used"] == 25
    assert report["figure_count"] == 7
    assert report["table_count"] == 8


def test_generated_latex_preserves_structural_parity() -> None:
    markdown = (ROOT / "manuscript/mdpi_information/main.md").read_text(encoding="utf-8")
    generated = revision.markdown_to_tex(markdown)
    persisted = (ROOT / "manuscript/mdpi_information/main.tex").read_text(encoding="utf-8")

    assert generated == persisted
    assert generated.count(r"\begin{figure}") == 7
    assert generated.count(r"\begin{table}") == 8
    assert "\\subsection{2.1" not in generated
    assert "@@" not in generated


def test_obsolete_v1_concept_is_rejected() -> None:
    claims = revision.load_approved_claims()
    markdown = (ROOT / "manuscript/mdpi_information/main.md").read_text(encoding="utf-8")
    changed = markdown.replace("The intended use", "A chatbot was evaluated. The intended use", 1)

    with pytest.raises(revision.ManuscriptRevisionError, match="Obsolete v1 concepts"):
        revision.validate_manuscript(changed, revision.markdown_to_tex(changed), claims)


def test_unverified_citation_is_rejected() -> None:
    claims = revision.load_approved_claims()
    markdown = (ROOT / "manuscript/mdpi_information/main.md").read_text(encoding="utf-8")
    changed = markdown.replace("[@tambe2019artificial", "[@unverified2026; @tambe2019artificial", 1)

    with pytest.raises(revision.ManuscriptRevisionError, match="Unverified citation"):
        revision.validate_manuscript(changed, revision.markdown_to_tex(changed), claims)


def test_results_comparison_is_numeric_claim_complete() -> None:
    path = ROOT / "reports/research_log/major_revision_v3/phase5b_manuscript/RESULTS_COMPARISON.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {
        row["claim_id"]
        for row in revision.load_approved_claims()
        if row["claim_type"] == "numerical"
    }

    assert len(rows) == 32
    assert {row["claim_id"] for row in rows} == expected
    assert all(row["source_sha256"] for row in rows)


def test_bibliography_uses_only_frozen_verified_keys() -> None:
    literature = json.loads((ROOT / "configs/literature_positioning_v3.json").read_text(encoding="utf-8"))
    expected = {study["citation_key"] for study in literature["studies"]}
    bibliography = (ROOT / "manuscript/mdpi_information/references.bib").read_text(encoding="utf-8")
    observed = {
        line.split("{", 1)[1].rstrip(",")
        for line in bibliography.splitlines()
        if line.startswith("@misc{")
    }

    assert observed == expected
    assert "10.26438/ijcse/v7si14.443447" not in bibliography


def test_final_assets_match_revised_manuscript_scope() -> None:
    package = ROOT / "reports/research_log/major_revision_v3/phase5b_manuscript"
    tables = list((package / "FINAL_TABLES").glob("*"))
    figures = list((package / "FINAL_FIGURES").glob("*"))
    searchable = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in [*tables, *figures]
        if path.suffix.lower() in {".md", ".csv", ".json", ".svg", ".txt"}
    ).lower()

    assert len(tables) == 24  # CSV, preview, and source map for each of eight tables.
    assert len(figures) == 28  # PNG, SVG, caption, and alt text for each of seven figures.
    assert "chatbot" not in searchable
    assert "multi-agent" not in searchable
    assert "counterfactual" not in searchable


def test_revision_diff_cannot_publish_api_key_pattern() -> None:
    path = ROOT / "reports/research_log/major_revision_v3/phase5b_manuscript/ORIGINAL_VS_REVISED.diff"
    assert re.search(r"sk-[A-Za-z0-9_-]{20,}", path.read_text(encoding="utf-8")) is None
