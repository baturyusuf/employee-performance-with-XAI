"""Generate and validate the approved Round 2 manuscript package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.governance.manuscript_revision_v3 import markdown_to_tex
from src.governance.round2_claim_matrix_v4 import validate_round2_claim_matrix


ROOT = Path(__file__).resolve().parents[2]
ROUND2_DIR = ROOT / "reports/research_log/major_revision_round2"
CLAIM_DIR = ROUND2_DIR / "round2_claim_matrix"
PACKAGE_DIR = ROUND2_DIR / "round2_manuscript"
MANUSCRIPT_DIR = ROOT / "manuscript/mdpi_information"
MARKDOWN_PATH = MANUSCRIPT_DIR / "main.md"
TEX_PATH = MANUSCRIPT_DIR / "main.tex"
BIB_PATH = MANUSCRIPT_DIR / "references.bib"
LEDGER_PATH = ROUND2_DIR / "SUPPLEMENTARY_EVIDENCE_LEDGER_ROUND2.csv"
CLAIM_DIGEST = "751208d036597461606bd02d68bfa9e642a4aa5ecf5df2df3fd7e46b99084aea"
ASSEMBLY_BASE_COMMIT = "08619b2ef7f6394b48bf02586a9161188fcd26c6"

REVIEWER_PATH = ROUND2_DIR / "REVIEWER_RESPONSE_ROUND2.md"
SIMULATION_PATH = ROUND2_DIR / "FINAL_REVIEW_SIMULATION_ROUND2.md"
AUTHOR_ACTION_PATH = ROUND2_DIR / "AUTHOR_ACTION_REQUIRED.md"
REVISION_REPORT_PATH = ROUND2_DIR / "ROUND2_REVISION_REPORT.md"
CORE_REFERENCE_PATH = ROUND2_DIR / "LITERATURE_V4/CORE_METHOD_REFERENCES.csv"
HISTORICAL_LITERATURE_CONFIG = ROOT / "configs/literature_positioning_v3.json"

PACKAGE_OUTPUTS = frozenset(
    {
        "README.md",
        "SUPPLEMENTARY_EVIDENCE_LEDGER_ROUND2.csv",
        "MANUSCRIPT_CLAIM_TRACE.csv",
        "MANUSCRIPT_VALIDATION.json",
        "ORIGINAL_VS_REVISED.diff",
        "REVIEWER_RESPONSE_ROUND2.md",
        "FINAL_REVIEW_SIMULATION_ROUND2.md",
        "AUTHOR_ACTION_REQUIRED.md",
        "ROUND2_REVISION_REPORT.md",
        "provenance_receipt.json",
        "manifest.json",
    }
)


class Round2ManuscriptError(RuntimeError):
    """Raised when the approved Round 2 manuscript contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Round2ManuscriptError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalized_text_bytes(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _csv_bytes(fieldnames: Sequence[str], rows: Iterable[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _validate_claim_boundary() -> dict[str, Any]:
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        receipt = validate_round2_claim_matrix(CLAIM_DIR)
    finally:
        os.chdir(previous)
    _require(receipt["status"] == "passed_approved_for_rewrite", "Round 2 claim boundary is not approved.")
    _require(receipt["claim_set_sha256"] == CLAIM_DIGEST, "Round 2 claim digest drifted.")
    _require(receipt["manuscript_editing_authorized"] is True, "Manuscript editing is not authorized.")
    _require(receipt["bibliography_editing_authorized"] is True, "Bibliography editing is not authorized.")
    _require(receipt["reviewer_response_editing_authorized"] is True, "Reviewer response is not authorized.")
    _require(receipt["paid_api_calls"] == 0, "Paid API calls are prohibited.")
    return receipt


def _active_claims() -> tuple[list[str], list[dict[str, str]]]:
    path = CLAIM_DIR / "ROUND2_CLAIM_MATRIX.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader if row["active_for_rewrite"] == "True"]
    _require(len(rows) == 117, f"Expected 117 active claims, found {len(rows)}.")
    _require(all(row["approval_status"] == "approved" for row in rows), "An active claim is not approved.")
    _require(sum(row["claim_type"] == "numerical" for row in rows) == 99, "Active numerical claim count drifted.")
    return fieldnames, rows


def generate_tex(markdown: str) -> str:
    generated = markdown_to_tex(markdown)
    return generated.replace(
        "src/governance/manuscript_revision_v3.py",
        "src/governance/round2_manuscript_v4.py",
        1,
    )


def _citation_keys(markdown: str) -> set[str]:
    return {
        key
        for citation in re.findall(r"\[(?:@[^\]]+)\]", markdown)
        for key in re.findall(r"@([A-Za-z0-9_:-]+)", citation)
    }


def _allowed_reference_keys() -> set[str]:
    historical = json.loads(HISTORICAL_LITERATURE_CONFIG.read_text(encoding="utf-8"))
    old = {row["citation_key"] for row in historical["studies"]}
    new = {row["citation_key"] for row in _read_csv(CORE_REFERENCE_PATH)}
    _require(len(old) == 25 and len(new) == 9 and old.isdisjoint(new), "Reference registry drifted.")
    return old | new


def _bibliography_keys(bibliography: str) -> list[str]:
    return re.findall(r"^@[A-Za-z]+\{([^,]+),", bibliography, flags=re.MULTILINE)


def validate_manuscript(markdown: str, tex: str, bibliography: str) -> dict[str, Any]:
    required_headings = (
        "## 3. Materials and Methods",
        "### 3.5 Nested benchmark and selection-objective sensitivity",
        "### 3.11 HR target mapping, CV design, and target-alias sensitivity",
        "### 4.1 Metric-specific benchmark leaders and extreme-class behavior",
        "### 4.2 Selection-objective sensitivity",
        "### 4.4 P3→P4 timing/information sensitivity",
        "### 4.6 Subgroup results across all prespecified attributes",
        "### 4.8 HR target-mapping and CV-design sensitivity",
        "### 4.9 HR target-alias and data-quality sensitivity",
        "### 5.1 Selection objective and extreme-class failure",
        "## 6. Limitations",
        "## 7. Conclusions",
    )
    missing = [heading for heading in required_headings if heading not in markdown]
    _require(not missing, f"Required Round 2 headings are missing: {missing}.")

    required_fragments = (
        "37 of 60",
        "6 of 9",
        "all 9",
        "rating-4 recall 0.0000",
        "rating-4 recall was only 0.0076",
        "−0.2094",
        "−0.3581",
        "−0.1937",
        "−0.3308",
        "0.2179/0.4388/0.1193",
        "31/243/37",
        "13/18/243/37",
        "11 of 14",
        "Canonical OOF restricted, fit-free",
        "Exclusion and refit",
        "Aggregate QWK or MAE improvement therefore does not guarantee class-4 success",
    )
    absent = [fragment for fragment in required_fragments if fragment not in markdown]
    _require(not absent, f"Required Round 2 statements are missing: {absent}.")

    abstract = markdown.split("## Abstract", 1)[1].split("**Keywords:**", 1)[0]
    for fragment in (
        "selection",
        "rating-4 recall",
        "P3",
        "P4",
        "six prespecified subgroup attributes",
        "three-class mapping",
        "four-class estimand",
        "11 of 14",
    ):
        _require(fragment in abstract, f"Abstract does not expose required Round 2 topic: {fragment}.")

    visible_claim_ids = re.findall(r"\b(?:C\d{3}|R2[NE]\d{3})\b", markdown)
    _require(not visible_claim_ids, f"Visible claim IDs remain in manuscript: {sorted(set(visible_claim_ids))}.")
    _require(CLAIM_DIGEST not in markdown, "The full Round 2 digest must not appear in ordinary prose.")
    _require("1664b188df14135d3b3de8642de3d080c25a3fbadc440615c2bd04b2f14ddabe" not in markdown, "Historical digest remains in manuscript.")

    _, active_claims = _active_claims()
    normalized_markdown = markdown.replace("−", "-")
    absent_numeric_claims = [
        row["claim_id"]
        for row in active_claims
        if row["claim_type"] == "numerical" and row["display_value"] not in normalized_markdown
    ]
    _require(
        not absent_numeric_claims,
        f"Active numerical display values are absent from the manuscript: {absent_numeric_claims}.",
    )

    prohibited_patterns = (
        r"\b(?:is|are|was|were) leakage-free\b",
        r"\bwe (?:prove|demonstrate) fairness\b",
        r"\bproves? discrimination\b",
        r"\bdeployment-ready\b",
        r"\bworld[- ]first contribution\b",
        r"\btarget formulations? (?:improved|outperformed)\b",
    )
    found = [pattern for pattern in prohibited_patterns if re.search(pattern, markdown, flags=re.IGNORECASE)]
    _require(not found, f"Prohibited overclaim wording remains: {found}.")

    figure_count = markdown.count("![Figure ")
    table_count = markdown.count("**Table ")
    _require(figure_count == 7, f"Expected 7 figures, found {figure_count}.")
    _require(table_count == 11, f"Expected 11 tables, found {table_count}.")
    _require(tex.count(r"\begin{figure}") == figure_count, "Markdown/LaTeX figure parity failed.")
    _require(tex.count(r"\begin{table}") == table_count, "Markdown/LaTeX table parity failed.")
    _require(generate_tex(markdown) == tex, "Persisted main.tex does not match generated Markdown conversion.")
    _require("@@" not in tex, "Unresolved Markdown-to-LaTeX placeholder remains.")

    allowed = _allowed_reference_keys()
    cited = _citation_keys(markdown)
    observed = _bibliography_keys(bibliography)
    _require(len(observed) == len(set(observed)), "Duplicate bibliography keys found.")
    _require(set(observed) == allowed, "Bibliography does not equal the 25+9 verified reference registry.")
    _require(cited == allowed, "Every verified bibliography item must be cited and no unverified item may be cited.")
    _require("10.26438/ijcse/v7si14.443447" not in bibliography, "Known incorrect DOI remains in bibliography.")

    for placeholder in (
        "[AUTHOR TO COMPLETE]",
        "[AUTHOR/INSTITUTION TO COMPLETE",
        "[AUTHOR AND JOURNAL-POLICY REVIEW REQUIRED",
    ):
        _require(placeholder in markdown, f"Required unresolved author placeholder is missing: {placeholder}.")

    return {
        "status": "scientifically_revised_author_actions_open",
        "claim_set_sha256": CLAIM_DIGEST,
        "active_claim_count": 117,
        "active_numerical_claim_count": 99,
        "citation_count": len(cited),
        "figure_count": figure_count,
        "table_count": table_count,
        "visible_claim_id_count": 0,
        "full_digest_in_ordinary_prose": False,
        "markdown_tex_parity": True,
        "paid_api_calls": 0,
        "release_authorized": False,
        "submission_ready": False,
    }


TRACE_SPECS = (
    ("Related Work 2.6", "R2N010", "The contribution is not the novelty of any single component"),
    ("Methods 3.1", "C001,C009", "cross-sectional sensitivity study"),
    ("Methods 3.4–3.6", "R2N009", "Preprocessing is fitted anew"),
    ("Results 4.1", "C101,C102,C103,C104,C105,C106,R2N003,R2E006,R2E007", "Aggregate ordinal performance did not guarantee extreme-class success"),
    ("Results 4.2", "R2N001,R2N002,R2E001,R2E002,R2E003,R2E004,R2E005,R2E008,R2E009,R2E010", "Changing the inner selection objective"),
    ("Results 4.3", "C201,C202,C203,C204,C205", "Across five repeated designs"),
    ("Results 4.4", "C301,C302,C303,R2E011,R2E012,R2E013,R2E014,R2E015,R2E016,R2E017,R2E018,R2E019,R2E020", "The restriction from P3 Primary Leakage-Aware"),
    ("Results 4.5", "C401,C402,C403,C501,C502", "The top five feature families"),
    ("Results 4.6", "R2N005,R2E101,R2E102,R2E103,R2E104,R2E105,R2E106,R2E107,R2E108,R2E109,R2E110,R2E111,R2E112,R2E113,R2E114,R2E115,R2E116,R2E117,R2E118", "At the prespecified support threshold of 30"),
    ("Results 4.7", "C601,C602,C603,C604,C605,C606", "Refitting P3 without JobRole"),
    ("Results 4.8", "C701,C702,C703,R2N006,R2N007,R2E201,R2E202,R2E203,R2E204,R2E205,R2E206,R2E207,R2E208,R2E209,R2E210,R2E211,R2E212,R2E213,R2E214,R2E215,R2E216,R2E217", "Across five repeated designs under the retained three-class mapping"),
    ("Results 4.9", "C802,C803,C804,R2N008,R2E218,R2E219,R2E220,R2E221,R2E222,R2E223,R2E224,R2E225,R2E226,R2E227,R2E228,R2E229", "The two target-text/identifier disagreement rows"),
)


def _trace_bytes(markdown: str, active_ids: set[str]) -> bytes:
    rows: list[dict[str, str]] = []
    for location, joined_ids, anchor in TRACE_SPECS:
        ids = joined_ids.split(",")
        _require(set(ids) <= active_ids, f"Trace uses inactive or unknown claim IDs: {sorted(set(ids) - active_ids)}.")
        _require(anchor in markdown, f"Trace anchor is absent from manuscript: {anchor!r}.")
        rows.append(
            {
                "manuscript_location": location,
                "claim_ids": joined_ids,
                "assertion_anchor": anchor,
                "claim_set_sha256": CLAIM_DIGEST,
            }
        )
    return _csv_bytes(("manuscript_location", "claim_ids", "assertion_anchor", "claim_set_sha256"), rows)


def _revision_diff() -> bytes:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-ext-diff",
            "--unified=3",
            ASSEMBLY_BASE_COMMIT,
            "--",
            "manuscript/mdpi_information/main.md",
            "manuscript/mdpi_information/main.tex",
            "manuscript/mdpi_information/references.bib",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    sanitized = re.sub(r"sk-[A-Za-z0-9_-]{20,}", "[REDACTED_API_KEY_PATTERN]", result.stdout)
    return sanitized.encode("utf-8")


def _readme() -> bytes:
    return (
        "# Approved Round 2 Manuscript Package\n\n"
        "Status: **scientifically revised; author actions and release gates remain open**.\n\n"
        "This compact package binds the manuscript rewrite, bibliography, reviewer response, final review "
        "simulation, and claim trace to the user-approved Round 2 claim set. The Supplementary Evidence "
        "Ledger retains complete claim identifiers, source selectors, exact values, qualifiers, and hashes.\n\n"
        "The package excludes employee-level rows, fold assignments, fitted models, raw data, and any "
        "authorization for release, tags, DOI creation, dataset redistribution, software licensing, ethics/IRB "
        "wording, or author declarations. No paid API call was made.\n"
    ).encode("utf-8")


def _provenance(validation: dict[str, Any]) -> bytes:
    return _json_bytes(
        {
            "schema_version": 1,
            "status": validation["status"],
            "assembly_base_commit": ASSEMBLY_BASE_COMMIT,
            "claim_set_sha256": CLAIM_DIGEST,
            "inputs": {
                "main_md": {"path": MARKDOWN_PATH.relative_to(ROOT).as_posix(), "normalized_text_sha256": _sha256_bytes(_normalized_text_bytes(MARKDOWN_PATH))},
                "main_tex": {"path": TEX_PATH.relative_to(ROOT).as_posix(), "normalized_text_sha256": _sha256_bytes(_normalized_text_bytes(TEX_PATH))},
                "references_bib": {"path": BIB_PATH.relative_to(ROOT).as_posix(), "normalized_text_sha256": _sha256_bytes(_normalized_text_bytes(BIB_PATH))},
                "claim_matrix": {
                    "path": (CLAIM_DIR / "ROUND2_CLAIM_MATRIX.csv").relative_to(ROOT).as_posix(),
                    "sha256": _sha256(CLAIM_DIR / "ROUND2_CLAIM_MATRIX.csv"),
                },
            },
            "controls": {
                "paid_api_calls": 0,
                "release_authorized": False,
                "tag_authorized": False,
                "doi_authorized": False,
                "raw_data_redistribution_authorized": False,
                "software_licence_resolved": False,
                "ethics_irb_resolved": False,
                "author_declarations_resolved": False,
            },
        }
    )


def _package_payloads() -> dict[str, bytes]:
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    tex = TEX_PATH.read_text(encoding="utf-8")
    bibliography = BIB_PATH.read_text(encoding="utf-8")
    validation = validate_manuscript(markdown, tex, bibliography)
    fieldnames, claims = _active_claims()
    ledger = _csv_bytes(fieldnames, claims)
    trace = _trace_bytes(markdown, {row["claim_id"] for row in claims})
    payloads = {
        "README.md": _readme(),
        "SUPPLEMENTARY_EVIDENCE_LEDGER_ROUND2.csv": ledger,
        "MANUSCRIPT_CLAIM_TRACE.csv": trace,
        "MANUSCRIPT_VALIDATION.json": _json_bytes(validation),
        "ORIGINAL_VS_REVISED.diff": _revision_diff(),
        "REVIEWER_RESPONSE_ROUND2.md": _normalized_text_bytes(REVIEWER_PATH),
        "FINAL_REVIEW_SIMULATION_ROUND2.md": _normalized_text_bytes(SIMULATION_PATH),
        "AUTHOR_ACTION_REQUIRED.md": _normalized_text_bytes(AUTHOR_ACTION_PATH),
        "ROUND2_REVISION_REPORT.md": _normalized_text_bytes(REVISION_REPORT_PATH),
        "provenance_receipt.json": _provenance(validation),
    }
    manifest = {
        "schema_version": 1,
        "status": validation["status"],
        "claim_set_sha256": CLAIM_DIGEST,
        "files": {
            name: {"sha256": _sha256_bytes(content), "bytes": len(content)}
            for name, content in sorted(payloads.items())
        },
    }
    payloads["manifest.json"] = _json_bytes(manifest)
    return payloads


def _validate_package_payloads(payloads: dict[str, bytes]) -> dict[str, Any]:
    _require(set(payloads) == PACKAGE_OUTPUTS, "Round 2 manuscript package inventory drifted.")
    manifest = json.loads(payloads["manifest.json"])
    for name, metadata in manifest["files"].items():
        _require(_sha256_bytes(payloads[name]) == metadata["sha256"], f"Manifest hash mismatch: {name}.")
        _require(len(payloads[name]) == metadata["bytes"], f"Manifest size mismatch: {name}.")
    validation = json.loads(payloads["MANUSCRIPT_VALIDATION.json"])
    return {
        **validation,
        "package_dir": PACKAGE_DIR.relative_to(ROOT).as_posix(),
        "package_file_count": len(payloads),
        "ledger_sha256": _sha256_bytes(payloads["SUPPLEMENTARY_EVIDENCE_LEDGER_ROUND2.csv"]),
    }


def build(*, replace: bool = False) -> dict[str, Any]:
    _validate_claim_boundary()
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    TEX_PATH.write_text(generate_tex(markdown), encoding="utf-8", newline="\n")
    payloads = _package_payloads()
    receipt = _validate_package_payloads(payloads)
    LEDGER_PATH.write_bytes(payloads["SUPPLEMENTARY_EVIDENCE_LEDGER_ROUND2.csv"])

    if PACKAGE_DIR.exists():
        _require(replace, f"Package already exists: {PACKAGE_DIR.relative_to(ROOT).as_posix()}.")
        _require(PACKAGE_DIR.resolve().parent == ROUND2_DIR.resolve(), "Refusing unsafe package replacement.")
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True)
    for name, content in payloads.items():
        (PACKAGE_DIR / name).write_bytes(content)
    return validate_package()


def validate_package() -> dict[str, Any]:
    _validate_claim_boundary()
    _require(PACKAGE_DIR.is_dir(), "Round 2 manuscript package is absent.")
    names = {path.name for path in PACKAGE_DIR.iterdir() if path.is_file()}
    _require(names == PACKAGE_OUTPUTS, "Round 2 manuscript package inventory drifted.")
    expected = _package_payloads()
    for name, content in expected.items():
        _require((PACKAGE_DIR / name).read_bytes() == content, f"Round 2 manuscript artifact drifted: {name}.")
    _require(_normalized_text_bytes(LEDGER_PATH) == expected["SUPPLEMENTARY_EVIDENCE_LEDGER_ROUND2.csv"], "Top-level evidence ledger drifted.")
    return _validate_package_payloads(expected)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = validate_package() if args.validate_only else build(replace=args.replace)
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
