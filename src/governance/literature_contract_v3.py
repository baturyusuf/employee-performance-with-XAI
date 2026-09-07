"""Validate and publish the source-verified Phase 4A literature contract."""

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
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CONTRACT = Path("configs/literature_positioning_v3.json")
DEFAULT_OUTPUT = Path("reports/research_log/major_revision_v3/phase4a_literature")
MANIFEST_NAME = "manifest.json"
EXPECTED_OUTPUTS = frozenset(
    {
        "README.md",
        "LITERATURE_SEARCH_PROTOCOL.md",
        "VERIFIED_BIBLIOGRAPHY.md",
        "NOVELTY_POSITIONING.md",
        "literature_evidence_log.csv",
        "positioning_table.csv",
        "excluded_candidates.csv",
        "provenance_receipt.json",
        MANIFEST_NAME,
    }
)
POSITIONING_FIELDS = (
    "employee_performance",
    "same_or_similar_dataset",
    "leakage_policy",
    "nested_cv",
    "calibration",
    "exact_fold_explanations",
    "shap_stability",
    "subgroup_analysis",
    "proxy_reconstruction",
    "second_dataset",
    "artifact_provenance",
)
REQUIRED_ROLES = frozenset(
    {
        "same_dataset_empirical",
        "similar_hr_empirical",
        "hr_governance",
        "xai_validity",
        "evaluation_calibration",
        "provenance_reproducibility",
    }
)
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LiteratureContractV3Error(RuntimeError):
    """Raised when the literature evidence contract is incomplete or unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LiteratureContractV3Error(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return stream.getvalue().encode("utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _git_is_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return not result.stdout.strip()


def _load_contract(contract_path: Path | str) -> tuple[Path, dict[str, Any]]:
    path = Path(contract_path)
    _require(path.is_file(), f"Literature contract is absent: {path.as_posix()}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "Literature contract root must be an object.")
    return path, payload


def validate_literature_contract_v3(
    contract_path: Path | str = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Validate source identity, matrix semantics, scope, and bounded novelty."""

    path, payload = _load_contract(contract_path)
    _require(payload.get("schema_version") == 1, "Literature schema version drifted.")
    _require(payload.get("phase") == "4A", "Literature phase identity drifted.")
    _require(payload.get("retrieval_date") == "2026-09-07", "Literature retrieval date drifted.")
    protocol = payload.get("search_protocol")
    _require(isinstance(protocol, dict), "Literature search protocol is absent.")
    _require(protocol.get("paid_api_calls") == 0, "Paid API policy drifted.")
    query_families = protocol.get("query_families")
    _require(isinstance(query_families, list) and len(query_families) >= 8, "Literature query coverage is incomplete.")
    authorities = protocol.get("preferred_authorities")
    _require(isinstance(authorities, list) and len(authorities) >= 4, "Literature source-authority order is incomplete.")

    vocabulary = payload.get("status_vocabulary")
    expected_vocabulary = ["yes", "no", "partial", "not_reported", "not_applicable"]
    _require(vocabulary == expected_vocabulary, "Positioning status vocabulary drifted.")
    definitions = payload.get("positioning_field_definitions")
    _require(isinstance(definitions, dict), "Positioning field definitions are absent.")
    _require(tuple(definitions) == POSITIONING_FIELDS, "Positioning field order or identity drifted.")

    studies = payload.get("studies")
    _require(isinstance(studies, list), "Literature studies are absent.")
    _require(15 <= len(studies) <= 25, "Literature set must contain 15 to 25 close studies.")
    _require(len(studies) == 25, "The frozen Phase 4A positioning set must contain 25 studies.")
    expected_ids = [f"S{index:02d}" for index in range(1, len(studies) + 1)]
    _require([study.get("study_id") for study in studies] == expected_ids, "Study IDs or order drifted.")

    required_study_fields = {
        "study_id",
        "citation_key",
        "authors",
        "year",
        "title",
        "venue",
        "publication_type",
        "doi",
        "doi_status",
        "primary_url",
        "metadata_sources",
        "evidence_role",
        "dataset_match",
        "verification_basis",
        "evidence_anchor",
        "flags",
        "assessment_note",
    }
    citation_keys: set[str] = set()
    dois: set[str] = set()
    roles: Counter[str] = Counter()
    dataset_matches: Counter[str] = Counter()
    flag_counts = {field: Counter() for field in POSITIONING_FIELDS}
    for study in studies:
        study_id = str(study.get("study_id"))
        _require(set(study) == required_study_fields, f"{study_id} field inventory drifted.")
        citation_key = study.get("citation_key")
        _require(isinstance(citation_key, str) and citation_key, f"{study_id} citation key is absent.")
        _require(citation_key not in citation_keys, f"Duplicate citation key: {citation_key}.")
        citation_keys.add(citation_key)
        _require(isinstance(study.get("year"), int) and 2000 <= study["year"] <= 2026, f"{study_id} year is invalid.")
        for field in ("authors", "title", "venue", "publication_type", "dataset_match", "verification_basis", "evidence_anchor", "assessment_note"):
            _require(isinstance(study.get(field), str) and study[field].strip(), f"{study_id} {field} is absent.")
        primary_url = study.get("primary_url")
        _require(isinstance(primary_url, str) and primary_url.startswith("https://"), f"{study_id} primary URL is not HTTPS.")
        sources = study.get("metadata_sources")
        _require(isinstance(sources, list) and sources, f"{study_id} metadata sources are absent.")
        _require(all(isinstance(url, str) and url.startswith("https://") for url in sources), f"{study_id} metadata source is not HTTPS.")
        role = study.get("evidence_role")
        _require(role in REQUIRED_ROLES, f"{study_id} evidence role is invalid.")
        roles[role] += 1
        dataset_matches[str(study["dataset_match"])] += 1

        doi = study.get("doi")
        doi_status = study.get("doi_status")
        _require(doi_status in {"registered", "publisher_asserted_unresolved", "not_assigned"}, f"{study_id} DOI status is invalid.")
        if doi_status == "registered":
            _require(isinstance(doi, str) and DOI_RE.fullmatch(doi), f"{study_id} registered DOI is invalid.")
            _require(any("api.crossref.org/works/" in url for url in sources), f"{study_id} registered DOI lacks Crossref verification.")
        elif doi_status == "publisher_asserted_unresolved":
            _require(isinstance(doi, str) and DOI_RE.fullmatch(doi), f"{study_id} asserted DOI is invalid.")
            _require("unresolved" in study["verification_basis"].lower() or "404" in study["verification_basis"], f"{study_id} unresolved DOI caveat is absent.")
        else:
            _require(doi is None, f"{study_id} unassigned DOI must be null.")
        if doi is not None:
            normalized_doi = doi.lower()
            _require(normalized_doi not in dois, f"Duplicate DOI: {doi}.")
            dois.add(normalized_doi)

        flags = study.get("flags")
        _require(isinstance(flags, dict), f"{study_id} flags are absent.")
        _require(tuple(flags) == POSITIONING_FIELDS, f"{study_id} positioning field order or identity drifted.")
        for field, status in flags.items():
            _require(status in vocabulary, f"{study_id} {field} status is invalid: {status}.")
            flag_counts[field][status] += 1

    _require(set(roles) == REQUIRED_ROLES, "Literature evidence-role coverage is incomplete.")
    _require(dataset_matches["exact_inx"] == 3, "Exact-INX primary-study count drifted.")
    _require(dataset_matches["exact_inx_secondary_dataset"] == 1, "Exact-INX secondary-study count drifted.")
    _require(dataset_matches["exact_hrdataset_v14"] == 1, "Exact-HRDataset_v14 study count drifted.")
    _require(sum(1 for study in studies if study["doi_status"] == "publisher_asserted_unresolved") == 1, "Unresolved DOI audit count drifted.")
    _require(flag_counts["exact_fold_explanations"]["yes"] == 0, "Prior exact-fold explanation count drifted.")
    _require(flag_counts["artifact_provenance"]["yes"] == 0, "Prior complete artifact-provenance count drifted.")

    novelty = payload.get("novelty_boundary")
    _require(isinstance(novelty, dict), "Novelty boundary is absent.")
    _require("Within this 25-work" in novelty.get("bounded_claim", ""), "Novelty claim is not bounded to the screened set.")
    _require("shared" in novelty.get("mechanism", "").lower(), "Shared evidence-contract mechanism is absent.")
    prevented = novelty.get("prevented_failures")
    prohibited = novelty.get("prohibited_claims")
    _require(isinstance(prevented, list) and len(prevented) >= 6, "Mechanistic failure coverage is incomplete.")
    _require(isinstance(prohibited, list) and len(prohibited) >= 6, "Prohibited novelty/impact claims are incomplete.")
    excluded = payload.get("excluded_candidates")
    _require(isinstance(excluded, list) and len(excluded) >= 5, "Candidate exclusion log is incomplete.")
    for candidate in excluded:
        _require(set(candidate) == {"candidate", "url", "decision", "reason"}, "Excluded-candidate schema drifted.")
        _require(candidate["url"].startswith("https://"), "Excluded-candidate URL is not HTTPS.")
        _require(candidate["decision"].startswith("excluded_"), "Excluded-candidate decision is invalid.")

    full_text = json.dumps(payload, ensure_ascii=False).lower()
    for claim in ("all leakage is eliminated", "fairness is established", "deployment readiness is established"):
        _require(full_text.count(claim) == 1, f"Prohibited claim escaped its explicit prohibition list: {claim}.")

    return {
        "status": "passed",
        "schema_version": 1,
        "contract_path": path.as_posix(),
        "contract_sha256": _sha256(path),
        "retrieval_date": payload["retrieval_date"],
        "study_count": len(studies),
        "registered_doi_count": sum(study["doi_status"] == "registered" for study in studies),
        "publisher_asserted_unresolved_doi_count": sum(study["doi_status"] == "publisher_asserted_unresolved" for study in studies),
        "no_assigned_doi_count": sum(study["doi_status"] == "not_assigned" for study in studies),
        "exact_inx_study_count": dataset_matches["exact_inx"] + dataset_matches["exact_inx_secondary_dataset"],
        "exact_hrdataset_v14_study_count": dataset_matches["exact_hrdataset_v14"],
        "role_counts": dict(sorted(roles.items())),
        "flag_counts": {field: dict(sorted(counts.items())) for field, counts in flag_counts.items()},
        "excluded_candidate_count": len(excluded),
        "paid_api_calls": 0,
    }


def _evidence_rows(studies: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for study in studies:
        rows.append(
            {
                "study_id": study["study_id"],
                "citation_key": study["citation_key"],
                "authors": study["authors"],
                "year": study["year"],
                "title": study["title"],
                "venue": study["venue"],
                "publication_type": study["publication_type"],
                "doi": study["doi"] or "",
                "doi_status": study["doi_status"],
                "primary_url": study["primary_url"],
                "metadata_sources": " | ".join(study["metadata_sources"]),
                "evidence_role": study["evidence_role"],
                "dataset_match": study["dataset_match"],
                "verification_basis": study["verification_basis"],
                "evidence_anchor": study["evidence_anchor"],
                "assessment_note": study["assessment_note"],
            }
        )
    return rows


def _positioning_rows(studies: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for study in studies:
        row = {
            "study_id": study["study_id"],
            "study": f"{study['authors'].split(';')[0]} ({study['year']})",
            "title": study["title"],
            "dataset_match": study["dataset_match"],
        }
        row.update(study["flags"])
        rows.append(row)
    return rows


def _readme(payload: Mapping[str, Any], generation_commit: str) -> str:
    return "\n".join(
        [
            "# Phase 4A Verified Literature Package",
            "",
            f"Evidence-generation commit: `{generation_commit}`",
            f"Online retrieval cutoff: `{payload['retrieval_date']}`",
            "",
            "This package freezes a bounded 25-work positioning set for later claim-matrix and manuscript work. It contains source identity, method-applicability judgments, exact/similar-dataset prior art, exclusions, and a mechanistic novelty boundary. It does not modify the manuscript or its bibliography.",
            "",
            "The package does not claim an exhaustive systematic review or a world-first contribution. `not_reported` means the requested design fact was not visible in the accessible primary evidence; it is not converted to `no`. One publisher-asserted 2019 DOI did not resolve in Crossref on the retrieval date and remains explicitly qualified.",
            "",
            "## Contents",
            "",
            "- `LITERATURE_SEARCH_PROTOCOL.md`: search scope, authorities, operational definitions, and DOI rule.",
            "- `literature_evidence_log.csv`: complete verified bibliographic and assessment evidence.",
            "- `positioning_table.csv`: requested cross-study capability matrix.",
            "- `VERIFIED_BIBLIOGRAPHY.md`: human-readable source list with DOI status.",
            "- `NOVELTY_POSITIONING.md`: bounded shared-evidence-contract contribution and prohibited overclaims.",
            "- `excluded_candidates.csv`: transparent screening exclusions.",
            "- `provenance_receipt.json` and `manifest.json`: source-contract hash, generation identity, controls, and file hashes.",
        ]
    ) + "\n"


def _search_protocol(payload: Mapping[str, Any]) -> str:
    protocol = payload["search_protocol"]
    definitions = payload["positioning_field_definitions"]
    lines = [
        "# Literature Search and Verification Protocol",
        "",
        f"Retrieval cutoff: `{payload['retrieval_date']}`",
        "",
        "## Objective",
        "",
        "Identify close employee-performance/HR-XAI studies and the methodological sources needed to evaluate leakage, nested selection, calibration, explanation validity, responsible HR inference, and reproducible artifact lineage. The target is a bounded positioning set, not an exhaustive meta-analysis.",
        "",
        "## Query families",
        "",
    ]
    lines.extend(f"- {item}" for item in protocol["query_families"])
    lines.extend(["", "## Source-authority order", ""])
    lines.extend(f"- {item}" for item in protocol["preferred_authorities"])
    lines.extend(
        [
            "",
            "Discovery results from general search, GitHub, resumes, portfolios, ResearchGate, and index aggregators were used only to locate primary records. A work entered the positioning set only after its identity was checked against a publisher/proceedings record, Crossref, or an authoritative repository. No paid API was called.",
            "",
            "## Selection and missing-evidence rules",
            "",
            protocol["selection_rule"],
            "",
            protocol["access_rule"],
            "",
            protocol["doi_rule"],
            "",
            "The 2019 IJCSE INX article is retained because the publisher record and PDF are direct prior-art evidence. Its printed DOI did not resolve in Crossref on the cutoff date, so downstream citation work must use the verified publisher URL and retain the unresolved-DOI caveat unless the identifier is later independently resolved.",
            "",
            "## Operational positioning fields",
            "",
        ]
    )
    for field, definition in definitions.items():
        lines.append(f"- `{field}`: {definition}")
    lines.extend(
        [
            "",
            "Statuses are `yes`, `no`, `partial`, `not_reported`, and `not_applicable`. `partial` records a related but narrower mechanism. `not_applicable` is used when a non-empirical method/governance paper cannot fairly be judged as an application study.",
            "",
            "## Search result boundary",
            "",
            "The selected set contains four exact-INX studies (three primary uses and one secondary-dataset use), one exact-HRDataset_v14 study, four other close HR empirical/XAI studies, four HR governance/review sources, five XAI-validity sources, five evaluation/leakage/calibration sources, and two reporting/reproducibility sources. The exclusion log preserves close candidates that were displaced by the 25-work cap or could not be verified deeply enough.",
        ]
    )
    return "\n".join(lines) + "\n"


def _bibliography(studies: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Verified Bibliography",
        "",
        "Entries use the publisher/proceedings year. DOI links are shown only as registered when Crossref returned the exact work on 2026-09-07. This is a verification ledger, not yet the manuscript bibliography.",
        "",
    ]
    for study in studies:
        if study["doi_status"] == "registered":
            identifier = f"Registered DOI: https://doi.org/{study['doi']}"
        elif study["doi_status"] == "publisher_asserted_unresolved":
            identifier = f"Publisher-asserted DOI, unresolved in Crossref on cutoff: {study['doi']}"
        else:
            identifier = "No DOI assigned in the verified primary record"
        lines.extend(
            [
                f"## {study['study_id']} — {study['title']}",
                "",
                f"{study['authors']} ({study['year']}). {study['venue']}.",
                "",
                f"- {identifier}",
                f"- Primary record: {study['primary_url']}",
                f"- Verification: {study['verification_basis']}",
                "",
            ]
        )
    return "\n".join(lines)


def _novelty(payload: Mapping[str, Any], studies: Sequence[Mapping[str, Any]]) -> str:
    novelty = payload["novelty_boundary"]
    empirical = [study for study in studies if study["evidence_role"] in {"same_dataset_empirical", "similar_hr_empirical"}]
    capabilities = {
        field: sum(study["flags"][field] == "yes" for study in studies)
        for field in POSITIONING_FIELDS
    }
    empirical_capabilities = {
        field: sum(study["flags"][field] == "yes" for study in empirical)
        for field in POSITIONING_FIELDS
    }
    lines = [
        "# Bounded Novelty Positioning",
        "",
        "## Claim",
        "",
        novelty["bounded_claim"],
        "",
        "This is a bounded comparison statement about the frozen positioning set. It is not a claim that no unobserved publication has related components, and it is not a checklist-based world-first claim.",
        "",
        "## What the prior literature establishes",
        "",
        "The evidence is distributed across separate traditions: same-dataset employee-performance prediction, HR governance and fairness scholarship, SHAP algorithms and explanation-validity tests, leakage/nested-evaluation methods, probability-calibration methods, and reproducibility/reporting frameworks. Those sources individually justify the components used here.",
        "",
        f"Across all {len(studies)} selected works, explicit `yes` counts are: leakage policy {capabilities['leakage_policy']}, nested CV {capabilities['nested_cv']}, calibration {capabilities['calibration']}, exact-fold explanations {capabilities['exact_fold_explanations']}, SHAP stability {capabilities['shap_stability']}, subgroup analysis {capabilities['subgroup_analysis']}, proxy reconstruction {capabilities['proxy_reconstruction']}, second dataset {capabilities['second_dataset']}, and complete artifact provenance {capabilities['artifact_provenance']}.",
        "",
        f"Among the {len(empirical)} closest HR empirical/XAI studies, the corresponding explicit `yes` counts are: leakage policy {empirical_capabilities['leakage_policy']}, nested CV {empirical_capabilities['nested_cv']}, calibration {empirical_capabilities['calibration']}, exact-fold explanations {empirical_capabilities['exact_fold_explanations']}, SHAP stability {empirical_capabilities['shap_stability']}, subgroup analysis {empirical_capabilities['subgroup_analysis']}, proxy reconstruction {empirical_capabilities['proxy_reconstruction']}, second dataset {empirical_capabilities['second_dataset']}, and complete artifact provenance {empirical_capabilities['artifact_provenance']}.",
        "",
        "These counts are descriptive of coded evidence, not quality scores. A `not_reported` value remains epistemically different from `no`.",
        "",
        "## Mechanistic contribution",
        "",
        novelty["mechanism"],
        "",
        "The shared contract matters because it prevents evidence assembled in one analytical stage from silently changing identity before another stage cites it. Specifically, it is designed to prevent:",
        "",
    ]
    lines.extend(f"- {failure}" for failure in novelty["prevented_failures"])
    lines.extend(
        [
            "",
            "## Dataset-specific prior-art implication",
            "",
            "Four verified publications use the exact INX data. The closest registered INX work uses an 80/20 split and a separate validation-curve presentation; another selects the highest accuracy across several train/test proportions. The exact HRDataset_v14 comparator reports a single 80/20 split and does not document removal of the target-alias field `PerfScoreID`. These are design differences, not grounds to dismiss the studies; they explain why the present contribution must be stated as an evidence-identity mechanism rather than another accuracy comparison.",
            "",
            "## Prohibited overclaims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in novelty["prohibited_claims"])
    lines.extend(
        [
            "",
            "Later manuscript prose must also preserve the existing boundaries: SHAP is model attribution rather than causation; subgroup results are descriptive rather than proof of fairness/discrimination; HRDataset_v14 is an independently trained mapped-target sensitivity rather than locked transport; and public source presence does not establish licence or authenticity.",
        ]
    )
    return "\n".join(lines) + "\n"


def export_literature_package_v3(
    contract_path: Path | str = DEFAULT_CONTRACT,
    output_dir: Path | str = DEFAULT_OUTPUT,
    *,
    generation_commit: str | None = None,
    require_clean_git: bool = True,
) -> dict[str, Any]:
    """Create the deterministic aggregate-only Phase 4A package."""

    contract, payload = _load_contract(contract_path)
    validation = validate_literature_contract_v3(contract)
    if generation_commit is None:
        _require(not require_clean_git or _git_is_clean(), "Literature package generation requires a clean Git worktree.")
        generation_commit = _git_head()
    _require(bool(re.fullmatch(r"[0-9a-f]{7,40}|test", generation_commit)), "Generation commit is invalid.")
    destination = Path(output_dir)
    _require(not destination.exists(), f"Literature package destination already exists: {destination.as_posix()}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        studies = payload["studies"]
        evidence_fields = [
            "study_id",
            "citation_key",
            "authors",
            "year",
            "title",
            "venue",
            "publication_type",
            "doi",
            "doi_status",
            "primary_url",
            "metadata_sources",
            "evidence_role",
            "dataset_match",
            "verification_basis",
            "evidence_anchor",
            "assessment_note",
        ]
        positioning_fields = ["study_id", "study", "title", "dataset_match", *POSITIONING_FIELDS]
        exclusion_fields = ["candidate", "url", "decision", "reason"]
        _write_bytes(staging / "README.md", _readme(payload, generation_commit).encode("utf-8"))
        _write_bytes(staging / "LITERATURE_SEARCH_PROTOCOL.md", _search_protocol(payload).encode("utf-8"))
        _write_bytes(staging / "VERIFIED_BIBLIOGRAPHY.md", _bibliography(studies).encode("utf-8"))
        _write_bytes(staging / "NOVELTY_POSITIONING.md", _novelty(payload, studies).encode("utf-8"))
        _write_bytes(staging / "literature_evidence_log.csv", _csv_bytes(evidence_fields, _evidence_rows(studies)))
        _write_bytes(staging / "positioning_table.csv", _csv_bytes(positioning_fields, _positioning_rows(studies)))
        _write_bytes(staging / "excluded_candidates.csv", _csv_bytes(exclusion_fields, payload["excluded_candidates"]))
        provenance = {
            "schema_version": 1,
            "package_kind": "phase4a_source_verified_literature_and_novelty",
            "generation_commit": generation_commit,
            "contract_path": contract.as_posix(),
            "contract_sha256": validation["contract_sha256"],
            "retrieval_date": payload["retrieval_date"],
            "contract_validation": validation,
            "publication_controls": {
                "paid_api_calls": 0,
                "network_calls_during_package_generation": 0,
                "full_text_source_copies_included": False,
                "raw_source_extracts_included": False,
                "manuscript_sources_modified": False,
                "manuscript_bibliography_modified": False,
                "exhaustive_systematic_review_claimed": False,
                "world_first_claimed": False,
                "unresolved_doi_presented_as_registered": False,
                "fairness_or_causal_validity_established": False,
                "source_licence_or_authenticity_established": False,
            },
        }
        _write_bytes(staging / "provenance_receipt.json", _json_bytes(provenance))
        records = [
            {"path": path.name, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in sorted(staging.iterdir())
            if path.is_file()
        ]
        manifest = {
            "schema_version": 1,
            "package_kind": "phase4a_source_verified_literature_and_novelty",
            "generation_commit": generation_commit,
            "contract_sha256": validation["contract_sha256"],
            "file_count_excluding_manifest": len(records),
            "files": records,
        }
        _write_bytes(staging / MANIFEST_NAME, _json_bytes(manifest))
        inventory = {path.name for path in staging.iterdir() if path.is_file()}
        _require(inventory == EXPECTED_OUTPUTS, f"Literature staging inventory drifted: {sorted(inventory ^ EXPECTED_OUTPUTS)}.")
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return validate_literature_package_v3(destination, contract_path=contract)


def validate_literature_package_v3(
    package_dir: Path | str = DEFAULT_OUTPUT,
    *,
    contract_path: Path | str = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Validate closed-world package hashes and exact contract-derived rows."""

    package = Path(package_dir)
    _require(package.is_dir(), f"Literature package is absent: {package.as_posix()}.")
    inventory = {path.name for path in package.iterdir() if path.is_file()}
    _require(inventory == EXPECTED_OUTPUTS, f"Literature package inventory drifted: {sorted(inventory ^ EXPECTED_OUTPUTS)}.")
    _require(not any(path.is_dir() for path in package.iterdir()), "Literature package contains a directory.")
    contract, payload = _load_contract(contract_path)
    validation = validate_literature_contract_v3(contract)
    manifest = json.loads((package / MANIFEST_NAME).read_text(encoding="utf-8"))
    records = manifest.get("files")
    _require(isinstance(records, list), "Literature manifest records are absent.")
    _require(manifest.get("file_count_excluding_manifest") == len(EXPECTED_OUTPUTS) - 1, "Literature manifest count drifted.")
    _require({record.get("path") for record in records} == EXPECTED_OUTPUTS - {MANIFEST_NAME}, "Literature manifest inventory drifted.")
    _require(manifest.get("contract_sha256") == validation["contract_sha256"], "Literature manifest contract hash drifted.")
    _require(SHA256_RE.fullmatch(str(manifest.get("contract_sha256"))) is not None, "Literature manifest contract hash is invalid.")
    for record in records:
        path = package / str(record["path"])
        _require(path.stat().st_size == int(record["size_bytes"]), f"Literature package size drifted for {path.name}.")
        _require(_sha256(path) == record["sha256"], f"Literature package hash drifted for {path.name}.")

    provenance = json.loads((package / "provenance_receipt.json").read_text(encoding="utf-8"))
    _require(provenance.get("contract_sha256") == validation["contract_sha256"], "Literature provenance contract hash drifted.")
    _require(provenance.get("contract_validation") == validation, "Literature provenance validation receipt drifted.")
    _require(provenance.get("retrieval_date") == payload["retrieval_date"], "Literature provenance retrieval date drifted.")
    controls = provenance.get("publication_controls")
    _require(isinstance(controls, dict), "Literature publication controls are absent.")
    _require(controls.get("paid_api_calls") == 0, "Literature package records paid API activity.")
    _require(controls.get("network_calls_during_package_generation") == 0, "Literature package generation records network activity.")
    for field, value in controls.items():
        if field not in {"paid_api_calls", "network_calls_during_package_generation"}:
            _require(value is False, f"Literature publication control drifted: {field}.")

    with (package / "literature_evidence_log.csv").open(encoding="utf-8", newline="") as stream:
        evidence_rows = list(csv.DictReader(stream))
    with (package / "positioning_table.csv").open(encoding="utf-8", newline="") as stream:
        positioning_rows = list(csv.DictReader(stream))
    with (package / "excluded_candidates.csv").open(encoding="utf-8", newline="") as stream:
        exclusion_rows = list(csv.DictReader(stream))
    _require(evidence_rows == [{key: str(value) if value is not None else "" for key, value in row.items()} for row in _evidence_rows(payload["studies"])], "Literature evidence rows drifted from contract.")
    _require(positioning_rows == [{key: str(value) for key, value in row.items()} for row in _positioning_rows(payload["studies"])], "Positioning rows drifted from contract.")
    _require(exclusion_rows == payload["excluded_candidates"], "Excluded-candidate rows drifted from contract.")

    protocol_text = (package / "LITERATURE_SEARCH_PROTOCOL.md").read_text(encoding="utf-8")
    novelty_text = (package / "NOVELTY_POSITIONING.md").read_text(encoding="utf-8")
    bibliography_text = (package / "VERIFIED_BIBLIOGRAPHY.md").read_text(encoding="utf-8")
    for required in (
        "bounded positioning set, not an exhaustive meta-analysis",
        "`not_reported`",
        "did not resolve in Crossref",
        "four exact-INX studies",
    ):
        _require(required.lower() in protocol_text.lower(), f"Literature protocol boundary is absent: {required}.")
    for required in (
        "not a checklist-based world-first claim",
        "explanation and evaluated-model identity mismatch",
        "calibration leakage from held-out outcomes",
        "PerfScoreID",
        "SHAP is model attribution rather than causation",
    ):
        _require(required.lower() in novelty_text.lower(), f"Novelty boundary is absent: {required}.")
    _require(bibliography_text.count("## S") == 25, "Verified bibliography entry count drifted.")
    _require("unresolved in Crossref on cutoff" in bibliography_text, "Unresolved DOI warning is absent.")

    return {
        "status": "passed",
        "package_dir": package.as_posix(),
        "generation_commit": manifest["generation_commit"],
        "contract_sha256": validation["contract_sha256"],
        "manifest_sha256": _sha256(package / MANIFEST_NAME),
        "file_count": len(EXPECTED_OUTPUTS),
        "size_bytes": sum(path.stat().st_size for path in package.iterdir() if path.is_file()),
        "study_count": len(evidence_rows),
        "positioning_row_count": len(positioning_rows),
        "excluded_candidate_count": len(exclusion_rows),
        "paid_api_calls": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-contract-only", action="store_true")
    parser.add_argument("--validate-package-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require(not (args.validate_contract_only and args.validate_package_only), "Choose only one validation mode.")
    if args.validate_contract_only:
        receipt = validate_literature_contract_v3(args.contract)
    elif args.validate_package_only:
        receipt = validate_literature_package_v3(args.output_dir, contract_path=args.contract)
    else:
        receipt = export_literature_package_v3(args.contract, args.output_dir)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONTRACT",
    "DEFAULT_OUTPUT",
    "EXPECTED_OUTPUTS",
    "LiteratureContractV3Error",
    "POSITIONING_FIELDS",
    "export_literature_package_v3",
    "validate_literature_contract_v3",
    "validate_literature_package_v3",
]
