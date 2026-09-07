"""Validate and publish the Phase 4B provenance and release-readiness contract."""

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
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CONTRACT = Path("configs/release_readiness_v3.json")
DEFAULT_OUTPUT = Path("reports/research_log/major_revision_v3/phase4b_release_readiness")
MANIFEST_NAME = "manifest.json"
EXPECTED_OUTPUTS = frozenset(
    {
        "README.md",
        "DATA_PROVENANCE_LICENSE_REGISTER.csv",
        "PROVENANCE_LICENSE_REPORT.md",
        "ETHICS_DECLARATIONS_REGISTER.csv",
        "ETHICS_DECLARATIONS_HANDOFF.md",
        "ARTIFACT_INVENTORY.csv",
        "BLOCKER_REGISTER.csv",
        "RELEASE_READINESS.md",
        "RELEASE_NOTES_DRAFT.md",
        "release_candidate_manifest.json",
        "provenance_receipt.json",
        MANIFEST_NAME,
    }
)
DATASET_IDS = (
    "inx_employee_performance",
    "hrdataset_v14",
    "ibm_hr_analytics",
    "employee_turnover",
)
DECLARATION_FIELDS = (
    "Author Contributions",
    "Funding",
    "Institutional Review Board Statement",
    "Informed Consent Statement",
    "Data Availability Statement",
    "Code Availability Statement",
    "Conflicts of Interest",
    "Use of Generative AI and AI-Assisted Technologies",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
TAG_RE = re.compile(r"^v\d+\.\d+\.\d+(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$")


class ReleaseReadinessV3Error(RuntimeError):
    """Raised when the Phase 4B contract or compact package is unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseReadinessV3Error(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(path: Path) -> str:
    result = subprocess.run(
        ["git", "hash-object", "--no-filters", "--", path.as_posix()],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


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


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
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


def _load_contract(path: Path | str) -> tuple[Path, dict[str, Any]]:
    contract = Path(path)
    _require(contract.is_file(), f"Release-readiness contract is absent: {contract.as_posix()}.")
    try:
        payload = json.loads(contract.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseReadinessV3Error(f"Release-readiness contract is invalid JSON: {exc}.") from exc
    _require(isinstance(payload, dict), "Release-readiness contract root must be an object.")
    return contract, payload


def _validate_url(value: Any, label: str) -> None:
    _require(isinstance(value, str) and value.startswith("https://"), f"{label} must be an HTTPS URL.")


def _dataset_rows(datasets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        source = dataset["primary_source"]
        license_record = dataset["license"]
        mirror = dataset["mirror"]
        rows.append(
            {
                "dataset_id": dataset["dataset_id"],
                "analytical_scope": dataset["analytical_scope"],
                "rows": dataset["rows"],
                "columns": dataset["columns"],
                "local_paths": " | ".join(item["path"] for item in dataset["local_files"]),
                "local_sha256": " | ".join(item["sha256"] for item in dataset["local_files"]),
                "primary_authority": source["authority"],
                "primary_record_url": source["record_url"],
                "retrieved_on": source["retrieved_on"],
                "mirror_permalink": mirror["url"] if mirror else "",
                "exact_local_mirror_blob_match": str(bool(mirror and mirror["exact_local_blob_match"])).lower(),
                "source_authenticity_status": dataset["source_authenticity_status"],
                "synthetic_status": dataset["synthetic_status"],
                "license_status": license_record["status"],
                "license_identifier": license_record["identifier"] or "",
                "license_evidence_url": license_record["evidence_url"] or "",
                "redistribution_status": dataset["redistribution_status"],
                "citation_status": dataset["citation_status"],
                "citation_text": dataset["citation_text"],
                "public_release_policy": dataset["public_release_policy"],
                "unresolved": dataset["unresolved"],
            }
        )
    return rows


def _declaration_rows(declarations: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "field": item["field"],
            "status": item["status"],
            "required_input": item["required_input"],
            "safe_interim": item["safe_interim"],
        }
        for item in declarations
    ]


def _artifact_rows(artifacts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": item["artifact_id"],
            "path": item["path"],
            "role": item["role"],
            "evidence_identity": item["evidence_identity"],
            "file_count": item["file_count"],
            "size_bytes": item["size_bytes"],
            "manifest_path": item["manifest_path"],
            "manifest_sha256": item["manifest_sha256"],
            "release_inclusion": item["release_inclusion"],
        }
        for item in artifacts
    ]


def _blocker_rows(blockers: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "blocker_id": item["blocker_id"],
            "category": item["category"],
            "status": item["status"],
            "item": item["item"],
        }
        for item in blockers
    ]


def validate_release_readiness_contract_v3(
    contract_path: Path | str = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Validate frozen provenance facts, local identities, and fail-closed release state."""

    contract, payload = _load_contract(contract_path)
    _require(payload.get("schema_version") == 1, "Release-readiness schema version must be 1.")
    _require(payload.get("phase") == "4B", "Release-readiness phase must be 4B.")
    _require(payload.get("snapshot_date") == "2026-09-07", "Release-readiness snapshot date drifted.")

    controls = payload.get("controls")
    _require(isinstance(controls, dict), "Release-readiness controls are absent.")
    _require(controls.get("paid_api_calls") == 0, "Paid API activity is prohibited.")
    for field in (
        "public_release_authorized",
        "tag_creation_authorized",
        "doi_minting_authorized",
        "history_rewrite_authorized",
        "raw_dataset_publication_authorized",
        "manuscript_editing_authorized",
        "network_required_for_package_generation",
    ):
        _require(controls.get(field) is False, f"Unsafe Phase 4B control: {field}.")
    _require("does not establish upstream" in controls.get("source_authority_rule", ""), "Mirror/source authority separation is absent.")
    _require("never inferred" in controls.get("unknown_rule", ""), "Unknown-field fail-closed rule is absent.")

    repository = payload.get("repository_snapshot")
    _require(isinstance(repository, dict), "Repository snapshot is absent.")
    _require(GIT_SHA1_RE.fullmatch(str(repository.get("snapshot_commit"))) is not None, "Repository snapshot commit is invalid.")
    _require(repository.get("repository_license_status") == "not_present", "Repository software-license state drifted.")
    _require(repository.get("github_release_count") == 0, "A GitHub release is unexpectedly claimed.")
    _require(repository.get("remote_tag_count") == 0, "A remote immutable tag is unexpectedly claimed.")
    _require(repository.get("current_tracked_raw_dataset_paths") == [], "Current tracked raw dataset paths must remain empty.")
    expected_history = {
        "data/external/employee_turnover/raw.csv",
        "data/external/hrdataset_v14/raw.csv",
        "data/external/ibm_hr_analytics/raw.csv",
        "data/raw/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls",
        "data/raw/inx_employee_performance.csv",
    }
    _require(set(repository.get("historical_raw_dataset_paths", [])) == expected_history, "Historical raw-data path register drifted.")
    _require(repository.get("historical_publication_status") == "unresolved_separate_authorization_required", "Historical Git blocker was weakened.")

    datasets = payload.get("datasets")
    _require(isinstance(datasets, list) and len(datasets) == 4, "Exactly four physical datasets are required.")
    _require(tuple(item.get("dataset_id") for item in datasets) == DATASET_IDS, "Dataset order or identity drifted.")
    license_statuses: dict[str, str] = {}
    authenticity_statuses: dict[str, str] = {}
    for dataset in datasets:
        dataset_id = str(dataset["dataset_id"])
        _require(int(dataset.get("rows", 0)) > 0 and int(dataset.get("columns", 0)) > 0, f"{dataset_id} shape is invalid.")
        local_files = dataset.get("local_files")
        _require(isinstance(local_files, list) and local_files, f"{dataset_id} local identity is absent.")
        for record in local_files:
            path = Path(record["path"])
            _require(path.is_file(), f"{dataset_id} local file is absent: {path.as_posix()}.")
            _require(SHA256_RE.fullmatch(str(record.get("sha256"))) is not None, f"{dataset_id} SHA-256 is invalid.")
            _require(GIT_SHA1_RE.fullmatch(str(record.get("git_blob_sha1"))) is not None, f"{dataset_id} Git blob is invalid.")
            _require(_sha256(path) == record["sha256"], f"{dataset_id} local SHA-256 drifted: {path.as_posix()}.")
            _require(path.stat().st_size == int(record["size_bytes"]), f"{dataset_id} local byte size drifted: {path.as_posix()}.")
            _require(_git_blob(path) == record["git_blob_sha1"], f"{dataset_id} local Git blob drifted: {path.as_posix()}.")
        source = dataset.get("primary_source")
        _require(isinstance(source, dict), f"{dataset_id} primary source is absent.")
        _validate_url(source.get("record_url"), f"{dataset_id} primary record")
        _require(source.get("retrieved_on") == payload["snapshot_date"], f"{dataset_id} retrieval date drifted.")
        _require(len(str(source.get("evidence", ""))) >= 80, f"{dataset_id} source evidence is too weak.")
        mirror = dataset.get("mirror")
        if mirror is not None:
            _validate_url(mirror.get("url"), f"{dataset_id} mirror permalink")
            _require(mirror.get("exact_local_blob_match") is True, f"{dataset_id} mirror match must be explicit.")
            _require(mirror.get("repository_license") is None, f"{dataset_id} mirror license must not be imputed.")
            _require(GIT_SHA1_RE.fullmatch(str(mirror.get("commit"))) is not None, f"{dataset_id} mirror commit is invalid.")
            _require(GIT_SHA1_RE.fullmatch(str(mirror.get("remote_blob_sha1"))) is not None, f"{dataset_id} mirror blob is invalid.")
            _require(mirror["remote_blob_sha1"] == local_files[0]["git_blob_sha1"], f"{dataset_id} mirror/local blob mismatch.")
        authenticity = str(dataset.get("source_authenticity_status", ""))
        _require(authenticity.startswith(("partial_", "unresolved_")), f"{dataset_id} authenticity is overclaimed.")
        authenticity_statuses[dataset_id] = authenticity
        license_record = dataset.get("license")
        _require(isinstance(license_record, dict), f"{dataset_id} license record is absent.")
        _require(license_record.get("status") != "verified_unrestricted", f"{dataset_id} license is overclaimed.")
        _require(len(str(license_record.get("evidence", ""))) >= 60, f"{dataset_id} license evidence is too weak.")
        license_statuses[dataset_id] = str(license_record["status"])
        _require(str(dataset.get("redistribution_status", "")).startswith("blocked_"), f"{dataset_id} raw redistribution must remain blocked.")
        _require("Exclude" in str(dataset.get("public_release_policy", "")), f"{dataset_id} raw exclusion policy is absent.")
        _require(len(str(dataset.get("citation_text", ""))) >= 60, f"{dataset_id} citation record is incomplete.")
        _require(len(str(dataset.get("unresolved", ""))) >= 60, f"{dataset_id} unresolved action is incomplete.")

    _require(license_statuses["employee_turnover"] == "unverified", "Employee-turnover license uncertainty was weakened.")
    _require(authenticity_statuses["employee_turnover"].startswith("unresolved_"), "Employee-turnover provenance uncertainty was weakened.")

    declarations = payload.get("declarations")
    _require(isinstance(declarations, list) and len(declarations) == len(DECLARATION_FIELDS), "Declaration register is incomplete.")
    _require(tuple(item.get("field") for item in declarations) == DECLARATION_FIELDS, "Declaration field order drifted.")
    for item in declarations:
        _require(str(item.get("status", "")).startswith(("pending_", "draft_ready_")), f"Declaration status is unsafe: {item.get('field')}.")
        _require(len(str(item.get("required_input", ""))) >= 30, f"Declaration input is incomplete: {item.get('field')}.")
        _require(len(str(item.get("safe_interim", ""))) >= 30, f"Declaration boundary is incomplete: {item.get('field')}.")

    artifacts = payload.get("artifact_inventory")
    _require(isinstance(artifacts, list) and len(artifacts) == 10, "Exactly ten compact component packages are required.")
    _require(len({item.get("artifact_id") for item in artifacts}) == 10, "Artifact identifiers are not unique.")
    for item in artifacts:
        root = Path(item["path"])
        manifest = Path(item["manifest_path"])
        _require(root.is_dir(), f"Artifact package is absent: {root.as_posix()}.")
        _require(manifest.is_file() and manifest.is_relative_to(root), f"Artifact manifest path is invalid: {manifest.as_posix()}.")
        files = [path for path in root.rglob("*") if path.is_file()]
        _require(len(files) == int(item["file_count"]), f"Artifact file count drifted: {item['artifact_id']}.")
        _require(sum(path.stat().st_size for path in files) == int(item["size_bytes"]), f"Artifact byte size drifted: {item['artifact_id']}.")
        _require(_sha256(manifest) == item["manifest_sha256"], f"Artifact manifest hash drifted: {item['artifact_id']}.")
        _require(SHA256_RE.fullmatch(str(item.get("manifest_sha256"))) is not None, f"Artifact manifest SHA-256 is invalid: {item['artifact_id']}.")
        _require(item.get("release_inclusion") == "candidate_tracked_component_after_final_gate", f"Artifact release status drifted: {item['artifact_id']}.")
        _require("data/raw" not in item["path"] and "/raw.csv" not in item["path"], f"Raw data entered release inventory: {item['artifact_id']}.")

    blockers = payload.get("blockers")
    _require(isinstance(blockers, list) and len(blockers) == 11, "Exactly eleven release blockers are required.")
    _require([item.get("blocker_id") for item in blockers] == [f"B{index:02d}" for index in range(1, 12)], "Blocker identifiers drifted.")
    _require(all(item.get("status") == "open" for item in blockers), "An unresolved blocker was silently closed.")

    release = payload.get("release_plan")
    _require(isinstance(release, dict), "Release plan is absent.")
    _require(release.get("status") == "blocked_preparation_only", "Release status is not fail-closed.")
    _require(TAG_RE.fullmatch(str(release.get("proposed_tag"))) is not None, "Proposed release tag is invalid.")
    for field in ("final_commit", "github_release_url", "archive_record_url", "version_doi", "concept_doi"):
        _require(release.get(field) is None, f"Uncreated release identity was populated: {field}.")
    _require(len(release.get("steps", [])) == 8, "Release preparation steps are incomplete.")
    _require(len(release.get("official_guidance", [])) == 5, "Official release guidance is incomplete.")
    for index, url in enumerate(release["official_guidance"], start=1):
        _validate_url(url, f"Official release guidance {index}")
    prohibited = payload.get("prohibited_assertions")
    _require(isinstance(prohibited, list) and len(prohibited) == 11, "Prohibited release assertions are incomplete.")

    return {
        "status": "passed",
        "contract_path": contract.as_posix(),
        "contract_sha256": _sha256(contract),
        "snapshot_date": payload["snapshot_date"],
        "dataset_count": len(datasets),
        "artifact_component_count": len(artifacts),
        "artifact_file_count": sum(int(item["file_count"]) for item in artifacts),
        "artifact_size_bytes": sum(int(item["size_bytes"]) for item in artifacts),
        "declaration_count": len(declarations),
        "open_blocker_count": len(blockers),
        "historical_raw_path_count": len(repository["historical_raw_dataset_paths"]),
        "raw_redistribution_allowed_count": 0,
        "github_release_count": repository["github_release_count"],
        "remote_tag_count": repository["remote_tag_count"],
        "final_commit_recorded": False,
        "doi_recorded": False,
        "paid_api_calls": 0,
    }


def _readme(payload: Mapping[str, Any], generation_commit: str) -> str:
    return "\n".join(
        [
            "# Phase 4B Provenance and Release-Readiness Package",
            "",
            f"Generation commit: `{generation_commit}`",
            f"Verification snapshot: `{payload['snapshot_date']}`",
            "",
            "This deterministic package records dataset provenance/license evidence, unresolved ethics and author declarations, compact artifact inventory, and an immutable-release procedure. It is preparation evidence only: the release state is `blocked_preparation_only`.",
            "",
            "No tag, GitHub release, archive record, DOI, repository software license, ethics determination, or raw-data publication is created or claimed here. All four raw datasets remain excluded from release payloads.",
            "",
            "## Contents",
            "",
            "- `DATA_PROVENANCE_LICENSE_REGISTER.csv` and `PROVENANCE_LICENSE_REPORT.md`: source, byte, license, citation, and redistribution evidence kept as distinct fields.",
            "- `ETHICS_DECLARATIONS_REGISTER.csv` and `ETHICS_DECLARATIONS_HANDOFF.md`: author/institution inputs and safe interim boundaries.",
            "- `ARTIFACT_INVENTORY.csv`: ten tracked compact evidence components and their manifest hashes.",
            "- `BLOCKER_REGISTER.csv`: eleven open release/submission blockers.",
            "- `RELEASE_READINESS.md` and `RELEASE_NOTES_DRAFT.md`: proposed tag and ordered release/DOI workflow without publication.",
            "- `release_candidate_manifest.json`: machine-readable candidate state with a null final commit and null publication identifiers.",
            "- `provenance_receipt.json` and `manifest.json`: contract/generation identity and closed-world hashes.",
        ]
    ) + "\n"


def _provenance_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Dataset Provenance, License, and Redistribution Report",
        "",
        "A checksum proves identity of bytes already held; it does not prove authorship, authenticity, consent, license ownership, or redistribution permission. A public mirror and its repository license, when present, are not treated as authority for third-party dataset rights.",
        "",
    ]
    for dataset in payload["datasets"]:
        source = dataset["primary_source"]
        license_record = dataset["license"]
        lines.extend(
            [
                f"## {dataset['dataset_id']}",
                "",
                f"- Scope/shape: `{dataset['analytical_scope']}`, {dataset['rows']} rows × {dataset['columns']} columns.",
                f"- Primary record: {source['record_url']}",
                f"- Source/authenticity status: `{dataset['source_authenticity_status']}`.",
                f"- Synthetic/fictitious evidence: `{dataset['synthetic_status']}`.",
                f"- License status: `{license_record['status']}`; identifier: `{license_record['identifier'] or 'not verified'}`.",
                f"- License evidence: {license_record['evidence']}",
                f"- Redistribution status: `{dataset['redistribution_status']}`.",
                f"- Citation status: `{dataset['citation_status']}`.",
                f"- Proposed citation: {dataset['citation_text']}",
                f"- Public-package rule: {dataset['public_release_policy']}",
                f"- Required resolution: {dataset['unresolved']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Cross-dataset decision",
            "",
            "No raw dataset is approved for redistribution. The current reproducibility boundary is hashes, schemas, acquisition instructions, and aggregate evidence. This is a conservative publication decision rather than a legal conclusion.",
        ]
    )
    return "\n".join(lines) + "\n"


def _ethics_handoff(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Ethics and Declarations Handoff",
        "",
        "The available source records describe the datasets as synthetic, fictional, or simulated, but the provenance chain is not complete for every local file. Therefore neither ethics approval nor an IRB/consent `not applicable` determination is inferred. Institution-approved wording remains required.",
        "",
    ]
    for item in payload["declarations"]:
        lines.extend(
            [
                f"## {item['field']}",
                "",
                f"Status: `{item['status']}`",
                "",
                f"Required input: {item['required_input']}",
                "",
                f"Safe interim boundary: {item['safe_interim']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Data/code availability drafting boundary",
            "",
            "A later manuscript statement may identify the exact public repository commit and compact aggregate evidence. It must say that raw data are excluded and must direct readers to the verified or qualified upstream records. It must not call the repository open source until a software license is actually selected and committed, and it must not cite a tag, release URL, or DOI until each exists remotely.",
        ]
    )
    return "\n".join(lines) + "\n"


def _release_readiness(payload: Mapping[str, Any]) -> str:
    repository = payload["repository_snapshot"]
    release = payload["release_plan"]
    lines = [
        "# Immutable Release Readiness",
        "",
        "Status: `blocked_preparation_only`",
        "",
        f"Proposed future tag: `{release['proposed_tag']}`",
        "Final release commit: `not recorded`",
        "GitHub release URL: `not created`",
        "Archive record/DOI: `not created`",
        "",
        "## Verified repository snapshot",
        "",
        f"- Audit commit: `{repository['snapshot_commit']}` on `{repository['active_branch']}`.",
        f"- Public repository/default branch: {repository['remote_url']} / `{repository['default_branch']}`.",
        "- Detected repository software license: none.",
        "- GitHub releases: 0; remote tags: 0.",
        f"- Historical local-only tag: `{repository['local_only_historical_tag']}` at `{repository['local_only_historical_tag_commit']}`; it is not an origin tag and must not be cited as the final immutable release.",
        "- Current tracked raw datasets: 0; five raw dataset paths remain in Git history and require a separately authorized strategy.",
        "",
        "## Open blockers",
        "",
    ]
    lines.extend(f"- `{item['blocker_id']}` ({item['category']}): {item['item']}" for item in payload["blockers"])
    lines.extend(["", "## Authorized payload boundary", "", release["allowed_payload"], "", "Excluded: " + release["excluded_payload"], "", "## Ordered release and DOI procedure", ""])
    lines.extend(f"{index}. {step}" for index, step in enumerate(release["steps"], start=1))
    lines.extend(["", "## Official guidance", ""])
    lines.extend(f"- {url}" for url in release["official_guidance"])
    return "\n".join(lines) + "\n"


def _release_notes(payload: Mapping[str, Any]) -> str:
    release = payload["release_plan"]
    return "\n".join(
        [
            "# Release Notes Draft — v1.0.0-paper-revision",
            "",
            "Status: draft only; do not publish.",
            "",
            "## Summary",
            "",
            "This candidate consolidates the immutable canonical-v2 technical evidence with additive v3 evidence for feature governance, ordinal baselines/models, repeated nested-CV sensitivity, fixed-versus-retuned policy analysis, SHAP stability/faithfulness, extended calibration, subgroup/proxy-use diagnostics, HRDataset_v14 target/CV sensitivity, core data quality, and source-verified literature positioning.",
            "",
            "## Evidence boundary",
            "",
            "The release candidate is a research and reproducibility artifact, not an HR decision system. It does not establish causal drivers, fairness or absence of discrimination, human usefulness, legal compliance, external organizational transport, or deployment readiness.",
            "",
            "## Included after final approval",
            "",
            f"{release['allowed_payload']}",
            "",
            "## Excluded",
            "",
            f"{release['excluded_payload']}",
            "",
            "## Required before publication",
            "",
            "Resolve every item in `BLOCKER_REGISTER.csv`; record the exact final commit; validate all manifests and rendered manuscript outputs; obtain explicit release authorization; only then create the remote tag/release and archival DOI.",
        ]
    ) + "\n"


def _candidate_manifest(payload: Mapping[str, Any], generation_commit: str, validation: Mapping[str, Any]) -> dict[str, Any]:
    release = payload["release_plan"]
    return {
        "schema_version": 1,
        "candidate_kind": "paper_revision_release_preparation",
        "status": release["status"],
        "snapshot_date": payload["snapshot_date"],
        "generation_commit": generation_commit,
        "proposed_tag": release["proposed_tag"],
        "final_commit": None,
        "github_release_url": None,
        "archive_record_url": None,
        "version_doi": None,
        "concept_doi": None,
        "component_artifacts": payload["artifact_inventory"],
        "open_blocker_ids": [item["blocker_id"] for item in payload["blockers"]],
        "raw_dataset_included": False,
        "release_authorized": False,
        "contract_sha256": validation["contract_sha256"],
    }


def _manifest(staging: Path, generation_commit: str, contract_sha256: str) -> dict[str, Any]:
    records = []
    for path in sorted(staging.iterdir(), key=lambda item: item.name):
        if path.is_file():
            records.append({"path": path.name, "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    return {
        "schema_version": 1,
        "package_kind": "phase4b_provenance_ethics_release_readiness",
        "generation_commit": generation_commit,
        "contract_sha256": contract_sha256,
        "file_count_excluding_manifest": len(records),
        "files": records,
    }


def export_release_readiness_package_v3(
    contract_path: Path | str = DEFAULT_CONTRACT,
    output_dir: Path | str = DEFAULT_OUTPUT,
    *,
    generation_commit: str | None = None,
    require_clean_git: bool = True,
) -> dict[str, Any]:
    """Atomically publish a deterministic, non-release Phase 4B package."""

    contract, payload = _load_contract(contract_path)
    validation = validate_release_readiness_contract_v3(contract)
    if require_clean_git:
        _require(_git_is_clean(), "Release-readiness export requires a clean Git worktree.")
    commit = generation_commit or _git_head()
    _require(commit == "test" or GIT_SHA1_RE.fullmatch(commit) is not None, "Generation commit is invalid.")
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    _require(not staging.exists(), "Release-readiness staging path unexpectedly exists.")
    staging.mkdir()
    try:
        dataset_fields = list(_dataset_rows(payload["datasets"])[0])
        declaration_fields = list(_declaration_rows(payload["declarations"])[0])
        artifact_fields = list(_artifact_rows(payload["artifact_inventory"])[0])
        blocker_fields = list(_blocker_rows(payload["blockers"])[0])
        _write_bytes(staging / "README.md", _readme(payload, commit).encode("utf-8"))
        _write_bytes(staging / "DATA_PROVENANCE_LICENSE_REGISTER.csv", _csv_bytes(dataset_fields, _dataset_rows(payload["datasets"])))
        _write_bytes(staging / "PROVENANCE_LICENSE_REPORT.md", _provenance_report(payload).encode("utf-8"))
        _write_bytes(staging / "ETHICS_DECLARATIONS_REGISTER.csv", _csv_bytes(declaration_fields, _declaration_rows(payload["declarations"])))
        _write_bytes(staging / "ETHICS_DECLARATIONS_HANDOFF.md", _ethics_handoff(payload).encode("utf-8"))
        _write_bytes(staging / "ARTIFACT_INVENTORY.csv", _csv_bytes(artifact_fields, _artifact_rows(payload["artifact_inventory"])))
        _write_bytes(staging / "BLOCKER_REGISTER.csv", _csv_bytes(blocker_fields, _blocker_rows(payload["blockers"])))
        _write_bytes(staging / "RELEASE_READINESS.md", _release_readiness(payload).encode("utf-8"))
        _write_bytes(staging / "RELEASE_NOTES_DRAFT.md", _release_notes(payload).encode("utf-8"))
        _write_bytes(staging / "release_candidate_manifest.json", _json_bytes(_candidate_manifest(payload, commit, validation)))
        provenance = {
            "schema_version": 1,
            "package_kind": "phase4b_provenance_ethics_release_readiness",
            "snapshot_date": payload["snapshot_date"],
            "generation_commit": commit,
            "contract_path": contract.as_posix(),
            "contract_sha256": validation["contract_sha256"],
            "contract_validation": validation,
            "publication_controls": {
                "paid_api_calls": 0,
                "network_calls_during_package_generation": 0,
                "raw_dataset_included": False,
                "tag_created": False,
                "github_release_created": False,
                "doi_created": False,
                "history_rewritten": False,
                "manuscript_modified": False,
                "repository_license_created": False,
                "ethics_determination_invented": False,
            },
        }
        _write_bytes(staging / "provenance_receipt.json", _json_bytes(provenance))
        _write_bytes(staging / MANIFEST_NAME, _json_bytes(_manifest(staging, commit, validation["contract_sha256"])))
        inventory = {path.name for path in staging.iterdir() if path.is_file()}
        _require(inventory == EXPECTED_OUTPUTS, f"Release-readiness staging inventory drifted: {sorted(inventory ^ EXPECTED_OUTPUTS)}.")
        _require(not destination.exists(), f"Release-readiness destination already exists: {destination.as_posix()}.")
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return validate_release_readiness_package_v3(destination, contract_path=contract)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_release_readiness_package_v3(
    package_dir: Path | str = DEFAULT_OUTPUT,
    *,
    contract_path: Path | str = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Validate closed-world hashes, exact rows, and fail-closed release claims."""

    package = Path(package_dir)
    _require(package.is_dir(), f"Release-readiness package is absent: {package.as_posix()}.")
    inventory = {path.name for path in package.iterdir() if path.is_file()}
    _require(inventory == EXPECTED_OUTPUTS, f"Release-readiness package inventory drifted: {sorted(inventory ^ EXPECTED_OUTPUTS)}.")
    _require(not any(path.is_dir() for path in package.iterdir()), "Release-readiness package contains a directory.")
    contract, payload = _load_contract(contract_path)
    validation = validate_release_readiness_contract_v3(contract)
    manifest = json.loads((package / MANIFEST_NAME).read_text(encoding="utf-8"))
    records = manifest.get("files")
    _require(isinstance(records, list), "Release-readiness manifest records are absent.")
    _require(manifest.get("file_count_excluding_manifest") == len(EXPECTED_OUTPUTS) - 1, "Release-readiness manifest count drifted.")
    _require({item.get("path") for item in records} == EXPECTED_OUTPUTS - {MANIFEST_NAME}, "Release-readiness manifest inventory drifted.")
    _require(manifest.get("contract_sha256") == validation["contract_sha256"], "Release-readiness manifest contract hash drifted.")
    for record in records:
        path = package / str(record["path"])
        _require(path.stat().st_size == int(record["size_bytes"]), f"Release-readiness package size drifted for {path.name}.")
        _require(_sha256(path) == record["sha256"], f"Release-readiness package hash drifted for {path.name}.")

    datasets = _read_csv(package / "DATA_PROVENANCE_LICENSE_REGISTER.csv")
    declarations = _read_csv(package / "ETHICS_DECLARATIONS_REGISTER.csv")
    artifacts = _read_csv(package / "ARTIFACT_INVENTORY.csv")
    blockers = _read_csv(package / "BLOCKER_REGISTER.csv")
    expected_datasets = [{key: str(value) for key, value in row.items()} for row in _dataset_rows(payload["datasets"])]
    expected_declarations = _declaration_rows(payload["declarations"])
    expected_artifacts = [{key: str(value) for key, value in row.items()} for row in _artifact_rows(payload["artifact_inventory"])]
    expected_blockers = _blocker_rows(payload["blockers"])
    _require(datasets == expected_datasets, "Dataset provenance rows drifted from contract.")
    _require(declarations == expected_declarations, "Declaration rows drifted from contract.")
    _require(artifacts == expected_artifacts, "Artifact inventory rows drifted from contract.")
    _require(blockers == expected_blockers, "Blocker rows drifted from contract.")

    candidate = json.loads((package / "release_candidate_manifest.json").read_text(encoding="utf-8"))
    _require(candidate.get("status") == "blocked_preparation_only", "Candidate release status drifted.")
    _require(candidate.get("final_commit") is None, "Candidate unexpectedly records a final commit.")
    for field in ("github_release_url", "archive_record_url", "version_doi", "concept_doi"):
        _require(candidate.get(field) is None, f"Candidate unexpectedly records {field}.")
    _require(candidate.get("open_blocker_ids") == [f"B{index:02d}" for index in range(1, 12)], "Candidate blocker inventory drifted.")
    _require(candidate.get("raw_dataset_included") is False and candidate.get("release_authorized") is False, "Candidate release boundary drifted.")
    _require(candidate.get("component_artifacts") == payload["artifact_inventory"], "Candidate artifact inventory drifted.")

    provenance = json.loads((package / "provenance_receipt.json").read_text(encoding="utf-8"))
    _require(provenance.get("contract_validation") == validation, "Release-readiness provenance validation receipt drifted.")
    publication_controls = provenance.get("publication_controls")
    _require(isinstance(publication_controls, dict), "Release-readiness publication controls are absent.")
    _require(publication_controls.get("paid_api_calls") == 0 and publication_controls.get("network_calls_during_package_generation") == 0, "Release-readiness package records network or paid API activity.")
    for field, value in publication_controls.items():
        if field not in {"paid_api_calls", "network_calls_during_package_generation"}:
            _require(value is False, f"Release-readiness publication control drifted: {field}.")

    provenance_text = (package / "PROVENANCE_LICENSE_REPORT.md").read_text(encoding="utf-8")
    ethics_text = (package / "ETHICS_DECLARATIONS_HANDOFF.md").read_text(encoding="utf-8")
    release_text = (package / "RELEASE_READINESS.md").read_text(encoding="utf-8")
    notes_text = (package / "RELEASE_NOTES_DRAFT.md").read_text(encoding="utf-8")
    for required in (
        "checksum proves identity of bytes already held; it does not prove authorship",
        "No raw dataset is approved for redistribution",
        "CC-BY-NC-ND-4.0",
        "ODbL-1.0-and-DbCL-1.0",
        "No accessible authoritative license statement was verified",
    ):
        _require(required.lower() in provenance_text.lower(), f"Provenance boundary is absent: {required}.")
    for required in (
        "neither ethics approval nor an IRB/consent `not applicable` determination is inferred",
        "must not call the repository open source",
        "raw data are excluded",
    ):
        _require(required.lower() in ethics_text.lower(), f"Ethics boundary is absent: {required}.")
    for required in (
        "blocked_preparation_only",
        "Final release commit: `not recorded`",
        "GitHub releases: 0; remote tags: 0",
        "five raw dataset paths remain in Git history",
        "do not publish",
    ):
        combined = release_text + "\n" + notes_text
        _require(required.lower() in combined.lower(), f"Release boundary is absent: {required}.")

    return {
        "status": "passed",
        "package_dir": package.as_posix(),
        "generation_commit": manifest["generation_commit"],
        "contract_sha256": validation["contract_sha256"],
        "manifest_sha256": _sha256(package / MANIFEST_NAME),
        "file_count": len(EXPECTED_OUTPUTS),
        "size_bytes": sum(path.stat().st_size for path in package.iterdir() if path.is_file()),
        "dataset_count": len(datasets),
        "artifact_component_count": len(artifacts),
        "declaration_count": len(declarations),
        "open_blocker_count": len(blockers),
        "raw_redistribution_allowed_count": 0,
        "github_release_count": 0,
        "remote_tag_count": 0,
        "doi_recorded": False,
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
        receipt = validate_release_readiness_contract_v3(args.contract)
    elif args.validate_package_only:
        receipt = validate_release_readiness_package_v3(args.output_dir, contract_path=args.contract)
    else:
        receipt = export_release_readiness_package_v3(args.contract, args.output_dir)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONTRACT",
    "DEFAULT_OUTPUT",
    "EXPECTED_OUTPUTS",
    "ReleaseReadinessV3Error",
    "export_release_readiness_package_v3",
    "validate_release_readiness_contract_v3",
    "validate_release_readiness_package_v3",
]
