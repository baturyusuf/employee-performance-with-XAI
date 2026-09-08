"""Validate and publish the Phase 5A sentence-level v3 claim matrix."""

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
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CONTRACT = Path("configs/claim_matrix_v3.json")
DEFAULT_OUTPUT = Path("reports/research_log/major_revision_v3/phase5a_claim_matrix")
MANIFEST_NAME = "manifest.json"
EXPECTED_OUTPUTS = frozenset(
    {
        "README.md",
        "CLAIM_MATRIX.csv",
        "CLAIM_MATRIX.md",
        "CLAIM_BOUNDARIES.md",
        "SOURCE_REGISTER.csv",
        "APPROVAL_REQUEST.md",
        "approval_record.json",
        "provenance_receipt.json",
        MANIFEST_NAME,
    }
)
CLAIM_TYPES = frozenset({"numerical", "narrative"})
SUPPORT_LEVELS = frozenset(
    {
        "direct_exact",
        "direct_descriptive",
        "bounded_synthesis",
        "procedural_boundary",
    }
)
REQUIRED_COMPONENTS = frozenset(
    {
        "feature_contract",
        "phase1b",
        "phase1c",
        "phase1d",
        "phase2a",
        "phase2b",
        "phase2c",
        "phase3a",
        "phase3b",
        "phase4a",
        "phase4b",
        "phase5a",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


class ClaimMatrixV3Error(RuntimeError):
    """Raised when the Phase 5A contract or package is incomplete or unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClaimMatrixV3Error(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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


def _load_contract(path: Path | str) -> tuple[Path, dict[str, Any]]:
    contract = Path(path)
    _require(contract.is_file(), f"Claim-matrix contract is absent: {contract.as_posix()}.")
    try:
        payload = json.loads(contract.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClaimMatrixV3Error(f"Claim-matrix contract is invalid JSON: {exc}.") from exc
    _require(isinstance(payload, dict), "Claim-matrix contract root must be an object.")
    return contract, payload


def _display_decimal(value: str, places: int, scale: str) -> str:
    try:
        number = Decimal(value) * Decimal(scale)
    except InvalidOperation as exc:
        raise ClaimMatrixV3Error(f"Non-numeric exact value: {value}.") from exc
    quantum = Decimal(1).scaleb(-places)
    return format(number.quantize(quantum, rounding=ROUND_HALF_EVEN), f".{places}f")


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _validate_and_enrich(
    contract_path: Path | str,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    contract, payload = _load_contract(contract_path)
    _require(payload.get("schema_version") == 1, "Claim-matrix schema version must be 1.")
    _require(payload.get("phase") == "5A", "Claim-matrix phase must be 5A.")
    _require(payload.get("snapshot_date") == "2026-09-08", "Claim-matrix snapshot date drifted.")
    state = payload.get("status")
    _require(state in {"pending_user_approval", "approved_for_phase5b"}, "Claim-matrix status is invalid.")

    approval = payload.get("user_approval")
    _require(isinstance(approval, dict), "User-approval record is absent.")
    _require(approval.get("status") in {"pending", "approved"}, "User-approval status is invalid.")
    _require((state == "approved_for_phase5b") == (approval.get("status") == "approved"), "Contract and approval states disagree.")

    controls = payload.get("controls")
    _require(isinstance(controls, dict), "Claim-matrix controls are absent.")
    approved = approval.get("status") == "approved"
    _require(controls.get("manuscript_editing_authorized") is approved, "Manuscript authorization does not match approval state.")
    _require(controls.get("bibliography_editing_authorized") is approved, "Bibliography authorization does not match approval state.")
    expected_false = (
        "release_authorized",
        "tag_creation_authorized",
        "doi_minting_authorized",
        "raw_data_publication_authorized",
        "network_required",
    )
    for field in expected_false:
        _require(controls.get(field) is False, f"Unsafe Phase 5A control: {field}.")
    _require(controls.get("paid_api_calls") == 0, "Paid API activity is prohibited.")

    overclaims = payload.get("global_prohibited_assertions")
    _require(isinstance(overclaims, list) and len(overclaims) >= 12, "Global prohibited assertions are incomplete.")
    _require(len(overclaims) == len(set(overclaims)), "Global prohibited assertions are duplicated.")

    claims = payload.get("claims")
    _require(isinstance(claims, list) and len(claims) >= 30, "At least 30 sentence-level claims are required.")
    claim_ids = [str(claim.get("claim_id", "")) for claim in claims]
    _require(len(claim_ids) == len(set(claim_ids)), "Claim identifiers are not unique.")
    _require(all(re.fullmatch(r"C\d{3}", claim_id) for claim_id in claim_ids), "Claim identifier format drifted.")
    _require({claim.get("component") for claim in claims} == REQUIRED_COMPONENTS, "Claim component coverage is incomplete.")

    csv_cache: dict[Path, list[dict[str, str]]] = {}
    enriched: list[dict[str, Any]] = []
    numerical_count = 0
    narrative_count = 0
    for claim in claims:
        claim_id = str(claim["claim_id"])
        claim_type = claim.get("claim_type")
        _require(claim_type in CLAIM_TYPES, f"{claim_id} claim type is invalid.")
        _require(claim.get("approval_status") == "pending_user_approval", f"{claim_id} approval was inferred.")
        _require(claim.get("support_level") in SUPPORT_LEVELS, f"{claim_id} support level is invalid.")
        _require(claim.get("component") in REQUIRED_COMPONENTS, f"{claim_id} component is invalid.")
        for field in ("manuscript_section", "proposed_claim", "evidence_scope", "mandatory_qualifier", "prohibited_overclaim"):
            _require(len(str(claim.get(field, "")).strip()) >= 12, f"{claim_id} field is incomplete: {field}.")

        source_path_text = str(claim.get("source_path", ""))
        source = Path(source_path_text)
        _require(source.is_file(), f"{claim_id} source is absent: {source_path_text}.")
        _require(source_path_text.startswith("reports/research_log/major_revision_v3/"), f"{claim_id} source is outside governed v3 evidence.")
        _require(not re.search(r"(^|/)(raw|runs?|models?|artifacts?)(/|$)", source_path_text, re.I), f"{claim_id} references excluded evidence.")
        declared_sha = str(claim.get("source_sha256", ""))
        _require(SHA256_RE.fullmatch(declared_sha) is not None, f"{claim_id} source SHA-256 is invalid.")
        _require(_sha256(source) == declared_sha, f"{claim_id} source SHA-256 drifted.")

        item = dict(claim)
        item["source_row_number"] = ""
        item["verified_exact_value"] = ""
        if claim_type == "numerical":
            numerical_count += 1
            _require(source.suffix.lower() == ".csv", f"{claim_id} numerical source must be CSV.")
            selector = claim.get("source_selector")
            _require(isinstance(selector, dict) and selector, f"{claim_id} source selector is absent.")
            _require(all(isinstance(k, str) and isinstance(v, str) for k, v in selector.items()), f"{claim_id} selector must contain strings.")
            rows = csv_cache.setdefault(source, _load_csv_rows(source))
            matches = [
                (index, row)
                for index, row in enumerate(rows, start=2)
                if all(row.get(key) == value for key, value in selector.items())
            ]
            _require(len(matches) == 1, f"{claim_id} selector matched {len(matches)} rows, expected one.")
            row_number, row = matches[0]
            column = str(claim.get("source_column", ""))
            _require(column in row, f"{claim_id} source column is absent.")
            exact = str(claim.get("exact_value", ""))
            _require(row[column] == exact, f"{claim_id} exact value drifted.")
            places = claim.get("rounding_places")
            scale = str(claim.get("display_scale", "1"))
            _require(isinstance(places, int) and 0 <= places <= 6, f"{claim_id} rounding rule is invalid.")
            expected_display = _display_decimal(exact, places, scale)
            _require(claim.get("display_value") == expected_display, f"{claim_id} displayed value is not deterministic.")
            _require(expected_display in str(claim["proposed_claim"]), f"{claim_id} proposed sentence omits its displayed value.")
            item["source_row_number"] = row_number
            item["verified_exact_value"] = row[column]
        else:
            narrative_count += 1
            anchor = str(claim.get("evidence_anchor", ""))
            _require(len(anchor) >= 20, f"{claim_id} evidence anchor is too short.")
            try:
                source_text = source.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ClaimMatrixV3Error(f"{claim_id} narrative source is not UTF-8 text.") from exc
            _require(anchor in source_text, f"{claim_id} evidence anchor drifted.")
        enriched.append(item)

    _require(numerical_count >= 20, "At least 20 exact numerical claims are required.")
    _require(narrative_count >= 10, "At least 10 bounded narrative claims are required.")
    claim_set_sha256 = _sha256_bytes(_json_bytes({"claims": payload["claims"]}))
    if approved:
        _require(approval.get("approved_by") == "user", "Approved claim set must identify the user as approver.")
        _require(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", str(approval.get("approved_at_utc", ""))) is not None, "Approval timestamp is invalid.")
        _require(approval.get("approved_claim_set_sha256") == claim_set_sha256, "Approved claim-set digest does not match the frozen claims.")
        _require(len(str(approval.get("approval_statement", ""))) >= 80, "Approval statement is incomplete.")
    else:
        for field in ("approved_by", "approved_at_utc", "approved_claim_set_sha256"):
            _require(approval.get(field) is None, f"Unapproved field must remain null: {field}.")
    return contract, payload, enriched


def validate_claim_matrix_contract_v3(
    contract_path: Path | str = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Validate exact rows, hashes, narrative anchors, and the approval boundary."""

    contract, payload, claims = _validate_and_enrich(contract_path)
    sources = {claim["source_path"] for claim in claims}
    return {
        "status": "passed",
        "contract_path": contract.as_posix(),
        "contract_sha256": _sha256(contract),
        "claim_set_sha256": _sha256_bytes(_json_bytes({"claims": payload["claims"]})),
        "claim_count": len(claims),
        "numerical_claim_count": sum(claim["claim_type"] == "numerical" for claim in claims),
        "narrative_claim_count": sum(claim["claim_type"] == "narrative" for claim in claims),
        "source_file_count": len(sources),
        "component_count": len({claim["component"] for claim in claims}),
        "approval_status": payload["user_approval"]["status"],
        "manuscript_editing_authorized": payload["controls"]["manuscript_editing_authorized"],
        "paid_api_calls": 0,
    }


def _claim_rows(claims: Sequence[Mapping[str, Any]], approval_status: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for claim in claims:
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "manuscript_section": claim["manuscript_section"],
                "component": claim["component"],
                "claim_type": claim["claim_type"],
                "support_level": claim["support_level"],
                "approval_status": approval_status,
                "proposed_claim": claim["proposed_claim"],
                "source_path": claim["source_path"],
                "source_sha256": claim["source_sha256"],
                "source_selector": json.dumps(claim.get("source_selector", {}), sort_keys=True, separators=(",", ":")),
                "source_row_number": claim["source_row_number"],
                "source_column": claim.get("source_column", ""),
                "exact_value": claim.get("exact_value", ""),
                "display_value": claim.get("display_value", ""),
                "rounding_places": claim.get("rounding_places", ""),
                "display_scale": claim.get("display_scale", ""),
                "evidence_anchor": claim.get("evidence_anchor", ""),
                "evidence_scope": claim["evidence_scope"],
                "mandatory_qualifier": claim["mandatory_qualifier"],
                "prohibited_overclaim": claim["prohibited_overclaim"],
            }
        )
    return rows


def _claim_matrix_md(claims: Sequence[Mapping[str, Any]], approval_status: str) -> str:
    status_line = (
        "Status: **APPROVED FOR PHASE 5B**. These sentences are the sole authorized claim boundary for manuscript and reviewer-response drafting."
        if approval_status == "approved"
        else "Status: **PENDING USER APPROVAL**. These sentences are candidates for later manuscript drafting; none is approved or inserted into the manuscript by this package."
    )
    lines = [
        "# Phase 5A Sentence-Level Claim Matrix",
        "",
        status_line,
        "",
    ]
    sections: list[str] = []
    for claim in claims:
        if claim["manuscript_section"] not in sections:
            sections.append(str(claim["manuscript_section"]))
    for section in sections:
        lines.extend([f"## {section}", ""])
        for claim in claims:
            if claim["manuscript_section"] != section:
                continue
            source_detail = f"row {claim['source_row_number']}, `{claim.get('source_column', '')}`" if claim["claim_type"] == "numerical" else "text anchor"
            lines.extend(
                [
                    f"### {claim['claim_id']} — {claim['support_level']}",
                    "",
                    str(claim["proposed_claim"]),
                    "",
                    f"- Evidence: `{claim['source_path']}` ({source_detail}; SHA-256 `{claim['source_sha256']}`).",
                    f"- Required qualifier: {claim['mandatory_qualifier']}",
                    f"- Prohibited overclaim: {claim['prohibited_overclaim']}",
                    "",
                ]
            )
    return "\n".join(lines)


def _boundaries_md(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Claim Boundaries",
            "",
            "These prohibitions apply to every later abstract, result, discussion, conclusion, caption, response-to-reviewers statement, and repository/release description.",
            "",
            *[f"- {item}" for item in payload["global_prohibited_assertions"]],
            "",
            "No `not_reported`, missing, unsupported, pending, or unverified value may be converted to `no`, zero, approved, exempt, licensed, authentic, released, or resolved.",
            "",
        ]
    )


def _approval_request(payload: Mapping[str, Any], claim_set_sha256: str) -> str:
    approval = payload["user_approval"]
    if approval["status"] == "approved":
        return "\n".join(
            [
                "# Explicit Approval Record",
                "",
                "Decision: `approved`",
                f"Approved by: `{approval['approved_by']}`",
                f"Approved at UTC: `{approval['approved_at_utc']}`",
                f"Approved claim-set SHA-256: `{approval['approved_claim_set_sha256']}`",
                "",
                str(approval["approval_statement"]),
                "",
                "This approval authorizes Phase 5B manuscript and reviewer-response drafting under this sole claim boundary. It does not authorize a release, tag, DOI, raw-data publication, Git-history rewrite, or invented declaration/licence/ethics value.",
                "",
            ]
        )
    return "\n".join(
        [
            "# Explicit Approval Request",
            "",
            "Decision requested: approve or reject the complete Phase 5A claim set for use as the sole claim boundary in the later manuscript rewrite.",
            "",
            f"Claim-set SHA-256: `{claim_set_sha256}`",
            f"Claims: {len(payload['claims'])}",
            "Current decision: `pending`",
            "",
            "Approval must be explicit and must identify this digest. A generic instruction to continue is not approval. Until approval is recorded, `manuscript/mdpi_information/main.md`, `main.tex`, and `references.bib` remain outside the authorized edit scope.",
            "",
            "Approval of the claim set would not resolve dataset rights, ethics/IRB, author declarations, repository licensing, Git-history, release, tag, or DOI blockers.",
            "",
        ]
    )


def _source_rows(claims: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for claim in claims:
        grouped.setdefault(str(claim["source_path"]), []).append(claim)
    rows: list[dict[str, Any]] = []
    for source_path, source_claims in sorted(grouped.items()):
        source = Path(source_path)
        manifest = source.parent / MANIFEST_NAME
        rows.append(
            {
                "source_path": source_path,
                "source_sha256": source_claims[0]["source_sha256"],
                "source_size_bytes": source.stat().st_size,
                "source_format": source.suffix.lower().lstrip("."),
                "claim_count": len(source_claims),
                "claim_ids": " | ".join(str(item["claim_id"]) for item in source_claims),
                "parent_manifest_path": manifest.as_posix() if manifest.is_file() else "",
                "parent_manifest_sha256": _sha256(manifest) if manifest.is_file() else "",
            }
        )
    return rows


def _build_outputs(
    contract_path: Path | str,
    generation_commit: str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    contract, payload, claims = _validate_and_enrich(contract_path)
    claim_set_sha256 = _sha256_bytes(_json_bytes({"claims": payload["claims"]}))
    approval_status = str(payload["user_approval"]["status"])
    rows = _claim_rows(claims, approval_status)
    source_rows = _source_rows(claims)
    fields = list(rows[0])
    source_fields = list(source_rows[0])
    approval_record = {
        "approved_at_utc": payload["user_approval"]["approved_at_utc"],
        "approved_by": payload["user_approval"]["approved_by"],
        "approved_claim_set_sha256": payload["user_approval"]["approved_claim_set_sha256"],
        "claim_count": len(claims),
        "claim_set_sha256": claim_set_sha256,
        "decision": approval_status,
        "manuscript_editing_authorized": payload["controls"]["manuscript_editing_authorized"],
        "schema_version": 1,
    }
    provenance = {
        "approval_status": approval_status,
        "claim_count": len(claims),
        "claim_set_sha256": claim_set_sha256,
        "component_count": len({claim["component"] for claim in claims}),
        "contract_path": contract.as_posix(),
        "contract_sha256": _sha256(contract),
        "generation_commit": generation_commit,
        "manuscript_editing_authorized": payload["controls"]["manuscript_editing_authorized"],
        "narrative_claim_count": sum(claim["claim_type"] == "narrative" for claim in claims),
        "network_calls": 0,
        "numerical_claim_count": sum(claim["claim_type"] == "numerical" for claim in claims),
        "package_kind": "phase5a_sentence_level_claim_matrix",
        "paid_api_calls": 0,
        "schema_version": 1,
        "source_file_count": len(source_rows),
        "status": "passed_approved_for_phase5b" if approval_status == "approved" else "passed_pending_user_approval",
    }
    readme = "\n".join(
        [
            "# Phase 5A Claim-Matrix Package",
            "",
            f"Generation commit: `{generation_commit}`",
            f"Claim-set SHA-256: `{claim_set_sha256}`",
            f"Status: `{payload['status']}`",
            "",
            "This deterministic package binds every proposed numerical sentence to one exact CSV row/value and every proposed narrative sentence to a hash-bound text anchor. Each row carries its evidence scope, required qualifier, and prohibited overclaim.",
            "",
            (
                "The recorded digest-specific approval authorizes Phase 5B manuscript and reviewer-response drafting under this claim boundary. It does not resolve the separate provenance/licence, ethics/declaration, Git-history, release, tag, or DOI blockers."
                if approval_status == "approved"
                else "The package does not edit or authorize edits to the manuscript or bibliography. It also does not resolve the separate provenance/licence, ethics/declaration, Git-history, release, tag, or DOI blockers."
            ),
            "",
            "## Contents",
            "",
            "- `CLAIM_MATRIX.csv` and `CLAIM_MATRIX.md`: machine-readable and review-readable claim sets.",
            "- `CLAIM_BOUNDARIES.md`: global non-negotiable language boundaries.",
            "- `SOURCE_REGISTER.csv`: source hashes, sizes, claim coverage, and parent manifest hashes.",
            "- `APPROVAL_REQUEST.md` and `approval_record.json`: the exact digest and explicit decision record.",
            "- `provenance_receipt.json` and `manifest.json`: offline generation identity and closed-world hashes.",
            "",
        ]
    )
    outputs = {
        "README.md": readme.encode("utf-8"),
        "CLAIM_MATRIX.csv": _csv_bytes(fields, rows),
        "CLAIM_MATRIX.md": _claim_matrix_md(claims, approval_status).encode("utf-8"),
        "CLAIM_BOUNDARIES.md": _boundaries_md(payload).encode("utf-8"),
        "SOURCE_REGISTER.csv": _csv_bytes(source_fields, source_rows),
        "APPROVAL_REQUEST.md": _approval_request(payload, claim_set_sha256).encode("utf-8"),
        "approval_record.json": _json_bytes(approval_record),
        "provenance_receipt.json": _json_bytes(provenance),
    }
    return outputs, provenance


def _manifest(outputs: Mapping[str, bytes], generation_commit: str, contract_sha256: str) -> dict[str, Any]:
    return {
        "contract_sha256": contract_sha256,
        "file_count_excluding_manifest": len(outputs),
        "files": [
            {"path": name, "sha256": _sha256_bytes(content), "size_bytes": len(content)}
            for name, content in sorted(outputs.items())
        ],
        "generation_commit": generation_commit,
        "package_kind": "phase5a_sentence_level_claim_matrix",
        "schema_version": 1,
    }


def export_claim_matrix_package_v3(
    contract_path: Path | str = DEFAULT_CONTRACT,
    output_dir: Path | str = DEFAULT_OUTPUT,
    *,
    generation_commit: str | None = None,
    require_clean_git: bool = True,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Atomically publish the deterministic pending-approval claim package."""

    if require_clean_git:
        _require(_git_is_clean(), "Claim-matrix export requires a clean Git worktree.")
    commit = generation_commit or _git_head()
    _require(GIT_SHA1_RE.fullmatch(commit) is not None or commit == "test", "Generation commit is invalid.")
    outputs, provenance = _build_outputs(contract_path, commit)
    contract_sha256 = provenance["contract_sha256"]
    manifest = _manifest(outputs, commit, contract_sha256)
    outputs = dict(outputs)
    outputs[MANIFEST_NAME] = _json_bytes(manifest)

    output = Path(output_dir)
    _require(replace_existing or not output.exists(), f"Claim-matrix output already exists: {output.as_posix()}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for name, content in outputs.items():
            path = staging / name
            with path.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        validate_claim_matrix_package_v3(staging, contract_path=contract_path)
        if output.exists():
            _require(output.is_dir(), f"Claim-matrix output is not a directory: {output.as_posix()}.")
            _require(output.resolve().parent == staging.resolve().parent, "Replacement target escaped the intended package parent.")
            backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
            output.replace(backup)
            try:
                staging.replace(output)
            except BaseException:
                backup.replace(output)
                raise
            shutil.rmtree(backup)
        else:
            staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_claim_matrix_package_v3(output, contract_path=contract_path)


def validate_claim_matrix_package_v3(
    output_dir: Path | str = DEFAULT_OUTPUT,
    *,
    contract_path: Path | str = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Independently validate the package inventory, bytes, and source derivation."""

    output = Path(output_dir)
    _require(output.is_dir(), f"Claim-matrix package is absent: {output.as_posix()}.")
    names = {path.name for path in output.iterdir() if path.is_file()}
    _require(names == EXPECTED_OUTPUTS, "Claim-matrix package closed-world inventory drifted.")
    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    _require(manifest.get("package_kind") == "phase5a_sentence_level_claim_matrix", "Claim-matrix manifest kind drifted.")
    generation_commit = str(manifest.get("generation_commit", ""))
    _require(GIT_SHA1_RE.fullmatch(generation_commit) is not None or generation_commit == "test", "Manifest generation commit is invalid.")
    expected_outputs, provenance = _build_outputs(contract_path, generation_commit)
    expected_manifest = _manifest(expected_outputs, generation_commit, provenance["contract_sha256"])
    _require(manifest == expected_manifest, "Claim-matrix manifest content drifted.")
    for name, expected in expected_outputs.items():
        _require((output / name).read_bytes() == expected, f"Claim-matrix derived file drifted: {name}.")
    approval = json.loads((output / "approval_record.json").read_text(encoding="utf-8"))
    _require(approval["decision"] == provenance["approval_status"], "Claim-matrix package approval state drifted.")
    return {
        "status": provenance["status"],
        "package_dir": output.as_posix(),
        "file_count": len(names),
        "size_bytes": sum(path.stat().st_size for path in output.iterdir() if path.is_file()),
        "manifest_sha256": _sha256(output / MANIFEST_NAME),
        "generation_commit": generation_commit,
        "contract_sha256": provenance["contract_sha256"],
        "claim_set_sha256": provenance["claim_set_sha256"],
        "claim_count": provenance["claim_count"],
        "numerical_claim_count": provenance["numerical_claim_count"],
        "narrative_claim_count": provenance["narrative_claim_count"],
        "source_file_count": provenance["source_file_count"],
        "component_count": provenance["component_count"],
        "approval_status": provenance["approval_status"],
        "manuscript_editing_authorized": provenance["manuscript_editing_authorized"],
        "paid_api_calls": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generation-commit")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--validate-contract-only", action="store_true")
    parser.add_argument("--validate-package-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require(not (args.validate_contract_only and args.validate_package_only), "Choose one validation mode.")
    if args.validate_contract_only:
        result = validate_claim_matrix_contract_v3(args.contract)
    elif args.validate_package_only:
        result = validate_claim_matrix_package_v3(args.output_dir, contract_path=args.contract)
    else:
        result = export_claim_matrix_package_v3(
            args.contract,
            args.output_dir,
            generation_commit=args.generation_commit,
            require_clean_git=not args.allow_dirty,
            replace_existing=args.replace_output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
