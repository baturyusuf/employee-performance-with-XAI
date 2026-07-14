"""Build and validate a history-free, allowlisted technical publication export.

This contract is deliberately narrower than a release package.  It proves that
the current commit tree does not track the declared local HR-data paths and
that an allowlisted ``git archive`` ZIP contains neither those paths nor Git
history.  It does not rewrite development history, decide redistribution
rights, publish a release, or promote scientific artifacts.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from src.governance.manuscript_contract import sha256_file, source_tree_hash
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_CONFIG = Path("configs/publication_export.yaml")
DEFAULT_OUTPUT = Path(
    "reports/research_log/finalization_v2/12_publication_export_receipt.json"
)
VALIDATOR_SOURCE_PATH = Path("src/governance/sanitized_publication_export.py")
ARCHIVE_METHOD = "git_archive_exact_commit_allowlist_zip_v1"
_MACHINE_PATH = re.compile(
    r"(?i)(?:[A-Z]:[\\/]Users[\\/][A-Za-z0-9._-]+[\\/]|/home/[A-Za-z0-9._-]+/|file:///)"
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*[\"'][A-Za-z0-9_-]{20,}[\"']"),
)
_TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


class PublicationExportError(RuntimeError):
    """Raised when current-tip sanitation or export validation fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _portable_path(root: Path, value: Any, *, field: str) -> tuple[str, Path]:
    raw = value.as_posix() if isinstance(value, Path) else str(value)
    if not raw or raw != raw.strip() or "\\" in raw or re.match(r"^[A-Za-z]:", raw):
        raise PublicationExportError(f"{field} must be a portable repository-relative path.")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise PublicationExportError(f"{field} is unsafe: {raw!r}.")
    resolved = (root / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PublicationExportError(f"{field} escapes the repository root.") from exc
    return relative.as_posix(), resolved


def _run_git(root: Path, arguments: Sequence[str], *, text: bool = True) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=text,
            encoding="utf-8" if text else None,
            errors="replace" if text else None,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = ""
        if isinstance(exc, subprocess.CalledProcessError):
            raw = exc.stderr
            stderr = raw.strip() if isinstance(raw, str) else raw.decode("utf-8", "replace").strip()
        detail = f": {stderr}" if stderr else ""
        raise PublicationExportError(f"Git command failed ({' '.join(arguments)}){detail}") from exc
    return completed.stdout


def _git_identity(root: Path, *, require_clean: bool) -> tuple[str, bool]:
    head = str(_run_git(root, ["rev-parse", "HEAD"])).strip()
    status = str(
        _run_git(root, ["status", "--porcelain", "--untracked-files=all"])
    ).strip()
    dirty = bool(status)
    if require_clean and dirty:
        raise PublicationExportError("Publication export requires a clean worktree.")
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise PublicationExportError("Git HEAD is not a full commit identity.")
    return head, dirty


def _tracked_paths(root: Path, commit: str) -> tuple[str, ...]:
    raw = _run_git(
        root,
        ["ls-tree", "-r", "--name-only", "-z", commit],
        text=False,
    )
    assert isinstance(raw, bytes)
    paths = tuple(item.decode("utf-8") for item in raw.split(b"\0") if item)
    for path in paths:
        _portable_path(root, path, field="tracked path")
    if len(paths) != len(set(paths)):
        raise PublicationExportError("Git tree contains duplicate path identities.")
    return paths


def _path_matches(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _forbidden_paths(
    paths: Sequence[str],
    *,
    patterns: Sequence[str],
    allowed_placeholders: Sequence[str],
) -> tuple[str, ...]:
    allowed = set(allowed_placeholders)
    return tuple(path for path in paths if path not in allowed and _path_matches(path, patterns))


def _selected_paths(paths: Sequence[str], includes: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(
        path
        for path in paths
        if any(path == include or path.startswith(f"{include}/") for include in includes)
    )
    missing = [
        include
        for include in includes
        if not any(path == include or path.startswith(f"{include}/") for path in paths)
    ]
    if missing:
        raise PublicationExportError(f"Allowlisted export paths are absent at the commit: {missing}")
    if not selected:
        raise PublicationExportError("Publication export allowlist selected no files.")
    return selected


def _local_preservation_summary(
    root: Path,
    entries: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not entries:
        raise PublicationExportError("local_preservation must not be empty.")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        relative, path = _portable_path(root, entry.get("path"), field=f"local_preservation[{index}].path")
        if relative in seen:
            raise PublicationExportError(f"Duplicate local preservation path: {relative}")
        seen.add(relative)
        if not path.is_file():
            raise PublicationExportError(f"Preserved local file is missing: {relative}")
        expected_hash = str(entry.get("sha256", ""))
        expected_size = int(entry.get("size_bytes", -1))
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        if actual_hash != expected_hash or actual_size != expected_size:
            raise PublicationExportError(f"Preserved local file bytes changed: {relative}")
        rows.append(
            {
                "path": relative,
                "sha256": actual_hash,
                "size_bytes": actual_size,
                "category": str(entry.get("category", "")),
            }
        )
    return {
        "count": len(rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "inventory_sha256": _sha256_json(rows),
        "paths": tuple(row["path"] for row in rows),
    }


def _is_portable_content_path(path: str, prefixes: Sequence[str]) -> bool:
    return any(
        path == prefix.rstrip("/") or path.startswith(prefix)
        for prefix in prefixes
    )


def _archive_evidence(
    archive_path: Path,
    *,
    expected_paths: Sequence[str],
    forbidden_patterns: Sequence[str],
    allowed_placeholders: Sequence[str],
    portable_content_prefixes: Sequence[str],
    maximum_archive_bytes: int,
    maximum_member_bytes: int,
) -> Mapping[str, Any]:
    archive_size = archive_path.stat().st_size
    if archive_size < 1 or archive_size > maximum_archive_bytes:
        raise PublicationExportError(
            f"Archive size {archive_size} is outside the declared limit {maximum_archive_bytes}."
        )
    manifest: list[dict[str, Any]] = []
    portable_findings: list[str] = []
    secret_findings: list[str] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in members]
        if len(names) != len(set(names)):
            raise PublicationExportError("Archive contains duplicate member paths.")
        if tuple(sorted(names)) != tuple(sorted(expected_paths)):
            missing = sorted(set(expected_paths) - set(names))
            extra = sorted(set(names) - set(expected_paths))
            raise PublicationExportError(
                f"Archive is not closed-world for the allowlist; missing={missing[:10]}, extra={extra[:10]}."
            )
        forbidden = _forbidden_paths(
            names,
            patterns=forbidden_patterns,
            allowed_placeholders=allowed_placeholders,
        )
        if forbidden:
            raise PublicationExportError(f"Archive contains forbidden local-data paths: {forbidden}")
        for info in members:
            name, _ = _portable_path(Path.cwd(), info.filename, field="archive member path")
            if name.startswith(".git/") or name == ".git":
                raise PublicationExportError("Archive contains Git metadata.")
            mode = info.external_attr >> 16
            if mode and stat.S_ISLNK(mode):
                raise PublicationExportError(f"Archive contains a symbolic link: {name}")
            if info.flag_bits & 0x1:
                raise PublicationExportError(f"Archive contains an encrypted member: {name}")
            if info.file_size > maximum_member_bytes:
                raise PublicationExportError(
                    f"Archive member exceeds the size limit: {name} ({info.file_size})."
                )
            content = archive.read(info)
            digest = hashlib.sha256(content).hexdigest()
            manifest.append({"path": name, "size_bytes": len(content), "sha256": digest})
            suffix = PurePosixPath(name).suffix.casefold()
            if suffix not in _TEXT_SUFFIXES:
                continue
            text = content.decode("utf-8", "replace")
            if _is_portable_content_path(name, portable_content_prefixes) and _MACHINE_PATH.search(text):
                portable_findings.append(name)
            if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
                secret_findings.append(name)
    if portable_findings:
        raise PublicationExportError(
            f"Archive documentation/config contains machine-local paths: {sorted(set(portable_findings))}"
        )
    if secret_findings:
        raise PublicationExportError(
            f"Archive contains secret-like text: {sorted(set(secret_findings))}"
        )
    manifest.sort(key=lambda row: row["path"])
    return {
        "archive_sha256": sha256_file(archive_path),
        "archive_size_bytes": archive_size,
        "member_count": len(manifest),
        "member_total_size_bytes": sum(row["size_bytes"] for row in manifest),
        "member_manifest_sha256": _sha256_json(manifest),
        "forbidden_member_count": 0,
        "portable_content_finding_count": 0,
        "secret_finding_count": 0,
        "git_metadata_present": False,
        "symlink_member_count": 0,
    }


def _build_archive_probe(
    root: Path,
    *,
    commit: str,
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    tracked = _tracked_paths(root, commit)
    patterns = [str(value) for value in contract.get("forbidden_tracked_globs", [])]
    allowed = [str(value) for value in contract.get("allowed_placeholder_paths", [])]
    forbidden = _forbidden_paths(tracked, patterns=patterns, allowed_placeholders=allowed)
    if forbidden:
        raise PublicationExportError(f"Commit tree still tracks forbidden local-data paths: {forbidden}")
    required = {str(value) for value in contract.get("required_tracked_paths", [])}
    missing_required = sorted(required - set(tracked))
    if missing_required:
        raise PublicationExportError(f"Required documentation/placeholders are untracked: {missing_required}")
    includes = [str(value) for value in contract.get("include_paths", [])]
    selected = _selected_paths(tracked, includes)
    with tempfile.TemporaryDirectory(prefix="publication-export-") as temporary:
        archive_path = Path(temporary) / "sanitized_publication_tip.zip"
        _run_git(
            root,
            [
                "archive",
                "--format=zip",
                f"--output={archive_path}",
                commit,
                "--",
                *includes,
            ],
        )
        evidence = _archive_evidence(
            archive_path,
            expected_paths=selected,
            forbidden_patterns=patterns,
            allowed_placeholders=allowed,
            portable_content_prefixes=[
                str(value) for value in contract.get("portable_content_prefixes", [])
            ],
            maximum_archive_bytes=int(contract.get("maximum_archive_bytes", 0)),
            maximum_member_bytes=int(contract.get("maximum_member_bytes", 0)),
        )
    return {
        **evidence,
        "git_tree_file_count": len(tracked),
        "allowlisted_member_count": len(selected),
        "tracked_forbidden_path_count": 0,
        "required_tracked_path_count": len(required),
        "include_path_count": len(includes),
        "archive_retained": False,
        "temporary_archive_removed": True,
        "history_included": False,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception as exc:
        try:
            os.unlink(temporary)
        except OSError as cleanup_exc:
            exc.add_note(
                "Atomic export-receipt cleanup also failed: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            )
        raise


def _load_contract(root: Path, config_path: str | Path) -> tuple[str, Path, Mapping[str, Any]]:
    relative, path = _portable_path(root, config_path, field="publication export config")
    payload = load_config(path)
    contract = payload.get("publication_export")
    if not isinstance(contract, Mapping):
        raise PublicationExportError("publication_export contract is missing.")
    if (
        contract.get("schema_version") != 1
        or contract.get("contract_status") != "approved_d3_forward_tip_sanitation"
        or contract.get("archive_method") != ARCHIVE_METHOD
    ):
        raise PublicationExportError("Publication export contract identity drifted.")
    return relative, path, contract


def validate_receipt(
    receipt_path: str | Path = DEFAULT_OUTPUT,
    *,
    project_root: str | Path = PROJECT_ROOT,
    rebuild_archive: bool = True,
) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    candidate = Path(receipt_path)
    if candidate.is_absolute():
        path = candidate.resolve()
        try:
            receipt_relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise PublicationExportError("Publication export receipt escapes the repository root.") from exc
    else:
        receipt_relative, path = _portable_path(
            root, receipt_path, field="publication export receipt"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("contract_kind") != "sanitized_publication_export":
        raise PublicationExportError("Publication export receipt is not a passing supported contract.")
    required_true = (
        "canonical_eligible",
        "local_files_preserved",
        "current_tip_raw_paths_absent",
        "archive_closed_world",
        "archive_paths_portable",
        "archive_bytes_verified",
    )
    if any(payload.get(field) is not True for field in required_true):
        raise PublicationExportError("Publication export receipt boolean invariants failed.")
    required_false = (
        "git_worktree_dirty",
        "archive_retained",
        "history_included",
        "git_metadata_present",
        "raw_employee_values_in_receipt",
    )
    if any(payload.get(field) is not False for field in required_false):
        raise PublicationExportError("Publication export receipt false invariants failed.")
    if payload.get("network_calls") != 0 or payload.get("paid_api_calls") != 0:
        raise PublicationExportError("Publication export must be offline and API-free.")
    config_relative, config_path, contract = _load_contract(
        root, payload.get("config_path", DEFAULT_CONFIG.as_posix())
    )
    if (
        config_relative != payload.get("config_path")
        or sha256_file(config_path) != payload.get("config_sha256")
        or receipt_relative != contract.get("receipt_path")
    ):
        raise PublicationExportError("Publication export config or receipt path changed.")
    if payload.get("source_tree_hash") != source_tree_hash(root):
        raise PublicationExportError("Scientific source tree changed after export validation.")
    if (
        payload.get("validator_source_path") != VALIDATOR_SOURCE_PATH.as_posix()
        or payload.get("validator_source_sha256") != sha256_file(Path(__file__))
    ):
        raise PublicationExportError("Publication export validator source changed.")
    commit = str(payload.get("git_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PublicationExportError("Publication export receipt lacks a Git commit identity.")
    _run_git(root, ["cat-file", "-e", f"{commit}^{{commit}}"])
    preservation = _local_preservation_summary(
        root, [entry for entry in contract.get("local_preservation", []) if isinstance(entry, Mapping)]
    )
    if (
        preservation["count"] != payload.get("local_preserved_file_count")
        or preservation["total_size_bytes"] != payload.get("local_preserved_total_size_bytes")
        or preservation["inventory_sha256"] != payload.get("local_preservation_inventory_sha256")
    ):
        raise PublicationExportError("Local preservation evidence changed.")
    if rebuild_archive:
        rebuilt = _build_archive_probe(root, commit=commit, contract=contract)
        fields = (
            "archive_sha256",
            "archive_size_bytes",
            "member_count",
            "member_total_size_bytes",
            "member_manifest_sha256",
            "git_tree_file_count",
            "allowlisted_member_count",
            "required_tracked_path_count",
            "include_path_count",
        )
        if any(rebuilt[field] != payload.get(field) for field in fields):
            raise PublicationExportError("Rebuilt sanitized archive evidence differs from the receipt.")
    return payload


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    project_root: str | Path = PROJECT_ROOT,
    require_clean: bool = True,
) -> Mapping[str, Any]:
    """Build an ephemeral sanitized export and write its compact receipt."""

    root = Path(project_root).resolve()
    config_relative, config, contract = _load_contract(root, config_path)
    output_relative, output = _portable_path(root, output_path, field="publication export receipt")
    if output_relative != contract.get("receipt_path"):
        raise PublicationExportError("Requested receipt differs from the export contract.")
    if output.exists():
        raise PublicationExportError(f"Publication export receipt already exists: {output_relative}")
    head, dirty = _git_identity(root, require_clean=require_clean)
    preservation = _local_preservation_summary(
        root, [entry for entry in contract.get("local_preservation", []) if isinstance(entry, Mapping)]
    )
    archive = _build_archive_probe(root, commit=head, contract=contract)
    payload = {
        "schema_version": 1,
        "contract_kind": "sanitized_publication_export",
        "status": "passed",
        "scope": contract.get("scope"),
        "claim_boundary": contract.get("claim_boundary"),
        "git_commit": head,
        "git_worktree_dirty": dirty,
        "source_tree_hash": source_tree_hash(root),
        "config_path": config_relative,
        "config_sha256": sha256_file(config),
        "validator_source_path": VALIDATOR_SOURCE_PATH.as_posix(),
        "validator_source_sha256": sha256_file(Path(__file__)),
        "archive_method": ARCHIVE_METHOD,
        **archive,
        "local_preserved_file_count": preservation["count"],
        "local_preserved_total_size_bytes": preservation["total_size_bytes"],
        "local_preservation_inventory_sha256": preservation["inventory_sha256"],
        "local_files_preserved": True,
        "current_tip_raw_paths_absent": archive["tracked_forbidden_path_count"] == 0,
        "archive_closed_world": archive["member_count"] == archive["allowlisted_member_count"],
        "archive_paths_portable": archive["portable_content_finding_count"] == 0,
        "archive_bytes_verified": True,
        "raw_employee_values_in_receipt": False,
        "canonical_eligible": require_clean and not dirty,
        "network_calls": 0,
        "paid_api_calls": 0,
        "validated_at": _utc_now(),
    }
    _atomic_write_json(output, payload)
    validate_receipt(output, project_root=root, rebuild_archive=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate a history-free allowlisted publication export."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    payload = run(arguments.config, output_path=arguments.output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "receipt": arguments.output,
                "git_commit": payload["git_commit"],
                "local_preserved_file_count": payload["local_preserved_file_count"],
                "archive_member_count": payload["member_count"],
                "archive_retained": payload["archive_retained"],
                "history_included": payload["history_included"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHIVE_METHOD",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT",
    "PublicationExportError",
    "run",
    "validate_receipt",
]
