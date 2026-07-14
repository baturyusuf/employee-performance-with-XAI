"""Fail-closed repository and release-candidate gates used by GitHub Actions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

from src.experiments.build_manuscript_evidence import validate_configured_release_candidate
from src.governance.dependency_contract import validate_contract
from src.governance.offline_runtime import enforce_offline_runtime
from src.utils.config_loader import PROJECT_ROOT


class CIRepositoryGateError(RuntimeError):
    """Raised when a repository or release-candidate invariant fails."""


_RAW_DATA_PATTERNS = (
    re.compile(r"^data/raw/(?!\.gitkeep$)"),
    re.compile(r"^data/external/[^/]+/raw\.csv$"),
    re.compile(r"^data/(?:interim|processed)/(?!\.gitkeep$)"),
)
_SECRET_PATTERNS = (
    re.compile(r"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*[\"'][A-Za-z0-9_./+=-]{20,}[\"']"
    ),
)
_MACHINE_PATH = re.compile(r"(?i)\b[A-Z]:[\\/](?:Users|Documents and Settings)[\\/]")
_README_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_paths(root: Path) -> list[str]:
    paths = [value for value in _git(root, "ls-files", "-z").split("\0") if value]
    if not paths:
        raise CIRepositoryGateError("Git tracked-file inventory is empty.")
    return sorted(paths)


def _validate_worktree(root: Path, *, allowed_untracked_root: str | None) -> None:
    entries = [line for line in _git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines() if line]
    unexpected: list[str] = []
    prefix = f"?? {allowed_untracked_root.rstrip('/')}/" if allowed_untracked_root else None
    for entry in entries:
        if prefix is not None and entry.startswith(prefix):
            continue
        unexpected.append(entry)
    if unexpected:
        raise CIRepositoryGateError(f"CI requires a clean worktree: {unexpected[:10]}")


def _validate_issue_register(root: Path) -> dict[str, Any]:
    path = root / "reports/research_log/finalization_v2/02_issue_register.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    identifiers = [str(row.get("issue_id", "")) for row in rows]
    if len(rows) < 32 or len(set(identifiers)) != len(rows):
        raise CIRepositoryGateError("The v2 issue register is truncated or has duplicate IDs.")
    if any(not re.fullmatch(r"V2-\d{3}", identifier) for identifier in identifiers):
        raise CIRepositoryGateError("The v2 issue register contains an invalid issue ID.")
    return {"path": path.relative_to(root).as_posix(), "row_count": len(rows), "sha256": _sha256(path)}


def _validate_readme_links(root: Path) -> dict[str, Any]:
    path = root / "README.md"
    missing: list[str] = []
    checked: list[str] = []
    for raw_target in _README_LINK.findall(path.read_text(encoding="utf-8")):
        target = raw_target.strip().split("#", 1)[0]
        if not target or re.match(r"^(?:https?|mailto):", target, flags=re.IGNORECASE):
            continue
        target = target.strip("<>").replace("%20", " ")
        checked.append(target)
        candidate = (root / target).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            missing.append(target)
            continue
        if not candidate.exists():
            missing.append(target)
    if missing:
        raise CIRepositoryGateError(f"README contains missing or escaping local links: {missing}")
    return {"path": "README.md", "local_link_count": len(checked), "sha256": _sha256(path)}


def _validate_tracked_inventory(root: Path, paths: list[str]) -> dict[str, Any]:
    raw_paths = [path for path in paths if any(pattern.search(path) for pattern in _RAW_DATA_PATTERNS)]
    env_paths = [
        path
        for path in paths
        if Path(path).name == ".env" or (Path(path).name.startswith(".env.") and Path(path).name != ".env.example")
    ]
    large_paths: list[dict[str, Any]] = []
    secret_paths: list[str] = []
    machine_path_findings: list[str] = []
    active_prefixes = (".github/", "configs/", "data/", "src/")
    for relative in paths:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            continue
        size = path.stat().st_size
        if size > 100 * 1024 * 1024:
            large_paths.append({"path": relative, "size_bytes": size})
        if not (relative == "README.md" or relative.startswith(active_prefixes)):
            continue
        if path.suffix.casefold() not in _TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
            secret_paths.append(relative)
        if _MACHINE_PATH.search(text):
            machine_path_findings.append(relative)
    findings = {
        "tracked_file_count": len(paths),
        "raw_data_paths": raw_paths,
        "environment_paths": env_paths,
        "large_paths": large_paths,
        "secret_paths": sorted(set(secret_paths)),
        "machine_path_findings": sorted(set(machine_path_findings)),
    }
    if any(findings[key] for key in findings if key != "tracked_file_count"):
        raise CIRepositoryGateError(f"Tracked inventory gate failed: {json.dumps(findings, sort_keys=True)}")
    return findings


def validate_repository(
    project_root: str | Path = PROJECT_ROOT,
    *,
    allowed_untracked_root: str | None = None,
) -> dict[str, Any]:
    """Validate immutable CI inputs, repository hygiene, and documentation links."""

    root = Path(project_root).resolve()
    if not (root / ".git").exists():
        raise CIRepositoryGateError("CI repository gate requires a Git worktree root.")
    _validate_worktree(root, allowed_untracked_root=allowed_untracked_root)
    paths = _tracked_paths(root)
    dependency = validate_contract(root)
    return {
        "schema_version": 1,
        "gate_kind": "publication_repository_ci",
        "status": "passed",
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_worktree_dirty": False,
        "dependency_config_sha256": dependency["file_sha256"][dependency["config_path"]],
        "dependency_lock_sha256": dependency["file_sha256"][dependency["constraints_path"]],
        "inventory": _validate_tracked_inventory(root, paths),
        "issue_register": _validate_issue_register(root),
        "readme": _validate_readme_links(root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT.as_posix())
    parser.add_argument("--config", default="configs/manuscript_final.yaml")
    parser.add_argument("--release-run-id", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    allowed_root = None
    if args.release_run_id is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.release_run_id):
            raise CIRepositoryGateError("Release run ID is not portable.")
        allowed_root = f"reports/manuscript_final/{args.release_run_id}"
    receipt = validate_repository(args.project_root, allowed_untracked_root=allowed_root)
    if args.release_run_id is not None:
        with enforce_offline_runtime():
            receipt["release_candidate"] = validate_configured_release_candidate(
                args.config,
                run_id=args.release_run_id,
            )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
