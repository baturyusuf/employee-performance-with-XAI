"""Fail-closed dependency split, lock, and clean-environment receipt contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.governance.manuscript_contract import source_tree_hash
from src.utils.config_loader import PROJECT_ROOT


DEFAULT_CONFIG = Path("configs/dependency_contract.yaml")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
_EXACT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)=="
    r"(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]*)"
    r"(?:;\s*(?P<marker>.+))?$"
)
_WINDOWS_MARKER = 'platform_system == "Windows"'


class DependencyContractError(RuntimeError):
    """Raised when a dependency file, lock, environment, or receipt fails."""


def _normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_sha256(lines: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _portable_path(raw: Any, *, field: str) -> str:
    if isinstance(raw, Path):
        raw = raw.as_posix()
    if not isinstance(raw, str) or not raw or raw != raw.strip() or "\\" in raw:
        raise DependencyContractError(f"Invalid portable {field}: {raw!r}.")
    candidate = Path(raw)
    if (
        candidate.is_absolute()
        or candidate.drive
        or raw.startswith("./")
        or any(part in {"", ".", ".."} for part in raw.split("/"))
    ):
        raise DependencyContractError(f"Invalid portable {field}: {raw!r}.")
    return raw


def _contained(project_root: Path, raw: Any, *, field: str) -> Path:
    relative = _portable_path(raw, field=field)
    root = project_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DependencyContractError(f"{field} escapes the repository: {relative}.") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise DependencyContractError(f"{field} is missing or link-like: {relative}.")
    return resolved


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DependencyContractError(f"Cannot load dependency contract: {path.as_posix()}.") from exc
    contract = payload.get("dependency_contract") if isinstance(payload, Mapping) else None
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise DependencyContractError("Dependency contract schema_version must equal 1.")
    return contract


def _parse_requirement_file(path: Path) -> dict[str, Any]:
    includes: list[str] = []
    constraints: list[str] = []
    packages: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            includes.append(_portable_path(line[3:].strip(), field=f"include at {path.name}:{line_number}"))
            continue
        if line.startswith("-c "):
            constraints.append(
                _portable_path(line[3:].strip(), field=f"constraint at {path.name}:{line_number}")
            )
            continue
        if line.startswith("-") or "://" in line or " @ " in line or line.startswith("."):
            raise DependencyContractError(f"Unsafe requirement directive at {path.name}:{line_number}.")
        match = _NAME_RE.match(line)
        if match is None:
            raise DependencyContractError(f"Invalid requirement at {path.name}:{line_number}.")
        name = _normalize_name(match.group(0))
        specifier = line[match.end() :].strip()
        range_specifier, separator, marker = specifier.partition(";")
        marker = marker.strip()
        if not range_specifier.strip().startswith(">=") or ",<" not in range_specifier:
            raise DependencyContractError(
                f"Direct requirement must have a bounded >=,< range at {path.name}:{line_number}."
            )
        if separator and (name != "pywin32" or marker != _WINDOWS_MARKER):
            raise DependencyContractError(
                f"Unsupported direct requirement marker at {path.name}:{line_number}."
            )
        if name in packages:
            raise DependencyContractError(f"Duplicate requirement {name!r} in {path.name}.")
        packages[name] = specifier
    return {
        "includes": sorted(includes),
        "constraints": sorted(constraints),
        "packages": packages,
    }


def _parse_constraints(path: Path) -> dict[str, dict[str, str | None]]:
    pins: dict[str, dict[str, str | None]] = {}
    order: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") or "://" in line or " @ " in line:
            raise DependencyContractError(f"Unsafe lock directive at {path.name}:{line_number}.")
        match = _EXACT_RE.fullmatch(line)
        if match is None:
            raise DependencyContractError(f"Lock entry is not an exact pin at {path.name}:{line_number}.")
        name = _normalize_name(match.group("name"))
        marker = match.group("marker")
        if marker is not None and marker != _WINDOWS_MARKER:
            raise DependencyContractError(f"Unsupported lock marker for {name!r}: {marker!r}.")
        if name in pins:
            raise DependencyContractError(f"Duplicate lock pin {name!r}.")
        pins[name] = {"version": match.group("version"), "marker": marker}
        order.append(name)
    if order != sorted(order):
        raise DependencyContractError("Lock entries must be sorted by normalized distribution name.")
    return pins


def _expected_names(raw: Any, *, field: str) -> list[str]:
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise DependencyContractError(f"{field} must be a string list.")
    normalized = [_normalize_name(value) for value in raw]
    if normalized != sorted(set(normalized)):
        raise DependencyContractError(f"{field} must be normalized, unique, and sorted.")
    return normalized


def validate_contract(
    project_root: str | Path = PROJECT_ROOT,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Validate the closed dependency taxonomy and exact constraints lock."""

    root = Path(project_root).resolve()
    config_relative = _portable_path(config_path, field="config path")
    config_file = _contained(root, config_relative, field="config path")
    contract = _load_config(config_file)
    python_minor = contract.get("python_minor")
    if not isinstance(python_minor, str) or not re.fullmatch(r"3\.\d+", python_minor):
        raise DependencyContractError("python_minor must be an explicit Python 3 minor.")

    constraints_relative = _portable_path(contract.get("constraints_path"), field="constraints path")
    constraints_file = _contained(root, constraints_relative, field="constraints path")
    lock = _parse_constraints(constraints_file)

    groups_raw = contract.get("groups")
    expected_group_names = ["core", "development", "legacy_optional", "supplementary"]
    if not isinstance(groups_raw, Mapping) or sorted(groups_raw) != expected_group_names:
        raise DependencyContractError(f"Dependency groups must be exactly {expected_group_names}.")
    groups: dict[str, dict[str, Any]] = {}
    file_hashes: dict[str, str] = {
        config_relative: _sha256(config_file),
        constraints_relative: _sha256(constraints_file),
    }
    for group_name in expected_group_names:
        raw_group = groups_raw[group_name]
        if not isinstance(raw_group, Mapping):
            raise DependencyContractError(f"Invalid dependency group: {group_name}.")
        relative = _portable_path(raw_group.get("path"), field=f"{group_name} path")
        group_file = _contained(root, relative, field=f"{group_name} path")
        parsed = _parse_requirement_file(group_file)
        expected_includes = sorted(
            _portable_path(value, field=f"{group_name} include")
            for value in raw_group.get("includes", [])
        )
        expected_constraints = sorted(
            _portable_path(value, field=f"{group_name} constraint")
            for value in raw_group.get("constraints", [])
        )
        expected_direct = _expected_names(
            raw_group.get("direct_packages"), field=f"{group_name}.direct_packages"
        )
        if parsed["includes"] != expected_includes:
            raise DependencyContractError(f"{group_name} include set differs from config.")
        if parsed["constraints"] != expected_constraints:
            raise DependencyContractError(f"{group_name} constraint set differs from config.")
        if sorted(parsed["packages"]) != expected_direct:
            raise DependencyContractError(f"{group_name} direct package set differs from config.")
        for include in expected_includes:
            _contained(root, include, field=f"{group_name} include")
        for constraint in expected_constraints:
            _contained(root, constraint, field=f"{group_name} constraint")
        groups[group_name] = {"path": relative, **parsed}
        file_hashes[relative] = _sha256(group_file)

    compatibility_relative = _portable_path(
        contract.get("compatibility_entrypoint"), field="compatibility entrypoint"
    )
    compatibility_file = _contained(root, compatibility_relative, field="compatibility entrypoint")
    compatibility = _parse_requirement_file(compatibility_file)
    expected_compatibility = sorted(
        _portable_path(value, field="compatibility include")
        for value in contract.get("compatibility_includes", [])
    )
    if compatibility["includes"] != expected_compatibility or compatibility["constraints"] or compatibility["packages"]:
        raise DependencyContractError("Compatibility entrypoint must include only the configured core file.")
    file_hashes[compatibility_relative] = _sha256(compatibility_file)

    forbidden = set(_expected_names(contract.get("forbidden_core_packages"), field="forbidden_core_packages"))
    core_and_supplementary = set(groups["core"]["packages"]) | set(groups["supplementary"]["packages"])
    if forbidden & core_and_supplementary:
        raise DependencyContractError(
            f"Forbidden packages entered core/supplementary: {sorted(forbidden & core_and_supplementary)}."
        )
    declared_direct = set().union(*(set(group["packages"]) for group in groups.values()))
    missing_pins = sorted(declared_direct - set(lock))
    if missing_pins:
        raise DependencyContractError(f"Direct packages missing from lock: {missing_pins}.")

    environment_file = _contained(root, "environment.yml", field="environment file")
    environment_text = environment_file.read_text(encoding="utf-8")
    if f"  - python={python_minor}\n" not in environment_text.replace("\r\n", "\n"):
        raise DependencyContractError("environment.yml does not pin the configured Python minor.")
    if "      - -r requirements-dev.txt\n" not in environment_text.replace("\r\n", "\n"):
        raise DependencyContractError("environment.yml does not install requirements-dev.txt.")
    file_hashes["environment.yml"] = _sha256(environment_file)

    return {
        "schema_version": 1,
        "python_minor": python_minor,
        "config_path": config_relative,
        "constraints_path": constraints_relative,
        "compatibility_entrypoint": compatibility_relative,
        "groups": groups,
        "lock": lock,
        "lock_package_count": len(lock),
        "declared_direct_package_count": len(declared_direct),
        "forbidden_core_packages": sorted(forbidden),
        "bootstrap_packages": _expected_names(contract.get("bootstrap_packages"), field="bootstrap_packages"),
        "receipt_path": _portable_path(contract.get("receipt_path"), field="receipt path"),
        "claim_boundary": str(contract.get("claim_boundary", "")),
        "file_sha256": dict(sorted(file_hashes.items())),
    }


def _installed_distributions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = _normalize_name(raw_name)
        version = distribution.version
        if name in installed and installed[name] != version:
            raise DependencyContractError(f"Multiple installed versions found for {name!r}.")
        installed[name] = version
    return installed


def _pip_check() -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode, output


def validate_environment(contract: Mapping[str, Any], *, profile: str = "core") -> dict[str, Any]:
    """Validate one isolated environment against the applicable exact lock."""

    if profile not in {"core", "development"}:
        raise DependencyContractError("Environment profile must be core or development.")
    expected_minor = str(contract["python_minor"])
    actual_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_minor != expected_minor:
        raise DependencyContractError(
            f"Python minor mismatch: expected {expected_minor}, found {actual_minor}."
        )
    groups = contract["groups"]
    direct_specs = dict(groups["core"]["packages"])
    if profile == "development":
        direct_specs.update(groups["legacy_optional"]["packages"])
        direct_specs.update(groups["development"]["packages"])
        direct_specs.update(groups["supplementary"]["packages"])
    direct = {
        name
        for name, specifier in direct_specs.items()
        if ";" not in specifier or platform.system() == "Windows"
    }
    installed = _installed_distributions()
    missing = sorted(direct - set(installed))
    lock = contract["lock"]
    bootstrap = set(contract["bootstrap_packages"])
    unlocked = sorted(set(installed) - set(lock) - bootstrap)
    mismatched = sorted(
        name
        for name, version in installed.items()
        if name in lock
        and (lock[name]["marker"] is None or platform.system() == "Windows")
        and version != lock[name]["version"]
    )
    if missing or unlocked or mismatched:
        raise DependencyContractError(
            f"Environment lock failure: missing={missing}, unlocked={unlocked}, mismatched={mismatched}."
        )
    pip_code, pip_output = _pip_check()
    if pip_code != 0:
        raise DependencyContractError(f"pip check failed: {pip_output}")
    inventory = [f"{name}=={installed[name]}" for name in sorted(installed) if name not in bootstrap]
    return {
        "profile": profile,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "validated_package_count": len(inventory),
        "installed_inventory_sha256": _semantic_sha256(inventory),
        "direct_versions": {name: installed[name] for name in sorted(direct)},
        "missing_package_count": 0,
        "unlocked_package_count": 0,
        "version_mismatch_count": 0,
        "pip_check_status": "passed",
    }


def _git(project_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException as primary:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError as cleanup_error:
                primary.add_note(f"Atomic receipt cleanup also failed: {cleanup_error}")
        raise


def build_receipt(
    project_root: str | Path = PROJECT_ROOT,
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    profile: str = "core",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one compact receipt from a clean exact Git commit and environment."""

    root = Path(project_root).resolve()
    contract = validate_contract(root, config_path)
    output_relative = _portable_path(
        output_path if output_path is not None else contract["receipt_path"], field="output path"
    )
    output = (root / output_relative).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise DependencyContractError("Receipt output escapes the repository.") from exc
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise DependencyContractError("Dependency receipt requires a clean Git worktree.")
    commit = _git(root, "rev-parse", "HEAD")
    environment = validate_environment(contract, profile=profile)
    receipt = {
        "schema_version": 1,
        "contract_kind": "python_dependency_lock",
        "status": "passed",
        "canonical_eligible": profile == "core",
        "git_commit": commit,
        "git_worktree_dirty": False,
        "source_tree_hash": source_tree_hash(root),
        "config_path": contract["config_path"],
        "config_sha256": contract["file_sha256"][contract["config_path"]],
        "constraints_path": contract["constraints_path"],
        "constraints_sha256": contract["file_sha256"][contract["constraints_path"]],
        "dependency_file_sha256": contract["file_sha256"],
        "lock_package_count": contract["lock_package_count"],
        "declared_direct_package_count": contract["declared_direct_package_count"],
        "core_forbidden_package_count": 0,
        "paid_api_calls": 0,
        "claim_boundary": contract["claim_boundary"],
        **environment,
    }
    _atomic_write_json(output, receipt)
    return receipt


def validate_receipt(
    receipt_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Revalidate a receipt's immutable contract hashes without changing it."""

    root = Path(project_root).resolve()
    receipt_relative = _portable_path(receipt_path, field="receipt path")
    receipt_file = _contained(root, receipt_relative, field="receipt path")
    try:
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DependencyContractError("Dependency receipt is not valid JSON.") from exc
    contract = validate_contract(root, config_path)
    if receipt.get("status") != "passed" or receipt.get("contract_kind") != "python_dependency_lock":
        raise DependencyContractError("Dependency receipt has invalid status or kind.")
    if receipt.get("dependency_file_sha256") != contract["file_sha256"]:
        raise DependencyContractError("Dependency files differ from the receipt.")
    if receipt.get("source_tree_hash") != source_tree_hash(root):
        raise DependencyContractError("Source tree differs from the dependency receipt.")
    generation_commit = str(receipt.get("git_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", generation_commit):
        raise DependencyContractError("Dependency receipt Git commit is invalid.")
    try:
        _git(root, "merge-base", "--is-ancestor", generation_commit, "HEAD")
    except subprocess.CalledProcessError as exc:
        raise DependencyContractError("Dependency receipt commit is not an ancestor of HEAD.") from exc
    if receipt.get("missing_package_count") != 0 or receipt.get("unlocked_package_count") != 0:
        raise DependencyContractError("Dependency receipt records an incomplete environment.")
    if receipt.get("version_mismatch_count") != 0 or receipt.get("pip_check_status") != "passed":
        raise DependencyContractError("Dependency receipt records a broken environment.")
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT.as_posix())
    parser.add_argument("--config", default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--profile", choices=("core", "development"), default="core")
    parser.add_argument("--output", default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.validate_only:
        contract = validate_contract(args.project_root, args.config)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "lock_package_count": contract["lock_package_count"],
                    "declared_direct_package_count": contract["declared_direct_package_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    receipt = build_receipt(
        args.project_root,
        args.config,
        profile=args.profile,
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
