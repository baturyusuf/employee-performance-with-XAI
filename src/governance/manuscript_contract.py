"""Canonical configuration and provenance contract for manuscript evidence.

This module deliberately contains no experiment logic.  It provides the shared
boundary that experiment stages use to resolve feature policies, bind outputs to
one run/configuration identity, and reject stale or incompatible artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.utils.config_loader import PROJECT_ROOT, load_config, resolve_config_path


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "manuscript_final.yaml"
MANIFEST_SCHEMA_VERSION = 1

REQUIRED_POLICY_NAMES = frozenset(
    {
        "full_feature_upper_bound",
        "no_salary_hike",
        "no_salary_hike_no_attrition",
        "no_salary_hike_no_attrition_no_department",
        "no_salary_hike_no_attrition_no_department_no_job_role",
        "no_salary_hike_no_attrition_sensitive_retaining_audit",
    }
)

STRUCTURED_FEATURE_FIELDS = frozenset(
    {
        "feature",
        "feature_name",
        "feature_family",
        "raw_feature",
        "raw_feature_name",
        "grouped_feature",
        "grouped_feature_name",
    }
)


class ManuscriptContractError(ValueError):
    """Base class for canonical manuscript-contract failures."""


class ManuscriptConfigError(ManuscriptContractError):
    """Raised when the canonical configuration is incomplete or inconsistent."""


class FeaturePolicyConsistencyError(ManuscriptContractError):
    """Raised when a module-local feature policy conflicts with the canonical one."""


class ForbiddenFeatureError(ManuscriptContractError):
    """Raised when a primary-model input or artifact names an excluded feature."""


class RunManifestError(ManuscriptContractError):
    """Raised when a run manifest or a referenced artifact is incompatible."""


def utc_now_iso() -> str:
    """Return a second-resolution UTC timestamp with an explicit timezone."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path) -> str:
    """Hash a regular file without loading it into memory."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot hash missing or non-file path: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_document(config: Mapping[str, Any]) -> dict[str, Any]:
    if "manuscript_final" in config:
        return dict(config)
    return {"manuscript_final": dict(config)}


def canonical_config_hash(config_or_path: Mapping[str, Any] | str | Path) -> str:
    """Return a semantic SHA-256 hash of the parsed canonical configuration.

    Hashing canonical JSON rather than source bytes means comments, indentation,
    and mapping order cannot create false run incompatibilities.
    """

    if isinstance(config_or_path, (str, Path)):
        config = load_config(config_or_path)
    elif isinstance(config_or_path, Mapping):
        config = _canonical_document(config_or_path)
    else:  # pragma: no cover - protected by the type contract
        raise TypeError("config_or_path must be a mapping or path")
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manuscript_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the validated inner ``manuscript_final`` settings mapping."""

    settings = config.get("manuscript_final")
    if not isinstance(settings, Mapping):
        raise ManuscriptConfigError("Config must contain a top-level 'manuscript_final' mapping.")
    return settings


def _require_mapping(parent: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ManuscriptConfigError(f"{context}.{key} must be a mapping.")
    return value


def _string_list(value: Any, context: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ManuscriptConfigError(f"{context} must be {qualifier} of strings.")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ManuscriptConfigError(f"{context} must contain only non-empty strings.")
    if len(value) != len(set(value)):
        raise ManuscriptConfigError(f"{context} contains duplicate values.")
    return list(value)


def feature_policy_definitions(config: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    settings = manuscript_settings(config)
    policies = _require_mapping(settings, "feature_policies", "manuscript_final")
    definitions = _require_mapping(policies, "definitions", "manuscript_final.feature_policies")
    return definitions  # type: ignore[return-value]


def primary_policy_name(config: Mapping[str, Any]) -> str:
    policies = _require_mapping(manuscript_settings(config), "feature_policies", "manuscript_final")
    name = policies.get("primary_policy")
    if not isinstance(name, str) or not name:
        raise ManuscriptConfigError("manuscript_final.feature_policies.primary_policy must be a name.")
    return name


def primary_policy_definition(config: Mapping[str, Any]) -> Mapping[str, Any]:
    name = primary_policy_name(config)
    definitions = feature_policy_definitions(config)
    definition = definitions.get(name)
    if not isinstance(definition, Mapping):
        raise ManuscriptConfigError(f"Primary feature policy is not defined: {name}")
    return definition


def primary_excluded_features(config: Mapping[str, Any]) -> tuple[str, ...]:
    definition = primary_policy_definition(config)
    return tuple(_string_list(definition.get("excluded_features"), "primary excluded_features"))


def _policy_exclusions(definition: Any, context: str) -> set[str]:
    if isinstance(definition, Mapping):
        raw = definition.get("excluded_features", definition.get("drop"))
    else:
        raw = definition
    if not isinstance(raw, (list, tuple, set, frozenset)):
        raise FeaturePolicyConsistencyError(
            f"{context} must be an exclusion sequence or a mapping containing "
            "'excluded_features' (legacy 'drop' is also recognized)."
        )
    values = list(raw)
    if any(not isinstance(value, str) or not value for value in values):
        raise FeaturePolicyConsistencyError(f"{context} contains a non-string feature name.")
    return set(values)


def canonical_policy_mapping(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return all canonical policy exclusions in deterministic order."""

    return {
        name: tuple(definition["excluded_features"])
        for name, definition in sorted(feature_policy_definitions(config).items())
    }


def validate_policy_consistency(
    config: Mapping[str, Any],
    policy_sources: Mapping[str, Mapping[str, Any]],
    *,
    reject_unknown_policies: bool = False,
) -> None:
    """Reject module-local definitions that disagree with canonical policies.

    ``policy_sources`` maps a source/module label to its named policy mapping.
    Values may be exclusion sequences, canonical ``excluded_features`` mappings,
    or legacy ``drop`` mappings.  Scientific stages should call this during
    preflight if they still accept externally supplied policy definitions.
    """

    canonical = {
        name: set(definition["excluded_features"])
        for name, definition in feature_policy_definitions(config).items()
    }
    for source_name, source_policies in policy_sources.items():
        if not isinstance(source_policies, Mapping):
            raise FeaturePolicyConsistencyError(f"Policy source {source_name!r} is not a mapping.")
        for policy_name, source_definition in source_policies.items():
            if policy_name not in canonical:
                if reject_unknown_policies:
                    raise FeaturePolicyConsistencyError(
                        f"Policy source {source_name!r} defines unknown policy {policy_name!r}."
                    )
                continue
            observed = _policy_exclusions(
                source_definition,
                f"policy {policy_name!r} from source {source_name!r}",
            )
            expected = canonical[policy_name]
            if observed != expected:
                missing = sorted(expected - observed)
                extra = sorted(observed - expected)
                raise FeaturePolicyConsistencyError(
                    f"Policy {policy_name!r} from source {source_name!r} conflicts with the "
                    f"canonical definition; missing={missing}, extra={extra}."
                )


def validate_manuscript_config(config: Mapping[str, Any]) -> None:
    """Validate required sections and cross-section scientific invariants."""

    settings = manuscript_settings(config)
    if settings.get("schema_version") != 1:
        raise ManuscriptConfigError("manuscript_final.schema_version must be 1.")

    required_sections = {
        "package",
        "datasets",
        "target",
        "model",
        "feature_policies",
        "governance_fields",
        "proxy_analysis",
        "evaluation",
        "calibration",
        "shap",
        "counterfactuals",
        "llm_agent_evaluation",
        "chatbot_guardrails",
        "output",
        "seeds",
        "figures",
        "provenance",
    }
    missing_sections = sorted(required_sections - set(settings))
    if missing_sections:
        raise ManuscriptConfigError(f"Canonical config is missing sections: {missing_sections}")
    for section in required_sections:
        _require_mapping(settings, section, "manuscript_final")

    package = _require_mapping(settings, "package", "manuscript_final")
    if package.get("autonomous_hr_decisions_allowed") is not False:
        raise ManuscriptConfigError("Canonical package must prohibit autonomous HR decisions.")

    target = _require_mapping(settings, "target", "manuscript_final")
    target_column = target.get("column")
    labels = target.get("labels")
    if not isinstance(target_column, str) or not target_column:
        raise ManuscriptConfigError("manuscript_final.target.column must be a non-empty string.")
    if labels != [2, 3, 4] or target.get("ordering") != [2, 3, 4]:
        raise ManuscriptConfigError("Primary target labels and ordering must be exactly [2, 3, 4].")

    datasets = _require_mapping(settings, "datasets", "manuscript_final")
    if target.get("primary_dataset") not in datasets:
        raise ManuscriptConfigError("target.primary_dataset must reference a configured dataset.")
    for name, definition in datasets.items():
        if not isinstance(definition, Mapping):
            raise ManuscriptConfigError(f"Dataset {name!r} must be a mapping.")
        for required in ("path", "role", "task_type", "target", "allowed_claim"):
            if not isinstance(definition.get(required), str) or not definition.get(required):
                raise ManuscriptConfigError(f"Dataset {name!r} requires non-empty {required!r}.")

    definitions = feature_policy_definitions(config)
    missing_policies = sorted(REQUIRED_POLICY_NAMES - set(definitions))
    if missing_policies:
        raise ManuscriptConfigError(f"Canonical config is missing feature policies: {missing_policies}")
    for name, definition in definitions.items():
        if not isinstance(definition, Mapping):
            raise ManuscriptConfigError(f"Feature policy {name!r} must be a mapping.")
        _string_list(definition.get("excluded_features"), f"feature policy {name!r}.excluded_features")
        if not isinstance(definition.get("role"), str) or not definition.get("role"):
            raise ManuscriptConfigError(f"Feature policy {name!r} requires a non-empty role.")
        if not isinstance(definition.get("audit_only"), bool):
            raise ManuscriptConfigError(f"Feature policy {name!r}.audit_only must be boolean.")

    primary_name = primary_policy_name(config)
    primary = primary_policy_definition(config)
    if primary.get("role") != "canonical_primary" or primary.get("audit_only") is not False:
        raise ManuscriptConfigError("The primary policy must be non-audit and have role 'canonical_primary'.")

    governance = _require_mapping(settings, "governance_fields", "manuscript_final")
    sensitive = set(_string_list(governance.get("fairness_sensitive_fields"), "fairness_sensitive_fields"))
    identifiers = set(_string_list(governance.get("identifier_fields"), "identifier_fields"))
    outcome_fields = set(
        _string_list(governance.get("outcome_or_post_outcome_fields"), "outcome_or_post_outcome_fields")
    )
    proxy = _require_mapping(settings, "proxy_analysis", "manuscript_final")
    proxy_target = proxy.get("target")
    if not isinstance(proxy_target, str) or not proxy_target:
        raise ManuscriptConfigError("proxy_analysis.target must be a non-empty feature name.")

    expected_primary = sensitive | identifiers | outcome_fields | {proxy_target}
    actual_primary = set(primary["excluded_features"])
    if actual_primary != expected_primary:
        raise ManuscriptConfigError(
            f"Primary policy {primary_name!r} must be the single exact union of sensitive, "
            f"identifier, outcome/post-outcome, and proxy-target exclusions; "
            f"missing={sorted(expected_primary - actual_primary)}, "
            f"extra={sorted(actual_primary - expected_primary)}."
        )
    if target_column not in actual_primary:
        raise ManuscriptConfigError("The primary target must be excluded from model features.")

    audit_name = "no_salary_hike_no_attrition_sensitive_retaining_audit"
    audit_definition = definitions[audit_name]
    expected_audit = identifiers | outcome_fields
    if set(audit_definition["excluded_features"]) != expected_audit or audit_definition["audit_only"] is not True:
        raise ManuscriptConfigError(
            f"{audit_name} must be audit-only and drop only identifier, target, salary-hike, and attrition fields."
        )

    governed_base = definitions["no_salary_hike_no_attrition"]
    if set(governed_base["excluded_features"]) != expected_audit | sensitive:
        raise ManuscriptConfigError(
            "no_salary_hike_no_attrition must add all sensitive exclusions to the pure leakage audit policy."
        )
    department_free = definitions["no_salary_hike_no_attrition_no_department"]
    if set(department_free["excluded_features"]) != set(governed_base["excluded_features"]) | {proxy_target}:
        raise ManuscriptConfigError("The canonical department-free policy must add only the proxy target.")
    strict = definitions["no_salary_hike_no_attrition_no_department_no_job_role"]
    strict_extra = set(strict["excluded_features"]) - set(department_free["excluded_features"])
    if strict_extra != {"EmpJobRole"}:
        raise ManuscriptConfigError("The strict policy must add exactly EmpJobRole to the primary exclusions.")

    metric_rules = _require_mapping(
        _require_mapping(settings, "evaluation", "manuscript_final"),
        "metric_applicability",
        "manuscript_final.evaluation",
    )
    for task in ("binary_attrition_transfer", "binary_turnover_transfer"):
        task_rules = _require_mapping(metric_rules, task, "metric_applicability")
        not_applicable = set(
            _string_list(task_rules.get("not_applicable"), f"metric_applicability.{task}.not_applicable")
        )
        if "severe_error_rate" not in not_applicable:
            raise ManuscriptConfigError(f"severe_error_rate must be N/A for {task}.")

    seeds = _require_mapping(settings, "seeds", "manuscript_final")
    if not seeds or any(not isinstance(seed, int) for seed in seeds.values()):
        raise ManuscriptConfigError("All canonical seeds must be explicit integers.")

    output = _require_mapping(settings, "output", "manuscript_final")
    if not isinstance(output.get("root"), str) or not output.get("root"):
        raise ManuscriptConfigError("output.root must be a non-empty relative path.")


def load_manuscript_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    policy_sources: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load and validate the canonical manuscript configuration."""

    data = load_config(path)
    validate_manuscript_config(data)
    if policy_sources:
        validate_policy_consistency(data, policy_sources)
    return data


def _candidate_matches_raw_feature(candidate: str, raw_feature: str) -> bool:
    text = candidate.strip().strip("'\"")
    # sklearn commonly prefixes transformed columns with ``transformer__``.
    base = text.rsplit("__", 1)[-1]
    base_folded = base.casefold()
    raw_folded = raw_feature.casefold()
    if base_folded == raw_folded:
        return True
    return any(base_folded.startswith(raw_folded + separator) for separator in ("_", "=", "[", ":", "-", " "))


def forbidden_feature_mentions(
    feature_names: Iterable[Any],
    config: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Map forbidden raw feature families to observed raw/encoded names."""

    forbidden = primary_excluded_features(config)
    found: dict[str, set[str]] = {}
    for value in feature_names:
        if value is None:
            continue
        candidate = str(value).strip()
        if not candidate:
            continue
        for raw_feature in forbidden:
            if _candidate_matches_raw_feature(candidate, raw_feature):
                found.setdefault(raw_feature, set()).add(candidate)
    return {name: sorted(values) for name, values in sorted(found.items())}


def validate_primary_feature_names(
    feature_names: Iterable[Any],
    config: Mapping[str, Any],
    *,
    context: str = "primary model input",
) -> None:
    mentions = forbidden_feature_mentions(feature_names, config)
    if mentions:
        raise ForbiddenFeatureError(f"Forbidden feature families appear in {context}: {mentions}")


def _collect_json_feature_values(value: Any, feature_fields: set[str]) -> list[Any]:
    collected: list[Any] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in feature_fields:
                if isinstance(nested, (list, tuple, set)):
                    collected.extend(nested)
                elif not isinstance(nested, Mapping):
                    collected.append(nested)
            collected.extend(_collect_json_feature_values(nested, feature_fields))
    elif isinstance(value, list):
        for nested in value:
            collected.extend(_collect_json_feature_values(nested, feature_fields))
    return collected


def artifact_feature_names(
    artifact_path: str | Path,
    *,
    feature_fields: Iterable[str] = STRUCTURED_FEATURE_FIELDS,
    scan_text: bool = False,
    forbidden_candidates: Iterable[str] = (),
) -> list[Any]:
    """Extract structured feature names (or explicit text mentions) from an artifact."""

    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Primary artifact is missing: {path}")
    fields = {field.casefold() for field in feature_fields}
    suffix = path.suffix.casefold()
    values: list[Any] = []

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames:
                selected = [name for name in reader.fieldnames if name.casefold() in fields]
                for row in reader:
                    values.extend(row.get(name) for name in selected)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        values.extend(_collect_json_feature_values(payload, fields))
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ForbiddenFeatureError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
                values.extend(_collect_json_feature_values(payload, fields))
    elif not scan_text:
        raise ForbiddenFeatureError(
            f"Cannot structurally inspect {path.suffix or 'extensionless'} artifact {path}; "
            "set scan_text=True for a text artifact."
        )

    if scan_text:
        text = path.read_text(encoding="utf-8")
        for candidate in forbidden_candidates:
            pattern = rf"(?<![A-Za-z0-9]){re.escape(candidate)}(?![A-Za-z0-9])"
            if re.search(pattern, text, flags=re.IGNORECASE):
                values.append(candidate)
    return values


def validate_artifact_forbidden_features(
    artifact_path: str | Path,
    config: Mapping[str, Any],
    *,
    feature_fields: Iterable[str] = STRUCTURED_FEATURE_FIELDS,
    scan_text: bool = False,
) -> None:
    """Reject a structured primary-model artifact containing excluded features."""

    forbidden = primary_excluded_features(config)
    names = artifact_feature_names(
        artifact_path,
        feature_fields=feature_fields,
        scan_text=scan_text,
        forbidden_candidates=forbidden,
    )
    validate_primary_feature_names(names, config, context=str(artifact_path))


def _run_git(project_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def source_tree_hash(
    project_root: str | Path,
    *,
    roots: Sequence[str] = ("src", "configs"),
    files: Sequence[str] = ("requirements.txt", "requirements-dev.txt"),
) -> str:
    """Hash experiment/config source content independently of Git state."""

    root = Path(project_root).resolve()
    candidates: set[Path] = set()
    for relative_root in roots:
        directory = root / relative_root
        if directory.is_dir():
            candidates.update(
                path
                for path in directory.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
            )
    for relative_file in files:
        path = root / relative_file
        if path.is_file():
            candidates.add(path)

    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _package_versions(package_names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    for package_name in package_names:
        if package_name.casefold() == "python":
            continue
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = "not_installed"
    return versions


def _resolve_from_root(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def make_run_id(config: Mapping[str, Any], config_hash: str, *, timestamp: datetime | None = None) -> str:
    settings = manuscript_settings(config)
    package = _require_mapping(settings, "package", "manuscript_final")
    prefix = str(package.get("run_id_prefix", "manuscript_final"))
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("_") or "manuscript_final"
    when = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"{safe_prefix}_{when.strftime('%Y%m%dT%H%M%SZ')}_{config_hash[:12]}"


def create_run_manifest(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_id: str | None = None,
    dataset_paths: Mapping[str, str | Path] | None = None,
    initial_command: str | None = None,
) -> dict[str, Any]:
    """Create an in-memory run manifest after validating and hashing all inputs."""

    root = Path(project_root).resolve()
    raw_config_path = Path(config_path)
    rooted_config_path = root / raw_config_path if not raw_config_path.is_absolute() else raw_config_path
    resolved_config_path = (
        rooted_config_path.resolve()
        if rooted_config_path.is_file()
        else resolve_config_path(config_path).resolve()
    )
    config = load_manuscript_config(resolved_config_path)
    config_hash = canonical_config_hash(config)
    settings = manuscript_settings(config)

    datasets = _require_mapping(settings, "datasets", "manuscript_final")
    selected_paths = dataset_paths or {
        name: str(definition["path"])
        for name, definition in datasets.items()
        if isinstance(definition, Mapping)
    }
    dataset_hashes: dict[str, dict[str, Any]] = {}
    for name, raw_path in selected_paths.items():
        path = _resolve_from_root(raw_path, root)
        if not path.is_file():
            raise RunManifestError(f"Configured dataset is missing for {name!r}: {path}")
        definition = datasets.get(name, {})
        dataset_hashes[name] = {
            "path": _portable_path(path, root),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "role": definition.get("role", "") if isinstance(definition, Mapping) else "",
            "task_type": definition.get("task_type", "") if isinstance(definition, Mapping) else "",
        }

    provenance = _require_mapping(settings, "provenance", "manuscript_final")
    package_names = provenance.get("package_names", [])
    if not isinstance(package_names, list):
        raise ManuscriptConfigError("provenance.package_names must be a list.")

    git_commit = _run_git(root, "rev-parse", "HEAD")
    git_status = _run_git(root, "status", "--porcelain", "--untracked-files=all")
    if provenance.get("git_commit_required") is True and git_commit == "unavailable" and root == PROJECT_ROOT.resolve():
        raise RunManifestError("A Git commit is required but could not be resolved.")

    manifest: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id or make_run_id(config, config_hash),
        "git_commit": git_commit,
        "git_worktree_dirty": bool(git_status and git_status != "unavailable"),
        "git_status_sha256": (
            hashlib.sha256(git_status.encode("utf-8")).hexdigest()
            if git_status != "unavailable"
            else "unavailable"
        ),
        "source_tree_hash": source_tree_hash(root),
        "config_path": _portable_path(resolved_config_path, root),
        "config_hash": config_hash,
        "dataset_hashes": dataset_hashes,
        "code_package_versions": _package_versions(package_names),
        "start_timestamp": utc_now_iso(),
        "end_timestamp": None,
        "random_seeds": dict(_require_mapping(settings, "seeds", "manuscript_final")),
        "commands": [],
        "output_files": [],
        "status": "running",
        "failure_information": [],
    }
    if initial_command:
        record_command(manifest, initial_command, stage="entrypoint", status="started")
    return manifest


def record_command(
    manifest: MutableMapping[str, Any],
    command: str,
    *,
    stage: str,
    status: str,
    started_at: str | None = None,
    ended_at: str | None = None,
    return_code: int | None = None,
) -> dict[str, Any]:
    if not isinstance(command, str) or not command.strip():
        raise RunManifestError("A recorded command must be a non-empty string.")
    commands = manifest.setdefault("commands", [])
    if not isinstance(commands, list):
        raise RunManifestError("manifest.commands must be a list.")
    record = {
        "command": command,
        "stage": stage,
        "status": status,
        "started_at": started_at or utc_now_iso(),
        "ended_at": ended_at,
        "return_code": return_code,
    }
    commands.append(record)
    return record


def register_artifact(
    manifest: MutableMapping[str, Any],
    artifact_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    stage: str,
    artifact_type: str,
    artifact_run_id: str | None = None,
    artifact_config_hash: str | None = None,
) -> dict[str, Any]:
    """Hash and register one artifact while enforcing its run/config identity."""

    root = Path(project_root).resolve()
    path = _resolve_from_root(artifact_path, root)
    if not path.is_file():
        raise RunManifestError(f"Cannot register missing artifact: {path}")

    expected_run_id = manifest.get("run_id")
    expected_config_hash = manifest.get("config_hash")
    observed_run_id = artifact_run_id or expected_run_id
    observed_config_hash = artifact_config_hash or expected_config_hash
    if observed_run_id != expected_run_id:
        raise RunManifestError(
            f"Artifact {path} has run_id={observed_run_id!r}; expected {expected_run_id!r}."
        )
    if observed_config_hash != expected_config_hash:
        raise RunManifestError(
            f"Artifact {path} has config_hash={observed_config_hash!r}; expected {expected_config_hash!r}."
        )

    outputs = manifest.setdefault("output_files", [])
    if not isinstance(outputs, list):
        raise RunManifestError("manifest.output_files must be a list.")
    portable = _portable_path(path, root)
    if any(isinstance(item, Mapping) and item.get("path") == portable for item in outputs):
        raise RunManifestError(f"Artifact is already registered: {portable}")

    record = {
        "path": portable,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "run_id": observed_run_id,
        "config_hash": observed_config_hash,
        "stage": stage,
        "artifact_type": artifact_type,
        "registered_at": utc_now_iso(),
    }
    outputs.append(record)
    return record


def record_failure(
    manifest: MutableMapping[str, Any],
    *,
    stage: str,
    error_type: str,
    message: str,
) -> dict[str, str]:
    failures = manifest.setdefault("failure_information", [])
    if not isinstance(failures, list):
        raise RunManifestError("manifest.failure_information must be a list.")
    record = {
        "timestamp": utc_now_iso(),
        "stage": stage,
        "error_type": error_type,
        "message": message,
    }
    failures.append(record)
    return record


def finalize_run_manifest(
    manifest: MutableMapping[str, Any],
    *,
    status: str,
) -> MutableMapping[str, Any]:
    if status not in {"complete", "failed"}:
        raise RunManifestError("Final manifest status must be 'complete' or 'failed'.")
    failures = manifest.get("failure_information", [])
    if status == "complete" and failures:
        raise RunManifestError("A run with recorded failures cannot be finalized as complete.")
    manifest["status"] = status
    manifest["end_timestamp"] = utc_now_iso()
    return manifest


def _load_manifest(manifest_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(manifest_or_path, Mapping):
        return dict(manifest_or_path)
    path = Path(manifest_or_path)
    if not path.is_file():
        raise RunManifestError(f"Run manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunManifestError(f"Invalid run manifest JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunManifestError("Run manifest root must be an object.")
    return payload


def validate_run_manifest(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    expected_config_hash: str | None = None,
    require_complete: bool = False,
    verify_source_tree: bool = False,
) -> dict[str, Any]:
    """Validate identity, hashes, and existence for all manifest references."""

    manifest = _load_manifest(manifest_or_path)
    root = Path(project_root).resolve()
    errors: list[str] = []

    required_fields = {
        "manifest_schema_version",
        "run_id",
        "git_commit",
        "config_path",
        "config_hash",
        "dataset_hashes",
        "code_package_versions",
        "start_timestamp",
        "end_timestamp",
        "random_seeds",
        "commands",
        "output_files",
        "status",
        "failure_information",
        "source_tree_hash",
    }
    missing_fields = sorted(required_fields - set(manifest))
    if missing_fields:
        errors.append(f"missing required fields: {missing_fields}")

    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(f"unsupported manifest schema version: {manifest.get('manifest_schema_version')!r}")
    run_id = manifest.get("run_id")
    config_hash = manifest.get("config_hash")
    if not isinstance(run_id, str) or not run_id:
        errors.append("run_id must be a non-empty string")
    if not isinstance(config_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", config_hash):
        errors.append("config_hash must be a lowercase SHA-256 digest")
    if expected_config_hash is not None and config_hash != expected_config_hash:
        errors.append(f"config_hash {config_hash!r} does not equal expected {expected_config_hash!r}")

    raw_config_path = manifest.get("config_path")
    if isinstance(raw_config_path, str) and raw_config_path:
        config_path = _resolve_from_root(raw_config_path, root)
        if not config_path.is_file():
            errors.append(f"config file is missing: {config_path}")
        else:
            try:
                actual_config_hash = canonical_config_hash(config_path)
            except Exception as exc:  # validation reports all manifest defects together
                errors.append(f"config cannot be loaded or hashed: {exc}")
            else:
                if actual_config_hash != config_hash:
                    errors.append(
                        f"config hash mismatch for {config_path}: manifest={config_hash}, actual={actual_config_hash}"
                    )
    else:
        errors.append("config_path must be a non-empty string")

    dataset_hashes = manifest.get("dataset_hashes")
    if not isinstance(dataset_hashes, Mapping) or not dataset_hashes:
        errors.append("dataset_hashes must be a non-empty mapping")
    else:
        for dataset_name, record in dataset_hashes.items():
            if not isinstance(record, Mapping):
                errors.append(f"dataset {dataset_name!r} record is not a mapping")
                continue
            raw_path = record.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                errors.append(f"dataset {dataset_name!r} has no path")
                continue
            path = _resolve_from_root(raw_path, root)
            if not path.is_file():
                errors.append(f"dataset {dataset_name!r} is missing: {path}")
                continue
            actual_hash = sha256_file(path)
            if record.get("sha256") != actual_hash:
                errors.append(
                    f"dataset {dataset_name!r} hash mismatch: manifest={record.get('sha256')}, actual={actual_hash}"
                )

    outputs = manifest.get("output_files")
    if not isinstance(outputs, list):
        errors.append("output_files must be a list")
    else:
        seen_paths: set[str] = set()
        for index, record in enumerate(outputs):
            label = f"output_files[{index}]"
            if not isinstance(record, Mapping):
                errors.append(f"{label} is not a mapping")
                continue
            raw_path = record.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                errors.append(f"{label} has no path")
                continue
            if raw_path in seen_paths:
                errors.append(f"duplicate artifact path in manifest: {raw_path}")
            seen_paths.add(raw_path)
            if record.get("run_id") != run_id:
                errors.append(f"{label} run_id does not match manifest run_id")
            if record.get("config_hash") != config_hash:
                errors.append(f"{label} config_hash does not match manifest config_hash")
            path = _resolve_from_root(raw_path, root)
            if not path.is_file():
                errors.append(f"manifest-referenced artifact is missing: {path}")
                continue
            actual_hash = sha256_file(path)
            if record.get("sha256") != actual_hash:
                errors.append(
                    f"artifact hash mismatch for {path}: manifest={record.get('sha256')}, actual={actual_hash}"
                )
            if record.get("size_bytes") != path.stat().st_size:
                errors.append(f"artifact size mismatch for {path}")

    status = manifest.get("status")
    if status not in {"running", "complete", "failed"}:
        errors.append(f"invalid run status: {status!r}")
    if require_complete and status != "complete":
        errors.append(f"run is not complete: status={status!r}")
    if status in {"complete", "failed"} and not manifest.get("end_timestamp"):
        errors.append("a finalized run requires end_timestamp")
    if status == "complete" and manifest.get("failure_information"):
        errors.append("a complete run cannot contain failure_information")

    if verify_source_tree:
        actual_source_hash = source_tree_hash(root)
        if manifest.get("source_tree_hash") != actual_source_hash:
            errors.append(
                "source tree hash mismatch: the experiment/config source changed after the run manifest was created"
            )

    if errors:
        raise RunManifestError("Invalid manuscript run manifest:\n- " + "\n- ".join(errors))
    return manifest


def write_run_manifest(
    manifest: Mapping[str, Any],
    path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    validate: bool = True,
    require_complete: bool = False,
) -> Path:
    """Atomically write a manifest after optional integrity validation."""

    if validate:
        validate_run_manifest(
            manifest,
            project_root=project_root,
            require_complete=require_complete,
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "FeaturePolicyConsistencyError",
    "ForbiddenFeatureError",
    "ManuscriptConfigError",
    "RunManifestError",
    "artifact_feature_names",
    "canonical_config_hash",
    "canonical_policy_mapping",
    "create_run_manifest",
    "feature_policy_definitions",
    "finalize_run_manifest",
    "forbidden_feature_mentions",
    "load_manuscript_config",
    "make_run_id",
    "manuscript_settings",
    "primary_excluded_features",
    "primary_policy_definition",
    "primary_policy_name",
    "record_command",
    "record_failure",
    "register_artifact",
    "sha256_file",
    "source_tree_hash",
    "validate_artifact_forbidden_features",
    "validate_manuscript_config",
    "validate_policy_consistency",
    "validate_primary_feature_names",
    "validate_run_manifest",
    "write_run_manifest",
]
