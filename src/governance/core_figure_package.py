"""Fail-closed validation for one production v2 core-figure stage.

The validator intentionally does not generate figures.  It admits only a
closed-world ``core_figures`` stage whose manifest is derived from the frozen
figure plan and whose sources belong to the same immutable run identity.
"""

from __future__ import annotations

import csv
import json
import math
import re
import struct
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree

from src.governance.core_figure_contract import (
    CORE_FIGURE_IDENTITY_FIELDS,
    CORE_FIGURE_KEYS,
    CORE_FIGURE_PLAN_VERSION,
    expected_core_figure_plan,
)
from src.governance.manuscript_contract import (
    ForbiddenFeatureError,
    artifact_feature_names,
    canonical_config_hash,
    manuscript_settings,
    sha256_file,
    validate_artifact_forbidden_features,
)


class CoreFigurePackageError(ValueError):
    """Raised when a core-figure artifact or its lineage fails validation."""


FIGURE_MANIFEST_FILENAME = "figure_manifest.json"
STAGE_CONTRACT_FILENAME = "stage_contract.json"
FIGURE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "manifest_kind",
        "status",
        "inventory_mode",
        "path_basis",
        "plan_version",
        "stage",
        "scope",
        "hash_algorithm",
        "n_figures",
        *CORE_FIGURE_IDENTITY_FIELDS,
        "figures",
    }
)
FIGURE_RECORD_KEYS = frozenset(
    {
        "figure_key",
        "number",
        "figure_id",
        "output_stem",
        "png_path",
        "png_sha256",
        "png_size_bytes",
        "png_width_px",
        "png_height_px",
        "svg_path",
        "svg_sha256",
        "svg_size_bytes",
        "svg_width_px",
        "svg_height_px",
        "source_data_path",
        "source_data_sha256",
        "source_data_size_bytes",
        "caption_path",
        "caption_sha256",
        "caption_size_bytes",
        "sources",
    }
)
SOURCE_RECORD_KEYS = frozenset({"stage", "path", "sha256", "size_bytes"})
OBSOLETE_CORE_STEMS = frozenset(
    {
        "figure_1_governance_architecture",
        "figure_2_structured_evidence_flow",
        "figure_3_multi_agent_audit",
        "figure_4_gxair_readiness_dashboard",
        "figure_5_calibration_ordinal_error",
        "figure_6_global_grouped_shap",
        "figure_7_local_reason_code",
    }
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SVG_NUMBER = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)(?:px|pt|pc|mm|cm|in)?\s*$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CoreFigurePackageError(message)


def _is_linklike(path: Path) -> bool:
    try:
        details = path.lstat()
    except OSError:
        return False
    return path.is_symlink() or bool(getattr(details, "st_file_attributes", 0) & 0x400)


def _read_json(path: Path, *, context: str) -> Mapping[str, Any]:
    _require(path.is_file() and not _is_linklike(path), f"{context} is missing or link-like: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CoreFigurePackageError(f"Cannot parse {context}: {exc}") from exc
    _require(isinstance(payload, Mapping), f"{context} must be a JSON object.")
    return payload


def _portable_path(value: Any, *, context: str) -> str:
    _require(
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and "\\" not in value,
        f"{context} must be a non-empty portable relative path.",
    )
    path = PurePosixPath(str(value))
    _require(
        not path.is_absolute()
        and not str(value).startswith("./")
        and all(part not in {"", ".", ".."} for part in path.parts)
        and ":" not in path.parts[0],
        f"{context} must be a contained portable relative path.",
    )
    folded_parts = {part.casefold() for part in path.parts}
    _require(
        "latest" not in folded_parts and "historical" not in folded_parts,
        f"{context} may not reference a historical or latest package.",
    )
    _require(
        not any(Path(part).stem.casefold() in OBSOLETE_CORE_STEMS for part in path.parts),
        f"{context} contains an obsolete v1 figure stem.",
    )
    return str(value)


def _contained_file(root: Path, relative: str, *, context: str) -> Path:
    lexical = root / Path(relative)
    _require(lexical.is_file() and not _is_linklike(lexical), f"{context} is missing or link-like: {relative}")
    resolved_root = root.resolve()
    resolved = lexical.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise CoreFigurePackageError(f"{context} escapes its allowed root: {relative}") from exc
    _require(resolved.stat().st_size > 0, f"{context} is empty: {relative}")
    return resolved


def _expected_identity(
    *,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    source_tree_hash: str,
) -> dict[str, str]:
    identity = {
        "run_id": run_id,
        "config_hash": config_hash,
        "scientific_input_hash": scientific_input_hash,
        "source_tree_hash": source_tree_hash,
    }
    _require(isinstance(run_id, str) and bool(run_id), "run_id must be non-empty.")
    for field in ("config_hash", "scientific_input_hash", "source_tree_hash"):
        value = identity[field]
        _require(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"{field} must be a lowercase SHA-256 digest.",
        )
    return identity


def _check_identity(payload: Mapping[str, Any], identity: Mapping[str, str], *, context: str) -> None:
    mismatches = {
        field: {"expected": expected, "observed": payload.get(field)}
        for field, expected in identity.items()
        if payload.get(field) != expected
    }
    _require(not mismatches, f"{context} identity mismatch: {json.dumps(mismatches, sort_keys=True)}")


def _actual_files(root: Path) -> set[str]:
    files: set[str] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        _require(not _is_linklike(candidate), f"Artifact tree contains a link-like path: {relative}")
        if candidate.is_file():
            files.add(relative)
        else:
            _require(candidate.is_dir(), f"Artifact tree contains a non-regular path: {relative}")
    return files


def _validate_stage_contract(
    run_root: Path,
    stage: str,
    identity: Mapping[str, str],
) -> dict[str, Mapping[str, Any]]:
    stage_root = run_root / stage
    _require(stage_root.is_dir() and not _is_linklike(stage_root), f"Stage root is absent or link-like: {stage}")
    receipt = _read_json(stage_root / STAGE_CONTRACT_FILENAME, context=f"{stage} stage contract")
    _require(receipt.get("stage") == stage, f"Stage contract has the wrong stage identity: {stage}")
    _require(receipt.get("status") == "complete", f"Stage contract is not complete: {stage}")
    _require(receipt.get("inventory_mode") == "closed_world", f"Stage contract is not closed-world: {stage}")
    _require(receipt.get("path_basis") == "stage_relative", f"Stage contract has a nonportable path basis: {stage}")
    _check_identity(receipt, identity, context=f"{stage} stage contract")
    rows = receipt.get("outputs")
    _require(isinstance(rows, list) and bool(rows), f"Stage contract has no output inventory: {stage}")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), f"Stage output row is malformed: {stage}")
        _require(set(row) == {"path", "sha256", "size_bytes"}, f"Stage output row has unexpected fields: {stage}")
        relative = _portable_path(row.get("path"), context=f"{stage} output path")
        _require("/" not in relative or relative.split("/", 1)[0] != stage, f"Stage output path must be stage-relative: {relative}")
        _require(relative != STAGE_CONTRACT_FILENAME, "A stage contract may not inventory itself.")
        _require(relative not in indexed and relative.casefold() not in {key.casefold() for key in indexed}, f"Duplicate stage output path: {relative}")
        path = _contained_file(stage_root, relative, context=f"{stage} output")
        _require(row.get("size_bytes") == path.stat().st_size, f"Stage output size mismatch: {stage}/{relative}")
        _require(row.get("sha256") == sha256_file(path), f"Stage output hash mismatch: {stage}/{relative}")
        indexed[relative] = row
    actual = _actual_files(stage_root)
    _require(
        actual == {*indexed, STAGE_CONTRACT_FILENAME},
        f"Stage inventory is not closed-world: {stage}; extra={sorted(actual - set(indexed) - {STAGE_CONTRACT_FILENAME})}, missing={sorted(set(indexed) - actual)}",
    )
    return indexed


def _validate_run_inputs(
    run_root: Path,
    identity: Mapping[str, str],
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    root = run_root / "run_inputs"
    _require(root.is_dir() and not _is_linklike(root), "Run-input snapshot root is absent or link-like.")
    contract = _read_json(root / "input_contract.json", context="run-input contract")
    _require(contract.get("schema_version") == 1, "Run-input contract schema version is invalid.")
    _require(contract.get("contract_kind") == "manuscript_run_inputs", "Run-input contract kind is invalid.")
    _require(contract.get("status") == "complete", "Run-input contract is not complete.")
    _require(contract.get("inventory_mode") == "closed_world", "Run-input contract is not closed-world.")
    _require(contract.get("path_basis") == "run_inputs_relative", "Run-input contract path basis is invalid.")
    _check_identity(contract, identity, context="run-input contract")
    rows = contract.get("snapshots")
    _require(isinstance(rows, list) and contract.get("n_snapshots") == len(rows), "Run-input snapshot count is invalid.")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "Run-input snapshot row is malformed.")
        relative = _portable_path(row.get("snapshot_path"), context="run-input snapshot path")
        _require(relative != "input_contract.json" and relative not in indexed, f"Duplicate or reserved run-input path: {relative}")
        path = _contained_file(root, relative, context="run-input snapshot")
        _require(row.get("snapshot_size_bytes") == path.stat().st_size, f"Run-input snapshot size mismatch: {relative}")
        _require(row.get("snapshot_sha256") == sha256_file(path), f"Run-input snapshot hash mismatch: {relative}")
        indexed[relative] = row
    actual = _actual_files(root)
    _require(actual == {*indexed, "input_contract.json"}, "Run-input snapshot inventory is not closed-world.")
    return contract, indexed


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    _require(
        len(header) == 24 and header[:8] == _PNG_SIGNATURE and header[12:16] == b"IHDR",
        f"PNG has no valid signature/IHDR header: {path.name}",
    )
    width, height = struct.unpack(">II", header[16:24])
    _require(width > 0 and height > 0, f"PNG dimensions are invalid: {path.name}")
    return width, height


def _svg_number(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    match = _SVG_NUMBER.fullmatch(value)
    return float(match.group(1)) if match else None


def _svg_dimensions(path: Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8")
    _require("<!DOCTYPE" not in text.upper() and "<!ENTITY" not in text.upper(), f"SVG contains a prohibited declaration: {path.name}")
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise CoreFigurePackageError(f"SVG is not valid XML: {path.name}: {exc}") from exc
    _require(root.tag.rsplit("}", 1)[-1] == "svg", f"SVG root element is invalid: {path.name}")
    width = _svg_number(root.attrib.get("width"))
    height = _svg_number(root.attrib.get("height"))
    if width is None or height is None:
        raw_viewbox = root.attrib.get("viewBox")
        try:
            viewbox = [float(value) for value in str(raw_viewbox).replace(",", " ").split()]
        except ValueError as exc:
            raise CoreFigurePackageError(f"SVG viewBox is invalid: {path.name}") from exc
        _require(len(viewbox) == 4, f"SVG has no usable dimensions: {path.name}")
        width, height = viewbox[2], viewbox[3]
    _require(math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0, f"SVG dimensions are invalid: {path.name}")
    return width, height


def _validate_file_record(
    root: Path,
    row: Mapping[str, Any],
    *,
    path_field: str,
    hash_field: str,
    size_field: str,
    expected_path: str,
    context: str,
) -> Path:
    observed = _portable_path(row.get(path_field), context=f"{context} path")
    _require(observed == expected_path, f"{context} path differs from the frozen plan.")
    path = _contained_file(root, observed, context=context)
    _require(row.get(size_field) == path.stat().st_size, f"{context} size mismatch.")
    _require(row.get(hash_field) == sha256_file(path), f"{context} hash mismatch.")
    return path


def _validate_source_data(
    path: Path,
    identity: Mapping[str, str],
    config: Mapping[str, Any],
    *,
    validate_main_primary_features: bool,
) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CoreFigurePackageError(f"Cannot parse figure source data {path.name}: {exc}") from exc
    _require(bool(rows), f"Figure source data has no rows: {path.name}")
    for field, expected in identity.items():
        _require(field in fieldnames, f"Figure source data omits {field}: {path.name}")
        observed = {str(row.get(field, "")) for row in rows}
        _require(observed == {expected}, f"Figure source data has mixed or wrong {field}: {path.name}")
    if validate_main_primary_features:
        try:
            validate_artifact_forbidden_features(path, config)
        except ForbiddenFeatureError as exc:
            raise CoreFigurePackageError(str(exc)) from exc


def _external_primary_forbidden_features(config: Mapping[str, Any]) -> tuple[str, ...]:
    external = manuscript_settings(config).get("external_replication")
    _require(isinstance(external, Mapping), "Canonical config has no external-replication contract.")
    policy = external.get("feature_policy_contract")
    _require(isinstance(policy, Mapping), "External replication has no feature-policy contract.")
    aliases = policy.get("always_forbidden_feature_aliases")
    governance = policy.get("primary_governance_exclusions")
    _require(isinstance(aliases, Mapping) and isinstance(governance, list), "External primary exclusions are malformed.")
    _require(
        all(isinstance(group, list) for group in aliases.values()),
        "External always-forbidden feature groups must be lists.",
    )
    values = [str(value) for group in aliases.values() for value in group]
    values.extend(str(value) for value in governance)
    return tuple(dict.fromkeys(values))


def _validate_additional_forbidden_features(path: Path, forbidden: Sequence[str]) -> None:
    if not forbidden:
        return
    names = artifact_feature_names(path)
    found: dict[str, list[str]] = {}
    for value in names:
        candidate = str(value).strip().strip("'\"")
        base = candidate.rsplit("__", 1)[-1].casefold()
        for raw in forbidden:
            folded = raw.casefold()
            if base == folded or any(base.startswith(folded + separator) for separator in ("_", "=", "[", ":", "-", " ")):
                found.setdefault(raw, []).append(candidate)
    _require(not found, f"Forbidden external-primary feature families appear in {path}: {found}")


def _validate_caption(path: Path, identity: Mapping[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    _require(bool(text.strip()), f"Figure caption is empty: {path.name}")
    for field, expected in identity.items():
        _require(f"{field}={expected}" in text, f"Figure caption omits exact {field}: {path.name}")


def validate_core_figure_package(
    figure_dir: str | Path,
    *,
    run_root: str | Path,
    config: Mapping[str, Any],
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    source_tree_hash: str,
) -> dict[str, Any]:
    """Validate one exact seven-figure, source-bound production stage."""

    root = Path(figure_dir)
    run = Path(run_root)
    _require(run.is_dir() and not _is_linklike(run), "Run root is absent or link-like.")
    _require(root.is_dir() and not _is_linklike(root), "Core-figure stage root is absent or link-like.")
    _require(root.resolve() == (run / "core_figures").resolve(), "Figure directory must be the exact current-run core_figures stage.")
    identity = _expected_identity(
        run_id=run_id,
        config_hash=config_hash,
        scientific_input_hash=scientific_input_hash,
        source_tree_hash=source_tree_hash,
    )
    _require(
        canonical_config_hash(config) == config_hash,
        "Supplied config_hash does not match the canonical configuration.",
    )
    settings = manuscript_settings(config)
    figure_plan = settings.get("figures")
    _require(isinstance(figure_plan, Mapping), "Canonical config has no figure plan.")
    _require(
        dict(figure_plan) == expected_core_figure_plan(),
        "Canonical figure plan differs from the frozen v2 contract.",
    )
    definitions = figure_plan.get("definitions")
    _require(isinstance(definitions, Mapping) and tuple(definitions) == CORE_FIGURE_KEYS, "Canonical figure definitions are incomplete or reordered.")

    core_outputs = _validate_stage_contract(run, "core_figures", identity)
    manifest = _read_json(root / FIGURE_MANIFEST_FILENAME, context="core-figure manifest")
    _require(set(manifest) == FIGURE_MANIFEST_KEYS, "Core-figure manifest fields differ from schema version 1.")
    expected_manifest = {
        "schema_version": 1,
        "manifest_kind": "core_figure_package",
        "status": "complete",
        "inventory_mode": "closed_world",
        "path_basis": "core_figures_relative",
        "plan_version": CORE_FIGURE_PLAN_VERSION,
        "stage": "core_figures",
        "scope": "core",
        "hash_algorithm": "sha256",
        "n_figures": 7,
    }
    for field, expected in expected_manifest.items():
        _require(manifest.get(field) == expected, f"Core-figure manifest has the wrong {field}.")
    _check_identity(manifest, identity, context="core-figure manifest")
    figure_rows = manifest.get("figures")
    _require(isinstance(figure_rows, list) and len(figure_rows) == 7, "Core-figure manifest must contain exactly seven figures.")

    run_input_contract, run_input_rows = _validate_run_inputs(run, identity)
    stage_cache: dict[str, dict[str, Mapping[str, Any]]] = {"core_figures": core_outputs}
    expected_output_paths = {FIGURE_MANIFEST_FILENAME}
    source_count = 0
    for index, key in enumerate(CORE_FIGURE_KEYS):
        row = figure_rows[index]
        definition = definitions[key]
        _require(isinstance(row, Mapping) and set(row) == FIGURE_RECORD_KEYS, f"{key} manifest row has unexpected fields.")
        for field in ("number", "figure_id", "output_stem"):
            _require(row.get(field) == definition.get(field), f"{key} has the wrong {field}.")
        _require(row.get("figure_key") == key, f"{key} has the wrong figure_key.")
        stem = str(definition["output_stem"])
        expected_png = f"{stem}.png"
        expected_svg = f"{stem}.svg"
        expected_source = f"{figure_plan['source_data_subdirectory']}/{definition['source_data_filename']}"
        expected_caption = f"{figure_plan['caption_subdirectory']}/{definition['caption_filename']}"
        png = _validate_file_record(root, row, path_field="png_path", hash_field="png_sha256", size_field="png_size_bytes", expected_path=expected_png, context=f"{key} PNG")
        svg = _validate_file_record(root, row, path_field="svg_path", hash_field="svg_sha256", size_field="svg_size_bytes", expected_path=expected_svg, context=f"{key} SVG")
        source_data = _validate_file_record(root, row, path_field="source_data_path", hash_field="source_data_sha256", size_field="source_data_size_bytes", expected_path=expected_source, context=f"{key} source data")
        caption = _validate_file_record(root, row, path_field="caption_path", hash_field="caption_sha256", size_field="caption_size_bytes", expected_path=expected_caption, context=f"{key} caption")
        png_dimensions = _png_dimensions(png)
        svg_dimensions = _svg_dimensions(svg)
        _require((row.get("png_width_px"), row.get("png_height_px")) == png_dimensions, f"{key} PNG dimension receipt mismatch.")
        _require((row.get("svg_width_px"), row.get("svg_height_px")) == svg_dimensions, f"{key} SVG dimension receipt mismatch.")
        additional_forbidden = (
            _external_primary_forbidden_features(config) if key == "figure_7" else ()
        )
        validate_main_features = key in {"figure_5", "figure_6"}
        _validate_source_data(
            source_data,
            identity,
            config,
            validate_main_primary_features=validate_main_features,
        )
        _validate_additional_forbidden_features(source_data, additional_forbidden)
        _validate_caption(caption, identity)
        expected_output_paths.update({expected_png, expected_svg, expected_source, expected_caption})

        source_rows = row.get("sources")
        declared_sources = definition.get("sources")
        _require(isinstance(source_rows, list) and len(source_rows) == len(declared_sources), f"{key} source inventory count mismatch.")
        for source_index, declared in enumerate(declared_sources):
            source_row = source_rows[source_index]
            _require(isinstance(source_row, Mapping) and set(source_row) == SOURCE_RECORD_KEYS, f"{key} source record is malformed.")
            _require(source_row.get("stage") == declared.get("stage") and source_row.get("path") == declared.get("path"), f"{key} source identity differs from the frozen plan.")
            stage = str(declared["stage"])
            configured_path = _portable_path(declared["path"], context=f"{key} configured source")
            _require(configured_path.startswith(stage + "/"), f"{key} source does not belong to its declared stage.")
            stage_relative = configured_path.split("/", 1)[1]
            if stage == "run_inputs":
                source_path = _contained_file(run / stage, stage_relative, context=f"{key} run-input source")
                if stage_relative == "input_contract.json":
                    expected_hash = sha256_file(source_path)
                    expected_size = source_path.stat().st_size
                    _check_identity(run_input_contract, identity, context=f"{key} run-input source")
                else:
                    snapshot_row = run_input_rows.get(stage_relative)
                    _require(snapshot_row is not None, f"{key} source is absent from the run-input contract.")
                    expected_hash = snapshot_row.get("snapshot_sha256")
                    expected_size = snapshot_row.get("snapshot_size_bytes")
            else:
                if stage not in stage_cache:
                    stage_cache[stage] = _validate_stage_contract(run, stage, identity)
                output_row = stage_cache[stage].get(stage_relative)
                _require(output_row is not None, f"{key} source is absent from its stage contract: {configured_path}")
                source_path = _contained_file(run / stage, stage_relative, context=f"{key} upstream source")
                expected_hash = output_row.get("sha256")
                expected_size = output_row.get("size_bytes")
            _require(source_row.get("sha256") == expected_hash == sha256_file(source_path), f"{key} source hash mismatch: {configured_path}")
            _require(source_row.get("size_bytes") == expected_size == source_path.stat().st_size, f"{key} source size mismatch: {configured_path}")
            if source_path.suffix.casefold() in {".csv", ".json", ".jsonl"}:
                if validate_main_features:
                    try:
                        validate_artifact_forbidden_features(source_path, config)
                    except ForbiddenFeatureError as exc:
                        raise CoreFigurePackageError(str(exc)) from exc
                _validate_additional_forbidden_features(source_path, additional_forbidden)
            source_count += 1

    _require(set(core_outputs) == expected_output_paths, "Core-figure stage contains unlisted, missing, or obsolete artifacts.")
    return {
        "status": "passed",
        **identity,
        "plan_version": CORE_FIGURE_PLAN_VERSION,
        "figure_count": 7,
        "format_count": 14,
        "source_count": source_count,
        "artifact_count_excluding_stage_contract": len(expected_output_paths),
        "closed_world": True,
    }


__all__ = [
    "CoreFigurePackageError",
    "FIGURE_MANIFEST_FILENAME",
    "OBSOLETE_CORE_STEMS",
    "validate_core_figure_package",
]
