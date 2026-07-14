"""Build atomic, source-bound manuscript-support tables for one evidence scope."""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import pandas as pd

from src.core.atomic_publish import atomic_replace_directory
from src.governance.manuscript_contract import canonical_config_hash, manuscript_settings, sha256_file
from src.governance.table_contract import (
    TABLE_IDENTITY_FIELDS,
    TABLE_PLAN_VERSION,
    expected_table_plan,
    validate_table_plan_declaration,
)
from src.models.task_schema import metric_schema_hash, metric_schema_records
from src.utils.config_loader import load_config


class ManuscriptTableError(RuntimeError):
    """Raised when source-table generation or validation violates the contract."""


_PROVENANCE_COLUMNS = (
    *TABLE_IDENTITY_FIELDS,
    "metric_schema_hash",
    "table_id",
    "dataset_identity",
    "model_identity",
    "evaluation_scope",
    "denominator",
    "uncertainty_method",
    "source_artifact",
    "source_sha256",
    "source_record_type",
    "source_row_number",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManuscriptTableError(message)


def _is_linklike(path: Path) -> bool:
    try:
        details = path.lstat()
    except OSError:
        return False
    return path.is_symlink() or bool(getattr(details, "st_file_attributes", 0) & 0x400)


def _portable(value: Any, *, context: str) -> str:
    _require(
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and "\\" not in value,
        f"{context} must be a non-empty portable relative path.",
    )
    candidate = PurePosixPath(value)
    _require(
        not candidate.is_absolute()
        and not value.startswith("./")
        and all(part not in {"", ".", ".."} for part in candidate.parts)
        and ":" not in candidate.parts[0],
        f"{context} must be a contained portable relative path.",
    )
    return value


def _source_path(run_root: Path, relative: str) -> Path:
    portable = _portable(relative, context="Table source")
    path = (run_root / Path(portable)).resolve()
    try:
        path.relative_to(run_root.resolve())
    except ValueError as exc:
        raise ManuscriptTableError(f"Table source escapes the active run root: {portable}") from exc
    _require(path.is_file() and not _is_linklike(path), f"Table source is missing or link-like: {portable}")
    _require(path.stat().st_size > 0, f"Table source is empty: {portable}")
    return path


def _read_json(path: Path, *, context: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManuscriptTableError(f"Cannot parse {context}: {exc}") from exc
    _require(isinstance(payload, Mapping), f"{context} must be a JSON object.")
    return payload


def _read_csv(path: Path, *, context: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ManuscriptTableError(f"Cannot parse {context}: {exc}") from exc
    _require(not frame.empty, f"{context} has no rows.")
    return frame


def _scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return str(value)


def _first_present(row: Mapping[str, Any], names: Sequence[str], *, default: str) -> str:
    for name in names:
        value = _scalar(row.get(name))
        if value:
            return value
    return default


def _composite_identity(row: Mapping[str, Any], names: Sequence[str], *, default: str) -> str:
    values = {name: _scalar(row.get(name)) for name in names if _scalar(row.get(name))}
    if not values:
        return default
    return json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _check_source_identity(
    frame: pd.DataFrame,
    identity: Mapping[str, str],
    *,
    context: str,
) -> None:
    for field in ("run_id", "config_hash"):
        _require(field in frame.columns, f"{context} omits {field}.")
        observed = set(frame[field].astype(str))
        _require(observed == {identity[field]}, f"{context} has mixed or wrong {field}: {sorted(observed)}")
    if "scientific_input_hash" in frame.columns:
        observed = set(frame["scientific_input_hash"].astype(str))
        _require(
            observed == {identity["scientific_input_hash"]},
            f"{context} has mixed or wrong scientific_input_hash: {sorted(observed)}",
        )
    if "source_tree_hash" in frame.columns:
        observed = set(frame["source_tree_hash"].astype(str))
        _require(
            observed == {identity["source_tree_hash"]},
            f"{context} has mixed or wrong source_tree_hash: {sorted(observed)}",
        )


def _source_records(
    frame: pd.DataFrame,
    *,
    definition: Mapping[str, Any],
    source_artifact: str,
    source_sha256: str,
    identity: Mapping[str, str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for position, raw in enumerate(frame.to_dict(orient="records"), start=1):
        row = dict(raw)
        dataset_identity = _composite_identity(
            row,
            (
                "dataset_identity",
                "dataset_key",
                "dataset_id",
                "canonical_dataset_key",
                "physical_dataset_id",
                "dataset_sha256",
                "raw_dataset_sha256",
                "canonical_content_sha256",
            ),
            default=str(definition["dataset_scope"]),
        )
        model_identity = _composite_identity(
            row,
            (
                "model_identity",
                "system_id",
                "model_name",
                "model",
                "model_sha256",
                "model_set_hash",
                "fitted_model_set_hash",
                "policy",
            ),
            default=str(definition["model_scope"]),
        )
        denominator = _first_present(
            row,
            (
                "denominator",
                "metric_denominator",
                "evaluation_denominator",
                "n_samples",
                "n_oof_samples",
                "sample_count",
                "row_count",
                "support",
                "n",
            ),
            default="not_applicable_or_resolved_by_source_contract",
        )
        row.update(identity)
        row.update(
            {
                "metric_schema_hash": metric_schema_hash(),
                "table_id": definition["table_id"],
                "dataset_identity": dataset_identity,
                "model_identity": model_identity,
                "evaluation_scope": definition["evaluation_scope"],
                "denominator": denominator,
                "uncertainty_method": definition["uncertainty_method"],
                "source_artifact": source_artifact,
                "source_sha256": source_sha256,
                "source_record_type": Path(source_artifact).stem,
                "source_row_number": position,
            }
        )
        records.append(row)
    result = pd.DataFrame(records)
    trailing = [column for column in result.columns if column not in _PROVENANCE_COLUMNS]
    return result.loc[:, [*_PROVENANCE_COLUMNS, *trailing]]


def _generic_table(
    definition: Mapping[str, Any],
    *,
    run_root: Path,
    identity: Mapping[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    receipts: list[dict[str, Any]] = []
    for relative in definition["sources"]:
        source = _source_path(run_root, str(relative))
        digest = sha256_file(source)
        if source.suffix.lower() == ".csv":
            frame = _read_csv(source, context=f"table source {relative}")
        elif source.suffix.lower() == ".json":
            payload = _read_json(source, context=f"table source {relative}")
            frame = pd.DataFrame(
                [{key: _scalar(value) for key, value in payload.items()}]
            )
        else:
            raise ManuscriptTableError(f"Generic table source must be CSV or JSON: {relative}")
        _check_source_identity(frame, identity, context=f"table source {relative}")
        frames.append(
            _source_records(
                frame,
                definition=definition,
                source_artifact=str(relative),
                source_sha256=digest,
                identity=identity,
            )
        )
        receipts.append(
            {"path": str(relative), "sha256": digest, "size_bytes": source.stat().st_size}
        )
    table = pd.concat(frames, ignore_index=True, sort=False)
    _require(not table.empty, f"Table {definition['table_id']} produced no rows.")
    return table, receipts


def _registry_table(
    definition: Mapping[str, Any],
    *,
    run_root: Path,
    identity: Mapping[str, str],
    settings: Mapping[str, Any],
    scope: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    receipts: list[dict[str, Any]] = []
    by_path: dict[str, tuple[Path, str]] = {}
    for relative in definition["sources"]:
        source = _source_path(run_root, str(relative))
        digest = sha256_file(source)
        by_path[str(relative)] = (source, digest)
        receipts.append(
            {"path": str(relative), "sha256": digest, "size_bytes": source.stat().st_size}
        )
    contract = _read_json(by_path["run_inputs/input_contract.json"][0], context="run-input contract")
    for field in TABLE_IDENTITY_FIELDS:
        _require(contract.get(field) == identity[field], f"Run-input contract has the wrong {field}.")

    if scope == "core":
        permitted_tasks = {"ordinal_multiclass_performance", "nominal_multiclass_proxy_diagnostic"}
    else:
        permitted_tasks = {
            "restricted_target_performance_robustness",
            "binary_attrition_transfer",
            "binary_turnover_transfer",
        }
    config_source = "run_inputs/canonical_config_snapshot.yaml"
    config_digest = by_path[config_source][1]
    rows: list[dict[str, Any]] = []
    for position, metric_row in enumerate(metric_schema_records(), start=1):
        if metric_row["task_type"] not in permitted_tasks:
            continue
        row = dict(metric_row)
        row.update(identity)
        row.update(
            {
                "metric_schema_hash": metric_schema_hash(),
                "table_id": definition["table_id"],
                "dataset_identity": str(metric_row["task_type"]),
                "model_identity": str(definition["model_scope"]),
                "evaluation_scope": str(definition["evaluation_scope"]),
                "denominator": str(metric_row["denominator"]),
                "uncertainty_method": str(metric_row["uncertainty_method"]),
                "source_artifact": config_source,
                "source_sha256": config_digest,
                "source_record_type": "metric_definition",
                "source_row_number": position,
            }
        )
        rows.append(row)

    table_plan = expected_table_plan(scope)
    for definition_key, source_definition in table_plan.items():
        row = {
            **identity,
            "metric_schema_hash": metric_schema_hash(),
            "table_id": definition["table_id"],
            "dataset_identity": source_definition["dataset_scope"],
            "model_identity": source_definition["model_scope"],
            "evaluation_scope": source_definition["evaluation_scope"],
            "denominator": "not_applicable_claim_contract",
            "uncertainty_method": source_definition["uncertainty_method"],
            "source_artifact": config_source,
            "source_sha256": config_digest,
            "source_record_type": "claim_boundary",
            "source_row_number": len(rows) + 1,
            "contract_table_key": definition_key,
            "contract_table_number": source_definition["number"],
            "contract_title": source_definition["title"],
            "claim_boundary": source_definition["claim_boundary"],
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    trailing = [column for column in result.columns if column not in _PROVENANCE_COLUMNS]
    return result.loc[:, [*_PROVENANCE_COLUMNS, *trailing]], receipts


def validate_table_package(
    output_dir: str | Path,
    *,
    run_root: str | Path,
    config: Mapping[str, Any],
    scope: str,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    source_tree_hash: str,
) -> Mapping[str, Any]:
    output = Path(output_dir).resolve()
    root = Path(run_root).resolve()
    _require(output.is_dir() and not _is_linklike(output), "Table package root is missing or link-like.")
    identity = {
        "run_id": run_id,
        "config_hash": config_hash,
        "scientific_input_hash": scientific_input_hash,
        "source_tree_hash": source_tree_hash,
    }
    manifest_path = output / "table_manifest.json"
    manifest = _read_json(manifest_path, context="table manifest")
    expected_manifest = {
        "schema_version": 1,
        "plan_version": TABLE_PLAN_VERSION,
        "status": "complete",
        "scope": scope,
        **identity,
        "metric_schema_hash": metric_schema_hash(),
        "inventory_mode": "closed_world_runner_owned",
    }
    for field, expected in expected_manifest.items():
        _require(manifest.get(field) == expected, f"Table manifest has the wrong {field}.")
    settings = manuscript_settings(config)
    validate_table_plan_declaration(settings.get("tables"))
    plan = expected_table_plan(scope)
    records = manifest.get("tables")
    _require(isinstance(records, list) and len(records) == len(plan), "Table manifest count is invalid.")
    _require(manifest.get("table_count") == len(plan), "Table manifest declared count is invalid.")
    by_id = {str(record.get("table_id")): record for record in records if isinstance(record, Mapping)}
    _require(len(by_id) == len(plan), "Table manifest IDs are missing or duplicated.")
    expected_files = {"table_manifest.json"}
    for definition in plan.values():
        table_id = str(definition["table_id"])
        record = by_id.get(table_id)
        _require(record is not None, f"Table manifest omits {table_id}.")
        filename = _portable(str(definition["filename"]), context="Table filename")
        _require(record.get("filename") == filename, f"Table manifest filename differs for {table_id}.")
        _require(
            record.get("evaluation_scope") == definition["evaluation_scope"]
            and record.get("claim_boundary") == definition["claim_boundary"],
            f"Table manifest contract fields differ for {table_id}.",
        )
        table_path = output / filename
        _require(table_path.is_file() and not _is_linklike(table_path), f"Table output is missing: {filename}")
        _require(record.get("sha256") == sha256_file(table_path), f"Table hash differs: {filename}")
        _require(record.get("size_bytes") == table_path.stat().st_size, f"Table size differs: {filename}")
        frame = _read_csv(table_path, context=f"table output {filename}")
        _require(record.get("rows") == len(frame), f"Table row count differs: {filename}")
        _require(record.get("columns") == list(frame.columns), f"Table columns differ: {filename}")
        missing = sorted(set(_PROVENANCE_COLUMNS).difference(frame.columns))
        _require(not missing, f"Table {filename} omits provenance columns: {missing}")
        for field in TABLE_IDENTITY_FIELDS:
            _require(set(frame[field].astype(str)) == {identity[field]}, f"Table {filename} has wrong {field}.")
        _require(
            set(frame["metric_schema_hash"].astype(str)) == {metric_schema_hash()},
            f"Table {filename} has the wrong metric schema hash.",
        )
        declared_sources = set(str(value) for value in definition["sources"])
        observed_sources = set(frame["source_artifact"].astype(str))
        _require(observed_sources.issubset(declared_sources), f"Table {filename} cites undeclared sources.")
        for column in (
            "dataset_identity",
            "model_identity",
            "evaluation_scope",
            "denominator",
            "uncertainty_method",
            "source_artifact",
            "source_sha256",
        ):
            _require(
                frame[column].astype(str).str.strip().ne("").all(),
                f"Table {filename} has an empty {column} value.",
            )
        source_records = record.get("sources")
        _require(
            isinstance(source_records, list) and len(source_records) == len(declared_sources),
            f"Table source receipt count differs: {filename}",
        )
        receipt_paths = [
            str(source_record.get("path"))
            for source_record in source_records
            if isinstance(source_record, Mapping)
        ]
        _require(
            len(receipt_paths) == len(source_records)
            and len(set(receipt_paths)) == len(receipt_paths)
            and set(receipt_paths) == declared_sources,
            f"Table source receipts are missing or duplicated: {filename}",
        )
        receipt_hashes: dict[str, str] = {}
        for source_record in source_records:
            _require(isinstance(source_record, Mapping), f"Invalid source receipt in {filename}.")
            relative = _portable(source_record.get("path"), context="Table source receipt")
            _require(relative in declared_sources, f"Table {filename} has an undeclared source receipt.")
            source = _source_path(root, relative)
            _require(source_record.get("sha256") == sha256_file(source), f"Source hash changed: {relative}")
            _require(source_record.get("size_bytes") == source.stat().st_size, f"Source size changed: {relative}")
            receipt_hashes[relative] = str(source_record["sha256"])
        for source_artifact, group in frame.groupby("source_artifact", sort=False):
            _require(
                set(group["source_sha256"].astype(str)) == {receipt_hashes[str(source_artifact)]},
                f"Table row source hashes differ from the receipt: {filename}/{source_artifact}",
            )
        expected_files.add(filename)
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "stage_contract.json"
    }
    _require(actual_files == expected_files, "Table package closed-world inventory differs from the plan.")
    return {
        "status": "complete",
        "scope": scope,
        "table_count": len(plan),
        "metric_schema_hash": metric_schema_hash(),
        **identity,
    }


def run(
    config_path: str | Path,
    *,
    run_root: str | Path,
    output_dir: str | Path,
    scope: str,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    source_tree_hash: str,
) -> Mapping[str, Path]:
    config = load_config(config_path)
    _require(canonical_config_hash(config) == config_hash, "Canonical config hash differs from the run identity.")
    settings = manuscript_settings(config)
    validate_table_plan_declaration(settings.get("tables"))
    plan = expected_table_plan(scope)
    root = Path(run_root).resolve()
    output = Path(output_dir).resolve()
    _require(root.is_dir() and not _is_linklike(root), "Active run root is missing or link-like.")
    _require(not output.exists(), f"Table output already exists and is immutable: {output.name}")
    _require(output.parent.resolve() == root, "Table output must be a direct child of the active run root.")
    identity = {
        "run_id": run_id,
        "config_hash": config_hash,
        "scientific_input_hash": scientific_input_hash,
        "source_tree_hash": source_tree_hash,
    }
    staging = output.parent / f"{output.name}.__staging__.{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    written: dict[str, Path] = {}
    table_receipts: list[dict[str, Any]] = []
    try:
        for definition in plan.values():
            if definition["renderer"] == "metric_and_claim_registry":
                frame, source_receipts = _registry_table(
                    definition,
                    run_root=root,
                    identity=identity,
                    settings=settings,
                    scope=scope,
                )
            else:
                frame, source_receipts = _generic_table(
                    definition,
                    run_root=root,
                    identity=identity,
                )
            filename = str(definition["filename"])
            path = staging / filename
            frame.to_csv(path, index=False, lineterminator="\n")
            _require(path.stat().st_size > 0, f"Table writer produced an empty file: {filename}")
            written[str(definition["table_id"])] = path
            table_receipts.append(
                {
                    "number": definition["number"],
                    "table_id": definition["table_id"],
                    "title": definition["title"],
                    "filename": filename,
                    "rows": len(frame),
                    "columns": list(frame.columns),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "evaluation_scope": definition["evaluation_scope"],
                    "claim_boundary": definition["claim_boundary"],
                    "sources": source_receipts,
                }
            )
        manifest = {
            "schema_version": 1,
            "plan_version": TABLE_PLAN_VERSION,
            "status": "complete",
            "scope": scope,
            **identity,
            "metric_schema_hash": metric_schema_hash(),
            "inventory_mode": "closed_world_runner_owned",
            "table_count": len(table_receipts),
            "tables": table_receipts,
        }
        manifest_path = staging / "table_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        validate_table_package(
            staging,
            run_root=root,
            config=config,
            scope=scope,
            **identity,
        )
        atomic_replace_directory(staging, output)
    except Exception:
        raise
    return {
        **{key: output / path.name for key, path in written.items()},
        "manifest": output / "table_manifest.json",
    }
