"""Fail-closed loading and acquisition verification for scientific datasets.

The scientific pipeline must never discover an interim, cache, or mirror by
convention.  A canonical config selects a logical dataset, the acquisition
manifest binds it to one physical file, and this module verifies the exact
bytes and scientific schema before returning any dataframe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd

from src.data.validate_schema import validate_dataframe
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_ACQUISITION_MANIFEST = PROJECT_ROOT / "configs" / "data_acquisition.yaml"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class CanonicalDataError(RuntimeError):
    """Base error for a canonical data-contract violation."""


class DataIntegrityError(CanonicalDataError):
    """Raised when local or downloaded bytes do not match the pinned contract."""


class AcquisitionNotApprovedError(CanonicalDataError):
    """Raised when data are absent and no approved acquisition path is enabled."""


@dataclass(frozen=True)
class CanonicalDataset:
    """Verified dataframe and a portable receipt describing actual consumption."""

    frame: pd.DataFrame
    receipt: Mapping[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = config.get("manuscript_final", config)
    if not isinstance(settings, Mapping):
        raise CanonicalDataError("Canonical config must contain a manuscript_final mapping.")
    return settings


def _acquisition_settings(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = manifest.get("data_acquisition", manifest)
    if not isinstance(settings, Mapping):
        raise CanonicalDataError("Acquisition manifest must contain a data_acquisition mapping.")
    if settings.get("schema_version") != 1:
        raise CanonicalDataError("Unsupported acquisition manifest schema_version; expected 1.")
    for required in ("physical_datasets", "logical_bindings"):
        if not isinstance(settings.get(required), Mapping) or not settings[required]:
            raise CanonicalDataError(f"Acquisition manifest requires a non-empty {required} mapping.")
    return settings


def _resolve(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _portable_configured_path(
    raw_path: str | Path,
    resolved: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> str:
    configured = Path(raw_path)
    if not configured.is_absolute():
        return configured.as_posix()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        # Absolute paths are permitted only for isolated unit fixtures. Production
        # artifact portability validation rejects them before packaging.
        return resolved.as_posix()


def _distribution(series: pd.Series) -> Dict[str, int]:
    counts = series.value_counts(dropna=False, sort=False)
    result: Dict[str, int] = {}
    for value, count in counts.items():
        if pd.isna(value):
            key = "<NA>"
        elif isinstance(value, float) and value.is_integer():
            key = str(int(value))
        else:
            key = str(value).strip()
        result[key] = int(count)
    return dict(sorted(result.items()))


def _normalise_expected_distribution(value: Any, context: str) -> Dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise CanonicalDataError(f"{context} must be a non-empty mapping.")
    try:
        return dict(sorted((str(label), int(count)) for label, count in value.items()))
    except (TypeError, ValueError) as exc:
        raise CanonicalDataError(f"{context} must map target labels to integer counts.") from exc


def _read_frame(path: Path, record: Mapping[str, Any]) -> pd.DataFrame:
    file_format = str(record.get("format", "csv")).casefold()
    if file_format != "csv":
        raise CanonicalDataError(
            f"Canonical modeling inputs must be pinned CSV files; received format={file_format!r}, path={path}."
        )
    delimiter = str(record.get("delimiter", ","))
    encoding = str(record.get("encoding", "utf-8-sig"))
    frame = pd.read_csv(path, sep=delimiter, encoding=encoding)
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    if str(record.get("validation_profile", "")) == "inx_primary":
        frame, report = validate_dataframe(frame)
        if not report.is_valid or report.unexpected_columns:
            raise DataIntegrityError(
                "INX validation failed: "
                f"missing={report.missing_required_columns}, unexpected={report.unexpected_columns}, "
                f"duplicates={report.duplicate_id_count}, null_target={report.null_target_count}."
            )
    return frame


def _write_mismatch_report(path: str | Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _validate_file(
    path: Path,
    *,
    record: Mapping[str, Any],
    target_profile: Mapping[str, Any],
    dataset_key: str,
    mismatch_report_path: str | Path | None,
    project_root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    expected_hash = str(record.get("expected_sha256", "")).casefold()
    if not SHA256_PATTERN.fullmatch(expected_hash):
        raise CanonicalDataError(f"{dataset_key}: expected_sha256 must be a lowercase SHA-256 digest.")
    actual_hash = sha256_file(path)
    expected_rows = int(record.get("expected_rows", -1))
    expected_columns_raw = record.get("expected_columns")
    if not isinstance(expected_columns_raw, list) or not expected_columns_raw:
        raise CanonicalDataError(f"{dataset_key}: expected_columns must be a non-empty ordered list.")
    expected_columns = [str(value) for value in expected_columns_raw]
    expected_column_count = int(record.get("expected_column_count", len(expected_columns)))
    target_column = str(target_profile.get("raw_target", ""))
    expected_target = _normalise_expected_distribution(
        target_profile.get("expected_distribution"),
        f"{dataset_key}.target_profile.expected_distribution",
    )

    frame: pd.DataFrame | None = None
    parse_error: str | None = None
    try:
        frame = _read_frame(path, record)
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"

    observed_columns = list(frame.columns) if frame is not None else []
    observed_rows = len(frame) if frame is not None else None
    observed_target = (
        _distribution(frame[target_column])
        if frame is not None and target_column in frame.columns
        else {}
    )
    comparison = {
        "dataset_key": dataset_key,
        "candidate_path": _portable_configured_path(path, path, project_root=project_root),
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
        "expected_rows": expected_rows,
        "actual_rows": observed_rows,
        "expected_column_count": expected_column_count,
        "actual_column_count": len(observed_columns) if frame is not None else None,
        "expected_columns": expected_columns,
        "actual_columns": observed_columns,
        "target_column": target_column,
        "expected_target_distribution": expected_target,
        "actual_target_distribution": observed_target,
        "parse_error": parse_error,
    }
    differences: list[str] = []
    if actual_hash != expected_hash:
        differences.append("sha256")
    if observed_rows != expected_rows:
        differences.append("row_count")
    if len(observed_columns) != expected_column_count:
        differences.append("column_count")
    if observed_columns != expected_columns:
        differences.append("ordered_schema")
    if target_column not in observed_columns:
        differences.append("target_column")
    elif observed_target != expected_target:
        differences.append("target_distribution")
    if parse_error is not None:
        differences.append("parse_error")
    comparison["status"] = "passed" if not differences else "failed"
    comparison["differences"] = differences
    if differences or frame is None:
        _write_mismatch_report(mismatch_report_path, comparison)
        raise DataIntegrityError(
            f"Dataset {dataset_key!r} failed the pinned acquisition contract: {differences}."
        )
    return frame, comparison


def _download_candidate(url: str, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".canonical_download_", suffix=".tmp", dir=destination_dir
    )
    try:
        with os.fdopen(descriptor, "wb") as target, urllib.request.urlopen(url, timeout=60) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return Path(temporary_name)


def load_canonical_dataset(
    config_path: str | Path,
    dataset_key: str,
    acquisition_manifest_path: str | Path | None = None,
    *,
    allow_download: bool = False,
    mismatch_report_path: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> CanonicalDataset:
    """Load one explicitly configured dataset after exact contract validation."""

    root = Path(project_root).resolve()
    resolved_config_path = _resolve(config_path, root)
    config = load_config(resolved_config_path)
    settings = _settings(config)
    datasets = settings.get("datasets")
    if not isinstance(datasets, Mapping) or dataset_key not in datasets:
        raise CanonicalDataError(f"Dataset {dataset_key!r} is not declared by the canonical config.")
    dataset_definition = datasets[dataset_key]
    if not isinstance(dataset_definition, Mapping):
        raise CanonicalDataError(f"Configured dataset {dataset_key!r} must be a mapping.")
    configured_path = dataset_definition.get("path")
    if not isinstance(configured_path, str) or not configured_path:
        raise CanonicalDataError(f"Configured dataset {dataset_key!r} has no explicit path.")

    if acquisition_manifest_path is None:
        provenance = settings.get("provenance", {})
        configured_manifest = (
            provenance.get("data_acquisition_manifest")
            if isinstance(provenance, Mapping)
            else None
        )
        acquisition_manifest_path = configured_manifest or DEFAULT_ACQUISITION_MANIFEST
    manifest_path = _resolve(acquisition_manifest_path, root)
    manifest = load_config(manifest_path)
    acquisition = _acquisition_settings(manifest)
    bindings = acquisition["logical_bindings"]
    if dataset_key not in bindings:
        raise CanonicalDataError(f"Dataset {dataset_key!r} has no acquisition-manifest binding.")
    binding = bindings[dataset_key]
    if not isinstance(binding, Mapping):
        raise CanonicalDataError(f"Acquisition binding for {dataset_key!r} must be a mapping.")
    physical_id = str(binding.get("physical_dataset", ""))
    target_profile_name = str(binding.get("target_profile", ""))
    physical_datasets = acquisition["physical_datasets"]
    if physical_id not in physical_datasets:
        raise CanonicalDataError(f"Unknown physical dataset {physical_id!r} for {dataset_key!r}.")
    record = physical_datasets[physical_id]
    if not isinstance(record, Mapping):
        raise CanonicalDataError(f"Physical dataset {physical_id!r} must be a mapping.")
    target_profiles = record.get("target_profiles")
    if not isinstance(target_profiles, Mapping) or target_profile_name not in target_profiles:
        raise CanonicalDataError(
            f"Physical dataset {physical_id!r} has no target profile {target_profile_name!r}."
        )
    target_profile = target_profiles[target_profile_name]
    if not isinstance(target_profile, Mapping):
        raise CanonicalDataError(f"Target profile {target_profile_name!r} must be a mapping.")

    manifest_local_path = record.get("local_path")
    if not isinstance(manifest_local_path, str) or not manifest_local_path:
        raise CanonicalDataError(f"Physical dataset {physical_id!r} has no local_path.")
    resolved_configured = _resolve(configured_path, root)
    resolved_manifest = _resolve(manifest_local_path, root)
    if resolved_configured != resolved_manifest:
        raise CanonicalDataError(
            f"Config/acquisition path mismatch for {dataset_key!r}: "
            f"config={configured_path!r}, acquisition={manifest_local_path!r}."
        )

    acquisition_method = "existing_local_file"
    candidate = resolved_configured
    downloaded: Path | None = None
    if not candidate.is_file():
        approved_url = record.get("approved_download_url")
        approved = record.get("automatic_download_allowed") is True
        if not allow_download or not approved or not isinstance(approved_url, str) or not approved_url:
            raise AcquisitionNotApprovedError(
                f"Dataset {dataset_key!r} is missing at {configured_path!r}; "
                "no approved automatic acquisition is enabled."
            )
        try:
            downloaded = _download_candidate(approved_url, candidate.parent)
        except Exception as exc:
            _write_mismatch_report(
                mismatch_report_path,
                {
                    "dataset_key": dataset_key,
                    "status": "download_failed",
                    "approved_download_url": approved_url,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise CanonicalDataError(f"Approved acquisition failed for {dataset_key!r}: {exc}") from exc
        candidate = downloaded
        acquisition_method = "approved_manifest_download"

    try:
        frame, _comparison = _validate_file(
            candidate,
            record=record,
            target_profile=target_profile,
            dataset_key=dataset_key,
            mismatch_report_path=mismatch_report_path,
            project_root=root,
        )
        if downloaded is not None:
            os.replace(downloaded, resolved_configured)
            candidate = resolved_configured
    except Exception:
        if downloaded is not None and downloaded.exists():
            downloaded.unlink()
        raise

    actual_hash = sha256_file(candidate)
    target_column = str(target_profile["raw_target"])
    receipt = {
        "dataset_key": dataset_key,
        "physical_dataset_id": physical_id,
        "actual_path": _portable_configured_path(
            configured_path, candidate, project_root=root
        ),
        "actual_sha256": actual_hash,
        "size_bytes": int(candidate.stat().st_size),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "schema_status": "valid",
        "schema_columns": [str(value) for value in frame.columns],
        "target_column": target_column,
        "target_distribution": _distribution(frame[target_column]),
        "acquisition_method": acquisition_method,
        "acquisition_manifest_path": _portable_configured_path(
            acquisition_manifest_path, manifest_path, project_root=root
        ),
        "acquisition_manifest_sha256": sha256_file(manifest_path),
        "automatic_download_allowed": bool(record.get("automatic_download_allowed") is True),
        "source_authenticity_status": str(
            record.get("source_authenticity_status", record.get("source_status", "manual_review_required"))
        ),
        "licence_verification_status": str(
            record.get("licence_verification_status", record.get("licence_status", "manual_review_required"))
        ),
    }
    return CanonicalDataset(frame=frame, receipt=receipt)


def verify_configured_datasets(
    config_path: str | Path,
    acquisition_manifest_path: str | Path | None = None,
    *,
    dataset_keys: Iterable[str] | None = None,
    allow_download: bool = False,
    report_dir: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> Dict[str, CanonicalDataset]:
    root = Path(project_root).resolve()
    resolved_config_path = _resolve(config_path, root)
    config = load_config(resolved_config_path)
    settings = _settings(config)
    datasets = settings.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise CanonicalDataError("Canonical config has no datasets mapping.")
    selected = list(dataset_keys) if dataset_keys is not None else list(datasets)
    unknown = sorted(set(selected).difference(datasets))
    if unknown:
        raise CanonicalDataError(f"Requested datasets are outside the canonical config: {unknown}.")
    output: Dict[str, CanonicalDataset] = {}
    for key in selected:
        mismatch = Path(report_dir) / f"{key}_acquisition_comparison.json" if report_dir else None
        output[key] = load_canonical_dataset(
            resolved_config_path,
            key,
            acquisition_manifest_path,
            allow_download=allow_download,
            mismatch_report_path=mismatch,
            project_root=root,
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify configured scientific datasets fail-closed.")
    parser.add_argument("--config", default="configs/manuscript_final.yaml")
    parser.add_argument("--acquisition-manifest", default=str(DEFAULT_ACQUISITION_MANIFEST))
    parser.add_argument("--datasets", default="all", help="Comma-separated logical keys or all.")
    parser.add_argument("--allow-approved-download", action="store_true")
    parser.add_argument("--report-dir", default="reports/data_preflight")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    selected_keys = None if arguments.datasets == "all" else [
        value.strip() for value in arguments.datasets.split(",") if value.strip()
    ]
    verified = verify_configured_datasets(
        arguments.config,
        arguments.acquisition_manifest,
        dataset_keys=selected_keys,
        allow_download=arguments.allow_approved_download,
        report_dir=arguments.report_dir,
    )
    print(json.dumps({key: dict(value.receipt) for key, value in verified.items()}, indent=2, sort_keys=True))
