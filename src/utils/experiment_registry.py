from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable

from src.utils.config_loader import PROJECT_ROOT


REGISTRY_COLUMNS = [
    "run_id",
    "date_time",
    "git_commit_if_available",
    "script",
    "config",
    "feature_set",
    "model",
    "seed",
    "cv_strategy",
    "primary_metrics",
    "output_dir",
    "notes",
    "decision_status",
]


def default_registry_path() -> Path:
    return PROJECT_ROOT / "reports" / "research_log" / "experiment_registry.csv"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def normalize_registry_row(row: Dict[str, Any]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for col in REGISTRY_COLUMNS:
        value = row.get(col, "")
        if isinstance(value, (dict, list, tuple)):
            normalized[col] = json.dumps(value, sort_keys=True)
        else:
            normalized[col] = "" if value is None else str(value)
    return normalized


def append_registry_row(row: Dict[str, Any], registry_path: Path | None = None) -> Path:
    path = registry_path or default_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(normalize_registry_row(row))

    return path


def read_registry(registry_path: Path | None = None) -> list[Dict[str, str]]:
    path = registry_path or default_registry_path()
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_package_versions(package_names: Iterable[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def write_run_metadata(output_dir: Path, metadata_payload: Dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "date_time_utc": utc_now_iso(),
        "git_commit_if_available": get_git_commit(),
        **metadata_payload,
    }
    path = output_dir / "run_metadata.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
