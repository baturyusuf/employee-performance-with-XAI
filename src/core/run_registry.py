from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from src.core.io_utils import ensure_dir
from src.utils.config import SETTINGS
from src.utils.experiment_registry import get_git_commit, utc_now_iso


RUN_REGISTRY_COLUMNS = [
    "run_id",
    "timestamp",
    "git_commit_hash",
    "command",
    "config_path",
    "dataset",
    "model",
    "feature_policy",
    "llm_provider",
    "llm_model",
    "seed",
    "output_files",
]


@dataclass
class RunRegistryEntry:
    run_id: str
    command: str
    config_path: str = ""
    dataset: str = ""
    model: str = ""
    feature_policy: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    seed: str = ""
    output_files: List[str] = field(default_factory=list)
    timestamp: str = ""
    git_commit_hash: str = ""

    def normalized(self) -> Dict[str, str]:
        payload = asdict(self)
        if not payload["timestamp"]:
            payload["timestamp"] = utc_now_iso()
        if not payload["git_commit_hash"]:
            payload["git_commit_hash"] = get_git_commit()
        payload["output_files"] = json.dumps(payload.get("output_files", []), sort_keys=True)
        return {col: str(payload.get(col, "")) for col in RUN_REGISTRY_COLUMNS}


def default_run_registry_path() -> Path:
    return SETTINGS.reports_dir / "research_log" / "run_registry.csv"


def append_run_entry(entry: RunRegistryEntry, path: Path | None = None) -> Path:
    registry_path = path or default_run_registry_path()
    ensure_dir(registry_path.parent)
    exists = registry_path.exists() and registry_path.stat().st_size > 0
    with registry_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_REGISTRY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(entry.normalized())
    return registry_path


def read_run_registry(path: Path | None = None) -> List[Dict[str, str]]:
    registry_path = path or default_run_registry_path()
    if not registry_path.exists():
        return []
    with registry_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
