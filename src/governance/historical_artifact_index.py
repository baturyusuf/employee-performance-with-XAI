"""Inventory pre-canonical report artifacts without admitting them to a new run.

Historical outputs remain useful for audit and diagnosis, but the absence of a
shared canonical run/config contract means they cannot be mixed with manuscript
evidence.  This module records that boundary while preserving the original files.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.governance.manuscript_contract import (
    artifact_feature_names,
    canonical_config_hash,
    forbidden_feature_mentions,
    load_manuscript_config,
    sha256_file,
)
from src.utils.config_loader import PROJECT_ROOT


DEFAULT_HISTORICAL_ROOTS = (
    "reports/model_selection",
    "reports/xai/final_candidates",
    "reports/counterfactuals/final_candidates",
    "reports/calibration/final_candidates",
    "reports/llm_explanations",
    "reports/agent_audits",
    "reports/chatbot_eval",
    "reports/governance_reports",
    "reports/external_validation",
    "reports/manuscript_assets/final_evidence_manifest",
    "reports/leakage",
    "reports/leakage_safe",
)


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _identity_from_artifact(path: Path) -> tuple[str, str]:
    """Best-effort structured identity extraction; values are never inferred."""

    try:
        if path.suffix.casefold() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                return str(payload.get("run_id", "")), str(payload.get("config_hash", ""))
        if path.suffix.casefold() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream), None)
            if row:
                return str(row.get("run_id", "")), str(row.get("config_hash", ""))
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error):
        pass
    return "", ""


def _forbidden_mentions(path: Path, config: Mapping[str, Any]) -> str:
    if path.suffix.casefold() not in {".csv", ".json", ".jsonl"}:
        return "not_structurally_scanned"
    try:
        names = artifact_feature_names(path)
    except Exception:
        return "scan_not_applicable_or_failed"
    mentions = forbidden_feature_mentions(names, config)
    return ";".join(sorted(mentions)) if mentions else "none_detected_in_structured_feature_fields"


def build_historical_artifact_index(
    output_path: str | Path,
    *,
    config_path: str | Path = "configs/manuscript_final.yaml",
    roots: Iterable[str | Path] = DEFAULT_HISTORICAL_ROOTS,
    canonical_run_id: str,
) -> Path:
    config = load_manuscript_config(config_path)
    current_hash = canonical_config_hash(config)
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            resolved = path.resolve()
            if not path.is_file() or resolved in seen or "__pycache__" in path.parts:
                continue
            seen.add(resolved)
            observed_run_id, observed_config_hash = _identity_from_artifact(path)
            reasons: list[str] = []
            if not observed_run_id:
                reasons.append("run_id_absent")
            elif observed_run_id != canonical_run_id:
                reasons.append("different_run_id")
            if not observed_config_hash:
                reasons.append("config_hash_absent")
            elif observed_config_hash != current_hash:
                reasons.append("different_config_hash")
            if not reasons:
                reasons.append("outside_versioned_canonical_run_root")
            stat = path.stat()
            records.append(
                {
                    "canonical_run_id": canonical_run_id,
                    "canonical_config_hash": current_hash,
                    "historical_path": _portable(path),
                    "sha256": sha256_file(path),
                    "size_bytes": stat.st_size,
                    "last_modified_utc": datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).isoformat(),
                    "observed_run_id": observed_run_id,
                    "observed_config_hash": observed_config_hash,
                    "compatibility_label": "historical_not_admitted_to_canonical_package",
                    "compatibility_reason": ";".join(reasons),
                    "forbidden_primary_feature_scan": _forbidden_mentions(path, config),
                    "original_file_preserved": True,
                    "scientific_values_reused": False,
                }
            )
    if not records:
        raise RuntimeError("No historical artifacts were found under the configured report roots.")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index and label historical pre-canonical artifacts.")
    parser.add_argument("--config", default="configs/manuscript_final.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        build_historical_artifact_index(
            arguments.output,
            config_path=arguments.config,
            canonical_run_id=arguments.run_id,
        )
    )
