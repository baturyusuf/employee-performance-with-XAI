"""Export validated aggregate-only Round 2 selection/prior evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.data.canonical_loader import sha256_file
from src.governance.selection_objective_sensitivity_run_validator_v4 import (
    validate_selection_objective_sensitivity_run_v4,
)


PUBLISHED_CSVS = (
    "aggregate_metrics.csv",
    "per_class_metrics.csv",
    "confusion_matrix.csv",
    "selection_schedule.csv",
    "selected_candidate_changes.csv",
    "metric_effect_magnitudes.csv",
    "ranking_changes.csv",
    "empirical_prior_fold_parameters.csv",
    "empirical_prior_metrics.csv",
)
EXPECTED_FILES = {*PUBLISHED_CSVS, "README.md", "provenance_receipt.json", "manifest.json"}


class SelectionObjectiveCompactExportV4Error(RuntimeError):
    """Raised when compact Round 2 export or validation fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionObjectiveCompactExportV4Error(message)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _write(path: Path, value: bytes) -> None:
    path.write_bytes(value)


def _format_value(value: float) -> str:
    return f"{float(value):.6f}"


def _markdown(run_id: str, root: Path) -> str:
    rankings = pd.read_csv(root / "ranking_changes.csv")
    changes = pd.read_csv(root / "selected_candidate_changes.csv")
    effects = pd.read_csv(root / "metric_effect_magnitudes.csv")
    prior = pd.read_csv(root / "empirical_prior_metrics.csv")
    changed_by_model = changes.groupby("model", sort=True)["selected_candidate_changed"].sum().astype(int)
    lines = [
        "# Selection-Objective Sensitivity and Empirical-Prior Baseline",
        "",
        f"Validated local source run: `{run_id}`.",
        "",
        "Macro-F1 selection reuses the exact Phase 1B OOF predictions. QWK selection swaps the primary and tie-break metrics inside the same inclusive 0.001 pool and refits every model in every outer fold. The empirical prior uses only the current outer-training labels.",
        "",
        "## Candidate changes",
        "",
        "| Model | Changed folds | Total folds |",
        "| --- | ---: | ---: |",
    ]
    for model, count in changed_by_model.items():
        lines.append(f"| {model} | {count} | 10 |")
    lines.extend(
        [
            "",
            "## Ranking diagnostics",
            "",
            "`leader_changed` and `full_ordering_changed` are reported separately. A small lower-order permutation is not labelled material dependence.",
            "",
            "| Metric | Leader (macro-F1 selection) | Leader (QWK selection) | Leader changed | Full ordering changed |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in rankings.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.macro_f1_selection_leader} | {row.qwk_selection_leader} | {str(bool(row.leader_changed)).lower()} | {str(bool(row.full_ordering_changed)).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Maximum observed model-level score change by metric",
            "",
            "These are descriptive absolute magnitudes, not a composite materiality decision.",
            "",
            "| Metric | Maximum absolute change |",
            "| --- | ---: |",
        ]
    )
    maxima = effects.groupby("metric", sort=True)["absolute_effect_magnitude"].max()
    for metric, value in maxima.items():
        lines.append(f"| {metric} | {_format_value(value)} |")
    lines.extend(
        [
            "",
            "## Outer-training empirical-prior probability baseline",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    probability_metrics = {"nll_log_loss", "multiclass_brier", "ranked_probability_score", "ece_confidence"}
    for row in prior.loc[prior["metric"].isin(probability_metrics)].sort_values("metric").itertuples(index=False):
        lines.append(f"| {row.metric} | {_format_value(row.value)} |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The tables describe sensitivity to a prespecified hyperparameter-selection objective under one dataset, policy, fold schedule, candidate registry, and seed contract. They do not establish a universal best model, statistical significance, deployment validity, or causal HR evidence. Row-level OOF predictions and fitted internals are intentionally excluded from this compact package.",
            "",
        ]
    )
    return "\n".join(lines)


def export_selection_objective_sensitivity_compact_v4(
    run_dir: Path | str, destination: Path | str
) -> dict[str, Any]:
    source = Path(run_dir)
    target = Path(destination)
    _require(not target.exists(), f"Destination already exists: {target.as_posix()}.")
    validation = validate_selection_objective_sensitivity_run_v4(source)
    metadata = json.loads((source / "stage_metadata.json").read_text(encoding="utf-8"))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for name in PUBLISHED_CSVS:
            shutil.copyfile(source / name, staging / name)
        _write(staging / "README.md", _markdown(metadata["run_id"], staging).encode("utf-8"))
        receipt = {
            "schema_version": 1, "package": "selection_objective_sensitivity_v4",
            "status": "independently_validated_aggregate_only", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run_id": metadata["run_id"], "source_generation_commit": metadata["git_identity"]["commit"],
            "source_scientific_input_sha256": metadata["scientific_input_sha256"],
            "source_output_hashes": metadata["output_hashes"], "run_validation": validation,
            "published_csvs": list(PUBLISHED_CSVS), "employee_level_oof_published": False,
            "fitted_models_published": False, "candidate_level_inner_scores_published": False,
            "composite_material_dependence_flag_present": False, "paid_api_calls": 0, "network_calls": 0,
        }
        _write(staging / "provenance_receipt.json", _json_bytes(receipt))
        payload_files = sorted(path for path in staging.iterdir() if path.is_file())
        manifest = {
            "schema_version": 1, "package": "selection_objective_sensitivity_v4",
            "file_count_excluding_manifest": len(payload_files),
            "files": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in payload_files
            ],
        }
        manifest["payload_set_sha256"] = hashlib.sha256(
            json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        _write(staging / "manifest.json", _json_bytes(manifest))
        os.replace(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    validate_selection_objective_sensitivity_compact_v4(target)
    return {
        "status": "complete", "destination": target.as_posix(),
        "source_run_id": metadata["run_id"], "file_count": len(EXPECTED_FILES),
        "employee_level_oof_published": False, "paid_api_calls": 0, "network_calls": 0,
    }


def validate_selection_objective_sensitivity_compact_v4(directory: Path | str) -> dict[str, Any]:
    root = Path(directory)
    files = {path.name for path in root.iterdir() if path.is_file()}
    _require(files == EXPECTED_FILES, f"Compact package closed-world inventory drifted: {sorted(files)}.")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_payload = sorted(EXPECTED_FILES - {"manifest.json"})
    _require([row["path"] for row in manifest["files"]] == expected_payload, "Manifest inventory drifted.")
    for row in manifest["files"]:
        path = root / row["path"]
        _require(path.stat().st_size == row["bytes"] and sha256_file(path) == row["sha256"], f"Manifest hash drifted: {path.name}.")
    _require(not any("oof" in name.lower() for name in files), "OOF file leaked into compact package.")
    receipt = json.loads((root / "provenance_receipt.json").read_text(encoding="utf-8"))
    _require(receipt["employee_level_oof_published"] is False, "OOF publication receipt drifted.")
    _require(receipt["composite_material_dependence_flag_present"] is False, "Composite materiality flag is prohibited.")
    rankings = pd.read_csv(root / "ranking_changes.csv")
    _require({"leader_changed", "full_ordering_changed"}.issubset(rankings.columns), "Ranking fields are incomplete.")
    _require(not any("material" in column.lower() for column in rankings.columns), "Ranking table contains a materiality field.")
    return {
        "status": "passed", "file_count": len(files), "payload_set_sha256": manifest["payload_set_sha256"],
        "source_run_id": receipt["source_run_id"], "employee_level_oof_published": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        validate_selection_objective_sensitivity_compact_v4(args.destination)
        if args.validate_only
        else export_selection_objective_sensitivity_compact_v4(args.run_dir, args.destination)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SelectionObjectiveCompactExportV4Error", "export_selection_objective_sensitivity_compact_v4",
    "validate_selection_objective_sensitivity_compact_v4",
]
