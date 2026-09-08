"""Export independently validated aggregate-only HR alias evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.data.canonical_loader import sha256_file
from src.governance.hr_target_alias_run_validator_v4 import validate_hr_target_alias_run_v4


PUBLISHED = (
    "aggregate_metrics.csv", "per_class_metrics.csv", "confusion_matrix.csv",
    "comparison_deltas.csv", "selected_hyperparameters.csv", "population_receipt.json",
)
EXPECTED = {*PUBLISHED, "README.md", "provenance_receipt.json", "manifest.json"}


class HRTargetAliasCompactExportV4Error(RuntimeError):
    """Raised when compact HR alias evidence violates its closed-world contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HRTargetAliasCompactExportV4Error(message)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _markdown(run_id: str, root: Path) -> str:
    aggregate = pd.read_csv(root / "aggregate_metrics.csv")
    deltas = pd.read_csv(root / "comparison_deltas.csv")
    selected = pd.read_csv(root / "selected_hyperparameters.csv")
    priority = ["macro_f1", "balanced_accuracy", "quadratic_weighted_kappa", "ordinal_mae", "ranked_probability_score", "nll_log_loss", "multiclass_brier", "ece_confidence"]
    lines = [
        "# HR Target-Alias Matched-Sample Sensitivity", "", f"Validated local source run: `{run_id}`.", "",
        "The historical 311-row Phase 3A result is preserved. Its existing predictions are also restricted fit-free to the 309 rows without target-alias disagreements. A separate model is selected and refitted after excluding those two rows, then evaluated on exactly the same 309-row population.", "",
        "## Arm metrics", "", "| Arm | N | Macro-F1 | Balanced accuracy | QWK | Ordinal MAE | RPS | Log loss | Brier | ECE |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    pivot = aggregate.loc[aggregate["metric"].isin(priority)].pivot(index="arm", columns="metric", values="value")
    counts = aggregate.groupby("arm")["sample_count"].first()
    for arm in ("historical_canonical_311", "restricted_canonical_309", "exclusion_refit_309"):
        lines.append(
            f"| {arm} | {int(counts.loc[arm])} | {pivot.loc[arm, 'macro_f1']:.6f} | {pivot.loc[arm, 'balanced_accuracy']:.6f} | {pivot.loc[arm, 'quadratic_weighted_kappa']:.6f} | {pivot.loc[arm, 'ordinal_mae']:.6f} | {pivot.loc[arm, 'ranked_probability_score']:.6f} | {pivot.loc[arm, 'nll_log_loss']:.6f} | {pivot.loc[arm, 'multiclass_brier']:.6f} | {pivot.loc[arm, 'ece_confidence']:.6f} |"
        )
    lines.extend(["", "## Comparison boundaries", ""])
    for comparison_id in ("sample_removal_effect", "matched_refit_effect"):
        scoped = deltas.loc[deltas["comparison_id"] == comparison_id]
        lines.append(f"- `{comparison_id}`: `{scoped.iloc[0]['left_arm']}` → `{scoped.iloc[0]['right_arm']}`; `{scoped.iloc[0]['interpretation']}`.")
    lines.extend(["", "The first contrast changes only the evaluated sample set and is fit-free. The second is the primary training/data-rule sensitivity because both arms contain identical sample, fold, and target identities.", "", "## Candidate schedule", "", "| Outer fold | Historical candidate | Exclusion/refit candidate | Changed |", "| ---: | ---: | ---: | --- |"])
    for row in selected.itertuples(index=False):
        lines.append(f"| {int(row.outer_fold)} | {int(row.historical_candidate_index)} | {int(row.exclusion_refit_candidate_index)} | {str(bool(row.selected_candidate_changed)).lower()} |")
    lines.extend(["", "## Interpretation boundary", "", "This limited repetition-1/raw-XGBoost analysis is a descriptive data-quality sensitivity. It is not an equivalence test, target-validity proof, robustness certification, calibration analysis, or cross-dataset performance claim. Row identities, OOF predictions, and candidate-level inner-fold scores are intentionally excluded.", ""])
    return "\n".join(lines)


def export_hr_target_alias_compact_v4(run_dir: Path | str, destination: Path | str) -> dict[str, Any]:
    source, target = Path(run_dir), Path(destination)
    _require(not target.exists(), f"Destination already exists: {target.as_posix()}.")
    validation = validate_hr_target_alias_run_v4(source)
    metadata = json.loads((source / "stage_metadata.json").read_text(encoding="utf-8"))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for name in PUBLISHED:
            shutil.copyfile(source / name, staging / name)
        (staging / "README.md").write_text(_markdown(metadata["run_id"], staging), encoding="utf-8", newline="\n")
        receipt = {
            "schema_version": 1, "package": "hr_target_alias_sensitivity_v4",
            "status": "independently_validated_aggregate_only", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run_id": metadata["run_id"], "source_generation_commit": metadata["git_identity"]["commit"],
            "source_scientific_input_sha256": metadata["scientific_input_sha256"],
            "source_output_hashes": metadata["output_hashes"], "run_validation": validation,
            "employee_level_oof_published": False, "candidate_level_scores_published": False,
            "row_identities_published": False, "fitted_models_published": False,
            "matched_refit_evaluation_population_identical": True,
            "sample_removal_effect_fit_free": True, "paid_api_calls": 0, "network_calls": 0,
        }
        (staging / "provenance_receipt.json").write_bytes(_json_bytes(receipt))
        payloads = sorted((path for path in staging.iterdir() if path.is_file()), key=lambda path: path.name.lower())
        file_rows = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in payloads]
        manifest = {
            "schema_version": 1, "package": "hr_target_alias_sensitivity_v4",
            "file_count_excluding_manifest": len(file_rows), "files": file_rows,
            "payload_set_sha256": hashlib.sha256(json.dumps(file_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        }
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    validation = validate_hr_target_alias_compact_v4(target)
    return {"status": "complete", "source_run_id": metadata["run_id"], "destination": target.as_posix(), **validation}


def validate_hr_target_alias_compact_v4(directory: Path | str) -> dict[str, Any]:
    root = Path(directory)
    files = {path.name for path in root.iterdir() if path.is_file()}
    _require(files == EXPECTED, f"Compact HR alias closed-world inventory drifted: {sorted(files)}.")
    _require(not any("oof" in name.lower() or "candidate_search" in name.lower() for name in files), "Row/candidate-level file leaked.")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_payload = sorted(EXPECTED - {"manifest.json"}, key=str.lower)
    _require([row["path"] for row in manifest["files"]] == expected_payload, "Manifest inventory drifted.")
    for row in manifest["files"]:
        path = root / row["path"]
        _require(path.stat().st_size == row["bytes"] and sha256_file(path) == row["sha256"], f"Manifest hash drifted: {path.name}.")
    population = json.loads((root / "population_receipt.json").read_text(encoding="utf-8"))
    _require(population["restricted_and_refit_sample_sets_identical"] is True, "Matched population receipt drifted.")
    _require(population["row_identities_publication_authorized"] is False, "Row-identity publication receipt drifted.")
    deltas = pd.read_csv(root / "comparison_deltas.csv")
    _require(deltas.loc[deltas["comparison_id"] == "matched_refit_effect", "evaluation_population_matched"].astype(bool).all(), "Matched-refit flags drifted.")
    return {"file_count": len(files), "payload_set_sha256": manifest["payload_set_sha256"], "row_identities_published": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = validate_hr_target_alias_compact_v4(args.destination) if args.validate_only else export_hr_target_alias_compact_v4(args.run_dir, args.destination)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["HRTargetAliasCompactExportV4Error", "export_hr_target_alias_compact_v4", "validate_hr_target_alias_compact_v4"]
