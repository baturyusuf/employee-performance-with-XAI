"""Create and validate a publication-safe compact Phase 2B evidence package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.data.canonical_loader import sha256_file
from src.governance.calibration_diagnostics_run_validator_v3 import (
    DEFAULT_CALIBRATION_DIAGNOSTICS_RUN,
    validate_calibration_diagnostics_run_v3,
)


DEFAULT_OUTPUT = Path(
    "reports/research_log/major_revision_v3/phase2b_calibration_diagnostics"
)
MANIFEST_NAME = "manifest.json"
DIRECT_EXPORTS = {
    "calibration_metric_summary.csv": "calibration_metric_summary.csv",
    "classwise_calibration_metrics.csv": "classwise_calibration_metrics.csv",
    "cumulative_calibration_metrics.csv": "cumulative_calibration_metrics.csv",
    "extended_reliability_bins.csv": "extended_reliability_bins.csv",
    "method_comparison.csv": "method_comparison.csv",
    "classwise_reliability.png": "classwise_reliability.png",
    "classwise_reliability.svg": "classwise_reliability.svg",
    "cumulative_reliability.png": "cumulative_reliability.png",
    "cumulative_reliability.svg": "cumulative_reliability.svg",
    "diagnostic_receipt.json": "diagnostic_receipt.json",
}
EXPECTED_EXPORT_FILES = frozenset(
    {"README.md", "provenance_receipt.json", MANIFEST_NAME, *DIRECT_EXPORTS}
)
CSV_ROW_COUNTS = {
    "calibration_metric_summary.csv": 2,
    "classwise_calibration_metrics.csv": 6,
    "cumulative_calibration_metrics.csv": 4,
    "extended_reliability_bins.csv": 120,
    "method_comparison.csv": 6,
}
FORBIDDEN_PUBLIC_COLUMN_TOKENS = (
    "sample_index",
    "employee",
    "empnumber",
    "y_true",
    "y_pred",
    "prob_class_",
    "sample_key",
    "outer_fold",
)


class V3CalibrationDiagnosticsCompactExportError(RuntimeError):
    """Raised when a compact Phase 2B export is unsafe or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3CalibrationDiagnosticsCompactExportError(message)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _markdown_summary(summary: pd.DataFrame, comparison: pd.DataFrame, *, run_id: str) -> str:
    values = summary.set_index("method")
    changes = comparison.set_index("metric")
    metrics = (
        ("nll_log_loss", "Log loss"),
        ("multiclass_brier", "Multiclass Brier"),
        ("top_label_ece", "Top-label ECE"),
        ("macro_classwise_ece", "Macro classwise ECE"),
        ("mean_cumulative_ece", "Mean cumulative ECE"),
        ("ranked_probability_score", "Normalized RPS"),
    )
    rows = [
        "| Metric | Raw | Sigmoid | Sigmoid − raw | Direction-aligned improvement |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, label in metrics:
        rows.append(
            f"| {label} | {values.loc['raw', metric]:.4f} | "
            f"{values.loc['sigmoid', metric]:.4f} | "
            f"{changes.loc[metric, 'raw_difference_sigmoid_minus_raw']:+.4f} | "
            f"{changes.loc[metric, 'direction_aligned_improvement']:+.4f} |"
        )
    return "\n".join(
        [
            "# Phase 2B Extended Calibration Diagnostics — Compact Evidence",
            "",
            f"Source run: `{run_id}`",
            "",
            "This package contains aggregate OOF calibration diagnostics and reliability figures only. Employee-level OOF probabilities, fold assignments, fitted performance models, probability-calibrator objects/parameters, and raw data are deliberately excluded.",
            "",
            "## Preserved method",
            "",
            "The primary sigmoid method was predeclared in canonical v2. Within each outer fold, three one-vs-rest Platt calibrators were trained only on five-fold cross-fitted outer-training probabilities; their positive outputs were divided by the row sum to restore the multiclass probability simplex. The untouched outer-test fold was evaluation-only. Phase 2B fits no performance model or probability calibrator and does not use these results to select a method.",
            "",
            "## Aggregate results",
            "",
            "All displayed metrics are lower-is-better. A positive direction-aligned value favors sigmoid.",
            "",
            *rows,
            "",
            "Sigmoid improves the log-loss, multiclass-Brier, macro classwise-ECE, mean cumulative-ECE, and normalized-RPS point estimates, but its top-label ECE is worse. The legacy paired interval for the top-label-ECE difference spans zero. No interval was estimated for the new diagnostic differences, so this package does not support an ‘all metrics improved’ claim.",
            "",
            "## Reliability and ordinal definitions",
            "",
            "- Top-label ECE bins maximum predicted probability against argmax correctness; classwise ECE treats each class as one-vs-rest; cumulative ECE uses the ordered events Y≤2 and Y≤3.",
            "- Every ECE uses ten fixed equal-width bins on [0,1]. The first bin is closed at both ends; later bins are left-open/right-closed. Empty bins remain explicit in `extended_reliability_bins.csv` and are omitted only from plotted lines.",
            "- Normalized RPS is the mean squared cumulative-distribution error over the two nontrivial thresholds. It equals the mean of their binary Brier scores.",
            "- Calibration intercept/slope values are unpenalized pooled exactly-once-OOF descriptive diagnostics fitted on the same prediction set. They are not confidence-bounded future-calibration validation.",
            "",
            "## Interpretation boundaries",
            "",
            "- ECE depends on the selected bins, and rare-class/high-probability bins can be empty or sparse. The diagrams must be read together with bin support.",
            "- Raw versus sigmoid is a predeclared evaluation comparison, not test-set selection among calibration methods.",
            "- The probabilities are model outputs for this cross-sectional research construct, not objective employee-outcome probabilities or validated decision thresholds.",
            "- The evidence does not establish prospective calibration, fairness, causal effects, human usefulness, legal compliance, or deployment readiness for HR decisions.",
            "",
            "## Files",
            "",
            "- `calibration_metric_summary.csv` and `method_comparison.csv`: aggregate raw/sigmoid metrics and bounded contrasts.",
            "- `classwise_calibration_metrics.csv` and `cumulative_calibration_metrics.csv`: per-event ECE, Brier, log loss, intercept, and slope.",
            "- `extended_reliability_bins.csv`: all 120 top-label/classwise/cumulative bins, including empty bins.",
            "- `classwise_reliability.*` and `cumulative_reliability.*`: 300-DPI PNG and editable SVG reliability diagrams.",
            "- `diagnostic_receipt.json`, `provenance_receipt.json`, and `manifest.json`: method, lineage, exclusions, independent validation, and byte hashes.",
        ]
    ) + "\n"


def export_calibration_diagnostics_compact_v3(
    source_run: Path | str = DEFAULT_CALIBRATION_DIAGNOSTICS_RUN,
    output_dir: Path | str = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate the full run and atomically export only compact safe evidence."""

    source = Path(source_run)
    destination = Path(output_dir)
    _require(not destination.exists(), f"Compact destination already exists: {destination.as_posix()}.")
    run_receipt = validate_calibration_diagnostics_run_v3(source)
    source_metadata = json.loads((source / "stage_metadata.json").read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for output_name, source_name in DIRECT_EXPORTS.items():
            shutil.copyfile(source / source_name, staging / output_name)
        readme = _markdown_summary(
            pd.read_csv(staging / "calibration_metric_summary.csv"),
            pd.read_csv(staging / "method_comparison.csv"),
            run_id=run_receipt["run_id"],
        )
        _write_bytes(staging / "README.md", readme.encode("utf-8"))
        source_files = {
            path.name: {
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in sorted(source.iterdir())
            if path.is_file()
        }
        provenance = {
            "schema_version": 1,
            "package_kind": "phase2b_calibration_diagnostics_compact_evidence",
            "source_run": source.as_posix(),
            "source_run_id": run_receipt["run_id"],
            "source_generation_commit": run_receipt["generation_commit"],
            "contract_sha256": run_receipt["contract_sha256"],
            "scientific_input_sha256": run_receipt["scientific_input_sha256"],
            "source_created_at_utc": source_metadata["created_at_utc"],
            "independent_run_validation": run_receipt,
            "source_files": source_files,
            "included_files": sorted(DIRECT_EXPORTS),
            "excluded_source_files": sorted(set(source_files) - set(DIRECT_EXPORTS)),
            "publication_controls": {
                "employee_level_oof_probabilities_included": False,
                "fold_assignments_included": False,
                "raw_data_included": False,
                "fitted_models_included": False,
                "fitted_calibrators_or_parameters_included": False,
                "new_calibration_method_selected": False,
                "inferential_claim_for_new_diagnostics_allowed": False,
                "future_calibration_claim_allowed": False,
                "all_metrics_improved_claim_allowed": False,
                "deployment_claim_allowed": False,
            },
            "row_counts": {
                filename: int(len(pd.read_csv(staging / filename)))
                for filename in CSV_ROW_COUNTS
            },
        }
        _write_bytes(staging / "provenance_receipt.json", _json_bytes(provenance))
        records = [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in sorted(staging.iterdir())
            if path.is_file()
        ]
        manifest = {
            "schema_version": 1,
            "package_kind": "phase2b_calibration_diagnostics_compact_evidence",
            "source_run_id": run_receipt["run_id"],
            "file_count_excluding_manifest": len(records),
            "files": records,
        }
        _write_bytes(staging / MANIFEST_NAME, _json_bytes(manifest))
        _require({path.name for path in staging.iterdir() if path.is_file()} == EXPECTED_EXPORT_FILES, "Compact Phase 2B staging inventory drifted.")
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            for child in sorted(staging.iterdir(), reverse=True):
                if child.is_file():
                    child.unlink()
            staging.rmdir()
        raise
    return validate_calibration_diagnostics_compact_v3(destination, source_run=source)


def validate_calibration_diagnostics_compact_v3(
    package_dir: Path | str = DEFAULT_OUTPUT,
    *,
    source_run: Path | str = DEFAULT_CALIBRATION_DIAGNOSTICS_RUN,
) -> dict[str, Any]:
    """Validate compact contents, source equivalence, exclusions, and hashes."""

    package = Path(package_dir)
    source = Path(source_run)
    _require(package.is_dir(), f"Compact Phase 2B package is absent: {package.as_posix()}.")
    inventory = {path.name for path in package.iterdir() if path.is_file()}
    _require(inventory == EXPECTED_EXPORT_FILES, f"Compact Phase 2B closed-world inventory drifted: {sorted(inventory ^ EXPECTED_EXPORT_FILES)}.")
    _require(not any(path.is_dir() for path in package.iterdir()), "Compact Phase 2B package contains a directory.")
    run_receipt = validate_calibration_diagnostics_run_v3(source)
    manifest = json.loads((package / MANIFEST_NAME).read_text(encoding="utf-8"))
    _require(manifest.get("source_run_id") == run_receipt["run_id"], "Compact Phase 2B manifest run id drifted.")
    records = manifest.get("files")
    _require(isinstance(records, list), "Compact Phase 2B manifest records are absent.")
    _require(manifest.get("file_count_excluding_manifest") == len(EXPECTED_EXPORT_FILES) - 1, "Compact Phase 2B manifest count drifted.")
    _require({record.get("path") for record in records} == EXPECTED_EXPORT_FILES - {MANIFEST_NAME}, "Compact Phase 2B manifest inventory drifted.")
    for record in records:
        path = package / str(record["path"])
        _require(path.stat().st_size == int(record["size_bytes"]), f"Compact Phase 2B size drifted for {path.name}.")
        _require(sha256_file(path) == record["sha256"], f"Compact Phase 2B hash drifted for {path.name}.")
    provenance = json.loads((package / "provenance_receipt.json").read_text(encoding="utf-8"))
    _require(provenance.get("source_run_id") == run_receipt["run_id"], "Compact Phase 2B provenance run id drifted.")
    _require(provenance.get("source_generation_commit") == run_receipt["generation_commit"], "Compact Phase 2B provenance commit drifted.")
    _require(provenance.get("independent_run_validation") == run_receipt, "Compact Phase 2B validation receipt drifted.")
    controls = provenance.get("publication_controls")
    _require(isinstance(controls, Mapping), "Compact Phase 2B controls are absent.")
    for field in (
        "employee_level_oof_probabilities_included",
        "fold_assignments_included",
        "raw_data_included",
        "fitted_models_included",
        "fitted_calibrators_or_parameters_included",
        "new_calibration_method_selected",
        "inferential_claim_for_new_diagnostics_allowed",
        "future_calibration_claim_allowed",
        "all_metrics_improved_claim_allowed",
        "deployment_claim_allowed",
    ):
        _require(controls.get(field) is False, f"Compact Phase 2B control {field} drifted.")
    row_counts: dict[str, int] = {}
    for output_name, source_name in DIRECT_EXPORTS.items():
        compact_path = package / output_name
        source_path = source / source_name
        _require(sha256_file(compact_path) == sha256_file(source_path), f"Compact/source bytes drifted for {output_name}.")
        if output_name.endswith(".csv"):
            frame = pd.read_csv(compact_path)
            row_counts[output_name] = len(frame)
            normalized = [str(column).casefold() for column in frame.columns]
            for token in FORBIDDEN_PUBLIC_COLUMN_TOKENS:
                _require(not any(token in column for column in normalized), f"Compact {output_name} exposes forbidden column token {token}.")
    _require(row_counts == CSV_ROW_COUNTS, "Compact Phase 2B row counts drifted.")
    _require(provenance.get("row_counts") == row_counts, "Compact Phase 2B provenance row counts drifted.")
    readme = (package / "README.md").read_text(encoding="utf-8")
    for required in (
        "top-label ECE is worse",
        "does not support an ‘all metrics improved’ claim",
        "ten fixed equal-width bins",
        "Empty bins remain explicit",
        "not confidence-bounded future-calibration validation",
        "not test-set selection",
        "not objective employee-outcome probabilities",
        "deployment readiness",
        "Employee-level OOF probabilities",
    ):
        _require(required in readme, f"Compact Phase 2B boundary is absent: {required}.")
    return {
        "status": "passed",
        "source_run_id": run_receipt["run_id"],
        "source_generation_commit": run_receipt["generation_commit"],
        "file_count": len(EXPECTED_EXPORT_FILES),
        "total_size_bytes": sum(path.stat().st_size for path in package.iterdir() if path.is_file()),
        "manifest_sha256": sha256_file(package / MANIFEST_NAME),
        "row_counts": row_counts,
        "employee_level_rows_included": False,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_CALIBRATION_DIAGNOSTICS_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt = (
        validate_calibration_diagnostics_compact_v3(args.output_dir, source_run=args.source_run)
        if args.validate_only
        else export_calibration_diagnostics_compact_v3(args.source_run, args.output_dir)
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
