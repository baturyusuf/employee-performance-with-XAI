"""Create and validate a publication-safe compact Phase 2C evidence package."""

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
from src.governance.subgroup_proxy_use_run_validator_v3 import (
    DEFAULT_SUBGROUP_PROXY_USE_RUN,
    validate_subgroup_proxy_use_run_v3,
)


DEFAULT_OUTPUT = Path(
    "reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use"
)
MANIFEST_NAME = "manifest.json"
DIRECT_EXPORTS = {
    "subgroup_metric_grid.csv": "subgroup_metric_grid.csv",
    "subgroup_gap_sensitivity.csv": "subgroup_gap_sensitivity.csv",
    "primary_gap_bootstrap_intervals.csv": "primary_gap_bootstrap_intervals.csv",
    "proxy_prediction_change_by_department.csv": "proxy_prediction_change_by_department.csv",
    "jobrole_permutation_repetition.csv": "jobrole_permutation_repetition.csv",
    "jobrole_permutation_summary.csv": "jobrole_permutation_summary.csv",
    "department_reconstructability_metrics.csv": "department_reconstructability_metrics.csv",
    "department_reconstructability_differences.csv": "department_reconstructability_differences.csv",
    "diagnostic_receipt.json": "diagnostic_receipt.json",
}
EXPECTED_EXPORT_FILES = frozenset(
    {"README.md", "provenance_receipt.json", MANIFEST_NAME, *DIRECT_EXPORTS}
)
CSV_ROW_COUNTS = {
    "subgroup_metric_grid.csv": 2025,
    "subgroup_gap_sensitivity.csv": 486,
    "primary_gap_bootstrap_intervals.csv": 162,
    "proxy_prediction_change_by_department.csv": 7,
    "jobrole_permutation_repetition.csv": 40,
    "jobrole_permutation_summary.csv": 2,
    "department_reconstructability_metrics.csv": 6,
    "department_reconstructability_differences.csv": 3,
}
FORBIDDEN_PUBLIC_COLUMN_TOKENS = (
    "sample_index",
    "employee",
    "empnumber",
    "y_true",
    "y_pred",
    "sample_key",
    "outer_fold",
)


class V3SubgroupProxyUseCompactExportError(RuntimeError):
    """Raised when a compact Phase 2C export is unsafe or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3SubgroupProxyUseCompactExportError(message)


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


def _markdown_summary(
    proxy_change: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    reconstructability: pd.DataFrame,
    *,
    run_id: str,
) -> str:
    overall = proxy_change[proxy_change["scope"] == "overall"].iloc[0]
    schemes = permutation_summary.set_index("scheme")
    marginal = schemes.loc["marginal_within_outer_test_fold"]
    conditional = schemes.loc[
        "department_conditional_within_outer_test_fold_and_department"
    ]
    reconstruction = reconstructability.pivot(
        index="system_id", columns="metric", values="point_estimate"
    )
    primary_system = "no_salary_hike_no_attrition_no_department"
    reduced_system = "no_salary_hike_no_attrition_no_department_no_job_role"
    return "\n".join(
        [
            "# Phase 2C Subgroup and Proxy-Use Diagnostics — Compact Evidence",
            "",
            f"Source run: `{run_id}`",
            "",
            "This package contains support-aware aggregate OOF diagnostics only. The 1,200 employee-level paired prediction rows, 48,000 employee-level permutation rows, folds, raw data, and fitted models are deliberately excluded.",
            "",
            "## Complete subgroup audit",
            "",
            "The package retains all 2,025 declared group-metric rows for three exact canonical systems, six attributes, nine metrics, and support thresholds 20/30/50. Unsupported group or class-denominator cells remain explicit with missing estimates and a status; they are excluded from maximum-minus-minimum gaps rather than hidden.",
            "",
            "All 486 declared gap cells are present. P3 has 162 pointwise and exploratory familywise simultaneous intervals based on 5,000 employee-level bootstrap repetitions stratified by outer fold and true class. Eligibility is fixed before resampling. The simultaneous family covers every estimable P3 attribute/threshold/metric cell, so this package does not select only the largest observed gap. These intervals condition on the fitted models/folds and are not confirmatory fairness inference.",
            "",
            "## Exact P3 versus P3-minus-JobRole prediction changes",
            "",
            f"Overall mean total variation is {overall['mean_total_variation']:.4f}; the argmax prediction changes for {overall['prediction_change_rate']:.2%} of cases; and the macro-F1 difference (proxy-reduced minus P3) is {overall['delta_macro_f1']:+.4f}. Department-specific rows retain probability, confidence-margin, ordinal-margin, prediction-change, and ordinal-shift summaries.",
            "",
            "The comparator is exactly the canonical P3 model refitted without JobRole. It is not v3 P5, which has a broader timing/proxy exclusion contract. The paired differences therefore combine direct feature removal with the resulting refit and are not isolated causal effects.",
            "",
            "## JobRole perturbation sensitivity",
            "",
            "| Scheme | Mean total variation | Prediction-change rate | Mean macro-F1 change | Mean raw-margin drop |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| Marginal within outer fold | {marginal['mean_total_variation_mean']:.4f} | {marginal['prediction_change_rate_mean']:.4f} | {marginal['delta_macro_f1_mean']:+.4f} | {marginal['mean_original_predicted_class_raw_margin_drop_mean']:.4f} |",
            f"| Department-conditional within outer fold | {conditional['mean_total_variation_mean']:.4f} | {conditional['prediction_change_rate_mean']:.4f} | {conditional['delta_macro_f1_mean']:+.4f} | {conditional['mean_original_predicted_class_raw_margin_drop_mean']:.4f} |",
            "",
            "Each scheme uses 20 prespecified outcome-blind shuffles and the exact persisted P3 outer-fold models, with no refitting. Repetition variation is descriptive perturbation variability, not a confidence interval. Marginal shuffling can create out-of-distribution combinations; conditioning only on department is not a fully conditional permutation test.",
            "",
            "## Department reconstructability is a different question",
            "",
            f"With JobRole in the reconstruction feature space, department reconstruction accuracy/balanced accuracy/macro-F1 are {reconstruction.loc[primary_system, 'accuracy']:.4f}/{reconstruction.loc[primary_system, 'balanced_accuracy']:.4f}/{reconstruction.loc[primary_system, 'macro_f1']:.4f}. Without JobRole they are {reconstruction.loc[reduced_system, 'accuracy']:.4f}/{reconstruction.loc[reduced_system, 'balanced_accuracy']:.4f}/{reconstruction.loc[reduced_system, 'macro_f1']:.4f}.",
            "",
            "Reconstructability shows that department information exists in a feature space. It does not prove that the performance model used department. The paired/refit and permutation tables address model-output dependence separately, and neither establishes causality or discrimination.",
            "",
            "## Interpretation boundaries",
            "",
            "- This is a same-dataset, exactly-once-OOF exploratory audit conditional on the observed sample, support rules, folds, models, policies, and perturbations.",
            "- Small groups and rare true-class denominators are not silently pooled or suppressed. Age 60+ and other ineligible cells remain visible as unsupported.",
            "- Maximum gaps are selected over groups and metrics; all cells and multiplicity-aware exploratory intervals must be considered together.",
            "- No table certifies fairness, proves absence or presence of discrimination, identifies causal JobRole or department effects, validates legal compliance, or supports autonomous HR decisions or deployment readiness.",
            "",
            "## Files",
            "",
            "- `subgroup_metric_grid.csv`: complete supported/unsupported group metric grid.",
            "- `subgroup_gap_sensitivity.csv`: all maximum-minus-minimum gaps at n=20/30/50.",
            "- `primary_gap_bootstrap_intervals.csv`: P3 pointwise and simultaneous exploratory intervals.",
            "- `proxy_prediction_change_by_department.csv`: paired aggregate output changes overall/by department.",
            "- `jobrole_permutation_repetition.csv` and `jobrole_permutation_summary.csv`: 40 repetition-level and two scheme-level perturbation summaries.",
            "- `department_reconstructability_metrics.csv` and `department_reconstructability_differences.csv`: preserved reconstruction evidence, explicitly separated from performance-model use.",
            "- `diagnostic_receipt.json`, `provenance_receipt.json`, and `manifest.json`: method scope, independent validation, exclusions, lineage, and byte hashes.",
        ]
    ) + "\n"


def export_subgroup_proxy_use_compact_v3(
    source_run: Path | str = DEFAULT_SUBGROUP_PROXY_USE_RUN,
    output_dir: Path | str = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate the full run and atomically export only compact aggregate evidence."""

    source = Path(source_run)
    destination = Path(output_dir)
    _require(
        not destination.exists(),
        f"Compact destination already exists: {destination.as_posix()}.",
    )
    run_receipt = validate_subgroup_proxy_use_run_v3(source)
    source_metadata = json.loads(
        (source / "stage_metadata.json").read_text(encoding="utf-8")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for output_name, source_name in DIRECT_EXPORTS.items():
            shutil.copyfile(source / source_name, staging / output_name)
        readme = _markdown_summary(
            pd.read_csv(staging / "proxy_prediction_change_by_department.csv"),
            pd.read_csv(staging / "jobrole_permutation_summary.csv"),
            pd.read_csv(staging / "department_reconstructability_metrics.csv"),
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
            "package_kind": "phase2c_subgroup_proxy_use_compact_evidence",
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
                "employee_level_prediction_comparison_rows_included": False,
                "employee_level_permutation_rows_included": False,
                "fold_assignments_included": False,
                "raw_data_included": False,
                "fitted_models_included": False,
                "fairness_certification_allowed": False,
                "discrimination_claim_allowed": False,
                "causal_feature_effect_claim_allowed": False,
                "reconstructability_proves_performance_use": False,
                "P3_minus_JobRole_labelled_as_P5": False,
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
            "package_kind": "phase2c_subgroup_proxy_use_compact_evidence",
            "source_run_id": run_receipt["run_id"],
            "file_count_excluding_manifest": len(records),
            "files": records,
        }
        _write_bytes(staging / MANIFEST_NAME, _json_bytes(manifest))
        _require(
            {path.name for path in staging.iterdir() if path.is_file()}
            == EXPECTED_EXPORT_FILES,
            "Compact Phase 2C staging inventory drifted.",
        )
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            for child in sorted(staging.iterdir(), reverse=True):
                if child.is_file():
                    child.unlink()
            staging.rmdir()
        raise
    return validate_subgroup_proxy_use_compact_v3(destination, source_run=source)


def validate_subgroup_proxy_use_compact_v3(
    package_dir: Path | str = DEFAULT_OUTPUT,
    *,
    source_run: Path | str = DEFAULT_SUBGROUP_PROXY_USE_RUN,
) -> dict[str, Any]:
    """Validate compact contents, source equivalence, exclusions, and hashes."""

    package = Path(package_dir)
    source = Path(source_run)
    _require(package.is_dir(), f"Compact Phase 2C package is absent: {package.as_posix()}.")
    inventory = {path.name for path in package.iterdir() if path.is_file()}
    _require(
        inventory == EXPECTED_EXPORT_FILES,
        f"Compact Phase 2C closed-world inventory drifted: {sorted(inventory ^ EXPECTED_EXPORT_FILES)}.",
    )
    _require(not any(path.is_dir() for path in package.iterdir()), "Compact Phase 2C package contains a directory.")
    run_receipt = validate_subgroup_proxy_use_run_v3(source)
    manifest = json.loads((package / MANIFEST_NAME).read_text(encoding="utf-8"))
    _require(manifest.get("source_run_id") == run_receipt["run_id"], "Compact Phase 2C manifest run id drifted.")
    records = manifest.get("files")
    _require(isinstance(records, list), "Compact Phase 2C manifest records are absent.")
    _require(manifest.get("file_count_excluding_manifest") == len(EXPECTED_EXPORT_FILES) - 1, "Compact Phase 2C manifest count drifted.")
    _require({record.get("path") for record in records} == EXPECTED_EXPORT_FILES - {MANIFEST_NAME}, "Compact Phase 2C manifest inventory drifted.")
    for record in records:
        path = package / str(record["path"])
        _require(path.stat().st_size == int(record["size_bytes"]), f"Compact Phase 2C size drifted for {path.name}.")
        _require(sha256_file(path) == record["sha256"], f"Compact Phase 2C hash drifted for {path.name}.")
    provenance = json.loads((package / "provenance_receipt.json").read_text(encoding="utf-8"))
    _require(provenance.get("source_run_id") == run_receipt["run_id"], "Compact Phase 2C provenance run id drifted.")
    _require(provenance.get("source_generation_commit") == run_receipt["generation_commit"], "Compact Phase 2C provenance commit drifted.")
    _require(provenance.get("independent_run_validation") == run_receipt, "Compact Phase 2C validation receipt drifted.")
    controls = provenance.get("publication_controls")
    _require(isinstance(controls, Mapping), "Compact Phase 2C controls are absent.")
    for field in (
        "employee_level_prediction_comparison_rows_included",
        "employee_level_permutation_rows_included",
        "fold_assignments_included",
        "raw_data_included",
        "fitted_models_included",
        "fairness_certification_allowed",
        "discrimination_claim_allowed",
        "causal_feature_effect_claim_allowed",
        "reconstructability_proves_performance_use",
        "P3_minus_JobRole_labelled_as_P5",
        "deployment_claim_allowed",
    ):
        _require(controls.get(field) is False, f"Compact Phase 2C control {field} drifted.")
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
                _require(
                    not any(token in column for column in normalized),
                    f"Compact {output_name} exposes forbidden column token {token}.",
                )
    _require(row_counts == CSV_ROW_COUNTS, "Compact Phase 2C row counts drifted.")
    _require(provenance.get("row_counts") == row_counts, "Compact Phase 2C provenance row counts drifted.")
    _require(
        "proxy_prediction_change_sample.csv" not in inventory
        and "jobrole_permutation_sample.csv" not in inventory
        and "stage_metadata.json" not in inventory,
        "Compact Phase 2C package contains prohibited source internals.",
    )
    readme = (package / "README.md").read_text(encoding="utf-8")
    for required in (
        "all 2,025 declared group-metric rows",
        "support thresholds 20/30/50",
        "does not select only the largest observed gap",
        "not confirmatory fairness inference",
        "It is not v3 P5",
        "not a fully conditional permutation test",
        "It does not prove that the performance model used department",
        "No table certifies fairness",
        "48,000 employee-level permutation rows",
    ):
        _require(required in readme, f"Compact Phase 2C boundary is absent: {required}.")
    return {
        "status": "passed",
        "source_run_id": run_receipt["run_id"],
        "source_generation_commit": run_receipt["generation_commit"],
        "file_count": len(EXPECTED_EXPORT_FILES),
        "total_size_bytes": sum(
            path.stat().st_size for path in package.iterdir() if path.is_file()
        ),
        "manifest_sha256": sha256_file(package / MANIFEST_NAME),
        "row_counts": row_counts,
        "employee_level_rows_included": False,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SUBGROUP_PROXY_USE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt = (
        validate_subgroup_proxy_use_compact_v3(
            args.output_dir, source_run=args.source_run
        )
        if args.validate_only
        else export_subgroup_proxy_use_compact_v3(args.source_run, args.output_dir)
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "EXPECTED_EXPORT_FILES",
    "V3SubgroupProxyUseCompactExportError",
    "export_subgroup_proxy_use_compact_v3",
    "validate_subgroup_proxy_use_compact_v3",
]
