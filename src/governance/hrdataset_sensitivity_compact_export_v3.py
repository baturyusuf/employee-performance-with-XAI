"""Create and validate a publication-safe compact Phase 3A evidence package."""

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
from src.governance.hrdataset_sensitivity_run_validator_v3 import (
    DEFAULT_HRDATASET_SENSITIVITY_RUN,
    validate_hrdataset_sensitivity_run_v3,
)
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_OUTPUT = Path(
    "reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity"
)
MANIFEST_NAME = "manifest.json"
DIRECT_EXPORTS = {
    "baseline_comparisons.csv": "baseline_comparisons.csv",
    "canonical_v2_confusion_matrix.csv": "canonical_v2_confusion_matrix.csv",
    "canonical_v2_per_class_metrics.csv": "canonical_v2_per_class_metrics.csv",
    "confusion_matrices.csv": "confusion_matrices.csv",
    "cv_design_sensitivity.csv": "cv_design_sensitivity.csv",
    "per_class_metrics.csv": "per_class_metrics.csv",
    "protocol_comparison.csv": "protocol_comparison.csv",
    "repetition_metrics.csv": "repetition_metrics.csv",
    "target_mapping_support.csv": "target_mapping_support.csv",
    "variability_summary.csv": "variability_summary.csv",
}
DERIVED_EXPORT = "selected_candidate_frequency.csv"
EXPECTED_EXPORT_FILES = frozenset(
    {"README.md", "provenance_receipt.json", MANIFEST_NAME, DERIVED_EXPORT, *DIRECT_EXPORTS}
)
CSV_ROW_COUNTS = {
    "baseline_comparisons.csv": 210,
    "canonical_v2_confusion_matrix.csv": 18,
    "canonical_v2_per_class_metrics.csv": 6,
    "confusion_matrices.csv": 625,
    "cv_design_sensitivity.csv": 14,
    "per_class_metrics.csv": 175,
    "protocol_comparison.csv": 8,
    "repetition_metrics.csv": 500,
    "selected_candidate_frequency.csv": 16,
    "target_mapping_support.csv": 10,
    "variability_summary.csv": 100,
}
FORBIDDEN_PUBLIC_COLUMN_TOKENS = (
    "sample_index",
    "sample_key",
    "employee",
    "empnumber",
    "outer_fold",
    "y_true",
    "y_pred",
    "prob_class_",
    "coefficient",
    "intercept",
)
PROHIBITED_SOURCE_EXPORTS = frozenset(
    {
        "calibration_training_oof.csv",
        "calibrator_parameters.csv",
        "candidate_search_results.csv",
        "fold_contracts.json",
        "fold_metrics.csv",
        "oof_predictions.csv",
        "selected_hyperparameters.csv",
        "stage_metadata.json",
    }
)


class HRDatasetSensitivityCompactExportV3Error(RuntimeError):
    """Raised when a compact Phase 3A export is unsafe or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HRDatasetSensitivityCompactExportV3Error(message)


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


def _selected_candidate_frequency(source: Path) -> pd.DataFrame:
    selected = pd.read_csv(
        source / "selected_hyperparameters.csv", float_precision="round_trip"
    )
    grid = load_config(PROJECT_ROOT / "configs/model_grid.yaml")
    candidates = grid["model_benchmark"]["models"]["xgboost"]["candidates"]
    rows: list[dict[str, Any]] = []
    for formulation_id in ("primary_three_class", "raw_order_four_class"):
        scoped = selected.loc[selected["formulation_id"].astype(str) == formulation_id]
        _require(len(scoped) == 25, f"Selection support drifted for {formulation_id}.")
        counts = scoped["selected_candidate_index"].astype(int).value_counts().to_dict()
        for candidate_index, parameters in enumerate(candidates):
            selected_count = int(counts.get(candidate_index, 0))
            rows.append(
                {
                    "formulation_id": formulation_id,
                    "candidate_index": candidate_index,
                    "candidate_parameters_json": json.dumps(
                        dict(parameters), sort_keys=True, separators=(",", ":")
                    ),
                    "selected_count": selected_count,
                    "total_outer_selections": 25,
                    "selection_fraction": selected_count / 25.0,
                    "interpretation": "descriptive_selection_frequency_not_model_probability",
                }
            )
    return pd.DataFrame(rows)


def _mean_metric(
    variability: pd.DataFrame,
    formulation_id: str,
    system: str,
    metric: str,
) -> float:
    scoped = variability.loc[
        (variability["formulation_id"].astype(str) == formulation_id)
        & (variability["system"].astype(str) == system)
        & (variability["metric"].astype(str) == metric)
    ]
    _require(len(scoped) == 1, f"Summary metric is absent: {formulation_id}/{system}/{metric}.")
    return float(scoped["mean"].iloc[0])


def _markdown_summary(
    variability: pd.DataFrame,
    cv_design: pd.DataFrame,
    *,
    run_id: str,
) -> str:
    def values(formulation_id: str, system: str) -> list[float]:
        return [
            _mean_metric(variability, formulation_id, system, metric)
            for metric in (
                "macro_f1",
                "quadratic_weighted_kappa",
                "ordinal_mae",
                "nll_log_loss",
                "multiclass_brier",
                "ece_confidence",
                "ranked_probability_score",
            )
        ]

    primary_raw = values("primary_three_class", "xgboost_raw")
    primary_sigmoid = values("primary_three_class", "xgboost_sigmoid")
    alternative_raw = values("raw_order_four_class", "xgboost_raw")
    alternative_sigmoid = values("raw_order_four_class", "xgboost_sigmoid")
    inside_count = int(cv_design["ten_fold_inside_repeated_range"].astype(bool).sum())
    return "\n".join(
        [
            "# Phase 3A HRDataset_v14 Sensitivity — Compact Evidence",
            "",
            f"Source run: `{run_id}`",
            "",
            "This package reports an independent, cross-dataset protocol replication on HRDataset_v14. It is not external validation of a locked INX model: the feature space, target semantics, fitted parameters, and sample population differ, and no INX model was transported.",
            "",
            "## Prespecified design",
            "",
            "Both target formulations use the same seven-feature conservative-primary policy, five fixed repetitions of 5-fold outer by 5-fold inner nested cross-validation, eight-candidate XGBoost selection by inner macro-F1 with QWK tie-breaking, outer-training-only cross-fitted sigmoid calibration, and three label-only naive baselines. Each of the 311 records appears exactly once in outer-test predictions for every system and repetition.",
            "",
            "The retained three-class mapping combines PIP and Needs Improvement (31/243/37 across labels 2/3/4). The supplementary raw-order four-class mapping keeps PIP and Needs Improvement separate (13/18/243/37 across labels 1/2/3/4). These are different estimands; metric differences between them are deliberately not computed and target equivalence is not claimed.",
            "",
            "## Five-repetition descriptive means",
            "",
            "| Target formulation | System | Macro-F1 | QWK | Ordinal MAE | Log loss | Brier | Confidence ECE | RPS |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| Retained three-class | XGBoost raw | {primary_raw[0]:.4f} | {primary_raw[1]:.4f} | {primary_raw[2]:.4f} | {primary_raw[3]:.4f} | {primary_raw[4]:.4f} | {primary_raw[5]:.4f} | {primary_raw[6]:.4f} |",
            f"| Retained three-class | XGBoost sigmoid | {primary_sigmoid[0]:.4f} | {primary_sigmoid[1]:.4f} | {primary_sigmoid[2]:.4f} | {primary_sigmoid[3]:.4f} | {primary_sigmoid[4]:.4f} | {primary_sigmoid[5]:.4f} | {primary_sigmoid[6]:.4f} |",
            f"| Raw-order four-class | XGBoost raw | {alternative_raw[0]:.4f} | {alternative_raw[1]:.4f} | {alternative_raw[2]:.4f} | {alternative_raw[3]:.4f} | {alternative_raw[4]:.4f} | {alternative_raw[5]:.4f} | {alternative_raw[6]:.4f} |",
            f"| Raw-order four-class | XGBoost sigmoid | {alternative_sigmoid[0]:.4f} | {alternative_sigmoid[1]:.4f} | {alternative_sigmoid[2]:.4f} | {alternative_sigmoid[3]:.4f} | {alternative_sigmoid[4]:.4f} | {alternative_sigmoid[5]:.4f} | {alternative_sigmoid[6]:.4f} |",
            "",
            "Calibration effects are metric-specific. In both formulations sigmoid improves QWK, ordinal MAE, log loss, Brier, confidence ECE, and RPS but lowers macro-F1. The method was predeclared, not selected from outer-test results.",
            "",
            "## CV-design sensitivity and baselines",
            "",
            f"For the retained mapping, {inside_count} of 14 canonical-v2 10×5 point estimates lie inside the observed five-repetition 5×5 ranges. Raw macro-F1 and sigmoid Brier/RPS fall outside those ranges. These are descriptive design comparisons, not equivalence tests or confidence intervals; the evidence does not authorize an unqualified claim of robustness to CV design.",
            "",
            "The baseline table retains every repetition-level comparison against majority, stratified, and ordinal-median predictors. Per-class precision/recall/F1/support and full zero-retaining confusion grids are included because aggregate scores can hide the rare PIP and Needs Improvement classes.",
            "",
            "## Interpretation boundaries",
            "",
            "- Repetition ranges and sample standard deviations describe training/split variability under five prespecified seeds; they are not confidence intervals.",
            "- The dataset is small and class-imbalanced, especially under the four-class mapping. Results are conditional on this public table and its unresolved source-authenticity/licence review.",
            "- No target equivalence, locked-model transport, external-validation, causal, fairness-certification, autonomous HR-decision, or deployment-readiness claim is allowed.",
            "- The row-level OOF predictions, calibration-training rows, fold assignments, candidate scores, selected-fold records, calibrator parameters, raw data, and fitted objects remain local and are excluded from Git.",
            "",
            "## Files",
            "",
            "- `target_mapping_support.csv`: observed support and rationale for both mappings.",
            "- `repetition_metrics.csv` and `variability_summary.csv`: complete repetition-level metrics and five-repetition summaries.",
            "- `per_class_metrics.csv` and `confusion_matrices.csv`: class-specific results and complete confusion grids.",
            "- `baseline_comparisons.csv`: matched-repetition raw-XGBoost versus naive-baseline comparisons.",
            "- `cv_design_sensitivity.csv`: retained-mapping 10×5 versus repeated-5×5 descriptive comparison.",
            "- `canonical_v2_per_class_metrics.csv` and `canonical_v2_confusion_matrix.csv`: independently recomputed canonical-v2 class evidence.",
            "- `selected_candidate_frequency.csv`: aggregate selection counts, including zero-count candidates.",
            "- `protocol_comparison.csv`: explicit INX/HRDataset implementation and semantic boundaries.",
            "- `provenance_receipt.json` and `manifest.json`: independent validation, exclusions, lineage, and byte hashes.",
        ]
    ) + "\n"


def export_hrdataset_sensitivity_compact_v3(
    source_run: Path | str = DEFAULT_HRDATASET_SENSITIVITY_RUN,
    output_dir: Path | str = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate the full run and atomically export only aggregate evidence."""

    source = Path(source_run)
    destination = Path(output_dir)
    _require(not destination.exists(), f"Compact destination already exists: {destination.as_posix()}.")
    run_receipt = validate_hrdataset_sensitivity_run_v3(source)
    source_metadata = json.loads((source / "stage_metadata.json").read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for output_name, source_name in DIRECT_EXPORTS.items():
            shutil.copyfile(source / source_name, staging / output_name)
        frequency = _selected_candidate_frequency(source)
        frequency.to_csv(staging / DERIVED_EXPORT, index=False)
        readme = _markdown_summary(
            pd.read_csv(staging / "variability_summary.csv", float_precision="round_trip"),
            pd.read_csv(staging / "cv_design_sensitivity.csv", float_precision="round_trip"),
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
            "package_kind": "phase3a_hrdataset_sensitivity_compact_evidence",
            "source_run": source.as_posix(),
            "source_run_id": run_receipt["run_id"],
            "source_generation_commit": run_receipt["generation_commit"],
            "contract_sha256": run_receipt["contract_sha256"],
            "scientific_input_sha256": run_receipt["scientific_input_sha256"],
            "source_created_at_utc": source_metadata["created_at_utc"],
            "independent_run_validation": run_receipt,
            "source_files": source_files,
            "directly_included_source_files": sorted(DIRECT_EXPORTS.values()),
            "derived_source_files": ["selected_hyperparameters.csv"],
            "excluded_source_files": sorted(set(source_files) - set(DIRECT_EXPORTS.values())),
            "publication_controls": {
                "employee_level_oof_included": False,
                "calibration_training_oof_included": False,
                "fold_assignments_or_fold_metrics_included": False,
                "candidate_search_or_fold_selections_included": False,
                "calibrator_parameters_included": False,
                "raw_data_included": False,
                "fitted_models_or_calibrators_included": False,
                "cross_formulation_metric_difference_computed": False,
                "target_equivalence_claim_allowed": False,
                "external_validation_claim_allowed": False,
                "locked_model_transport_claim_allowed": False,
                "cv_design_robustness_claim_allowed_without_qualification": False,
                "fairness_certification_allowed": False,
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
            "package_kind": "phase3a_hrdataset_sensitivity_compact_evidence",
            "source_run_id": run_receipt["run_id"],
            "file_count_excluding_manifest": len(records),
            "files": records,
        }
        _write_bytes(staging / MANIFEST_NAME, _json_bytes(manifest))
        _require(
            {path.name for path in staging.iterdir() if path.is_file()} == EXPECTED_EXPORT_FILES,
            "Compact Phase 3A staging inventory drifted.",
        )
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            for child in sorted(staging.iterdir(), reverse=True):
                if child.is_file():
                    child.unlink()
            staging.rmdir()
        raise
    return validate_hrdataset_sensitivity_compact_v3(destination, source_run=source)


def validate_hrdataset_sensitivity_compact_v3(
    package_dir: Path | str = DEFAULT_OUTPUT,
    *,
    source_run: Path | str = DEFAULT_HRDATASET_SENSITIVITY_RUN,
) -> dict[str, Any]:
    """Validate compact contents, source equivalence, exclusions, and hashes."""

    package = Path(package_dir)
    source = Path(source_run)
    _require(package.is_dir(), f"Compact Phase 3A package is absent: {package.as_posix()}.")
    inventory = {path.name for path in package.iterdir() if path.is_file()}
    _require(
        inventory == EXPECTED_EXPORT_FILES,
        f"Compact Phase 3A closed-world inventory drifted: {sorted(inventory ^ EXPECTED_EXPORT_FILES)}.",
    )
    _require(not any(path.is_dir() for path in package.iterdir()), "Compact Phase 3A package contains a directory.")
    _require(not (inventory & PROHIBITED_SOURCE_EXPORTS), "Compact Phase 3A package contains prohibited source evidence.")
    manifest = json.loads((package / MANIFEST_NAME).read_text(encoding="utf-8"))
    records = manifest.get("files")
    _require(isinstance(records, list), "Compact Phase 3A manifest records are absent.")
    _require(manifest.get("file_count_excluding_manifest") == len(EXPECTED_EXPORT_FILES) - 1, "Compact Phase 3A manifest count drifted.")
    _require({record.get("path") for record in records} == EXPECTED_EXPORT_FILES - {MANIFEST_NAME}, "Compact Phase 3A manifest inventory drifted.")
    for record in records:
        path = package / str(record["path"])
        _require(path.stat().st_size == int(record["size_bytes"]), f"Compact Phase 3A size drifted for {path.name}.")
        _require(sha256_file(path) == record["sha256"], f"Compact Phase 3A hash drifted for {path.name}.")
    run_receipt = validate_hrdataset_sensitivity_run_v3(source)
    _require(manifest.get("source_run_id") == run_receipt["run_id"], "Compact Phase 3A manifest run id drifted.")
    provenance = json.loads((package / "provenance_receipt.json").read_text(encoding="utf-8"))
    _require(provenance.get("source_run_id") == run_receipt["run_id"], "Compact Phase 3A provenance run id drifted.")
    _require(provenance.get("source_generation_commit") == run_receipt["generation_commit"], "Compact Phase 3A provenance commit drifted.")
    _require(provenance.get("independent_run_validation") == run_receipt, "Compact Phase 3A validation receipt drifted.")
    controls = provenance.get("publication_controls")
    _require(isinstance(controls, Mapping), "Compact Phase 3A publication controls are absent.")
    for field in (
        "employee_level_oof_included",
        "calibration_training_oof_included",
        "fold_assignments_or_fold_metrics_included",
        "candidate_search_or_fold_selections_included",
        "calibrator_parameters_included",
        "raw_data_included",
        "fitted_models_or_calibrators_included",
        "cross_formulation_metric_difference_computed",
        "target_equivalence_claim_allowed",
        "external_validation_claim_allowed",
        "locked_model_transport_claim_allowed",
        "cv_design_robustness_claim_allowed_without_qualification",
        "fairness_certification_allowed",
        "deployment_claim_allowed",
    ):
        _require(controls.get(field) is False, f"Compact Phase 3A control {field} drifted.")
    row_counts: dict[str, int] = {}
    for output_name, source_name in DIRECT_EXPORTS.items():
        compact_path = package / output_name
        source_path = source / source_name
        _require(sha256_file(compact_path) == sha256_file(source_path), f"Compact/source bytes drifted for {output_name}.")
        frame = pd.read_csv(compact_path)
        row_counts[output_name] = len(frame)
        normalized = [str(column).casefold() for column in frame.columns]
        for token in FORBIDDEN_PUBLIC_COLUMN_TOKENS:
            _require(not any(token in column for column in normalized), f"Compact Phase 3A column exposes prohibited token {token}: {output_name}.")
    expected_frequency = _selected_candidate_frequency(source)
    observed_frequency = pd.read_csv(package / DERIVED_EXPORT, float_precision="round_trip")
    try:
        pd.testing.assert_frame_equal(
            observed_frequency,
            expected_frequency,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as exc:
        raise HRDatasetSensitivityCompactExportV3Error(
            f"Compact selected-candidate frequency drifted: {exc}"
        ) from exc
    row_counts[DERIVED_EXPORT] = len(observed_frequency)
    _require(row_counts == CSV_ROW_COUNTS, f"Compact Phase 3A row counts drifted: {row_counts}.")
    _require(provenance.get("row_counts") == CSV_ROW_COUNTS, "Compact Phase 3A provenance row counts drifted.")
    excluded = set(provenance.get("excluded_source_files", ()))
    _require(PROHIBITED_SOURCE_EXPORTS.issubset(excluded), "Compact Phase 3A exclusions are incomplete.")
    readme = (package / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "cross-dataset protocol replication",
        "not external validation",
        "no INX model was transported",
        "different estimands",
        "not equivalence tests or confidence intervals",
        "does not authorize an unqualified claim of robustness to CV design",
        "target equivalence",
        "deployment-readiness",
    ):
        _require(phrase.casefold() in readme.casefold(), f"Compact Phase 3A README boundary is absent: {phrase}.")
    return {
        "status": "passed",
        "package_dir": package.as_posix(),
        "source_run_id": run_receipt["run_id"],
        "source_generation_commit": run_receipt["generation_commit"],
        "file_count": len(EXPECTED_EXPORT_FILES),
        "total_size_bytes": sum(path.stat().st_size for path in package.iterdir() if path.is_file()),
        "manifest_sha256": sha256_file(package / MANIFEST_NAME),
        "row_counts": row_counts,
        "employee_level_rows_included": False,
        "calibration_training_rows_included": False,
        "fold_level_rows_included": False,
        "target_equivalence_claim_allowed": False,
        "external_validation_claim_allowed": False,
        "locked_model_transport_claim_allowed": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_HRDATASET_SENSITIVITY_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.validate_only:
        receipt = validate_hrdataset_sensitivity_compact_v3(
            args.output, source_run=args.source_run
        )
    else:
        receipt = export_hrdataset_sensitivity_compact_v3(
            args.source_run, args.output
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CSV_ROW_COUNTS",
    "DEFAULT_OUTPUT",
    "DIRECT_EXPORTS",
    "EXPECTED_EXPORT_FILES",
    "HRDatasetSensitivityCompactExportV3Error",
    "export_hrdataset_sensitivity_compact_v3",
    "validate_hrdataset_sensitivity_compact_v3",
]
