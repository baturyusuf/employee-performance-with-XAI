"""Create and validate a publication-safe compact Phase 1C evidence package."""

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
from src.experiments.repeated_nested_cv_v3 import (
    ALL_MODEL_NAMES,
    TUNED_MODEL_NAMES,
    summarize_repeated_metrics_v3,
)
from src.governance.repeated_nested_cv_run_validator_v3 import (
    validate_repeated_nested_cv_run_v3,
)


DEFAULT_SOURCE_RUN = Path(
    "reports/major_revision_v3_runs/"
    "phase1c_v3_20260903T215015Z_78649c4/repeated_nested_cv"
)
DEFAULT_OUTPUT = Path(
    "reports/research_log/major_revision_v3/phase1c_repeated_nested_cv"
)
MANIFEST_NAME = "manifest.json"
DIRECT_EXPORTS = {
    "repetition_metrics.csv": "repetition_metrics.csv",
    "variability_summary.csv": "variability_summary.csv",
    "rank_by_repetition.csv": "rank_by_repetition.csv",
    "model_rank_summary.csv": "model_rank_summary.csv",
    "ordering_stability.csv": "ordering_stability.csv",
    "selected_candidate_frequency.csv": "selected_candidate_frequency.csv",
}
EXPECTED_EXPORT_FILES = frozenset(
    {"README.md", "provenance_receipt.json", MANIFEST_NAME, *DIRECT_EXPORTS}
)
FORBIDDEN_PUBLIC_COLUMN_TOKENS = (
    "sample_index",
    "employee",
    "empnumber",
    "y_true",
    "y_pred",
    "prob_class_",
    "sample_key",
)


class V3RepeatedNestedCVCompactExportError(RuntimeError):
    """Raised when the compact repeated-CV export is unsafe or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3RepeatedNestedCVCompactExportError(message)


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


def _assert_frame_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_columns: Sequence[str],
    context: str,
) -> None:
    _require(set(observed.columns) == set(expected.columns), f"{context} schema drifted.")
    columns = list(expected.columns)
    try:
        pd.testing.assert_frame_equal(
            observed.loc[:, columns].sort_values(list(sort_columns)).reset_index(drop=True),
            expected.loc[:, columns].sort_values(list(sort_columns)).reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-14,
        )
    except AssertionError as exc:
        raise V3RepeatedNestedCVCompactExportError(
            f"{context} does not match compact recomputation: {exc}"
        ) from exc


def _markdown_summary(
    variability: pd.DataFrame,
    rank_summary: pd.DataFrame,
    stability: pd.DataFrame,
    *,
    run_id: str,
) -> str:
    values = variability.pivot(index="model_name", columns="metric")
    macro_ranks = rank_summary[rank_summary["metric"] == "macro_f1"].set_index("model_name")
    balanced_ranks = rank_summary[rank_summary["metric"] == "balanced_accuracy"].set_index("model_name")
    qwk_ranks = rank_summary[rank_summary["metric"] == "quadratic_weighted_kappa"].set_index("model_name")
    mae_ranks = rank_summary[rank_summary["metric"] == "ordinal_mae"].set_index("model_name")
    stability_by_metric = stability.set_index("metric")
    display_names = {
        "logistic_regression": "Multinomial logistic regression",
        "random_forest": "Random Forest",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "proportional_odds_logistic": "Proportional-odds logistic",
        "cumulative_threshold_xgboost": "Cumulative-threshold XGBoost",
        "majority_baseline": "Majority baseline",
        "stratified_baseline": "Stratified baseline",
        "ordinal_median_baseline": "Ordinal-median baseline",
    }
    order = values["mean"]["macro_f1"].sort_values(ascending=False).index.tolist()
    lines = [
        "# Phase 1C Repeated Nested-CV — Compact Evidence",
        "",
        f"Source run: `{run_id}`",
        "",
        "This package contains repetition-level and higher-order summaries only. Employee-level OOF predictions, fold assignments, fold metrics, raw data, candidate-search rows, and fitted models are deliberately excluded.",
        "",
        "## Five-repetition priority metrics",
        "",
        "Values are mean ± sample SD across five prespecified 5×5 nested-CV repetitions.",
        "",
        "| Model | Macro-F1 | Balanced accuracy | QWK | Ordinal MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model_name in order:
        lines.append(
            "| "
            + " | ".join(
                [
                    display_names[str(model_name)],
                    f"{values.loc[model_name, ('mean', 'macro_f1')]:.4f} ± {values.loc[model_name, ('sample_sd', 'macro_f1')]:.4f}",
                    f"{values.loc[model_name, ('mean', 'balanced_accuracy')]:.4f} ± {values.loc[model_name, ('sample_sd', 'balanced_accuracy')]:.4f}",
                    f"{values.loc[model_name, ('mean', 'quadratic_weighted_kappa')]:.4f} ± {values.loc[model_name, ('sample_sd', 'quadratic_weighted_kappa')]:.4f}",
                    f"{values.loc[model_name, ('mean', 'ordinal_mae')]:.4f} ± {values.loc[model_name, ('sample_sd', 'ordinal_mae')]:.4f}",
                ]
            )
            + " |"
        )
    xgb = "xgboost"
    lgbm = "lightgbm"
    cumulative = "cumulative_threshold_xgboost"
    forest = "random_forest"
    lines.extend(
        [
            "",
            "## Bounded interpretation",
            "",
            f"- Macro-F1 does not have a single repetition winner: LightGBM wins 3/5 and XGBoost 2/5. Their mean ranks are both {macro_ranks.loc[lgbm, 'mean_rank']:.1f}, while mean macro-F1 is {values.loc[xgb, ('mean', 'macro_f1')]:.4f} for XGBoost and {values.loc[lgbm, ('mean', 'macro_f1')]:.4f} for LightGBM.",
            f"- Cumulative-threshold XGBoost has the highest mean balanced accuracy ({values.loc[cumulative, ('mean', 'balanced_accuracy')]:.4f}) and wins 4/5 repetitions; LightGBM wins the remaining repetition.",
            f"- Random Forest wins all 5/5 repetitions on both QWK (mean {values.loc[forest, ('mean', 'quadratic_weighted_kappa')]:.4f}, SD {values.loc[forest, ('sample_sd', 'quadratic_weighted_kappa')]:.4f}) and ordinal MAE (mean {values.loc[forest, ('mean', 'ordinal_mae')]:.4f}, SD {values.loc[forest, ('sample_sd', 'ordinal_mae')]:.4f}).",
            f"- Mean pairwise rank Spearman correlations are {stability_by_metric.loc['macro_f1', 'mean_pairwise_rank_spearman']:.3f} for macro-F1, {stability_by_metric.loc['balanced_accuracy', 'mean_pairwise_rank_spearman']:.3f} for balanced accuracy, {stability_by_metric.loc['quadratic_weighted_kappa', 'mean_pairwise_rank_spearman']:.3f} for QWK, and {stability_by_metric.loc['ordinal_mae', 'mean_pairwise_rank_spearman']:.3f} for ordinal MAE.",
            "- Model ordering is therefore metric-specific: ordinal-error ordering is stable at the winner, whereas the classification winner varies across repetitions. No universally best model is identified.",
            "- The reported minimum–maximum ranges are empirical repetition ranges, not confidence intervals and not sample-level uncertainty estimates.",
            "- Five repetitions were fixed before result inspection as the bounded-cost option (5,725 planned estimator fits); no seed or repetition was selected after seeing results.",
            "- Every system was refitted in every repetition. Canonical-v2 OOF predictions were not reused for this estimand.",
            "- These remain cross-sectional P3 results under timestamp-unverified feature availability. They do not establish prospective validity, causality, fairness, or deployment readiness.",
            "",
            "## Files",
            "",
            "- `repetition_metrics.csv`: all 16 metrics for nine systems in each repetition.",
            "- `variability_summary.csv`: mean, sample SD, median, minimum, and maximum for four priority metrics.",
            "- `rank_by_repetition.csv`: six tuned-model ranks and winners for each priority metric.",
            "- `model_rank_summary.csv`: descriptive rank and winner-frequency summaries.",
            "- `ordering_stability.csv`: pairwise rank-correlation and winner-stability summaries.",
            "- `selected_candidate_frequency.csv`: fold-level tuning-choice frequencies without outer-test results or employee rows.",
            "- `provenance_receipt.json` and `manifest.json`: source validation, immutable identities, and byte hashes.",
        ]
    )
    _require(int(balanced_ranks.loc[cumulative, "winner_count"]) == 4, "Balanced-accuracy summary drifted.")
    _require(int(qwk_ranks.loc[forest, "winner_count"]) == 5, "QWK summary drifted.")
    _require(int(mae_ranks.loc[forest, "winner_count"]) == 5, "Ordinal-MAE summary drifted.")
    return "\n".join(lines) + "\n"


def export_repeated_nested_cv_compact_v3(
    source_run: Path | str = DEFAULT_SOURCE_RUN,
    output_dir: Path | str = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate a local run and atomically export only compact safe evidence."""

    source = Path(source_run)
    destination = Path(output_dir)
    if destination.exists():
        raise V3RepeatedNestedCVCompactExportError(
            f"Compact destination already exists: {destination.as_posix()}."
        )
    run_receipt = validate_repeated_nested_cv_run_v3(source)
    source_metadata = json.loads((source / "stage_metadata.json").read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for output_name, source_name in DIRECT_EXPORTS.items():
            shutil.copyfile(source / source_name, staging / output_name)
        variability = pd.read_csv(staging / "variability_summary.csv")
        rank_summary = pd.read_csv(staging / "model_rank_summary.csv")
        stability = pd.read_csv(staging / "ordering_stability.csv")
        _write_bytes(
            staging / "README.md",
            _markdown_summary(
                variability,
                rank_summary,
                stability,
                run_id=run_receipt["run_id"],
            ).encode("utf-8"),
        )
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
            "package_kind": "phase1c_repeated_nested_cv_compact_evidence",
            "status": "passed",
            "source_run_id": run_receipt["run_id"],
            "source_generation_commit": run_receipt["generation_commit"],
            "source_scientific_input_sha256": run_receipt["scientific_input_sha256"],
            "source_repeated_contract_sha256": run_receipt["repeated_contract_sha256"],
            "source_file_hashes": source_files,
            "source_validation": run_receipt,
            "design": {
                "repetitions": 5,
                "outer_folds_per_repetition": 5,
                "inner_folds": 5,
                "planned_estimator_fit_calls": source_metadata["planned_estimator_fit_calls"],
                "all_models_refitted_in_every_repetition": True,
                "seed_or_repetition_selected_from_results": False,
            },
            "range_interpretation": "empirical_repetition_range_not_confidence_interval",
            "employee_level_rows_included": False,
            "fold_assignments_included": False,
            "fold_level_metrics_included": False,
            "candidate_search_rows_included": False,
            "raw_data_included": False,
            "fitted_models_included": False,
            "paid_api_calls": 0,
            "network_calls": 0,
        }
        _write_bytes(staging / "provenance_receipt.json", _json_bytes(provenance))
        manifest_records = [
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
            "manifest_kind": "closed_world_phase1c_repeated_nested_cv_compact_evidence",
            "status": "passed",
            "source_run_id": run_receipt["run_id"],
            "file_count_excluding_manifest": len(manifest_records),
            "files": manifest_records,
        }
        _write_bytes(staging / MANIFEST_NAME, _json_bytes(manifest))
        validate_repeated_nested_cv_compact_v3(staging)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            for child in staging.iterdir():
                if child.is_file():
                    child.unlink()
            staging.rmdir()
        raise
    return validate_repeated_nested_cv_compact_v3(destination)


def validate_repeated_nested_cv_compact_v3(
    output_dir: Path | str = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate tracked aggregate evidence without requiring local row-level files."""

    root = Path(output_dir)
    _require(root.is_dir(), f"Compact package is absent: {root.as_posix()}.")
    files = {path.name for path in root.iterdir() if path.is_file()}
    directories = [path.name for path in root.iterdir() if path.is_dir()]
    _require(not directories, f"Compact package has unexpected directories: {directories}.")
    _require(
        files == EXPECTED_EXPORT_FILES,
        "Compact closed-world inventory drifted: "
        f"missing={sorted(EXPECTED_EXPORT_FILES - files)}, "
        f"unexpected={sorted(files - EXPECTED_EXPORT_FILES)}.",
    )
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    _require(manifest.get("status") == "passed", "Compact manifest status drifted.")
    records = manifest.get("files")
    _require(isinstance(records, list) and len(records) == 8, "Compact manifest must contain eight file records.")
    _require(manifest.get("file_count_excluding_manifest") == len(records), "Compact manifest count drifted.")
    _require({record.get("path") for record in records} == files - {MANIFEST_NAME}, "Compact manifest path inventory drifted.")
    for record in records:
        path = root / str(record["path"])
        _require(record.get("sha256") == sha256_file(path), f"Compact hash drifted: {path.name}.")
        _require(record.get("size_bytes") == path.stat().st_size, f"Compact size drifted: {path.name}.")

    repetition_metrics = pd.read_csv(root / "repetition_metrics.csv")
    variability = pd.read_csv(root / "variability_summary.csv")
    ranks = pd.read_csv(root / "rank_by_repetition.csv")
    rank_summary = pd.read_csv(root / "model_rank_summary.csv")
    stability = pd.read_csv(root / "ordering_stability.csv")
    frequency = pd.read_csv(root / "selected_candidate_frequency.csv")
    _require(len(repetition_metrics) == 720, "Compact repetition metric grid must contain 720 rows.")
    _require(set(repetition_metrics["model_name"].unique()) == set(ALL_MODEL_NAMES), "Compact model set drifted.")
    _require(repetition_metrics.groupby(["repetition", "model_name"])["metric"].nunique().eq(16).all(), "Compact repetition metric grid is incomplete.")
    _require(len(variability) == 36, "Compact variability grid must contain 36 rows.")
    _require(len(ranks) == 120, "Compact rank grid must contain 120 rows.")
    _require(len(rank_summary) == 24, "Compact rank-summary grid must contain 24 rows.")
    _require(len(stability) == 4, "Compact ordering-stability grid must contain four rows.")
    _require(len(frequency) == 22, "Compact candidate-frequency grid must contain 22 rows.")
    _require(set(ranks["model_name"].unique()) == set(TUNED_MODEL_NAMES), "Compact tuned-model rank set drifted.")
    expected_variability, expected_ranks, expected_rank_summary, expected_stability = summarize_repeated_metrics_v3(repetition_metrics)
    _assert_frame_equal(variability, expected_variability, sort_columns=("model_name", "metric"), context="Variability summary")
    _assert_frame_equal(ranks, expected_ranks, sort_columns=("metric", "repetition", "rank", "model_name"), context="Ranks by repetition")
    _assert_frame_equal(rank_summary, expected_rank_summary, sort_columns=("metric", "model_name"), context="Model-rank summary")
    _assert_frame_equal(stability, expected_stability, sort_columns=("metric",), context="Ordering stability")
    _require(frequency.groupby("model_name")["selection_count"].sum().eq(25).all(), "Candidate selection counts do not sum to 25 per model.")
    _require(frequency["selection_opportunities"].astype(int).eq(25).all(), "Candidate selection opportunities drifted.")
    _require(
        ((frequency["selection_count"] / 25) - frequency["selection_frequency"]).abs().le(1e-14).all(),
        "Candidate selection frequencies drifted.",
    )
    for csv_path in root.glob("*.csv"):
        columns = [str(column).casefold() for column in pd.read_csv(csv_path, nrows=0).columns]
        forbidden = [column for column in columns if any(token in column for token in FORBIDDEN_PUBLIC_COLUMN_TOKENS)]
        _require(not forbidden, f"Unsafe row-level columns in {csv_path.name}: {forbidden}.")
    provenance = json.loads((root / "provenance_receipt.json").read_text(encoding="utf-8"))
    _require(provenance.get("status") == "passed", "Compact provenance status drifted.")
    _require(provenance.get("source_run_id") == manifest.get("source_run_id"), "Compact source run identity drifted.")
    for field in (
        "employee_level_rows_included",
        "fold_assignments_included",
        "fold_level_metrics_included",
        "candidate_search_rows_included",
        "raw_data_included",
        "fitted_models_included",
    ):
        _require(provenance.get(field) is False, f"Provenance safety field drifted: {field}.")
    _require(provenance.get("paid_api_calls") == 0, "Paid API count drifted.")
    _require(provenance.get("network_calls") == 0, "Network call count drifted.")
    source_hashes = provenance.get("source_file_hashes")
    _require(isinstance(source_hashes, Mapping), "Source file hash inventory is absent.")
    for output_name, source_name in DIRECT_EXPORTS.items():
        _require(source_name in source_hashes, f"Source hash is absent for {source_name}.")
        _require(sha256_file(root / output_name) == source_hashes[source_name]["sha256"], f"Compact/source byte identity drifted for {output_name}.")
    total_bytes = sum(path.stat().st_size for path in root.iterdir() if path.is_file())
    return {
        "status": "passed",
        "source_run_id": manifest["source_run_id"],
        "file_count_including_manifest": len(files),
        "total_bytes": int(total_bytes),
        "repetition_metric_rows": len(repetition_metrics),
        "variability_rows": len(variability),
        "rank_rows": len(ranks),
        "rank_summary_rows": len(rank_summary),
        "ordering_stability_rows": len(stability),
        "candidate_frequency_rows": len(frequency),
        "manifest_sha256": sha256_file(root / MANIFEST_NAME),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    receipt = (
        validate_repeated_nested_cv_compact_v3(args.output)
        if args.validate_only
        else export_repeated_nested_cv_compact_v3(args.source_run, args.output)
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_SOURCE_RUN",
    "EXPECTED_EXPORT_FILES",
    "V3RepeatedNestedCVCompactExportError",
    "export_repeated_nested_cv_compact_v3",
    "validate_repeated_nested_cv_compact_v3",
]
