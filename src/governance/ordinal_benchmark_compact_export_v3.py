"""Create and validate a publication-safe compact Phase 1B result package."""

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
from src.governance.ordinal_benchmark_run_validator_v3 import (
    validate_ordinal_benchmark_run_v3,
)
from src.models.canonical_models import CANONICAL_MODEL_NAMES
from src.models.ordinal_models_v3 import (
    V3_NAIVE_BASELINE_NAMES,
    V3_ORDINAL_MODEL_NAMES,
)


DEFAULT_SOURCE_RUN = Path(
    "reports/major_revision_v3_runs/"
    "phase1b_v3_20260903T130912Z_dc5cb8b/ordinal_benchmark"
)
DEFAULT_OUTPUT = Path(
    "reports/research_log/major_revision_v3/phase1b_ordinal_benchmark"
)
MANIFEST_NAME = "manifest.json"
EXPECTED_EXPORT_FILES = frozenset(
    {
        "README.md",
        "aggregate_metrics.csv",
        "candidate_search_summary.csv",
        "confusion_matrix.csv",
        "extension_fold_metrics.csv",
        "per_class_metrics.csv",
        "provenance_receipt.json",
        "selected_hyperparameters_by_fold.csv",
        MANIFEST_NAME,
    }
)
FORBIDDEN_PUBLIC_COLUMN_TOKENS = (
    "sample_index",
    "employee",
    "empnumber",
    "y_true",
    "y_pred",
    "prob_class_",
)


class V3OrdinalCompactExportError(RuntimeError):
    """Raised when the compact result export is unsafe or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3OrdinalCompactExportError(message)


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


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
    )


def _markdown_summary(aggregate: pd.DataFrame, *, run_id: str) -> str:
    pivot = aggregate.pivot(index="model_name", columns="metric", values="value")
    order = pivot["macro_f1"].sort_values(ascending=False).index.tolist()
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
    lines = [
        "# Phase 1B Ordinal Benchmark — Compact Evidence",
        "",
        f"Source run: `{run_id}`",
        "",
        "This package contains aggregate and fold-level evidence only. Employee-level OOF rows, raw data, and fitted models are deliberately excluded.",
        "",
        "## Nine-system OOF comparison",
        "",
        "| Model | Macro-F1 | Balanced accuracy | QWK | Ordinal MAE | Two-level reversal | RPS | Log loss | Brier | ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in order:
        row = pivot.loc[model_name]
        lines.append(
            "| "
            + " | ".join(
                [
                    display_names[model_name],
                    f"{row['macro_f1']:.4f}",
                    f"{row['balanced_accuracy']:.4f}",
                    f"{row['quadratic_weighted_kappa']:.4f}",
                    f"{row['ordinal_mae']:.4f}",
                    f"{row['two_level_reversal_rate']:.4f}",
                    f"{row['ranked_probability_score']:.4f}",
                    f"{row['nll_log_loss']:.4f}",
                    f"{row['multiclass_brier']:.4f}",
                    f"{row['ece_confidence']:.4f}",
                ]
            )
            + " |"
        )
    cumulative = pivot.loc["cumulative_threshold_xgboost"]
    xgboost = pivot.loc["xgboost"]
    random_forest = pivot.loc["random_forest"]
    lightgbm = pivot.loc["lightgbm"]
    proportional = pivot.loc["proportional_odds_logistic"]
    lines.extend(
        [
            "",
            "## Bounded interpretation",
            "",
            f"- Cumulative-threshold XGBoost has the highest macro-F1 ({cumulative['macro_f1']:.4f}) and balanced accuracy ({cumulative['balanced_accuracy']:.4f}), but its advantage over nominal XGBoost in macro-F1 is only {cumulative['macro_f1'] - xgboost['macro_f1']:.4f} and no interval or significance conclusion is yet attached to that contrast.",
            f"- Random Forest remains strongest on QWK ({random_forest['quadratic_weighted_kappa']:.4f}) and ordinal MAE ({random_forest['ordinal_mae']:.4f}). LightGBM has the lowest RPS ({lightgbm['ranked_probability_score']:.4f}) and multiclass Brier score ({lightgbm['multiclass_brier']:.4f}); nominal XGBoost has the lowest log loss ({xgboost['nll_log_loss']:.4f}).",
            f"- The proportional-odds model does not improve the benchmark: macro-F1 is {proportional['macro_f1']:.4f}, QWK is {proportional['quadratic_weighted_kappa']:.4f}, and ordinal MAE is {proportional['ordinal_mae']:.4f}.",
            f"- Cumulative-threshold XGBoost's raw log loss ({cumulative['nll_log_loss']:.4f}) is materially worse than its classification ranking suggests. Calibration and probability-quality claims must therefore remain metric-specific.",
            "- Majority and ordinal-median baselines coincide because class 3 is both the training majority and ordinal median in every outer fold. Their zero two-level-reversal rate is achieved by always predicting the middle class and must not be treated as overall superiority.",
            "- These are cross-sectional P3 results under timestamp-unverified availability assumptions. They do not establish prospective validity, causality, fairness, deployment readiness, or a universally best model.",
            "- Four nominal OOF prediction sets are immutable canonical-v2 evidence reused without refitting or relabelling. The two ordinal models and three naive baselines are newly fitted on the exact same persisted folds.",
            "",
            "## Files",
            "",
            "- `aggregate_metrics.csv`: full-OOF values for 16 metrics and all nine systems.",
            "- `per_class_metrics.csv`: precision, recall, F1, and support for each class/system.",
            "- `confusion_matrix.csv`: complete ordered 3×3 confusion grid for every system.",
            "- `extension_fold_metrics.csv`: five newly fitted systems across ten outer folds.",
            "- `selected_hyperparameters_by_fold.csv`: selection records without employee rows.",
            "- `candidate_search_summary.csv`: candidate-level cross-fold selection summary.",
            "- `provenance_receipt.json` and `manifest.json`: immutable identities and file hashes.",
        ]
    )
    return "\n".join(lines) + "\n"


def export_ordinal_benchmark_compact_v3(
    source_run: Path | str = DEFAULT_SOURCE_RUN,
    output_dir: Path | str = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate a local run and atomically export only compact safe evidence."""

    source = Path(source_run)
    destination = Path(output_dir)
    if destination.exists():
        raise V3OrdinalCompactExportError(
            f"Compact destination already exists: {destination.as_posix()}."
        )
    run_receipt = validate_ordinal_benchmark_run_v3(source)
    source_metadata = json.loads(
        (source / "stage_metadata.json").read_text(encoding="utf-8")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        direct_exports = {
            "aggregate_metrics.csv": "aggregate_metrics.csv",
            "per_class_metrics.csv": "per_class_metrics.csv",
            "confusion_matrix.csv": "confusion_matrix.csv",
            "extension_fold_metrics.csv": "extension_fold_metrics.csv",
            "selected_hyperparameters_by_fold.csv": "selected_hyperparameters.csv",
        }
        for output_name, source_name in direct_exports.items():
            shutil.copyfile(source / source_name, staging / output_name)

        candidates = pd.read_csv(source / "candidate_search_results.csv")
        candidate_summary = (
            candidates.groupby(
                ["model", "candidate_index", "parameters_json"],
                as_index=False,
                dropna=False,
            )
            .agg(
                outer_fold_count=("outer_fold", "nunique"),
                selected_outer_fold_count=("selected_by_protocol", "sum"),
                inner_macro_f1_mean_across_outer_folds=("inner_macro_f1_mean", "mean"),
                inner_macro_f1_min_across_outer_folds=("inner_macro_f1_mean", "min"),
                inner_macro_f1_max_across_outer_folds=("inner_macro_f1_mean", "max"),
                inner_qwk_mean_across_outer_folds=("inner_qwk_mean", "mean"),
                inner_qwk_min_across_outer_folds=("inner_qwk_mean", "min"),
                inner_qwk_max_across_outer_folds=("inner_qwk_mean", "max"),
            )
            .sort_values(["model", "candidate_index"])
            .reset_index(drop=True)
        )
        _write_csv(staging / "candidate_search_summary.csv", candidate_summary)
        aggregate = pd.read_csv(staging / "aggregate_metrics.csv")
        _write_bytes(
            staging / "README.md",
            _markdown_summary(aggregate, run_id=run_receipt["run_id"]).encode("utf-8"),
        )

        source_files = {
            path.name: {"sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
            for path in sorted(source.iterdir())
            if path.is_file()
        }
        provenance = {
            "schema_version": 1,
            "package_kind": "phase1b_ordinal_benchmark_compact_evidence",
            "status": "passed",
            "source_run_id": run_receipt["run_id"],
            "source_generation_commit": run_receipt["generation_commit"],
            "source_scientific_input_sha256": run_receipt["scientific_input_sha256"],
            "source_benchmark_contract_sha256": source_metadata[
                "benchmark_contract_sha256"
            ],
            "source_fold_contract_hash": run_receipt["fold_contract_hash"],
            "source_file_hashes": source_files,
            "source_validation": run_receipt,
            "nominal_evidence_rule": "canonical_v2_reused_without_refit_or_relabelling",
            "new_fit_models": [*V3_ORDINAL_MODEL_NAMES, *V3_NAIVE_BASELINE_NAMES],
            "employee_level_rows_included": False,
            "raw_data_included": False,
            "fitted_models_included": False,
            "paid_api_calls": 0,
            "network_calls": 0,
        }
        _write_bytes(staging / "provenance_receipt.json", _json_bytes(provenance))

        manifest_records = []
        for path in sorted(staging.iterdir()):
            if path.is_file():
                manifest_records.append(
                    {
                        "path": path.name,
                        "sha256": sha256_file(path),
                        "size_bytes": int(path.stat().st_size),
                    }
                )
        manifest = {
            "schema_version": 1,
            "manifest_kind": "closed_world_phase1b_ordinal_benchmark_compact_evidence",
            "status": "passed",
            "source_run_id": run_receipt["run_id"],
            "file_count_excluding_manifest": len(manifest_records),
            "files": manifest_records,
        }
        _write_bytes(staging / MANIFEST_NAME, _json_bytes(manifest))
        validate_ordinal_benchmark_compact_v3(staging)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            for child in staging.iterdir():
                if child.is_file():
                    child.unlink()
            staging.rmdir()
        raise
    return validate_ordinal_benchmark_compact_v3(destination)


def validate_ordinal_benchmark_compact_v3(
    output_dir: Path | str = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate the tracked compact package without requiring local row-level sources."""

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
    _require(isinstance(records, list), "Compact manifest files must be a list.")
    _require(len(records) == 8, "Compact manifest must contain eight file records.")
    _require(
        manifest.get("file_count_excluding_manifest") == len(records),
        "Compact manifest count drifted.",
    )
    _require(
        {record.get("path") for record in records} == files - {MANIFEST_NAME},
        "Compact manifest path inventory drifted.",
    )
    for record in records:
        path = root / str(record["path"])
        _require(record.get("sha256") == sha256_file(path), f"Compact hash drifted: {path.name}.")
        _require(record.get("size_bytes") == path.stat().st_size, f"Compact size drifted: {path.name}.")

    aggregate = pd.read_csv(root / "aggregate_metrics.csv")
    per_class = pd.read_csv(root / "per_class_metrics.csv")
    confusion = pd.read_csv(root / "confusion_matrix.csv")
    fold_metrics = pd.read_csv(root / "extension_fold_metrics.csv")
    selected = pd.read_csv(root / "selected_hyperparameters_by_fold.csv")
    candidate_summary = pd.read_csv(root / "candidate_search_summary.csv")
    expected_models = {*CANONICAL_MODEL_NAMES, *V3_ORDINAL_MODEL_NAMES, *V3_NAIVE_BASELINE_NAMES}
    _require(len(aggregate) == 144, "Compact aggregate metric grid must contain 144 rows.")
    _require(set(aggregate["model_name"].unique()) == expected_models, "Compact model set drifted.")
    _require(aggregate.groupby("model_name")["metric"].nunique().eq(16).all(), "Metric grid is incomplete.")
    _require(len(per_class) == 27, "Compact per-class grid must contain 27 rows.")
    _require(len(confusion) == 81, "Compact confusion grid must contain 81 rows.")
    _require(len(fold_metrics) == 50, "Compact fold-metric grid must contain 50 rows.")
    _require(len(selected) == 50, "Compact selection grid must contain 50 rows.")
    _require(len(candidate_summary) == 14, "Compact candidate summary must contain 14 rows.")
    for csv_path in root.glob("*.csv"):
        columns = [str(column).casefold() for column in pd.read_csv(csv_path, nrows=0).columns]
        forbidden = [
            column
            for column in columns
            if any(token in column for token in FORBIDDEN_PUBLIC_COLUMN_TOKENS)
        ]
        _require(not forbidden, f"Unsafe row-level columns in {csv_path.name}: {forbidden}.")
    provenance = json.loads((root / "provenance_receipt.json").read_text(encoding="utf-8"))
    for field in (
        "employee_level_rows_included",
        "raw_data_included",
        "fitted_models_included",
    ):
        _require(provenance.get(field) is False, f"Provenance safety field drifted: {field}.")
    _require(provenance.get("paid_api_calls") == 0, "Paid API count drifted.")
    _require(provenance.get("network_calls") == 0, "Network call count drifted.")
    total_bytes = sum(path.stat().st_size for path in root.iterdir() if path.is_file())
    return {
        "status": "passed",
        "source_run_id": manifest["source_run_id"],
        "file_count_including_manifest": len(files),
        "total_bytes": int(total_bytes),
        "aggregate_metric_rows": len(aggregate),
        "per_class_rows": len(per_class),
        "confusion_rows": len(confusion),
        "fold_metric_rows": len(fold_metrics),
        "selection_rows": len(selected),
        "candidate_summary_rows": len(candidate_summary),
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
        validate_ordinal_benchmark_compact_v3(args.output)
        if args.validate_only
        else export_ordinal_benchmark_compact_v3(args.source_run, args.output)
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_SOURCE_RUN",
    "EXPECTED_EXPORT_FILES",
    "V3OrdinalCompactExportError",
    "export_ordinal_benchmark_compact_v3",
    "validate_ordinal_benchmark_compact_v3",
]
