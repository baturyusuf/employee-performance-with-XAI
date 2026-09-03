"""Independent closed-world validator for a completed v3 ordinal benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.canonical_loader import sha256_file
from src.experiments.ordinal_benchmark_v3 import (
    DEFAULT_CANONICAL_V2_ROOT,
    EXTENSION_MODEL_NAMES,
    summarize_combined_oof_v3,
)
from src.models.canonical_models import CANONICAL_MODEL_NAMES
from src.models.ordinal_models_v3 import (
    V3_NAIVE_BASELINE_NAMES,
    V3_ORDINAL_MODEL_NAMES,
)
from src.utils.config_loader import PROJECT_ROOT


EXPECTED_FILES = frozenset(
    {
        "aggregate_metrics.csv",
        "candidate_search_results.csv",
        "combined_oof_predictions.csv",
        "confusion_matrix.csv",
        "extension_fold_metrics.csv",
        "extension_oof_predictions.csv",
        "per_class_metrics.csv",
        "selected_hyperparameters.csv",
        "stage_metadata.json",
    }
)
OUTPUT_HASH_FILES: Mapping[str, str] = {
    "aggregate_metrics": "aggregate_metrics.csv",
    "candidate_search_results": "candidate_search_results.csv",
    "combined_oof_predictions": "combined_oof_predictions.csv",
    "confusion_matrix": "confusion_matrix.csv",
    "extension_oof_predictions": "extension_oof_predictions.csv",
    "fold_metrics": "extension_fold_metrics.csv",
    "per_class_metrics": "per_class_metrics.csv",
    "selected_hyperparameters": "selected_hyperparameters.csv",
}
EXPECTED_MODELS = frozenset({*CANONICAL_MODEL_NAMES, *EXTENSION_MODEL_NAMES})


class V3OrdinalRunValidationError(RuntimeError):
    """Raised when a persisted v3 benchmark run is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3OrdinalRunValidationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V3OrdinalRunValidationError(f"Could not read {path.as_posix()}: {exc}") from exc
    _require(isinstance(payload, dict), f"{path.name} must contain a JSON object.")
    return payload


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_blob(commit: str, relative_path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise V3OrdinalRunValidationError(
            f"Could not resolve generation blob {commit}:{relative_path}: {exc}"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise V3OrdinalRunValidationError(f"Could not parse {path.name}: {exc}") from exc


def _assert_frame_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_columns: Sequence[str],
    context: str,
) -> None:
    observed_sorted = observed.sort_values(list(sort_columns)).reset_index(drop=True)
    expected_sorted = expected.sort_values(list(sort_columns)).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            observed_sorted,
            expected_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-14,
        )
    except AssertionError as exc:
        raise V3OrdinalRunValidationError(f"{context} does not match OOF recomputation: {exc}") from exc


def validate_ordinal_benchmark_run_v3(
    run_dir: Path | str,
    *,
    canonical_v2_root: Path | str = DEFAULT_CANONICAL_V2_ROOT,
) -> dict[str, Any]:
    """Rehash and recompute a complete nine-system benchmark package."""

    root = Path(run_dir)
    _require(root.is_dir(), f"Run directory does not exist: {root.as_posix()}.")
    physical_files = {path.name for path in root.iterdir() if path.is_file()}
    physical_directories = [path.name for path in root.iterdir() if path.is_dir()]
    _require(not physical_directories, f"Unexpected run subdirectories: {physical_directories}.")
    _require(
        physical_files == EXPECTED_FILES,
        "Run closed-world inventory drifted: "
        f"missing={sorted(EXPECTED_FILES - physical_files)}, "
        f"unexpected={sorted(physical_files - EXPECTED_FILES)}.",
    )
    metadata = _load_json(root / "stage_metadata.json")
    expected_metadata = {
        "schema_version": 1,
        "stage": "ordinal_benchmark_v3",
        "status": "complete",
        "evidence_status": "complete_exactly_once_oof",
        "feature_policy": "P3",
        "feature_count": 20,
        "ordered_labels": [2, 3, 4],
        "model_count": 9,
        "sample_count_per_model": 1200,
        "outer_folds": 10,
        "inner_folds": 5,
        "outer_test_used_for_selection": False,
        "paid_api_calls": 0,
        "network_calls": 0,
        "employee_level_outputs_publication_authorized": False,
    }
    for key, expected in expected_metadata.items():
        _require(metadata.get(key) == expected, f"Metadata {key} drifted.")
    _require(
        metadata.get("nominal_models_reused") == list(CANONICAL_MODEL_NAMES),
        "Nominal-model provenance drifted.",
    )
    _require(
        metadata.get("new_ordinal_models_fitted") == list(V3_ORDINAL_MODEL_NAMES),
        "Ordinal-model provenance drifted.",
    )
    _require(
        metadata.get("new_naive_baselines_fitted") == list(V3_NAIVE_BASELINE_NAMES),
        "Naive-baseline provenance drifted.",
    )

    output_hashes = metadata.get("output_hashes")
    _require(isinstance(output_hashes, Mapping), "Metadata output_hashes must be an object.")
    _require(set(output_hashes) == set(OUTPUT_HASH_FILES), "Output-hash inventory drifted.")
    for key, filename in OUTPUT_HASH_FILES.items():
        _require(
            output_hashes[key] == sha256_file(root / filename),
            f"Output byte hash drifted for {filename}.",
        )

    scientific_inputs = metadata.get("scientific_inputs")
    _require(isinstance(scientific_inputs, Mapping), "scientific_inputs must be an object.")
    _require(
        metadata.get("scientific_input_sha256") == _canonical_json_sha256(scientific_inputs),
        "Scientific-input composite hash drifted.",
    )
    _require(
        metadata.get("benchmark_contract_sha256")
        == scientific_inputs.get("benchmark_contract_sha256"),
        "Benchmark-contract identity is inconsistent.",
    )
    _require(
        metadata.get("source_tree_hash") == scientific_inputs.get("source_tree_hash"),
        "Source-tree receipt is inconsistent.",
    )

    git_identity = metadata.get("git_identity")
    _require(isinstance(git_identity, Mapping), "Git identity must be an object.")
    _require(git_identity == scientific_inputs.get("git_identity"), "Git identity is inconsistent.")
    commit = str(git_identity.get("commit", ""))
    _require(len(commit) == 40, "Generation commit is not a full digest.")
    contract_blob = _git_blob(commit, "configs/ordinal_benchmark_v3.json")
    _require(
        _sha256_bytes(contract_blob) == metadata["benchmark_contract_sha256"],
        "Generation-commit benchmark contract hash drifted.",
    )
    implementation_hashes = scientific_inputs.get("implementation_hashes")
    _require(isinstance(implementation_hashes, Mapping), "Implementation hashes are absent.")
    expected_implementations = {
        "src/experiments/ordinal_benchmark_v3.py",
        "src/models/ordinal_evaluation_v3.py",
        "src/models/ordinal_models_v3.py",
    }
    _require(
        set(implementation_hashes) == expected_implementations,
        "Generation implementation inventory drifted.",
    )
    for relative_path, expected_hash in implementation_hashes.items():
        _require(
            _sha256_bytes(_git_blob(commit, str(relative_path))) == expected_hash,
            f"Generation implementation hash drifted for {relative_path}.",
        )

    canonical_root = Path(canonical_v2_root)
    v2_hashes = scientific_inputs.get("canonical_v2_artifact_hashes")
    _require(isinstance(v2_hashes, Mapping), "Canonical-v2 source hashes are absent.")
    v2_files = {
        "fold_contract": "core/shared_folds/fold_contract.json",
        "outer_assignments": "core/shared_folds/fold_assignments.csv",
        "inner_assignments": "core/shared_folds/inner_fold_assignments.csv",
        "nominal_oof_predictions": "core/model_benchmarks/oof_predictions.csv",
    }
    _require(set(v2_hashes) == set(v2_files), "Canonical-v2 source inventory drifted.")
    for key, relative_path in v2_files.items():
        _require(
            sha256_file(canonical_root / relative_path) == v2_hashes[key],
            f"Canonical-v2 source hash drifted for {key}.",
        )

    extension = _read_csv(root / "extension_oof_predictions.csv")
    combined = _read_csv(root / "combined_oof_predictions.csv")
    aggregate = _read_csv(root / "aggregate_metrics.csv")
    per_class = _read_csv(root / "per_class_metrics.csv")
    confusion = _read_csv(root / "confusion_matrix.csv")
    candidates = _read_csv(root / "candidate_search_results.csv")
    selected = _read_csv(root / "selected_hyperparameters.csv")
    fold_metrics = _read_csv(root / "extension_fold_metrics.csv")

    _require(len(extension) == 5 * 1200, "Extension OOF row count drifted.")
    _require(len(combined) == 9 * 1200, "Combined OOF row count drifted.")
    _require(set(extension["model"].unique()) == set(EXTENSION_MODEL_NAMES), "Extension model set drifted.")
    _require(set(combined["model"].unique()) == EXPECTED_MODELS, "Combined model set drifted.")
    _require(
        set(extension["evidence_source"].unique()) == {"v3_new_outer_fold_fit"},
        "Extension evidence-source labels drifted.",
    )
    nominal_rows = combined[combined["model"].isin(CANONICAL_MODEL_NAMES)].copy()
    _require(
        set(nominal_rows["evidence_source"].unique())
        == {"canonical_v2_reused_without_refit_or_relabelling"},
        "Nominal evidence-source labels drifted.",
    )
    for model_name, rows in combined.groupby("model", sort=False):
        _require(len(rows) == 1200, f"OOF count drifted for {model_name}.")
        _require(rows["sample_index"].nunique() == 1200, f"OOF uniqueness drifted for {model_name}.")
        _require(set(rows["outer_fold"].astype(int)) == set(range(1, 11)), f"Fold coverage drifted for {model_name}.")
        probability = rows[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float)
        _require(np.all(np.isfinite(probability)), f"Non-finite probabilities for {model_name}.")
        _require(
            np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-9),
            f"Probability simplex drifted for {model_name}.",
        )
        _require(
            np.array_equal(
                np.asarray([2, 3, 4])[np.argmax(probability, axis=1)],
                rows["y_pred"].to_numpy(int),
            ),
            f"Prediction/probability disagreement for {model_name}.",
        )

    outer_assignments = _read_csv(canonical_root / v2_files["outer_assignments"])
    fold_lookup = outer_assignments.set_index("sample_index")[["outer_fold", "y_true"]].sort_index()
    for model_name, rows in combined.groupby("model", sort=False):
        observed = rows.set_index("sample_index")[["outer_fold", "y_true"]].astype(int).sort_index()
        _require(observed.equals(fold_lookup.astype(int)), f"Fold/target lineage drifted for {model_name}.")

    nominal_source = _read_csv(canonical_root / v2_files["nominal_oof_predictions"])
    core_columns = [
        "model",
        "sample_index",
        "outer_fold",
        "y_true",
        "y_pred",
        "prob_class_2",
        "prob_class_3",
        "prob_class_4",
    ]
    _assert_frame_equal(
        nominal_rows[core_columns],
        nominal_source[core_columns],
        sort_columns=("model", "sample_index"),
        context="Reused nominal OOF evidence",
    )
    extension_in_combined = combined[combined["model"].isin(EXTENSION_MODEL_NAMES)]
    _assert_frame_equal(
        extension_in_combined[core_columns],
        extension[core_columns],
        sort_columns=("model", "sample_index"),
        context="Extension OOF combination",
    )

    recomputed_aggregate, recomputed_per_class, recomputed_confusion = summarize_combined_oof_v3(
        combined
    )
    _assert_frame_equal(
        aggregate,
        recomputed_aggregate,
        sort_columns=("model_name", "metric"),
        context="Aggregate metrics",
    )
    _assert_frame_equal(
        per_class,
        recomputed_per_class,
        sort_columns=("model_name", "class_label"),
        context="Per-class metrics",
    )
    _assert_frame_equal(
        confusion,
        recomputed_confusion,
        sort_columns=("model_name", "true_label", "predicted_label"),
        context="Confusion matrix",
    )
    _require(len(aggregate) == 9 * 16, "Aggregate metric grid is incomplete.")
    _require(len(per_class) == 9 * 3, "Per-class metric grid is incomplete.")
    _require(len(confusion) == 9 * 9, "Confusion grid is incomplete.")
    _require(np.all(np.isfinite(aggregate["value"].to_numpy(float))), "Aggregate metrics are non-finite.")

    _require(len(candidates) == 10 * (6 + 8), "Candidate-search row count drifted.")
    _require(not candidates["outer_test_used_for_selection"].astype(bool).any(), "Outer test entered candidate selection.")
    selected_flags = candidates.groupby(["outer_fold", "model"])["selected_by_protocol"].sum()
    _require(selected_flags.eq(1).all(), "Each ordinal model/fold must select exactly one candidate.")
    _require(len(selected) == 10 * 5, "Selected-hyperparameter row count drifted.")
    _require(not selected["outer_test_used_for_selection"].astype(bool).any(), "Outer test entered selected records.")
    ordinal_selected = selected[selected["model"].isin(V3_ORDINAL_MODEL_NAMES)]
    baseline_selected = selected[selected["model"].isin(V3_NAIVE_BASELINE_NAMES)]
    _require(ordinal_selected["selection_performed"].astype(bool).all(), "Ordinal selection flags drifted.")
    _require(not baseline_selected["selection_performed"].astype(bool).any(), "A baseline entered selection.")
    _require(len(fold_metrics) == 10 * 5, "Extension fold-metric grid is incomplete.")
    _require(
        set(fold_metrics["model"].unique()) == set(EXTENSION_MODEL_NAMES),
        "Extension fold-metric model set drifted.",
    )

    metric_pivot = aggregate.pivot(index="model_name", columns="metric", values="value")
    return {
        "status": "passed",
        "run_id": metadata["run_id"],
        "generation_commit": commit,
        "scientific_input_sha256": metadata["scientific_input_sha256"],
        "fold_contract_hash": metadata["fold_contract_hash"],
        "file_count": len(EXPECTED_FILES),
        "model_count": 9,
        "sample_count_per_model": 1200,
        "combined_oof_row_count": len(combined),
        "candidate_search_row_count": len(candidates),
        "aggregate_metric_row_count": len(aggregate),
        "per_class_row_count": len(per_class),
        "confusion_row_count": len(confusion),
        "best_macro_f1_model": str(metric_pivot["macro_f1"].idxmax()),
        "best_qwk_model": str(metric_pivot["quadratic_weighted_kappa"].idxmax()),
        "best_rps_model": str(metric_pivot["ranked_probability_score"].idxmin()),
        "paid_api_calls": 0,
        "network_calls": 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--canonical-v2-root", type=Path, default=DEFAULT_CANONICAL_V2_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    receipt = validate_ordinal_benchmark_run_v3(
        args.run_dir,
        canonical_v2_root=args.canonical_v2_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_FILES",
    "V3OrdinalRunValidationError",
    "validate_ordinal_benchmark_run_v3",
]
