"""Fail-closed reader for prediction-producing XGBoost OOF artifacts.

The benchmark stage persists one fitted pipeline for every outer fold.  SHAP
and other downstream consumers must load those exact pipelines rather than
refitting a look-alike model.  This module binds the persisted models to the
shared-fold, selection, OOF-prediction, and transformed-feature-lineage
evidence before exposing them to a consumer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.experiments.shared_folds import (
    CONTRACT_FILENAME,
    INNER_ASSIGNMENT_FILENAME,
    OUTER_ASSIGNMENT_FILENAME,
    SharedFoldArtifacts,
    SharedFoldContractError,
    read_shared_folds,
)
from src.governance.manuscript_contract import sha256_file
from src.models.canonical_models import (
    CanonicalModelError,
    CanonicalXGBClassifier,
    aligned_predict_proba,
)


XGBOOST_MODEL_NAME = "xgboost"
REQUIRED_OUTER_FOLDS = 10
REQUIRED_INNER_FOLDS = 5
BENCHMARK_SCHEMA_VERSION = 3
BENCHMARK_PROTOCOL_NAME = "restrained_nested_tuning_v2_10x5"
PERSISTED_PROBABILITY_ATOL = 1e-6
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_IDENTITY_COLUMNS = (
    "run_id",
    "config_hash",
    "scientific_input_hash",
    "fold_contract_hash",
)


class BenchmarkArtifactContractError(RuntimeError):
    """Raised when persisted benchmark evidence cannot be trusted."""


@dataclass(frozen=True)
class BenchmarkArtifactIdentity:
    """Scientific identity shared by folds, models, and OOF evidence."""

    run_id: str
    config_hash: str
    scientific_input_hash: str
    fold_contract_hash: str


@dataclass(frozen=True)
class XGBoostFoldModel:
    """One validated prediction-producing outer-fold XGBoost pipeline."""

    outer_fold: int
    pipeline: Pipeline = field(repr=False, compare=False)
    path: Path
    sha256: str
    size_bytes: int
    selected_candidate_index: int
    transformed_feature_names: tuple[str, ...]
    test_sample_indices: tuple[int, ...]


@dataclass(frozen=True)
class XGBoostOOFArtifacts:
    """Validated XGBoost benchmark evidence ready for downstream use."""

    identity: BenchmarkArtifactIdentity
    folds: SharedFoldArtifacts
    oof_predictions: pd.DataFrame = field(repr=False, compare=False)
    selected_hyperparameters: pd.DataFrame = field(repr=False, compare=False)
    model_index: pd.DataFrame = field(repr=False, compare=False)
    transformed_lineage: pd.DataFrame = field(repr=False, compare=False)
    fold_models: Mapping[int, XGBoostFoldModel] = field(repr=False, compare=False)
    model_set_sha256: str
    upstream_file_hashes: Mapping[str, str]
    baseline_gate: Mapping[str, Any]
    labels: tuple[int, ...]
    raw_feature_order: tuple[str, ...]
    benchmark_dir: Path = field(repr=False, compare=False)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validated_identity(identity: BenchmarkArtifactIdentity) -> BenchmarkArtifactIdentity:
    if not isinstance(identity, BenchmarkArtifactIdentity):
        raise BenchmarkArtifactContractError(
            "expected_identity must be a BenchmarkArtifactIdentity."
        )
    if not isinstance(identity.run_id, str) or not identity.run_id.strip():
        raise BenchmarkArtifactContractError("Benchmark run_id must be non-blank.")
    for field_name in ("config_hash", "scientific_input_hash", "fold_contract_hash"):
        value = getattr(identity, field_name)
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise BenchmarkArtifactContractError(
                f"Benchmark identity {field_name} must be a lowercase SHA-256."
            )
    return identity


def _validated_labels(labels: Sequence[int]) -> tuple[int, ...]:
    values: list[int] = []
    for value in labels:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise BenchmarkArtifactContractError("Declared labels must be integers.")
        values.append(int(value))
    if not values or len(values) != len(set(values)):
        raise BenchmarkArtifactContractError("Declared labels must be non-empty and unique.")
    return tuple(values)


def _validated_raw_feature_order(raw_feature_order: Sequence[str]) -> tuple[str, ...]:
    values = tuple(raw_feature_order)
    if (
        not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
        or len(values) != len(set(values))
    ):
        raise BenchmarkArtifactContractError(
            "Raw feature order must contain unique, non-blank string names."
        )
    return values


def _read_json(path: Path, *, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise BenchmarkArtifactContractError(f"Required {name} is missing: {path.name}.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkArtifactContractError(f"Required {name} cannot be read: {exc}") from exc
    if not isinstance(value, Mapping):
        raise BenchmarkArtifactContractError(f"Required {name} must contain a JSON object.")
    return value


def _read_csv(path: Path, *, name: str, required_columns: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise BenchmarkArtifactContractError(f"Required {name} is missing: {path.name}.")
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise BenchmarkArtifactContractError(f"Required {name} cannot be read: {exc}") from exc
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise BenchmarkArtifactContractError(
            f"Required {name} columns are missing: {missing}."
        )
    if frame.empty:
        raise BenchmarkArtifactContractError(f"Required {name} is empty.")
    return frame


def _identity_values(identity: BenchmarkArtifactIdentity) -> dict[str, str]:
    return {
        "run_id": identity.run_id,
        "config_hash": identity.config_hash,
        "scientific_input_hash": identity.scientific_input_hash,
        "fold_contract_hash": identity.fold_contract_hash,
    }


def _validate_mapping_identity(
    value: Mapping[str, Any],
    *,
    identity: BenchmarkArtifactIdentity,
    name: str,
) -> None:
    for column, expected in _identity_values(identity).items():
        if str(value.get(column)) != expected:
            raise BenchmarkArtifactContractError(
                f"{name} {column} does not match the expected benchmark identity."
            )


def _validate_frame_identity(
    frame: pd.DataFrame,
    *,
    identity: BenchmarkArtifactIdentity,
    name: str,
) -> None:
    for column, expected in _identity_values(identity).items():
        if column not in frame.columns or set(frame[column].astype(str)) != {expected}:
            raise BenchmarkArtifactContractError(
                f"{name} does not carry exactly one expected {column} identity."
            )


def _integer_series(series: pd.Series, *, context: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise BenchmarkArtifactContractError(f"{context} must contain integers.") from exc
    values = numeric.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or not np.equal(values, np.floor(values)).all():
        raise BenchmarkArtifactContractError(f"{context} must contain finite integers.")
    return pd.Series(values.astype(np.int64), index=series.index)


def _boolean_series(series: pd.Series, *, context: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        1: True,
        0: False,
    }
    converted = series.map(mapping)
    if converted.isna().any():
        raise BenchmarkArtifactContractError(f"{context} contains a non-boolean value.")
    return converted.astype(bool)


def _require_exact_outer_grid(frame: pd.DataFrame, *, name: str) -> pd.Series:
    outer = _integer_series(frame["outer_fold"], context=f"{name}.outer_fold")
    if set(outer) != set(range(1, REQUIRED_OUTER_FOLDS + 1)):
        raise BenchmarkArtifactContractError(f"{name} does not contain outer folds 1..10.")
    if len(frame) != REQUIRED_OUTER_FOLDS or outer.duplicated().any():
        raise BenchmarkArtifactContractError(
            f"{name} must contain exactly one row per outer fold."
        )
    return outer


def _snapshot_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise BenchmarkArtifactContractError(
                f"Required upstream file {name!r} is missing or is a symbolic link."
            )
        result[name] = sha256_file(path)
    return result


def _safe_relative_model_path(benchmark_dir: Path, value: Any) -> tuple[Path, Path]:
    text = str(value)
    relative = Path(text)
    if (
        not text
        or "\\" in text
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != text
        or relative.suffix != ".joblib"
    ):
        raise BenchmarkArtifactContractError(
            f"Model index contains a non-portable model path: {text!r}."
        )
    resolved = (benchmark_dir / relative).resolve()
    try:
        resolved.relative_to(benchmark_dir)
    except ValueError as exc:
        raise BenchmarkArtifactContractError("Model index path escapes the benchmark directory.") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise BenchmarkArtifactContractError(f"Persisted model file is missing: {text}.")
    return relative, resolved


def _read_parameter_mapping(value: Any, *, context: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise BenchmarkArtifactContractError(f"{context} is not valid JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise BenchmarkArtifactContractError(f"{context} must encode a JSON object.")
    return dict(parsed)


def _validate_pipeline(
    pipeline: Any,
    *,
    raw_feature_order: tuple[str, ...],
    labels: tuple[int, ...],
    transformed_feature_names: tuple[str, ...],
    selected_row: pd.Series,
    outer_fold: int,
) -> Pipeline:
    if not isinstance(pipeline, Pipeline) or tuple(pipeline.named_steps) != (
        "preprocessor",
        "model",
    ):
        raise BenchmarkArtifactContractError(
            f"Outer fold {outer_fold} artifact is not the canonical two-step pipeline."
        )
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    if not isinstance(estimator, CanonicalXGBClassifier):
        raise BenchmarkArtifactContractError(
            f"Outer fold {outer_fold} artifact is not a CanonicalXGBClassifier pipeline."
        )
    pipeline_features = tuple(str(value) for value in getattr(pipeline, "feature_names_in_", ()))
    preprocessor_features = tuple(
        str(value) for value in getattr(preprocessor, "feature_names_in_", ())
    )
    if pipeline_features != raw_feature_order or preprocessor_features != raw_feature_order:
        raise BenchmarkArtifactContractError(
            f"Outer fold {outer_fold} raw feature order differs from the canonical policy."
        )
    try:
        fitted_names = tuple(str(value) for value in preprocessor.get_feature_names_out())
    except Exception as exc:
        raise BenchmarkArtifactContractError(
            f"Outer fold {outer_fold} preprocessor has no readable fitted feature lineage."
        ) from exc
    if fitted_names != transformed_feature_names:
        raise BenchmarkArtifactContractError(
            f"Outer fold {outer_fold} fitted feature lineage differs from persisted evidence."
        )
    if not fitted_names or len(fitted_names) != len(set(fitted_names)):
        raise BenchmarkArtifactContractError(
            f"Outer fold {outer_fold} transformed feature names are empty or duplicated."
        )
    classes = tuple(int(value) for value in getattr(pipeline, "classes_", ()))
    if classes != labels:
        raise BenchmarkArtifactContractError(
            f"Outer fold {outer_fold} model classes {classes} differ from labels {labels}."
        )
    fixed = _read_parameter_mapping(
        selected_row["fixed_parameters_json"],
        context=f"outer fold {outer_fold} fixed_parameters_json",
    )
    selected = _read_parameter_mapping(
        selected_row["selected_candidate_parameters_json"],
        context=f"outer fold {outer_fold} selected_candidate_parameters_json",
    )
    overlap = set(fixed).intersection(selected)
    if overlap:
        raise BenchmarkArtifactContractError(
            f"Outer fold {outer_fold} fixed and candidate parameter sets overlap: {sorted(overlap)}."
        )
    estimator_parameters = estimator.get_params(deep=False)
    for parameter, expected in {**fixed, **selected}.items():
        if parameter not in estimator_parameters or estimator_parameters[parameter] != expected:
            raise BenchmarkArtifactContractError(
                f"Outer fold {outer_fold} fitted model parameter {parameter!r} differs from "
                "the selected-hyperparameter evidence."
            )
    return pipeline


def _model_set_hash(
    *,
    identity: BenchmarkArtifactIdentity,
    labels: tuple[int, ...],
    raw_feature_order: tuple[str, ...],
    fold_models: Mapping[int, XGBoostFoldModel],
) -> str:
    payload = {
        "schema_version": 1,
        "identity": _identity_values(identity),
        "model": XGBOOST_MODEL_NAME,
        "labels": list(labels),
        "raw_feature_order": list(raw_feature_order),
        "fold_models": [
            {
                "outer_fold": model.outer_fold,
                "path": model.path.as_posix(),
                "sha256": model.sha256,
                "size_bytes": model.size_bytes,
                "selected_candidate_index": model.selected_candidate_index,
                "transformed_feature_names": list(model.transformed_feature_names),
                "test_sample_indices": list(model.test_sample_indices),
            }
            for model in (fold_models[fold] for fold in sorted(fold_models))
        ],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def read_xgboost_oof_artifacts(
    shared_folds_dir: str | Path,
    model_benchmarks_dir: str | Path,
    *,
    expected_run_id: str,
    expected_config_hash: str,
    expected_scientific_input_hash: str,
    expected_feature_columns: Sequence[str],
    expected_labels: Sequence[int],
) -> XGBoostOOFArtifacts:
    """Read and cross-validate the exact persisted XGBoost outer-fold models."""

    declared_labels = _validated_labels(expected_labels)
    declared_raw_order = _validated_raw_feature_order(expected_feature_columns)
    benchmark = Path(model_benchmarks_dir).resolve()
    shared = Path(shared_folds_dir).resolve()
    if not benchmark.is_dir():
        raise BenchmarkArtifactContractError(f"Benchmark directory is missing: {benchmark}.")
    if not shared.is_dir():
        raise BenchmarkArtifactContractError(f"Shared-fold directory is missing: {shared}.")

    upstream_paths = {
        "shared_fold_contract": shared / CONTRACT_FILENAME,
        "shared_outer_assignments": shared / OUTER_ASSIGNMENT_FILENAME,
        "shared_inner_assignments": shared / INNER_ASSIGNMENT_FILENAME,
        "benchmark_stage_metadata": benchmark / "stage_metadata.json",
        "baseline_xgboost_gate": benchmark / "baseline_xgboost_gate.json",
        "paired_model_differences": benchmark / "paired_model_differences.csv",
        "candidate_search_results": benchmark / "candidate_search_results.csv",
        "selected_hyperparameters": benchmark / "selected_hyperparameters.csv",
        "oof_predictions": benchmark / "oof_predictions.csv",
        "fitted_model_index": benchmark / "fitted_model_index.csv",
        "transformed_feature_lineage": benchmark / "transformed_feature_lineage.csv",
    }
    upstream_hashes = _snapshot_hashes(upstream_paths)
    try:
        folds = read_shared_folds(shared)
    except SharedFoldContractError as exc:
        raise BenchmarkArtifactContractError(f"Shared-fold evidence is invalid: {exc}") from exc
    identity = _validated_identity(
        BenchmarkArtifactIdentity(
            run_id=expected_run_id,
            config_hash=expected_config_hash,
            scientific_input_hash=expected_scientific_input_hash,
            fold_contract_hash=str(folds.contract.get("fold_contract_hash", "")),
        )
    )
    if int(folds.contract.get("outer_splits", -1)) != REQUIRED_OUTER_FOLDS:
        raise BenchmarkArtifactContractError("Shared-fold evidence must use exactly 10 outer folds.")
    if int(folds.contract.get("inner_splits", -1)) != REQUIRED_INNER_FOLDS:
        raise BenchmarkArtifactContractError("Shared-fold evidence must use exactly five inner folds.")
    _validate_mapping_identity(folds.contract, identity=identity, name="Shared-fold contract")
    fold_labels = tuple(int(value) for value in folds.contract.get("target_labels", ()))
    if fold_labels != declared_labels:
        raise BenchmarkArtifactContractError(
            "Shared-fold target-label order differs from the declared benchmark labels."
        )

    metadata = _read_json(upstream_paths["benchmark_stage_metadata"], name="stage metadata")
    _validate_mapping_identity(metadata, identity=identity, name="Benchmark stage metadata")
    expected_metadata = {
        "stage": "model_benchmarks",
        "status": "complete",
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_protocol_name": BENCHMARK_PROTOCOL_NAME,
        "selection_metric": "macro_f1",
        "selection_tie_break_metric": "quadratic_weighted_kappa",
    }
    for field_name, expected in expected_metadata.items():
        if metadata.get(field_name) != expected:
            raise BenchmarkArtifactContractError(
                f"Benchmark stage metadata {field_name} must be {expected!r}."
            )
    models = metadata.get("models")
    if not isinstance(models, list) or XGBOOST_MODEL_NAME not in map(str, models):
        raise BenchmarkArtifactContractError("Benchmark metadata does not declare XGBoost.")

    gate = _read_json(upstream_paths["baseline_xgboost_gate"], name="baseline gate")
    _validate_mapping_identity(gate, identity=identity, name="Baseline gate")
    expected_gate_fields = {
        "comparison_direction": "baseline_improvement_over_xgboost",
        "gate_metric": "macro_f1",
        "gate_triggered": False,
        "trigger_rule": "point_estimate_gt_zero_and_paired_ci_low_gt_zero",
        "user_decision_required_if_triggered": True,
    }
    for field_name, expected in expected_gate_fields.items():
        observed = gate.get(field_name)
        matches = observed is expected if isinstance(expected, bool) else observed == expected
        if not matches:
            raise BenchmarkArtifactContractError(
                f"Baseline gate {field_name} must be {expected!r} before XGBoost SHAP."
            )
    if "decision_required" in gate and gate.get("decision_required") is not False:
        raise BenchmarkArtifactContractError(
            "Baseline gate decision_required must be false before XGBoost SHAP."
        )
    triggered_comparisons = gate.get("triggered_comparisons")
    if not isinstance(triggered_comparisons, list) or triggered_comparisons:
        raise BenchmarkArtifactContractError(
            "Baseline gate has a triggered comparison; XGBoost reference decision is unresolved."
        )
    if metadata.get("baseline_gate") != gate:
        raise BenchmarkArtifactContractError(
            "Benchmark stage metadata baseline gate differs from its standalone evidence."
        )

    paired = _read_csv(
        upstream_paths["paired_model_differences"],
        name="paired model-difference evidence",
        required_columns=(
            *_IDENTITY_COLUMNS,
            "comparison_id",
            "metric",
            "improvement_oriented_difference",
            "improvement_ci_low",
            "gate_eligible",
            "gate_triggered",
            "resample_hash",
        ),
    )
    _validate_frame_identity(paired, identity=identity, name="paired model-difference evidence")
    expected_comparisons = {
        "logistic_regression_minus_xgboost",
        "random_forest_minus_xgboost",
        "lightgbm_minus_xgboost",
    }
    gate_rows = paired[
        (paired["metric"].astype(str) == "macro_f1")
        & (paired["comparison_id"].astype(str).isin(expected_comparisons))
    ].copy()
    if (
        len(gate_rows) != len(expected_comparisons)
        or set(gate_rows["comparison_id"].astype(str)) != expected_comparisons
        or gate_rows["comparison_id"].astype(str).duplicated().any()
    ):
        raise BenchmarkArtifactContractError(
            "Paired evidence must contain exactly three baseline-minus-XGBoost macro-F1 gate rows."
        )
    if not _boolean_series(gate_rows["gate_eligible"], context="paired gate_eligible").all():
        raise BenchmarkArtifactContractError("All three macro-F1 comparison rows must be gate eligible.")
    try:
        point = pd.to_numeric(
            gate_rows["improvement_oriented_difference"], errors="raise"
        ).astype(float)
        ci_low = pd.to_numeric(gate_rows["improvement_ci_low"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise BenchmarkArtifactContractError("Paired gate values must be numeric.") from exc
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(ci_low)):
        raise BenchmarkArtifactContractError("Paired gate values must be finite.")
    recomputed_trigger = (point > 0.0) & (ci_low > 0.0)
    recorded_trigger = _boolean_series(
        gate_rows["gate_triggered"], context="paired gate_triggered"
    )
    if not recorded_trigger.equals(recomputed_trigger):
        raise BenchmarkArtifactContractError(
            "Paired gate_triggered values violate the predeclared point-plus-CI rule."
        )
    if recomputed_trigger.any():
        raise BenchmarkArtifactContractError(
            "A baseline superiority gate is triggered; XGBoost SHAP requires a user decision."
        )
    resample_hash = str(gate.get("resample_hash", ""))
    if (
        _SHA256_PATTERN.fullmatch(resample_hash) is None
        or set(gate_rows["resample_hash"].astype(str)) != {resample_hash}
    ):
        raise BenchmarkArtifactContractError(
            "Baseline gate resample hash differs from paired comparison evidence."
        )

    candidate = _read_csv(
        upstream_paths["candidate_search_results"],
        name="candidate-search evidence",
        required_columns=(
            *_IDENTITY_COLUMNS,
            "model",
            "outer_fold",
            "candidate_index",
            "parameters_json",
            "selected_by_protocol",
            "outer_test_used_for_selection",
        ),
    )
    selected = _read_csv(
        upstream_paths["selected_hyperparameters"],
        name="selected-hyperparameter evidence",
        required_columns=(
            *_IDENTITY_COLUMNS,
            "model",
            "outer_fold",
            "selected_candidate_index",
            "selected_candidate_parameters_json",
            "fixed_parameters_json",
            "outer_test_used_for_selection",
        ),
    )
    oof = _read_csv(
        upstream_paths["oof_predictions"],
        name="OOF prediction evidence",
        required_columns=(
            *_IDENTITY_COLUMNS,
            "system_id",
            "model",
            "sample_index",
            "outer_fold",
            "y_true",
            "y_pred",
            "selected_candidate_index",
            *(f"prob_class_{label}" for label in declared_labels),
        ),
    )
    model_index = _read_csv(
        upstream_paths["fitted_model_index"],
        name="fitted-model index",
        required_columns=(
            *_IDENTITY_COLUMNS,
            "model",
            "outer_fold",
            "path",
            "sha256",
            "size_bytes",
        ),
    )
    lineage = _read_csv(
        upstream_paths["transformed_feature_lineage"],
        name="transformed-feature lineage",
        required_columns=(
            *_IDENTITY_COLUMNS,
            "model",
            "outer_fold",
            "transformed_feature_index",
            "transformed_feature_name",
        ),
    )
    for name, frame in (
        ("candidate-search evidence", candidate),
        ("selected-hyperparameter evidence", selected),
        ("OOF prediction evidence", oof),
        ("fitted-model index", model_index),
        ("transformed-feature lineage", lineage),
    ):
        _validate_frame_identity(frame, identity=identity, name=name)

    candidate = candidate[candidate["model"].astype(str) == XGBOOST_MODEL_NAME].copy()
    selected = selected[selected["model"].astype(str) == XGBOOST_MODEL_NAME].copy()
    oof = oof[oof["model"].astype(str) == XGBOOST_MODEL_NAME].copy()
    model_index = model_index[model_index["model"].astype(str) == XGBOOST_MODEL_NAME].copy()
    lineage = lineage[lineage["model"].astype(str) == XGBOOST_MODEL_NAME].copy()
    if any(frame.empty for frame in (candidate, selected, oof, model_index, lineage)):
        raise BenchmarkArtifactContractError("One or more XGBoost benchmark evidence tables are empty.")
    if set(oof["system_id"].astype(str)) != {XGBOOST_MODEL_NAME}:
        raise BenchmarkArtifactContractError("XGBoost OOF system_id differs from its model name.")

    selected_outer = _require_exact_outer_grid(selected, name="selected_hyperparameters")
    selected["outer_fold"] = selected_outer
    selected_indices = _integer_series(
        selected["selected_candidate_index"], context="selected candidate index"
    )
    if (selected_indices < 0).any():
        raise BenchmarkArtifactContractError("Selected candidate indices must be nonnegative.")
    selected["selected_candidate_index"] = selected_indices
    if _boolean_series(
        selected["outer_test_used_for_selection"],
        context="selected outer_test_used_for_selection",
    ).any():
        raise BenchmarkArtifactContractError("Outer-test evidence was used for candidate selection.")
    selected_lookup = selected.set_index("outer_fold")["selected_candidate_index"].astype(int)

    candidate["outer_fold"] = _integer_series(
        candidate["outer_fold"], context="candidate outer_fold"
    )
    candidate["candidate_index"] = _integer_series(
        candidate["candidate_index"], context="candidate index"
    )
    if set(candidate["outer_fold"]) != set(range(1, REQUIRED_OUTER_FOLDS + 1)):
        raise BenchmarkArtifactContractError("Candidate evidence does not cover outer folds 1..10.")
    if candidate.duplicated(["outer_fold", "candidate_index"]).any():
        raise BenchmarkArtifactContractError("Candidate evidence repeats an outer-fold candidate.")
    selected_flags = _boolean_series(
        candidate["selected_by_protocol"], context="candidate selected_by_protocol"
    )
    if _boolean_series(
        candidate["outer_test_used_for_selection"],
        context="candidate outer_test_used_for_selection",
    ).any():
        raise BenchmarkArtifactContractError("Outer-test evidence was used during candidate search.")
    selected_candidate_rows = candidate[selected_flags].copy()
    if len(selected_candidate_rows) != REQUIRED_OUTER_FOLDS:
        raise BenchmarkArtifactContractError("Every outer fold must select exactly one XGBoost candidate.")
    candidate_lookup = selected_candidate_rows.set_index("outer_fold")["candidate_index"].astype(int)
    if not candidate_lookup.sort_index().equals(selected_lookup.sort_index()):
        raise BenchmarkArtifactContractError(
            "Selected candidate indices disagree across candidate and selection evidence."
        )
    for outer_fold, selected_row in selected.set_index("outer_fold").iterrows():
        candidate_row = selected_candidate_rows[
            selected_candidate_rows["outer_fold"].astype(int) == int(outer_fold)
        ].iloc[0]
        if _read_parameter_mapping(
            candidate_row["parameters_json"],
            context=f"outer fold {outer_fold} candidate parameters_json",
        ) != _read_parameter_mapping(
            selected_row["selected_candidate_parameters_json"],
            context=f"outer fold {outer_fold} selected_candidate_parameters_json",
        ):
            raise BenchmarkArtifactContractError(
                f"Outer fold {outer_fold} selected parameters disagree with candidate evidence."
            )

    outer_assignments = folds.outer_assignments.copy()
    outer_assignments["sample_index"] = _integer_series(
        outer_assignments["sample_index"], context="shared sample_index"
    )
    outer_assignments["outer_fold"] = _integer_series(
        outer_assignments["outer_fold"], context="shared outer_fold"
    )
    expected_samples = set(outer_assignments["sample_index"])
    oof["sample_index"] = _integer_series(oof["sample_index"], context="OOF sample_index")
    oof["outer_fold"] = _integer_series(oof["outer_fold"], context="OOF outer_fold")
    oof["y_true"] = _integer_series(oof["y_true"], context="OOF y_true")
    oof["y_pred"] = _integer_series(oof["y_pred"], context="OOF y_pred")
    oof["selected_candidate_index"] = _integer_series(
        oof["selected_candidate_index"], context="OOF selected candidate index"
    )
    if (
        len(oof) != len(outer_assignments)
        or oof["sample_index"].duplicated().any()
        or set(oof["sample_index"]) != expected_samples
    ):
        raise BenchmarkArtifactContractError("XGBoost OOF coverage is not exactly once per sample.")
    expected_outer = outer_assignments.set_index("sample_index").sort_index()
    keyed_oof = oof.set_index("sample_index").sort_index()
    if not keyed_oof["outer_fold"].astype(int).equals(expected_outer["outer_fold"].astype(int)):
        raise BenchmarkArtifactContractError("XGBoost OOF folds differ from shared assignments.")
    if not keyed_oof["y_true"].astype(int).equals(expected_outer["y_true"].astype(int)):
        raise BenchmarkArtifactContractError("XGBoost OOF targets differ from shared assignments.")
    expected_selected = keyed_oof["outer_fold"].map(selected_lookup).astype(int)
    if not keyed_oof["selected_candidate_index"].astype(int).equals(expected_selected):
        raise BenchmarkArtifactContractError(
            "XGBoost OOF selected candidate indices differ from selection evidence."
        )
    for column in ("y_true", "y_pred"):
        if not set(keyed_oof[column].astype(int)).issubset(set(declared_labels)):
            raise BenchmarkArtifactContractError(f"XGBoost OOF {column} has an undeclared label.")
    probability_columns = [f"prob_class_{label}" for label in declared_labels]
    try:
        probabilities = keyed_oof[probability_columns].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise BenchmarkArtifactContractError("XGBoost OOF probabilities are not numeric.") from exc
    probability_values = probabilities.to_numpy(dtype=float)
    if (
        not np.all(np.isfinite(probability_values))
        or np.any(probability_values < 0.0)
        or np.any(probability_values > 1.0)
        or not np.allclose(
            probability_values.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=PERSISTED_PROBABILITY_ATOL,
        )
    ):
        raise BenchmarkArtifactContractError(
            "XGBoost OOF probabilities are not finite normalized probabilities."
        )
    probability_predictions = np.asarray(declared_labels)[np.argmax(probability_values, axis=1)]
    if not np.array_equal(probability_predictions, keyed_oof["y_pred"].to_numpy(dtype=int)):
        raise BenchmarkArtifactContractError(
            "XGBoost OOF predictions disagree with the persisted probability argmax."
        )

    model_outer = _require_exact_outer_grid(model_index, name="fitted_model_index")
    model_index["outer_fold"] = model_outer
    size_bytes = _integer_series(model_index["size_bytes"], context="model size_bytes")
    if (size_bytes <= 0).any():
        raise BenchmarkArtifactContractError("Persisted model sizes must be positive.")
    model_index["size_bytes"] = size_bytes
    if model_index["path"].astype(str).duplicated().any():
        raise BenchmarkArtifactContractError("Fitted-model index repeats a model path.")

    lineage["outer_fold"] = _integer_series(
        lineage["outer_fold"], context="lineage outer_fold"
    )
    lineage["transformed_feature_index"] = _integer_series(
        lineage["transformed_feature_index"], context="transformed feature index"
    )
    if set(lineage["outer_fold"]) != set(range(1, REQUIRED_OUTER_FOLDS + 1)):
        raise BenchmarkArtifactContractError("Transformed lineage does not cover outer folds 1..10.")

    fold_models: dict[int, XGBoostFoldModel] = {}
    selected_by_fold = selected.set_index("outer_fold")
    for index_row in model_index.sort_values("outer_fold").itertuples(index=False):
        outer_fold = int(index_row.outer_fold)
        digest = str(index_row.sha256)
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise BenchmarkArtifactContractError(
                f"Outer fold {outer_fold} model hash is not a lowercase SHA-256."
            )
        relative_path, resolved_path = _safe_relative_model_path(benchmark, index_row.path)
        expected_size = int(index_row.size_bytes)
        if resolved_path.stat().st_size != expected_size or sha256_file(resolved_path) != digest:
            raise BenchmarkArtifactContractError(
                f"Outer fold {outer_fold} model fails its persisted hash/size contract."
            )
        fold_lineage = lineage[lineage["outer_fold"].astype(int) == outer_fold].sort_values(
            "transformed_feature_index"
        )
        indices = fold_lineage["transformed_feature_index"].astype(int).tolist()
        if indices != list(range(len(indices))):
            raise BenchmarkArtifactContractError(
                f"Outer fold {outer_fold} transformed feature indices are not consecutive from zero."
            )
        transformed_names = tuple(fold_lineage["transformed_feature_name"].astype(str))
        if any(not value.strip() for value in transformed_names):
            raise BenchmarkArtifactContractError(
                f"Outer fold {outer_fold} transformed lineage contains a blank name."
            )
        try:
            pipeline = joblib.load(resolved_path)
        except Exception as exc:
            raise BenchmarkArtifactContractError(
                f"Outer fold {outer_fold} persisted model cannot be loaded: {type(exc).__name__}: {exc}"
            ) from exc
        if resolved_path.stat().st_size != expected_size or sha256_file(resolved_path) != digest:
            raise BenchmarkArtifactContractError(
                f"Outer fold {outer_fold} model changed while it was being loaded."
            )
        selected_row = selected_by_fold.loc[outer_fold]
        pipeline = _validate_pipeline(
            pipeline,
            raw_feature_order=declared_raw_order,
            labels=declared_labels,
            transformed_feature_names=transformed_names,
            selected_row=selected_row,
            outer_fold=outer_fold,
        )
        test_sample_indices = tuple(
            outer_assignments.loc[
                outer_assignments["outer_fold"].astype(int) == outer_fold,
                "sample_index",
            ].astype(int)
        )
        if not test_sample_indices:
            raise BenchmarkArtifactContractError(f"Outer fold {outer_fold} has no test samples.")
        fold_models[outer_fold] = XGBoostFoldModel(
            outer_fold=outer_fold,
            pipeline=pipeline,
            path=relative_path,
            sha256=digest,
            size_bytes=expected_size,
            selected_candidate_index=int(selected_row["selected_candidate_index"]),
            transformed_feature_names=transformed_names,
            test_sample_indices=test_sample_indices,
        )

    if set(fold_models) != set(range(1, REQUIRED_OUTER_FOLDS + 1)):
        raise BenchmarkArtifactContractError("Exactly 10 validated XGBoost fold models are required.")
    final_upstream_hashes = _snapshot_hashes(upstream_paths)
    if final_upstream_hashes != upstream_hashes:
        raise BenchmarkArtifactContractError("Upstream benchmark files changed while being read.")
    model_set_sha256 = _model_set_hash(
        identity=identity,
        labels=declared_labels,
        raw_feature_order=declared_raw_order,
        fold_models=fold_models,
    )
    return XGBoostOOFArtifacts(
        identity=identity,
        folds=folds,
        oof_predictions=keyed_oof.reset_index().sort_values("sample_index").reset_index(drop=True),
        selected_hyperparameters=selected.sort_values("outer_fold").reset_index(drop=True),
        model_index=model_index.sort_values("outer_fold").reset_index(drop=True),
        transformed_lineage=lineage.sort_values(
            ["outer_fold", "transformed_feature_index"]
        ).reset_index(drop=True),
        fold_models=MappingProxyType(dict(sorted(fold_models.items()))),
        model_set_sha256=model_set_sha256,
        upstream_file_hashes=MappingProxyType(dict(sorted(upstream_hashes.items()))),
        baseline_gate=MappingProxyType(dict(gate)),
        labels=declared_labels,
        raw_feature_order=declared_raw_order,
        benchmark_dir=benchmark,
    )


def validate_xgboost_oof_replay(
    artifacts: XGBoostOOFArtifacts,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    labels: Sequence[int],
    probability_atol: float = PERSISTED_PROBABILITY_ATOL,
) -> None:
    """Replay every OOF row through its exact persisted fold pipeline.

    The function never calls ``fit``.  It rechecks model bytes immediately
    before prediction and raises on any label or probability disagreement.
    """

    if not isinstance(artifacts, XGBoostOOFArtifacts):
        raise BenchmarkArtifactContractError("artifacts must be XGBoostOOFArtifacts.")
    if (
        isinstance(probability_atol, bool)
        or not isinstance(probability_atol, (int, float, np.integer, np.floating))
        or not math.isfinite(float(probability_atol))
        or float(probability_atol) < 0.0
        or float(probability_atol) > PERSISTED_PROBABILITY_ATOL
    ):
        raise BenchmarkArtifactContractError(
            f"Replay probability_atol must be finite and within [0, {PERSISTED_PROBABILITY_ATOL}]."
        )
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise BenchmarkArtifactContractError("Replay features must be a non-empty DataFrame.")
    if not isinstance(target, pd.Series) or target.empty:
        raise BenchmarkArtifactContractError("Replay target must be a non-empty Series.")
    if not features.index.is_unique:
        raise BenchmarkArtifactContractError("Replay feature sample indices must be unique.")
    if not target.index.is_unique or not features.index.equals(target.index):
        raise BenchmarkArtifactContractError(
            "Replay feature and target indices must be unique and identically ordered."
        )
    replay_labels = _validated_labels(labels)
    if replay_labels != artifacts.labels:
        raise BenchmarkArtifactContractError(
            "Replay labels differ from the persisted benchmark label order."
        )
    if tuple(features.columns) != artifacts.raw_feature_order:
        raise BenchmarkArtifactContractError(
            "Replay raw feature order differs from the persisted model contract."
        )
    try:
        feature_indices = {int(value) for value in features.index}
    except (TypeError, ValueError, OverflowError) as exc:
        raise BenchmarkArtifactContractError("Replay feature indices must be integers.") from exc
    expected_indices = set(artifacts.oof_predictions["sample_index"].astype(int))
    if feature_indices != expected_indices or len(features) != len(expected_indices):
        raise BenchmarkArtifactContractError(
            "Replay feature samples differ from the exactly-once OOF sample set."
        )
    replay_target = _integer_series(target, context="replay target")
    persisted_target = (
        artifacts.oof_predictions.set_index("sample_index")["y_true"].astype(int).sort_index()
    )
    observed_target = pd.Series(
        replay_target.to_numpy(dtype=int),
        index=pd.Index([int(value) for value in target.index], name="sample_index"),
    ).sort_index()
    if not observed_target.equals(persisted_target):
        raise BenchmarkArtifactContractError(
            "Replay target values differ from the shared-fold/OOF target evidence."
        )

    replay_rows: list[dict[str, Any]] = []
    probability_columns = [f"prob_class_{label}" for label in artifacts.labels]
    persisted = artifacts.oof_predictions.set_index("sample_index").sort_index()
    for outer_fold, fold_model in artifacts.fold_models.items():
        resolved_path = (artifacts.benchmark_dir / fold_model.path).resolve()
        try:
            resolved_path.relative_to(artifacts.benchmark_dir)
        except ValueError as exc:
            raise BenchmarkArtifactContractError("Fold model path escapes the benchmark directory.") from exc
        if (
            not resolved_path.is_file()
            or resolved_path.is_symlink()
            or resolved_path.stat().st_size != fold_model.size_bytes
            or sha256_file(resolved_path) != fold_model.sha256
        ):
            raise BenchmarkArtifactContractError(
                f"Outer fold {outer_fold} model bytes changed after artifact loading."
            )
        sample_indices = list(fold_model.test_sample_indices)
        fold_features = features.loc[sample_indices]
        try:
            prediction = np.asarray(fold_model.pipeline.predict(fold_features), dtype=int)
            probability = aligned_predict_proba(
                fold_model.pipeline,
                fold_features,
                labels=artifacts.labels,
            )
        except (CanonicalModelError, TypeError, ValueError) as exc:
            raise BenchmarkArtifactContractError(
                f"Outer fold {outer_fold} OOF replay failed: {type(exc).__name__}: {exc}"
            ) from exc
        expected = persisted.loc[sample_indices]
        if not np.array_equal(prediction, expected["y_pred"].to_numpy(dtype=int)):
            mismatch_count = int(
                np.count_nonzero(prediction != expected["y_pred"].to_numpy(dtype=int))
            )
            raise BenchmarkArtifactContractError(
                f"Outer fold {outer_fold} replay has {mismatch_count} prediction mismatches."
            )
        expected_probability = expected[probability_columns].to_numpy(dtype=float)
        if not np.allclose(
            probability,
            expected_probability,
            rtol=0.0,
            atol=float(probability_atol),
        ):
            maximum_delta = float(np.max(np.abs(probability - expected_probability)))
            raise BenchmarkArtifactContractError(
                f"Outer fold {outer_fold} replay probability mismatch; max delta={maximum_delta:.3g}."
            )
        for row_position, sample_index in enumerate(sample_indices):
            row = {
                "run_id": artifacts.identity.run_id,
                "config_hash": artifacts.identity.config_hash,
                "scientific_input_hash": artifacts.identity.scientific_input_hash,
                "fold_contract_hash": artifacts.identity.fold_contract_hash,
                "model_set_sha256": artifacts.model_set_sha256,
                "model_sha256": fold_model.sha256,
                "model": XGBOOST_MODEL_NAME,
                "sample_index": int(sample_index),
                "outer_fold": int(outer_fold),
                "y_true": int(expected.iloc[row_position]["y_true"]),
                "y_pred": int(prediction[row_position]),
                "selected_candidate_index": fold_model.selected_candidate_index,
            }
            row.update(
                {
                    column: float(probability[row_position, column_index])
                    for column_index, column in enumerate(probability_columns)
                }
            )
            replay_rows.append(row)
    replay = pd.DataFrame(replay_rows).sort_values("sample_index").reset_index(drop=True)
    if len(replay) != len(expected_indices) or replay["sample_index"].duplicated().any():
        raise BenchmarkArtifactContractError("Replay did not produce exactly one row per OOF sample.")
    return None


__all__ = [
    "BENCHMARK_PROTOCOL_NAME",
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkArtifactContractError",
    "BenchmarkArtifactIdentity",
    "PERSISTED_PROBABILITY_ATOL",
    "XGBOOST_MODEL_NAME",
    "XGBoostFoldModel",
    "XGBoostOOFArtifacts",
    "read_xgboost_oof_artifacts",
    "validate_xgboost_oof_replay",
]
