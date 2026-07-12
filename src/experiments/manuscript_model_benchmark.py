"""Restrained nested OOF benchmark for the four approved model families."""

from __future__ import annotations

import json
import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info, threadpool_limits

from src.data.canonical_loader import load_canonical_dataset
from src.experiments.shared_folds import (
    SharedFoldArtifacts,
    read_shared_folds,
    validate_consumer_fold_assignments,
    validate_shared_folds,
)
from src.governance.manuscript_contract import (
    canonical_config_hash,
    primary_excluded_features,
    sha256_file,
)
from src.models.canonical_models import (
    CANONICAL_ESTIMATOR_PATHS,
    CANONICAL_MODEL_NAMES,
    aligned_predict_proba,
    build_estimator,
    build_model_pipeline,
    merge_model_parameters,
    validate_model_feature_frame,
)
from src.models.evaluate import classification_metrics
from src.models.oof_bootstrap import (
    BootstrapProtocol,
    BootstrapResult,
    ComparisonSpec,
    compute_paired_oof_bootstrap,
    metric_definition,
    validate_aligned_oof_predictions,
)
from src.models.task_schema import ORDINAL_MULTICLASS_PERFORMANCE
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_CONFIG = Path("configs/manuscript_final.yaml")
DEFAULT_MODEL_GRID = Path("configs/model_grid.yaml")
EXPECTED_CANDIDATE_COUNTS = {
    "logistic_regression": 6,
    "random_forest": 8,
    "lightgbm": 8,
    "xgboost": 8,
}
BENCHMARK_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "quadratic_weighted_kappa",
    "ordinal_mae",
    "severe_error_rate",
    "nll_log_loss",
    "multiclass_brier",
    "ece_confidence",
)
FIT_THREAD_LIMIT = 1


class ModelBenchmarkError(RuntimeError):
    """Raised when any nested-search or OOF benchmark invariant fails."""


@dataclass(frozen=True)
class BenchmarkResult:
    candidate_search_results: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    model_summary: pd.DataFrame
    paired_model_differences: pd.DataFrame
    baseline_gate: Mapping[str, Any]
    bootstrap_metadata: Mapping[str, Any]
    fitted_outer_models: Mapping[tuple[str, int], Any]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _benchmark_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = config.get("model_benchmark", config)
    if not isinstance(settings, Mapping):
        raise ModelBenchmarkError("Model-grid config must contain model_benchmark mapping.")
    return settings


def validate_benchmark_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the frozen model/search contract before touching any dataset."""

    settings = _benchmark_settings(config)
    if settings.get("schema_version") != 2:
        raise ModelBenchmarkError("model_benchmark.schema_version must be 2.")
    if settings.get("candidate_failure_policy") != "fail_entire_stage":
        raise ModelBenchmarkError("Candidate failures must fail the entire benchmark stage.")
    if settings.get("tie_breaking") != "highest_inner_mean_then_lowest_candidate_index":
        raise ModelBenchmarkError("The deterministic candidate tie-break contract is missing.")
    selection_metric = settings.get("selection_metric")
    gate_metric = settings.get("baseline_gate_metric")
    if not isinstance(selection_metric, str) or not selection_metric.strip():
        raise ModelBenchmarkError(
            "selection_metric is pending; real nested benchmark execution is prohibited."
        )
    if not isinstance(gate_metric, str) or not gate_metric.strip():
        raise ModelBenchmarkError(
            "baseline_gate_metric is pending; real nested benchmark execution is prohibited."
        )
    for metric in (selection_metric, gate_metric):
        try:
            metric_definition(metric)
        except Exception as exc:
            raise ModelBenchmarkError(f"Unsupported benchmark metric {metric!r}: {exc}") from exc

    models = settings.get("models")
    if not isinstance(models, Mapping) or set(models) != set(CANONICAL_MODEL_NAMES):
        raise ModelBenchmarkError(
            f"Benchmark must define exactly the approved models {list(CANONICAL_MODEL_NAMES)}."
        )
    for model_name in CANONICAL_MODEL_NAMES:
        definition = models[model_name]
        if not isinstance(definition, Mapping):
            raise ModelBenchmarkError(f"Model definition {model_name!r} must be a mapping.")
        expected_estimator = CANONICAL_ESTIMATOR_PATHS[model_name]
        if definition.get("estimator") != expected_estimator:
            raise ModelBenchmarkError(
                f"Model {model_name!r} estimator must be {expected_estimator!r}; "
                f"observed {definition.get('estimator')!r}."
            )
        fixed = definition.get("fixed_params")
        candidates = definition.get("candidates")
        if not isinstance(fixed, Mapping):
            raise ModelBenchmarkError(f"Model {model_name!r} requires fixed_params.")
        if not isinstance(candidates, list) or len(candidates) != EXPECTED_CANDIDATE_COUNTS[model_name]:
            raise ModelBenchmarkError(
                f"Model {model_name!r} must declare exactly "
                f"{EXPECTED_CANDIDATE_COUNTS[model_name]} candidates."
            )
        if any(not isinstance(candidate, Mapping) for candidate in candidates):
            raise ModelBenchmarkError(f"Every candidate for {model_name!r} must be a mapping.")
        serialized = [_json_dumps(dict(candidate)) for candidate in candidates]
        if len(serialized) != len(set(serialized)):
            raise ModelBenchmarkError(f"Model {model_name!r} contains duplicate candidates.")
        if model_name in {"random_forest", "lightgbm", "xgboost"} and fixed.get("n_jobs") != 1:
            raise ModelBenchmarkError(f"Model {model_name!r} must be single-threaded.")
        for candidate_index, candidate in enumerate(candidates):
            try:
                merged = merge_model_parameters(fixed, candidate)
                build_estimator(model_name, merged, random_state=0)
            except Exception as exc:
                raise ModelBenchmarkError(
                    f"Model {model_name!r} candidate {candidate_index} fails estimator preflight: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
    return settings


def select_candidate_index(scores: Sequence[float], *, better_direction: str) -> int:
    """Select the best finite mean, resolving exact ties by candidate index."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ModelBenchmarkError("Candidate means must be a non-empty finite vector.")
    if better_direction == "higher":
        optimum = float(values.max())
    elif better_direction == "lower":
        optimum = float(values.min())
    else:
        raise ModelBenchmarkError(f"Unknown metric direction: {better_direction!r}.")
    return int(np.flatnonzero(values == optimum)[0])


def _metric_value(
    metric: str,
    y_true: pd.Series,
    prediction: np.ndarray,
    probability: np.ndarray,
    labels: Sequence[int],
    *,
    task_type: str,
) -> float:
    metrics = classification_metrics(
        y_true,
        prediction,
        probability,
        list(labels),
        task_type=task_type,
    )
    value = metrics.get(metric)
    if value is None or not math.isfinite(float(value)):
        raise ModelBenchmarkError(f"Metric {metric!r} is unavailable or non-finite.")
    return float(value)


def thread_determinism_metadata() -> dict[str, Any]:
    """Describe the native thread pools constrained during every model fit."""

    libraries = []
    for record in threadpool_info():
        libraries.append(
            {
                "user_api": record.get("user_api"),
                "internal_api": record.get("internal_api"),
                "prefix": record.get("prefix"),
                "version": record.get("version"),
                "detected_num_threads_before_fit_limit": record.get("num_threads"),
            }
        )
    return {
        "fit_thread_limit": FIT_THREAD_LIMIT,
        "control": "threadpoolctl.threadpool_limits",
        "estimator_parallelism": "n_jobs=1 where supported",
        "detected_libraries": libraries,
    }


def _fit_pipeline_or_fail(pipeline: Any, X: pd.DataFrame, y: pd.Series, *, context: str) -> Any:
    try:
        with threadpool_limits(limits=FIT_THREAD_LIMIT):
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                return pipeline.fit(X, y)
    except Exception as exc:
        raise ModelBenchmarkError(f"{context} failed: {type(exc).__name__}: {exc}") from exc


def evaluate_nested_benchmark(
    features: pd.DataFrame,
    target: pd.Series,
    folds: SharedFoldArtifacts,
    benchmark_config: Mapping[str, Any],
    *,
    labels: Sequence[int],
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    random_state: int,
    task_type: str = ORDINAL_MULTICLASS_PERFORMANCE,
    forbidden_features: Sequence[str] = (),
    bootstrap_protocol: BootstrapProtocol = BootstrapProtocol(),
) -> BenchmarkResult:
    """Evaluate all candidates using supplied inner folds and outer tests once."""

    settings = validate_benchmark_config(benchmark_config)
    validate_model_feature_frame(features, forbidden_features=forbidden_features)
    validate_shared_folds(folds)
    if not features.index.is_unique or not target.index.is_unique:
        raise ModelBenchmarkError("Feature and target sample indices must be unique.")
    if not features.index.equals(target.index):
        raise ModelBenchmarkError("Feature and target indices must be identically ordered.")
    outer = folds.outer_assignments.copy()
    inner = folds.inner_assignments.copy()
    expected_samples = set(outer["sample_index"].astype(int))
    if set(int(value) for value in features.index) != expected_samples:
        raise ModelBenchmarkError("Feature samples do not match shared-fold assignments.")
    persisted_target = outer.set_index("sample_index")["y_true"].sort_index()
    observed_target = target.sort_index()
    if not np.array_equal(persisted_target.to_numpy(), observed_target.to_numpy()):
        raise ModelBenchmarkError("Target values do not match the shared-fold contract.")
    if int(folds.contract.get("inner_splits", -1)) != 3:
        raise ModelBenchmarkError("Restrained nested tuning requires exactly three inner folds.")
    for field, expected in (
        ("run_id", run_id),
        ("config_hash", config_hash),
        ("scientific_input_hash", scientific_input_hash),
    ):
        if str(folds.contract.get(field)) != str(expected):
            raise ModelBenchmarkError(f"Shared-fold {field} does not match benchmark identity.")

    labels = tuple(int(label) for label in labels)
    if set(target.astype(int).unique()) != set(labels):
        raise ModelBenchmarkError("Observed target support does not equal declared benchmark labels.")
    selection_metric = str(settings["selection_metric"])
    gate_metric = str(settings["baseline_gate_metric"])
    selection_direction = metric_definition(selection_metric).better_direction
    model_definitions = settings["models"]
    fold_contract_hash = str(folds.contract["fold_contract_hash"])

    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fitted_models: dict[tuple[str, int], Any] = {}

    for outer_fold in sorted(outer["outer_fold"].astype(int).unique()):
        outer_test_ids = outer.loc[
            outer["outer_fold"].astype(int) == outer_fold, "sample_index"
        ].astype(int).tolist()
        outer_train_ids = outer.loc[
            outer["outer_fold"].astype(int) != outer_fold, "sample_index"
        ].astype(int).tolist()
        scoped_inner = inner[inner["outer_fold"].astype(int) == outer_fold].copy()
        if set(scoped_inner["sample_index"].astype(int)) != set(outer_train_ids):
            raise ModelBenchmarkError(
                f"Inner assignments for outer fold {outer_fold} are not the exact outer training set."
            )

        for model_name in CANONICAL_MODEL_NAMES:
            definition = model_definitions[model_name]
            fixed_parameters = dict(definition["fixed_params"])
            candidates = [dict(value) for value in definition["candidates"]]
            candidate_means: list[float] = []
            for candidate_index, candidate_parameters in enumerate(candidates):
                inner_scores: list[float] = []
                for inner_fold in (1, 2, 3):
                    validation_ids = scoped_inner.loc[
                        scoped_inner["inner_fold"].astype(int) == inner_fold,
                        "sample_index",
                    ].astype(int).tolist()
                    development_ids = sorted(set(outer_train_ids) - set(validation_ids))
                    if not validation_ids or not development_ids:
                        raise ModelBenchmarkError(
                            f"Empty inner partition at outer={outer_fold}, inner={inner_fold}."
                        )
                    pipeline = build_model_pipeline(
                        model_name,
                        features.loc[development_ids],
                        fixed_parameters=fixed_parameters,
                        candidate_parameters=candidate_parameters,
                        random_state=random_state,
                        forbidden_features=forbidden_features,
                    )
                    _fit_pipeline_or_fail(
                        pipeline,
                        features.loc[development_ids],
                        target.loc[development_ids],
                        context=(
                            f"model={model_name}, outer={outer_fold}, candidate={candidate_index}, "
                            f"inner={inner_fold}"
                        ),
                    )
                    validation_prediction = np.asarray(
                        pipeline.predict(features.loc[validation_ids]), dtype=int
                    )
                    validation_probability = aligned_predict_proba(
                        pipeline,
                        features.loc[validation_ids],
                        labels=labels,
                    )
                    inner_scores.append(
                        _metric_value(
                            selection_metric,
                            target.loc[validation_ids],
                            validation_prediction,
                            validation_probability,
                            labels,
                            task_type=task_type,
                        )
                    )
                mean_score = float(np.mean(inner_scores))
                candidate_means.append(mean_score)
                candidate_rows.append(
                    {
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "scientific_input_hash": scientific_input_hash,
                        "fold_contract_hash": fold_contract_hash,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "candidate_index": candidate_index,
                        "parameters_json": _json_dumps(candidate_parameters),
                        "selection_metric": selection_metric,
                        "inner_fold_scores_json": _json_dumps(inner_scores),
                        "inner_mean": mean_score,
                        "inner_std": float(np.std(inner_scores, ddof=1)),
                        "n_inner_folds": 3,
                        "candidate_status": "complete",
                        "outer_test_used_for_selection": False,
                    }
                )

            selected_index = select_candidate_index(
                candidate_means,
                better_direction=selection_direction,
            )
            selected_candidate = candidates[selected_index]
            final_pipeline = build_model_pipeline(
                model_name,
                features.loc[outer_train_ids],
                fixed_parameters=fixed_parameters,
                candidate_parameters=selected_candidate,
                random_state=random_state,
                forbidden_features=forbidden_features,
            )
            _fit_pipeline_or_fail(
                final_pipeline,
                features.loc[outer_train_ids],
                target.loc[outer_train_ids],
                context=f"outer refit model={model_name}, outer={outer_fold}",
            )
            outer_prediction = np.asarray(
                final_pipeline.predict(features.loc[outer_test_ids]), dtype=int
            )
            outer_probability = aligned_predict_proba(
                final_pipeline,
                features.loc[outer_test_ids],
                labels=labels,
            )
            metrics = classification_metrics(
                target.loc[outer_test_ids],
                outer_prediction,
                outer_probability,
                list(labels),
                task_type=task_type,
            )
            fitted_models[(model_name, outer_fold)] = final_pipeline
            selected_rows.append(
                {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "scientific_input_hash": scientific_input_hash,
                    "fold_contract_hash": fold_contract_hash,
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "selected_candidate_index": selected_index,
                    "selected_candidate_parameters_json": _json_dumps(selected_candidate),
                    "fixed_parameters_json": _json_dumps(fixed_parameters),
                    "selection_metric": selection_metric,
                    "selected_inner_mean": candidate_means[selected_index],
                    "tie_breaking": settings["tie_breaking"],
                }
            )
            fold_rows.append(
                {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "scientific_input_hash": scientific_input_hash,
                    "fold_contract_hash": fold_contract_hash,
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "n_train": len(outer_train_ids),
                    "n_test": len(outer_test_ids),
                    **metrics,
                }
            )
            for row_position, sample_index in enumerate(outer_test_ids):
                row = {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "scientific_input_hash": scientific_input_hash,
                    "fold_contract_hash": fold_contract_hash,
                    "system_id": model_name,
                    "model": model_name,
                    "sample_index": sample_index,
                    "outer_fold": outer_fold,
                    "y_true": int(target.loc[sample_index]),
                    "y_pred": int(outer_prediction[row_position]),
                    "selected_candidate_index": selected_index,
                }
                row.update(
                    {
                        f"prob_class_{label}": float(outer_probability[row_position, column])
                        for column, label in enumerate(labels)
                    }
                )
                prediction_rows.append(row)

    candidate_frame = pd.DataFrame(candidate_rows)
    selected_frame = pd.DataFrame(selected_rows)
    fold_frame = pd.DataFrame(fold_rows)
    oof_frame = pd.DataFrame(prediction_rows).sort_values(
        ["model", "sample_index"]
    ).reset_index(drop=True)
    validate_consumer_fold_assignments(
        folds,
        oof_frame,
        group_columns=("model",),
    )
    metrics_for_uncertainty = tuple(
        dict.fromkeys([*BENCHMARK_METRICS, selection_metric, gate_metric])
    )
    validate_aligned_oof_predictions(
        oof_frame,
        labels=labels,
        task_type=task_type,
        metrics=metrics_for_uncertainty,
    )
    comparisons = tuple(
        ComparisonSpec(
            comparison_id=f"{baseline}_minus_xgboost",
            system_a=baseline,
            system_b="xgboost",
            primary_gate=True,
        )
        for baseline in CANONICAL_MODEL_NAMES
        if baseline != "xgboost"
    )
    bootstrap: BootstrapResult = compute_paired_oof_bootstrap(
        oof_frame,
        labels=labels,
        task_type=task_type,
        metrics=metrics_for_uncertainty,
        comparisons=comparisons,
        primary_metric=gate_metric,
        protocol=bootstrap_protocol,
    )
    gate_rows = bootstrap.paired_differences[
        bootstrap.paired_differences["gate_eligible"].astype(bool)
    ]
    gate_payload = {
        "gate_metric": gate_metric,
        "comparison_direction": "baseline_improvement_over_xgboost",
        "trigger_rule": "paired_improvement_ci_low_greater_than_zero",
        "gate_triggered": bool(gate_rows["gate_triggered"].any()),
        "triggered_comparisons": gate_rows.loc[
            gate_rows["gate_triggered"].astype(bool), "comparison_id"
        ].tolist(),
        "user_decision_required_if_triggered": True,
        "n_resamples": bootstrap_protocol.n_resamples,
        "resample_hash": bootstrap.metadata["resample_hash"],
    }
    model_summary = bootstrap.metric_intervals.copy()
    paired_differences = bootstrap.paired_differences.copy()
    for identity_frame in (model_summary, paired_differences):
        identity_frame.insert(0, "fold_contract_hash", fold_contract_hash)
        identity_frame.insert(0, "scientific_input_hash", scientific_input_hash)
        identity_frame.insert(0, "config_hash", config_hash)
        identity_frame.insert(0, "run_id", run_id)
    gate_payload.update(
        {
            "run_id": run_id,
            "config_hash": config_hash,
            "scientific_input_hash": scientific_input_hash,
            "fold_contract_hash": fold_contract_hash,
        }
    )
    return BenchmarkResult(
        candidate_search_results=candidate_frame,
        selected_hyperparameters=selected_frame,
        fold_metrics=fold_frame,
        oof_predictions=oof_frame,
        model_summary=model_summary,
        paired_model_differences=paired_differences,
        baseline_gate=gate_payload,
        bootstrap_metadata=bootstrap.metadata,
        fitted_outer_models=fitted_models,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return path


def exact_primary_feature_frame(
    frame: pd.DataFrame,
    *,
    excluded_features: Sequence[str],
) -> pd.DataFrame:
    """Materialise exactly raw columns minus the canonical primary exclusions."""

    excluded = tuple(str(value) for value in excluded_features)
    if not excluded or len(set(excluded)) != len(excluded):
        raise ModelBenchmarkError("Canonical primary exclusions must be non-empty and unique.")
    unknown = sorted(set(excluded).difference(map(str, frame.columns)))
    if unknown:
        raise ModelBenchmarkError(f"Canonical primary exclusions are absent from the dataset: {unknown}.")
    expected_columns = [str(column) for column in frame.columns if str(column) not in set(excluded)]
    if not expected_columns:
        raise ModelBenchmarkError("Canonical primary exclusions leave no benchmark features.")
    features = frame.loc[:, expected_columns].copy()
    if features.columns.tolist() != expected_columns:
        raise ModelBenchmarkError("Benchmark feature order differs from the exact canonical policy.")
    validate_model_feature_frame(features, forbidden_features=excluded)
    return features


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    model_grid_path: str | Path = DEFAULT_MODEL_GRID,
    shared_folds_dir: str | Path,
    output_dir: str | Path,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    model_grid_sha256: str,
) -> dict[str, Path]:
    """Run and persist the real benchmark; pending metric choices block first."""

    model_grid_path = Path(model_grid_path)
    observed_model_grid_hash = sha256_file(model_grid_path)
    if observed_model_grid_hash != model_grid_sha256:
        raise ModelBenchmarkError(
            "Model search-space hash does not match the scoped run manifest."
        )
    model_grid = load_config(model_grid_path)
    settings = validate_benchmark_config(model_grid)  # intentionally before data access/output
    output = Path(output_dir)
    if output.exists() and not output.is_dir():
        raise ModelBenchmarkError(f"Benchmark output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise ModelBenchmarkError(f"Benchmark output directory must be empty: {output}")
    manuscript = load_config(config_path)
    observed_config_hash = canonical_config_hash(manuscript)
    if observed_config_hash != config_hash:
        raise ModelBenchmarkError("Supplied config_hash does not match the canonical config.")
    manuscript_settings = manuscript.get("manuscript_final", manuscript)
    folds = read_shared_folds(shared_folds_dir)
    canonical = load_canonical_dataset(config_path, "inx_primary")
    if canonical.receipt.get("actual_sha256") != folds.contract.get("dataset_sha256"):
        raise ModelBenchmarkError(
            "Current canonical dataset hash does not match the shared-fold contract."
        )
    frame = canonical.frame
    target_settings = manuscript_settings["target"]
    target_column = str(target_settings["column"])
    labels = [int(value) for value in target_settings["labels"]]
    identifier_fields = manuscript_settings["governance_fields"]["identifier_fields"]
    excluded = list(primary_excluded_features(manuscript))
    required_exclusions = {target_column, *map(str, identifier_fields)}
    if not required_exclusions.issubset(excluded):
        raise ModelBenchmarkError(
            "Canonical primary exclusions must contain the target and every identifier field."
        )
    features = exact_primary_feature_frame(frame, excluded_features=excluded)
    target = frame[target_column].astype(int)
    seed = int(manuscript_settings["seeds"]["model"])
    bootstrap_settings = manuscript_settings["evaluation"]["bootstrap"]
    protocol = BootstrapProtocol(
        n_resamples=int(bootstrap_settings["n_resamples"]),
        confidence_level=float(bootstrap_settings["confidence_level"]),
        seed=int(manuscript_settings["seeds"]["bootstrap"]),
    )
    result = evaluate_nested_benchmark(
        features,
        target,
        folds,
        settings,
        labels=labels,
        run_id=run_id,
        config_hash=config_hash,
        scientific_input_hash=scientific_input_hash,
        random_state=seed,
        forbidden_features=excluded,
        bootstrap_protocol=protocol,
    )

    if sha256_file(model_grid_path) != model_grid_sha256:
        raise ModelBenchmarkError("Model search space changed during benchmark execution.")

    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate_search_results": output / "candidate_search_results.csv",
        "selected_hyperparameters": output / "selected_hyperparameters.csv",
        "fold_metrics": output / "fold_metrics.csv",
        "oof_predictions": output / "oof_predictions.csv",
        "model_summary": output / "model_summary.csv",
        "paired_model_differences": output / "paired_model_differences.csv",
        "baseline_gate": output / "baseline_xgboost_gate.json",
        "metadata": output / "stage_metadata.json",
        "feature_lineage": output / "transformed_feature_lineage.csv",
        "model_index": output / "fitted_model_index.csv",
    }
    for frame_value, name in (
        (result.candidate_search_results, "candidate_search_results"),
        (result.selected_hyperparameters, "selected_hyperparameters"),
        (result.fold_metrics, "fold_metrics"),
        (result.oof_predictions, "oof_predictions"),
        (result.model_summary, "model_summary"),
        (result.paired_model_differences, "paired_model_differences"),
    ):
        frame_value.to_csv(paths[name], index=False)
    _write_json(paths["baseline_gate"], result.baseline_gate)

    model_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    for (model_name, outer_fold), fitted in sorted(result.fitted_outer_models.items()):
        model_path = output / "models" / model_name / f"outer_fold_{outer_fold:02d}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted, model_path)
        model_rows.append(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "scientific_input_hash": scientific_input_hash,
                "fold_contract_hash": folds.contract["fold_contract_hash"],
                "model": model_name,
                "outer_fold": outer_fold,
                "path": model_path.relative_to(output).as_posix(),
                "sha256": sha256_file(model_path),
                "size_bytes": model_path.stat().st_size,
            }
        )
        names = fitted.named_steps["preprocessor"].get_feature_names_out().tolist()
        lineage_rows.extend(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "scientific_input_hash": scientific_input_hash,
                "fold_contract_hash": folds.contract["fold_contract_hash"],
                "model": model_name,
                "outer_fold": outer_fold,
                "transformed_feature_index": index,
                "transformed_feature_name": str(name),
            }
            for index, name in enumerate(names)
        )
    pd.DataFrame(model_rows).to_csv(paths["model_index"], index=False)
    pd.DataFrame(lineage_rows).to_csv(paths["feature_lineage"], index=False)
    _write_json(
        paths["metadata"],
        {
            "stage": "model_benchmarks",
            "status": "complete",
            "run_id": run_id,
            "config_hash": config_hash,
            "scientific_input_hash": scientific_input_hash,
            "model_grid_sha256": model_grid_sha256,
            "fold_contract_hash": folds.contract["fold_contract_hash"],
            "selection_metric": settings["selection_metric"],
            "baseline_gate_metric": settings["baseline_gate_metric"],
            "models": list(CANONICAL_MODEL_NAMES),
            "candidate_counts": EXPECTED_CANDIDATE_COUNTS,
            "bootstrap": dict(result.bootstrap_metadata),
            "baseline_gate": dict(result.baseline_gate),
            "thread_determinism": thread_determinism_metadata(),
            "paid_api_calls": 0,
        },
    )
    return paths


__all__ = [
    "BENCHMARK_METRICS",
    "BenchmarkResult",
    "EXPECTED_CANDIDATE_COUNTS",
    "ModelBenchmarkError",
    "exact_primary_feature_frame",
    "evaluate_nested_benchmark",
    "run",
    "select_candidate_index",
    "thread_determinism_metadata",
    "validate_benchmark_config",
]
