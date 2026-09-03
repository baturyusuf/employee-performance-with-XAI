"""Run the additive v3 ordinal/baseline benchmark on canonical v2 folds.

The four nominal-model OOF predictions are reused as immutable v2 evidence.
Only the two ordinal models and three naive baselines are newly fitted, on the
exact persisted outer/inner assignments used by those nominal comparators.
Row-level OOF output remains local and is never part of the Git publication
surface; compact aggregate, per-class, and confusion evidence is exported by a
separate governed step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.data.canonical_loader import load_canonical_dataset, sha256_file
from src.experiments.manuscript_model_benchmark import select_candidate_index
from src.experiments.shared_folds import (
    SharedFoldArtifacts,
    read_shared_folds,
    validate_consumer_fold_assignments,
    validate_shared_folds,
)
from src.governance.feature_availability_contract import (
    validate_feature_availability_contract,
)
from src.governance.manuscript_contract import primary_excluded_features, source_tree_hash
from src.governance.ordinal_benchmark_contract_v3 import (
    DEFAULT_BENCHMARK_CONTRACT_PATH,
    validate_ordinal_benchmark_contract_v3,
)
from src.models.canonical_models import (
    CANONICAL_MODEL_NAMES,
    aligned_predict_proba,
    build_common_preprocessor,
    validate_model_feature_frame,
)
from src.models.evaluate import classification_metrics
from src.models.ordinal_evaluation_v3 import ordinal_evaluation_bundle_v3
from src.models.ordinal_models_v3 import (
    V3_NAIVE_BASELINE_NAMES,
    V3_ORDINAL_MODEL_NAMES,
    build_v3_naive_baseline,
    build_v3_ordinal_estimator,
)
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_CANONICAL_V2_ROOT = Path(
    "reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f"
)
DEFAULT_LOCAL_RUN_ROOT = Path("reports/major_revision_v3_runs")
FIT_THREAD_LIMIT = 1
EXTENSION_MODEL_NAMES = (*V3_ORDINAL_MODEL_NAMES, *V3_NAIVE_BASELINE_NAMES)


class V3OrdinalBenchmarkError(RuntimeError):
    """Raised when source, fold, fit, OOF, or publication invariants fail."""


@dataclass(frozen=True)
class V3OrdinalExtensionResult:
    candidate_search_results: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    evidence_status: str


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V3OrdinalBenchmarkError(f"Could not read {path.as_posix()}: {exc}") from exc
    if not isinstance(payload, dict):
        raise V3OrdinalBenchmarkError(f"{path.as_posix()} must contain a JSON object.")
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


def _clean_git_identity() -> dict[str, str]:
    """Return exact local Git identity or fail before scientific execution."""

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise V3OrdinalBenchmarkError(f"Could not establish Git identity: {exc}") from exc
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise V3OrdinalBenchmarkError("Git HEAD is not a full lowercase commit digest.")
    if status:
        raise V3OrdinalBenchmarkError(
            "Scientific execution requires a clean tracked/untracked worktree; "
            f"status={status.splitlines()[:10]}."
        )
    return {"commit": head, "branch": branch}


def _fit_or_fail(estimator: Any, X: pd.DataFrame, y: pd.Series, *, context: str) -> Any:
    try:
        with threadpool_limits(limits=FIT_THREAD_LIMIT):
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                return estimator.fit(X, y)
    except Exception as exc:
        raise V3OrdinalBenchmarkError(
            f"{context} failed: {type(exc).__name__}: {exc}"
        ) from exc


def _v3_pipeline(
    model_name: str,
    training_features: pd.DataFrame,
    *,
    fixed_parameters: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
    random_state: int,
) -> Pipeline:
    overlap = sorted(set(fixed_parameters).intersection(candidate_parameters))
    if overlap:
        raise V3OrdinalBenchmarkError(
            f"{model_name} candidate overwrites fixed parameters: {overlap}."
        )
    estimator = build_v3_ordinal_estimator(
        model_name,
        {**dict(fixed_parameters), **dict(candidate_parameters)},
        random_state=random_state,
    )
    return Pipeline(
        [
            ("preprocessor", build_common_preprocessor(training_features)),
            ("model", estimator),
        ]
    )


def _selection_scores(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    labels: Sequence[int],
) -> tuple[float, float]:
    metrics = classification_metrics(y_true, y_pred, y_proba, list(labels))
    macro_f1 = metrics.get("macro_f1")
    qwk = metrics.get("quadratic_weighted_kappa")
    if macro_f1 is None or qwk is None or not all(
        math.isfinite(float(value)) for value in (macro_f1, qwk)
    ):
        raise V3OrdinalBenchmarkError("Inner selection metrics are unavailable or non-finite.")
    return float(macro_f1), float(qwk)


def exact_p3_feature_frame(
    frame: pd.DataFrame,
    contract: Mapping[str, Any],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Materialize raw INX columns minus the exact P3 contract exclusions."""

    policies = contract.get("policies")
    if not isinstance(policies, list):
        raise V3OrdinalBenchmarkError("Feature contract has no policy list.")
    p3_rows = [row for row in policies if isinstance(row, Mapping) and row.get("policy_id") == "P3"]
    if len(p3_rows) != 1:
        raise V3OrdinalBenchmarkError("Feature contract must contain exactly one P3 policy.")
    exclusions = tuple(str(value) for value in p3_rows[0].get("excluded_features", ()))
    if not exclusions or len(set(exclusions)) != len(exclusions):
        raise V3OrdinalBenchmarkError("P3 exclusions must be non-empty and unique.")
    unknown = sorted(set(exclusions).difference(map(str, frame.columns)))
    if unknown:
        raise V3OrdinalBenchmarkError(f"P3 exclusions are absent from the dataset: {unknown}.")
    retained = [str(column) for column in frame.columns if str(column) not in set(exclusions)]
    features = frame.loc[:, retained].copy()
    validate_model_feature_frame(features, forbidden_features=exclusions)
    return features, exclusions


def _validate_fold_alignment(
    features: pd.DataFrame,
    target: pd.Series,
    folds: SharedFoldArtifacts,
    *,
    labels: Sequence[int],
) -> None:
    validate_shared_folds(folds)
    if not features.index.is_unique or not target.index.is_unique:
        raise V3OrdinalBenchmarkError("Feature and target sample indices must be unique.")
    if not features.index.equals(target.index):
        raise V3OrdinalBenchmarkError("Feature and target indices must align exactly.")
    outer = folds.outer_assignments
    expected_samples = set(outer["sample_index"].astype(int))
    if set(int(value) for value in features.index) != expected_samples:
        raise V3OrdinalBenchmarkError("Feature samples differ from persisted outer folds.")
    persisted = outer.set_index("sample_index")["y_true"].sort_index().astype(int)
    observed = target.sort_index().astype(int)
    if not np.array_equal(persisted.to_numpy(), observed.to_numpy()):
        raise V3OrdinalBenchmarkError("Targets differ from persisted fold labels.")
    if set(observed.unique()) != set(map(int, labels)):
        raise V3OrdinalBenchmarkError("Observed target support differs from ordered labels.")


def evaluate_ordinal_extension_v3(
    features: pd.DataFrame,
    target: pd.Series,
    folds: SharedFoldArtifacts,
    benchmark_contract: Mapping[str, Any],
    *,
    run_id: str,
    benchmark_contract_sha256: str,
    scientific_input_sha256: str,
    outer_fold_subset: Sequence[int] | None = None,
) -> V3OrdinalExtensionResult:
    """Fit the two ordinal models and three baselines on supplied shared folds.

    ``outer_fold_subset`` exists only for isolated diagnostics/tests. Any subset
    receives an explicit inadmissible status and cannot be mistaken for a full
    exactly-once OOF result.
    """

    if not str(run_id).strip():
        raise V3OrdinalBenchmarkError("run_id must be non-empty.")
    for name, digest in (
        ("benchmark_contract_sha256", benchmark_contract_sha256),
        ("scientific_input_sha256", scientific_input_sha256),
    ):
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise V3OrdinalBenchmarkError(f"{name} must be a lowercase SHA-256 digest.")
    labels = tuple(int(value) for value in benchmark_contract.get("ordered_labels", ()))
    if labels != (2, 3, 4):
        raise V3OrdinalBenchmarkError("The extension ordered labels must equal (2, 3, 4).")
    _validate_fold_alignment(features, target, folds, labels=labels)
    outer = folds.outer_assignments.copy()
    inner = folds.inner_assignments.copy()
    all_outer_folds = tuple(sorted(outer["outer_fold"].astype(int).unique()))
    declared_outer_splits = int(benchmark_contract["shared_nested_cv"]["outer_splits"])
    if int(folds.contract.get("outer_splits", -1)) != declared_outer_splits or len(
        all_outer_folds
    ) != declared_outer_splits:
        raise V3OrdinalBenchmarkError("Persisted outer-fold count differs from the v3 contract.")
    if outer_fold_subset is None:
        selected_outer_folds = all_outer_folds
        evidence_status = "complete_exactly_once_oof"
    else:
        selected_outer_folds = tuple(sorted(set(map(int, outer_fold_subset))))
        if not selected_outer_folds or not set(selected_outer_folds).issubset(all_outer_folds):
            raise V3OrdinalBenchmarkError("Diagnostic outer-fold subset is invalid.")
        evidence_status = "diagnostic_incomplete_never_canonical"

    settings = benchmark_contract["selection"]
    tolerance = float(settings["primary_tie_tolerance"])
    model_seed = int(benchmark_contract["shared_nested_cv"]["model_seed"])
    inner_splits = int(benchmark_contract["shared_nested_cv"]["inner_splits"])
    if int(folds.contract.get("inner_splits", -1)) != inner_splits:
        raise V3OrdinalBenchmarkError("Persisted inner-fold count differs from the v3 contract.")
    ordinal_definitions = benchmark_contract["ordinal_models"]
    fold_contract_hash = str(folds.contract["fold_contract_hash"])
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for outer_fold in selected_outer_folds:
        test_ids = outer.loc[
            outer["outer_fold"].astype(int) == outer_fold, "sample_index"
        ].astype(int).tolist()
        train_ids = outer.loc[
            outer["outer_fold"].astype(int) != outer_fold, "sample_index"
        ].astype(int).tolist()
        scoped_inner = inner[inner["outer_fold"].astype(int) == outer_fold].copy()
        if set(scoped_inner["sample_index"].astype(int)) != set(train_ids):
            raise V3OrdinalBenchmarkError(
                f"Inner assignments at outer fold {outer_fold} differ from outer training data."
            )

        for model_name in V3_ORDINAL_MODEL_NAMES:
            definition = ordinal_definitions[model_name]
            fixed = dict(definition["fixed_params"])
            candidates = [dict(value) for value in definition["candidates"]]
            candidate_macro_f1: list[float] = []
            candidate_qwk: list[float] = []
            scoped_candidate_rows: list[dict[str, Any]] = []
            for candidate_index, candidate in enumerate(candidates):
                inner_macro_f1: list[float] = []
                inner_qwk: list[float] = []
                for inner_fold in range(1, inner_splits + 1):
                    validation_ids = scoped_inner.loc[
                        scoped_inner["inner_fold"].astype(int) == inner_fold,
                        "sample_index",
                    ].astype(int).tolist()
                    development_ids = sorted(set(train_ids) - set(validation_ids))
                    if not validation_ids or not development_ids:
                        raise V3OrdinalBenchmarkError(
                            f"Empty inner partition at outer={outer_fold}, inner={inner_fold}."
                        )
                    pipeline = _v3_pipeline(
                        model_name,
                        features.loc[development_ids],
                        fixed_parameters=fixed,
                        candidate_parameters=candidate,
                        random_state=model_seed,
                    )
                    _fit_or_fail(
                        pipeline,
                        features.loc[development_ids],
                        target.loc[development_ids],
                        context=(
                            f"model={model_name}, outer={outer_fold}, "
                            f"candidate={candidate_index}, inner={inner_fold}"
                        ),
                    )
                    prediction = np.asarray(
                        pipeline.predict(features.loc[validation_ids]), dtype=int
                    )
                    probability = aligned_predict_proba(
                        pipeline, features.loc[validation_ids], labels=labels
                    )
                    macro_f1, qwk = _selection_scores(
                        target.loc[validation_ids],
                        prediction,
                        probability,
                        labels=labels,
                    )
                    inner_macro_f1.append(macro_f1)
                    inner_qwk.append(qwk)
                candidate_macro_f1.append(float(np.mean(inner_macro_f1)))
                candidate_qwk.append(float(np.mean(inner_qwk)))
                scoped_candidate_rows.append(
                    {
                        "run_id": run_id,
                        "benchmark_contract_sha256": benchmark_contract_sha256,
                        "scientific_input_sha256": scientific_input_sha256,
                        "fold_contract_hash": fold_contract_hash,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "candidate_index": candidate_index,
                        "parameters_json": json.dumps(
                            candidate, sort_keys=True, separators=(",", ":")
                        ),
                        "inner_macro_f1_scores_json": json.dumps(inner_macro_f1),
                        "inner_macro_f1_mean": float(np.mean(inner_macro_f1)),
                        "inner_qwk_scores_json": json.dumps(inner_qwk),
                        "inner_qwk_mean": float(np.mean(inner_qwk)),
                        "n_inner_folds": inner_splits,
                        "outer_test_used_for_selection": False,
                        "candidate_status": "complete",
                    }
                )
            selected_index = select_candidate_index(
                candidate_macro_f1,
                candidate_qwk,
                practical_tie_tolerance=tolerance,
                better_direction="higher",
            )
            for row in scoped_candidate_rows:
                row["selected_by_protocol"] = row["candidate_index"] == selected_index
            candidate_rows.extend(scoped_candidate_rows)
            selected_candidate = candidates[selected_index]
            pipeline = _v3_pipeline(
                model_name,
                features.loc[train_ids],
                fixed_parameters=fixed,
                candidate_parameters=selected_candidate,
                random_state=model_seed,
            )
            _fit_or_fail(
                pipeline,
                features.loc[train_ids],
                target.loc[train_ids],
                context=f"outer refit model={model_name}, outer={outer_fold}",
            )
            prediction = np.asarray(pipeline.predict(features.loc[test_ids]), dtype=int)
            probability = aligned_predict_proba(
                pipeline, features.loc[test_ids], labels=labels
            )
            selected_rows.append(
                {
                    "run_id": run_id,
                    "benchmark_contract_sha256": benchmark_contract_sha256,
                    "scientific_input_sha256": scientific_input_sha256,
                    "fold_contract_hash": fold_contract_hash,
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "selection_performed": True,
                    "selected_candidate_index": selected_index,
                    "selected_candidate_parameters_json": json.dumps(
                        selected_candidate, sort_keys=True, separators=(",", ":")
                    ),
                    "fixed_parameters_json": json.dumps(
                        fixed, sort_keys=True, separators=(",", ":")
                    ),
                    "selected_inner_macro_f1_mean": candidate_macro_f1[selected_index],
                    "selected_inner_qwk_mean": candidate_qwk[selected_index],
                    "outer_test_used_for_selection": False,
                }
            )
            _append_fold_and_predictions(
                fold_rows,
                prediction_rows,
                run_id=run_id,
                benchmark_contract_sha256=benchmark_contract_sha256,
                scientific_input_sha256=scientific_input_sha256,
                fold_contract_hash=fold_contract_hash,
                outer_fold=outer_fold,
                model_name=model_name,
                test_ids=test_ids,
                target=target,
                prediction=prediction,
                probability=probability,
                labels=labels,
                selected_candidate_index=selected_index,
            )

        for baseline_name in V3_NAIVE_BASELINE_NAMES:
            baseline = build_v3_naive_baseline(
                baseline_name, random_state=model_seed
            )
            _fit_or_fail(
                baseline,
                features.loc[train_ids],
                target.loc[train_ids],
                context=f"baseline={baseline_name}, outer={outer_fold}",
            )
            prediction = np.asarray(baseline.predict(features.loc[test_ids]), dtype=int)
            probability = aligned_predict_proba(
                baseline, features.loc[test_ids], labels=labels
            )
            selected_rows.append(
                {
                    "run_id": run_id,
                    "benchmark_contract_sha256": benchmark_contract_sha256,
                    "scientific_input_sha256": scientific_input_sha256,
                    "fold_contract_hash": fold_contract_hash,
                    "outer_fold": outer_fold,
                    "model": baseline_name,
                    "selection_performed": False,
                    "selected_candidate_index": None,
                    "selected_candidate_parameters_json": "{}",
                    "fixed_parameters_json": json.dumps(
                        {"strategy": baseline.strategy}, separators=(",", ":")
                    ),
                    "selected_inner_macro_f1_mean": None,
                    "selected_inner_qwk_mean": None,
                    "outer_test_used_for_selection": False,
                }
            )
            _append_fold_and_predictions(
                fold_rows,
                prediction_rows,
                run_id=run_id,
                benchmark_contract_sha256=benchmark_contract_sha256,
                scientific_input_sha256=scientific_input_sha256,
                fold_contract_hash=fold_contract_hash,
                outer_fold=outer_fold,
                model_name=baseline_name,
                test_ids=test_ids,
                target=target,
                prediction=prediction,
                probability=probability,
                labels=labels,
                selected_candidate_index=None,
            )

    oof = pd.DataFrame(prediction_rows).sort_values(
        ["model", "sample_index"]
    ).reset_index(drop=True)
    expected_ids = set(
        outer.loc[outer["outer_fold"].isin(selected_outer_folds), "sample_index"].astype(int)
    )
    _validate_extension_oof(
        oof,
        expected_sample_ids=expected_ids,
        folds=outer,
        labels=labels,
        full_run=outer_fold_subset is None,
    )
    return V3OrdinalExtensionResult(
        candidate_search_results=pd.DataFrame(candidate_rows),
        selected_hyperparameters=pd.DataFrame(selected_rows),
        fold_metrics=pd.DataFrame(fold_rows),
        oof_predictions=oof,
        evidence_status=evidence_status,
    )


def _append_fold_and_predictions(
    fold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    run_id: str,
    benchmark_contract_sha256: str,
    scientific_input_sha256: str,
    fold_contract_hash: str,
    outer_fold: int,
    model_name: str,
    test_ids: Sequence[int],
    target: pd.Series,
    prediction: np.ndarray,
    probability: np.ndarray,
    labels: Sequence[int],
    selected_candidate_index: int | None,
) -> None:
    bundle = ordinal_evaluation_bundle_v3(
        target.loc[list(test_ids)],
        prediction,
        probability,
        labels=labels,
        dataset_key="inx_primary",
        model_name=model_name,
    )
    fold_rows.append(
        {
            "run_id": run_id,
            "benchmark_contract_sha256": benchmark_contract_sha256,
            "scientific_input_sha256": scientific_input_sha256,
            "fold_contract_hash": fold_contract_hash,
            "outer_fold": int(outer_fold),
            "model": model_name,
            "n_test": len(test_ids),
            **bundle["aggregate_metrics"],
        }
    )
    for position, sample_index in enumerate(test_ids):
        row = {
            "run_id": run_id,
            "benchmark_contract_sha256": benchmark_contract_sha256,
            "scientific_input_sha256": scientific_input_sha256,
            "fold_contract_hash": fold_contract_hash,
            "evidence_source": "v3_new_outer_fold_fit",
            "model": model_name,
            "sample_index": int(sample_index),
            "outer_fold": int(outer_fold),
            "y_true": int(target.loc[sample_index]),
            "y_pred": int(prediction[position]),
            "selected_candidate_index": selected_candidate_index,
        }
        row.update(
            {
                f"prob_class_{label}": float(probability[position, column])
                for column, label in enumerate(labels)
            }
        )
        prediction_rows.append(row)


def _validate_extension_oof(
    oof: pd.DataFrame,
    *,
    expected_sample_ids: set[int],
    folds: pd.DataFrame,
    labels: Sequence[int],
    full_run: bool,
) -> None:
    if oof.empty or set(oof["model"].unique()) != set(EXTENSION_MODEL_NAMES):
        raise V3OrdinalBenchmarkError("Extension OOF model coverage is incomplete.")
    fold_lookup = folds.set_index("sample_index")["outer_fold"].astype(int)
    for model_name, rows in oof.groupby("model", sort=False):
        ids = rows["sample_index"].astype(int)
        if ids.duplicated().any() or set(ids) != expected_sample_ids:
            raise V3OrdinalBenchmarkError(
                f"OOF sample coverage is invalid for {model_name}."
            )
        observed_folds = rows.set_index("sample_index")["outer_fold"].astype(int).sort_index()
        expected_folds = fold_lookup.loc[list(expected_sample_ids)].sort_index()
        if not observed_folds.equals(expected_folds):
            raise V3OrdinalBenchmarkError(
                f"OOF fold assignments drifted for {model_name}."
            )
        probability = rows[[f"prob_class_{label}" for label in labels]].to_numpy(float)
        if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-9):
            raise V3OrdinalBenchmarkError(
                f"OOF probability rows do not sum to one for {model_name}."
            )
        predicted_from_probability = np.asarray(labels)[np.argmax(probability, axis=1)]
        if not np.array_equal(predicted_from_probability, rows["y_pred"].to_numpy(int)):
            raise V3OrdinalBenchmarkError(
                f"OOF labels disagree with probability argmax for {model_name}."
            )
    if full_run and len(expected_sample_ids) != len(folds):
        raise V3OrdinalBenchmarkError("Full OOF run did not cover every persisted sample.")


def summarize_combined_oof_v3(
    combined_oof: pd.DataFrame,
    *,
    labels: Sequence[int] = (2, 3, 4),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build aggregate, per-class, and confusion records for all nine comparators."""

    required = {
        "model",
        "sample_index",
        "outer_fold",
        "y_true",
        "y_pred",
        *(f"prob_class_{label}" for label in labels),
    }
    missing = sorted(required.difference(combined_oof.columns))
    if missing:
        raise V3OrdinalBenchmarkError(f"Combined OOF evidence lacks columns: {missing}.")
    expected_models = {*CANONICAL_MODEL_NAMES, *EXTENSION_MODEL_NAMES}
    if set(combined_oof["model"].unique()) != expected_models:
        raise V3OrdinalBenchmarkError("Combined OOF evidence must contain exactly nine models.")
    aggregate_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    reference_ids: set[int] | None = None
    reference_target: pd.Series | None = None
    for model_name, rows in combined_oof.groupby("model", sort=True):
        rows = rows.sort_values("sample_index")
        ids = rows["sample_index"].astype(int)
        if ids.duplicated().any():
            raise V3OrdinalBenchmarkError(f"Duplicate OOF samples for {model_name}.")
        if reference_ids is None:
            reference_ids = set(ids)
            reference_target = rows.set_index("sample_index")["y_true"].astype(int).sort_index()
        elif set(ids) != reference_ids:
            raise V3OrdinalBenchmarkError("Combined models do not share exact OOF samples.")
        target = rows.set_index("sample_index")["y_true"].astype(int).sort_index()
        if reference_target is None or not target.equals(reference_target):
            raise V3OrdinalBenchmarkError("Combined models do not share exact OOF targets.")
        probability = rows[[f"prob_class_{label}" for label in labels]].to_numpy(float)
        if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-9):
            raise V3OrdinalBenchmarkError(
                f"Combined OOF probabilities do not sum to one for {model_name}."
            )
        probability_prediction = np.asarray(labels)[np.argmax(probability, axis=1)]
        if not np.array_equal(
            probability_prediction, rows["y_pred"].to_numpy(dtype=int)
        ):
            raise V3OrdinalBenchmarkError(
                f"Combined OOF labels disagree with probability argmax for {model_name}."
            )
        bundle = ordinal_evaluation_bundle_v3(
            rows["y_true"].astype(int),
            rows["y_pred"].astype(int),
            probability,
            labels=labels,
            dataset_key="inx_primary",
            model_name=str(model_name),
        )
        aggregate_rows.extend(
            {
                "dataset_key": "inx_primary",
                "model_name": str(model_name),
                "metric": metric,
                "value": value,
            }
            for metric, value in bundle["aggregate_metrics"].items()
        )
        per_class_rows.extend(bundle["per_class_metrics"])
        confusion_rows.extend(bundle["confusion_matrix"])
    return (
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(per_class_rows),
        pd.DataFrame(confusion_rows),
    )


def _validate_and_load_v2_sources(
    canonical_v2_root: Path,
    benchmark_contract: Mapping[str, Any],
) -> tuple[SharedFoldArtifacts, pd.DataFrame, dict[str, str]]:
    comparison = benchmark_contract["canonical_v2_comparison_source"]
    artifact_hashes: dict[str, str] = {}
    for key in (
        "fold_contract",
        "outer_assignments",
        "inner_assignments",
        "nominal_oof_predictions",
    ):
        record = comparison[key]
        path = canonical_v2_root / str(record["path"])
        if not path.is_file():
            raise V3OrdinalBenchmarkError(
                f"Required canonical v2 source is absent: {path.as_posix()}."
            )
        observed = sha256_file(path)
        if observed != record["sha256"]:
            raise V3OrdinalBenchmarkError(f"Canonical v2 source hash drifted for {key}.")
        artifact_hashes[key] = observed
    shared_folds_dir = canonical_v2_root / "core" / "shared_folds"
    folds = read_shared_folds(shared_folds_dir)
    validate_shared_folds(folds)
    if folds.contract.get("run_id") != comparison["run_id"]:
        raise V3OrdinalBenchmarkError("Canonical v2 fold run identity drifted.")
    nominal_oof = pd.read_csv(
        canonical_v2_root / comparison["nominal_oof_predictions"]["path"]
    )
    if set(nominal_oof["model"].unique()) != set(CANONICAL_MODEL_NAMES):
        raise V3OrdinalBenchmarkError("Canonical v2 nominal OOF model set drifted.")
    validate_consumer_fold_assignments(
        folds,
        nominal_oof,
        group_columns=("model",),
    )
    labels = (2, 3, 4)
    for model_name, rows in nominal_oof.groupby("model", sort=False):
        probability = rows[[f"prob_class_{label}" for label in labels]].to_numpy(float)
        if not np.all(np.isfinite(probability)) or not np.allclose(
            probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-6
        ):
            raise V3OrdinalBenchmarkError(
                f"Canonical v2 nominal probabilities are invalid for {model_name}."
            )
        predicted = np.asarray(labels)[np.argmax(probability, axis=1)]
        if not np.array_equal(predicted, rows["y_pred"].to_numpy(dtype=int)):
            raise V3OrdinalBenchmarkError(
                f"Canonical v2 nominal labels disagree with probability argmax for {model_name}."
            )
    return folds, nominal_oof, artifact_hashes


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def preflight_ordinal_benchmark_v3(
    *,
    contract_path: Path | str = DEFAULT_BENCHMARK_CONTRACT_PATH,
    canonical_v2_root: Path | str = DEFAULT_CANONICAL_V2_ROOT,
) -> dict[str, Any]:
    """Validate every local source and alignment invariant without fitting a model."""

    contract_file = Path(contract_path)
    contract_receipt = validate_ordinal_benchmark_contract_v3(contract_file)
    contract = _load_json(contract_file)
    folds, nominal_oof, v2_artifact_hashes = _validate_and_load_v2_sources(
        Path(canonical_v2_root), contract
    )
    data_source = contract["data_source"]
    canonical = load_canonical_dataset(
        data_source["canonical_loader_config_path"],
        "inx_primary",
        data_source["acquisition_manifest_path"],
        allow_download=False,
    )
    feature_contract_path = Path(contract["information_contract"]["path"])
    feature_contract = _load_json(feature_contract_path)
    features, exclusions = exact_p3_feature_frame(canonical.frame, feature_contract)
    v2_config = load_config(data_source["canonical_loader_config_path"])
    if set(exclusions) != set(primary_excluded_features(v2_config)):
        raise V3OrdinalBenchmarkError(
            "V3 P3 exclusions differ from the canonical v2 primary feature policy."
        )
    target = canonical.frame[contract["target"]].astype(int)
    _validate_fold_alignment(
        features,
        target,
        folds,
        labels=tuple(contract["ordered_labels"]),
    )
    if canonical.receipt["actual_sha256"] != folds.contract["dataset_sha256"]:
        raise V3OrdinalBenchmarkError("Current dataset hash differs from canonical v2 folds.")
    return {
        "status": "passed",
        "model_fit_count": 0,
        "benchmark_contract_sha256": contract_receipt["contract_sha256"],
        "dataset_sha256": canonical.receipt["actual_sha256"],
        "feature_count": int(features.shape[1]),
        "sample_count": int(len(features)),
        "target_distribution": canonical.receipt["target_distribution"],
        "outer_folds": int(folds.contract["outer_splits"]),
        "inner_folds": int(folds.contract["inner_splits"]),
        "fold_contract_hash": folds.contract["fold_contract_hash"],
        "nominal_oof_row_count": int(len(nominal_oof)),
        "nominal_model_count": int(nominal_oof["model"].nunique()),
        "canonical_v2_artifact_hashes": v2_artifact_hashes,
        "paid_api_calls": 0,
        "network_calls": 0,
    }


def run_ordinal_benchmark_v3(
    *,
    contract_path: Path | str = DEFAULT_BENCHMARK_CONTRACT_PATH,
    canonical_v2_root: Path | str = DEFAULT_CANONICAL_V2_ROOT,
    output_dir: Path | str,
    run_id: str,
) -> dict[str, Any]:
    """Execute and atomically publish the complete local nine-model v3 benchmark."""

    contract_file = Path(contract_path)
    git_identity = _clean_git_identity()
    contract_receipt = validate_ordinal_benchmark_contract_v3(contract_file)
    contract = _load_json(contract_file)
    destination = Path(output_dir)
    if destination.exists():
        raise V3OrdinalBenchmarkError(f"Output destination already exists: {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        folds, nominal_oof, v2_artifact_hashes = _validate_and_load_v2_sources(
            Path(canonical_v2_root), contract
        )
        data_source = contract["data_source"]
        canonical = load_canonical_dataset(
            data_source["canonical_loader_config_path"],
            "inx_primary",
            data_source["acquisition_manifest_path"],
            allow_download=False,
        )
        feature_contract_path = Path(contract["information_contract"]["path"])
        validate_feature_availability_contract(feature_contract_path)
        feature_contract = _load_json(feature_contract_path)
        features, exclusions = exact_p3_feature_frame(canonical.frame, feature_contract)
        v2_config = load_config(data_source["canonical_loader_config_path"])
        if set(exclusions) != set(primary_excluded_features(v2_config)):
            raise V3OrdinalBenchmarkError(
                "V3 P3 exclusions differ from the canonical v2 primary feature policy."
            )
        target = canonical.frame[contract["target"]].astype(int)
        if canonical.receipt["actual_sha256"] != folds.contract["dataset_sha256"]:
            raise V3OrdinalBenchmarkError("Current dataset hash differs from canonical v2 folds.")

        implementation_paths = (
            Path("src/models/ordinal_models_v3.py"),
            Path("src/models/ordinal_evaluation_v3.py"),
            Path("src/experiments/ordinal_benchmark_v3.py"),
        )
        implementation_hashes = {
            path.as_posix(): sha256_file(path) for path in implementation_paths
        }
        current_source_tree_hash = source_tree_hash(PROJECT_ROOT)
        scientific_inputs = {
            "git_identity": git_identity,
            "source_tree_hash": current_source_tree_hash,
            "benchmark_contract_sha256": contract_receipt["contract_sha256"],
            "dataset_sha256": canonical.receipt["actual_sha256"],
            "feature_contract_sha256": contract_receipt["information_contract_sha256"],
            "nominal_model_registry_sha256": contract_receipt[
                "nominal_model_registry_sha256"
            ],
            "canonical_v2_artifact_hashes": v2_artifact_hashes,
            "implementation_hashes": implementation_hashes,
        }
        scientific_input_sha256 = _canonical_json_sha256(scientific_inputs)
        result = evaluate_ordinal_extension_v3(
            features,
            target,
            folds,
            contract,
            run_id=run_id,
            benchmark_contract_sha256=contract_receipt["contract_sha256"],
            scientific_input_sha256=scientific_input_sha256,
        )

        nominal = nominal_oof.copy()
        nominal["evidence_source"] = "canonical_v2_reused_without_refit_or_relabelling"
        nominal["benchmark_contract_sha256"] = contract_receipt["contract_sha256"]
        nominal["v3_scientific_input_sha256"] = scientific_input_sha256
        extension = result.oof_predictions.copy()
        combined_columns = [
            "model",
            "sample_index",
            "outer_fold",
            "y_true",
            "y_pred",
            "prob_class_2",
            "prob_class_3",
            "prob_class_4",
            "evidence_source",
        ]
        combined = pd.concat(
            [nominal[combined_columns], extension[combined_columns]],
            ignore_index=True,
        ).sort_values(["model", "sample_index"]).reset_index(drop=True)
        aggregate, per_class, confusion = summarize_combined_oof_v3(combined)

        paths = {
            "candidate_search_results": staging / "candidate_search_results.csv",
            "selected_hyperparameters": staging / "selected_hyperparameters.csv",
            "fold_metrics": staging / "extension_fold_metrics.csv",
            "extension_oof_predictions": staging / "extension_oof_predictions.csv",
            "combined_oof_predictions": staging / "combined_oof_predictions.csv",
            "aggregate_metrics": staging / "aggregate_metrics.csv",
            "per_class_metrics": staging / "per_class_metrics.csv",
            "confusion_matrix": staging / "confusion_matrix.csv",
            "metadata": staging / "stage_metadata.json",
        }
        result.candidate_search_results.to_csv(paths["candidate_search_results"], index=False)
        result.selected_hyperparameters.to_csv(paths["selected_hyperparameters"], index=False)
        result.fold_metrics.to_csv(paths["fold_metrics"], index=False)
        result.oof_predictions.to_csv(paths["extension_oof_predictions"], index=False)
        combined.to_csv(paths["combined_oof_predictions"], index=False)
        aggregate.to_csv(paths["aggregate_metrics"], index=False)
        per_class.to_csv(paths["per_class_metrics"], index=False)
        confusion.to_csv(paths["confusion_matrix"], index=False)
        output_hashes = {
            name: sha256_file(path) for name, path in paths.items() if name != "metadata"
        }
        metadata = {
            "schema_version": 1,
            "stage": "ordinal_benchmark_v3",
            "status": "complete",
            "evidence_status": result.evidence_status,
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark_contract_sha256": contract_receipt["contract_sha256"],
            "scientific_input_sha256": scientific_input_sha256,
            "scientific_inputs": scientific_inputs,
            "git_identity": git_identity,
            "source_tree_hash": current_source_tree_hash,
            "fold_contract_hash": folds.contract["fold_contract_hash"],
            "feature_policy": "P3",
            "feature_count": int(features.shape[1]),
            "ordered_labels": [2, 3, 4],
            "nominal_models_reused": list(CANONICAL_MODEL_NAMES),
            "new_ordinal_models_fitted": list(V3_ORDINAL_MODEL_NAMES),
            "new_naive_baselines_fitted": list(V3_NAIVE_BASELINE_NAMES),
            "model_count": 9,
            "sample_count_per_model": len(features),
            "outer_folds": int(folds.contract["outer_splits"]),
            "inner_folds": int(folds.contract["inner_splits"]),
            "outer_test_used_for_selection": False,
            "paid_api_calls": 0,
            "network_calls": 0,
            "employee_level_outputs_publication_authorized": False,
            "output_hashes": output_hashes,
        }
        if _clean_git_identity() != git_identity:
            raise V3OrdinalBenchmarkError("Git identity changed during scientific execution.")
        if source_tree_hash(PROJECT_ROOT) != current_source_tree_hash:
            raise V3OrdinalBenchmarkError("Scientific source tree changed during execution.")
        if sha256_file(contract_file) != contract_receipt["contract_sha256"]:
            raise V3OrdinalBenchmarkError("Benchmark contract changed during execution.")
        dataset_path = Path(str(canonical.receipt["actual_path"]))
        if not dataset_path.is_absolute():
            dataset_path = PROJECT_ROOT / dataset_path
        if sha256_file(dataset_path) != canonical.receipt["actual_sha256"]:
            raise V3OrdinalBenchmarkError("Canonical dataset changed during execution.")
        for key, expected_hash in v2_artifact_hashes.items():
            source_path = Path(canonical_v2_root) / contract[
                "canonical_v2_comparison_source"
            ][key]["path"]
            if sha256_file(source_path) != expected_hash:
                raise V3OrdinalBenchmarkError(
                    f"Canonical v2 source changed during execution: {key}."
                )
        _atomic_write_json(paths["metadata"], metadata)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            for child in sorted(staging.iterdir(), reverse=True):
                if child.is_file():
                    child.unlink()
            staging.rmdir()
        raise
    return {
        "status": "complete",
        "run_id": run_id,
        "output_dir": destination.as_posix(),
        "benchmark_contract_sha256": contract_receipt["contract_sha256"],
        "scientific_input_sha256": scientific_input_sha256,
        "model_count": 9,
        "sample_count_per_model": len(features),
        "paid_api_calls": 0,
        "network_calls": 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_BENCHMARK_CONTRACT_PATH)
    parser.add_argument("--canonical-v2-root", type=Path, default=DEFAULT_CANONICAL_V2_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_LOCAL_RUN_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.preflight_only:
        receipt = preflight_ordinal_benchmark_v3(
            contract_path=args.contract,
            canonical_v2_root=args.canonical_v2_root,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if not isinstance(args.run_id, str) or not args.run_id.strip():
        raise V3OrdinalBenchmarkError("--run-id is required unless --preflight-only is used.")
    output = args.output_root / args.run_id / "ordinal_benchmark"
    receipt = run_ordinal_benchmark_v3(
        contract_path=args.contract,
        canonical_v2_root=args.canonical_v2_root,
        output_dir=output,
        run_id=args.run_id,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CANONICAL_V2_ROOT",
    "DEFAULT_LOCAL_RUN_ROOT",
    "EXTENSION_MODEL_NAMES",
    "V3OrdinalBenchmarkError",
    "V3OrdinalExtensionResult",
    "evaluate_ordinal_extension_v3",
    "exact_p3_feature_frame",
    "preflight_ordinal_benchmark_v3",
    "run_ordinal_benchmark_v3",
    "summarize_combined_oof_v3",
]
