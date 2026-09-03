"""Run the bounded v3 5×5×5 repeated nested-CV sensitivity.

Unlike the Phase 1B common-fold comparison, this sensitivity refits all four
nominal models, both ordinal models, and three naive baselines in each of five
prespecified repetitions. Row-level predictions and fold assignments remain in
the ignored local run root; only a later governed compact export may be tracked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
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
from src.experiments.manuscript_model_benchmark import (
    select_candidate_index,
    validate_benchmark_config,
)
from src.experiments.ordinal_benchmark_v3 import (
    exact_p3_feature_frame,
    summarize_combined_oof_v3,
)
from src.experiments.shared_folds import (
    SharedFoldArtifacts,
    generate_shared_folds,
    validate_shared_folds,
)
from src.governance.feature_availability_contract import (
    validate_feature_availability_contract,
)
from src.governance.manuscript_contract import primary_excluded_features, source_tree_hash
from src.governance.offline_runtime import enforce_offline_runtime
from src.governance.repeated_nested_cv_contract_v3 import (
    DEFAULT_REPEATED_CV_CONTRACT_PATH,
    PRIORITY_METRICS,
    validate_repeated_nested_cv_contract_v3,
)
from src.models.canonical_models import (
    CANONICAL_MODEL_NAMES,
    aligned_predict_proba,
    build_common_preprocessor,
    build_model_pipeline,
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


DEFAULT_LOCAL_RUN_ROOT = Path("reports/major_revision_v3_runs")
FIT_THREAD_LIMIT = 1
TUNED_MODEL_NAMES = (*CANONICAL_MODEL_NAMES, *V3_ORDINAL_MODEL_NAMES)
ALL_MODEL_NAMES = (*TUNED_MODEL_NAMES, *V3_NAIVE_BASELINE_NAMES)
EXPECTED_LOCAL_FILES = frozenset(
    {
        "candidate_search_results.csv",
        "fold_contracts.json",
        "fold_metrics.csv",
        "model_rank_summary.csv",
        "oof_predictions.csv",
        "ordering_stability.csv",
        "rank_by_repetition.csv",
        "repetition_metrics.csv",
        "selected_candidate_frequency.csv",
        "selected_hyperparameters.csv",
        "stage_metadata.json",
        "variability_summary.csv",
    }
)


class RepeatedNestedCVError(RuntimeError):
    """Raised when a repeated-CV source, fold, fit, or output invariant fails."""


@dataclass(frozen=True)
class RepeatedNestedCVResult:
    candidate_search_results: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    repetition_metrics: pd.DataFrame
    variability_summary: pd.DataFrame
    rank_by_repetition: pd.DataFrame
    model_rank_summary: pd.DataFrame
    ordering_stability: pd.DataFrame
    selected_candidate_frequency: pd.DataFrame
    fold_contracts: tuple[Mapping[str, Any], ...]
    evidence_status: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RepeatedNestedCVError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RepeatedNestedCVError(f"Could not read {path.as_posix()}: {exc}") from exc
    _require(isinstance(payload, dict), f"{path.as_posix()} must contain a JSON object.")
    return payload


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_digest(value: str, *, length: int = 64) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _clean_git_identity() -> dict[str, str]:
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
        raise RepeatedNestedCVError(f"Could not establish Git identity: {exc}") from exc
    _require(
        _valid_digest(head, length=40),
        "Git HEAD must be a full lowercase commit digest.",
    )
    _require(
        not status,
        "Scientific execution requires a clean worktree; "
        f"status={status.splitlines()[:10]}.",
    )
    return {"commit": head, "branch": branch}


def _fit_or_fail(estimator: Any, X: pd.DataFrame, y: pd.Series, *, context: str) -> Any:
    try:
        with threadpool_limits(limits=FIT_THREAD_LIMIT):
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                return estimator.fit(X, y)
    except Exception as exc:
        raise RepeatedNestedCVError(
            f"{context} failed: {type(exc).__name__}: {exc}"
        ) from exc


def _model_definitions(
    nominal_config: Mapping[str, Any],
    ordinal_config: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    nominal_settings = validate_benchmark_config(nominal_config)
    definitions: dict[str, Mapping[str, Any]] = {
        name: nominal_settings["models"][name] for name in CANONICAL_MODEL_NAMES
    }
    ordinal = ordinal_config.get("ordinal_models")
    _require(isinstance(ordinal, Mapping), "Ordinal model definitions are absent.")
    definitions.update({name: ordinal[name] for name in V3_ORDINAL_MODEL_NAMES})
    _require(set(definitions) == set(TUNED_MODEL_NAMES), "Tuned model registry drifted.")
    return definitions


def _pipeline(
    model_name: str,
    training_features: pd.DataFrame,
    *,
    fixed_parameters: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
    random_state: int,
    forbidden_features: Sequence[str],
) -> Pipeline:
    if model_name in CANONICAL_MODEL_NAMES:
        return build_model_pipeline(
            model_name,
            training_features,
            fixed_parameters=fixed_parameters,
            candidate_parameters=candidate_parameters,
            random_state=random_state,
            forbidden_features=forbidden_features,
        )
    overlap = sorted(set(fixed_parameters).intersection(candidate_parameters))
    _require(not overlap, f"{model_name} candidate overwrites fixed values: {overlap}.")
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
    values = (metrics.get("macro_f1"), metrics.get("quadratic_weighted_kappa"))
    _require(
        all(value is not None and math.isfinite(float(value)) for value in values),
        "Inner macro-F1 or QWK is unavailable/non-finite.",
    )
    return float(values[0]), float(values[1])


def _append_fold_and_oof(
    fold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    run_id: str,
    repeated_contract_sha256: str,
    scientific_input_sha256: str,
    repetition: int,
    seed_record: Mapping[str, int],
    folds: SharedFoldArtifacts,
    outer_fold: int,
    model_name: str,
    train_ids: Sequence[int],
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
    identity = {
        "run_id": run_id,
        "repeated_contract_sha256": repeated_contract_sha256,
        "scientific_input_sha256": scientific_input_sha256,
        "repetition": int(repetition),
        "outer_seed": int(seed_record["outer_seed"]),
        "inner_seed": int(seed_record["inner_seed"]),
        "model_seed": int(seed_record["model_seed"]),
        "fold_contract_hash": str(folds.contract["fold_contract_hash"]),
    }
    fold_rows.append(
        {
            **identity,
            "outer_fold": int(outer_fold),
            "model": model_name,
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            **bundle["aggregate_metrics"],
        }
    )
    for position, sample_index in enumerate(test_ids):
        row = {
            **identity,
            "evidence_source": "v3_repeated_nested_outer_fold_fit",
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


def _evaluate_repetition(
    features: pd.DataFrame,
    target: pd.Series,
    folds: SharedFoldArtifacts,
    definitions: Mapping[str, Mapping[str, Any]],
    *,
    run_id: str,
    repeated_contract_sha256: str,
    scientific_input_sha256: str,
    repetition: int,
    seed_record: Mapping[str, int],
    labels: Sequence[int],
    tie_tolerance: float,
    forbidden_features: Sequence[str],
    outer_fold_subset: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_shared_folds(folds)
    outer = folds.outer_assignments.copy()
    inner = folds.inner_assignments.copy()
    all_outer_folds = tuple(sorted(outer["outer_fold"].astype(int).unique()))
    if outer_fold_subset is None:
        selected_outer_folds = all_outer_folds
    else:
        selected_outer_folds = tuple(sorted(set(map(int, outer_fold_subset))))
        _require(
            selected_outer_folds
            and set(selected_outer_folds).issubset(all_outer_folds),
            "Diagnostic outer-fold subset is invalid.",
        )
    inner_splits = int(folds.contract["inner_splits"])
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    base_identity = {
        "run_id": run_id,
        "repeated_contract_sha256": repeated_contract_sha256,
        "scientific_input_sha256": scientific_input_sha256,
        "repetition": int(repetition),
        "outer_seed": int(seed_record["outer_seed"]),
        "inner_seed": int(seed_record["inner_seed"]),
        "model_seed": int(seed_record["model_seed"]),
        "fold_contract_hash": str(folds.contract["fold_contract_hash"]),
    }

    for outer_fold in selected_outer_folds:
        test_ids = outer.loc[
            outer["outer_fold"].astype(int) == outer_fold, "sample_index"
        ].astype(int).tolist()
        train_ids = outer.loc[
            outer["outer_fold"].astype(int) != outer_fold, "sample_index"
        ].astype(int).tolist()
        scoped_inner = inner[inner["outer_fold"].astype(int) == outer_fold].copy()
        _require(
            set(scoped_inner["sample_index"].astype(int)) == set(train_ids),
            f"Inner assignments differ from outer training at repetition={repetition}, outer={outer_fold}.",
        )

        for model_name in TUNED_MODEL_NAMES:
            definition = definitions[model_name]
            fixed = dict(definition["fixed_params"])
            candidates = [dict(candidate) for candidate in definition["candidates"]]
            primary_means: list[float] = []
            tie_break_means: list[float] = []
            scoped_candidates: list[dict[str, Any]] = []
            for candidate_index, candidate in enumerate(candidates):
                primary_scores: list[float] = []
                tie_break_scores: list[float] = []
                for inner_fold in range(1, inner_splits + 1):
                    validation_ids = scoped_inner.loc[
                        scoped_inner["inner_fold"].astype(int) == inner_fold,
                        "sample_index",
                    ].astype(int).tolist()
                    development_ids = sorted(set(train_ids) - set(validation_ids))
                    _require(
                        bool(validation_ids) and bool(development_ids),
                        f"Empty inner partition at repetition={repetition}, outer={outer_fold}, inner={inner_fold}.",
                    )
                    pipeline = _pipeline(
                        model_name,
                        features.loc[development_ids],
                        fixed_parameters=fixed,
                        candidate_parameters=candidate,
                        random_state=int(seed_record["model_seed"]),
                        forbidden_features=forbidden_features,
                    )
                    _fit_or_fail(
                        pipeline,
                        features.loc[development_ids],
                        target.loc[development_ids],
                        context=(
                            f"repetition={repetition}, model={model_name}, outer={outer_fold}, "
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
                    primary_scores.append(macro_f1)
                    tie_break_scores.append(qwk)
                primary_mean = float(np.mean(primary_scores))
                tie_break_mean = float(np.mean(tie_break_scores))
                primary_means.append(primary_mean)
                tie_break_means.append(tie_break_mean)
                scoped_candidates.append(
                    {
                        **base_identity,
                        "outer_fold": int(outer_fold),
                        "model": model_name,
                        "candidate_index": int(candidate_index),
                        "parameters_json": json.dumps(
                            candidate, sort_keys=True, separators=(",", ":")
                        ),
                        "inner_macro_f1_scores_json": json.dumps(primary_scores),
                        "inner_macro_f1_mean": primary_mean,
                        "inner_qwk_scores_json": json.dumps(tie_break_scores),
                        "inner_qwk_mean": tie_break_mean,
                        "n_inner_folds": inner_splits,
                        "candidate_status": "complete",
                        "outer_test_used_for_selection": False,
                    }
                )
            selected_index = select_candidate_index(
                primary_means,
                tie_break_means,
                practical_tie_tolerance=tie_tolerance,
                better_direction="higher",
            )
            for row in scoped_candidates:
                row["selected_by_protocol"] = row["candidate_index"] == selected_index
            candidate_rows.extend(scoped_candidates)
            selected_candidate = candidates[selected_index]
            final_pipeline = _pipeline(
                model_name,
                features.loc[train_ids],
                fixed_parameters=fixed,
                candidate_parameters=selected_candidate,
                random_state=int(seed_record["model_seed"]),
                forbidden_features=forbidden_features,
            )
            _fit_or_fail(
                final_pipeline,
                features.loc[train_ids],
                target.loc[train_ids],
                context=f"outer refit repetition={repetition}, model={model_name}, outer={outer_fold}",
            )
            prediction = np.asarray(
                final_pipeline.predict(features.loc[test_ids]), dtype=int
            )
            probability = aligned_predict_proba(
                final_pipeline, features.loc[test_ids], labels=labels
            )
            selected_rows.append(
                {
                    **base_identity,
                    "outer_fold": int(outer_fold),
                    "model": model_name,
                    "selection_performed": True,
                    "selected_candidate_index": int(selected_index),
                    "selected_candidate_parameters_json": json.dumps(
                        selected_candidate, sort_keys=True, separators=(",", ":")
                    ),
                    "fixed_parameters_json": json.dumps(
                        fixed, sort_keys=True, separators=(",", ":")
                    ),
                    "selected_inner_macro_f1_mean": primary_means[selected_index],
                    "selected_inner_qwk_mean": tie_break_means[selected_index],
                    "outer_test_used_for_selection": False,
                }
            )
            _append_fold_and_oof(
                fold_rows,
                prediction_rows,
                run_id=run_id,
                repeated_contract_sha256=repeated_contract_sha256,
                scientific_input_sha256=scientific_input_sha256,
                repetition=repetition,
                seed_record=seed_record,
                folds=folds,
                outer_fold=outer_fold,
                model_name=model_name,
                train_ids=train_ids,
                test_ids=test_ids,
                target=target,
                prediction=prediction,
                probability=probability,
                labels=labels,
                selected_candidate_index=selected_index,
            )

        for baseline_name in V3_NAIVE_BASELINE_NAMES:
            baseline = build_v3_naive_baseline(
                baseline_name, random_state=int(seed_record["model_seed"])
            )
            _fit_or_fail(
                baseline,
                features.loc[train_ids],
                target.loc[train_ids],
                context=f"baseline repetition={repetition}, model={baseline_name}, outer={outer_fold}",
            )
            prediction = np.asarray(baseline.predict(features.loc[test_ids]), dtype=int)
            probability = aligned_predict_proba(
                baseline, features.loc[test_ids], labels=labels
            )
            selected_rows.append(
                {
                    **base_identity,
                    "outer_fold": int(outer_fold),
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
            _append_fold_and_oof(
                fold_rows,
                prediction_rows,
                run_id=run_id,
                repeated_contract_sha256=repeated_contract_sha256,
                scientific_input_sha256=scientific_input_sha256,
                repetition=repetition,
                seed_record=seed_record,
                folds=folds,
                outer_fold=outer_fold,
                model_name=baseline_name,
                train_ids=train_ids,
                test_ids=test_ids,
                target=target,
                prediction=prediction,
                probability=probability,
                labels=labels,
                selected_candidate_index=None,
            )

    return (
        pd.DataFrame(candidate_rows),
        pd.DataFrame(selected_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
    )


def _validate_oof(
    oof: pd.DataFrame,
    folds_by_repetition: Mapping[int, SharedFoldArtifacts],
    *,
    labels: Sequence[int],
    full_run: bool,
) -> None:
    _require(not oof.empty, "Repeated OOF evidence is empty.")
    expected_repetitions = set(folds_by_repetition)
    _require(set(oof["repetition"].astype(int)) == expected_repetitions, "OOF repetitions drifted.")
    probability_columns = [f"prob_class_{label}" for label in labels]
    for repetition, scoped in oof.groupby("repetition", sort=True):
        repetition = int(repetition)
        folds = folds_by_repetition[repetition]
        outer = folds.outer_assignments
        expected_ids = set(outer["sample_index"].astype(int))
        _require(set(scoped["model"].unique()) == set(ALL_MODEL_NAMES), f"Model grid drifted in repetition {repetition}.")
        fold_lookup = outer.set_index("sample_index")[["outer_fold", "y_true"]].astype(int).sort_index()
        for model_name, rows in scoped.groupby("model", sort=False):
            ids = rows["sample_index"].astype(int)
            if full_run:
                _require(
                    len(rows) == len(expected_ids)
                    and ids.nunique() == len(expected_ids)
                    and set(ids) == expected_ids,
                    f"Exactly-once OOF coverage failed for repetition={repetition}, model={model_name}.",
                )
            observed = rows.set_index("sample_index")[["outer_fold", "y_true"]].astype(int).sort_index()
            _require(
                observed.equals(fold_lookup.loc[observed.index]),
                f"Fold/target lineage drifted for repetition={repetition}, model={model_name}.",
            )
            probability = rows[probability_columns].to_numpy(float)
            _require(np.all(np.isfinite(probability)), "OOF probabilities must be finite.")
            _require(
                np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-9),
                f"Probability simplex drifted for repetition={repetition}, model={model_name}.",
            )
            predicted = np.asarray(labels)[np.argmax(probability, axis=1)]
            _require(
                np.array_equal(predicted, rows["y_pred"].to_numpy(int)),
                f"Probability/prediction mismatch for repetition={repetition}, model={model_name}.",
            )


def summarize_repeated_metrics_v3(
    repetition_metrics: pd.DataFrame,
    *,
    repetitions: Sequence[int] = (1, 2, 3, 4, 5),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize priority variability and six-model ordering stability."""

    required = {"repetition", "model_name", "metric", "value"}
    _require(required.issubset(repetition_metrics.columns), "Repetition metrics schema is incomplete.")
    expected_repetitions = tuple(map(int, repetitions))
    _require(
        tuple(sorted(repetition_metrics["repetition"].astype(int).unique()))
        == expected_repetitions,
        "Repetition metric identities drifted.",
    )
    priority = repetition_metrics[
        repetition_metrics["metric"].isin(PRIORITY_METRICS)
    ].copy()
    expected_grid = len(expected_repetitions) * len(ALL_MODEL_NAMES) * len(PRIORITY_METRICS)
    _require(len(priority) == expected_grid, "Priority repetition metric grid is incomplete.")
    _require(
        not priority.duplicated(["repetition", "model_name", "metric"]).any(),
        "Priority repetition metric grid has duplicates.",
    )
    _require(
        set(priority["model_name"].unique()) == set(ALL_MODEL_NAMES),
        "Priority metric model set drifted.",
    )
    _require(
        np.all(np.isfinite(priority["value"].to_numpy(float))),
        "Priority repetition metrics must be finite.",
    )

    variability_rows: list[dict[str, Any]] = []
    for (model_name, metric), rows in priority.groupby(["model_name", "metric"], sort=True):
        values = rows.sort_values("repetition")["value"].to_numpy(float)
        variability_rows.append(
            {
                "model_name": str(model_name),
                "metric": str(metric),
                "repetition_count": len(values),
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "range_interpretation": "empirical_repetition_range_not_confidence_interval",
            }
        )
    variability = pd.DataFrame(variability_rows)

    tuned = priority[priority["model_name"].isin(TUNED_MODEL_NAMES)].copy()
    rank_rows: list[dict[str, Any]] = []
    directions = {
        "macro_f1": "higher",
        "balanced_accuracy": "higher",
        "quadratic_weighted_kappa": "higher",
        "ordinal_mae": "lower",
    }
    for (repetition, metric), rows in tuned.groupby(["repetition", "metric"], sort=True):
        direction = directions[str(metric)]
        rows = rows.sort_values("model_name").copy()
        rows["rank"] = rows["value"].rank(
            method="min", ascending=direction == "lower"
        )
        for record in rows.to_dict(orient="records"):
            rank_rows.append(
                {
                    "repetition": int(repetition),
                    "metric": str(metric),
                    "direction": direction,
                    "model_name": str(record["model_name"]),
                    "value": float(record["value"]),
                    "rank": int(record["rank"]),
                    "is_winner": int(record["rank"]) == 1,
                }
            )
    ranks = pd.DataFrame(rank_rows).sort_values(
        ["metric", "repetition", "rank", "model_name"]
    ).reset_index(drop=True)

    rank_summary_rows: list[dict[str, Any]] = []
    for (metric, model_name), rows in ranks.groupby(["metric", "model_name"], sort=True):
        rank_values = rows.sort_values("repetition")["rank"].to_numpy(float)
        winner_count = int(rows["is_winner"].astype(bool).sum())
        rank_summary_rows.append(
            {
                "metric": str(metric),
                "direction": str(rows["direction"].iloc[0]),
                "model_name": str(model_name),
                "repetition_count": len(rank_values),
                "mean_rank": float(np.mean(rank_values)),
                "sample_sd_rank": float(np.std(rank_values, ddof=1)),
                "median_rank": float(np.median(rank_values)),
                "minimum_rank": int(np.min(rank_values)),
                "maximum_rank": int(np.max(rank_values)),
                "winner_count": winner_count,
                "winner_frequency": winner_count / len(rank_values),
            }
        )
    rank_summary = pd.DataFrame(rank_summary_rows)

    stability_rows: list[dict[str, Any]] = []
    for metric, rows in ranks.groupby("metric", sort=True):
        matrix = rows.pivot(index="repetition", columns="model_name", values="rank").sort_index()
        pairwise: list[float] = []
        for left_index in range(len(matrix.index)):
            for right_index in range(left_index + 1, len(matrix.index)):
                correlation = matrix.iloc[left_index].corr(
                    matrix.iloc[right_index], method="spearman"
                )
                _require(
                    correlation is not None and math.isfinite(float(correlation)),
                    f"Rank correlation is non-finite for {metric}.",
                )
                pairwise.append(float(correlation))
        winner_counts = (
            rows.groupby("model_name")["is_winner"].sum().astype(int).sort_index()
        )
        highest_count = int(winner_counts.max())
        modal_winners = sorted(winner_counts[winner_counts == highest_count].index.astype(str))
        winner_models = sorted(
            rows.loc[rows["is_winner"].astype(bool), "model_name"].astype(str).unique()
        )
        stability_rows.append(
            {
                "metric": str(metric),
                "direction": str(rows["direction"].iloc[0]),
                "repetition_count": len(matrix.index),
                "model_count": len(matrix.columns),
                "repetition_pair_count": len(pairwise),
                "unique_winner_count": len(winner_models),
                "winner_models_json": json.dumps(winner_models, separators=(",", ":")),
                "modal_winner_models_json": json.dumps(modal_winners, separators=(",", ":")),
                "modal_winner_frequency": highest_count / len(matrix.index),
                "mean_pairwise_rank_spearman": float(np.mean(pairwise)),
                "minimum_pairwise_rank_spearman": float(np.min(pairwise)),
                "maximum_pairwise_rank_spearman": float(np.max(pairwise)),
                "interpretation": "descriptive_five_repetition_ordering_stability",
            }
        )
    stability = pd.DataFrame(stability_rows)
    return variability, ranks, rank_summary, stability


def selected_candidate_frequency_v3(selected: pd.DataFrame) -> pd.DataFrame:
    """Count fold-level selected candidates without using outer-test performance."""

    required = {
        "repetition",
        "outer_fold",
        "model",
        "selection_performed",
        "selected_candidate_index",
        "selected_candidate_parameters_json",
        "outer_test_used_for_selection",
    }
    _require(required.issubset(selected.columns), "Selection record schema is incomplete.")
    _require(
        not selected["outer_test_used_for_selection"].astype(bool).any(),
        "Outer test entered repeated-CV selection records.",
    )
    tuned = selected[selected["model"].isin(TUNED_MODEL_NAMES)].copy()
    _require(tuned["selection_performed"].astype(bool).all(), "A tuned model lacks selection.")
    try:
        tuned["selected_candidate_index"] = pd.to_numeric(
            tuned["selected_candidate_index"], errors="raise"
        ).astype(int)
    except (TypeError, ValueError) as exc:
        raise RepeatedNestedCVError(
            "A tuned model has an invalid selected-candidate index."
        ) from exc
    _require(
        not tuned.duplicated(["repetition", "outer_fold", "model"]).any(),
        "Tuned selection records contain duplicates.",
    )
    grouped = (
        tuned.groupby(
            ["model", "selected_candidate_index", "selected_candidate_parameters_json"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "selection_count", "model": "model_name"})
    )
    totals = tuned.groupby("model").size().rename("selection_opportunities")
    grouped["selection_opportunities"] = grouped["model_name"].map(totals).astype(int)
    grouped["selection_frequency"] = (
        grouped["selection_count"] / grouped["selection_opportunities"]
    )
    return grouped.sort_values(["model_name", "selected_candidate_index"]).reset_index(drop=True)


def _outer_assignment_semantic_sha256(folds: SharedFoldArtifacts) -> str:
    records = (
        folds.outer_assignments[["sample_index", "outer_fold"]]
        .astype(int)
        .sort_values("sample_index")
        .to_dict(orient="records")
    )
    encoded = json.dumps(records, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_repeated_nested_cv_v3(
    source_frame: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    repeated_contract: Mapping[str, Any],
    nominal_config: Mapping[str, Any],
    ordinal_config: Mapping[str, Any],
    *,
    run_id: str,
    repeated_contract_sha256: str,
    scientific_input_sha256: str,
    dataset_sha256: str,
    id_column: str = "EmpNumber",
    repetition_subset: Sequence[int] | None = None,
    outer_fold_subset: Sequence[int] | None = None,
) -> RepeatedNestedCVResult:
    """Fit every system on prespecified repeated folds and return local evidence."""

    _require(bool(str(run_id).strip()), "run_id must be non-empty.")
    for name, digest in (
        ("repeated_contract_sha256", repeated_contract_sha256),
        ("scientific_input_sha256", scientific_input_sha256),
        ("dataset_sha256", dataset_sha256),
    ):
        _require(_valid_digest(str(digest)), f"{name} must be a lowercase SHA-256 digest.")
    _require(source_frame.index.equals(features.index), "Source/features indices drifted.")
    _require(features.index.equals(target.index), "Feature/target indices drifted.")
    _require(id_column in source_frame.columns, f"Identifier {id_column!r} is absent.")
    labels = tuple(int(value) for value in repeated_contract.get("ordered_labels", ()))
    _require(labels == (2, 3, 4), "Repeated-CV ordered labels must equal (2, 3, 4).")
    _require(set(target.astype(int).unique()) == set(labels), "Target support drifted.")
    design = repeated_contract["design"]
    seed_schedule = [dict(record) for record in design["seed_schedule"]]
    all_repetitions = tuple(int(record["repetition"]) for record in seed_schedule)
    if repetition_subset is None:
        selected_repetitions = all_repetitions
    else:
        selected_repetitions = tuple(sorted(set(map(int, repetition_subset))))
        _require(
            selected_repetitions
            and set(selected_repetitions).issubset(all_repetitions),
            "Diagnostic repetition subset is invalid.",
        )
    full_run = repetition_subset is None and outer_fold_subset is None
    evidence_status = (
        "complete_five_repetition_exactly_once_oof"
        if full_run
        else "diagnostic_incomplete_never_canonical"
    )
    definitions = _model_definitions(nominal_config, ordinal_config)
    tie_tolerance = float(repeated_contract["selection"]["primary_tie_tolerance"])
    candidate_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    oof_frames: list[pd.DataFrame] = []
    folds_by_repetition: dict[int, SharedFoldArtifacts] = {}
    fold_contract_records: list[Mapping[str, Any]] = []

    for seed_record in seed_schedule:
        repetition = int(seed_record["repetition"])
        if repetition not in selected_repetitions:
            continue
        folds = generate_shared_folds(
            source_frame,
            target_column=str(repeated_contract["target"]),
            id_column=id_column,
            run_id=f"{run_id}_rep{repetition}",
            config_hash=repeated_contract_sha256,
            scientific_input_hash=scientific_input_sha256,
            dataset_key=str(repeated_contract["dataset_key"]),
            dataset_sha256=dataset_sha256,
            outer_splits=int(design["outer_splits"]),
            inner_splits=int(design["inner_splits"]),
            seed=int(seed_record["outer_seed"]),
            inner_seed=int(seed_record["inner_seed"]),
        )
        validate_shared_folds(folds)
        folds_by_repetition[repetition] = folds
        semantic_hash = _outer_assignment_semantic_sha256(folds)
        fold_contract_records.append(
            {
                "repetition": repetition,
                "outer_seed": int(seed_record["outer_seed"]),
                "inner_seed": int(seed_record["inner_seed"]),
                "model_seed": int(seed_record["model_seed"]),
                "outer_assignment_semantic_sha256": semantic_hash,
                "contract": folds.contract,
            }
        )
        candidate, selected, fold_metrics, oof = _evaluate_repetition(
            features,
            target,
            folds,
            definitions,
            run_id=run_id,
            repeated_contract_sha256=repeated_contract_sha256,
            scientific_input_sha256=scientific_input_sha256,
            repetition=repetition,
            seed_record=seed_record,
            labels=labels,
            tie_tolerance=tie_tolerance,
            forbidden_features=tuple(
                column for column in source_frame.columns if column not in features.columns
            ),
            outer_fold_subset=outer_fold_subset,
        )
        candidate_frames.append(candidate)
        selected_frames.append(selected)
        fold_frames.append(fold_metrics)
        oof_frames.append(oof)

    semantic_hashes = [
        str(record["outer_assignment_semantic_sha256"])
        for record in fold_contract_records
    ]
    _require(
        len(set(semantic_hashes)) == len(semantic_hashes),
        "Two repetitions produced the same semantic outer-fold assignment.",
    )
    candidates = pd.concat(candidate_frames, ignore_index=True).sort_values(
        ["repetition", "outer_fold", "model", "candidate_index"]
    ).reset_index(drop=True)
    selected = pd.concat(selected_frames, ignore_index=True).sort_values(
        ["repetition", "outer_fold", "model"]
    ).reset_index(drop=True)
    fold_metrics = pd.concat(fold_frames, ignore_index=True).sort_values(
        ["repetition", "outer_fold", "model"]
    ).reset_index(drop=True)
    oof = pd.concat(oof_frames, ignore_index=True).sort_values(
        ["repetition", "model", "sample_index"]
    ).reset_index(drop=True)
    _validate_oof(oof, folds_by_repetition, labels=labels, full_run=full_run)
    _require(
        not candidates["outer_test_used_for_selection"].astype(bool).any(),
        "Outer test entered candidate selection.",
    )
    selected_flags = candidates.groupby(
        ["repetition", "outer_fold", "model"]
    )["selected_by_protocol"].sum()
    _require(selected_flags.eq(1).all(), "A tuned model/fold lacks exactly one selection.")
    _require(
        not selected["outer_test_used_for_selection"].astype(bool).any(),
        "Outer test entered selected-hyperparameter records.",
    )

    repetition_metric_frames: list[pd.DataFrame] = []
    for repetition, rows in oof.groupby("repetition", sort=True):
        aggregate, _, _ = summarize_combined_oof_v3(rows, labels=labels)
        identity_row = rows.iloc[0]
        aggregate.insert(0, "fold_contract_hash", str(identity_row["fold_contract_hash"]))
        aggregate.insert(0, "model_seed", int(identity_row["model_seed"]))
        aggregate.insert(0, "inner_seed", int(identity_row["inner_seed"]))
        aggregate.insert(0, "outer_seed", int(identity_row["outer_seed"]))
        aggregate.insert(0, "repetition", int(repetition))
        aggregate.insert(0, "scientific_input_sha256", scientific_input_sha256)
        aggregate.insert(0, "repeated_contract_sha256", repeated_contract_sha256)
        aggregate.insert(0, "run_id", run_id)
        repetition_metric_frames.append(aggregate)
    repetition_metrics = pd.concat(repetition_metric_frames, ignore_index=True).sort_values(
        ["repetition", "model_name", "metric"]
    ).reset_index(drop=True)
    if full_run:
        variability, ranks, rank_summary, stability = summarize_repeated_metrics_v3(
            repetition_metrics,
            repetitions=all_repetitions,
        )
    else:
        variability = pd.DataFrame()
        ranks = pd.DataFrame()
        rank_summary = pd.DataFrame()
        stability = pd.DataFrame()
    frequency = selected_candidate_frequency_v3(selected)
    return RepeatedNestedCVResult(
        candidate_search_results=candidates,
        selected_hyperparameters=selected,
        fold_metrics=fold_metrics,
        oof_predictions=oof,
        repetition_metrics=repetition_metrics,
        variability_summary=variability,
        rank_by_repetition=ranks,
        model_rank_summary=rank_summary,
        ordering_stability=stability,
        selected_candidate_frequency=frequency,
        fold_contracts=tuple(fold_contract_records),
        evidence_status=evidence_status,
    )


def _prepare_inputs(
    contract_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Any,
    pd.DataFrame,
    tuple[str, ...],
    pd.Series,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    contract_receipt = validate_repeated_nested_cv_contract_v3(contract_path)
    contract = _load_json(contract_path)
    sources = contract["source_contracts"]
    canonical = load_canonical_dataset(
        sources["canonical_loader_config"]["path"],
        str(contract["dataset_key"]),
        sources["acquisition_manifest"]["path"],
        allow_download=False,
    )
    feature_contract_path = Path(sources["feature_availability"]["path"])
    validate_feature_availability_contract(feature_contract_path)
    feature_contract = _load_json(feature_contract_path)
    features, exclusions = exact_p3_feature_frame(canonical.frame, feature_contract)
    manuscript_config = load_config(sources["canonical_loader_config"]["path"])
    _require(
        set(exclusions) == set(primary_excluded_features(manuscript_config)),
        "Repeated-CV P3 exclusions differ from canonical-v2 primary exclusions.",
    )
    target = canonical.frame[str(contract["target"])].astype(int)
    nominal_config = load_config(sources["nominal_model_grid"]["path"])
    ordinal_config = _load_json(Path(sources["ordinal_benchmark"]["path"]))
    _model_definitions(nominal_config, ordinal_config)
    return (
        contract,
        contract_receipt,
        canonical,
        features,
        exclusions,
        target,
        nominal_config,
        ordinal_config,
    )


def preflight_repeated_nested_cv_v3(
    *,
    contract_path: Path | str = DEFAULT_REPEATED_CV_CONTRACT_PATH,
) -> dict[str, Any]:
    """Validate inputs and generate all fold identities without fitting a model."""

    path = Path(contract_path)
    (
        contract,
        receipt,
        canonical,
        features,
        _,
        target,
        _,
        _,
    ) = _prepare_inputs(path)
    fold_hashes: list[str] = []
    semantic_hashes: list[str] = []
    for seed_record in contract["design"]["seed_schedule"]:
        repetition = int(seed_record["repetition"])
        folds = generate_shared_folds(
            canonical.frame,
            target_column=str(contract["target"]),
            id_column="EmpNumber",
            run_id=f"preflight_repeated_cv_v3_rep{repetition}",
            config_hash=receipt["contract_sha256"],
            scientific_input_hash=receipt["contract_sha256"],
            dataset_key=str(contract["dataset_key"]),
            dataset_sha256=canonical.receipt["actual_sha256"],
            outer_splits=5,
            inner_splits=5,
            seed=int(seed_record["outer_seed"]),
            inner_seed=int(seed_record["inner_seed"]),
        )
        validate_shared_folds(folds)
        fold_hashes.append(str(folds.contract["fold_contract_hash"]))
        semantic_hashes.append(_outer_assignment_semantic_sha256(folds))
    _require(len(set(semantic_hashes)) == 5, "Preflight repetitions do not have distinct folds.")
    return {
        "status": "passed",
        "contract_sha256": receipt["contract_sha256"],
        "dataset_sha256": canonical.receipt["actual_sha256"],
        "sample_count": len(features),
        "feature_count": features.shape[1],
        "target_distribution": canonical.receipt["target_distribution"],
        "repetitions": 5,
        "outer_folds_per_repetition": 5,
        "inner_folds": 5,
        "fold_contract_hashes": fold_hashes,
        "outer_assignment_semantic_hashes": semantic_hashes,
        "distinct_outer_assignment_count": len(set(semantic_hashes)),
        "planned_estimator_fit_calls": receipt["planned_estimator_fit_calls"],
        "model_fit_count": 0,
        "paid_api_calls": 0,
        "network_calls": 0,
    }


def diagnostic_repeated_nested_cv_v3(
    *,
    contract_path: Path | str = DEFAULT_REPEATED_CV_CONTRACT_PATH,
    repetition: int = 1,
    outer_fold: int = 1,
) -> dict[str, Any]:
    """Run one in-memory outer fold with an explicitly inadmissible status."""

    path = Path(contract_path)
    with enforce_offline_runtime() as offline_state:
        (
            contract,
            contract_receipt,
            canonical,
            features,
            _,
            target,
            nominal_config,
            ordinal_config,
        ) = _prepare_inputs(path)
        diagnostic_scientific_hash = _canonical_json_sha256(
            {
                "contract_sha256": contract_receipt["contract_sha256"],
                "dataset_sha256": canonical.receipt["actual_sha256"],
                "diagnostic_only": True,
            }
        )
        result = evaluate_repeated_nested_cv_v3(
            canonical.frame,
            features,
            target,
            contract,
            nominal_config,
            ordinal_config,
            run_id="diagnostic_incomplete_never_canonical",
            repeated_contract_sha256=contract_receipt["contract_sha256"],
            scientific_input_sha256=diagnostic_scientific_hash,
            dataset_sha256=canonical.receipt["actual_sha256"],
            repetition_subset=[repetition],
            outer_fold_subset=[outer_fold],
        )
        _require(
            result.evidence_status == "diagnostic_incomplete_never_canonical",
            "Bounded diagnostic acquired an admissible evidence status.",
        )
        runtime_receipt = offline_state.receipt()
    return {
        "status": "diagnostic_incomplete_never_canonical",
        "persisted": False,
        "repetition": repetition,
        "outer_fold": outer_fold,
        "model_count": len(ALL_MODEL_NAMES),
        "candidate_search_row_count": len(result.candidate_search_results),
        "selected_hyperparameter_row_count": len(result.selected_hyperparameters),
        "fold_metric_row_count": len(result.fold_metrics),
        "oof_prediction_row_count": len(result.oof_predictions),
        "repetition_metric_row_count": len(result.repetition_metrics),
        "estimator_fit_call_count": 229,
        "runtime_policy": runtime_receipt,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _write_json(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _run_repeated_nested_cv_v3_impl(
    *,
    contract_path: Path,
    output_dir: Path,
    run_id: str,
    offline_state: Any,
) -> dict[str, Any]:
    git_identity = _clean_git_identity()
    (
        contract,
        contract_receipt,
        canonical,
        features,
        exclusions,
        target,
        nominal_config,
        ordinal_config,
    ) = _prepare_inputs(contract_path)
    _require(not output_dir.exists(), f"Output destination already exists: {output_dir}.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        implementation_paths = (
            Path("src/experiments/repeated_nested_cv_v3.py"),
            Path("src/governance/repeated_nested_cv_contract_v3.py"),
            Path("src/experiments/shared_folds.py"),
            Path("src/models/canonical_models.py"),
            Path("src/models/ordinal_models_v3.py"),
            Path("src/models/ordinal_evaluation_v3.py"),
        )
        implementation_hashes = {
            path.as_posix(): sha256_file(path) for path in implementation_paths
        }
        current_source_tree_hash = source_tree_hash(PROJECT_ROOT)
        scientific_inputs = {
            "git_identity": git_identity,
            "source_tree_hash": current_source_tree_hash,
            "repeated_contract_sha256": contract_receipt["contract_sha256"],
            "bound_source_hashes": contract_receipt["source_hashes"],
            "dataset_sha256": canonical.receipt["actual_sha256"],
            "implementation_hashes": implementation_hashes,
        }
        scientific_input_sha256 = _canonical_json_sha256(scientific_inputs)
        result = evaluate_repeated_nested_cv_v3(
            canonical.frame,
            features,
            target,
            contract,
            nominal_config,
            ordinal_config,
            run_id=run_id,
            repeated_contract_sha256=contract_receipt["contract_sha256"],
            scientific_input_sha256=scientific_input_sha256,
            dataset_sha256=canonical.receipt["actual_sha256"],
        )
        frames = {
            "candidate_search_results.csv": result.candidate_search_results,
            "fold_metrics.csv": result.fold_metrics,
            "model_rank_summary.csv": result.model_rank_summary,
            "oof_predictions.csv": result.oof_predictions,
            "ordering_stability.csv": result.ordering_stability,
            "rank_by_repetition.csv": result.rank_by_repetition,
            "repetition_metrics.csv": result.repetition_metrics,
            "selected_candidate_frequency.csv": result.selected_candidate_frequency,
            "selected_hyperparameters.csv": result.selected_hyperparameters,
            "variability_summary.csv": result.variability_summary,
        }
        for filename, frame in frames.items():
            frame.to_csv(staging / filename, index=False)
        _write_json(staging / "fold_contracts.json", list(result.fold_contracts))
        output_hashes = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        metadata = {
            "schema_version": 1,
            "stage": "repeated_nested_cv_v3",
            "status": "complete",
            "evidence_status": result.evidence_status,
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repeated_contract_sha256": contract_receipt["contract_sha256"],
            "scientific_input_sha256": scientific_input_sha256,
            "scientific_inputs": scientific_inputs,
            "git_identity": git_identity,
            "source_tree_hash": current_source_tree_hash,
            "dataset_sha256": canonical.receipt["actual_sha256"],
            "feature_policy": "P3",
            "feature_count": int(features.shape[1]),
            "sample_count": int(len(features)),
            "ordered_labels": [2, 3, 4],
            "repetitions": 5,
            "outer_folds_per_repetition": 5,
            "inner_folds": 5,
            "model_count": len(ALL_MODEL_NAMES),
            "tuned_model_count": len(TUNED_MODEL_NAMES),
            "planned_estimator_fit_calls": contract_receipt[
                "planned_estimator_fit_calls"
            ],
            "candidate_search_row_count": len(result.candidate_search_results),
            "selected_hyperparameter_row_count": len(result.selected_hyperparameters),
            "fold_metric_row_count": len(result.fold_metrics),
            "oof_prediction_row_count": len(result.oof_predictions),
            "repetition_metric_row_count": len(result.repetition_metrics),
            "outer_test_used_for_selection": False,
            "seed_or_repetition_selected_from_results": False,
            "employee_level_outputs_publication_authorized": False,
            "runtime_policy": offline_state.receipt(),
            "network_calls": 0,
            "paid_api_calls": 0,
            "output_hashes": output_hashes,
        }
        _require(_clean_git_identity() == git_identity, "Git identity changed during execution.")
        _require(
            source_tree_hash(PROJECT_ROOT) == current_source_tree_hash,
            "Scientific source tree changed during execution.",
        )
        repeat_receipt = validate_repeated_nested_cv_contract_v3(contract_path)
        _require(
            repeat_receipt["contract_sha256"] == contract_receipt["contract_sha256"],
            "Repeated-CV contract changed during execution.",
        )
        dataset_path = Path(str(canonical.receipt["actual_path"]))
        if not dataset_path.is_absolute():
            dataset_path = PROJECT_ROOT / dataset_path
        _require(
            sha256_file(dataset_path) == canonical.receipt["actual_sha256"],
            "Canonical dataset changed during execution.",
        )
        _write_json(staging / "stage_metadata.json", metadata)
        _require(
            {path.name for path in staging.iterdir() if path.is_file()}
            == EXPECTED_LOCAL_FILES,
            "Repeated-CV local output inventory drifted before publication.",
        )
        os.replace(staging, output_dir)
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
        "output_dir": output_dir.as_posix(),
        "repeated_contract_sha256": contract_receipt["contract_sha256"],
        "scientific_input_sha256": scientific_input_sha256,
        "repetitions": 5,
        "model_count": len(ALL_MODEL_NAMES),
        "oof_prediction_row_count": len(result.oof_predictions),
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def run_repeated_nested_cv_v3(
    *,
    contract_path: Path | str = DEFAULT_REPEATED_CV_CONTRACT_PATH,
    output_dir: Path | str,
    run_id: str,
) -> dict[str, Any]:
    """Execute a full clean-commit run under the process-wide offline boundary."""

    with enforce_offline_runtime() as offline_state:
        return _run_repeated_nested_cv_v3_impl(
            contract_path=Path(contract_path),
            output_dir=Path(output_dir),
            run_id=run_id,
            offline_state=offline_state,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_REPEATED_CV_CONTRACT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_LOCAL_RUN_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--diagnostic-repetition", type=int)
    parser.add_argument("--diagnostic-outer-fold", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    diagnostic_requested = (
        args.diagnostic_repetition is not None or args.diagnostic_outer_fold is not None
    )
    _require(
        not (args.preflight_only and diagnostic_requested),
        "Preflight and diagnostic modes are mutually exclusive.",
    )
    if args.preflight_only:
        receipt = preflight_repeated_nested_cv_v3(contract_path=args.contract)
    elif diagnostic_requested:
        _require(
            args.diagnostic_repetition is not None
            and args.diagnostic_outer_fold is not None,
            "Both diagnostic repetition and outer fold are required.",
        )
        receipt = diagnostic_repeated_nested_cv_v3(
            contract_path=args.contract,
            repetition=args.diagnostic_repetition,
            outer_fold=args.diagnostic_outer_fold,
        )
    else:
        _require(
            isinstance(args.run_id, str) and bool(args.run_id.strip()),
            "--run-id is required unless --preflight-only is used.",
        )
        receipt = run_repeated_nested_cv_v3(
            contract_path=args.contract,
            output_dir=args.output_root / args.run_id / "repeated_nested_cv",
            run_id=args.run_id,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALL_MODEL_NAMES",
    "DEFAULT_LOCAL_RUN_ROOT",
    "EXPECTED_LOCAL_FILES",
    "RepeatedNestedCVError",
    "RepeatedNestedCVResult",
    "TUNED_MODEL_NAMES",
    "diagnostic_repeated_nested_cv_v3",
    "evaluate_repeated_nested_cv_v3",
    "preflight_repeated_nested_cv_v3",
    "run_repeated_nested_cv_v3",
    "selected_candidate_frequency_v3",
    "summarize_repeated_metrics_v3",
]
