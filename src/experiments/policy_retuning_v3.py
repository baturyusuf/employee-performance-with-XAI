"""Run the v3 fixed-schedule versus independently retuned policy comparison."""

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
from threadpoolctl import threadpool_limits

from src.data.canonical_loader import load_canonical_dataset, sha256_file
from src.experiments.manuscript_model_benchmark import (
    select_candidate_index,
    validate_benchmark_config,
)
from src.experiments.shared_folds import SharedFoldArtifacts, read_shared_folds
from src.governance.manuscript_contract import source_tree_hash
from src.governance.offline_runtime import enforce_offline_runtime
from src.governance.policy_retuning_contract_v3 import (
    CANONICAL_V2_ROOT,
    DEFAULT_POLICY_RETUNING_CONTRACT_PATH,
    HEADLINE_METRICS,
    POLICY_IDS,
    POLICY_NAMES,
    validate_policy_retuning_contract_v3,
)
from src.models.canonical_models import aligned_predict_proba, build_model_pipeline
from src.models.evaluate import classification_metrics
from src.models.ordinal_evaluation_v3 import ordinal_evaluation_bundle_v3
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_LOCAL_RUN_ROOT = Path("reports/major_revision_v3_runs")
FIXED_ESTIMAND = "fixed_primary_schedule_feature_access_sensitivity"
RETUNED_ESTIMAND = "independently_retuned_policy_performance"
FIT_THREAD_LIMIT = 1
EXPECTED_LOCAL_FILES = frozenset(
    {
        "aggregate_metrics.csv",
        "candidate_search_results.csv",
        "combined_oof_predictions.csv",
        "fixed_oof_predictions.csv",
        "fold_metrics.csv",
        "headline_policy_comparison.csv",
        "metric_comparison.csv",
        "policy_feature_contract.csv",
        "retuned_oof_predictions.csv",
        "selected_candidate_frequency.csv",
        "selected_hyperparameters.csv",
        "stage_metadata.json",
    }
)


class PolicyRetuningV3Error(RuntimeError):
    """Raised when a policy input, fit, selection, or result invariant fails."""


@dataclass(frozen=True)
class PolicyRetuningV3Result:
    candidate_search_results: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    selected_candidate_frequency: pd.DataFrame
    fixed_oof_predictions: pd.DataFrame
    retuned_oof_predictions: pd.DataFrame
    combined_oof_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    metric_comparison: pd.DataFrame
    headline_policy_comparison: pd.DataFrame
    policy_feature_contract: pd.DataFrame
    evidence_status: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyRetuningV3Error(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyRetuningV3Error(f"Could not read {path.as_posix()}: {exc}") from exc
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
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


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
        raise PolicyRetuningV3Error(f"Could not establish Git identity: {exc}") from exc
    _require(_valid_digest(head, length=40), "Git HEAD must be a full lowercase commit digest.")
    _require(not status, f"Scientific execution requires a clean worktree; status={status.splitlines()[:10]}.")
    return {"commit": head, "branch": branch}


def _feature_frames(
    source_frame: pd.DataFrame,
    feature_contract: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    all_features = [str(record["feature_name"]) for record in feature_contract["features"]]
    _require(set(all_features) == set(source_frame.columns), "Feature contract does not cover the exact source schema.")
    policy_records = {str(record["policy_id"]): record for record in feature_contract["policies"]}
    _require(tuple(policy_records) == POLICY_IDS, "Feature-policy order drifted.")
    frames: dict[str, pd.DataFrame] = {}
    contract_rows: list[dict[str, Any]] = []
    for policy_order, (policy_id, policy_name) in enumerate(zip(POLICY_IDS, POLICY_NAMES)):
        record = policy_records[policy_id]
        _require(str(record["name"]) == policy_name, f"Policy name drifted for {policy_id}.")
        excluded = list(map(str, record["excluded_features"]))
        retained = [feature for feature in all_features if feature not in set(excluded)]
        _require(retained and "PerformanceRating" not in retained and "EmpNumber" not in retained, f"Forbidden feature retained for {policy_id}.")
        frames[policy_id] = source_frame.loc[:, retained].copy()
        contract_rows.append(
            {
                "policy_id": policy_id,
                "policy_order": policy_order,
                "policy_name": policy_name,
                "policy_role": str(record["role"]),
                "n_features": len(retained),
                "retained_features_json": json.dumps(retained, separators=(",", ":")),
                "excluded_features_json": json.dumps(excluded, separators=(",", ":")),
            }
        )
    return frames, pd.DataFrame(contract_rows)


def _fit_or_fail(estimator: Any, X: pd.DataFrame, y: pd.Series, *, context: str) -> Any:
    try:
        with threadpool_limits(limits=FIT_THREAD_LIMIT):
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                return estimator.fit(X, y)
    except Exception as exc:
        raise PolicyRetuningV3Error(f"{context} failed: {type(exc).__name__}: {exc}") from exc


def _pipeline(
    training_features: pd.DataFrame,
    *,
    model_definition: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
    forbidden_features: Sequence[str],
    random_state: int,
) -> Any:
    return build_model_pipeline(
        "xgboost",
        training_features,
        fixed_parameters=dict(model_definition["fixed_params"]),
        candidate_parameters=dict(candidate_parameters),
        random_state=random_state,
        forbidden_features=forbidden_features,
    )


def _selection_scores(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float]:
    metrics = classification_metrics(y_true, y_pred, y_proba, [2, 3, 4])
    values = (metrics.get("macro_f1"), metrics.get("quadratic_weighted_kappa"))
    _require(
        all(value is not None and math.isfinite(float(value)) for value in values),
        "Inner macro-F1 or QWK is unavailable/non-finite.",
    )
    return float(values[0]), float(values[1])


def _base_identity(
    *,
    run_id: str,
    policy_contract_sha256: str,
    scientific_input_sha256: str,
    fold_contract_hash: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "policy_contract_sha256": policy_contract_sha256,
        "scientific_input_sha256": scientific_input_sha256,
        "fold_contract_hash": fold_contract_hash,
    }


def _prediction_rows(
    *,
    identity: Mapping[str, Any],
    estimand: str,
    policy_record: Mapping[str, Any],
    evidence_source: str,
    source_policy: str | None,
    sample_indices: Sequence[int],
    outer_fold: Sequence[int] | int,
    target: Sequence[int],
    prediction: Sequence[int],
    probability: np.ndarray,
    selected_candidate_index: Sequence[int] | int,
) -> list[dict[str, Any]]:
    indices = list(map(int, sample_indices))
    folds = (
        [int(outer_fold)] * len(indices)
        if isinstance(outer_fold, (int, np.integer))
        else list(map(int, outer_fold))
    )
    selected = (
        [int(selected_candidate_index)] * len(indices)
        if isinstance(selected_candidate_index, (int, np.integer))
        else list(map(int, selected_candidate_index))
    )
    probability = np.asarray(probability, dtype=float)
    _require(probability.shape == (len(indices), 3), "Prediction probability shape drifted.")
    target_values = list(map(int, target))
    prediction_values = list(map(int, prediction))
    _require(
        len(target_values) == len(prediction_values) == len(indices),
        "Prediction target/label length drifted.",
    )
    rows: list[dict[str, Any]] = []
    for position, sample_index in enumerate(indices):
        rows.append(
            {
                **dict(identity),
                "estimand": estimand,
                "policy_id": str(policy_record["policy_id"]),
                "policy_name": str(policy_record["policy_name"]),
                "n_features": int(policy_record["n_features"]),
                "evidence_source": evidence_source,
                "source_policy": source_policy,
                "model": "xgboost",
                "sample_index": sample_index,
                "outer_fold": folds[position],
                "y_true": target_values[position],
                "y_pred": prediction_values[position],
                "selected_candidate_index": selected[position],
                "prob_class_2": float(probability[position, 0]),
                "prob_class_3": float(probability[position, 1]),
                "prob_class_4": float(probability[position, 2]),
            }
        )
    return rows


def _validate_oof(
    oof: pd.DataFrame,
    folds: SharedFoldArtifacts,
    *,
    policies: Sequence[str],
    outer_folds: Sequence[int],
    full_run: bool,
    expected_sample_count: int,
) -> None:
    expected_folds = set(map(int, outer_folds))
    reference = folds.outer_assignments.set_index("sample_index")[["outer_fold", "y_true"]].astype(int).sort_index()
    _require(set(oof["estimand"].unique()) == {FIXED_ESTIMAND, RETUNED_ESTIMAND}, "OOF estimand set drifted.")
    for (estimand, policy_id), rows in oof.groupby(["estimand", "policy_id"], sort=False):
        _require(policy_id in policies, "OOF policy set drifted.")
        _require(set(rows["outer_fold"].astype(int)) == expected_folds, f"OOF fold coverage drifted for {estimand}/{policy_id}.")
        if full_run:
            _require(
                len(rows) == expected_sample_count
                and rows["sample_index"].nunique() == expected_sample_count,
                f"Exactly-once OOF coverage failed for {estimand}/{policy_id}.",
            )
        observed = rows.set_index("sample_index")[["outer_fold", "y_true"]].astype(int).sort_index()
        _require(observed.equals(reference.loc[observed.index]), f"Fold/target lineage drifted for {estimand}/{policy_id}.")
        probability = rows[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float)
        _require(np.all(np.isfinite(probability)), "OOF probabilities must be finite.")
        _require(np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-9), "OOF probability simplex drifted.")
        expected_prediction = np.asarray([2, 3, 4])[np.argmax(probability, axis=1)]
        _require(np.array_equal(expected_prediction, rows["y_pred"].to_numpy(int)), "OOF prediction/probability mismatch.")


def summarize_policy_oof_v3(
    combined_oof: pd.DataFrame,
    *,
    policy_feature_contract: pd.DataFrame,
    total_sample_count: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build fold, aggregate, full-metric comparison, and headline records."""

    required = {
        "run_id",
        "policy_contract_sha256",
        "scientific_input_sha256",
        "fold_contract_hash",
        "estimand",
        "policy_id",
        "policy_name",
        "n_features",
        "model",
        "sample_index",
        "outer_fold",
        "y_true",
        "y_pred",
        "prob_class_2",
        "prob_class_3",
        "prob_class_4",
    }
    _require(required.issubset(combined_oof.columns), "Combined policy OOF schema is incomplete.")
    _require(set(combined_oof["estimand"].unique()) == {FIXED_ESTIMAND, RETUNED_ESTIMAND}, "Combined estimand set drifted.")
    identity_columns = (
        "run_id",
        "policy_contract_sha256",
        "scientific_input_sha256",
        "fold_contract_hash",
    )
    aggregate_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    probability_columns = ["prob_class_2", "prob_class_3", "prob_class_4"]
    for (estimand, policy_id), rows in combined_oof.groupby(["estimand", "policy_id"], sort=True):
        rows = rows.sort_values("sample_index")
        identity = {column: rows.iloc[0][column] for column in identity_columns}
        policy = {
            "estimand": str(estimand),
            "policy_id": str(policy_id),
            "policy_name": str(rows.iloc[0]["policy_name"]),
            "n_features": int(rows.iloc[0]["n_features"]),
            "model_name": "xgboost",
        }
        bundle = ordinal_evaluation_bundle_v3(
            rows["y_true"].astype(int),
            rows["y_pred"].astype(int),
            rows[probability_columns].to_numpy(float),
            labels=(2, 3, 4),
            dataset_key="inx_primary",
            model_name="xgboost",
        )
        aggregate_rows.extend(
            {**identity, **policy, "metric": metric, "value": float(value)}
            for metric, value in bundle["aggregate_metrics"].items()
        )
        total = int(total_sample_count) if total_sample_count is not None else rows["sample_index"].nunique()
        _require(total >= rows["sample_index"].nunique(), "Total sample count is smaller than observed OOF coverage.")
        for outer_fold, fold_rows_source in rows.groupby("outer_fold", sort=True):
            fold_rows_source = fold_rows_source.sort_values("sample_index")
            fold_bundle = ordinal_evaluation_bundle_v3(
                fold_rows_source["y_true"].astype(int),
                fold_rows_source["y_pred"].astype(int),
                fold_rows_source[probability_columns].to_numpy(float),
                labels=(2, 3, 4),
                dataset_key="inx_primary",
                model_name="xgboost",
            )
            fold_rows.append(
                {
                    **identity,
                    **policy,
                    "outer_fold": int(outer_fold),
                    "n_train": int(total - len(fold_rows_source)),
                    "n_test": int(len(fold_rows_source)),
                    **fold_bundle["aggregate_metrics"],
                }
            )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(["policy_id", "estimand", "metric"]).reset_index(drop=True)
    fold_metrics = pd.DataFrame(fold_rows).sort_values(["policy_id", "estimand", "outer_fold"]).reset_index(drop=True)
    pivot = aggregate.pivot(index=["policy_id", "policy_name", "n_features", "metric"], columns="estimand", values="value").reset_index()
    _require({FIXED_ESTIMAND, RETUNED_ESTIMAND}.issubset(pivot.columns), "Metric comparison lacks an estimand.")
    pivot = pivot.rename(columns={FIXED_ESTIMAND: "fixed_value", RETUNED_ESTIMAND: "retuned_value"})
    pivot["raw_difference_retuned_minus_fixed"] = pivot["retuned_value"] - pivot["fixed_value"]
    higher_metrics = {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "weighted_precision",
        "macro_recall",
        "weighted_recall",
        "quadratic_weighted_kappa",
        "adjacent_accuracy",
    }
    pivot["better_direction"] = np.where(pivot["metric"].isin(higher_metrics), "higher", "lower")
    pivot["direction_aligned_improvement"] = np.where(
        pivot["better_direction"] == "higher",
        pivot["raw_difference_retuned_minus_fixed"],
        -pivot["raw_difference_retuned_minus_fixed"],
    )
    metric_comparison = pivot.sort_values(["policy_id", "metric"]).reset_index(drop=True)

    feature_lookup = policy_feature_contract.set_index("policy_id")
    _require(feature_lookup.index.is_unique, "Policy feature contract has duplicate policy ids.")
    headline_rows: list[dict[str, Any]] = []
    for policy_id in policy_feature_contract["policy_id"].astype(str):
        source = metric_comparison[metric_comparison["policy_id"] == policy_id].set_index("metric")
        row: dict[str, Any] = {
            "policy_id": policy_id,
            "policy_name": str(feature_lookup.loc[policy_id, "policy_name"]),
            "n_features": int(feature_lookup.loc[policy_id, "n_features"]),
        }
        for metric in HEADLINE_METRICS:
            row[f"fixed_{metric}"] = float(source.loc[metric, "fixed_value"])
            row[f"retuned_{metric}"] = float(source.loc[metric, "retuned_value"])
            row[f"raw_difference_{metric}"] = float(source.loc[metric, "raw_difference_retuned_minus_fixed"])
            row[f"direction_aligned_improvement_{metric}"] = float(source.loc[metric, "direction_aligned_improvement"])
        headline_rows.append(row)
    headline = pd.DataFrame(headline_rows)
    return fold_metrics, aggregate, metric_comparison, headline


def selected_candidate_frequency_v3(selected: pd.DataFrame) -> pd.DataFrame:
    """Summarize policy-specific training-only candidate choices."""

    required = {
        "policy_id",
        "outer_fold",
        "selected_candidate_index",
        "selected_candidate_parameters_json",
        "outer_test_used_for_selection",
    }
    _require(required.issubset(selected.columns), "Selected candidate schema is incomplete.")
    _require(not selected["outer_test_used_for_selection"].astype(bool).any(), "Outer test entered policy selection.")
    _require(not selected.duplicated(["policy_id", "outer_fold"]).any(), "Selected candidate grid has duplicates.")
    grouped = (
        selected.groupby(
            ["policy_id", "policy_name", "selected_candidate_index", "selected_candidate_parameters_json"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "selection_count"})
    )
    totals = selected.groupby("policy_id").size().rename("selection_opportunities")
    grouped["selection_opportunities"] = grouped["policy_id"].map(totals).astype(int)
    grouped["selection_frequency"] = grouped["selection_count"] / grouped["selection_opportunities"]
    return grouped.sort_values(["policy_id", "selected_candidate_index"]).reset_index(drop=True)


def evaluate_policy_retuning_v3(
    source_frame: pd.DataFrame,
    feature_frames: Mapping[str, pd.DataFrame],
    target: pd.Series,
    folds: SharedFoldArtifacts,
    model_definition: Mapping[str, Any],
    primary_schedule: pd.DataFrame,
    fixed_source_oof: pd.DataFrame,
    policy_contract: Mapping[str, Any],
    policy_feature_contract: pd.DataFrame,
    *,
    run_id: str,
    policy_contract_sha256: str,
    scientific_input_sha256: str,
    policy_subset: Sequence[str] | None = None,
    outer_fold_subset: Sequence[int] | None = None,
) -> PolicyRetuningV3Result:
    """Evaluate both policy estimands on one exact persisted fold system."""

    _require(bool(str(run_id).strip()), "run_id must be non-empty.")
    _require(_valid_digest(policy_contract_sha256), "policy_contract_sha256 is invalid.")
    _require(_valid_digest(scientific_input_sha256), "scientific_input_sha256 is invalid.")
    _require(source_frame.index.equals(target.index), "Source/target indices drifted.")
    _require(set(feature_frames) == set(POLICY_IDS), "Policy feature-frame registry drifted.")
    _require(all(frame.index.equals(source_frame.index) for frame in feature_frames.values()), "Policy feature-frame indices drifted.")
    selected_policies = tuple(POLICY_IDS if policy_subset is None else policy_subset)
    _require(selected_policies and set(selected_policies).issubset(POLICY_IDS), "Policy subset is invalid.")
    all_outer_folds = tuple(sorted(folds.outer_assignments["outer_fold"].astype(int).unique()))
    selected_outer_folds = tuple(all_outer_folds if outer_fold_subset is None else sorted(set(map(int, outer_fold_subset))))
    _require(selected_outer_folds and set(selected_outer_folds).issubset(all_outer_folds), "Outer-fold subset is invalid.")
    full_run = selected_policies == POLICY_IDS and selected_outer_folds == all_outer_folds == tuple(range(1, 11))
    evidence_status = "complete_two_estimand_exactly_once_oof" if full_run else "diagnostic_incomplete_never_canonical"
    identity = _base_identity(
        run_id=run_id,
        policy_contract_sha256=policy_contract_sha256,
        scientific_input_sha256=scientific_input_sha256,
        fold_contract_hash=str(folds.contract["fold_contract_hash"]),
    )
    feature_lookup = policy_feature_contract.set_index("policy_id")
    policy_config_records = {str(record["policy_id"]): record for record in policy_contract["fixed_evidence_crosswalk"]}
    schedule = primary_schedule[primary_schedule["model"] == "xgboost"].set_index("outer_fold")
    _require(set(schedule.index.astype(int)) == set(range(1, 11)), "Primary XGBoost schedule is incomplete.")
    candidates = [dict(item) for item in model_definition["candidates"]]
    _require(len(candidates) == 8, "XGBoost candidate registry drifted.")
    random_state = 42

    fixed_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    retuned_rows: list[dict[str, Any]] = []
    for policy_id in selected_policies:
        feature_record = feature_lookup.loc[policy_id].to_dict()
        policy_record = {"policy_id": policy_id, **feature_record}
        crosswalk = policy_config_records[policy_id]
        source_policy = crosswalk["source_policy"]
        if source_policy is not None:
            source_rows = fixed_source_oof[
                (fixed_source_oof["system_id"] == source_policy)
                & (fixed_source_oof["outer_fold"].astype(int).isin(selected_outer_folds))
            ].sort_values("sample_index")
            fixed_rows.extend(
                _prediction_rows(
                    identity=identity,
                    estimand=FIXED_ESTIMAND,
                    policy_record=policy_record,
                    evidence_source="canonical_v2_fixed_policy_oof_exact_feature_set_reuse",
                    source_policy=str(source_policy),
                    sample_indices=source_rows["sample_index"].astype(int).tolist(),
                    outer_fold=source_rows["outer_fold"].astype(int).tolist(),
                    target=source_rows["y_true"].astype(int).tolist(),
                    prediction=source_rows["y_pred"].astype(int).tolist(),
                    probability=source_rows[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float),
                    selected_candidate_index=source_rows["selected_candidate_index"].astype(int).tolist(),
                )
            )
        else:
            for outer_fold in selected_outer_folds:
                test_ids = folds.outer_assignments.loc[
                    folds.outer_assignments["outer_fold"].astype(int) == outer_fold,
                    "sample_index",
                ].astype(int).tolist()
                train_ids = folds.outer_assignments.loc[
                    folds.outer_assignments["outer_fold"].astype(int) != outer_fold,
                    "sample_index",
                ].astype(int).tolist()
                schedule_row = schedule.loc[outer_fold]
                selected_index = int(schedule_row["selected_candidate_index"])
                _require(0 <= selected_index < len(candidates), "Primary schedule candidate index is invalid.")
                estimator = _pipeline(
                    feature_frames[policy_id].loc[train_ids],
                    model_definition=model_definition,
                    candidate_parameters=candidates[selected_index],
                    forbidden_features=tuple(column for column in source_frame.columns if column not in feature_frames[policy_id].columns),
                    random_state=random_state,
                )
                _fit_or_fail(
                    estimator,
                    feature_frames[policy_id].loc[train_ids],
                    target.loc[train_ids],
                    context=f"fixed schedule policy={policy_id}, outer={outer_fold}",
                )
                prediction = np.asarray(estimator.predict(feature_frames[policy_id].loc[test_ids]), dtype=int)
                probability = aligned_predict_proba(estimator, feature_frames[policy_id].loc[test_ids], labels=(2, 3, 4))
                fixed_rows.extend(
                    _prediction_rows(
                        identity=identity,
                        estimand=FIXED_ESTIMAND,
                        policy_record=policy_record,
                        evidence_source="v3_new_outer_train_fit_with_primary_P3_schedule",
                        source_policy=None,
                        sample_indices=test_ids,
                        outer_fold=outer_fold,
                        target=target.loc[test_ids].astype(int).tolist(),
                        prediction=prediction,
                        probability=probability,
                        selected_candidate_index=selected_index,
                    )
                )

        for outer_fold in selected_outer_folds:
            test_ids = folds.outer_assignments.loc[
                folds.outer_assignments["outer_fold"].astype(int) == outer_fold,
                "sample_index",
            ].astype(int).tolist()
            train_ids = folds.outer_assignments.loc[
                folds.outer_assignments["outer_fold"].astype(int) != outer_fold,
                "sample_index",
            ].astype(int).tolist()
            scoped_inner = folds.inner_assignments[
                folds.inner_assignments["outer_fold"].astype(int) == outer_fold
            ]
            _require(set(scoped_inner["sample_index"].astype(int)) == set(train_ids), "Inner assignments differ from outer training.")
            macro_means: list[float] = []
            qwk_means: list[float] = []
            scoped_candidate_rows: list[dict[str, Any]] = []
            for candidate_index, candidate in enumerate(candidates):
                macro_scores: list[float] = []
                qwk_scores: list[float] = []
                for inner_fold in range(1, 6):
                    validation_ids = scoped_inner.loc[
                        scoped_inner["inner_fold"].astype(int) == inner_fold,
                        "sample_index",
                    ].astype(int).tolist()
                    development_ids = sorted(set(train_ids) - set(validation_ids))
                    estimator = _pipeline(
                        feature_frames[policy_id].loc[development_ids],
                        model_definition=model_definition,
                        candidate_parameters=candidate,
                        forbidden_features=tuple(column for column in source_frame.columns if column not in feature_frames[policy_id].columns),
                        random_state=random_state,
                    )
                    _fit_or_fail(
                        estimator,
                        feature_frames[policy_id].loc[development_ids],
                        target.loc[development_ids],
                        context=(
                            f"retune policy={policy_id}, outer={outer_fold}, "
                            f"candidate={candidate_index}, inner={inner_fold}"
                        ),
                    )
                    prediction = np.asarray(estimator.predict(feature_frames[policy_id].loc[validation_ids]), dtype=int)
                    probability = aligned_predict_proba(estimator, feature_frames[policy_id].loc[validation_ids], labels=(2, 3, 4))
                    macro_f1, qwk = _selection_scores(target.loc[validation_ids], prediction, probability)
                    macro_scores.append(macro_f1)
                    qwk_scores.append(qwk)
                macro_mean = float(np.mean(macro_scores))
                qwk_mean = float(np.mean(qwk_scores))
                macro_means.append(macro_mean)
                qwk_means.append(qwk_mean)
                scoped_candidate_rows.append(
                    {
                        **identity,
                        "estimand": RETUNED_ESTIMAND,
                        "policy_id": policy_id,
                        "policy_name": str(policy_record["policy_name"]),
                        "n_features": int(policy_record["n_features"]),
                        "outer_fold": outer_fold,
                        "model": "xgboost",
                        "candidate_index": candidate_index,
                        "parameters_json": json.dumps(candidate, sort_keys=True, separators=(",", ":")),
                        "inner_macro_f1_scores_json": json.dumps(macro_scores),
                        "inner_macro_f1_mean": macro_mean,
                        "inner_qwk_scores_json": json.dumps(qwk_scores),
                        "inner_qwk_mean": qwk_mean,
                        "n_inner_folds": 5,
                        "candidate_status": "complete",
                        "outer_test_used_for_selection": False,
                    }
                )
            selected_index = select_candidate_index(
                macro_means,
                qwk_means,
                practical_tie_tolerance=0.001,
                better_direction="higher",
            )
            for row in scoped_candidate_rows:
                row["selected_by_protocol"] = row["candidate_index"] == selected_index
            candidate_rows.extend(scoped_candidate_rows)
            selected_candidate = candidates[selected_index]
            selected_rows.append(
                {
                    **identity,
                    "estimand": RETUNED_ESTIMAND,
                    "policy_id": policy_id,
                    "policy_name": str(policy_record["policy_name"]),
                    "n_features": int(policy_record["n_features"]),
                    "outer_fold": outer_fold,
                    "model": "xgboost",
                    "selected_candidate_index": selected_index,
                    "selected_candidate_parameters_json": json.dumps(selected_candidate, sort_keys=True, separators=(",", ":")),
                    "fixed_parameters_json": json.dumps(dict(model_definition["fixed_params"]), sort_keys=True, separators=(",", ":")),
                    "selected_inner_macro_f1_mean": macro_means[selected_index],
                    "selected_inner_qwk_mean": qwk_means[selected_index],
                    "outer_test_used_for_selection": False,
                }
            )
            estimator = _pipeline(
                feature_frames[policy_id].loc[train_ids],
                model_definition=model_definition,
                candidate_parameters=selected_candidate,
                forbidden_features=tuple(column for column in source_frame.columns if column not in feature_frames[policy_id].columns),
                random_state=random_state,
            )
            _fit_or_fail(
                estimator,
                feature_frames[policy_id].loc[train_ids],
                target.loc[train_ids],
                context=f"retuned outer fit policy={policy_id}, outer={outer_fold}",
            )
            prediction = np.asarray(estimator.predict(feature_frames[policy_id].loc[test_ids]), dtype=int)
            probability = aligned_predict_proba(estimator, feature_frames[policy_id].loc[test_ids], labels=(2, 3, 4))
            retuned_rows.extend(
                _prediction_rows(
                    identity=identity,
                    estimand=RETUNED_ESTIMAND,
                    policy_record=policy_record,
                    evidence_source="v3_policy_specific_inner_selection_and_outer_refit",
                    source_policy=None,
                    sample_indices=test_ids,
                    outer_fold=outer_fold,
                    target=target.loc[test_ids].astype(int).tolist(),
                    prediction=prediction,
                    probability=probability,
                    selected_candidate_index=selected_index,
                )
            )

    fixed = pd.DataFrame(fixed_rows).sort_values(["policy_id", "sample_index"]).reset_index(drop=True)
    retuned = pd.DataFrame(retuned_rows).sort_values(["policy_id", "sample_index"]).reset_index(drop=True)
    combined = pd.concat([fixed, retuned], ignore_index=True).sort_values(["estimand", "policy_id", "sample_index"]).reset_index(drop=True)
    candidate_frame = pd.DataFrame(candidate_rows).sort_values(["policy_id", "outer_fold", "candidate_index"]).reset_index(drop=True)
    selected_frame = pd.DataFrame(selected_rows).sort_values(["policy_id", "outer_fold"]).reset_index(drop=True)
    _validate_oof(
        combined,
        folds,
        policies=selected_policies,
        outer_folds=selected_outer_folds,
        full_run=full_run,
        expected_sample_count=len(source_frame),
    )
    _require(not candidate_frame["outer_test_used_for_selection"].astype(bool).any(), "Outer test entered candidate search.")
    selected_flags = candidate_frame.groupby(["policy_id", "outer_fold"])["selected_by_protocol"].sum()
    _require(selected_flags.eq(1).all(), "A policy/fold lacks exactly one selected candidate.")
    _require(not selected_frame["outer_test_used_for_selection"].astype(bool).any(), "Outer test entered selected records.")

    if "P3" in selected_policies:
        fixed_p3 = fixed[fixed["policy_id"] == "P3"].set_index("sample_index").sort_index()
        retuned_p3 = retuned[retuned["policy_id"] == "P3"].set_index("sample_index").sort_index()
        _require(np.array_equal(fixed_p3["y_pred"].to_numpy(int), retuned_p3["y_pred"].to_numpy(int)), "Retuned P3 labels do not replay the primary benchmark.")
        _require(np.array_equal(fixed_p3["selected_candidate_index"].to_numpy(int), retuned_p3["selected_candidate_index"].to_numpy(int)), "Retuned P3 candidate schedule drifted from the primary benchmark.")
        maximum_probability_error = float(
            np.max(
                np.abs(
                    fixed_p3[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float)
                    - retuned_p3[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float)
                )
            )
        )
        _require(maximum_probability_error <= 1e-12, f"Retuned P3 probability replay exceeded tolerance: {maximum_probability_error}.")

    fold_metrics, aggregate, comparison, headline = summarize_policy_oof_v3(
        combined,
        policy_feature_contract=policy_feature_contract[
            policy_feature_contract["policy_id"].isin(selected_policies)
        ],
        total_sample_count=len(source_frame),
    )
    frequency = selected_candidate_frequency_v3(selected_frame)
    return PolicyRetuningV3Result(
        candidate_search_results=candidate_frame,
        selected_hyperparameters=selected_frame,
        selected_candidate_frequency=frequency,
        fixed_oof_predictions=fixed,
        retuned_oof_predictions=retuned,
        combined_oof_predictions=combined,
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate,
        metric_comparison=comparison,
        headline_policy_comparison=headline,
        policy_feature_contract=policy_feature_contract,
        evidence_status=evidence_status,
    )


def _prepare_inputs(contract_path: Path) -> tuple[Any, ...]:
    contract_receipt = validate_policy_retuning_contract_v3(contract_path)
    contract = _load_json(contract_path)
    sources = contract["source_contracts"]
    for name, record in sources.items():
        source_path = Path(str(record["path"]))
        _require(source_path.is_file(), f"Required policy source is absent: {name}.")
        _require(sha256_file(source_path) == record["sha256"], f"Policy source hash drifted: {name}.")
    canonical = load_canonical_dataset(
        sources["canonical_loader_config"]["path"],
        "inx_primary",
        sources["acquisition_manifest"]["path"],
        allow_download=False,
    )
    _require(canonical.receipt["actual_sha256"] == "b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a", "Canonical dataset hash drifted.")
    feature_contract = _load_json(Path(sources["feature_availability"]["path"]))
    frames, feature_rows = _feature_frames(canonical.frame, feature_contract)
    folds = read_shared_folds(CANONICAL_V2_ROOT / "core/shared_folds")
    _require(folds.contract["dataset_sha256"] == canonical.receipt["actual_sha256"], "Fold/dataset identity drifted.")
    model_definition = validate_benchmark_config(load_config(sources["xgboost_candidate_registry"]["path"]))["models"]["xgboost"]
    primary_schedule = pd.read_csv(sources["canonical_v2_selected_hyperparameters"]["path"])
    fixed_source_oof = pd.read_csv(sources["canonical_v2_fixed_policy_oof"]["path"])
    target = canonical.frame["PerformanceRating"].astype(int)
    return (
        contract,
        contract_receipt,
        canonical,
        frames,
        feature_rows,
        target,
        folds,
        model_definition,
        primary_schedule,
        fixed_source_oof,
    )


def preflight_policy_retuning_v3(
    *,
    contract_path: Path | str = DEFAULT_POLICY_RETUNING_CONTRACT_PATH,
) -> dict[str, Any]:
    """Validate all real-data sources and grids without fitting a model."""

    (
        _,
        receipt,
        canonical,
        frames,
        feature_rows,
        target,
        folds,
        model_definition,
        primary_schedule,
        fixed_source_oof,
    ) = _prepare_inputs(Path(contract_path))
    _require(set(target.unique()) == {2, 3, 4}, "Target support drifted.")
    _require(len(primary_schedule[primary_schedule["model"] == "xgboost"]) == 10, "Primary schedule count drifted.")
    _require(len(fixed_source_oof[fixed_source_oof["system_id"].isin([
        "full_feature_upper_bound",
        "no_salary_hike_no_attrition_sensitive_retaining_audit",
        "no_salary_hike_no_attrition",
        "no_salary_hike_no_attrition_no_department",
    ])]) == 4800, "Reusable fixed OOF count drifted.")
    return {
        "status": "passed",
        "contract_sha256": receipt["contract_sha256"],
        "dataset_sha256": canonical.receipt["actual_sha256"],
        "sample_count": len(target),
        "target_support": {str(label): int(count) for label, count in target.value_counts().sort_index().items()},
        "policy_feature_counts": feature_rows.set_index("policy_id")["n_features"].astype(int).to_dict(),
        "outer_splits": int(folds.contract["outer_splits"]),
        "inner_splits": int(folds.contract["inner_splits"]),
        "fold_contract_hash": folds.contract["fold_contract_hash"],
        "candidate_count": len(model_definition["candidates"]),
        "primary_schedule_rows": 10,
        "reusable_fixed_oof_rows": 4800,
        "planned_new_estimator_fit_calls": receipt["planned_new_estimator_fit_calls"],
        "model_fit_count": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def diagnostic_policy_retuning_v3(
    *,
    contract_path: Path | str = DEFAULT_POLICY_RETUNING_CONTRACT_PATH,
    policy_id: str = "P3",
    outer_fold: int = 1,
) -> dict[str, Any]:
    """Run one explicit in-memory policy/fold diagnostic without persistence."""

    with enforce_offline_runtime() as offline_state:
        (
            contract,
            receipt,
            canonical,
            frames,
            feature_rows,
            target,
            folds,
            model_definition,
            primary_schedule,
            fixed_source_oof,
        ) = _prepare_inputs(Path(contract_path))
        result = evaluate_policy_retuning_v3(
            canonical.frame,
            frames,
            target,
            folds,
            model_definition,
            primary_schedule,
            fixed_source_oof,
            contract,
            feature_rows,
            run_id=f"diagnostic_{policy_id}_fold{outer_fold}",
            policy_contract_sha256=receipt["contract_sha256"],
            scientific_input_sha256="0" * 64,
            policy_subset=(policy_id,),
            outer_fold_subset=(outer_fold,),
        )
        runtime = offline_state.receipt()
    return {
        "status": result.evidence_status,
        "policy_id": policy_id,
        "outer_fold": outer_fold,
        "candidate_search_rows": len(result.candidate_search_results),
        "selected_hyperparameter_rows": len(result.selected_hyperparameters),
        "fixed_oof_rows": len(result.fixed_oof_predictions),
        "retuned_oof_rows": len(result.retuned_oof_predictions),
        "runtime_policy": runtime,
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


def _run_policy_retuning_v3_impl(
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
        frames,
        feature_rows,
        target,
        folds,
        model_definition,
        primary_schedule,
        fixed_source_oof,
    ) = _prepare_inputs(contract_path)
    _require(not output_dir.exists(), f"Output destination already exists: {output_dir}.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        implementation_paths = (
            Path("src/experiments/policy_retuning_v3.py"),
            Path("src/governance/policy_retuning_contract_v3.py"),
            Path("src/experiments/shared_folds.py"),
            Path("src/models/canonical_models.py"),
            Path("src/models/ordinal_evaluation_v3.py"),
        )
        scientific_inputs = {
            "git_identity": git_identity,
            "source_tree_hash": source_tree_hash(PROJECT_ROOT),
            "policy_contract_sha256": contract_receipt["contract_sha256"],
            "dataset_sha256": canonical.receipt["actual_sha256"],
            "fold_contract_hash": folds.contract["fold_contract_hash"],
            "bound_source_hashes": {
                name: str(record["sha256"])
                for name, record in contract["source_contracts"].items()
            },
            "implementation_hashes": {
                path.as_posix(): sha256_file(path) for path in implementation_paths
            },
        }
        scientific_input_sha256 = _canonical_json_sha256(scientific_inputs)
        result = evaluate_policy_retuning_v3(
            canonical.frame,
            frames,
            target,
            folds,
            model_definition,
            primary_schedule,
            fixed_source_oof,
            contract,
            feature_rows,
            run_id=run_id,
            policy_contract_sha256=contract_receipt["contract_sha256"],
            scientific_input_sha256=scientific_input_sha256,
        )
        frames_to_write = {
            "aggregate_metrics.csv": result.aggregate_metrics,
            "candidate_search_results.csv": result.candidate_search_results,
            "combined_oof_predictions.csv": result.combined_oof_predictions,
            "fixed_oof_predictions.csv": result.fixed_oof_predictions,
            "fold_metrics.csv": result.fold_metrics,
            "headline_policy_comparison.csv": result.headline_policy_comparison,
            "metric_comparison.csv": result.metric_comparison,
            "policy_feature_contract.csv": result.policy_feature_contract,
            "retuned_oof_predictions.csv": result.retuned_oof_predictions,
            "selected_candidate_frequency.csv": result.selected_candidate_frequency,
            "selected_hyperparameters.csv": result.selected_hyperparameters,
        }
        for filename, frame in frames_to_write.items():
            frame.to_csv(staging / filename, index=False)
        output_hashes = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        fixed_counts = result.fixed_oof_predictions["evidence_source"].value_counts().to_dict()
        metadata = {
            "schema_version": 1,
            "stage": "policy_retuning_v3",
            "status": "complete",
            "evidence_status": result.evidence_status,
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy_contract_sha256": contract_receipt["contract_sha256"],
            "scientific_input_sha256": scientific_input_sha256,
            "scientific_inputs": scientific_inputs,
            "git_identity": git_identity,
            "source_tree_hash": scientific_inputs["source_tree_hash"],
            "dataset_sha256": canonical.receipt["actual_sha256"],
            "fold_contract_hash": folds.contract["fold_contract_hash"],
            "model": "xgboost",
            "policy_count": 6,
            "sample_count": 1200,
            "outer_folds": 10,
            "inner_folds": 5,
            "candidate_count": 8,
            "planned_new_estimator_fit_calls": 2480,
            "candidate_search_row_count": len(result.candidate_search_results),
            "selected_hyperparameter_row_count": len(result.selected_hyperparameters),
            "fixed_oof_row_count": len(result.fixed_oof_predictions),
            "retuned_oof_row_count": len(result.retuned_oof_predictions),
            "combined_oof_row_count": len(result.combined_oof_predictions),
            "fold_metric_row_count": len(result.fold_metrics),
            "aggregate_metric_row_count": len(result.aggregate_metrics),
            "metric_comparison_row_count": len(result.metric_comparison),
            "headline_policy_row_count": len(result.headline_policy_comparison),
            "fixed_evidence_source_counts": fixed_counts,
            "outer_test_used_for_selection": False,
            "seed_or_policy_selected_from_results": False,
            "employee_level_outputs_publication_authorized": False,
            "runtime_policy": offline_state.receipt(),
            "network_calls": 0,
            "paid_api_calls": 0,
            "output_hashes": output_hashes,
        }
        _require(_clean_git_identity() == git_identity, "Git identity changed during execution.")
        _require(source_tree_hash(PROJECT_ROOT) == scientific_inputs["source_tree_hash"], "Scientific source tree changed during execution.")
        repeat_receipt = validate_policy_retuning_contract_v3(contract_path)
        _require(repeat_receipt["contract_sha256"] == contract_receipt["contract_sha256"], "Policy-retuning contract changed during execution.")
        for name, record in contract["source_contracts"].items():
            _require(sha256_file(record["path"]) == record["sha256"], f"Policy source changed during execution: {name}.")
        _write_json(staging / "stage_metadata.json", metadata)
        _require({path.name for path in staging.iterdir() if path.is_file()} == EXPECTED_LOCAL_FILES, "Policy-retuning local output inventory drifted.")
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
        "policy_contract_sha256": contract_receipt["contract_sha256"],
        "scientific_input_sha256": scientific_input_sha256,
        "policy_count": 6,
        "combined_oof_row_count": len(result.combined_oof_predictions),
        "planned_new_estimator_fit_calls": 2480,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def run_policy_retuning_v3(
    *,
    contract_path: Path | str = DEFAULT_POLICY_RETUNING_CONTRACT_PATH,
    output_dir: Path | str,
    run_id: str,
) -> dict[str, Any]:
    """Execute a full clean-commit run under the process-wide offline boundary."""

    with enforce_offline_runtime() as offline_state:
        return _run_policy_retuning_v3_impl(
            contract_path=Path(contract_path),
            output_dir=Path(output_dir),
            run_id=run_id,
            offline_state=offline_state,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_POLICY_RETUNING_CONTRACT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_LOCAL_RUN_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--diagnostic-policy", choices=POLICY_IDS)
    parser.add_argument("--diagnostic-outer-fold", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    diagnostic_requested = args.diagnostic_policy is not None or args.diagnostic_outer_fold is not None
    _require(not (args.preflight_only and diagnostic_requested), "Preflight and diagnostic modes are mutually exclusive.")
    if args.preflight_only:
        receipt = preflight_policy_retuning_v3(contract_path=args.contract)
    elif diagnostic_requested:
        _require(args.diagnostic_policy is not None and args.diagnostic_outer_fold is not None, "Both diagnostic policy and outer fold are required.")
        receipt = diagnostic_policy_retuning_v3(
            contract_path=args.contract,
            policy_id=args.diagnostic_policy,
            outer_fold=args.diagnostic_outer_fold,
        )
    else:
        _require(isinstance(args.run_id, str) and bool(args.run_id.strip()), "--run-id is required unless preflight/diagnostic mode is used.")
        receipt = run_policy_retuning_v3(
            contract_path=args.contract,
            output_dir=args.output_root / args.run_id / "policy_retuning",
            run_id=args.run_id,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_LOCAL_RUN_ROOT",
    "EXPECTED_LOCAL_FILES",
    "FIXED_ESTIMAND",
    "PolicyRetuningV3Error",
    "PolicyRetuningV3Result",
    "RETUNED_ESTIMAND",
    "diagnostic_policy_retuning_v3",
    "evaluate_policy_retuning_v3",
    "preflight_policy_retuning_v3",
    "run_policy_retuning_v3",
    "selected_candidate_frequency_v3",
    "summarize_policy_oof_v3",
]
