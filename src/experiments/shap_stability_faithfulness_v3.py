"""Run v3 grouped-SHAP stability and model-level deletion faithfulness evidence."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
import shap
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from src.data.canonical_loader import load_canonical_dataset, sha256_file
from src.experiments.benchmark_artifact_contract import read_xgboost_oof_artifacts
from src.experiments.manuscript_model_benchmark import validate_benchmark_config
from src.explainability.canonical_shap_axis import (
    build_canonical_shap_axis,
    group_canonical_shap_values,
    normalize_multiclass_shap_values,
)
from src.governance.manuscript_contract import source_tree_hash
from src.governance.offline_runtime import enforce_offline_runtime
from src.governance.shap_stability_faithfulness_contract_v3 import (
    CANONICAL_V2_ROOT,
    DEFAULT_SHAP_STABILITY_FAITHFULNESS_CONTRACT,
    validate_shap_stability_faithfulness_contract_v3,
)
from src.models.canonical_models import aligned_predict_proba, build_model_pipeline
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_LOCAL_RUN_ROOT = Path("reports/major_revision_v3_runs")
EXPECTED_LOCAL_FILES = frozenset(
    {
        "additivity_checks.csv",
        "aggregation_receipt.json",
        "deletion_auc_contrast.csv",
        "deletion_auc_sample.csv",
        "deletion_auc_summary.csv",
        "faithfulness_contrasts.csv",
        "faithfulness_feature_frequency.csv",
        "faithfulness_sample_results.csv",
        "faithfulness_summary.csv",
        "fold_feature_importance.csv",
        "stability_pairwise.csv",
        "stability_run_rankings.csv",
        "stability_summary.csv",
        "stage_metadata.json",
    }
)
TOP_K_VALUES = (5, 10, 15)
DELETION_COUNTS = (1, 3, 5)
LABELS = (2, 3, 4)
FIT_THREAD_LIMIT = 1


class ShapStabilityFaithfulnessV3Error(RuntimeError):
    """Raised when a Phase 2A execution invariant fails."""


@dataclass(frozen=True)
class ExplainedRun:
    fold_importance: pd.DataFrame
    additivity_checks: pd.DataFrame
    rankings: pd.DataFrame
    grouped_oof: np.ndarray | None = None


@dataclass(frozen=True)
class Phase2AResult:
    fold_feature_importance: pd.DataFrame
    additivity_checks: pd.DataFrame
    stability_run_rankings: pd.DataFrame
    stability_pairwise: pd.DataFrame
    stability_summary: pd.DataFrame
    faithfulness_sample_results: pd.DataFrame
    faithfulness_summary: pd.DataFrame
    faithfulness_contrasts: pd.DataFrame
    deletion_auc_sample: pd.DataFrame
    deletion_auc_summary: pd.DataFrame
    deletion_auc_contrast: pd.DataFrame
    faithfulness_feature_frequency: pd.DataFrame
    aggregation_receipt: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ShapStabilityFaithfulnessV3Error(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ShapStabilityFaithfulnessV3Error(f"Could not read {path.as_posix()}: {exc}") from exc
    _require(isinstance(payload, dict), f"{path.name} must contain a JSON object.")
    return payload


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clean_git_identity() -> dict[str, str]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ShapStabilityFaithfulnessV3Error(f"Could not establish Git identity: {exc}") from exc
    _require(len(head) == 40 and all(c in "0123456789abcdef" for c in head), "Git HEAD digest is invalid.")
    _require(not status, f"Scientific execution requires a clean worktree; status={status.splitlines()[:10]}.")
    return {"commit": head, "branch": branch}


def _p3_features(frame: pd.DataFrame, feature_contract: Mapping[str, Any]) -> tuple[pd.DataFrame, tuple[str, ...]]:
    all_features = [str(record["feature_name"]) for record in feature_contract["features"]]
    _require(set(all_features) == set(frame.columns), "Feature contract/source schema drifted.")
    policies = {str(record["policy_id"]): record for record in feature_contract["policies"]}
    excluded = tuple(map(str, policies["P3"]["excluded_features"]))
    retained = [feature for feature in all_features if feature not in set(excluded)]
    _require(len(retained) == 20 and "PerformanceRating" not in retained and "EmpNumber" not in retained, "P3 feature set drifted.")
    return frame.loc[:, retained].copy(), excluded


def _prepare_inputs(contract_path: Path) -> tuple[Any, ...]:
    receipt = validate_shap_stability_faithfulness_contract_v3(contract_path)
    contract = _load_json(contract_path)
    sources = contract["source_contracts"]
    canonical = load_canonical_dataset(
        sources["canonical_loader_config"]["path"],
        "inx_primary",
        sources["acquisition_manifest"]["path"],
        allow_download=False,
    )
    feature_contract = _load_json(Path(sources["feature_availability"]["path"]))
    features, excluded = _p3_features(canonical.frame, feature_contract)
    identity = contract["canonical_identity"]
    artifacts = read_xgboost_oof_artifacts(
        CANONICAL_V2_ROOT / "core/shared_folds",
        CANONICAL_V2_ROOT / "core/model_benchmarks",
        expected_run_id=identity["run_id"],
        expected_config_hash=identity["config_hash"],
        expected_scientific_input_hash=identity["scientific_input_hash"],
        expected_feature_columns=features.columns,
        expected_labels=LABELS,
    )
    model_definition = validate_benchmark_config(load_config(sources["xgboost_candidate_registry"]["path"]))["models"]["xgboost"]
    target = canonical.frame["PerformanceRating"].astype(int)
    return contract, receipt, canonical, features, excluded, target, artifacts, model_definition


def preflight_shap_stability_faithfulness_v3(
    *, contract_path: Path | str = DEFAULT_SHAP_STABILITY_FAITHFULNESS_CONTRACT,
) -> dict[str, Any]:
    contract, receipt, canonical, features, _, target, artifacts, model_definition = _prepare_inputs(Path(contract_path))
    return {
        "status": "passed",
        "contract_sha256": receipt["contract_sha256"],
        "dataset_sha256": canonical.receipt["actual_sha256"],
        "sample_count": len(features),
        "feature_count": features.shape[1],
        "target_support": {str(k): int(v) for k, v in target.value_counts().sort_index().items()},
        "outer_model_count": len(artifacts.fold_models),
        "candidate_count": len(model_definition["candidates"]),
        "seed_stability_new_fit_calls": contract["seed_stability"]["new_estimator_fit_calls"],
        "resampling_stability_new_fit_calls": contract["resampling_stability"]["new_estimator_fit_calls"],
        "planned_new_estimator_fit_calls": contract["computational_scope"]["total_new_estimator_fit_calls"],
        "model_fit_count": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _fit_pipeline(
    training_features: pd.DataFrame,
    training_target: pd.Series,
    *,
    model_definition: Mapping[str, Any],
    candidate: Mapping[str, Any],
    forbidden_features: Sequence[str],
    model_seed: int,
    context: str,
) -> Any:
    pipeline = build_model_pipeline(
        "xgboost",
        training_features,
        fixed_parameters=dict(model_definition["fixed_params"]),
        candidate_parameters=dict(candidate),
        random_state=int(model_seed),
        forbidden_features=tuple(forbidden_features),
    )
    try:
        with threadpool_limits(limits=FIT_THREAD_LIMIT):
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                pipeline.fit(training_features, training_target)
    except Exception as exc:
        raise ShapStabilityFaithfulnessV3Error(f"{context} failed: {type(exc).__name__}: {exc}") from exc
    return pipeline


def _expected_values(explainer: Any, n_classes: int) -> np.ndarray:
    values = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    _require(values.shape == (n_classes,) and np.all(np.isfinite(values)), "TreeSHAP expected-value shape drifted.")
    return values


def _explain_pipeline(
    pipeline: Any,
    test_features: pd.DataFrame,
    *,
    raw_feature_order: Sequence[str],
    forbidden_features: Sequence[str],
    maximum_additivity_error: float,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["model"]
    transformed = preprocessor.transform(test_features)
    axis = build_canonical_shap_axis(
        preprocessor,
        raw_feature_order=raw_feature_order,
        forbidden_features=forbidden_features,
    )
    axis.validate_transformed_matrix(transformed, n_samples=len(test_features))
    explainer = shap.TreeExplainer(
        classifier.model_,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    raw_values = explainer.shap_values(transformed, check_additivity=True)
    normalized = normalize_multiclass_shap_values(
        raw_values,
        n_samples=len(test_features),
        n_classes=len(LABELS),
        n_transformed_features=len(axis.transformed_feature_names),
    )
    grouped = group_canonical_shap_values(normalized, axis)
    _require(tuple(axis.raw_feature_names) == tuple(raw_feature_order), "Grouped SHAP raw-feature order drifted.")
    margins = np.asarray(classifier.model_.predict(transformed, output_margin=True), dtype=float)
    expected = _expected_values(explainer, len(LABELS))
    additive = normalized.sum(axis=2) + expected.reshape(1, -1)
    maximum_error = float(np.max(np.abs(additive - margins)))
    grouping_error = float(np.max(np.abs(normalized.sum(axis=2) - grouped.sum(axis=2))))
    _require(maximum_error <= maximum_additivity_error, f"TreeSHAP raw-margin additivity exceeded tolerance: {maximum_error}.")
    _require(grouping_error <= 1e-12, f"Grouped SHAP sum preservation exceeded tolerance: {grouping_error}.")
    return grouped, {
        "n_test": len(test_features),
        "n_transformed_features": len(axis.transformed_feature_names),
        "maximum_raw_margin_additivity_error": maximum_error,
        "maximum_grouped_sum_error": grouping_error,
        "feature_perturbation": str(explainer.feature_perturbation),
        "model_output": "raw_margin",
    }


def _rank_from_fold_importance(fold_importance: pd.DataFrame) -> pd.DataFrame:
    keys = ["stability_type", "run_label", "model_seed", "subsample_seed"]
    rows: list[dict[str, Any]] = []
    for key, group in fold_importance.groupby(keys, dropna=False, sort=True):
        weighted = (
            group.assign(weighted=lambda x: x["mean_abs_grouped_shap"] * x["n_test"])
            .groupby("feature", as_index=False)
            .agg(weighted_sum=("weighted", "sum"), n_oof=("n_test", "sum"))
        )
        weighted["mean_abs_grouped_shap"] = weighted["weighted_sum"] / weighted["n_oof"]
        weighted = weighted.sort_values(["mean_abs_grouped_shap", "feature"], ascending=[False, True]).reset_index(drop=True)
        for rank, record in enumerate(weighted.itertuples(index=False), start=1):
            rows.append(
                {
                    "stability_type": key[0],
                    "run_label": key[1],
                    "model_seed": key[2],
                    "subsample_seed": key[3],
                    "feature": record.feature,
                    "mean_abs_grouped_shap": float(record.mean_abs_grouped_shap),
                    "rank": rank,
                    "n_oof": int(record.n_oof),
                }
            )
    return pd.DataFrame(rows)


def explain_stability_run_v3(
    *,
    stability_type: str,
    run_label: str,
    model_seed: int,
    subsample_seed: int | None,
    features: pd.DataFrame,
    target: pd.Series,
    artifacts: Any,
    model_definition: Mapping[str, Any],
    forbidden_features: Sequence[str],
    maximum_additivity_error: float,
    retain_grouped_oof: bool = False,
) -> ExplainedRun:
    """Fit/reuse one ten-fold model set and calculate global OOF grouped SHAP."""

    feature_names = list(features.columns)
    grouped_oof = np.full((len(features), len(LABELS), len(feature_names)), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    candidates = [dict(value) for value in model_definition["candidates"]]
    for outer_fold, fold_model in artifacts.fold_models.items():
        test_ids = list(map(int, fold_model.test_sample_indices))
        train_ids = sorted(set(map(int, features.index)) - set(test_ids))
        selected_index = int(fold_model.selected_candidate_index)
        if stability_type == "canonical_reference":
            pipeline = fold_model.pipeline
            used_train_ids = train_ids
            model_source = "exact_canonical_v2_outer_model"
        else:
            used_train_ids = train_ids
            if stability_type == "outer_train_resample":
                _require(subsample_seed is not None, "Resampling stability requires a subsample seed.")
                used_train_ids, _ = train_test_split(
                    train_ids,
                    train_size=0.8,
                    random_state=int(subsample_seed) + int(outer_fold),
                    stratify=target.loc[train_ids],
                )
                used_train_ids = sorted(map(int, used_train_ids))
            pipeline = _fit_pipeline(
                features.loc[used_train_ids],
                target.loc[used_train_ids],
                model_definition=model_definition,
                candidate=candidates[selected_index],
                forbidden_features=forbidden_features,
                model_seed=model_seed,
                context=f"{stability_type}/{run_label}/outer_fold={outer_fold}",
            )
            model_source = "new_contract_bound_refit"
        grouped, receipt = _explain_pipeline(
            pipeline,
            features.loc[test_ids],
            raw_feature_order=feature_names,
            forbidden_features=forbidden_features,
            maximum_additivity_error=maximum_additivity_error,
        )
        grouped_oof[np.asarray(test_ids, dtype=int)] = grouped
        importance = np.mean(np.abs(grouped), axis=(0, 1))
        membership_hash = _canonical_json_sha256(used_train_ids)
        for feature, value in zip(feature_names, importance):
            fold_rows.append(
                {
                    "stability_type": stability_type,
                    "run_label": run_label,
                    "model_seed": int(model_seed),
                    "subsample_seed": subsample_seed,
                    "outer_fold": int(outer_fold),
                    "feature": feature,
                    "mean_abs_grouped_shap": float(value),
                    "n_train": len(used_train_ids),
                    "n_test": len(test_ids),
                    "training_membership_sha256": membership_hash,
                    "selected_candidate_index": selected_index,
                    "model_source": model_source,
                }
            )
        check_rows.append(
            {
                "stability_type": stability_type,
                "run_label": run_label,
                "model_seed": int(model_seed),
                "subsample_seed": subsample_seed,
                "outer_fold": int(outer_fold),
                "n_train": len(used_train_ids),
                "training_membership_sha256": membership_hash,
                "selected_candidate_index": selected_index,
                **receipt,
            }
        )
    _require(not np.isnan(grouped_oof).any(), f"OOF SHAP coverage is incomplete for {run_label}.")
    fold_frame = pd.DataFrame(fold_rows)
    rankings = _rank_from_fold_importance(fold_frame)
    return ExplainedRun(
        fold_importance=fold_frame,
        additivity_checks=pd.DataFrame(check_rows),
        rankings=rankings,
        grouped_oof=grouped_oof if retain_grouped_oof else None,
    )


def stability_pairwise_v3(rankings: pd.DataFrame) -> pd.DataFrame:
    """Compute prespecified top-k Jaccard and all-feature Spearman values."""

    rows: list[dict[str, Any]] = []
    for stability_type, scoped in rankings.groupby("stability_type", sort=True):
        by_run = {
            str(label): group.sort_values("rank")["feature"].astype(str).tolist()
            for label, group in scoped.groupby("run_label", sort=True)
        }
        for left, right in itertools.combinations(sorted(by_run), 2):
            left_features, right_features = by_run[left], by_run[right]
            _require(set(left_features) == set(right_features), "Stability feature universe drifted.")
            right_rank = {feature: rank for rank, feature in enumerate(right_features, start=1)}
            rho = float(spearmanr(range(1, len(left_features) + 1), [right_rank[f] for f in left_features]).statistic)
            for top_k in TOP_K_VALUES:
                left_set, right_set = set(left_features[:top_k]), set(right_features[:top_k])
                rows.append(
                    {
                        "stability_type": stability_type,
                        "left_run": left,
                        "right_run": right,
                        "top_k": top_k,
                        "jaccard": len(left_set & right_set) / len(left_set | right_set),
                        "all_feature_spearman": rho,
                        "pair_independence_assumed": False,
                        "confidence_interval_applicable": False,
                    }
                )
    return pd.DataFrame(rows)


def stability_summary_v3(pairwise: pd.DataFrame, canonical_fold_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (stability_type, top_k), values in pairwise.groupby(["stability_type", "top_k"], sort=True):
        rows.append(
            {
                "stability_type": stability_type,
                "top_k": int(top_k),
                "n_pairs": len(values),
                "jaccard_mean": float(values["jaccard"].mean()),
                "jaccard_sample_sd": float(values["jaccard"].std(ddof=1)),
                "jaccard_median": float(values["jaccard"].median()),
                "jaccard_min": float(values["jaccard"].min()),
                "jaccard_max": float(values["jaccard"].max()),
                "spearman_mean": float(values["all_feature_spearman"].mean()),
                "spearman_sample_sd": float(values["all_feature_spearman"].std(ddof=1)),
                "spearman_median": float(values["all_feature_spearman"].median()),
                "spearman_min": float(values["all_feature_spearman"].min()),
                "spearman_max": float(values["all_feature_spearman"].max()),
                "pair_independence_assumed": False,
                "confidence_interval_applicable": False,
            }
        )
    for record in canonical_fold_summary.itertuples(index=False):
        rows.append(
            {
                "stability_type": "canonical_outer_fold_pair",
                "top_k": int(record.top_k),
                "n_pairs": int(record.n_fold_pairs),
                "jaccard_mean": float(record.jaccard_mean),
                "jaccard_sample_sd": float(record.jaccard_std),
                "jaccard_median": float(record.jaccard_median),
                "jaccard_min": float(record.jaccard_min),
                "jaccard_max": float(record.jaccard_max),
                "spearman_mean": float(record.spearman_mean),
                "spearman_sample_sd": float(record.spearman_std),
                "spearman_median": float(record.spearman_median),
                "spearman_min": float(record.spearman_min),
                "spearman_max": float(record.spearman_max),
                "pair_independence_assumed": False,
                "confidence_interval_applicable": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["stability_type", "top_k"]).reset_index(drop=True)


def _mask_reference(training: pd.DataFrame) -> dict[str, Any]:
    references: dict[str, Any] = {}
    for column in training.columns:
        series = training[column].dropna()
        _require(not series.empty, f"Training-fold mask reference is empty for {column}.")
        if pd.api.types.is_numeric_dtype(training[column]):
            references[column] = float(series.median())
        else:
            modes = list(series.mode(dropna=True))
            _require(bool(modes), f"Training-fold categorical mode is unavailable for {column}.")
            references[column] = min(modes, key=lambda value: str(value))
    return references


def _masked_frame(
    source: pd.DataFrame,
    feature_sets: Sequence[Sequence[str]],
    references: Mapping[str, Any],
) -> pd.DataFrame:
    _require(len(source) == len(feature_sets), "Mask feature-set count drifted.")
    masked = source.copy()
    selected_features = {feature for features in feature_sets for feature in features}
    for feature in selected_features:
        if pd.api.types.is_numeric_dtype(masked[feature]):
            masked[feature] = masked[feature].astype(float)
    for row_position, features in enumerate(feature_sets):
        for feature in features:
            _require(feature in references, f"Unknown mask feature: {feature}.")
            masked.iat[row_position, masked.columns.get_loc(feature)] = references[feature]
    return masked


def _predict_outputs(pipeline: Any, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = aligned_predict_proba(pipeline, frame, labels=LABELS)
    predictions = np.asarray(pipeline.predict(frame), dtype=int)
    transformed = pipeline.named_steps["preprocessor"].transform(frame)
    margins = np.asarray(pipeline.named_steps["model"].model_.predict(transformed, output_margin=True), dtype=float)
    _require(probabilities.shape == margins.shape == (len(frame), len(LABELS)), "Prediction output shape drifted.")
    return predictions, probabilities, margins


def faithfulness_deletion_v3(
    *,
    features: pd.DataFrame,
    target: pd.Series,
    artifacts: Any,
    grouped_reference: np.ndarray,
    random_seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mask SHAP-ranked or equally sized random feature families on exact OOF models."""

    feature_names = np.asarray(features.columns.astype(str))
    result_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    persisted = artifacts.oof_predictions.set_index("sample_index").sort_index()
    label_position = {label: index for index, label in enumerate(LABELS)}
    for outer_fold, fold_model in artifacts.fold_models.items():
        test_ids = list(map(int, fold_model.test_sample_indices))
        train_ids = sorted(set(map(int, features.index)) - set(test_ids))
        X_test = features.loc[test_ids]
        pipeline = fold_model.pipeline
        predictions, baseline_probability, baseline_margin = _predict_outputs(pipeline, X_test)
        expected = persisted.loc[test_ids]
        _require(np.array_equal(predictions, expected["y_pred"].to_numpy(int)), "Faithfulness baseline labels drifted from canonical OOF.")
        _require(np.allclose(baseline_probability, expected[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float), rtol=0.0, atol=1e-12), "Faithfulness baseline probabilities drifted from canonical OOF.")
        predicted_positions = np.asarray([label_position[int(value)] for value in predictions], dtype=int)
        local_importance = np.abs(grouped_reference[np.asarray(test_ids), predicted_positions, :])
        guided_orders = [
            np.lexsort((feature_names, -local_importance[position])).tolist()
            for position in range(len(test_ids))
        ]
        references = _mask_reference(features.loc[train_ids])
        for count in DELETION_COUNTS:
            feature_sets = [[str(feature_names[index]) for index in order[:count]] for order in guided_orders]
            masked = _masked_frame(X_test, feature_sets, references)
            _, probabilities, margins = _predict_outputs(pipeline, masked)
            for position, sample_index in enumerate(test_ids):
                class_position = predicted_positions[position]
                result_rows.append(
                    {
                        "sample_index": sample_index,
                        "outer_fold": int(outer_fold),
                        "method": "shap_guided",
                        "random_repetition": 0,
                        "deleted_feature_count": count,
                        "deleted_features_json": json.dumps(feature_sets[position], separators=(",", ":")),
                        "original_predicted_class": int(predictions[position]),
                        "y_true": int(target.loc[sample_index]),
                        "baseline_probability": float(baseline_probability[position, class_position]),
                        "perturbed_probability": float(probabilities[position, class_position]),
                        "probability_drop": float(baseline_probability[position, class_position] - probabilities[position, class_position]),
                        "baseline_raw_margin": float(baseline_margin[position, class_position]),
                        "perturbed_raw_margin": float(margins[position, class_position]),
                        "raw_margin_drop": float(baseline_margin[position, class_position] - margins[position, class_position]),
                    }
                )
                for feature in feature_sets[position]:
                    frequency_rows.append({"outer_fold": int(outer_fold), "deleted_feature_count": count, "feature": feature})
        for repetition, seed in enumerate(random_seeds, start=1):
            random_orders = [
                np.random.default_rng(int(seed) + int(sample_index) * 1009).permutation(len(feature_names)).tolist()
                for sample_index in test_ids
            ]
            for count in DELETION_COUNTS:
                feature_sets = [[str(feature_names[index]) for index in order[:count]] for order in random_orders]
                masked = _masked_frame(X_test, feature_sets, references)
                _, probabilities, margins = _predict_outputs(pipeline, masked)
                for position, sample_index in enumerate(test_ids):
                    class_position = predicted_positions[position]
                    result_rows.append(
                        {
                            "sample_index": sample_index,
                            "outer_fold": int(outer_fold),
                            "method": "random",
                            "random_repetition": repetition,
                            "deleted_feature_count": count,
                            "deleted_features_json": json.dumps(feature_sets[position], separators=(",", ":")),
                            "original_predicted_class": int(predictions[position]),
                            "y_true": int(target.loc[sample_index]),
                            "baseline_probability": float(baseline_probability[position, class_position]),
                            "perturbed_probability": float(probabilities[position, class_position]),
                            "probability_drop": float(baseline_probability[position, class_position] - probabilities[position, class_position]),
                            "baseline_raw_margin": float(baseline_margin[position, class_position]),
                            "perturbed_raw_margin": float(margins[position, class_position]),
                            "raw_margin_drop": float(baseline_margin[position, class_position] - margins[position, class_position]),
                        }
                    )
    results = pd.DataFrame(result_rows).sort_values(["method", "random_repetition", "sample_index", "deleted_feature_count"]).reset_index(drop=True)
    expected_rows = (1 + len(random_seeds)) * len(features) * len(DELETION_COUNTS)
    _require(len(results) == expected_rows, "Faithfulness perturbation row count drifted.")
    frequency = (
        pd.DataFrame(frequency_rows)
        .groupby(["deleted_feature_count", "feature"], as_index=False)
        .size()
        .rename(columns={"size": "selection_count"})
    )
    frequency["selection_opportunities"] = 1200
    frequency["selection_frequency"] = frequency["selection_count"] / 1200
    return results, frequency.sort_values(["deleted_feature_count", "selection_count", "feature"], ascending=[True, False, True]).reset_index(drop=True)


def summarize_faithfulness_v3(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize probability/margin drops and normalized deletion AUC contrasts."""

    summary = (
        results.groupby(["method", "random_repetition", "deleted_feature_count"], as_index=False)
        .agg(
            n_samples=("sample_index", "size"),
            mean_probability_drop=("probability_drop", "mean"),
            sample_sd_probability_drop=("probability_drop", "std"),
            median_probability_drop=("probability_drop", "median"),
            mean_raw_margin_drop=("raw_margin_drop", "mean"),
            sample_sd_raw_margin_drop=("raw_margin_drop", "std"),
            median_raw_margin_drop=("raw_margin_drop", "median"),
            positive_probability_drop_fraction=("probability_drop", lambda values: float(np.mean(np.asarray(values) > 0))),
        )
        .sort_values(["method", "random_repetition", "deleted_feature_count"])
        .reset_index(drop=True)
    )
    contrast_rows: list[dict[str, Any]] = []
    for count in DELETION_COUNTS:
        guided = summary[(summary["method"] == "shap_guided") & (summary["deleted_feature_count"] == count)].iloc[0]
        random = summary[(summary["method"] == "random") & (summary["deleted_feature_count"] == count)]
        for metric in ("mean_probability_drop", "mean_raw_margin_drop"):
            random_values = random[metric].to_numpy(float)
            guided_value = float(guided[metric])
            contrast_rows.append(
                {
                    "deleted_feature_count": count,
                    "metric": metric,
                    "guided_value": guided_value,
                    "random_repetition_mean": float(np.mean(random_values)),
                    "random_repetition_sample_sd": float(np.std(random_values, ddof=1)),
                    "random_repetition_min": float(np.min(random_values)),
                    "random_repetition_max": float(np.max(random_values)),
                    "guided_minus_random_mean": guided_value - float(np.mean(random_values)),
                    "guided_exceeds_random_repetition_fraction": float(np.mean(guided_value > random_values)),
                    "inferential_p_value": np.nan,
                }
            )
    contrasts = pd.DataFrame(contrast_rows)

    auc_rows: list[dict[str, Any]] = []
    x = np.asarray([0.0, 0.2, 0.6, 1.0])
    for (method, repetition, sample_index), group in results.groupby(["method", "random_repetition", "sample_index"], sort=True):
        values = group.set_index("deleted_feature_count")["probability_drop"]
        y = np.asarray([0.0, float(values.loc[1]), float(values.loc[3]), float(values.loc[5])])
        auc_rows.append(
            {
                "method": method,
                "random_repetition": int(repetition),
                "sample_index": int(sample_index),
                "deletion_auc_probability_drop": float(np.trapezoid(y, x)),
            }
        )
    auc_sample = pd.DataFrame(auc_rows)
    auc_summary = (
        auc_sample.groupby(["method", "random_repetition"], as_index=False)
        .agg(
            n_samples=("sample_index", "size"),
            mean_deletion_auc=("deletion_auc_probability_drop", "mean"),
            sample_sd_deletion_auc=("deletion_auc_probability_drop", "std"),
            median_deletion_auc=("deletion_auc_probability_drop", "median"),
        )
    )
    guided_auc = float(auc_summary.loc[auc_summary["method"] == "shap_guided", "mean_deletion_auc"].iloc[0])
    random_auc = auc_summary.loc[auc_summary["method"] == "random", "mean_deletion_auc"].to_numpy(float)
    auc_contrast = pd.DataFrame(
        [
            {
                "guided_mean_deletion_auc": guided_auc,
                "random_repetition_mean": float(np.mean(random_auc)),
                "random_repetition_sample_sd": float(np.std(random_auc, ddof=1)),
                "random_repetition_min": float(np.min(random_auc)),
                "random_repetition_max": float(np.max(random_auc)),
                "guided_minus_random_mean": guided_auc - float(np.mean(random_auc)),
                "guided_exceeds_random_repetition_fraction": float(np.mean(guided_auc > random_auc)),
                "inferential_p_value": np.nan,
            }
        ]
    )
    return summary, contrasts, auc_sample, auc_summary, auc_contrast


def evaluate_shap_stability_faithfulness_v3(
    contract: Mapping[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
    artifacts: Any,
    model_definition: Mapping[str, Any],
) -> Phase2AResult:
    implementation = contract["grouped_shap_implementation"]
    maximum_error = float(implementation["maximum_additivity_absolute_error"])
    runs: list[ExplainedRun] = []
    feature_contract = _load_json(Path(contract["source_contracts"]["feature_availability"]["path"]))
    excluded = tuple(
        next(record for record in feature_contract["policies"] if record["policy_id"] == "P3")["excluded_features"]
    )
    reference = explain_stability_run_v3(
        stability_type="canonical_reference",
        run_label="canonical_seed_42",
        model_seed=42,
        subsample_seed=None,
        features=features,
        target=target,
        artifacts=artifacts,
        model_definition=model_definition,
        forbidden_features=excluded,
        maximum_additivity_error=maximum_error,
        retain_grouped_oof=True,
    )
    runs.append(reference)
    for record in contract["seed_stability"]["new_runs"]:
        runs.append(
            explain_stability_run_v3(
                stability_type="model_seed",
                run_label=record["run_label"],
                model_seed=int(record["model_seed"]),
                subsample_seed=None,
                features=features,
                target=target,
                artifacts=artifacts,
                model_definition=model_definition,
                forbidden_features=excluded,
                maximum_additivity_error=maximum_error,
            )
        )
    # Include the canonical reference in the seed comparison without duplicating
    # its explanation or fit.
    seed_reference_fold = reference.fold_importance.assign(stability_type="model_seed")
    seed_reference_checks = reference.additivity_checks.assign(stability_type="model_seed")
    seed_reference_rankings = reference.rankings.assign(stability_type="model_seed")
    for seed in contract["resampling_stability"]["subsample_seeds"]:
        runs.append(
            explain_stability_run_v3(
                stability_type="outer_train_resample",
                run_label=f"resample_{seed}",
                model_seed=int(contract["resampling_stability"]["model_seed"]),
                subsample_seed=int(seed),
                features=features,
                target=target,
                artifacts=artifacts,
                model_definition=model_definition,
                forbidden_features=excluded,
                maximum_additivity_error=maximum_error,
            )
        )
    fold_importance = pd.concat([seed_reference_fold, *[run.fold_importance for run in runs[1:]]], ignore_index=True)
    additivity = pd.concat([seed_reference_checks, *[run.additivity_checks for run in runs[1:]]], ignore_index=True)
    rankings = pd.concat([seed_reference_rankings, *[run.rankings for run in runs[1:]]], ignore_index=True)
    pairwise = stability_pairwise_v3(rankings)
    canonical_fold_summary = pd.read_csv(contract["source_contracts"]["canonical_v2_shap_stability"]["path"])
    stability_summary = stability_summary_v3(pairwise, canonical_fold_summary)

    _require(reference.grouped_oof is not None, "Reference grouped SHAP was not retained.")
    faithfulness_rows, feature_frequency = faithfulness_deletion_v3(
        features=features,
        target=target,
        artifacts=artifacts,
        grouped_reference=reference.grouped_oof,
        random_seeds=contract["faithfulness"]["random_baseline_seeds"],
    )
    faithfulness_summary, contrasts, auc_sample, auc_summary, auc_contrast = summarize_faithfulness_v3(faithfulness_rows)
    aggregation_receipt = {
        "schema_version": 1,
        "library": "shap",
        "library_version": shap.__version__,
        "explainer": "shap.TreeExplainer",
        "background_data": None,
        "feature_perturbation": "tree_path_dependent",
        "model_output": "raw_margin",
        "class_labels": list(LABELS),
        "normalized_axis_order": "sample_class_transformed_feature",
        "operation_order": [
            "normalize_library_output_to_sample_class_transformed_feature",
            "sum_signed_encoded_values_within_raw_feature_family",
            "take_absolute_grouped_value",
            "mean_across_classes",
            "mean_across_exactly_once_oof_samples",
        ],
        "absolute_before_transformed_grouping": False,
        "grouped_sum_preservation_required_per_sample_class": True,
        "maximum_observed_raw_margin_additivity_error": float(additivity["maximum_raw_margin_additivity_error"].max()),
        "maximum_observed_grouped_sum_error": float(additivity["maximum_grouped_sum_error"].max()),
        "additivity_tolerance": maximum_error,
        "probability_space_explanation": False,
        "stability_is_faithfulness": False,
    }
    return Phase2AResult(
        fold_feature_importance=fold_importance,
        additivity_checks=additivity,
        stability_run_rankings=rankings,
        stability_pairwise=pairwise,
        stability_summary=stability_summary,
        faithfulness_sample_results=faithfulness_rows,
        faithfulness_summary=faithfulness_summary,
        faithfulness_contrasts=contrasts,
        deletion_auc_sample=auc_sample,
        deletion_auc_summary=auc_summary,
        deletion_auc_contrast=auc_contrast,
        faithfulness_feature_frequency=feature_frequency,
        aggregation_receipt=aggregation_receipt,
    )


def _write_json(path: Path, payload: Any) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def run_shap_stability_faithfulness_v3(
    *,
    contract_path: Path | str,
    output_dir: Path | str,
    run_id: str,
) -> dict[str, Any]:
    contract_path, output_dir = Path(contract_path), Path(output_dir)
    _require(bool(str(run_id).strip()), "run_id must be non-empty.")
    with enforce_offline_runtime() as offline_state:
        git_identity = _clean_git_identity()
        contract, contract_receipt, canonical, features, _, target, artifacts, model_definition = _prepare_inputs(contract_path)
        _require(not output_dir.exists(), f"Output destination already exists: {output_dir}.")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = output_dir.parent / f".{output_dir.name}.staging.{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            implementation_paths = (
                Path("src/experiments/shap_stability_faithfulness_v3.py"),
                Path("src/governance/shap_stability_faithfulness_contract_v3.py"),
                Path("src/explainability/canonical_shap_axis.py"),
                Path("src/experiments/benchmark_artifact_contract.py"),
                Path("src/models/canonical_models.py"),
            )
            scientific_inputs = {
                "git_identity": git_identity,
                "source_tree_hash": source_tree_hash(PROJECT_ROOT),
                "contract_sha256": contract_receipt["contract_sha256"],
                "dataset_sha256": canonical.receipt["actual_sha256"],
                "bound_source_hashes": {name: record["sha256"] for name, record in contract["source_contracts"].items()},
                "implementation_hashes": {path.as_posix(): sha256_file(path) for path in implementation_paths},
            }
            scientific_input_sha256 = _canonical_json_sha256(scientific_inputs)
            result = evaluate_shap_stability_faithfulness_v3(contract, features, target, artifacts, model_definition)
            for observed, expected, label in (
                (len(result.fold_feature_importance), 2200, "fold feature importance"),
                (len(result.additivity_checks), 110, "additivity checks"),
                (len(result.stability_run_rankings), 220, "stability rankings"),
                (len(result.stability_pairwise), 75, "stability pairwise"),
                (len(result.stability_summary), 9, "stability summary"),
                (len(result.faithfulness_sample_results), 75600, "faithfulness sample results"),
                (len(result.faithfulness_summary), 63, "faithfulness summary"),
                (len(result.faithfulness_contrasts), 6, "faithfulness contrasts"),
                (len(result.deletion_auc_sample), 25200, "deletion AUC samples"),
                (len(result.deletion_auc_summary), 21, "deletion AUC summary"),
            ):
                _require(observed == expected, f"Phase 2A {label} row count drifted.")
            frames = {
                "additivity_checks.csv": result.additivity_checks,
                "deletion_auc_contrast.csv": result.deletion_auc_contrast,
                "deletion_auc_sample.csv": result.deletion_auc_sample,
                "deletion_auc_summary.csv": result.deletion_auc_summary,
                "faithfulness_contrasts.csv": result.faithfulness_contrasts,
                "faithfulness_feature_frequency.csv": result.faithfulness_feature_frequency,
                "faithfulness_sample_results.csv": result.faithfulness_sample_results,
                "faithfulness_summary.csv": result.faithfulness_summary,
                "fold_feature_importance.csv": result.fold_feature_importance,
                "stability_pairwise.csv": result.stability_pairwise,
                "stability_run_rankings.csv": result.stability_run_rankings,
                "stability_summary.csv": result.stability_summary,
            }
            for filename, frame in frames.items():
                frame.to_csv(staging / filename, index=False)
            aggregation = {
                **result.aggregation_receipt,
                "run_id": run_id,
                "contract_sha256": contract_receipt["contract_sha256"],
                "scientific_input_sha256": scientific_input_sha256,
            }
            _write_json(staging / "aggregation_receipt.json", aggregation)
            output_hashes = {path.name: sha256_file(path) for path in sorted(staging.iterdir()) if path.is_file()}
            metadata = {
                "schema_version": 1,
                "stage": "shap_stability_faithfulness_v3",
                "status": "complete",
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract_sha256": contract_receipt["contract_sha256"],
                "scientific_input_sha256": scientific_input_sha256,
                "scientific_inputs": scientific_inputs,
                "git_identity": git_identity,
                "dataset_sha256": canonical.receipt["actual_sha256"],
                "canonical_model_set_sha256": artifacts.model_set_sha256,
                "sample_count": 1200,
                "feature_count": 20,
                "outer_folds": 10,
                "seed_stability_run_count_including_reference": 6,
                "resampling_run_count": 5,
                "planned_new_estimator_fit_calls": 100,
                "fold_feature_importance_row_count": len(result.fold_feature_importance),
                "additivity_check_row_count": len(result.additivity_checks),
                "stability_ranking_row_count": len(result.stability_run_rankings),
                "stability_pairwise_row_count": len(result.stability_pairwise),
                "stability_summary_row_count": len(result.stability_summary),
                "faithfulness_sample_row_count": len(result.faithfulness_sample_results),
                "faithfulness_summary_row_count": len(result.faithfulness_summary),
                "faithfulness_contrast_row_count": len(result.faithfulness_contrasts),
                "deletion_auc_sample_row_count": len(result.deletion_auc_sample),
                "deletion_auc_summary_row_count": len(result.deletion_auc_summary),
                "masking_out_of_distribution_risk": True,
                "human_usefulness_claim_allowed": False,
                "runtime_policy": offline_state.receipt(),
                "network_calls": 0,
                "paid_api_calls": 0,
                "output_hashes": output_hashes,
            }
            _require(_clean_git_identity() == git_identity, "Git identity changed during Phase 2A execution.")
            _require(source_tree_hash(PROJECT_ROOT) == scientific_inputs["source_tree_hash"], "Source tree changed during Phase 2A execution.")
            repeated_contract = validate_shap_stability_faithfulness_contract_v3(contract_path)
            _require(repeated_contract["contract_sha256"] == contract_receipt["contract_sha256"], "Phase 2A contract changed during execution.")
            for name, record in contract["source_contracts"].items():
                _require(sha256_file(record["path"]) == record["sha256"], f"Phase 2A source changed during execution: {name}.")
            for fold_model in artifacts.fold_models.values():
                _require(sha256_file(fold_model.path) == fold_model.sha256, f"Canonical fold model changed during execution: {fold_model.outer_fold}.")
            _write_json(staging / "stage_metadata.json", metadata)
            _require({path.name for path in staging.iterdir() if path.is_file()} == EXPECTED_LOCAL_FILES, "Phase 2A output inventory drifted.")
            os.replace(staging, output_dir)
        except Exception:
            if staging.exists():
                for child in staging.iterdir():
                    if child.is_file():
                        child.unlink()
                staging.rmdir()
            raise
    return {
        "status": "complete",
        "run_id": run_id,
        "output_dir": output_dir.as_posix(),
        "contract_sha256": contract_receipt["contract_sha256"],
        "scientific_input_sha256": scientific_input_sha256,
        "planned_new_estimator_fit_calls": 100,
        "faithfulness_sample_row_count": len(result.faithfulness_sample_results),
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_SHAP_STABILITY_FAITHFULNESS_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_LOCAL_RUN_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.preflight_only:
        receipt = preflight_shap_stability_faithfulness_v3(contract_path=args.contract)
    else:
        _require(isinstance(args.run_id, str) and bool(args.run_id.strip()), "--run-id is required for production execution.")
        receipt = run_shap_stability_faithfulness_v3(
            contract_path=args.contract,
            output_dir=args.output_root / args.run_id / "shap_stability_faithfulness",
            run_id=args.run_id,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
