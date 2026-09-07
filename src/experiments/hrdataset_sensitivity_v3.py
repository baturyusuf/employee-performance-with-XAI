"""Execute HRDataset_v14 target-mapping and repeated nested-CV sensitivities.

The tracked contract freezes two semantically motivated target formulations,
one exact seven-feature policy, five 5x5 nested-CV repetitions, a single tuned
XGBoost family, predeclared cross-fitted sigmoid calibration, and three
training-only naive baselines. Row-level evidence is written only below the
ignored local run root. No network or paid API is used.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.data.canonical_loader import sha256_file
from src.data.external_adapters import build_feature_columns, load_external_dataset
from src.experiments.manuscript_calibration import (
    apply_sigmoid_calibrator,
    fit_sigmoid_calibrator,
)
from src.experiments.manuscript_model_benchmark import select_candidate_index
from src.experiments.shared_folds import (
    SharedFoldArtifacts,
    generate_shared_folds,
    validate_shared_folds,
)
from src.governance.hrdataset_sensitivity_contract_v3 import (
    ALTERNATIVE_FORMULATION,
    BASELINES,
    DEFAULT_HRDATASET_SENSITIVITY_CONTRACT,
    FEATURES,
    FORMULATIONS,
    METRICS,
    PRIMARY_FORMULATION,
    PRIORITY_METRICS,
    validate_hrdataset_sensitivity_contract_v3,
)
from src.governance.manuscript_contract import source_tree_hash
from src.governance.offline_runtime import enforce_offline_runtime
from src.models.canonical_models import aligned_predict_proba, build_model_pipeline
from src.models.evaluate import classification_metrics
from src.models.ordinal_evaluation_v3 import ordinal_evaluation_bundle_v3
from src.models.ordinal_models_v3 import build_v3_naive_baseline
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_LOCAL_RUN_ROOT = Path("reports/major_revision_v3_runs")
MODEL_NAME = "xgboost"
SYSTEMS = ("xgboost_raw", "xgboost_sigmoid", *BASELINES)
FIT_THREAD_LIMIT = 1
EXPECTED_LOCAL_FILES = frozenset(
    {
        "baseline_comparisons.csv",
        "calibration_training_oof.csv",
        "calibrator_parameters.csv",
        "candidate_search_results.csv",
        "canonical_v2_confusion_matrix.csv",
        "canonical_v2_per_class_metrics.csv",
        "confusion_matrices.csv",
        "cv_design_sensitivity.csv",
        "fold_contracts.json",
        "fold_metrics.csv",
        "oof_predictions.csv",
        "per_class_metrics.csv",
        "protocol_comparison.csv",
        "repetition_metrics.csv",
        "selected_hyperparameters.csv",
        "stage_metadata.json",
        "target_mapping_support.csv",
        "variability_summary.csv",
    }
)


class HRDatasetSensitivityV3Error(RuntimeError):
    """Raised when Phase 3A execution violates a scientific invariant."""


@dataclass(frozen=True)
class HRDatasetSensitivityResult:
    candidate_search_results: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    calibration_training_oof: pd.DataFrame
    calibrator_parameters: pd.DataFrame
    repetition_metrics: pd.DataFrame
    variability_summary: pd.DataFrame
    per_class_metrics: pd.DataFrame
    confusion_matrices: pd.DataFrame
    baseline_comparisons: pd.DataFrame
    target_mapping_support: pd.DataFrame
    cv_design_sensitivity: pd.DataFrame
    canonical_v2_per_class_metrics: pd.DataFrame
    canonical_v2_confusion_matrix: pd.DataFrame
    protocol_comparison: pd.DataFrame
    fold_contracts: tuple[Mapping[str, Any], ...]
    evidence_status: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HRDatasetSensitivityV3Error(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HRDatasetSensitivityV3Error(f"Could not read {path.as_posix()}: {exc}") from exc
    _require(isinstance(payload, dict), f"{path.name} must contain an object.")
    return payload


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_digest(value: Any, *, length: int = 64) -> bool:
    observed = str(value)
    return len(observed) == length and all(character in "0123456789abcdef" for character in observed)


def _clean_git_identity() -> dict[str, str]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HRDatasetSensitivityV3Error(f"Could not establish Git identity: {exc}") from exc
    _require(_valid_digest(head, length=40), "Git HEAD must be a full lowercase digest.")
    _require(not status, f"Scientific execution requires a clean worktree: {status.splitlines()[:10]}.")
    return {"commit": head, "branch": branch}


def _fit_or_fail(estimator: Any, features: pd.DataFrame, target: pd.Series, *, context: str) -> Any:
    try:
        with threadpool_limits(limits=FIT_THREAD_LIMIT):
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                return estimator.fit(features, target)
    except Exception as exc:
        raise HRDatasetSensitivityV3Error(
            f"{context} failed: {type(exc).__name__}: {exc}"
        ) from exc


def _formulation_map(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    formulations = contract["target_formulations"]
    indexed = {str(item["formulation_id"]): item for item in formulations}
    _require(tuple(indexed) == FORMULATIONS, "Target formulation registry drifted.")
    return indexed


def _prepare_inputs(contract_path: Path) -> tuple[dict[str, Any], dict[str, Any], Any, pd.DataFrame, Mapping[str, Any]]:
    receipt = validate_hrdataset_sensitivity_contract_v3(contract_path)
    contract = _load_json(contract_path)
    sources = contract["source_contracts"]
    dataset = load_external_dataset(
        "hrdataset_v14",
        raw_path=PROJECT_ROOT / sources["raw_dataset"]["path"],
        schema_mapping_path=PROJECT_ROOT / sources["schema_mapping"]["path"],
    )
    feature_columns = tuple(build_feature_columns(dataset, "conservative_primary"))
    _require(feature_columns == FEATURES, "Phase 3A feature order drifted.")
    features = dataset.canonical.loc[:, list(feature_columns)].copy()
    _require(features.index.equals(pd.RangeIndex(len(features))), "Canonical employee index drifted.")
    model_grid = load_config(PROJECT_ROOT / sources["model_grid"]["path"])
    xgb_definition = model_grid["model_benchmark"]["models"][MODEL_NAME]
    _require(len(xgb_definition["candidates"]) == 8, "XGBoost grid drifted.")
    return contract, receipt, dataset, features, xgb_definition


def _target_series(dataset: Any, formulation: Mapping[str, Any]) -> pd.Series:
    raw = dataset.raw["PerformanceScore"].astype(str).str.strip()
    target = raw.map(dict(formulation["mapping"]))
    _require(not target.isna().any(), f"{formulation['formulation_id']} target mapping is incomplete.")
    target = target.astype(int)
    _require(set(target) == set(formulation["ordered_labels"]), "Mapped target support drifted.")
    target.index = dataset.canonical.index
    return target


def _fold_source(dataset: Any, target: pd.Series) -> pd.DataFrame:
    source = dataset.canonical.copy()
    source["Phase3ATarget"] = target.astype(int)
    _require(source["ExternalSampleId"].astype(int).tolist() == list(range(len(source))), "External sample IDs drifted.")
    return source


def _outer_assignment_semantic_sha256(folds: SharedFoldArtifacts) -> str:
    rows = folds.outer_assignments[["sample_index", "outer_fold", "y_true"]].astype(int).sort_values("sample_index")
    return _canonical_json_sha256(rows.to_dict(orient="records"))


def _fold_membership(folds: SharedFoldArtifacts, outer_fold: int) -> tuple[list[int], list[int], pd.DataFrame]:
    outer = folds.outer_assignments
    test_ids = outer.loc[outer["outer_fold"].astype(int) == outer_fold, "sample_index"].astype(int).tolist()
    train_ids = outer.loc[outer["outer_fold"].astype(int) != outer_fold, "sample_index"].astype(int).tolist()
    inner = folds.inner_assignments.loc[
        folds.inner_assignments["outer_fold"].astype(int) == outer_fold
    ].copy()
    _require(set(inner["sample_index"].astype(int)) == set(train_ids), "Inner membership differs from outer training.")
    return train_ids, test_ids, inner


def _selection_scores(target: pd.Series, prediction: np.ndarray, probability: np.ndarray, labels: Sequence[int]) -> tuple[float, float]:
    metrics = classification_metrics(target, prediction, probability, list(labels))
    values = (metrics.get("macro_f1"), metrics.get("quadratic_weighted_kappa"))
    _require(all(value is not None and math.isfinite(float(value)) for value in values), "Inner selection metric is non-finite.")
    return float(values[0]), float(values[1])


def _pipeline(
    training_features: pd.DataFrame,
    definition: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    model_seed: int,
    forbidden_features: Sequence[str],
) -> Any:
    return build_model_pipeline(
        MODEL_NAME,
        training_features,
        fixed_parameters=definition["fixed_params"],
        candidate_parameters=candidate,
        random_state=model_seed,
        forbidden_features=forbidden_features,
    )


def _bundle(target: pd.Series, prediction: np.ndarray, probability: np.ndarray, labels: Sequence[int], system: str) -> Mapping[str, Any]:
    return ordinal_evaluation_bundle_v3(
        target,
        prediction,
        probability,
        labels=labels,
        dataset_key="hrdataset_v14",
        model_name=system,
    )


def _prediction_rows(
    *,
    identity: Mapping[str, Any],
    formulation_id: str,
    labels: Sequence[int],
    system: str,
    outer_fold: int,
    sample_ids: Sequence[int],
    target: pd.Series,
    prediction: np.ndarray,
    probability: np.ndarray,
    selected_candidate_index: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, sample_index in enumerate(sample_ids):
        row = {
            **identity,
            "formulation_id": formulation_id,
            "ordered_labels_json": json.dumps(list(labels), separators=(",", ":")),
            "system": system,
            "outer_fold": int(outer_fold),
            "sample_index": int(sample_index),
            "y_true": int(target.loc[sample_index]),
            "y_pred": int(prediction[position]),
            "selected_candidate_index": selected_candidate_index,
        }
        row.update(
            {
                f"prob_class_{int(label)}": float(probability[position, column])
                for column, label in enumerate(labels)
            }
        )
        rows.append(row)
    return rows


def _calibrator_parameter_rows(
    calibrator: Any,
    *,
    identity: Mapping[str, Any],
    formulation_id: str,
    outer_fold: int,
    selected_candidate_index: int,
) -> list[dict[str, Any]]:
    shared = {
        **identity,
        "formulation_id": formulation_id,
        "outer_fold": int(outer_fold),
        "selected_candidate_index": int(selected_candidate_index),
        "sigmoid_parameter_sha256": calibrator.parameter_sha256,
        "training_probability_sha256": calibrator.training_probability_sha256,
        "training_labels_sha256": calibrator.training_labels_sha256,
        "algorithm": "one_vs_rest_platt_logit_then_row_renormalize",
        "calibration_seed": calibrator.seed,
        "solver": calibrator.solver,
        "regularization": calibrator.regularization,
        "l1_ratio": calibrator.l1_ratio,
        "C": calibrator.c_value,
        "fit_intercept": calibrator.fit_intercept,
        "max_iter": calibrator.max_iter,
        "tol": calibrator.tolerance,
        "probability_clip": calibrator.probability_clip,
        "threadpool_limit": calibrator.threadpool_limit,
        "outer_test_used_for_fit_or_selection": False,
    }
    return [{**shared, **asdict(parameters)} for parameters in calibrator.class_parameters]


def _evaluate_outer_fold(
    features: pd.DataFrame,
    target: pd.Series,
    folds: SharedFoldArtifacts,
    definition: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    formulation_id: str,
    labels: Sequence[int],
    outer_fold: int,
    model_seed: int,
    calibration_seed: int,
    baseline_seed: int,
    tie_tolerance: float,
    forbidden_features: Sequence[str],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    train_ids, test_ids, inner = _fold_membership(folds, outer_fold)
    candidates = [dict(value) for value in definition["candidates"]]
    candidate_rows: list[dict[str, Any]] = []
    primary_means: list[float] = []
    secondary_means: list[float] = []
    for candidate_index, candidate in enumerate(candidates):
        primary_scores: list[float] = []
        secondary_scores: list[float] = []
        for inner_fold in range(1, 6):
            validation_ids = inner.loc[
                inner["inner_fold"].astype(int) == inner_fold, "sample_index"
            ].astype(int).tolist()
            development_ids = sorted(set(train_ids) - set(validation_ids))
            pipeline = _pipeline(
                features.loc[development_ids],
                definition,
                candidate,
                model_seed=model_seed,
                forbidden_features=forbidden_features,
            )
            _fit_or_fail(
                pipeline,
                features.loc[development_ids],
                target.loc[development_ids],
                context=f"{formulation_id}/outer={outer_fold}/candidate={candidate_index}/inner={inner_fold}",
            )
            probability = aligned_predict_proba(pipeline, features.loc[validation_ids], labels=labels)
            prediction = np.asarray(labels, dtype=int)[np.argmax(probability, axis=1)]
            primary, secondary = _selection_scores(target.loc[validation_ids], prediction, probability, labels)
            primary_scores.append(primary)
            secondary_scores.append(secondary)
        primary_means.append(float(np.mean(primary_scores)))
        secondary_means.append(float(np.mean(secondary_scores)))
        candidate_rows.append(
            {
                **identity,
                "formulation_id": formulation_id,
                "outer_fold": int(outer_fold),
                "candidate_index": int(candidate_index),
                "candidate_parameters_json": json.dumps(candidate, sort_keys=True, separators=(",", ":")),
                "inner_macro_f1_scores_json": json.dumps(primary_scores, separators=(",", ":")),
                "inner_macro_f1_mean": primary_means[-1],
                "inner_qwk_scores_json": json.dumps(secondary_scores, separators=(",", ":")),
                "inner_qwk_mean": secondary_means[-1],
                "outer_test_used_for_selection": False,
            }
        )
    selected_index = select_candidate_index(
        primary_means,
        secondary_means,
        practical_tie_tolerance=tie_tolerance,
        better_direction="higher",
    )
    for row in candidate_rows:
        row["selected_by_protocol"] = int(row["candidate_index"]) == selected_index
    selected_candidate = candidates[selected_index]
    selected_row = {
        **identity,
        "formulation_id": formulation_id,
        "outer_fold": int(outer_fold),
        "selected_candidate_index": int(selected_index),
        "selected_candidate_parameters_json": json.dumps(selected_candidate, sort_keys=True, separators=(",", ":")),
        "fixed_parameters_json": json.dumps(dict(definition["fixed_params"]), sort_keys=True, separators=(",", ":")),
        "selected_inner_macro_f1_mean": primary_means[selected_index],
        "selected_inner_qwk_mean": secondary_means[selected_index],
        "outer_test_used_for_selection": False,
    }
    final_model = _pipeline(
        features.loc[train_ids],
        definition,
        selected_candidate,
        model_seed=model_seed,
        forbidden_features=forbidden_features,
    )
    _fit_or_fail(final_model, features.loc[train_ids], target.loc[train_ids], context=f"{formulation_id}/outer={outer_fold}/final")
    raw_probability = aligned_predict_proba(final_model, features.loc[test_ids], labels=labels)
    raw_prediction = np.asarray(labels, dtype=int)[np.argmax(raw_probability, axis=1)]

    calibration_probability = np.full((len(train_ids), len(labels)), np.nan, dtype=float)
    training_position = {sample_index: position for position, sample_index in enumerate(train_ids)}
    for inner_fold in range(1, 6):
        validation_ids = inner.loc[
            inner["inner_fold"].astype(int) == inner_fold, "sample_index"
        ].astype(int).tolist()
        development_ids = sorted(set(train_ids) - set(validation_ids))
        calibration_model = _pipeline(
            features.loc[development_ids],
            definition,
            selected_candidate,
            model_seed=model_seed,
            forbidden_features=forbidden_features,
        )
        _fit_or_fail(
            calibration_model,
            features.loc[development_ids],
            target.loc[development_ids],
            context=f"{formulation_id}/outer={outer_fold}/calibration-inner={inner_fold}",
        )
        scoped = aligned_predict_proba(calibration_model, features.loc[validation_ids], labels=labels)
        for row_position, sample_index in enumerate(validation_ids):
            calibration_probability[training_position[sample_index], :] = scoped[row_position, :]
    _require(np.isfinite(calibration_probability).all(), "Calibration OOF probabilities are incomplete.")
    calibration_training_rows: list[dict[str, Any]] = []
    for position, sample_index in enumerate(train_ids):
        row = {
            **identity,
            "formulation_id": formulation_id,
            "ordered_labels_json": json.dumps(list(labels), separators=(",", ":")),
            "outer_fold": int(outer_fold),
            "sample_index": int(sample_index),
            "y_true": int(target.loc[sample_index]),
            "selected_candidate_index": int(selected_index),
            "outer_test_used_for_fit_or_selection": False,
        }
        row.update(
            {
                f"prob_class_{int(label)}": float(calibration_probability[position, column])
                for column, label in enumerate(labels)
            }
        )
        calibration_training_rows.append(row)
    calibrator = fit_sigmoid_calibrator(
        calibration_probability,
        target.loc[train_ids].to_numpy(int),
        labels,
        seed=calibration_seed + outer_fold,
    )
    sigmoid_probability = apply_sigmoid_calibrator(calibrator, raw_probability)
    sigmoid_prediction = np.asarray(labels, dtype=int)[np.argmax(sigmoid_probability, axis=1)]

    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for system, prediction, probability in (
        ("xgboost_raw", raw_prediction, raw_probability),
        ("xgboost_sigmoid", sigmoid_prediction, sigmoid_probability),
    ):
        prediction_rows.extend(
            _prediction_rows(
                identity=identity,
                formulation_id=formulation_id,
                labels=labels,
                system=system,
                outer_fold=outer_fold,
                sample_ids=test_ids,
                target=target,
                prediction=prediction,
                probability=probability,
                selected_candidate_index=selected_index,
            )
        )
        fold_rows.append(
            {
                **identity,
                "formulation_id": formulation_id,
                "system": system,
                "outer_fold": int(outer_fold),
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                **_bundle(target.loc[test_ids], prediction, probability, labels, system)["aggregate_metrics"],
            }
        )

    for baseline_position, baseline_name in enumerate(BASELINES):
        effective_seed = baseline_seed + outer_fold * 10 + baseline_position
        baseline = build_v3_naive_baseline(baseline_name, random_state=effective_seed)
        # These label-only comparators deliberately receive a finite inert design.
        # Passing the scientific feature frame would make their behavior depend on
        # missing-value validation despite never using feature values.
        baseline_train = pd.DataFrame(
            {"inert_baseline_design": np.zeros(len(train_ids), dtype=np.int8)},
            index=train_ids,
        )
        baseline_test = pd.DataFrame(
            {"inert_baseline_design": np.zeros(len(test_ids), dtype=np.int8)},
            index=test_ids,
        )
        _fit_or_fail(baseline, baseline_train, target.loc[train_ids], context=f"{formulation_id}/outer={outer_fold}/{baseline_name}")
        probability = aligned_predict_proba(baseline, baseline_test, labels=labels)
        prediction = np.asarray(labels, dtype=int)[np.argmax(probability, axis=1)]
        prediction_rows.extend(
            _prediction_rows(
                identity=identity,
                formulation_id=formulation_id,
                labels=labels,
                system=baseline_name,
                outer_fold=outer_fold,
                sample_ids=test_ids,
                target=target,
                prediction=prediction,
                probability=probability,
                selected_candidate_index=None,
            )
        )
        fold_rows.append(
            {
                **identity,
                "formulation_id": formulation_id,
                "system": baseline_name,
                "outer_fold": int(outer_fold),
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                **_bundle(target.loc[test_ids], prediction, probability, labels, baseline_name)["aggregate_metrics"],
            }
        )
    calibrator_rows = _calibrator_parameter_rows(
        calibrator,
        identity=identity,
        formulation_id=formulation_id,
        outer_fold=outer_fold,
        selected_candidate_index=selected_index,
    )
    return (
        candidate_rows,
        selected_row,
        fold_rows,
        prediction_rows,
        calibration_training_rows,
        calibrator_rows,
    )


def _summarize_oof(
    oof: pd.DataFrame,
    formulations: Mapping[str, Mapping[str, Any]],
    *,
    full_run: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for (formulation_id, repetition, system), rows in oof.groupby(
        ["formulation_id", "repetition", "system"], sort=True
    ):
        labels = tuple(int(value) for value in formulations[str(formulation_id)]["ordered_labels"])
        probability = rows[[f"prob_class_{label}" for label in labels]].to_numpy(float)
        bundle = _bundle(rows["y_true"], rows["y_pred"].to_numpy(int), probability, labels, str(system))
        identity = {
            "run_id": str(rows["run_id"].iloc[0]),
            "contract_sha256": str(rows["contract_sha256"].iloc[0]),
            "scientific_input_sha256": str(rows["scientific_input_sha256"].iloc[0]),
            "formulation_id": str(formulation_id),
            "repetition": int(repetition),
            "system": str(system),
            "ordered_labels_json": json.dumps(list(labels), separators=(",", ":")),
            "n_samples": len(rows),
        }
        for metric, value in bundle["aggregate_metrics"].items():
            if metric in METRICS:
                metric_rows.append({**identity, "metric": metric, "value": float(value)})
        class_rows.extend({**identity, **record} for record in bundle["per_class_metrics"])
        confusion_rows.extend({**identity, **record} for record in bundle["confusion_matrix"])
    repetition_metrics = pd.DataFrame(metric_rows).sort_values(
        ["formulation_id", "repetition", "system", "metric"]
    ).reset_index(drop=True)
    per_class = pd.DataFrame(class_rows).drop(columns=["dataset_key", "model_name"]).sort_values(
        ["formulation_id", "repetition", "system", "class_label"]
    ).reset_index(drop=True)
    confusion = pd.DataFrame(confusion_rows).drop(columns=["dataset_key", "model_name"]).sort_values(
        ["formulation_id", "repetition", "system", "true_label", "predicted_label"]
    ).reset_index(drop=True)
    variability_rows: list[dict[str, Any]] = []
    if full_run:
        expected = len(FORMULATIONS) * 5 * len(SYSTEMS) * len(METRICS)
        _require(len(repetition_metrics) == expected, "Repetition metric grid is incomplete.")
        for (formulation_id, system, metric), rows in repetition_metrics.groupby(
            ["formulation_id", "system", "metric"], sort=True
        ):
            values = rows.sort_values("repetition")["value"].to_numpy(float)
            _require(len(values) == 5 and np.isfinite(values).all(), "Five-repetition variability grid is incomplete.")
            variability_rows.append(
                {
                    "formulation_id": formulation_id,
                    "system": system,
                    "metric": metric,
                    "repetition_count": 5,
                    "mean": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "range_interpretation": "empirical_training_and_fold_variability_not_confidence_interval",
                }
            )
    variability = pd.DataFrame(variability_rows)
    return repetition_metrics, variability, per_class, confusion


def _validate_oof(
    oof: pd.DataFrame,
    fold_lookup: Mapping[tuple[str, int], pd.DataFrame],
    formulations: Mapping[str, Mapping[str, Any]],
    *,
    full_run: bool,
) -> None:
    _require(not oof.empty, "Phase 3A OOF evidence is empty.")
    for (formulation_id, repetition, system), rows in oof.groupby(
        ["formulation_id", "repetition", "system"], sort=True
    ):
        labels = tuple(int(value) for value in formulations[str(formulation_id)]["ordered_labels"])
        if full_run:
            _require(len(rows) == 311 and rows["sample_index"].nunique() == 311, "Exactly-once OOF coverage failed.")
            _require(set(rows["sample_index"].astype(int)) == set(range(311)), "OOF sample set drifted.")
        expected = fold_lookup[(str(formulation_id), int(repetition))]
        observed = rows.set_index("sample_index")[["outer_fold", "y_true"]].astype(int).sort_index()
        _require(observed.equals(expected.loc[observed.index]), "OOF fold/target lineage drifted.")
        probability = rows[[f"prob_class_{label}" for label in labels]].to_numpy(float)
        _require(np.isfinite(probability).all(), "OOF probabilities are non-finite.")
        _require(np.all((0 <= probability) & (probability <= 1)), "OOF probability escaped [0,1].")
        _require(np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-9), "OOF simplex drifted.")
        predicted = np.asarray(labels, dtype=int)[np.argmax(probability, axis=1)]
        _require(np.array_equal(predicted, rows["y_pred"].to_numpy(int)), "OOF prediction/probability mismatch.")


def _target_mapping_support(dataset: Any, formulations: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    raw = dataset.raw["PerformanceScore"].astype(str).str.strip()
    raw_support = raw.value_counts().to_dict()
    rows: list[dict[str, Any]] = []
    for formulation_id in FORMULATIONS:
        formulation = formulations[formulation_id]
        for raw_value, mapped_label in formulation["mapping"].items():
            rows.append(
                {
                    "formulation_id": formulation_id,
                    "formulation_role": formulation["role"],
                    "raw_target_value": raw_value,
                    "mapped_label": int(mapped_label),
                    "observed_support": int(raw_support.get(raw_value, 0)),
                    "observed_in_dataset": raw_value in raw_support,
                    "mapping_rationale": formulation["rationale"],
                    "target_equivalence_claim_allowed": False,
                }
            )
    return pd.DataFrame(rows)


def _baseline_comparisons(repetition_metrics: pd.DataFrame) -> pd.DataFrame:
    priority = repetition_metrics[repetition_metrics["metric"].isin(PRIORITY_METRICS)].copy()
    rows: list[dict[str, Any]] = []
    for (formulation_id, repetition, metric), scoped in priority.groupby(
        ["formulation_id", "repetition", "metric"], sort=True
    ):
        values = scoped.set_index("system")["value"]
        _require("xgboost_raw" in values.index, "Raw XGBoost baseline reference is absent.")
        for baseline in BASELINES:
            _require(baseline in values.index, f"Naive baseline is absent: {baseline}.")
            rows.append(
                {
                    "formulation_id": formulation_id,
                    "repetition": int(repetition),
                    "metric": metric,
                    "comparison": f"xgboost_raw_minus_{baseline}",
                    "xgboost_raw_value": float(values["xgboost_raw"]),
                    "baseline_value": float(values[baseline]),
                    "signed_arithmetic_difference": float(values["xgboost_raw"] - values[baseline]),
                    "better_direction": "higher" if metric in {"macro_f1", "quadratic_weighted_kappa"} else "lower",
                    "inference": "descriptive_matched_repetition_no_confidence_interval",
                }
            )
    return pd.DataFrame(rows)


def _canonical_v2_class_results(contract: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Mapping[str, float]]]:
    sources = contract["source_contracts"]
    outputs: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    metrics: dict[str, Mapping[str, float]] = {}
    for method, source_name in (("xgboost_raw", "v2_raw_oof"), ("xgboost_sigmoid", "v2_calibrated_oof")):
        frame = pd.read_csv(PROJECT_ROOT / sources[source_name]["path"])
        frame = frame.loc[frame["policy"].astype(str) == "conservative_primary"].sort_values("sample_index")
        probability = frame[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float)
        bundle = _bundle(frame["y_true"], frame["y_pred"].to_numpy(int), probability, (2, 3, 4), method)
        metrics[method] = {name: float(value) for name, value in bundle["aggregate_metrics"].items() if name in METRICS}
        common = {
            "dataset_key": "hrdataset_v14",
            "design": "canonical_v2_single_10_outer_x_5_inner_nested_cv",
            "formulation_id": PRIMARY_FORMULATION,
            "system": method,
            "n_samples": 311,
            "source_run_id": str(frame["run_id"].iloc[0]),
        }
        class_frame = pd.DataFrame([{**common, **row} for row in bundle["per_class_metrics"]]).drop(columns="model_name")
        confusion_frame = pd.DataFrame([{**common, **row} for row in bundle["confusion_matrix"]]).drop(columns="model_name")
        outputs.append((class_frame, confusion_frame))
    return (
        pd.concat([item[0] for item in outputs], ignore_index=True),
        pd.concat([item[1] for item in outputs], ignore_index=True),
        metrics,
    )


def _cv_design_sensitivity(variability: pd.DataFrame, v2_metrics: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scoped = variability.loc[variability["formulation_id"] == PRIMARY_FORMULATION]
    for system in ("xgboost_raw", "xgboost_sigmoid"):
        for metric in PRIORITY_METRICS:
            current = scoped.loc[(scoped["system"] == system) & (scoped["metric"] == metric)]
            _require(len(current) == 1, f"Repeated-CV summary cell is absent for {system}/{metric}.")
            record = current.iloc[0]
            reference = float(v2_metrics[system][metric])
            minimum = float(record["minimum"])
            maximum = float(record["maximum"])
            rows.append(
                {
                    "formulation_id": PRIMARY_FORMULATION,
                    "system": system,
                    "metric": metric,
                    "canonical_v2_10x5_point_estimate": reference,
                    "repeated_5x5_mean": float(record["mean"]),
                    "repeated_5x5_sample_sd": float(record["sample_sd"]),
                    "repeated_5x5_minimum": minimum,
                    "repeated_5x5_maximum": maximum,
                    "ten_fold_inside_repeated_range": minimum <= reference <= maximum,
                    "signed_repeated_mean_minus_ten_fold": float(record["mean"]) - reference,
                    "interpretation": "descriptive_cv_design_sensitivity_not_equivalence_test",
                }
            )
    return pd.DataFrame(rows)


def _protocol_comparison() -> pd.DataFrame:
    records = [
        ("folds", "canonical 10x5 plus v3 repeated 5x5", "canonical 10x5 plus v3 repeated 5x5", "same strategy; dataset-specific assignments", "not a common sample population"),
        ("tuning", "nested candidate selection for benchmark systems", "nested eight-candidate XGBoost selection", "same XGBoost selection implementation", "different dataset and fitted parameters"),
        ("calibration", "predeclared cross-fitted one-vs-rest sigmoid", "predeclared cross-fitted one-vs-rest sigmoid", "same algorithm", "probability quality is dataset conditional"),
        ("SHAP", "exact-fold OOF XGBoost raw-margin SHAP", "canonical-v2 exact-fold OOF XGBoost raw-margin SHAP", "same attribution implementation", "different feature families and no causal meaning"),
        ("subgroup", "support-aware descriptive diagnostics", "support-aware descriptive diagnostics", "same diagnostic framing", "different attributes/support; no fairness certification"),
        ("proxy", "department reconstructability and proxy-use sensitivity", "department reconstructability in canonical v2", "partially shared diagnostic design", "reconstructability is not performance-model use"),
        ("target", "native INX organisational rating 2/3/4", "dataset-specific mapped PerformanceScore; 3-class primary plus raw-order 4-class sensitivity", "different mapping implementation", "semantics and prevalence are not equivalent"),
        ("features", "20-feature P3 primary policy", "seven-feature conservative-primary policy", "same training-only preprocessing", "feature spaces and availability semantics differ"),
    ]
    return pd.DataFrame(
        [
            {
                "component": component,
                "INX": inx,
                "HRDataset_v14": hr,
                "same_implementation": implementation,
                "same_semantics": False if component in {"target", "features", "subgroup", "proxy"} else "partial",
                "notes": notes,
            }
            for component, inx, hr, implementation, notes in records
        ]
    )


def evaluate_hrdataset_sensitivity_v3(
    dataset: Any,
    features: pd.DataFrame,
    contract: Mapping[str, Any],
    definition: Mapping[str, Any],
    *,
    run_id: str,
    contract_sha256: str,
    scientific_input_sha256: str,
    dataset_sha256: str,
    repetition_subset: Sequence[int] | None = None,
    outer_fold_subset: Sequence[int] | None = None,
) -> HRDatasetSensitivityResult:
    """Return complete or explicitly diagnostic Phase 3A in-memory evidence."""

    for name, value in (("contract_sha256", contract_sha256), ("scientific_input_sha256", scientific_input_sha256), ("dataset_sha256", dataset_sha256)):
        _require(_valid_digest(value), f"{name} must be a lowercase SHA-256.")
    formulations = _formulation_map(contract)
    seed_schedule = [dict(row) for row in contract["design"]["seed_schedule"]]
    all_repetitions = tuple(int(row["repetition"]) for row in seed_schedule)
    selected_repetitions = all_repetitions if repetition_subset is None else tuple(sorted(set(map(int, repetition_subset))))
    _require(selected_repetitions and set(selected_repetitions).issubset(all_repetitions), "Diagnostic repetition subset is invalid.")
    selected_outer_folds = tuple(range(1, 6)) if outer_fold_subset is None else tuple(sorted(set(map(int, outer_fold_subset))))
    _require(selected_outer_folds and set(selected_outer_folds).issubset(set(range(1, 6))), "Diagnostic outer-fold subset is invalid.")
    full_run = repetition_subset is None and outer_fold_subset is None
    evidence_status = "complete_five_repetition_two_mapping_exactly_once_oof" if full_run else "diagnostic_incomplete_never_canonical"
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    calibration_training_rows: list[dict[str, Any]] = []
    calibrator_rows: list[dict[str, Any]] = []
    fold_contracts: list[Mapping[str, Any]] = []
    fold_lookup: dict[tuple[str, int], pd.DataFrame] = {}
    forbidden_features = tuple(column for column in dataset.canonical.columns if column not in FEATURES)
    for formulation_id in FORMULATIONS:
        formulation = formulations[formulation_id]
        labels = tuple(int(value) for value in formulation["ordered_labels"])
        target = _target_series(dataset, formulation)
        source = _fold_source(dataset, target)
        for seed_record in seed_schedule:
            repetition = int(seed_record["repetition"])
            if repetition not in selected_repetitions:
                continue
            folds = generate_shared_folds(
                source,
                target_column="Phase3ATarget",
                id_column="ExternalSampleId",
                run_id=f"{run_id}_{formulation_id}_rep{repetition}",
                config_hash=contract_sha256,
                scientific_input_hash=scientific_input_sha256,
                dataset_key=f"hrdataset_v14_{formulation_id}",
                dataset_sha256=dataset_sha256,
                outer_splits=5,
                inner_splits=5,
                seed=int(seed_record["outer_seed"]),
                inner_seed=int(seed_record["inner_seed"]),
            )
            validate_shared_folds(folds)
            semantic_hash = _outer_assignment_semantic_sha256(folds)
            fold_contracts.append(
                {
                    "formulation_id": formulation_id,
                    "ordered_labels": list(labels),
                    "repetition": repetition,
                    "outer_seed": int(seed_record["outer_seed"]),
                    "inner_seed": int(seed_record["inner_seed"]),
                    "model_seed": int(seed_record["model_seed"]),
                    "calibration_seed": int(seed_record["calibration_seed"]),
                    "baseline_seed": int(seed_record["baseline_seed"]),
                    "outer_assignment_semantic_sha256": semantic_hash,
                    "contract": folds.contract,
                }
            )
            fold_lookup[(formulation_id, repetition)] = folds.outer_assignments.set_index("sample_index")[["outer_fold", "y_true"]].astype(int).sort_index()
            identity = {
                "run_id": run_id,
                "contract_sha256": contract_sha256,
                "scientific_input_sha256": scientific_input_sha256,
                "dataset_sha256": dataset_sha256,
                "repetition": repetition,
                "outer_seed": int(seed_record["outer_seed"]),
                "inner_seed": int(seed_record["inner_seed"]),
                "model_seed": int(seed_record["model_seed"]),
                "calibration_seed": int(seed_record["calibration_seed"]),
                "baseline_seed": int(seed_record["baseline_seed"]),
                "fold_contract_hash": str(folds.contract["fold_contract_hash"]),
            }
            for outer_fold in selected_outer_folds:
                (
                    candidate,
                    selected,
                    fold_metric,
                    predictions,
                    calibration_training,
                    calibrators,
                ) = _evaluate_outer_fold(
                    features,
                    target,
                    folds,
                    definition,
                    identity=identity,
                    formulation_id=formulation_id,
                    labels=labels,
                    outer_fold=outer_fold,
                    model_seed=int(seed_record["model_seed"]),
                    calibration_seed=int(seed_record["calibration_seed"]),
                    baseline_seed=int(seed_record["baseline_seed"]),
                    tie_tolerance=float(contract["model_protocol"]["primary_tie_tolerance"]),
                    forbidden_features=forbidden_features,
                )
                candidate_rows.extend(candidate)
                selected_rows.append(selected)
                fold_rows.extend(fold_metric)
                prediction_rows.extend(predictions)
                calibration_training_rows.extend(calibration_training)
                calibrator_rows.extend(calibrators)
    candidates = pd.DataFrame(candidate_rows).sort_values(["formulation_id", "repetition", "outer_fold", "candidate_index"]).reset_index(drop=True)
    selected = pd.DataFrame(selected_rows).sort_values(["formulation_id", "repetition", "outer_fold"]).reset_index(drop=True)
    fold_metrics = pd.DataFrame(fold_rows).sort_values(["formulation_id", "repetition", "outer_fold", "system"]).reset_index(drop=True)
    oof = pd.DataFrame(prediction_rows).sort_values(["formulation_id", "repetition", "system", "sample_index"]).reset_index(drop=True)
    calibration_training = pd.DataFrame(calibration_training_rows).sort_values(
        ["formulation_id", "repetition", "outer_fold", "sample_index"]
    ).reset_index(drop=True)
    calibrators = pd.DataFrame(calibrator_rows).sort_values(["formulation_id", "repetition", "outer_fold", "class_label"]).reset_index(drop=True)
    _validate_oof(oof, fold_lookup, formulations, full_run=full_run)
    _require(not candidates["outer_test_used_for_selection"].astype(bool).any(), "Outer test entered selection.")
    _require(candidates.groupby(["formulation_id", "repetition", "outer_fold"])["selected_by_protocol"].sum().eq(1).all(), "A fold lacks exactly one selected candidate.")
    repetition_metrics, variability, per_class, confusion = _summarize_oof(oof, formulations, full_run=full_run)
    v2_class, v2_confusion, v2_metrics = _canonical_v2_class_results(contract)
    baseline = _baseline_comparisons(repetition_metrics)
    cv_design = _cv_design_sensitivity(variability, v2_metrics) if full_run else pd.DataFrame()
    return HRDatasetSensitivityResult(
        candidate_search_results=candidates,
        selected_hyperparameters=selected,
        fold_metrics=fold_metrics,
        oof_predictions=oof,
        calibration_training_oof=calibration_training,
        calibrator_parameters=calibrators,
        repetition_metrics=repetition_metrics,
        variability_summary=variability,
        per_class_metrics=per_class,
        confusion_matrices=confusion,
        baseline_comparisons=baseline,
        target_mapping_support=_target_mapping_support(dataset, formulations),
        cv_design_sensitivity=cv_design,
        canonical_v2_per_class_metrics=v2_class,
        canonical_v2_confusion_matrix=v2_confusion,
        protocol_comparison=_protocol_comparison(),
        fold_contracts=tuple(fold_contracts),
        evidence_status=evidence_status,
    )


def preflight_hrdataset_sensitivity_v3(
    *, contract_path: Path | str = DEFAULT_HRDATASET_SENSITIVITY_CONTRACT
) -> dict[str, Any]:
    """Validate sources, mappings, features and all fold contracts without fitting."""

    path = Path(contract_path)
    contract, receipt, dataset, features, _ = _prepare_inputs(path)
    formulations = _formulation_map(contract)
    fold_hashes: list[str] = []
    semantic_hashes: list[str] = []
    support: dict[str, dict[int, int]] = {}
    for formulation_id in FORMULATIONS:
        target = _target_series(dataset, formulations[formulation_id])
        support[formulation_id] = target.value_counts().sort_index().astype(int).to_dict()
        source = _fold_source(dataset, target)
        for seed_record in contract["design"]["seed_schedule"]:
            folds = generate_shared_folds(
                source,
                target_column="Phase3ATarget",
                id_column="ExternalSampleId",
                run_id=f"phase3a_preflight_{formulation_id}_rep{seed_record['repetition']}",
                config_hash=receipt["contract_sha256"],
                scientific_input_hash=receipt["contract_sha256"],
                dataset_key=f"hrdataset_v14_{formulation_id}",
                dataset_sha256=receipt["source_hashes"]["raw_dataset"],
                outer_splits=5,
                inner_splits=5,
                seed=int(seed_record["outer_seed"]),
                inner_seed=int(seed_record["inner_seed"]),
            )
            validate_shared_folds(folds)
            fold_hashes.append(str(folds.contract["fold_contract_hash"]))
            semantic_hashes.append(_outer_assignment_semantic_sha256(folds))
    _require(len(set(fold_hashes)) == 10 and len(set(semantic_hashes)) == 10, "Phase 3A fold identities are not distinct.")
    return {
        "status": "passed",
        "contract_sha256": receipt["contract_sha256"],
        "dataset_sha256": receipt["source_hashes"]["raw_dataset"],
        "sample_count": len(features),
        "feature_count": features.shape[1],
        "target_support": support,
        "formulations": 2,
        "repetitions_per_formulation": 5,
        "outer_folds": 5,
        "inner_folds": 5,
        "distinct_fold_contracts": len(set(fold_hashes)),
        "distinct_outer_assignments": len(set(semantic_hashes)),
        "planned_xgboost_fit_calls": 2300,
        "planned_baseline_fit_calls": 150,
        "model_fit_count": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def diagnostic_hrdataset_sensitivity_v3(
    *, contract_path: Path | str = DEFAULT_HRDATASET_SENSITIVITY_CONTRACT
) -> dict[str, Any]:
    """Run both formulations for one outer fold without persisting evidence."""

    path = Path(contract_path)
    with enforce_offline_runtime() as offline_state:
        contract, receipt, dataset, features, definition = _prepare_inputs(path)
        scientific_hash = _canonical_json_sha256(
            {"contract_sha256": receipt["contract_sha256"], "dataset_sha256": receipt["source_hashes"]["raw_dataset"], "diagnostic": True}
        )
        result = evaluate_hrdataset_sensitivity_v3(
            dataset,
            features,
            contract,
            definition,
            run_id="phase3a_diagnostic_incomplete_never_canonical",
            contract_sha256=receipt["contract_sha256"],
            scientific_input_sha256=scientific_hash,
            dataset_sha256=receipt["source_hashes"]["raw_dataset"],
            repetition_subset=[1],
            outer_fold_subset=[1],
        )
        runtime = offline_state.receipt()
    return {
        "status": result.evidence_status,
        "persisted": False,
        "formulations": 2,
        "outer_folds_evaluated": 2,
        "xgboost_fit_calls": 92,
        "baseline_fit_calls": 6,
        "candidate_rows": len(result.candidate_search_results),
        "oof_rows": len(result.oof_predictions),
        "runtime_policy": runtime,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _write_json(path: Path, payload: Any) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _run_impl(*, contract_path: Path, output_dir: Path, run_id: str, offline_state: Any) -> dict[str, Any]:
    git_identity = _clean_git_identity()
    contract, receipt, dataset, features, definition = _prepare_inputs(contract_path)
    _require(not output_dir.exists(), f"Output destination already exists: {output_dir}.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        implementation_paths = (
            Path("src/experiments/hrdataset_sensitivity_v3.py"),
            Path("src/governance/hrdataset_sensitivity_contract_v3.py"),
            Path("src/experiments/shared_folds.py"),
            Path("src/models/canonical_models.py"),
            Path("src/models/ordinal_models_v3.py"),
            Path("src/models/ordinal_evaluation_v3.py"),
            Path("src/experiments/manuscript_calibration.py"),
        )
        current_source_tree = source_tree_hash(PROJECT_ROOT)
        scientific_inputs = {
            "git_identity": git_identity,
            "source_tree_hash": current_source_tree,
            "contract_sha256": receipt["contract_sha256"],
            "source_hashes": receipt["source_hashes"],
            "implementation_hashes": {path.as_posix(): sha256_file(path) for path in implementation_paths},
        }
        scientific_hash = _canonical_json_sha256(scientific_inputs)
        result = evaluate_hrdataset_sensitivity_v3(
            dataset,
            features,
            contract,
            definition,
            run_id=run_id,
            contract_sha256=receipt["contract_sha256"],
            scientific_input_sha256=scientific_hash,
            dataset_sha256=receipt["source_hashes"]["raw_dataset"],
        )
        frames = {
            "baseline_comparisons.csv": result.baseline_comparisons,
            "calibration_training_oof.csv": result.calibration_training_oof,
            "calibrator_parameters.csv": result.calibrator_parameters,
            "candidate_search_results.csv": result.candidate_search_results,
            "canonical_v2_confusion_matrix.csv": result.canonical_v2_confusion_matrix,
            "canonical_v2_per_class_metrics.csv": result.canonical_v2_per_class_metrics,
            "confusion_matrices.csv": result.confusion_matrices,
            "cv_design_sensitivity.csv": result.cv_design_sensitivity,
            "fold_metrics.csv": result.fold_metrics,
            "oof_predictions.csv": result.oof_predictions,
            "per_class_metrics.csv": result.per_class_metrics,
            "protocol_comparison.csv": result.protocol_comparison,
            "repetition_metrics.csv": result.repetition_metrics,
            "selected_hyperparameters.csv": result.selected_hyperparameters,
            "target_mapping_support.csv": result.target_mapping_support,
            "variability_summary.csv": result.variability_summary,
        }
        for filename, frame in frames.items():
            frame.to_csv(staging / filename, index=False)
        _write_json(staging / "fold_contracts.json", list(result.fold_contracts))
        output_hashes = {path.name: sha256_file(path) for path in sorted(staging.iterdir()) if path.is_file()}
        metadata = {
            "schema_version": 1,
            "stage": "hrdataset_sensitivity_v3",
            "status": "complete",
            "evidence_status": result.evidence_status,
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "contract_sha256": receipt["contract_sha256"],
            "scientific_input_sha256": scientific_hash,
            "scientific_inputs": scientific_inputs,
            "git_identity": git_identity,
            "dataset_sha256": receipt["source_hashes"]["raw_dataset"],
            "sample_count": 311,
            "feature_policy": "conservative_primary",
            "feature_count": 7,
            "formulations": list(FORMULATIONS),
            "repetitions_per_formulation": 5,
            "outer_folds": 5,
            "inner_folds": 5,
            "systems": list(SYSTEMS),
            "xgboost_fit_calls": 2300,
            "baseline_fit_calls": 150,
            "candidate_search_row_count": len(result.candidate_search_results),
            "oof_prediction_row_count": len(result.oof_predictions),
            "calibration_training_oof_row_count": len(result.calibration_training_oof),
            "repetition_metric_row_count": len(result.repetition_metrics),
            "calibrator_parameter_row_count": len(result.calibrator_parameters),
            "outer_test_used_for_selection_or_calibration": False,
            "cross_formulation_metric_difference_computed": False,
            "locked_model_transport": False,
            "employee_level_output_publication_authorized": False,
            "runtime_policy": offline_state.receipt(),
            "network_calls": 0,
            "paid_api_calls": 0,
            "output_hashes": output_hashes,
        }
        _require(_clean_git_identity() == git_identity, "Git identity changed during execution.")
        _require(source_tree_hash(PROJECT_ROOT) == current_source_tree, "Scientific source tree changed during execution.")
        repeated_receipt = validate_hrdataset_sensitivity_contract_v3(contract_path)
        _require(repeated_receipt["contract_sha256"] == receipt["contract_sha256"], "Phase 3A contract changed during execution.")
        _write_json(staging / "stage_metadata.json", metadata)
        _require({path.name for path in staging.iterdir() if path.is_file()} == EXPECTED_LOCAL_FILES, "Phase 3A local output inventory drifted.")
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
        "contract_sha256": receipt["contract_sha256"],
        "scientific_input_sha256": scientific_hash,
        "xgboost_fit_calls": 2300,
        "baseline_fit_calls": 150,
        "oof_prediction_row_count": len(result.oof_predictions),
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def run_hrdataset_sensitivity_v3(
    *,
    output_dir: Path | str,
    run_id: str,
    contract_path: Path | str = DEFAULT_HRDATASET_SENSITIVITY_CONTRACT,
) -> dict[str, Any]:
    with enforce_offline_runtime() as offline_state:
        return _run_impl(
            contract_path=Path(contract_path),
            output_dir=Path(output_dir),
            run_id=run_id,
            offline_state=offline_state,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_HRDATASET_SENSITIVITY_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_LOCAL_RUN_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _require(not (args.preflight_only and args.diagnostic), "Preflight and diagnostic modes are mutually exclusive.")
    if args.preflight_only:
        receipt = preflight_hrdataset_sensitivity_v3(contract_path=args.contract)
    elif args.diagnostic:
        receipt = diagnostic_hrdataset_sensitivity_v3(contract_path=args.contract)
    else:
        _require(isinstance(args.run_id, str) and bool(args.run_id.strip()), "--run-id is required for a complete run.")
        receipt = run_hrdataset_sensitivity_v3(
            contract_path=args.contract,
            output_dir=args.output_root / args.run_id / "hrdataset_sensitivity",
            run_id=args.run_id,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
