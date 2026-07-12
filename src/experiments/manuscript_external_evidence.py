"""Canonical, versioned external evidence for the manuscript package.

This stage intentionally does not call the legacy external-report writer.  The
legacy writer owns historical paths and also produces unrelated XAI/LLM
artifacts.  Here every write is rooted below the caller-supplied output
directory and every table is bound to one manuscript ``run_id`` and semantic
``config_hash``.

The four task/dataset roles are deliberately kept separate:

* HRDataset_v14: independently trained performance-target replication;
* IBM performance: restricted-target robustness (observed classes 3/4 only);
* IBM attrition and Employee Turnover: related binary task transfer.

No result from this module is locked-INX-model transport evidence and no result
authorises an autonomous HR decision.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data.external_adapters import (
    ExternalDataset,
    audit_attribute_columns,
    build_feature_columns,
    load_external_dataset,
    role_columns,
)
from src.data.load_data import _read_csv_with_best_effort
from src.experiments.final_evidence_common import align_proba, predict_labels_from_proba
from src.experiments.final_shap_stability import get_group_mapping, group_shap_values, normalize_shap_values
from src.experiments.fairness_sensitivity import (
    compute_disparity_summary,
    compute_group_metrics,
    compute_small_group_warnings,
)
from src.experiments.leakage_safe_cv import LabelEncodedXGBClassifier, infer_columns, make_preprocessor
from src.governance.external_claims import external_allowed_claim
from src.governance.manuscript_contract import canonical_config_hash, sha256_file
from src.models.evaluate import classification_metrics
from src.models.task_schema import (
    BINARY_ATTRITION_TRANSFER,
    BINARY_TURNOVER_TRANSFER,
    KNOWN_METRICS,
    ORDINAL_MULTICLASS_PERFORMANCE,
    RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS,
    get_task_schema,
)
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_CONFIG = Path("configs/manuscript_final.yaml")
MODEL_NAME = "xgboost"
MINIMUM_TRANSPORT_FEATURES = 5
REPORT_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "weighted_precision",
    "macro_recall",
    "weighted_recall",
    "quadratic_weighted_kappa",
    "ordinal_mae",
    "adjacent_accuracy",
    "severe_error_rate",
    "nll_log_loss",
    "multiclass_brier",
    "binary_brier",
    "roc_auc",
    "average_precision",
    "ece_confidence",
)


class ExternalEvidenceError(RuntimeError):
    """Raised when external evidence violates its task or provenance contract."""


@dataclass(frozen=True)
class ExternalRunSpec:
    key: str
    config_dataset_key: str
    dataset_name: str
    target_kind: str
    task_type: str
    role: str
    expected_config_role: str
    expected_labels: tuple[int, ...]
    policies: tuple[str, ...]


RUN_SPECS: tuple[ExternalRunSpec, ...] = (
    ExternalRunSpec(
        key="hrdataset_v14",
        config_dataset_key="hrdataset_v14",
        dataset_name="hrdataset_v14",
        target_kind="primary",
        task_type=ORDINAL_MULTICLASS_PERFORMANCE,
        role="independent external performance-target replication",
        expected_config_role="independent_external_performance_target_replication",
        expected_labels=(2, 3, 4),
        policies=("department_including", "department_free", "department_job_role_free"),
    ),
    ExternalRunSpec(
        key="ibm_performance",
        config_dataset_key="ibm_hr_analytics",
        dataset_name="ibm_hr_analytics",
        target_kind="primary",
        task_type=RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS,
        role="restricted-target performance robustness",
        expected_config_role="restricted_target_performance_robustness",
        expected_labels=(3, 4),
        policies=("department_including", "department_free", "department_job_role_free"),
    ),
    ExternalRunSpec(
        key="ibm_attrition",
        config_dataset_key="ibm_hr_analytics_attrition",
        dataset_name="ibm_hr_analytics",
        target_kind="attrition",
        task_type=BINARY_ATTRITION_TRANSFER,
        role="related HR attrition task transfer",
        expected_config_role="related_task_transfer",
        expected_labels=(0, 1),
        policies=("department_including", "department_free", "department_job_role_free"),
    ),
    ExternalRunSpec(
        key="employee_turnover",
        config_dataset_key="employee_turnover",
        dataset_name="employee_turnover",
        target_kind="primary",
        task_type=BINARY_TURNOVER_TRANSFER,
        role="related HR turnover task transfer",
        expected_config_role="related_task_transfer",
        expected_labels=(0, 1),
        policies=("with_last_evaluation", "without_last_evaluation"),
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = config.get("manuscript_final", config)
    if not isinstance(settings, Mapping):
        raise ExternalEvidenceError("Canonical config must contain a manuscript_final mapping.")
    return settings


def _resolve_seed(settings: Mapping[str, Any], value: Any) -> int:
    if isinstance(value, int):
        return value
    seeds = settings.get("seeds", {})
    if not isinstance(seeds, Mapping) or value not in seeds or not isinstance(seeds[value], int):
        raise ExternalEvidenceError(f"Cannot resolve canonical seed {value!r}.")
    return int(seeds[value])


def configured_run_specs(config: Mapping[str, Any]) -> tuple[ExternalRunSpec, ...]:
    """Validate canonical dataset declarations against the immutable claim registry."""

    settings = _settings(config)
    datasets = settings.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ExternalEvidenceError("manuscript_final.datasets must be a mapping.")
    for spec in RUN_SPECS:
        entry = datasets.get(spec.config_dataset_key)
        if not isinstance(entry, Mapping):
            raise ExternalEvidenceError(f"Missing canonical dataset declaration: {spec.config_dataset_key}")
        if entry.get("task_type") != spec.task_type:
            raise ExternalEvidenceError(
                f"Task drift for {spec.config_dataset_key}: expected {spec.task_type!r}, "
                f"observed {entry.get('task_type')!r}."
            )
        if entry.get("role") != spec.expected_config_role:
            raise ExternalEvidenceError(
                f"Role drift for {spec.config_dataset_key}: expected {spec.expected_config_role!r}, "
                f"observed {entry.get('role')!r}."
            )
        registered = external_allowed_claim(spec.dataset_name, spec.target_kind)
        if registered != spec.role:
            raise ExternalEvidenceError(
                f"External claim registry drift for {spec.key}: expected {spec.role!r}, observed {registered!r}."
            )
        configured_claim = str(entry.get("allowed_claim", "")).strip().lower()
        if not configured_claim:
            raise ExternalEvidenceError(f"Canonical allowed claim is blank for {spec.config_dataset_key}.")
        if spec.key == "hrdataset_v14" and spec.role not in configured_claim:
            raise ExternalEvidenceError("HRDataset_v14 must be labelled as independent replication.")
        if spec.key == "ibm_performance" and spec.role not in configured_claim:
            raise ExternalEvidenceError("IBM performance must be labelled restricted-target robustness.")
        if spec.key in {"ibm_attrition", "employee_turnover"} and "not employee-performance validation" not in configured_claim:
            raise ExternalEvidenceError(f"Related-task claim boundary is missing for {spec.config_dataset_key}.")
    return RUN_SPECS


def target_mapping_table(
    dataset: ExternalDataset,
    *,
    run_id: str,
    config_hash: str,
    spec: ExternalRunSpec,
) -> pd.DataFrame:
    """Return observed raw-to-canonical mapping counts, never inferred counts."""

    raw = dataset.raw[dataset.target_raw_column].astype("string").fillna("__MISSING__")
    mapped = dataset.canonical[dataset.target_column].astype(int)
    rows = pd.DataFrame({"raw_target_value": raw, "mapped_label": mapped}).value_counts(sort=False).reset_index(name="n_rows")
    rows.insert(0, "config_hash", config_hash)
    rows.insert(0, "run_id", run_id)
    rows.insert(2, "dataset_key", spec.key)
    rows["target_raw_column"] = dataset.target_raw_column
    rows["target_canonical_column"] = dataset.target_column
    rows["mapping_complete"] = not bool(dataset.unmapped_target_values)
    return rows.sort_values(["mapped_label", "raw_target_value"], kind="stable").reset_index(drop=True)


def target_support_table(
    dataset: ExternalDataset,
    *,
    run_id: str,
    config_hash: str,
    spec: ExternalRunSpec,
    requested_splits: int,
) -> pd.DataFrame:
    labels = tuple(int(v) for v in dataset.labels)
    if labels != spec.expected_labels:
        raise ExternalEvidenceError(
            f"Unexpected labels for {spec.key}: expected {spec.expected_labels}, observed {labels}."
        )
    if dataset.unmapped_target_values:
        raise ExternalEvidenceError(f"Unmapped target values for {spec.key}: {dataset.unmapped_target_values}")
    counts = dataset.canonical[dataset.target_column].astype(int).value_counts().sort_index()
    effective_splits = min(requested_splits, int(counts.min()))
    if effective_splits < 2:
        raise ExternalEvidenceError(f"Insufficient class support for stratified CV in {spec.key}.")
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "dataset_key": spec.key,
                "task_type": spec.task_type,
                "label": int(label),
                "n_rows": int(count),
                "proportion": float(count / counts.sum()),
                "requested_cv_splits": requested_splits,
                "effective_cv_splits": effective_splits,
                "minimum_class_support": int(counts.min()),
                "mapping_verified": True,
            }
            for label, count in counts.items()
        ]
    )


def _model_parameters(settings: Mapping[str, Any]) -> dict[str, Any]:
    model = settings.get("model", {})
    raw = model.get("xgboost", {}) if isinstance(model, Mapping) else {}
    if not isinstance(raw, Mapping):
        raise ExternalEvidenceError("manuscript_final.model.xgboost must be a mapping.")
    parameters = dict(raw)
    parameters.pop("random_state_seed", None)
    return parameters


def _fit_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    parameters: Mapping[str, Any],
    seed: int,
    labels: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    model_parameters = dict(parameters)
    model_parameters["random_state"] = seed
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(X_train)),
            ("model", LabelEncodedXGBClassifier(**model_parameters)),
        ]
    )
    pipeline.fit(X_train, y_train)
    model = pipeline.named_steps["model"]
    probabilities = align_proba(pipeline.predict_proba(X_test), model.classes_, list(labels))
    predictions = predict_labels_from_proba(probabilities, list(labels))
    return np.asarray(predictions, dtype=int), np.asarray(probabilities, dtype=float), pipeline


def _metric_row(
    y_true: Sequence[int] | pd.Series,
    y_pred: Sequence[int] | pd.Series,
    probabilities: np.ndarray,
    *,
    labels: Sequence[int],
    task_type: str,
) -> dict[str, Any]:
    return classification_metrics(
        y_true,
        y_pred,
        probabilities,
        labels=list(labels),
        task_type=task_type,
    )


def validate_task_metric_rows(frame: pd.DataFrame) -> None:
    """Fail if an inapplicable task metric is encoded as a scientific value."""

    if frame.empty:
        raise ExternalEvidenceError("External metric table is empty.")
    if "task_type" not in frame or "role" not in frame:
        raise ExternalEvidenceError("External metric rows require task_type and role fields.")
    for row in frame.to_dict(orient="records"):
        schema = get_task_schema(str(row["task_type"]))
        for metric in REPORT_METRICS:
            if metric not in row or schema.is_metric_applicable(metric):
                continue
            value = row[metric]
            if value is not None and not (isinstance(value, float) and math.isnan(value)) and str(value).strip() != "":
                raise ExternalEvidenceError(
                    f"Inapplicable metric {metric!r} has value {value!r} for task {schema.name!r}."
                )
        role = str(row["role"]).lower()
        if schema.comparison_group == "related_binary_task_transfer" and "task transfer" not in role:
            raise ExternalEvidenceError("A related binary task was given a performance-validation role.")
        if schema.name == RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS and "restricted-target" not in role:
            raise ExternalEvidenceError("Restricted-target evidence was given a comparable-performance role.")


def _fold_local_shap_rows(
    *,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predicted: np.ndarray,
    test_positions: np.ndarray,
    fold: int,
    labels: Sequence[int],
    dataset: ExternalDataset,
    spec: ExternalRunSpec,
    policy: str,
    run_id: str,
    config_hash: str,
) -> list[dict[str, Any]]:
    """Explain the exact fold model that produced each external OOF prediction."""

    import shap

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["model"]
    numeric_columns, categorical_columns = infer_columns(X_train)
    group_names, mapping = get_group_mapping(preprocessor, numeric_columns, categorical_columns)
    if set(group_names) != set(X_train.columns):
        raise ExternalEvidenceError("External SHAP grouping lost or added a raw feature family.")
    transformed = preprocessor.transform(X_test)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    raw_values = shap.TreeExplainer(classifier.model_).shap_values(transformed)
    normalized = normalize_shap_values(
        raw_values,
        n_samples=len(X_test),
        n_features=transformed.shape[1],
        n_classes=len(labels),
    )
    grouped = group_shap_values(normalized, group_names, mapping)
    rows: list[dict[str, Any]] = []
    for row_offset, sample_position in enumerate(test_positions):
        sample_index = int(dataset.canonical.index[int(sample_position)])
        predicted_class = int(predicted[row_offset])
        for class_index, class_label in enumerate(labels):
            for feature_index, feature in enumerate(group_names):
                value = float(grouped[row_offset, class_index, feature_index])
                rows.append(
                    {
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "dataset_key": spec.key,
                        "task_type": spec.task_type,
                        "role": spec.role,
                        "policy": policy,
                        "feature_set": policy,
                        "fold": fold,
                        "sample_index": sample_index,
                        "external_sample_id": int(dataset.canonical.iloc[int(sample_position)]["ExternalSampleId"]),
                        "y_true": int(y_test.iloc[row_offset]),
                        "predicted_class": predicted_class,
                        "class_label": int(class_label),
                        "is_predicted_class": bool(int(class_label) == predicted_class),
                        "feature": feature,
                        "grouped_shap_value": value,
                        "abs_grouped_shap_value": abs(value),
                        "evaluation_scope": "case_specific_oof_fold_model",
                        "prediction_identity_verified": True,
                        "control_type": "external_context_dependent",
                        "sensitive_or_proxy": bool(feature in role_columns(dataset, "proxy")),
                        "leakage_risk": False,
                        "governance_notes": (
                            "OOF model attribution only; not a causal effect, fairness finding, or employee prescription."
                        ),
                    }
                )
    return rows


def _representative_cases(
    predictions: pd.DataFrame,
    *,
    run_id: str,
    config_hash: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, group in predictions.groupby("policy", sort=False):
        candidates: list[tuple[str, pd.DataFrame, bool]] = [
            ("correct_high_confidence", group[group["correct"]], False),
            ("correct_low_confidence", group[group["correct"]], True),
            ("incorrect_high_confidence", group[~group["correct"]], False),
            ("incorrect_low_confidence", group[~group["correct"]], True),
        ]
        seen: set[int] = set()
        for case_type, subset, ascending in candidates:
            if subset.empty:
                continue
            for item in subset.sort_values(["confidence", "sample_index"], ascending=[ascending, True]).itertuples(index=False):
                if int(item.sample_index) in seen:
                    continue
                seen.add(int(item.sample_index))
                rows.append(
                    {
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "dataset_key": item.dataset_key,
                        "task_type": item.task_type,
                        "role": item.role,
                        "policy": policy,
                        "case_type": case_type,
                        "sample_index": int(item.sample_index),
                        "external_sample_id": int(item.external_sample_id),
                        "fold": int(item.fold),
                        "true_class": int(item.y_true),
                        "predicted_class": int(item.y_pred),
                        "confidence": float(item.confidence),
                        "correct": bool(item.correct),
                    }
                )
                break
        distribution = group["y_true"].value_counts()
        minority_label = int(distribution.sort_values().index[0])
        minority = group[group["y_true"] == minority_label].sort_values(
            ["correct", "confidence", "sample_index"], ascending=[False, False, True]
        )
        if not minority.empty:
            item = minority.iloc[0]
            if int(item["sample_index"]) not in seen:
                rows.append(
                    {
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "dataset_key": item["dataset_key"],
                        "task_type": item["task_type"],
                        "role": item["role"],
                        "policy": policy,
                        "case_type": f"minority_true_class_{minority_label}",
                        "sample_index": int(item["sample_index"]),
                        "external_sample_id": int(item["external_sample_id"]),
                        "fold": int(item["fold"]),
                        "true_class": int(item["y_true"]),
                        "predicted_class": int(item["y_pred"]),
                        "confidence": float(item["confidence"]),
                        "correct": bool(item["correct"]),
                    }
                )
    return pd.DataFrame(rows)


def _hr_fairness_outputs(
    dataset: ExternalDataset,
    predictions: pd.DataFrame,
    *,
    output_dir: Path,
    labels: Sequence[int],
    run_id: str,
    config_hash: str,
    min_support: int,
) -> dict[str, Path]:
    fairness_dir = output_dir / "fairness_proxy"
    fairness_dir.mkdir(parents=True, exist_ok=True)
    attributes = audit_attribute_columns(dataset)
    group_metrics = compute_group_metrics(predictions, dataset.canonical, attributes, list(labels), min_support)
    disparity = compute_disparity_summary(group_metrics, min_support) if not group_metrics.empty else pd.DataFrame()
    warnings = compute_small_group_warnings(dataset.canonical, attributes, min_support)
    for frame in (group_metrics, disparity, warnings):
        frame.insert(0, "config_hash", config_hash)
        frame.insert(0, "run_id", run_id)
        frame.insert(2, "dataset_key", "hrdataset_v14")
        frame.insert(3, "task_type", ORDINAL_MULTICLASS_PERFORMANCE)
        frame.insert(4, "role", "independent external performance-target replication")
    paths = {
        "fairness_group_metrics": fairness_dir / "fairness_group_metrics.csv",
        "fairness_disparity_summary": fairness_dir / "fairness_disparity_summary.csv",
        "small_group_warnings": fairness_dir / "small_group_warnings.csv",
    }
    group_metrics.to_csv(paths["fairness_group_metrics"], index=False)
    disparity.to_csv(paths["fairness_disparity_summary"], index=False)
    warnings.to_csv(paths["small_group_warnings"], index=False)
    return paths


def _run_dataset_task(
    spec: ExternalRunSpec,
    *,
    output_dir: Path,
    settings: Mapping[str, Any],
    run_id: str,
    config_hash: str,
) -> dict[str, Path]:
    dataset = load_external_dataset(spec.dataset_name, target_kind=spec.target_kind)
    if dataset.task_type != spec.task_type:
        raise ExternalEvidenceError(
            f"Adapter task drift for {spec.key}: expected {spec.task_type}, observed {dataset.task_type}."
        )
    configured_policies = set(dataset.config.feature_policy_variants)
    missing_policies = sorted(set(spec.policies).difference(configured_policies))
    if missing_policies:
        raise ExternalEvidenceError(f"Missing mapped feature policies for {spec.key}: {missing_policies}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cv = settings.get("evaluation", {}).get("cv", {})
    requested_splits = int(cv.get("n_splits", 10))
    seed = _resolve_seed(settings, cv.get("seed", "cv"))
    support = target_support_table(
        dataset,
        run_id=run_id,
        config_hash=config_hash,
        spec=spec,
        requested_splits=requested_splits,
    )
    mapping = target_mapping_table(dataset, run_id=run_id, config_hash=config_hash, spec=spec)
    effective_splits = int(support["effective_cv_splits"].iloc[0])
    labels = list(spec.expected_labels)
    y = dataset.canonical[dataset.target_column].astype(int)
    splitter = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=bool(cv.get("shuffle", True)),
        random_state=seed,
    )
    folds = list(splitter.split(dataset.canonical, y))
    parameters = _model_parameters(settings)

    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    oof_local_shap_rows: list[dict[str, Any]] = []
    for policy in spec.policies:
        features = build_feature_columns(dataset, policy)
        forbidden = sorted(
            set(features).intersection(
                set(role_columns(dataset, "id"))
                | set(role_columns(dataset, "leakage"))
                | set(role_columns(dataset, "sensitive"))
            )
        )
        if forbidden:
            raise ExternalEvidenceError(f"Forbidden mapped features in {spec.key}/{policy}: {forbidden}")
        policy_rows.append(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "dataset_key": spec.key,
                "task_type": spec.task_type,
                "role": spec.role,
                "policy": policy,
                "n_features": len(features),
                "feature_columns": ";".join(features),
                "excluded_id_columns": ";".join(role_columns(dataset, "id")),
                "excluded_leakage_columns": ";".join(role_columns(dataset, "leakage")),
                "excluded_sensitive_columns": ";".join(role_columns(dataset, "sensitive")),
                "forbidden_feature_count": 0,
            }
        )
        X = dataset.canonical.loc[:, features].copy()
        for fold, (train_positions, test_positions) in enumerate(folds, start=1):
            predicted, probabilities, pipeline = _fit_predict(
                X.iloc[train_positions],
                y.iloc[train_positions],
                X.iloc[test_positions],
                parameters=parameters,
                seed=seed,
                labels=labels,
            )
            y_test = y.iloc[test_positions]
            if spec.key == "hrdataset_v14" and policy == "department_free":
                oof_local_shap_rows.extend(
                    _fold_local_shap_rows(
                        pipeline=pipeline,
                        X_train=X.iloc[train_positions],
                        X_test=X.iloc[test_positions],
                        y_test=y_test,
                        predicted=predicted,
                        test_positions=np.asarray(test_positions),
                        fold=fold,
                        labels=labels,
                        dataset=dataset,
                        spec=spec,
                        policy=policy,
                        run_id=run_id,
                        config_hash=config_hash,
                    )
                )
            metrics = _metric_row(y_test, predicted, probabilities, labels=labels, task_type=spec.task_type)
            fold_metric_rows.append(
                {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "dataset_key": spec.key,
                    "task_type": spec.task_type,
                    "task_comparison_group": get_task_schema(spec.task_type).comparison_group,
                    "role": spec.role,
                    "policy": policy,
                    "feature_set": policy,
                    "model": MODEL_NAME,
                    "fold": fold,
                    "n_test": len(test_positions),
                    **metrics,
                }
            )
            for row_offset, sample_position in enumerate(test_positions):
                row = {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "dataset_key": spec.key,
                    "task_type": spec.task_type,
                    "role": spec.role,
                    "policy": policy,
                    "feature_set": policy,
                    "model": MODEL_NAME,
                    "fold": fold,
                    "sample_index": int(dataset.canonical.index[sample_position]),
                    "external_sample_id": int(dataset.canonical.iloc[sample_position]["ExternalSampleId"]),
                    "y_true": int(y_test.iloc[row_offset]),
                    "y_pred": int(predicted[row_offset]),
                    "correct": bool(int(y_test.iloc[row_offset]) == int(predicted[row_offset])),
                    "confidence": float(np.max(probabilities[row_offset])),
                }
                for label_index, label in enumerate(labels):
                    row[f"prob_class_{label}"] = float(probabilities[row_offset, label_index])
                prediction_rows.append(row)

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    validate_task_metric_rows(fold_metrics)
    summaries: list[dict[str, Any]] = []
    confidence_level = float(settings.get("evaluation", {}).get("bootstrap", {}).get("confidence_level", 0.95))
    if confidence_level != 0.95:
        raise ExternalEvidenceError("External fold uncertainty currently requires the canonical 0.95 confidence level.")
    for policy, prediction_group in predictions.groupby("policy", sort=False):
        probability_columns = [f"prob_class_{label}" for label in labels]
        aggregate = _metric_row(
            prediction_group["y_true"],
            prediction_group["y_pred"],
            prediction_group[probability_columns].to_numpy(dtype=float),
            labels=labels,
            task_type=spec.task_type,
        )
        row: dict[str, Any] = {
            "run_id": run_id,
            "config_hash": config_hash,
            "dataset_key": spec.key,
            "task_type": spec.task_type,
            "task_comparison_group": get_task_schema(spec.task_type).comparison_group,
            "role": spec.role,
            "policy": policy,
            "model": MODEL_NAME,
            "n_rows": len(prediction_group),
            "n_features": int(next(value["n_features"] for value in policy_rows if value["policy"] == policy)),
            "labels": ";".join(str(label) for label in labels),
            "n_folds": effective_splits,
            "locked_inx_model_transported": False,
            **aggregate,
        }
        fold_group = fold_metrics[fold_metrics["policy"] == policy]
        for metric in REPORT_METRICS:
            if metric not in fold_group or not get_task_schema(spec.task_type).is_metric_applicable(metric):
                row[f"{metric}_fold_mean"] = None
                row[f"{metric}_fold_std"] = None
                row[f"{metric}_fold_ci95_low"] = None
                row[f"{metric}_fold_ci95_high"] = None
                continue
            values = pd.to_numeric(fold_group[metric], errors="coerce").dropna()
            if values.empty:
                row[f"{metric}_fold_mean"] = None
                row[f"{metric}_fold_std"] = None
                row[f"{metric}_fold_ci95_low"] = None
                row[f"{metric}_fold_ci95_high"] = None
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half_width = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{metric}_fold_mean"] = mean
            row[f"{metric}_fold_std"] = std
            row[f"{metric}_fold_ci95_low"] = mean - half_width
            row[f"{metric}_fold_ci95_high"] = mean + half_width
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    validate_task_metric_rows(summary)

    paths = {
        "target_mapping": output_dir / "target_mapping.csv",
        "target_support": output_dir / "target_support.csv",
        "feature_policy": output_dir / "feature_policy_audit.csv",
        "predictions": output_dir / "model_predictions.csv",
        "fold_metrics": output_dir / "fold_metrics.csv",
        "policy_summary": output_dir / "performance_metrics.csv",
        "representative_cases": output_dir / "representative_cases.csv",
        "actionability_summary": output_dir / "actionability_summary.csv",
        "metadata": output_dir / "experiment_metadata.json",
    }
    mapping.to_csv(paths["target_mapping"], index=False)
    support.to_csv(paths["target_support"], index=False)
    pd.DataFrame(policy_rows).to_csv(paths["feature_policy"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    summary.to_csv(paths["policy_summary"], index=False)
    _representative_cases(predictions, run_id=run_id, config_hash=config_hash).to_csv(
        paths["representative_cases"], index=False
    )
    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "dataset_key": spec.key,
                "task_type": spec.task_type,
                "role": spec.role,
                "policy": policy,
                "actionability_status": "not_evaluated_for_external_dataset_replication",
                "validity": None,
                "denominator": 0,
                "warning": "No external counterfactual validity claim; model scenarios would not be employee prescriptions.",
            }
            for policy in spec.policies
        ]
    ).to_csv(paths["actionability_summary"], index=False)

    if spec.key == "hrdataset_v14":
        local_shap = pd.DataFrame(oof_local_shap_rows)
        expected_case_class_feature_rows = len(dataset.canonical) * len(labels) * int(
            next(row["n_features"] for row in policy_rows if row["policy"] == "department_free")
        )
        if len(local_shap) != expected_case_class_feature_rows:
            raise ExternalEvidenceError(
                "Incomplete HRDataset OOF local SHAP coverage: "
                f"expected {expected_case_class_feature_rows}, observed {len(local_shap)}."
            )
        identity = local_shap.groupby(["sample_index", "fold", "predicted_class"], dropna=False).size().reset_index()
        predicted_identity = predictions[predictions["policy"] == "department_free"][
            ["sample_index", "fold", "y_pred"]
        ].rename(columns={"y_pred": "predicted_class"})
        if len(identity) != len(dataset.canonical) or not identity[
            ["sample_index", "fold", "predicted_class"]
        ].sort_values("sample_index").reset_index(drop=True).equals(
            predicted_identity[["sample_index", "fold", "predicted_class"]]
            .sort_values("sample_index")
            .reset_index(drop=True)
        ):
            raise ExternalEvidenceError("HRDataset local SHAP does not match the OOF prediction identity.")
        shap_dir = output_dir / "shap" / "department_free"
        shap_dir.mkdir(parents=True, exist_ok=True)
        paths["oof_local_shap"] = shap_dir / "local_grouped_shap_values.csv"
        paths["oof_global_shap"] = shap_dir / "global_grouped_shap_importance.csv"
        local_shap.to_csv(paths["oof_local_shap"], index=False)
        (
            local_shap.groupby("feature", as_index=False)["abs_grouped_shap_value"]
            .mean()
            .rename(columns={"abs_grouped_shap_value": "mean_abs_grouped_shap"})
            .sort_values("mean_abs_grouped_shap", ascending=False)
            .assign(
                run_id=run_id,
                config_hash=config_hash,
                dataset_key=spec.key,
                task_type=spec.task_type,
                role=spec.role,
                policy="department_free",
                evaluation_scope="out_of_fold_only",
            )[
                [
                    "run_id",
                    "config_hash",
                    "dataset_key",
                    "task_type",
                    "role",
                    "policy",
                    "evaluation_scope",
                    "feature",
                    "mean_abs_grouped_shap",
                ]
            ]
            .to_csv(paths["oof_global_shap"], index=False)
        )
        min_support = int(settings.get("fairness", {}).get("minimum_group_support", 30))
        paths.update(
            _hr_fairness_outputs(
                dataset,
                predictions,
                output_dir=output_dir,
                labels=labels,
                run_id=run_id,
                config_hash=config_hash,
                min_support=min_support,
            )
        )
    source_path = dataset.config.raw_path
    schema_path = source_path.parent / "schema_mapping.json"
    paths["metadata"].write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "status": "completed",
                "dataset_key": spec.key,
                "dataset_name": spec.dataset_name,
                "target_kind": spec.target_kind,
                "task_type": spec.task_type,
                "task_comparison_group": get_task_schema(spec.task_type).comparison_group,
                "role": spec.role,
                "locked_inx_model_transported": False,
                "mapping_verified": True,
                "n_rows": len(dataset.canonical),
                "labels": labels,
                "requested_cv_splits": requested_splits,
                "effective_cv_splits": effective_splits,
                "seed": seed,
                "policies": list(spec.policies),
                "raw_dataset_path": str(source_path),
                "raw_dataset_sha256": sha256_file(source_path),
                "schema_mapping_path": str(schema_path),
                "schema_mapping_sha256": sha256_file(schema_path),
                "completed_at": _utc_now(),
                "claim_limitations": [
                    "Research-grade decision support only; no autonomous HR decision use.",
                    "Dataset-specific models are not locked-INX-model transport.",
                    "SHAP attribution, if generated elsewhere, is not causal evidence.",
                    "Related attrition/turnover tasks are not performance-model replication.",
                ],
                "outputs": {name: str(path.relative_to(output_dir)) for name, path in paths.items() if name != "metadata"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def compute_transport_assessment(
    config: Mapping[str, Any],
    *,
    run_id: str,
    config_hash: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute the safe INX/HRDataset overlap without training a transport model."""

    settings = _settings(config)
    policies = settings.get("feature_policies", {})
    definitions = policies.get("definitions", {}) if isinstance(policies, Mapping) else {}
    primary_name = policies.get("primary_policy") if isinstance(policies, Mapping) else None
    primary = definitions.get(primary_name) if isinstance(definitions, Mapping) else None
    if not isinstance(primary, Mapping):
        raise ExternalEvidenceError("Canonical primary feature policy cannot be resolved for transport gate.")
    excluded = {str(value) for value in primary.get("excluded_features", [])}
    dataset_declarations = settings.get("datasets", {})
    inx_declaration = dataset_declarations.get("inx_primary", {}) if isinstance(dataset_declarations, Mapping) else {}
    inx_path = Path(str(inx_declaration.get("path", "")))
    if not inx_path.is_absolute():
        inx_path = PROJECT_ROOT / inx_path
    if not inx_path.is_file():
        raise ExternalEvidenceError(f"Canonical INX dataset is missing for transport gate: {inx_path}")
    inx = _read_csv_with_best_effort(inx_path)
    inx_features = {str(column) for column in inx.columns if column not in excluded}
    hr = load_external_dataset("hrdataset_v14")
    hr_features = set(build_feature_columns(hr, "department_free"))
    common = sorted(inx_features.intersection(hr_features))
    rows = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "feature": feature,
                "in_inx_canonical_primary": feature in inx_features,
                "in_hrdataset_department_free": feature in hr_features,
                "common_safe_feature": feature in common,
            }
            for feature in sorted(inx_features.union(hr_features))
        ]
    )
    feasible = len(common) >= MINIMUM_TRANSPORT_FEATURES and set(hr.labels) == {2, 3, 4}
    assessment = {
        "run_id": run_id,
        "config_hash": config_hash,
        "status": "protocol_required_before_transport" if feasible else "infeasible_or_too_limited",
        "locked_inx_model_transported": False,
        "n_common_safe_features": len(common),
        "common_safe_features": common,
        "minimum_feature_gate": MINIMUM_TRANSPORT_FEATURES,
        "hr_target_labels_verified": sorted(hr.labels),
        "interpretation": (
            "The schema overlap is too limited for a scientifically defensible locked-model transport result. "
            "HRDataset_v14 remains independently trained performance-target replication evidence."
            if not feasible
            else "The schema gate alone is satisfied, but no locked model was transported by this stage."
        ),
    }
    return rows, assessment


def _roles_table(config: Mapping[str, Any], *, run_id: str, config_hash: str) -> pd.DataFrame:
    settings = _settings(config)
    datasets = settings["datasets"]
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "dataset_key": spec.key,
                "canonical_dataset_key": spec.config_dataset_key,
                "dataset_name": spec.dataset_name,
                "target_kind": spec.target_kind,
                "task_type": spec.task_type,
                "task_comparison_group": get_task_schema(spec.task_type).comparison_group,
                "role": spec.role,
                "configured_allowed_claim": datasets[spec.config_dataset_key]["allowed_claim"],
                "locked_inx_model_transported": False,
                "comparable_to_primary_three_class_task": spec.task_type == ORDINAL_MULTICLASS_PERFORMANCE,
                "research_grade_decision_support_only": True,
                "autonomous_hr_decisions_allowed": False,
            }
            for spec in RUN_SPECS
        ]
    )


def _metric_applicability_table(*, run_id: str, config_hash: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in RUN_SPECS:
        schema = get_task_schema(spec.task_type)
        for metric in REPORT_METRICS:
            applicable = metric in KNOWN_METRICS and schema.is_metric_applicable(metric)
            rows.append(
                {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "dataset_key": spec.key,
                    "task_type": spec.task_type,
                    "task_comparison_group": schema.comparison_group,
                    "metric": metric,
                    "applicable": applicable,
                    "inapplicable_representation": "" if applicable else "N/A",
                    "applicability_note": schema.applicability_note,
                }
            )
    return pd.DataFrame(rows)


def _interpretation_markdown(
    *,
    run_id: str,
    config_hash: str,
    assessment: Mapping[str, Any],
) -> str:
    common = ", ".join(str(v) for v in assessment["common_safe_features"]) or "none"
    return "\n".join(
        [
            "# Canonical External Evidence Interpretation",
            "",
            f"Run ID: `{run_id}`  ",
            f"Config hash: `{config_hash}`",
            "",
            "## Separated evidence roles",
            "",
            "- HRDataset_v14 is independent external performance-target replication using independently trained, dataset-specific models.",
            "- IBM PerformanceRating is restricted-target robustness because only classes 3 and 4 are observed.",
            "- IBM attrition and Employee Turnover are related binary task-transfer evidence; they supply no validation evidence for the employee-performance model.",
            "- These task families are reported in separate tables and must not be ranked as directly equivalent outcomes.",
            "",
            "## Locked-model transport gate",
            "",
            f"Status: `{assessment['status']}`. Common safe features: {assessment['n_common_safe_features']} ({common}).",
            "No locked INX model was transported. The overlap result is a feasibility finding, not a transport performance estimate.",
            "",
            "## Claim limits",
            "",
            "- Research-grade decision support only; no autonomous hiring, firing, promotion, ranking, or discipline decisions.",
            "- Target mappings and class support are recorded beside each task and constrain interpretation.",
            "- Binary-task ordinal metrics are N/A, never zero.",
            "- Restricted-target ordinal-distance metrics are N/A and non-comparable.",
            "- Removing group variables does not establish fairness, and model attribution is not causality.",
            "",
        ]
    )


def _write_stage_manifest(output_dir: Path, *, run_id: str, config_hash: str) -> Path:
    path = output_dir / "stage_artifact_manifest.csv"
    rows = []
    for artifact in sorted(output_dir.rglob("*")):
        if not artifact.is_file() or artifact == path:
            continue
        rows.append(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "artifact_path": artifact.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(artifact),
                "size_bytes": artifact.stat().st_size,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_dir: str | Path,
    run_id: str,
    config_hash: str | None = None,
) -> dict[str, Path]:
    """Generate the complete canonical external-evidence stage."""

    if not str(run_id).strip():
        raise ExternalEvidenceError("run_id must be non-empty.")
    config = load_config(config_path)
    configured_run_specs(config)
    observed_hash = canonical_config_hash(config)
    if config_hash is not None and config_hash != observed_hash:
        raise ExternalEvidenceError(
            f"Supplied config_hash {config_hash!r} does not match canonical hash {observed_hash!r}."
        )
    config_hash = observed_hash
    output = Path(output_dir)
    if output.exists() and not output.is_dir():
        raise ExternalEvidenceError(f"External output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    task_paths: dict[str, dict[str, Path]] = {}
    task_summaries: dict[str, pd.DataFrame] = {}
    for spec in RUN_SPECS:
        paths = _run_dataset_task(
            spec,
            output_dir=output / spec.key,
            settings=_settings(config),
            run_id=run_id,
            config_hash=config_hash,
        )
        task_paths[spec.key] = paths
        task_summaries[spec.key] = pd.read_csv(paths["policy_summary"])

    outputs: dict[str, Path] = {
        "external_dataset_roles": output / "external_dataset_roles.csv",
        "metric_applicability": output / "external_metric_applicability.csv",
        "performance_target_replication": output / "performance_target_replication.csv",
        "restricted_target_robustness": output / "restricted_target_robustness.csv",
        "related_binary_task_transfer": output / "related_binary_task_transfer.csv",
        "transport_feature_overlap": output / "cross_dataset_transport" / "feature_overlap.csv",
        "transport_feasibility": output / "cross_dataset_transport" / "transport_feasibility.json",
        "transport_interpretation": output / "cross_dataset_transport" / "transport_feasibility.md",
        "interpretation": output / "external_evidence_interpretation.md",
        "metadata": output / "stage_metadata.json",
    }
    _roles_table(config, run_id=run_id, config_hash=config_hash).to_csv(outputs["external_dataset_roles"], index=False)
    _metric_applicability_table(run_id=run_id, config_hash=config_hash).to_csv(outputs["metric_applicability"], index=False)
    task_summaries["hrdataset_v14"].to_csv(outputs["performance_target_replication"], index=False)
    task_summaries["ibm_performance"].to_csv(outputs["restricted_target_robustness"], index=False)
    pd.concat(
        [task_summaries["ibm_attrition"], task_summaries["employee_turnover"]],
        ignore_index=True,
    ).to_csv(outputs["related_binary_task_transfer"], index=False)

    overlap, assessment = compute_transport_assessment(config, run_id=run_id, config_hash=config_hash)
    outputs["transport_feature_overlap"].parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(outputs["transport_feature_overlap"], index=False)
    outputs["transport_feasibility"].write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs["transport_interpretation"].write_text(
        "# INX-to-HRDataset Locked-Model Transport Feasibility\n\n"
        f"Run ID: `{run_id}`  \nConfig hash: `{config_hash}`\n\n"
        + _interpretation_markdown(run_id=run_id, config_hash=config_hash, assessment=assessment).split(
            "## Locked-model transport gate", 1
        )[1].split("## Claim limits", 1)[0].strip()
        + "\n",
        encoding="utf-8",
    )
    outputs["interpretation"].write_text(
        _interpretation_markdown(run_id=run_id, config_hash=config_hash, assessment=assessment),
        encoding="utf-8",
    )
    outputs["metadata"].write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "status": "completed",
                "started_and_completed_by": "manuscript_external_evidence",
                "completed_at": _utc_now(),
                "task_keys": [spec.key for spec in RUN_SPECS],
                "locked_inx_model_transported": False,
                "paid_api_calls": 0,
                "outputs": {name: path.relative_to(output).as_posix() for name, path in outputs.items() if name != "metadata"},
                "task_outputs": {
                    key: {name: path.relative_to(output).as_posix() for name, path in paths.items()}
                    for key, paths in task_paths.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["stage_artifact_manifest"] = _write_stage_manifest(output, run_id=run_id, config_hash=config_hash)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical versioned external evidence.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        run(
            args.config,
            output_dir=args.output_dir,
            run_id=args.run_id,
            config_hash=args.config_hash,
        )
    )
