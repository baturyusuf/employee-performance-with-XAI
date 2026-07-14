"""Task-bounded supplementary external evidence with exact nested OOF lineage.

This is the production V2-013 runner.  It intentionally excludes the
HRDataset_v14 core replication, locked-model transport, SHAP, fairness, and
cross-task score aggregation.  Each admitted supplementary task is fitted and
reported in its own stratum using one deterministic 10 outer x 5 inner fold
contract, primary-policy-only XGBoost selection, same-fold parameter reuse for
policy audits, exact persisted outer models, and a 5,000-draw paired sample
level OOF bootstrap.

The IBM 3/4 PerformanceRating task is a restricted binary robustness task.
IBM attrition and Employee Turnover are related binary task-transfer evidence.
None is direct validation of the primary three-class performance task and none
is locked-model transport.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.data.canonical_loader import CanonicalDataset, load_canonical_dataset
from src.data.external_adapters import (
    ExternalDataset,
    build_feature_columns,
    load_external_dataset,
    role_columns,
)
from src.experiments.final_evidence_common import predict_labels_from_proba
from src.experiments.manuscript_model_benchmark import select_candidate_index
from src.experiments.shared_folds import (
    SharedFoldArtifacts,
    generate_shared_folds,
    validate_consumer_fold_assignments,
)
from src.governance.manuscript_contract import canonical_config_hash, sha256_file
from src.models.canonical_models import aligned_predict_proba, build_model_pipeline
from src.models.evaluate import classification_metrics
from src.models.oof_bootstrap import (
    BootstrapProtocol,
    ComparisonSpec,
    compute_paired_oof_bootstrap,
)
from src.models.task_schema import (
    BINARY_ATTRITION_TRANSFER,
    BINARY_TURNOVER_TRANSFER,
    KNOWN_METRICS,
    RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS,
    get_task_schema,
)
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_CONFIG = Path("configs/manuscript_final.yaml")
MODEL_GRID_SIDE_INPUT_KEY = "model_search_space"
PRODUCTION_OUTER_SPLITS = 10
PRODUCTION_INNER_SPLITS = 5
PRODUCTION_CANDIDATE_COUNT = 8
PRODUCTION_BOOTSTRAP_RESAMPLES = 5000
MODEL_NAME = "xgboost"
MODEL_THREADS = 1
SELECTION_METRIC = "macro_f1"
SELECTION_TIE_BREAK_METRIC = "balanced_accuracy"
PRACTICAL_TIE_TOLERANCE = 0.001
SUPPLEMENTARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "average_precision",
    "nll_log_loss",
    "binary_brier",
    "ece_confidence",
)
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
COMMON_IDENTITY_FIELDS = (
    "run_id",
    "config_hash",
    "scientific_input_hash",
    "source_tree_hash",
    "git_commit",
    "scope_contract_hash",
    "dataset_key",
    "dataset_sha256",
    "canonical_content_sha256",
    "schema_mapping_sha256",
    "feature_policy_contract_sha256",
    "fold_contract_hash",
)


class SupplementaryExternalError(RuntimeError):
    """Raised when supplementary external evidence violates its contract."""


@dataclass(frozen=True)
class SupplementaryTaskSpec:
    key: str
    canonical_dataset_key: str
    dataset_name: str
    target_kind: str
    task_type: str
    role: str
    publication_stratum: str
    labels: tuple[int, ...]
    positive_label: int
    primary_policy: str
    audit_policies: tuple[str, ...]
    schema_side_input_key: str
    claim_boundary: str

    @property
    def policies(self) -> tuple[str, ...]:
        return (self.primary_policy, *self.audit_policies)


TASK_SPECS: Mapping[str, SupplementaryTaskSpec] = MappingProxyType(
    {
        "ibm_performance": SupplementaryTaskSpec(
            key="ibm_performance",
            canonical_dataset_key="ibm_hr_analytics",
            dataset_name="ibm_hr_analytics",
            target_kind="primary",
            task_type=RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS,
            role="restricted-target performance robustness",
            publication_stratum="ibm_restricted_target_performance_robustness",
            labels=(3, 4),
            positive_label=4,
            primary_policy="department_job_role_free",
            audit_policies=("department_free", "department_including"),
            schema_side_input_key="external_ibm_hr_analytics_schema_mapping",
            claim_boundary=(
                "Supplementary restricted 3/4-target robustness only; not direct three-class "
                "employee-performance external validation and not transport."
            ),
        ),
        "ibm_attrition": SupplementaryTaskSpec(
            key="ibm_attrition",
            canonical_dataset_key="ibm_hr_analytics_attrition",
            dataset_name="ibm_hr_analytics",
            target_kind="attrition",
            task_type=BINARY_ATTRITION_TRANSFER,
            role="related HR attrition task transfer",
            publication_stratum="ibm_attrition_related_task_transfer",
            labels=(0, 1),
            positive_label=1,
            primary_policy="department_job_role_free",
            audit_policies=("department_free", "department_including"),
            schema_side_input_key="external_ibm_hr_analytics_schema_mapping",
            claim_boundary=(
                "Supplementary related attrition-task transfer only; not employee-performance "
                "validation and not transport."
            ),
        ),
        "employee_turnover": SupplementaryTaskSpec(
            key="employee_turnover",
            canonical_dataset_key="employee_turnover",
            dataset_name="employee_turnover",
            target_kind="primary",
            task_type=BINARY_TURNOVER_TRANSFER,
            role="related HR turnover task transfer",
            publication_stratum="employee_turnover_related_task_transfer",
            labels=(0, 1),
            positive_label=1,
            primary_policy="without_last_evaluation",
            audit_policies=("with_last_evaluation",),
            schema_side_input_key="external_employee_turnover_schema_mapping",
            claim_boundary=(
                "Supplementary related turnover-task transfer only; not employee-performance "
                "validation and not transport."
            ),
        ),
    }
)


@dataclass(frozen=True)
class SupplementaryExternalTestOnlyOverrides:
    """Visible noncanonical reductions for unit tests and bounded diagnostics."""

    candidate_indices: tuple[int, ...] = (0,)
    bootstrap_resamples: int = 50
    task_keys: tuple[str, ...] = ("ibm_performance",)

    def __post_init__(self) -> None:
        if (
            not self.candidate_indices
            or tuple(sorted(set(self.candidate_indices))) != self.candidate_indices
            or any(not isinstance(value, int) or isinstance(value, bool) for value in self.candidate_indices)
            or min(self.candidate_indices) < 0
            or max(self.candidate_indices) >= PRODUCTION_CANDIDATE_COUNT
        ):
            raise SupplementaryExternalError(
                "Test-only candidate indices must be sorted, unique, and within the production pool."
            )
        if not 2 <= self.bootstrap_resamples < PRODUCTION_BOOTSTRAP_RESAMPLES:
            raise SupplementaryExternalError(
                "Test-only bootstrap_resamples must lie in [2, 4999]."
            )
        if (
            not self.task_keys
            or tuple(dict.fromkeys(self.task_keys)) != self.task_keys
            or not set(self.task_keys).issubset(TASK_SPECS)
        ):
            raise SupplementaryExternalError("Test-only task keys must be a unique admitted subset.")


@dataclass(frozen=True)
class TaskEvidence:
    spec: SupplementaryTaskSpec
    folds: SharedFoldArtifacts
    target_mapping: pd.DataFrame
    target_support: pd.DataFrame
    feature_policies: pd.DataFrame
    candidate_fit_receipts: pd.DataFrame
    candidate_search_results: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    outer_model_receipts: pd.DataFrame
    oof_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    fold_descriptive_summary: pd.DataFrame
    metric_intervals: pd.DataFrame
    paired_policy_differences: pd.DataFrame
    model_bytes: Mapping[tuple[str, int], bytes]
    bootstrap_metadata: Mapping[str, Any]
    task_metadata: Mapping[str, Any]
    canonical_eligible: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_identity(name: str, value: Any, *, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SupplementaryExternalError(f"{name} must be a non-empty string.")
    observed = value.strip()
    if sha256 and (
        len(observed) != 64
        or any(character not in "0123456789abcdef" for character in observed)
    ):
        raise SupplementaryExternalError(f"{name} must be a lowercase SHA-256 digest.")
    return observed


def _serialize_model(model: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=0, protocol=4)
    payload = buffer.getvalue()
    if not payload:
        raise SupplementaryExternalError("A fitted model serialized to empty bytes.")
    return payload


def _hash_sample_ids(values: Sequence[int]) -> str:
    return _sha256_json(sorted(int(value) for value in values))


def _feature_order_sha256(columns: Sequence[str]) -> str:
    return _sha256_json([str(column) for column in columns])


def _canonical_csv_content_sha256(path: Path) -> tuple[str, int, int]:
    """Hash parsed CSV cells independently of byte quoting and line endings."""

    digest = hashlib.sha256()
    row_count = 0
    width: int | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if width is None:
                width = len(row)
            if len(row) != width:
                raise SupplementaryExternalError(
                    f"CSV row width drift in {path} at row {row_count + 1}."
                )
            digest.update(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            digest.update(b"\n")
            row_count += 1
    if width is None or row_count <= 1:
        raise SupplementaryExternalError(f"CSV content is empty: {path}")
    return digest.hexdigest(), row_count - 1, int(width)


def _safe_relative_path(value: Any, *, field: str) -> str:
    raw = str(value)
    if not raw or raw != raw.strip() or "\\" in raw:
        raise SupplementaryExternalError(f"{field} is not a portable repository-relative path.")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise SupplementaryExternalError(f"{field} is not a contained relative path: {raw!r}.")
    candidate = (PROJECT_ROOT / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise SupplementaryExternalError(f"{field} escapes the repository root.") from exc
    return relative.as_posix()


def _validated_side_input(
    expected: Mapping[str, Any], key: str, configured_path: str | Path
) -> tuple[Path, str]:
    record = expected.get(key)
    if not isinstance(record, Mapping):
        raise SupplementaryExternalError(f"Missing scoped side-input receipt {key!r}.")
    reference = _safe_relative_path(record.get("path"), field=f"{key}.path")
    candidate = Path(configured_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    expected_path = (PROJECT_ROOT / reference).resolve()
    if candidate != expected_path or not candidate.is_file():
        raise SupplementaryExternalError(f"Configured side input {key!r} does not match its receipt.")
    digest = sha256_file(candidate)
    if digest != record.get("sha256") or candidate.stat().st_size != record.get("size_bytes"):
        raise SupplementaryExternalError(f"Scientific side input {key!r} changed after manifest creation.")
    return candidate, digest


def _validate_dataset_receipt(
    observed: Mapping[str, Any], expected_receipts: Mapping[str, Any], dataset_key: str
) -> Mapping[str, Any]:
    expected = expected_receipts.get(dataset_key)
    if not isinstance(expected, Mapping):
        raise SupplementaryExternalError(f"Missing actual-input receipt for {dataset_key!r}.")
    fields = (
        "dataset_key",
        "physical_dataset_id",
        "actual_path",
        "actual_sha256",
        "size_bytes",
        "row_count",
        "column_count",
        "schema_status",
        "target_column",
    )
    differences = {
        field: {"expected": expected.get(field), "observed": observed.get(field)}
        for field in fields
        if expected.get(field) != observed.get(field)
    }
    if differences:
        raise SupplementaryExternalError(
            f"Actual-input receipt drift for {dataset_key!r}: "
            + json.dumps(differences, sort_keys=True, ensure_ascii=True)
        )
    path = _safe_relative_path(observed.get("actual_path"), field=f"{dataset_key}.actual_path")
    raw_path = (PROJECT_ROOT / path).resolve()
    if not raw_path.is_file() or sha256_file(raw_path) != observed.get("actual_sha256"):
        raise SupplementaryExternalError(f"Consumed bytes changed for {dataset_key!r}.")
    return dict(observed)


def _settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = config.get("manuscript_final", config)
    if not isinstance(settings, Mapping):
        raise SupplementaryExternalError("Canonical manuscript settings are missing.")
    return settings


def validate_protocol(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail closed on any drift in the accepted V2-013 contract."""

    settings = _settings(config)
    protocol = settings.get("supplementary_external_evidence")
    if not isinstance(protocol, Mapping):
        raise SupplementaryExternalError("supplementary_external_evidence is required.")
    expected_scalars = {
        "scope": "supplementary_task_bounded_robustness_only",
        "publication_role": "supplementary_only",
        "direct_external_validation_of_primary_allowed": False,
        "locked_model_transport_performed": False,
        "transport_claim_allowed": False,
        "cross_task_comparison_allowed": False,
    }
    for field, expected in expected_scalars.items():
        if protocol.get(field) != expected:
            raise SupplementaryExternalError(
                f"Supplementary external protocol drift at {field}: expected {expected!r}."
            )
    model = protocol.get("model_protocol")
    cv = protocol.get("cv")
    uncertainty = protocol.get("uncertainty")
    outputs = protocol.get("outputs")
    if not all(isinstance(value, Mapping) for value in (model, cv, uncertainty, outputs)):
        raise SupplementaryExternalError("Supplementary model/CV/uncertainty/output contracts are required.")
    expected_model = {
        "model": MODEL_NAME,
        "candidate_count": PRODUCTION_CANDIDATE_COUNT,
        "selection_primary_metric": SELECTION_METRIC,
        "selection_tie_break_metric": SELECTION_TIE_BREAK_METRIC,
        "ordinal_tie_break_metrics_allowed": False,
        "primary_practical_tie_tolerance": PRACTICAL_TIE_TOLERANCE,
        "outer_test_used_for_selection": False,
        "candidate_failure_policy": "fail_entire_stage",
        "estimator_threads": MODEL_THREADS,
    }
    for field, expected in expected_model.items():
        if model.get(field) != expected:
            raise SupplementaryExternalError(f"Supplementary model protocol drift at {field}.")
    if (
        cv.get("outer_splits") != PRODUCTION_OUTER_SPLITS
        or cv.get("inner_splits") != PRODUCTION_INNER_SPLITS
        or cv.get("silent_fold_reduction_allowed") is not False
        or cv.get("same_outer_folds_across_policies") is not True
    ):
        raise SupplementaryExternalError("Supplementary CV must remain exact 10x5 without reduction.")
    if (
        uncertainty.get("n_resamples") != PRODUCTION_BOOTSTRAP_RESAMPLES
        or uncertainty.get("method") != "paired_stratified_percentile_bootstrap"
        or uncertainty.get("stratify_by") != ["outer_fold", "y_true"]
        or uncertainty.get("confidence_level") != 0.95
    ):
        raise SupplementaryExternalError("Supplementary uncertainty must remain the 5,000-draw OOF contract.")
    if (
        outputs.get("separate_task_strata_required") is not True
        or outputs.get("combined_cross_task_score_table_allowed") is not False
        or outputs.get("explicit_metric_applicability_required") is not True
        or outputs.get("inapplicable_metrics_representation") != "N/A"
        or outputs.get("exact_model_replay_required") is not True
        or outputs.get("closed_world_manifest_required") is not True
        or outputs.get("atomic_publication_required") is not True
    ):
        raise SupplementaryExternalError("Supplementary output/claim contract drifted.")
    task_config = protocol.get("tasks")
    metric_config = protocol.get("metrics")
    dataset_config = settings.get("datasets")
    if not isinstance(task_config, Mapping) or set(task_config) != set(TASK_SPECS):
        raise SupplementaryExternalError("The three supplementary tasks must be exact and exhaustive.")
    if not isinstance(metric_config, Mapping) or not isinstance(dataset_config, Mapping):
        raise SupplementaryExternalError("Task-specific supplementary metrics are required.")
    for key, spec in TASK_SPECS.items():
        entry = task_config.get(key)
        if not isinstance(entry, Mapping):
            raise SupplementaryExternalError(f"Missing supplementary task contract {key!r}.")
        expected_task = {
            "canonical_dataset_key": spec.canonical_dataset_key,
            "task_type": spec.task_type,
            "publication_stratum": spec.publication_stratum,
            "primary_policy": spec.primary_policy,
            "audit_policies": list(spec.audit_policies),
            "positive_label": spec.positive_label,
            "claim_boundary": spec.claim_boundary,
        }
        if any(entry.get(field) != value for field, value in expected_task.items()):
            raise SupplementaryExternalError(f"Supplementary task contract drift for {key!r}.")
        dataset_entry = dataset_config.get(spec.canonical_dataset_key)
        expected_role = (
            "restricted_target_performance_robustness"
            if spec.key == "ibm_performance"
            else "related_task_transfer"
        )
        if (
            not isinstance(dataset_entry, Mapping)
            or dataset_entry.get("task_type") != spec.task_type
            or dataset_entry.get("role") != expected_role
        ):
            raise SupplementaryExternalError(f"Canonical dataset role drift for {key!r}.")
        allowed_claim = str(dataset_entry.get("allowed_claim", "")).lower()
        required_claim_text = (
            "not direct three-class external validation"
            if spec.key == "ibm_performance"
            else "not employee-performance validation"
        )
        if required_claim_text not in allowed_claim:
            raise SupplementaryExternalError(f"Canonical claim boundary drift for {key!r}.")
        metrics = metric_config.get(spec.task_type)
        if metrics != list(SUPPLEMENTARY_METRICS):
            raise SupplementaryExternalError(f"Invalid metric list for {spec.task_type!r}.")
        schema = get_task_schema(spec.task_type)
        for metric in metrics:
            if metric not in KNOWN_METRICS or not schema.is_metric_applicable(str(metric)):
                raise SupplementaryExternalError(
                    f"Metric {metric!r} is inapplicable to {spec.task_type!r}."
                )
        if any(schema.is_metric_applicable(metric) for metric in (
            "quadratic_weighted_kappa", "ordinal_mae", "adjacent_accuracy", "severe_error_rate"
        )):
            raise SupplementaryExternalError(
                f"Ordinal metric applicability leaked into supplementary task {spec.task_type!r}."
            )
    return protocol


def _model_grid(
    settings: Mapping[str, Any], expected_side_inputs: Mapping[str, Any]
) -> tuple[Mapping[str, Any], str]:
    configured_path = settings.get("model", {}).get("search_space_config")
    if not isinstance(configured_path, str):
        raise SupplementaryExternalError("Canonical model search-space path is missing.")
    path, digest = _validated_side_input(
        expected_side_inputs, MODEL_GRID_SIDE_INPUT_KEY, configured_path
    )
    payload = load_config(path)
    benchmark = payload.get("model_benchmark") if isinstance(payload, Mapping) else None
    models = benchmark.get("models") if isinstance(benchmark, Mapping) else None
    definition = models.get(MODEL_NAME) if isinstance(models, Mapping) else None
    if not isinstance(definition, Mapping):
        raise SupplementaryExternalError("The canonical XGBoost grid is missing.")
    candidates = definition.get("candidates")
    fixed = definition.get("fixed_params")
    if (
        not isinstance(candidates, list)
        or len(candidates) != PRODUCTION_CANDIDATE_COUNT
        or not all(isinstance(value, Mapping) for value in candidates)
        or not isinstance(fixed, Mapping)
        or fixed.get("n_jobs") != MODEL_THREADS
    ):
        raise SupplementaryExternalError("The restrained eight-candidate XGBoost grid drifted.")
    return definition, digest


def _fit_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    fixed_parameters: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
    random_state: int,
    forbidden_features: Sequence[str],
    context: str,
) -> Any:
    pipeline = build_model_pipeline(
        MODEL_NAME,
        X,
        fixed_parameters=fixed_parameters,
        candidate_parameters=candidate_parameters,
        random_state=random_state,
        forbidden_features=forbidden_features,
    )
    try:
        with threadpool_limits(limits=MODEL_THREADS):
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                pipeline.fit(X, y)
    except Exception as exc:
        raise SupplementaryExternalError(
            f"{context} failed: {type(exc).__name__}: {exc}"
        ) from exc
    return pipeline


def _target_mapping(dataset: ExternalDataset, identity: Mapping[str, Any]) -> pd.DataFrame:
    raw = dataset.raw[dataset.target_raw_column].astype("string").fillna("__MISSING__")
    mapped = dataset.canonical[dataset.target_column].astype(int)
    frame = (
        pd.DataFrame({"raw_target_value": raw, "mapped_label": mapped})
        .value_counts(sort=False)
        .reset_index(name="n_rows")
        .sort_values(["mapped_label", "raw_target_value"], kind="stable")
        .reset_index(drop=True)
    )
    frame["target_raw_column"] = dataset.target_raw_column
    frame["target_canonical_column"] = dataset.target_column
    frame["mapping_complete"] = not bool(dataset.unmapped_target_values)
    return _add_identity(frame, identity)


def _target_support(
    dataset: ExternalDataset, spec: SupplementaryTaskSpec, identity: Mapping[str, Any]
) -> pd.DataFrame:
    counts = dataset.canonical[dataset.target_column].astype(int).value_counts().sort_index()
    if tuple(int(value) for value in counts.index) != spec.labels:
        raise SupplementaryExternalError(
            f"Observed labels for {spec.key!r} differ from {list(spec.labels)}."
        )
    if int(counts.min()) < PRODUCTION_OUTER_SPLITS:
        raise SupplementaryExternalError(
            f"{spec.key!r} cannot support exact 10-fold OOF; minimum class support={int(counts.min())}."
        )
    rows = [
        {
            "label": int(label),
            "n_rows": int(count),
            "proportion": float(count / counts.sum()),
            "positive_class": int(label) == spec.positive_label,
            "requested_outer_splits": PRODUCTION_OUTER_SPLITS,
            "effective_outer_splits": PRODUCTION_OUTER_SPLITS,
            "silent_fold_reduction": False,
            "mapping_verified": True,
        }
        for label, count in counts.items()
    ]
    return _add_identity(pd.DataFrame(rows), identity)


def _add_identity(frame: pd.DataFrame, identity: Mapping[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    for position, field in enumerate(reversed(tuple(identity))):
        result.insert(0, field, identity[field])
    return result


def _fold_membership(
    folds: SharedFoldArtifacts, outer_fold: int
) -> tuple[list[int], list[int], pd.DataFrame]:
    outer = folds.outer_assignments
    test_ids = outer.loc[outer["outer_fold"].astype(int) == outer_fold, "sample_index"].astype(int).tolist()
    train_ids = outer.loc[outer["outer_fold"].astype(int) != outer_fold, "sample_index"].astype(int).tolist()
    inner = folds.inner_assignments[
        folds.inner_assignments["outer_fold"].astype(int) == outer_fold
    ].copy()
    if set(inner["sample_index"].astype(int)) != set(train_ids):
        raise SupplementaryExternalError(f"Inner folds do not equal outer training fold {outer_fold}.")
    return train_ids, test_ids, inner


def _metric_value(
    metric: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    *,
    labels: Sequence[int],
    task_type: str,
) -> float:
    value = classification_metrics(
        y_true,
        y_pred,
        probabilities,
        list(labels),
        task_type=task_type,
    ).get(metric)
    if value is None or not math.isfinite(float(value)):
        raise SupplementaryExternalError(f"Selection metric {metric!r} is invalid.")
    return float(value)


def _descriptive_fold_summary(fold_metrics: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, group in fold_metrics.groupby("policy", sort=False):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                raise SupplementaryExternalError(
                    f"Applicable fold metric {metric!r} is missing for policy {policy!r}."
                )
            rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "fold_mean": float(values.mean()),
                    "fold_std": float(values.std(ddof=1)),
                    "fold_min": float(values.min()),
                    "fold_max": float(values.max()),
                    "n_outer_folds": int(len(values)),
                    "uncertainty_role": "descriptive_variability_only_no_population_ci",
                }
            )
    return pd.DataFrame(rows)


def evaluate_task(
    dataset: ExternalDataset,
    *,
    spec: SupplementaryTaskSpec,
    receipt: Mapping[str, Any],
    schema_mapping_sha256: str,
    canonical_content_sha256: str,
    model_definition: Mapping[str, Any],
    metric_names: Sequence[str],
    identity_base: Mapping[str, str],
    outer_seed: int,
    inner_seed: int,
    model_seed: int,
    bootstrap_seed: int,
    test_only_overrides: SupplementaryExternalTestOnlyOverrides | None = None,
) -> TaskEvidence:
    """Compute one isolated supplementary task and validate every OOF model link."""

    if dataset.task_type != spec.task_type:
        raise SupplementaryExternalError(
            f"Adapter task drift for {spec.key!r}: {dataset.task_type!r}."
        )
    if len(dataset.canonical) != int(receipt.get("row_count", -1)):
        raise SupplementaryExternalError(f"Adapted row denominator drift for {spec.key!r}.")
    policies = spec.policies
    if set(policies).difference(dataset.config.feature_policy_variants):
        raise SupplementaryExternalError(f"Mapped feature policies are incomplete for {spec.key!r}.")
    policy_features = {policy: build_feature_columns(dataset, policy) for policy in policies}
    forbidden_roles = set(role_columns(dataset, "id")) | set(role_columns(dataset, "leakage")) | set(
        role_columns(dataset, "sensitive")
    )
    for policy, features in policy_features.items():
        forbidden = sorted(forbidden_roles.intersection(features))
        if forbidden:
            raise SupplementaryExternalError(
                f"Forbidden features entered {spec.key}/{policy}: {forbidden}."
            )
    feature_policy_hash = _sha256_json(
        {
            "task_key": spec.key,
            "primary_policy": spec.primary_policy,
            "audit_policies": list(spec.audit_policies),
            "exact_features": policy_features,
            "schema_mapping_sha256": schema_mapping_sha256,
        }
    )
    dataset_hash = _require_identity("dataset_sha256", receipt.get("actual_sha256"), sha256=True)
    fold_source = dataset.canonical[["ExternalSampleId", dataset.target_column]].copy()
    folds = generate_shared_folds(
        fold_source,
        target_column=dataset.target_column,
        id_column="ExternalSampleId",
        run_id=identity_base["run_id"],
        config_hash=identity_base["config_hash"],
        scientific_input_hash=identity_base["scientific_input_hash"],
        dataset_key=spec.canonical_dataset_key,
        dataset_sha256=dataset_hash,
        outer_splits=PRODUCTION_OUTER_SPLITS,
        inner_splits=PRODUCTION_INNER_SPLITS,
        seed=outer_seed,
        inner_seed=inner_seed,
    )
    fold_hash = str(folds.contract["fold_contract_hash"])
    identity: dict[str, Any] = {
        **identity_base,
        "dataset_key": spec.canonical_dataset_key,
        "dataset_sha256": dataset_hash,
        "canonical_content_sha256": canonical_content_sha256,
        "canonical_content_hash_algorithm": "sha256_utf8_json_rows_csv_utf8_sig_v1",
        "schema_mapping_sha256": schema_mapping_sha256,
        "feature_policy_contract_sha256": feature_policy_hash,
        "fold_contract_hash": fold_hash,
    }
    target_mapping = _target_mapping(dataset, identity)
    target_support = _target_support(dataset, spec, identity)
    y = dataset.canonical[dataset.target_column].astype(int)
    fixed = dict(model_definition["fixed_params"])
    candidates = [dict(value) for value in model_definition["candidates"]]
    candidate_indices = (
        tuple(range(PRODUCTION_CANDIDATE_COUNT))
        if test_only_overrides is None
        else test_only_overrides.candidate_indices
    )
    feature_rows = [
        {
            **identity,
            "task_key": spec.key,
            "task_type": spec.task_type,
            "role": spec.role,
            "publication_stratum": spec.publication_stratum,
            "policy": policy,
            "policy_role": "primary" if policy == spec.primary_policy else "audit_sensitivity",
            "n_features": len(features),
            "feature_columns": ";".join(features),
            "feature_order_sha256": _feature_order_sha256(features),
            "excluded_id_columns": ";".join(sorted(role_columns(dataset, "id"))),
            "excluded_leakage_columns": ";".join(sorted(role_columns(dataset, "leakage"))),
            "excluded_sensitive_columns": ";".join(sorted(role_columns(dataset, "sensitive"))),
            "forbidden_feature_count": 0,
        }
        for policy, features in policy_features.items()
    ]
    candidate_fit_rows: list[dict[str, Any]] = []
    candidate_search_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    outer_receipt_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    model_bytes: dict[tuple[str, int], bytes] = {}
    fitted_models: dict[tuple[str, int], Any] = {}
    labels = list(spec.labels)
    primary_X = dataset.canonical.loc[:, policy_features[spec.primary_policy]].copy()
    for outer_fold in range(1, PRODUCTION_OUTER_SPLITS + 1):
        train_ids, test_ids, inner = _fold_membership(folds, outer_fold)
        candidate_primary_means: list[float] = []
        candidate_tie_means: list[float] = []
        outer_candidate_rows: list[dict[str, Any]] = []
        for candidate_index in candidate_indices:
            candidate = candidates[candidate_index]
            primary_scores: list[float] = []
            tie_scores: list[float] = []
            fit_hashes: list[str] = []
            for inner_fold in range(1, PRODUCTION_INNER_SPLITS + 1):
                validation_ids = inner.loc[
                    inner["inner_fold"].astype(int) == inner_fold, "sample_index"
                ].astype(int).tolist()
                development_ids = sorted(set(train_ids).difference(validation_ids))
                if not development_ids or not validation_ids:
                    raise SupplementaryExternalError(
                        f"Empty nested partition for {spec.key}, outer={outer_fold}, inner={inner_fold}."
                    )
                pipeline = _fit_pipeline(
                    primary_X.loc[development_ids],
                    y.loc[development_ids],
                    fixed_parameters=fixed,
                    candidate_parameters=candidate,
                    random_state=model_seed,
                    forbidden_features=tuple(forbidden_roles),
                    context=(
                        f"{spec.key} candidate fit outer={outer_fold}, candidate={candidate_index}, "
                        f"inner={inner_fold}"
                    ),
                )
                probabilities = aligned_predict_proba(
                    pipeline, primary_X.loc[validation_ids], labels=labels
                )
                predicted = predict_labels_from_proba(probabilities, labels)
                primary_scores.append(
                    _metric_value(
                        SELECTION_METRIC,
                        y.loc[validation_ids],
                        predicted,
                        probabilities,
                        labels=labels,
                        task_type=spec.task_type,
                    )
                )
                tie_scores.append(
                    _metric_value(
                        SELECTION_TIE_BREAK_METRIC,
                        y.loc[validation_ids],
                        predicted,
                        probabilities,
                        labels=labels,
                        task_type=spec.task_type,
                    )
                )
                serialized = _serialize_model(pipeline)
                model_sha = hashlib.sha256(serialized).hexdigest()
                fit_contract = {
                    "task_key": spec.key,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "candidate_index": candidate_index,
                    "parameters": candidate,
                    "development_sample_set_sha256": _hash_sample_ids(development_ids),
                    "validation_sample_set_sha256": _hash_sample_ids(validation_ids),
                    "feature_order_sha256": _feature_order_sha256(
                        policy_features[spec.primary_policy]
                    ),
                    "model_sha256": model_sha,
                }
                fit_hash = _sha256_json(fit_contract)
                fit_hashes.append(fit_hash)
                candidate_fit_rows.append(
                    {
                        **identity,
                        "task_key": spec.key,
                        "task_type": spec.task_type,
                        "publication_stratum": spec.publication_stratum,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "candidate_index": candidate_index,
                        "candidate_parameters_json": json.dumps(candidate, sort_keys=True),
                        "selection_metric": SELECTION_METRIC,
                        "selection_metric_value": primary_scores[-1],
                        "selection_tie_break_metric": SELECTION_TIE_BREAK_METRIC,
                        "selection_tie_break_value": tie_scores[-1],
                        "n_development": len(development_ids),
                        "n_validation": len(validation_ids),
                        "development_sample_set_sha256": fit_contract[
                            "development_sample_set_sha256"
                        ],
                        "validation_sample_set_sha256": fit_contract[
                            "validation_sample_set_sha256"
                        ],
                        "feature_order_sha256": fit_contract["feature_order_sha256"],
                        "model_sha256": model_sha,
                        "model_fit_contract_sha256": fit_hash,
                        "outer_test_used": False,
                    }
                )
            primary_mean = float(np.mean(primary_scores))
            tie_mean = float(np.mean(tie_scores))
            candidate_primary_means.append(primary_mean)
            candidate_tie_means.append(tie_mean)
            outer_candidate_rows.append(
                {
                    **identity,
                    "task_key": spec.key,
                    "task_type": spec.task_type,
                    "publication_stratum": spec.publication_stratum,
                    "outer_fold": outer_fold,
                    "candidate_index": candidate_index,
                    "candidate_parameters_json": json.dumps(candidate, sort_keys=True),
                    "selection_metric": SELECTION_METRIC,
                    "inner_fold_scores_json": json.dumps(primary_scores),
                    "inner_mean": primary_mean,
                    "inner_std": float(np.std(primary_scores, ddof=1)),
                    "selection_tie_break_metric": SELECTION_TIE_BREAK_METRIC,
                    "tie_break_inner_fold_scores_json": json.dumps(tie_scores),
                    "tie_break_inner_mean": tie_mean,
                    "tie_break_inner_std": float(np.std(tie_scores, ddof=1)),
                    "candidate_fit_set_sha256": _sha256_json(fit_hashes),
                    "n_inner_folds": PRODUCTION_INNER_SPLITS,
                    "outer_test_used_for_selection": False,
                }
            )
        selected_local = select_candidate_index(
            candidate_primary_means,
            candidate_tie_means,
            better_direction="higher",
            practical_tie_tolerance=PRACTICAL_TIE_TOLERANCE,
        )
        selected_index = candidate_indices[selected_local]
        selected_candidate = candidates[selected_index]
        for local_index, row in enumerate(outer_candidate_rows):
            row["selected_by_protocol"] = local_index == selected_local
            candidate_search_rows.append(row)
        selected_rows.append(
            {
                **identity,
                "task_key": spec.key,
                "task_type": spec.task_type,
                "publication_stratum": spec.publication_stratum,
                "outer_fold": outer_fold,
                "selection_source_policy": spec.primary_policy,
                "selected_candidate_index": selected_index,
                "selected_candidate_parameters_json": json.dumps(
                    selected_candidate, sort_keys=True
                ),
                "fixed_parameters_json": json.dumps(fixed, sort_keys=True),
                "selection_metric": SELECTION_METRIC,
                "selected_inner_mean": candidate_primary_means[selected_local],
                "selection_tie_break_metric": SELECTION_TIE_BREAK_METRIC,
                "selected_tie_break_inner_mean": candidate_tie_means[selected_local],
                "selected_candidate_fit_set_sha256": outer_candidate_rows[selected_local][
                    "candidate_fit_set_sha256"
                ],
                "primary_practical_tie_tolerance": PRACTICAL_TIE_TOLERANCE,
                "outer_test_used_for_selection": False,
                "policy_parameter_reuse": "same_selected_candidate_for_all_policies_in_outer_fold",
            }
        )
        for policy in policies:
            features = policy_features[policy]
            X = dataset.canonical.loc[:, features].copy()
            pipeline = _fit_pipeline(
                X.loc[train_ids],
                y.loc[train_ids],
                fixed_parameters=fixed,
                candidate_parameters=selected_candidate,
                random_state=model_seed,
                forbidden_features=tuple(forbidden_roles),
                context=f"{spec.key} outer model policy={policy}, outer={outer_fold}",
            )
            probabilities = aligned_predict_proba(pipeline, X.loc[test_ids], labels=labels)
            predicted = predict_labels_from_proba(probabilities, labels)
            if (
                not np.isfinite(probabilities).all()
                or (probabilities < 0.0).any()
                or (probabilities > 1.0).any()
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
            ):
                raise SupplementaryExternalError(
                    f"Invalid probabilities for {spec.key}/{policy}/fold {outer_fold}."
                )
            serialized = _serialize_model(pipeline)
            model_sha = hashlib.sha256(serialized).hexdigest()
            restored = joblib.load(io.BytesIO(serialized))
            replay_probabilities = aligned_predict_proba(restored, X.loc[test_ids], labels=labels)
            replay_predictions = predict_labels_from_proba(replay_probabilities, labels)
            replay_error = float(np.max(np.abs(replay_probabilities - probabilities)))
            if replay_error > 1e-12 or not np.array_equal(replay_predictions, predicted):
                raise SupplementaryExternalError(
                    f"Serialized model replay failed for {spec.key}/{policy}/fold {outer_fold}."
                )
            model_bytes[(policy, outer_fold)] = serialized
            fitted_models[(policy, outer_fold)] = pipeline
            feature_hash = _feature_order_sha256(features)
            fit_contract = {
                "task_key": spec.key,
                "policy": policy,
                "outer_fold": outer_fold,
                "selected_candidate_index": selected_index,
                "selected_candidate_parameters": selected_candidate,
                "train_sample_set_sha256": _hash_sample_ids(train_ids),
                "test_sample_set_sha256": _hash_sample_ids(test_ids),
                "feature_order_sha256": feature_hash,
                "model_sha256": model_sha,
            }
            model_fit_hash = _sha256_json(fit_contract)
            outer_receipt_rows.append(
                {
                    **identity,
                    "task_key": spec.key,
                    "task_type": spec.task_type,
                    "role": spec.role,
                    "publication_stratum": spec.publication_stratum,
                    "policy": policy,
                    "policy_role": "primary" if policy == spec.primary_policy else "audit_sensitivity",
                    "outer_fold": outer_fold,
                    "n_train": len(train_ids),
                    "n_test": len(test_ids),
                    "selected_candidate_index": selected_index,
                    "selected_candidate_parameters_json": json.dumps(
                        selected_candidate, sort_keys=True
                    ),
                    "parameter_source": "primary_policy_nested_selection_same_outer_fold",
                    "feature_order_sha256": feature_hash,
                    "train_sample_set_sha256": fit_contract["train_sample_set_sha256"],
                    "test_sample_set_sha256": fit_contract["test_sample_set_sha256"],
                    "model_sha256": model_sha,
                    "model_fit_contract_sha256": model_fit_hash,
                    "model_artifact_path": f"models/{policy}/outer_fold_{outer_fold:02d}.joblib",
                    "exact_serialized_model_replay": True,
                    "max_probability_replay_error": replay_error,
                    "outer_test_used_for_selection": False,
                }
            )
            metrics = classification_metrics(
                y.loc[test_ids],
                predicted,
                probabilities,
                labels,
                task_type=spec.task_type,
            )
            fold_metric_rows.append(
                {
                    **identity,
                    "task_key": spec.key,
                    "task_type": spec.task_type,
                    "role": spec.role,
                    "publication_stratum": spec.publication_stratum,
                    "policy": policy,
                    "outer_fold": outer_fold,
                    "model": MODEL_NAME,
                    "model_sha256": model_sha,
                    "model_fit_contract_sha256": model_fit_hash,
                    "model_artifact_path": f"models/{policy}/outer_fold_{outer_fold:02d}.joblib",
                    "selected_candidate_index": selected_index,
                    "n_test": len(test_ids),
                    **{name: metrics.get(name) for name in REPORT_METRICS},
                }
            )
            for position, sample_index in enumerate(test_ids):
                row = {
                    **identity,
                    "task_key": spec.key,
                    "task_type": spec.task_type,
                    "role": spec.role,
                    "publication_stratum": spec.publication_stratum,
                    "policy": policy,
                    "system_id": policy,
                    "sample_index": int(sample_index),
                    "outer_fold": outer_fold,
                    "y_true": int(y.loc[sample_index]),
                    "y_pred": int(predicted[position]),
                    "model": MODEL_NAME,
                    "model_sha256": model_sha,
                    "model_fit_contract_sha256": model_fit_hash,
                    "model_artifact_path": f"models/{policy}/outer_fold_{outer_fold:02d}.joblib",
                    "selected_candidate_index": selected_index,
                    "probability_source": "raw_exact_outer_fold_model",
                }
                for column, label in enumerate(labels):
                    row[f"prob_class_{label}"] = float(probabilities[position, column])
                prediction_rows.append(row)

    outer_receipts = pd.DataFrame(outer_receipt_rows)
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["policy", "sample_index"], kind="stable"
    ).reset_index(drop=True)
    fold_metrics = pd.DataFrame(fold_metric_rows).sort_values(
        ["policy", "outer_fold"], kind="stable"
    ).reset_index(drop=True)
    if len(outer_receipts) != len(policies) * PRODUCTION_OUTER_SPLITS:
        raise SupplementaryExternalError(f"Incomplete outer-model receipts for {spec.key!r}.")
    policy_model_hashes: dict[str, str] = {}
    for policy in policies:
        scoped = outer_receipts[outer_receipts["policy"] == policy].sort_values("outer_fold")
        if len(scoped) != PRODUCTION_OUTER_SPLITS or scoped["outer_fold"].duplicated().any():
            raise SupplementaryExternalError(f"Outer-model identity is incomplete for {spec.key}/{policy}.")
        policy_model_hashes[policy] = _sha256_json(
            scoped[["outer_fold", "model_sha256"]].to_dict(orient="records")
        )
    task_model_set_sha = _sha256_json(policy_model_hashes)
    outer_receipts["policy_model_set_sha256"] = outer_receipts["policy"].map(policy_model_hashes)
    outer_receipts["task_model_set_sha256"] = task_model_set_sha
    predictions["policy_model_set_sha256"] = predictions["policy"].map(policy_model_hashes)
    predictions["task_model_set_sha256"] = task_model_set_sha
    fold_metrics["policy_model_set_sha256"] = fold_metrics["policy"].map(policy_model_hashes)
    fold_metrics["task_model_set_sha256"] = task_model_set_sha
    expected_samples = set(int(value) for value in dataset.canonical.index)
    for policy, scoped in predictions.groupby("policy", sort=False):
        if len(scoped) != len(dataset.canonical) or set(scoped["sample_index"].astype(int)) != expected_samples:
            raise SupplementaryExternalError(f"Incomplete OOF coverage for {spec.key}/{policy}.")
        if scoped["sample_index"].duplicated().any():
            raise SupplementaryExternalError(f"Duplicate OOF sample for {spec.key}/{policy}.")
    validate_consumer_fold_assignments(folds, predictions, group_columns=("policy",))
    receipt_lookup = outer_receipts.set_index(["policy", "outer_fold"])
    for row in predictions.itertuples(index=False):
        receipt_row = receipt_lookup.loc[(str(row.policy), int(row.outer_fold))]
        if (
            str(row.model_sha256) != str(receipt_row.model_sha256)
            or str(row.model_fit_contract_sha256) != str(receipt_row.model_fit_contract_sha256)
            or int(row.selected_candidate_index) != int(receipt_row.selected_candidate_index)
        ):
            raise SupplementaryExternalError("OOF row does not resolve to its exact outer model.")
    bootstrap_resamples = (
        PRODUCTION_BOOTSTRAP_RESAMPLES
        if test_only_overrides is None
        else test_only_overrides.bootstrap_resamples
    )
    bootstrap = compute_paired_oof_bootstrap(
        predictions,
        labels=labels,
        task_type=spec.task_type,
        metrics=tuple(str(value) for value in metric_names),
        comparisons=tuple(
            ComparisonSpec(
                comparison_id=f"{policy}_minus_{spec.primary_policy}",
                system_a=policy,
                system_b=spec.primary_policy,
            )
            for policy in spec.audit_policies
        ),
        protocol=BootstrapProtocol(
            n_resamples=bootstrap_resamples,
            confidence_level=0.95,
            seed=bootstrap_seed,
            strata_columns=("outer_fold", "y_true"),
            method="paired_stratified_percentile",
            quantile_method="linear",
        ),
    )
    intervals = bootstrap.metric_intervals.rename(columns={"system_id": "policy"})
    intervals = _add_identity(intervals, identity)
    intervals.insert(len(identity), "task_key", spec.key)
    intervals["role"] = spec.role
    intervals["publication_stratum"] = spec.publication_stratum
    intervals["policy_model_set_sha256"] = intervals["policy"].map(policy_model_hashes)
    intervals["task_model_set_sha256"] = task_model_set_sha
    intervals["denominator"] = len(dataset.canonical)
    intervals["uncertainty_method"] = "paired_stratified_percentile_bootstrap"
    differences = _add_identity(bootstrap.paired_differences, identity)
    differences.insert(len(identity), "task_key", spec.key)
    differences["role"] = spec.role
    differences["publication_stratum"] = spec.publication_stratum
    differences["system_a_model_set_sha256"] = differences["system_a"].map(policy_model_hashes)
    differences["system_b_model_set_sha256"] = differences["system_b"].map(policy_model_hashes)
    differences["task_model_set_sha256"] = task_model_set_sha
    fold_summary = _add_identity(
        _descriptive_fold_summary(fold_metrics, metric_names), identity
    )
    fold_summary.insert(len(identity), "task_key", spec.key)
    fold_summary["role"] = spec.role
    fold_summary["publication_stratum"] = spec.publication_stratum
    fold_summary["policy_model_set_sha256"] = fold_summary["policy"].map(policy_model_hashes)
    fold_summary["task_model_set_sha256"] = task_model_set_sha
    task_metadata = {
        **identity,
        "schema_version": 1,
        "status": "complete",
        "task_key": spec.key,
        "task_type": spec.task_type,
        "role": spec.role,
        "publication_stratum": spec.publication_stratum,
        "claim_boundary": spec.claim_boundary,
        "direct_external_validation_of_primary": False,
        "comparable_to_primary_three_class_task": False,
        "locked_model_transport": False,
        "transport_claim_allowed": False,
        "n_rows": len(dataset.canonical),
        "labels": labels,
        "positive_label": spec.positive_label,
        "primary_policy": spec.primary_policy,
        "audit_policies": list(spec.audit_policies),
        "outer_splits": PRODUCTION_OUTER_SPLITS,
        "inner_splits": PRODUCTION_INNER_SPLITS,
        "candidate_indices_evaluated": list(candidate_indices),
        "selection_metric": SELECTION_METRIC,
        "selection_tie_break_metric": SELECTION_TIE_BREAK_METRIC,
        "ordinal_tie_break_used": False,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_resample_hash": bootstrap.metadata["resample_hash"],
        "bootstrap_strata": list(bootstrap.metadata["strata_columns"]),
        "policy_model_set_sha256": policy_model_hashes,
        "task_model_set_sha256": task_model_set_sha,
        "max_probability_replay_error": float(
            outer_receipts["max_probability_replay_error"].max()
        ),
        "probability_source": "raw_exact_outer_fold_model",
        "calibration_claim": "none_for_supplementary_task_strata",
        "canonical_eligible": test_only_overrides is None,
        "test_only_reduction": None
        if test_only_overrides is None
        else {
            "candidate_indices": list(candidate_indices),
            "bootstrap_resamples": bootstrap_resamples,
            "task_keys": list(test_only_overrides.task_keys),
        },
        "network_calls": 0,
        "paid_api_calls": 0,
        "source_authenticity_status": receipt.get("source_authenticity_status"),
        "licence_verification_status": receipt.get("licence_verification_status"),
        "raw_dataset_path": _safe_relative_path(
            receipt.get("actual_path"), field=f"{spec.key}.actual_path"
        ),
        "completed_at": _utc_now(),
    }
    return TaskEvidence(
        spec=spec,
        folds=folds,
        target_mapping=target_mapping,
        target_support=target_support,
        feature_policies=pd.DataFrame(feature_rows),
        candidate_fit_receipts=pd.DataFrame(candidate_fit_rows),
        candidate_search_results=pd.DataFrame(candidate_search_rows),
        selected_hyperparameters=pd.DataFrame(selected_rows),
        outer_model_receipts=outer_receipts,
        oof_predictions=predictions,
        fold_metrics=fold_metrics,
        fold_descriptive_summary=fold_summary,
        metric_intervals=intervals,
        paired_policy_differences=differences,
        model_bytes=MappingProxyType(dict(model_bytes)),
        bootstrap_metadata=dict(bootstrap.metadata),
        task_metadata=task_metadata,
        canonical_eligible=test_only_overrides is None,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _artifact_manifest(
    root: Path,
    *,
    manifest_path: Path,
    identity: Mapping[str, Any],
) -> Path:
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.resolve() != manifest_path.resolve()
        ),
        key=lambda value: value.relative_to(root).as_posix(),
    )
    rows = [
        {
            **identity,
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise SupplementaryExternalError("Closed-world artifact inventory is empty or duplicated.")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def _write_task(evidence: TaskEvidence, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "target_mapping": output / "target_mapping.csv",
        "target_support": output / "target_support.csv",
        "outer_folds": output / "outer_fold_assignments.csv",
        "inner_folds": output / "inner_fold_assignments.csv",
        "fold_contract": output / "fold_contract.json",
        "feature_policies": output / "feature_policy_audit.csv",
        "candidate_fit_receipts": output / "candidate_fit_receipts.csv",
        "candidate_search": output / "candidate_search_results.csv",
        "selected_hyperparameters": output / "selected_hyperparameters.csv",
        "outer_model_receipts": output / "outer_model_receipts.csv",
        "oof_predictions": output / "oof_predictions.csv",
        "fold_metrics": output / "fold_metrics.csv",
        "fold_summary": output / "fold_descriptive_summary.csv",
        "metric_intervals": output / "metric_intervals.csv",
        "policy_differences": output / "paired_policy_differences.csv",
        "bootstrap_metadata": output / "bootstrap_metadata.json",
        "metadata": output / "task_metadata.json",
        "manifest": output / "artifact_manifest.csv",
    }
    frames = {
        "target_mapping": evidence.target_mapping,
        "target_support": evidence.target_support,
        "outer_folds": evidence.folds.outer_assignments,
        "inner_folds": evidence.folds.inner_assignments,
        "feature_policies": evidence.feature_policies,
        "candidate_fit_receipts": evidence.candidate_fit_receipts,
        "candidate_search": evidence.candidate_search_results,
        "selected_hyperparameters": evidence.selected_hyperparameters,
        "outer_model_receipts": evidence.outer_model_receipts,
        "oof_predictions": evidence.oof_predictions,
        "fold_metrics": evidence.fold_metrics,
        "fold_summary": evidence.fold_descriptive_summary,
        "metric_intervals": evidence.metric_intervals,
        "policy_differences": evidence.paired_policy_differences,
    }
    for name, frame in frames.items():
        if frame.empty:
            raise SupplementaryExternalError(
                f"Required task artifact {evidence.spec.key}/{name} is empty."
            )
        frame.to_csv(paths[name], index=False)
    _write_json(paths["fold_contract"], evidence.folds.contract)
    _write_json(paths["bootstrap_metadata"], evidence.bootstrap_metadata)
    model_paths: list[Path] = []
    for (policy, outer_fold), payload in sorted(evidence.model_bytes.items()):
        model_path = output / "models" / policy / f"outer_fold_{outer_fold:02d}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(payload)
        expected = evidence.outer_model_receipts[
            (evidence.outer_model_receipts["policy"] == policy)
            & (evidence.outer_model_receipts["outer_fold"].astype(int) == outer_fold)
        ]
        if len(expected) != 1 or sha256_file(model_path) != str(expected.iloc[0]["model_sha256"]):
            raise SupplementaryExternalError(
                f"Persisted outer model hash drift for {evidence.spec.key}/{policy}/{outer_fold}."
            )
        model_paths.append(model_path)
    metadata = dict(evidence.task_metadata)
    metadata["outer_model_artifacts"] = [
        path.relative_to(output).as_posix() for path in model_paths
    ]
    metadata["output_files"] = sorted(
        [
            path.relative_to(output).as_posix()
            for name, path in paths.items()
            if name not in {"metadata", "manifest"}
        ]
        + [path.relative_to(output).as_posix() for path in model_paths]
    )
    _write_json(paths["metadata"], metadata)
    manifest_identity = {
        field: evidence.task_metadata[field]
        for field in COMMON_IDENTITY_FIELDS
    }
    _artifact_manifest(output, manifest_path=paths["manifest"], identity=manifest_identity)
    expected_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path != paths["manifest"]
    }
    observed_files = set(pd.read_csv(paths["manifest"])["path"].astype(str))
    if expected_files != observed_files:
        raise SupplementaryExternalError(f"Task closed-world manifest drift for {evidence.spec.key}.")
    return sorted(
        [path for path in output.rglob("*") if path.is_file()], key=lambda value: str(value)
    )


def _metric_applicability_table(
    specs: Sequence[SupplementaryTaskSpec], identity: Mapping[str, str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        schema = get_task_schema(spec.task_type)
        for metric in REPORT_METRICS:
            applicable = schema.is_metric_applicable(metric)
            rows.append(
                {
                    **identity,
                    "task_key": spec.key,
                    "task_type": spec.task_type,
                    "publication_stratum": spec.publication_stratum,
                    "metric": metric,
                    "applicable": applicable,
                    "inapplicable_representation": "" if applicable else "N/A",
                    "positive_label": spec.positive_label,
                    "applicability_note": schema.applicability_note,
                    "comparable_to_primary_three_class_task": False,
                }
            )
    return pd.DataFrame(rows)


def _source_table(evidence: TaskEvidence, *, source_artifact: str) -> pd.DataFrame:
    table = evidence.metric_intervals.copy()
    table["source_artifact"] = source_artifact
    table["evaluation_scope"] = "exact_nested_out_of_fold"
    table["direct_external_validation_of_primary"] = False
    table["comparable_to_primary_three_class_task"] = False
    table["locked_model_transport"] = False
    table["claim_boundary"] = evidence.spec.claim_boundary
    return table


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_dir: str | Path,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    source_tree_hash: str,
    git_commit: str,
    scope_contract_hash: str,
    expected_actual_input_receipts: Mapping[str, Any],
    expected_side_input_hashes: Mapping[str, Any],
    expected_git_worktree_dirty: bool,
    test_only_overrides: SupplementaryExternalTestOnlyOverrides | None = None,
) -> dict[str, Path]:
    """Atomically publish the exact supplementary external stage."""

    identity_base = {
        "run_id": _require_identity("run_id", run_id),
        "config_hash": _require_identity("config_hash", config_hash, sha256=True),
        "scientific_input_hash": _require_identity(
            "scientific_input_hash", scientific_input_hash, sha256=True
        ),
        "source_tree_hash": _require_identity("source_tree_hash", source_tree_hash, sha256=True),
        "git_commit": _require_identity("git_commit", git_commit),
        "scope_contract_hash": _require_identity(
            "scope_contract_hash", scope_contract_hash, sha256=True
        ),
    }
    if len(identity_base["git_commit"]) != 40 or any(
        character not in "0123456789abcdef" for character in identity_base["git_commit"]
    ):
        raise SupplementaryExternalError("git_commit must be a lowercase 40-character Git object ID.")
    if not isinstance(expected_git_worktree_dirty, bool):
        raise SupplementaryExternalError("expected_git_worktree_dirty must be boolean.")
    if expected_git_worktree_dirty:
        raise SupplementaryExternalError(
            "Production supplementary evidence requires a clean source-tree identity."
        )
    config = load_config(config_path)
    observed_config_hash = canonical_config_hash(config)
    if observed_config_hash != identity_base["config_hash"]:
        raise SupplementaryExternalError("Supplied config hash does not match the canonical config.")
    protocol = validate_protocol(config)
    settings = _settings(config)
    model_definition, model_grid_sha256 = _model_grid(settings, expected_side_input_hashes)
    acquisition_record = expected_side_input_hashes.get("data_acquisition_contract")
    if not isinstance(acquisition_record, Mapping):
        raise SupplementaryExternalError("The acquisition contract side input is missing.")
    task_keys = (
        tuple(TASK_SPECS)
        if test_only_overrides is None
        else test_only_overrides.task_keys
    )
    specs = tuple(TASK_SPECS[key] for key in task_keys)
    output = Path(output_dir).resolve()
    try:
        output.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        if test_only_overrides is None:
            raise SupplementaryExternalError(
                "Production supplementary output must be contained in the repository."
            ) from exc
    if output.exists():
        raise SupplementaryExternalError(f"Supplementary output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f"{output.name}.__staging__.", dir=output.parent)
    )
    seeds = settings.get("seeds")
    if not isinstance(seeds, Mapping):
        raise SupplementaryExternalError("Canonical seeds are missing.")
    evidence_by_task: dict[str, TaskEvidence] = {}
    all_written: list[Path] = []
    try:
        for spec in specs:
            canonical: CanonicalDataset = load_canonical_dataset(
                config_path,
                spec.canonical_dataset_key,
                allow_download=False,
            )
            receipt = _validate_dataset_receipt(
                canonical.receipt,
                expected_actual_input_receipts,
                spec.canonical_dataset_key,
            )
            if receipt.get("acquisition_manifest_sha256") != acquisition_record.get("sha256"):
                raise SupplementaryExternalError(
                    f"Acquisition-contract hash drift for {spec.canonical_dataset_key!r}."
                )
            dataset_entry = settings.get("datasets", {}).get(spec.canonical_dataset_key)
            if not isinstance(dataset_entry, Mapping):
                raise SupplementaryExternalError(
                    f"Canonical dataset entry is missing for {spec.canonical_dataset_key!r}."
                )
            mapping_path, schema_hash = _validated_side_input(
                expected_side_input_hashes,
                spec.schema_side_input_key,
                str(dataset_entry.get("schema_mapping_path", "")),
            )
            dataset = load_external_dataset(
                spec.dataset_name,
                target_kind=spec.target_kind,
                raw_frame=canonical.frame,
                schema_mapping_path=mapping_path,
            )
            raw_relative = _safe_relative_path(
                receipt.get("actual_path"), field=f"{spec.key}.actual_path"
            )
            content_hash, content_rows, content_columns = _canonical_csv_content_sha256(
                (PROJECT_ROOT / raw_relative).resolve()
            )
            if content_rows != int(receipt["row_count"]) or content_columns != int(
                receipt["column_count"]
            ):
                raise SupplementaryExternalError(
                    f"Canonical content dimensions drift for {spec.canonical_dataset_key!r}."
                )
            metrics = protocol["metrics"][spec.task_type]
            evidence = evaluate_task(
                dataset,
                spec=spec,
                receipt=receipt,
                schema_mapping_sha256=schema_hash,
                canonical_content_sha256=content_hash,
                model_definition=model_definition,
                metric_names=metrics,
                identity_base=identity_base,
                outer_seed=int(seeds["external_replication"]),
                inner_seed=int(seeds["inner_cv"]),
                model_seed=int(seeds["model"]),
                bootstrap_seed=int(seeds["bootstrap"]),
                test_only_overrides=test_only_overrides,
            )
            evidence_by_task[spec.key] = evidence
            all_written.extend(_write_task(evidence, staging / spec.key))

        top_paths = {
            "task_strata_index": staging / "task_strata_index.csv",
            "metric_applicability": staging / "metric_applicability.csv",
            "restricted_target_source": staging
            / "ibm_restricted_target_performance_robustness.csv",
            "ibm_attrition_source": staging / "ibm_attrition_task_transfer.csv",
            "turnover_source": staging / "employee_turnover_task_transfer.csv",
            "interpretation": staging / "supplementary_external_interpretation.md",
            "metadata": staging / "stage_metadata.json",
            "manifest": staging / "stage_artifact_manifest.csv",
        }
        strata_rows = []
        for spec in specs:
            evidence = evidence_by_task[spec.key]
            metadata = evidence.task_metadata
            strata_rows.append(
                {
                    **identity_base,
                    "task_key": spec.key,
                    "canonical_dataset_key": spec.canonical_dataset_key,
                    "dataset_sha256": metadata["dataset_sha256"],
                    "canonical_content_sha256": metadata["canonical_content_sha256"],
                    "schema_mapping_sha256": metadata["schema_mapping_sha256"],
                    "feature_policy_contract_sha256": metadata[
                        "feature_policy_contract_sha256"
                    ],
                    "fold_contract_hash": metadata["fold_contract_hash"],
                    "task_model_set_sha256": metadata["task_model_set_sha256"],
                    "task_type": spec.task_type,
                    "role": spec.role,
                    "publication_stratum": spec.publication_stratum,
                    "denominator": metadata["n_rows"],
                    "labels": ";".join(str(value) for value in spec.labels),
                    "positive_label": spec.positive_label,
                    "primary_policy": spec.primary_policy,
                    "n_outer_folds": PRODUCTION_OUTER_SPLITS,
                    "n_inner_folds": PRODUCTION_INNER_SPLITS,
                    "bootstrap_resamples": metadata["bootstrap_resamples"],
                    "uncertainty_method": "paired_stratified_percentile_bootstrap",
                    "source_artifact": f"{spec.key}/metric_intervals.csv",
                    "direct_external_validation_of_primary": False,
                    "comparable_to_primary_three_class_task": False,
                    "locked_model_transport": False,
                    "claim_boundary": spec.claim_boundary,
                }
            )
        pd.DataFrame(strata_rows).to_csv(top_paths["task_strata_index"], index=False)
        _metric_applicability_table(specs, identity_base).to_csv(
            top_paths["metric_applicability"], index=False
        )
        if "ibm_performance" in evidence_by_task:
            _source_table(
                evidence_by_task["ibm_performance"],
                source_artifact="ibm_performance/metric_intervals.csv",
            ).to_csv(top_paths["restricted_target_source"], index=False)
        else:
            top_paths.pop("restricted_target_source")
        if "ibm_attrition" in evidence_by_task:
            _source_table(
                evidence_by_task["ibm_attrition"],
                source_artifact="ibm_attrition/metric_intervals.csv",
            ).to_csv(top_paths["ibm_attrition_source"], index=False)
        else:
            top_paths.pop("ibm_attrition_source")
        if "employee_turnover" in evidence_by_task:
            _source_table(
                evidence_by_task["employee_turnover"],
                source_artifact="employee_turnover/metric_intervals.csv",
            ).to_csv(top_paths["turnover_source"], index=False)
        else:
            top_paths.pop("turnover_source")
        top_paths["interpretation"].write_text(
            "# Supplementary task-bounded external evidence\n\n"
            "The IBM 3/4 PerformanceRating stratum is restricted-target robustness only. "
            "IBM attrition and Employee Turnover are related binary task-transfer strata. "
            "The three strata are not directly comparable with the primary three-class task, "
            "are not direct employee-performance external validation, and are not locked-model "
            "transport. Inapplicable ordinal metrics are recorded as N/A. Fold summaries are "
            "descriptive only; reported intervals use the predeclared paired sample-level OOF "
            "bootstrap.\n",
            encoding="utf-8",
        )
        stage_metadata = {
            **identity_base,
            "schema_version": 1,
            "status": "complete",
            "stage": "external_robustness",
            "scope": "supplementary_task_bounded_robustness_only",
            "task_keys": list(task_keys),
            "separate_task_strata": True,
            "combined_cross_task_score_table": False,
            "direct_external_validation_of_primary": False,
            "locked_model_transport": False,
            "transport_claim_allowed": False,
            "model_grid_sha256": model_grid_sha256,
            "outer_splits": PRODUCTION_OUTER_SPLITS,
            "inner_splits": PRODUCTION_INNER_SPLITS,
            "selection_metric": SELECTION_METRIC,
            "selection_tie_break_metric": SELECTION_TIE_BREAK_METRIC,
            "ordinal_tie_break_used": False,
            "bootstrap_resamples": PRODUCTION_BOOTSTRAP_RESAMPLES
            if test_only_overrides is None
            else test_only_overrides.bootstrap_resamples,
            "canonical_eligible": test_only_overrides is None,
            "atomic_publication": True,
            "artifact_inventory_scope": "runner_owned_files_before_orchestrator_stage_contract",
            "orchestrator_stage_contract_path": "stage_contract.json",
            "network_calls": 0,
            "paid_api_calls": 0,
            "completed_at": _utc_now(),
        }
        _write_json(top_paths["metadata"], stage_metadata)
        manifest_identity = dict(identity_base)
        _artifact_manifest(
            staging,
            manifest_path=top_paths["manifest"],
            identity=manifest_identity,
        )
        inventory = pd.read_csv(top_paths["manifest"])
        expected_inventory = {
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file() and path != top_paths["manifest"]
        }
        if set(inventory["path"].astype(str)) != expected_inventory:
            raise SupplementaryExternalError("Stage closed-world inventory is incomplete.")
        os.replace(staging, output)
    except Exception:
        # Preserve a failed staging directory for forensic recovery.  The builder
        # refuses to reuse or overwrite it.
        raise
    result = {
        name: output / path.relative_to(staging)
        for name, path in top_paths.items()
    }
    result["all_artifacts"] = sorted(
        [path for path in output.rglob("*") if path.is_file()], key=lambda value: str(value)
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build task-bounded supplementary external evidence.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--scientific-input-hash", required=True)
    parser.add_argument("--source-tree-hash", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--scope-contract-hash", required=True)
    parser.add_argument("--manifest-inputs", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    payload = json.loads(Path(arguments.manifest_inputs).read_text(encoding="utf-8"))
    paths = run(
        arguments.config,
        output_dir=arguments.output_dir,
        run_id=arguments.run_id,
        config_hash=arguments.config_hash,
        scientific_input_hash=arguments.scientific_input_hash,
        source_tree_hash=arguments.source_tree_hash,
        git_commit=arguments.git_commit,
        scope_contract_hash=arguments.scope_contract_hash,
        expected_actual_input_receipts=payload["actual_input_receipts"],
        expected_side_input_hashes=payload["side_input_hashes"],
        expected_git_worktree_dirty=bool(payload.get("git_worktree_dirty", False)),
    )
    print({key: str(value) for key, value in paths.items() if key != "all_artifacts"})


if __name__ == "__main__":
    main()


__all__ = [
    "PRODUCTION_BOOTSTRAP_RESAMPLES",
    "PRODUCTION_CANDIDATE_COUNT",
    "PRODUCTION_INNER_SPLITS",
    "PRODUCTION_OUTER_SPLITS",
    "REPORT_METRICS",
    "SUPPLEMENTARY_METRICS",
    "SupplementaryExternalError",
    "SupplementaryExternalTestOnlyOverrides",
    "SupplementaryTaskSpec",
    "TASK_SPECS",
    "TaskEvidence",
    "evaluate_task",
    "run",
    "validate_protocol",
]
