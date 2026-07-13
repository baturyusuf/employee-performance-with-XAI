"""Fail-closed HRDataset_v14 nested OOF replication engine.

This module is intentionally free of data loading, report paths, configuration
discovery, and legacy experiment helpers.  Callers must supply verified policy
frames, labels, sample identifiers, scientific identities, and the validated
canonical model-grid mapping.  The engine owns only the scientific computation:

* one deterministic dataset-specific 10 outer x 5 inner fold contract;
* macro-F1 XGBoost selection with the predeclared QWK tie-break;
* exact once-per-sample raw OOF predictions for every supplied policy;
* reuse of the primary policy's fold-selected candidate for every policy fit;
* five-fold cross-fitted sigmoid calibration inside each outer-training split;
* paired 5,000-draw sample-level stratified bootstrap uncertainty; and
* auditable fit, model, probability, feature-lineage, and calibrator identities.

No function in this module writes an artifact.  The orchestrator is responsible
for atomic persistence and must reject results whose ``canonical_eligible`` flag
is false.  Reduced candidates/resamples are available only through the visibly
test-only override type and can never be canonical evidence.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import warnings
import zlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.experiments.manuscript_calibration import (
    SigmoidCalibrator,
    apply_sigmoid_calibrator,
    fit_sigmoid_calibrator,
)
from src.experiments.manuscript_model_benchmark import (
    select_candidate_index,
    validate_benchmark_config,
)
from src.experiments.shared_folds import (
    SharedFoldArtifacts,
    generate_shared_folds,
    validate_consumer_fold_assignments,
    validate_shared_folds,
)
from src.models.canonical_models import (
    CANONICAL_ESTIMATOR_PATHS,
    aligned_predict_proba,
    build_model_pipeline,
    merge_model_parameters,
    validate_model_feature_frame,
)
from src.models.evaluate import classification_metrics
from src.models.oof_bootstrap import (
    BootstrapProtocol,
    ComparisonSpec,
    compute_paired_oof_bootstrap,
    validate_aligned_oof_predictions,
)
from src.models.task_schema import ORDINAL_MULTICLASS_PERFORMANCE


DATASET_KEY = "hrdataset_v14"
MODEL_NAME = "xgboost"
PRODUCTION_OUTER_SPLITS = 10
PRODUCTION_INNER_SPLITS = 5
PRODUCTION_BOOTSTRAP_RESAMPLES = 5000
PRODUCTION_CANDIDATE_COUNT = 8
PRIMARY_SELECTION_METRIC = "macro_f1"
SELECTION_TIE_BREAK_METRIC = "quadratic_weighted_kappa"
PRIMARY_PRACTICAL_TIE_TOLERANCE = 0.001
EXPECTED_LABELS = (2, 3, 4)
FIT_THREAD_LIMIT = 1
CONDITIONAL_INFERENCE_NOTE = (
    "Intervals condition on the observed HRDataset_v14 employees and fixed nested "
    "model-training protocol; they do not estimate dataset-source, target-mapping, "
    "or model-training instability."
)
FOLD_DESCRIPTIVE_NOTE = (
    "Outer-fold mean, standard deviation, minimum, and maximum are descriptive "
    "variability only; they are not population confidence intervals."
)
REPLICATION_METRICS = (
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
COMMON_IDENTITY_FIELDS = (
    "run_id",
    "config_hash",
    "scientific_input_hash",
    "dataset_key",
    "dataset_sha256",
    "fold_contract_hash",
)


class HRDatasetReplicationError(RuntimeError):
    """Raised when HRDataset replication violates its scientific contract."""


@dataclass(frozen=True)
class HRDatasetReplicationProtocol:
    """The immutable production protocol; arbitrary scientific variants are rejected."""

    outer_splits: int = PRODUCTION_OUTER_SPLITS
    inner_splits: int = PRODUCTION_INNER_SPLITS
    outer_seed: int = 42
    inner_seed: int = 43
    model_seed: int = 42
    calibration_seed: int = 42
    bootstrap_seed: int = 42
    bootstrap_resamples: int = PRODUCTION_BOOTSTRAP_RESAMPLES
    confidence_level: float = 0.95
    selection_metric: str = PRIMARY_SELECTION_METRIC
    tie_break_metric: str = SELECTION_TIE_BREAK_METRIC
    primary_practical_tie_tolerance: float = PRIMARY_PRACTICAL_TIE_TOLERANCE
    calibration_method: str = "sigmoid"
    bootstrap_method: str = "paired_stratified_percentile"
    bootstrap_strata: tuple[str, ...] = ("outer_fold", "y_true")
    quantile_method: str = "linear"
    estimator_threads: int = FIT_THREAD_LIMIT

    def __post_init__(self) -> None:
        expected = {
            "outer_splits": PRODUCTION_OUTER_SPLITS,
            "inner_splits": PRODUCTION_INNER_SPLITS,
            "outer_seed": 42,
            "inner_seed": 43,
            "model_seed": 42,
            "calibration_seed": 42,
            "bootstrap_seed": 42,
            "bootstrap_resamples": PRODUCTION_BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
            "selection_metric": PRIMARY_SELECTION_METRIC,
            "tie_break_metric": SELECTION_TIE_BREAK_METRIC,
            "primary_practical_tie_tolerance": PRIMARY_PRACTICAL_TIE_TOLERANCE,
            "calibration_method": "sigmoid",
            "bootstrap_method": "paired_stratified_percentile",
            "bootstrap_strata": ("outer_fold", "y_true"),
            "quantile_method": "linear",
            "estimator_threads": FIT_THREAD_LIMIT,
        }
        observed = {field: getattr(self, field) for field in expected}
        if observed != expected:
            raise HRDatasetReplicationError(
                "HRDataset production protocol drifted: "
                + json.dumps(
                    {
                        field: {"expected": expected[field], "observed": observed[field]}
                        for field in expected
                        if observed[field] != expected[field]
                    },
                    sort_keys=True,
                    default=list,
                )
            )


@dataclass(frozen=True)
class HRDatasetTestOnlyOverrides:
    """Explicit noncanonical reductions allowed only for unit-test execution."""

    candidate_indices: tuple[int, ...] = (0,)
    bootstrap_resamples: int = 50

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_indices, tuple)
            or not self.candidate_indices
            or any(isinstance(value, bool) or not isinstance(value, int) for value in self.candidate_indices)
            or tuple(sorted(set(self.candidate_indices))) != self.candidate_indices
        ):
            raise HRDatasetReplicationError(
                "Test-only candidate indices must be a non-empty sorted unique integer tuple."
            )
        if (
            isinstance(self.bootstrap_resamples, bool)
            or not isinstance(self.bootstrap_resamples, int)
            or not 2 <= self.bootstrap_resamples < PRODUCTION_BOOTSTRAP_RESAMPLES
        ):
            raise HRDatasetReplicationError(
                "Test-only bootstrap_resamples must lie in [2, 4999]."
            )


@dataclass(frozen=True)
class HRDatasetReplicationResult:
    """Complete in-memory evidence returned to an atomic artifact orchestrator."""

    folds: SharedFoldArtifacts
    candidate_fit_receipts: pd.DataFrame
    candidate_search_results: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    outer_model_receipts: pd.DataFrame
    transformed_feature_lineage: pd.DataFrame
    raw_oof_predictions: pd.DataFrame
    calibrated_oof_predictions: pd.DataFrame
    calibration_training_oof: pd.DataFrame
    calibration_fit_receipts: pd.DataFrame
    calibrator_parameters: pd.DataFrame
    calibrator_model_relationships: pd.DataFrame
    fold_metrics: pd.DataFrame
    fold_descriptive_summary: pd.DataFrame
    raw_metric_intervals: pd.DataFrame
    raw_policy_differences: pd.DataFrame
    calibration_metric_intervals: pd.DataFrame
    calibration_differences: pd.DataFrame
    bootstrap_resample_plan: "PersistableResamplePlanEvidence"
    fitted_outer_models: Mapping[tuple[str, int], Any]
    serialized_outer_models: Mapping[tuple[str, int], bytes]
    calibrators: Mapping[int, SigmoidCalibrator]
    protocol_metadata: Mapping[str, Any]
    canonical_eligible: bool


@dataclass(frozen=True)
class PersistableResamplePlanEvidence:
    """Portable deterministic bytes and sample order for the shared bootstrap plan."""

    sample_order: pd.DataFrame
    compressed_indices_bytes: bytes
    receipt: Mapping[str, Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_identity(name: str, value: Any, *, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HRDatasetReplicationError(f"{name} must be a non-empty string.")
    observed = value.strip()
    if sha256 and (
        len(observed) != 64 or any(character not in "0123456789abcdef" for character in observed)
    ):
        raise HRDatasetReplicationError(f"{name} must be a lowercase SHA-256 digest.")
    return observed


def _json_mapping(value: Mapping[str, Any]) -> str:
    return _canonical_json(dict(value)).decode("utf-8")


def _sample_set_sha256(values: Sequence[int] | set[int]) -> str:
    return _sha256_bytes(_canonical_json(sorted(int(value) for value in values)))


def _feature_order_sha256(columns: Sequence[str]) -> str:
    return _sha256_bytes(_canonical_json([str(value) for value in columns]))


def _array_sha256(values: Any, *, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    payload = _canonical_json({"dtype": dtype, "shape": list(array.shape)}) + array.tobytes(order="C")
    return _sha256_bytes(payload)


def _serialize_model(model: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=0, protocol=4)
    payload = buffer.getvalue()
    if not payload:
        raise HRDatasetReplicationError("A fitted outer model serialized to empty bytes.")
    return payload


def _persistable_resample_plan(
    *,
    base_predictions: pd.DataFrame,
    plan: Any,
    identity: Mapping[str, str],
) -> PersistableResamplePlanEvidence:
    """Encode one central ResamplePlan without lossy/tabular expansion."""

    sorted_ids = tuple(int(value) for value in plan.sorted_sample_ids)
    base = (
        base_predictions[["sample_index", "outer_fold", "y_true"]]
        .drop_duplicates()
        .set_index("sample_index")
    )
    if set(base.index.astype(int)) != set(sorted_ids) or len(base) != len(sorted_ids):
        raise HRDatasetReplicationError(
            "Bootstrap sample-order evidence does not equal the exactly-once OOF population."
        )
    sample_order = base.loc[list(sorted_ids)].reset_index()
    sample_order.insert(0, "sample_position", np.arange(len(sample_order), dtype=int))
    for position, field in enumerate(COMMON_IDENTITY_FIELDS):
        sample_order.insert(position, field, identity[field])
    sample_order["resample_hash"] = str(plan.resample_hash)
    indices = np.ascontiguousarray(np.asarray(plan.indices, dtype="<i8"))
    if indices.ndim != 2 or indices.shape[1] != len(sample_order):
        raise HRDatasetReplicationError("Bootstrap resample index matrix shape is invalid.")
    buffer = io.BytesIO()
    np.save(buffer, indices, allow_pickle=False)
    uncompressed = buffer.getvalue()
    compressed = zlib.compress(uncompressed, level=9)
    if not compressed:
        raise HRDatasetReplicationError("Bootstrap resample-plan encoding is empty.")
    receipt = MappingProxyType(
        {
            **identity,
            "format": "zlib_compressed_numpy_npy_v1",
            "compression": "zlib_level_9",
            "dtype": "<i8",
            "shape": list(indices.shape),
            "n_samples": len(sample_order),
            "n_resamples": int(indices.shape[0]),
            "sample_id_column": "sample_index",
            "sample_order_sha256": _sha256_bytes(
                sample_order[
                    ["sample_position", "sample_index", "outer_fold", "y_true"]
                ].to_csv(index=False, lineterminator="\n").encode("utf-8")
            ),
            "uncompressed_npy_sha256": _sha256_bytes(uncompressed),
            "compressed_indices_sha256": _sha256_bytes(compressed),
            "compressed_size_bytes": len(compressed),
            "uncompressed_size_bytes": len(uncompressed),
            "resample_hash": str(plan.resample_hash),
            "strata_columns": ["outer_fold", "y_true"],
            "portable_write_mode": "write compressed_indices_bytes verbatim",
        }
    )
    return PersistableResamplePlanEvidence(
        sample_order=sample_order,
        compressed_indices_bytes=compressed,
        receipt=receipt,
    )


def _prediction_labels(probabilities: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    label_array = np.asarray([int(value) for value in labels], dtype=int)
    return label_array[np.argmax(probabilities, axis=1)]


def _finite_metric(metrics: Mapping[str, Any], name: str, *, context: str) -> float:
    value = metrics.get(name)
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise HRDatasetReplicationError(f"{context} produced nonnumeric {name}: {value!r}.") from exc
    if not math.isfinite(observed):
        raise HRDatasetReplicationError(f"{context} produced non-finite {name}: {observed!r}.")
    return observed


def _fit_pipeline(
    pipeline: Any,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    context: str,
) -> Any:
    try:
        with threadpool_limits(limits=FIT_THREAD_LIMIT):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                pipeline.fit(features, target)
    except Exception as exc:
        raise HRDatasetReplicationError(
            f"{context} failed: {type(exc).__name__}: {exc}"
        ) from exc
    return pipeline


def _predict_pipeline(
    pipeline: Any,
    features: pd.DataFrame,
    *,
    labels: Sequence[int],
    context: str,
) -> np.ndarray:
    try:
        with threadpool_limits(limits=FIT_THREAD_LIMIT):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                probability = aligned_predict_proba(
                    pipeline,
                    features,
                    labels=labels,
                )
    except Exception as exc:
        raise HRDatasetReplicationError(
            f"{context} prediction failed: {type(exc).__name__}: {exc}"
        ) from exc
    return probability


def _identity(
    *,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    dataset_sha256: str,
    fold_contract_hash: str,
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "config_hash": config_hash,
        "scientific_input_hash": scientific_input_hash,
        "dataset_key": DATASET_KEY,
        "dataset_sha256": dataset_sha256,
        "fold_contract_hash": fold_contract_hash,
    }


def _validate_inputs(
    policy_frames: Mapping[str, pd.DataFrame],
    policy_roles: Mapping[str, str],
    forbidden_features_by_policy: Mapping[str, Sequence[str]],
    target: pd.Series,
    sample_identifiers: pd.Series,
    *,
    primary_policy: str,
) -> tuple[dict[str, pd.DataFrame], pd.Series, pd.Series, dict[str, tuple[str, ...]]]:
    if not isinstance(policy_frames, Mapping) or not policy_frames:
        raise HRDatasetReplicationError("At least one explicitly supplied policy frame is required.")
    if any(not isinstance(value, str) for value in policy_frames):
        raise HRDatasetReplicationError("Policy-frame keys must be strings; implicit key coercion is forbidden.")
    policy_names = tuple(policy_frames)
    if any(not value.strip() for value in policy_names) or len(set(policy_names)) != len(policy_names):
        raise HRDatasetReplicationError("Policy names must be unique non-empty strings.")
    if not isinstance(primary_policy, str) or not primary_policy.strip():
        raise HRDatasetReplicationError("primary_policy must be a non-empty string.")
    if primary_policy not in policy_frames:
        raise HRDatasetReplicationError(f"Unknown primary policy {primary_policy!r}.")
    for name, mapping in (
        ("policy_roles", policy_roles),
        ("forbidden_features_by_policy", forbidden_features_by_policy),
    ):
        if (
            not isinstance(mapping, Mapping)
            or any(not isinstance(value, str) for value in mapping)
            or set(mapping) != set(policy_names)
        ):
            raise HRDatasetReplicationError(f"{name} keys must equal the supplied policy-frame keys.")
    if not isinstance(target, pd.Series) or target.empty or not target.index.is_unique:
        raise HRDatasetReplicationError("Target must be a non-empty uniquely indexed Series.")
    if not isinstance(sample_identifiers, pd.Series) or not sample_identifiers.index.is_unique:
        raise HRDatasetReplicationError("Sample identifiers must be a uniquely indexed Series.")
    if not target.index.equals(sample_identifiers.index):
        raise HRDatasetReplicationError("Target and sample identifiers must have identical ordered indices.")
    if not all(isinstance(value, (int, np.integer)) for value in target.index):
        raise HRDatasetReplicationError("Sample indices must be integers.")
    if target.isna().any() or sample_identifiers.isna().any() or sample_identifiers.duplicated().any():
        raise HRDatasetReplicationError("Targets and sample identifiers must be non-null; identifiers unique.")
    try:
        numeric_target = pd.to_numeric(target, errors="raise")
    except (TypeError, ValueError) as exc:
        raise HRDatasetReplicationError("Target labels must be integer-valued 2/3/4.") from exc
    if not np.equal(numeric_target, np.round(numeric_target)).all():
        raise HRDatasetReplicationError("Target labels must be integer-valued 2/3/4.")
    clean_target = numeric_target.astype(int)
    if tuple(sorted(clean_target.unique())) != EXPECTED_LABELS:
        raise HRDatasetReplicationError(
            f"HRDataset mapped target must have exact labels {EXPECTED_LABELS}; "
            f"observed={tuple(sorted(clean_target.unique()))}."
        )
    clean_frames: dict[str, pd.DataFrame] = {}
    forbidden_contracts: dict[str, tuple[str, ...]] = {}
    for policy in policy_names:
        role = policy_roles[policy]
        if not isinstance(role, str) or not role.strip():
            raise HRDatasetReplicationError(f"Policy {policy!r} has no declared role.")
        frame = policy_frames[policy]
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise HRDatasetReplicationError(f"Policy {policy!r} must supply a non-empty DataFrame.")
        if set(frame.index) != set(target.index) or len(frame) != len(target):
            raise HRDatasetReplicationError(f"Policy {policy!r} sample coverage differs from target.")
        frame = frame.loc[target.index].copy()
        forbidden_values = forbidden_features_by_policy[policy]
        if isinstance(forbidden_values, (str, bytes)):
            raise HRDatasetReplicationError(f"Policy {policy!r} forbidden features must be a sequence.")
        forbidden = tuple(str(value) for value in forbidden_values)
        if (
            not forbidden
            or any(not value.strip() for value in forbidden)
            or len({value.casefold() for value in forbidden}) != len(forbidden)
        ):
            raise HRDatasetReplicationError(
                f"Policy {policy!r} requires a non-empty case-insensitively unique forbidden contract."
            )
        observed_casefold = {str(column).casefold(): str(column) for column in frame.columns}
        leaked = sorted(
            observed_casefold[value.casefold()]
            for value in forbidden
            if value.casefold() in observed_casefold
        )
        if leaked:
            raise HRDatasetReplicationError(
                f"Policy {policy!r} contains forbidden model features: {leaked}."
            )
        try:
            validate_model_feature_frame(frame, forbidden_features=forbidden)
        except Exception as exc:
            raise HRDatasetReplicationError(
                f"Policy {policy!r} feature contract failed: {type(exc).__name__}: {exc}"
            ) from exc
        clean_frames[policy] = frame
        forbidden_contracts[policy] = forbidden
    return clean_frames, clean_target, sample_identifiers.copy(), forbidden_contracts


def _validate_model_grid(
    benchmark_config: Mapping[str, Any],
    test_only_overrides: HRDatasetTestOnlyOverrides | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[int, ...]]:
    try:
        settings = validate_benchmark_config(benchmark_config)
    except Exception as exc:
        raise HRDatasetReplicationError(
            f"Canonical model-grid contract failed: {type(exc).__name__}: {exc}"
        ) from exc
    if (
        settings.get("selection_metric") != PRIMARY_SELECTION_METRIC
        or settings.get("selection_tie_break_metric") != SELECTION_TIE_BREAK_METRIC
        or float(settings.get("primary_practical_tie_tolerance"))
        != PRIMARY_PRACTICAL_TIE_TOLERANCE
    ):
        raise HRDatasetReplicationError("HRDataset selection metric/tie-break protocol drifted.")
    definition = settings["models"].get(MODEL_NAME)
    if not isinstance(definition, Mapping):
        raise HRDatasetReplicationError("Canonical model-grid has no XGBoost definition.")
    if definition.get("estimator") != CANONICAL_ESTIMATOR_PATHS[MODEL_NAME]:
        raise HRDatasetReplicationError("XGBoost estimator path drifted from the canonical factory.")
    candidates = definition.get("candidates")
    fixed = definition.get("fixed_params")
    if not isinstance(candidates, list) or len(candidates) != PRODUCTION_CANDIDATE_COUNT:
        raise HRDatasetReplicationError("Production HRDataset XGBoost requires exactly eight candidates.")
    if not isinstance(fixed, Mapping) or fixed.get("n_jobs") != FIT_THREAD_LIMIT:
        raise HRDatasetReplicationError("Production HRDataset XGBoost must be single-threaded.")
    indices = (
        tuple(range(PRODUCTION_CANDIDATE_COUNT))
        if test_only_overrides is None
        else test_only_overrides.candidate_indices
    )
    if any(value < 0 or value >= len(candidates) for value in indices):
        raise HRDatasetReplicationError("Test-only candidate index falls outside the canonical grid.")
    return settings, definition, indices


def _outer_membership(folds: SharedFoldArtifacts, outer_fold: int) -> tuple[list[int], list[int]]:
    outer = folds.outer_assignments
    train_ids = outer.loc[outer["outer_fold"].astype(int) != outer_fold, "sample_index"].astype(int).tolist()
    test_ids = outer.loc[outer["outer_fold"].astype(int) == outer_fold, "sample_index"].astype(int).tolist()
    if not train_ids or not test_ids or set(train_ids).intersection(test_ids):
        raise HRDatasetReplicationError(f"Outer fold {outer_fold} membership is invalid.")
    return train_ids, test_ids


def _inner_membership(
    folds: SharedFoldArtifacts,
    outer_fold: int,
    inner_fold: int,
) -> tuple[list[int], list[int]]:
    scoped = folds.inner_assignments[
        folds.inner_assignments["outer_fold"].astype(int) == outer_fold
    ]
    validation = scoped.loc[scoped["inner_fold"].astype(int) == inner_fold, "sample_index"].astype(int).tolist()
    development = scoped.loc[scoped["inner_fold"].astype(int) != inner_fold, "sample_index"].astype(int).tolist()
    train_ids, test_ids = _outer_membership(folds, outer_fold)
    if (
        not validation
        or not development
        or set(validation).intersection(development)
        or set(validation).union(development) != set(train_ids)
        or set(validation).intersection(test_ids)
        or set(development).intersection(test_ids)
    ):
        raise HRDatasetReplicationError(
            f"Outer {outer_fold}/inner {inner_fold} partition isolation failed."
        )
    return development, validation


def _model_lineage_rows(
    pipeline: Any,
    *,
    identity: Mapping[str, str],
    policy: str,
    policy_role: str,
    outer_fold: int,
    model_sha256: str,
) -> list[dict[str, Any]]:
    expected_raw = tuple(str(value) for value in pipeline.feature_names_in_)
    preprocessor = pipeline.named_steps.get("preprocessor")
    observed_raw = tuple(str(value) for value in getattr(preprocessor, "feature_names_in_", ()))
    if observed_raw != expected_raw:
        raise HRDatasetReplicationError(
            f"Policy {policy}/outer {outer_fold} preprocessor raw lineage drifted."
        )
    transformed = tuple(str(value) for value in preprocessor.get_feature_names_out())
    if not transformed or len(set(transformed)) != len(transformed):
        raise HRDatasetReplicationError(
            f"Policy {policy}/outer {outer_fold} transformed lineage is empty or duplicated."
        )
    feature_order_hash = _feature_order_sha256(expected_raw)
    return [
        {
            **identity,
            "policy": policy,
            "policy_role": policy_role,
            "outer_fold": outer_fold,
            "model": MODEL_NAME,
            "model_sha256": model_sha256,
            "raw_feature_order_sha256": feature_order_hash,
            "raw_feature_count": len(expected_raw),
            "transformed_feature_count": len(transformed),
            "transformed_feature_index": index,
            "transformed_feature_name": name,
        }
        for index, name in enumerate(transformed)
    ]


def _add_identity(frame: pd.DataFrame, identity: Mapping[str, str]) -> pd.DataFrame:
    output = frame.copy()
    for position, field in enumerate(COMMON_IDENTITY_FIELDS):
        output.insert(position, field, identity[field])
    output["conditional_inference_note"] = CONDITIONAL_INFERENCE_NOTE
    return output


def _fold_descriptive_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (policy, role), group in fold_metrics.groupby(["policy", "policy_role"], sort=False):
        for metric in REPLICATION_METRICS:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise HRDatasetReplicationError(
                    f"Fold descriptive metric {policy}/{metric} contains non-finite values."
                )
            rows.append(
                {
                    **{field: group[field].iloc[0] for field in COMMON_IDENTITY_FIELDS},
                    "policy": policy,
                    "policy_role": role,
                    "model": MODEL_NAME,
                    "metric": metric,
                    "n_folds": len(values),
                    "fold_mean": float(np.mean(values)),
                    "fold_std": float(np.std(values, ddof=1)),
                    "fold_min": float(np.min(values)),
                    "fold_max": float(np.max(values)),
                    "population_confidence_interval_applicable": False,
                    "interpretation": FOLD_DESCRIPTIVE_NOTE,
                }
            )
    return pd.DataFrame(rows)


def _calibration_evidence(
    *,
    primary_features: pd.DataFrame,
    target: pd.Series,
    folds: SharedFoldArtifacts,
    raw_primary: pd.DataFrame,
    selected_rows: pd.DataFrame,
    fixed_parameters: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    primary_forbidden: Sequence[str],
    primary_policy: str,
    primary_role: str,
    outer_models: Mapping[tuple[str, int], Any],
    outer_model_bytes: Mapping[tuple[str, int], bytes],
    labels: Sequence[int],
    model_seed: int,
    calibration_seed: int,
    identity: Mapping[str, str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Mapping[int, SigmoidCalibrator],
]:
    probability_columns = [f"prob_class_{label}" for label in labels]
    training_rows: list[dict[str, Any]] = []
    fit_receipts: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    calibrated_rows: list[dict[str, Any]] = []
    relationship_rows: list[dict[str, Any]] = []
    calibrators: dict[int, SigmoidCalibrator] = {}
    for outer_fold in range(1, PRODUCTION_OUTER_SPLITS + 1):
        selected = selected_rows[selected_rows["outer_fold"].astype(int) == outer_fold]
        if len(selected) != 1:
            raise HRDatasetReplicationError(
                f"Outer fold {outer_fold} requires exactly one selected primary candidate."
            )
        selected_index = int(selected.iloc[0]["selected_candidate_index"])
        selected_candidate = dict(candidates[selected_index])
        source_bytes = outer_model_bytes[(primary_policy, outer_fold)]
        source_hash = _sha256_bytes(source_bytes)
        source_model = outer_models[(primary_policy, outer_fold)]
        if _sha256_bytes(_serialize_model(source_model)) != source_hash:
            raise HRDatasetReplicationError(
                f"Primary outer model {outer_fold} changed before calibration."
            )
        outer_train_ids, outer_test_ids = _outer_membership(folds, outer_fold)
        fold_training_rows: list[dict[str, Any]] = []
        for inner_fold in range(1, PRODUCTION_INNER_SPLITS + 1):
            development_ids, validation_ids = _inner_membership(folds, outer_fold, inner_fold)
            pipeline = build_model_pipeline(
                MODEL_NAME,
                primary_features.loc[development_ids],
                fixed_parameters=fixed_parameters,
                candidate_parameters=selected_candidate,
                random_state=model_seed,
                forbidden_features=primary_forbidden,
            )
            _fit_pipeline(
                pipeline,
                primary_features.loc[development_ids],
                target.loc[development_ids],
                context=f"calibration cross-fit outer={outer_fold}, inner={inner_fold}",
            )
            probability = _predict_pipeline(
                pipeline,
                primary_features.loc[validation_ids],
                labels=labels,
                context=f"calibration cross-fit outer={outer_fold}, inner={inner_fold}",
            )
            inner_model_hash = _sha256_bytes(_serialize_model(pipeline))
            probability_hash = _array_sha256(probability, dtype="<f8")
            fit_contract = {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "selected_candidate_index": selected_index,
                "fixed_parameters": dict(fixed_parameters),
                "selected_candidate_parameters": selected_candidate,
                "development_sample_sha256": _sample_set_sha256(development_ids),
                "validation_sample_sha256": _sample_set_sha256(validation_ids),
                "outer_test_sample_sha256": _sample_set_sha256(outer_test_ids),
                "feature_order_sha256": _feature_order_sha256(primary_features.columns),
                "source_outer_model_sha256": source_hash,
                "crossfit_model_sha256": inner_model_hash,
                "crossfit_probability_sha256": probability_hash,
            }
            contract_hash = _sha256_bytes(_canonical_json(fit_contract))
            fit_receipts.append(
                {
                    **identity,
                    "policy": primary_policy,
                    "policy_role": primary_role,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "selected_candidate_index": selected_index,
                    "fixed_parameters_json": _json_mapping(fixed_parameters),
                    "selected_candidate_parameters_json": _json_mapping(selected_candidate),
                    "n_inner_development": len(development_ids),
                    "n_inner_validation": len(validation_ids),
                    "n_outer_test": len(outer_test_ids),
                    "inner_development_sample_sha256": fit_contract["development_sample_sha256"],
                    "inner_validation_sample_sha256": fit_contract["validation_sample_sha256"],
                    "outer_test_sample_sha256": fit_contract["outer_test_sample_sha256"],
                    "feature_order_sha256": fit_contract["feature_order_sha256"],
                    "source_outer_model_sha256": source_hash,
                    "crossfit_model_sha256": inner_model_hash,
                    "inner_validation_probability_sha256": probability_hash,
                    "crossfit_fit_contract_sha256": contract_hash,
                    "preprocessing_fit_scope": "inner_development_partition_only",
                    "outer_test_used_for_fit": False,
                    "outer_test_used_for_calibrator_fit": False,
                    "threadpool_limit": FIT_THREAD_LIMIT,
                    "warning_count": 0,
                }
            )
            predictions = _prediction_labels(probability, labels)
            for position, sample_index in enumerate(validation_ids):
                row = {
                    **identity,
                    "policy": primary_policy,
                    "policy_role": primary_role,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "sample_index": int(sample_index),
                    "y_true": int(target.loc[sample_index]),
                    "y_pred": int(predictions[position]),
                    "selected_candidate_index": selected_index,
                    "source_outer_model_sha256": source_hash,
                    "crossfit_model_sha256": inner_model_hash,
                    "crossfit_fit_contract_sha256": contract_hash,
                }
                row.update(
                    {
                        column: float(probability[position, column_index])
                        for column_index, column in enumerate(probability_columns)
                    }
                )
                training_rows.append(row)
                fold_training_rows.append(row)
        training = pd.DataFrame(fold_training_rows).sort_values("sample_index").reset_index(drop=True)
        if (
            len(training) != len(outer_train_ids)
            or training["sample_index"].duplicated().any()
            or set(training["sample_index"].astype(int)) != set(outer_train_ids)
            or set(training["sample_index"].astype(int)).intersection(outer_test_ids)
        ):
            raise HRDatasetReplicationError(
                f"Outer fold {outer_fold} calibration training evidence is not exactly-once inner OOF."
            )
        calibrator = fit_sigmoid_calibrator(
            training[probability_columns].to_numpy(dtype=float),
            training["y_true"].astype(int),
            labels,
            seed=calibration_seed,
        )
        calibrators[outer_fold] = calibrator
        raw_test = raw_primary[raw_primary["outer_fold"].astype(int) == outer_fold].copy()
        raw_test = raw_test.sort_values("sample_index").reset_index(drop=True)
        if (
            set(raw_test["sample_index"].astype(int)) != set(outer_test_ids)
            or set(raw_test["source_outer_model_sha256"].astype(str)) != {source_hash}
        ):
            raise HRDatasetReplicationError(
                f"Outer fold {outer_fold} raw test probabilities do not match the source model."
            )
        calibrated = apply_sigmoid_calibrator(
            calibrator,
            raw_test[probability_columns].to_numpy(dtype=float),
        )
        raw_outer_probability_hash = _array_sha256(
            raw_test[probability_columns].to_numpy(dtype=float), dtype="<f8"
        )
        if set(raw_test["outer_test_probability_sha256"].astype(str)) != {
            raw_outer_probability_hash
        }:
            raise HRDatasetReplicationError(
                f"Outer fold {outer_fold} raw OOF probabilities differ from the exact "
                "prediction-producing outer-model receipt."
            )
        calibrated_outer_probability_hash = _array_sha256(calibrated, dtype="<f8")
        calibrated_prediction = _prediction_labels(calibrated, labels)
        for position, raw_row in enumerate(raw_test.itertuples(index=False)):
            row = {
                **identity,
                "system_id": "sigmoid",
                "policy": primary_policy,
                "policy_role": primary_role,
                "model": MODEL_NAME,
                "probability_method": "predeclared_cross_fitted_sigmoid",
                "outer_fold": outer_fold,
                "sample_index": int(raw_row.sample_index),
                "y_true": int(raw_row.y_true),
                "y_pred": int(calibrated_prediction[position]),
                "selected_candidate_index": selected_index,
                "source_outer_model_sha256": source_hash,
                "calibrator_parameter_sha256": calibrator.parameter_sha256,
            }
            row.update(
                {
                    column: float(calibrated[position, column_index])
                    for column_index, column in enumerate(probability_columns)
                }
            )
            calibrated_rows.append(row)
        for parameters in calibrator.class_parameters:
            parameter_rows.append(
                {
                    **identity,
                    "policy": primary_policy,
                    "policy_role": primary_role,
                    "outer_fold": outer_fold,
                    "selected_candidate_index": selected_index,
                    "source_outer_model_sha256": source_hash,
                    "calibration_method": "sigmoid",
                    "calibrator_parameter_sha256": calibrator.parameter_sha256,
                    "training_probability_sha256": calibrator.training_probability_sha256,
                    "training_labels_sha256": calibrator.training_labels_sha256,
                    "n_calibration_training": len(training),
                    "class_label": parameters.class_label,
                    "coefficient": parameters.coefficient,
                    "intercept": parameters.intercept,
                    "n_positive": parameters.n_positive,
                    "n_negative": parameters.n_negative,
                    "n_iter": parameters.n_iter,
                    "outer_test_used_for_fit": False,
                    "method_selected_from_outer_test": False,
                }
            )
        relationship_rows.append(
            {
                **identity,
                "policy": primary_policy,
                "policy_role": primary_role,
                "outer_fold": outer_fold,
                "model": MODEL_NAME,
                "selected_candidate_index": selected_index,
                "selected_candidate_parameters_json": _json_mapping(selected_candidate),
                "source_outer_model_sha256": source_hash,
                "source_outer_raw_probability_sha256": raw_outer_probability_hash,
                "outer_test_sample_sha256": _sample_set_sha256(outer_test_ids),
                "calibration_training_sample_sha256": _sample_set_sha256(outer_train_ids),
                "calibration_training_probability_sha256": calibrator.training_probability_sha256,
                "calibration_training_labels_sha256": calibrator.training_labels_sha256,
                "calibrator_parameter_sha256": calibrator.parameter_sha256,
                "calibrated_outer_probability_sha256": calibrated_outer_probability_hash,
                "calibrator_applied_to_exact_source_outer_probabilities": True,
                "source_outer_model_preserved": True,
                "outer_test_used_for_model_selection": False,
                "outer_test_used_for_model_fit": False,
                "outer_test_used_for_calibrator_fit": False,
                "calibration_method_selected_from_outer_test": False,
            }
        )
        if _sha256_bytes(_serialize_model(source_model)) != source_hash:
            raise HRDatasetReplicationError(
                f"Primary outer model {outer_fold} changed during calibration."
            )
    calibrated_frame = pd.DataFrame(calibrated_rows).sort_values("sample_index").reset_index(drop=True)
    if (
        len(calibrated_frame) != len(target)
        or calibrated_frame["sample_index"].duplicated().any()
        or set(calibrated_frame["sample_index"].astype(int)) != set(target.index.astype(int))
    ):
        raise HRDatasetReplicationError("Sigmoid OOF evidence is not exactly once per sample.")
    validate_consumer_fold_assignments(folds, calibrated_frame, group_columns=("system_id",))
    validate_aligned_oof_predictions(
        calibrated_frame,
        labels=labels,
        task_type=ORDINAL_MULTICLASS_PERFORMANCE,
        metrics=REPLICATION_METRICS,
    )
    return (
        calibrated_frame,
        pd.DataFrame(training_rows).sort_values(["outer_fold", "sample_index"]).reset_index(drop=True),
        pd.DataFrame(fit_receipts),
        pd.DataFrame(parameter_rows),
        pd.DataFrame(relationship_rows),
        MappingProxyType(calibrators),
    )


def evaluate_hrdataset_replication(
    policy_frames: Mapping[str, pd.DataFrame],
    policy_roles: Mapping[str, str],
    forbidden_features_by_policy: Mapping[str, Sequence[str]],
    target: pd.Series,
    sample_identifiers: pd.Series,
    benchmark_config: Mapping[str, Any],
    *,
    primary_policy: str,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    dataset_sha256: str,
    protocol: HRDatasetReplicationProtocol = HRDatasetReplicationProtocol(),
    test_only_overrides: HRDatasetTestOnlyOverrides | None = None,
) -> HRDatasetReplicationResult:
    """Run the complete XGBoost HRDataset_v14 replication computation.

    Production calls must omit ``test_only_overrides``.  A result created with
    an override is permanently marked noncanonical and must not be persisted in
    a manuscript evidence manifest.
    """

    if not isinstance(protocol, HRDatasetReplicationProtocol):
        raise HRDatasetReplicationError("protocol must be HRDatasetReplicationProtocol.")
    if test_only_overrides is not None and not isinstance(
        test_only_overrides, HRDatasetTestOnlyOverrides
    ):
        raise HRDatasetReplicationError(
            "Reduced execution requires an explicit HRDatasetTestOnlyOverrides instance."
        )
    run_id = _require_identity("run_id", run_id)
    config_hash = _require_identity("config_hash", config_hash, sha256=True)
    scientific_input_hash = _require_identity(
        "scientific_input_hash", scientific_input_hash, sha256=True
    )
    dataset_sha256 = _require_identity("dataset_sha256", dataset_sha256, sha256=True)
    frames, target, sample_identifiers, forbidden_contracts = _validate_inputs(
        policy_frames,
        policy_roles,
        forbidden_features_by_policy,
        target,
        sample_identifiers,
        primary_policy=primary_policy,
    )
    settings, definition, candidate_indices = _validate_model_grid(
        benchmark_config, test_only_overrides
    )
    fixed_parameters = dict(definition["fixed_params"])
    candidates = [dict(value) for value in definition["candidates"]]
    fold_source = pd.DataFrame(
        {
            "__hrdataset_sample_identifier__": sample_identifiers,
            "__hrdataset_mapped_target__": target,
        },
        index=target.index,
    )
    try:
        folds = generate_shared_folds(
            fold_source,
            target_column="__hrdataset_mapped_target__",
            id_column="__hrdataset_sample_identifier__",
            run_id=run_id,
            config_hash=config_hash,
            scientific_input_hash=scientific_input_hash,
            dataset_key=DATASET_KEY,
            dataset_sha256=dataset_sha256,
            outer_splits=protocol.outer_splits,
            inner_splits=protocol.inner_splits,
            seed=protocol.outer_seed,
            inner_seed=protocol.inner_seed,
        )
        validate_shared_folds(folds, source_frame=fold_source)
    except Exception as exc:
        raise HRDatasetReplicationError(
            f"HRDataset shared-fold construction failed: {type(exc).__name__}: {exc}"
        ) from exc
    fold_contract_hash = str(folds.contract["fold_contract_hash"])
    identity = _identity(
        run_id=run_id,
        config_hash=config_hash,
        scientific_input_hash=scientific_input_hash,
        dataset_sha256=dataset_sha256,
        fold_contract_hash=fold_contract_hash,
    )

    candidate_fit_rows: list[dict[str, Any]] = []
    candidate_search_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    outer_receipts: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    fitted_outer_models: dict[tuple[str, int], Any] = {}
    serialized_outer_models: dict[tuple[str, int], bytes] = {}
    primary_features = frames[primary_policy]
    primary_role = str(policy_roles[primary_policy])

    for outer_fold in range(1, PRODUCTION_OUTER_SPLITS + 1):
        outer_train_ids, outer_test_ids = _outer_membership(folds, outer_fold)
        candidate_primary_means: list[float] = []
        candidate_tie_means: list[float] = []
        evaluated_candidate_indices: list[int] = []
        outer_candidate_rows: list[dict[str, Any]] = []
        for candidate_index in candidate_indices:
            candidate = candidates[candidate_index]
            primary_scores: list[float] = []
            tie_scores: list[float] = []
            for inner_fold in range(1, PRODUCTION_INNER_SPLITS + 1):
                development_ids, validation_ids = _inner_membership(
                    folds, outer_fold, inner_fold
                )
                pipeline = build_model_pipeline(
                    MODEL_NAME,
                    primary_features.loc[development_ids],
                    fixed_parameters=fixed_parameters,
                    candidate_parameters=candidate,
                    random_state=protocol.model_seed,
                    forbidden_features=forbidden_contracts[primary_policy],
                )
                _fit_pipeline(
                    pipeline,
                    primary_features.loc[development_ids],
                    target.loc[development_ids],
                    context=(
                        f"candidate search outer={outer_fold}, candidate={candidate_index}, "
                        f"inner={inner_fold}"
                    ),
                )
                probability = _predict_pipeline(
                    pipeline,
                    primary_features.loc[validation_ids],
                    labels=EXPECTED_LABELS,
                    context=(
                        f"candidate selection outer={outer_fold}, inner={inner_fold}, "
                        f"candidate={candidate_index}"
                    ),
                )
                prediction = _prediction_labels(probability, EXPECTED_LABELS)
                metrics = classification_metrics(
                    target.loc[validation_ids],
                    prediction,
                    probability,
                    list(EXPECTED_LABELS),
                    task_type=ORDINAL_MULTICLASS_PERFORMANCE,
                )
                primary_score = _finite_metric(
                    metrics,
                    PRIMARY_SELECTION_METRIC,
                    context=f"outer={outer_fold}/candidate={candidate_index}/inner={inner_fold}",
                )
                tie_score = _finite_metric(
                    metrics,
                    SELECTION_TIE_BREAK_METRIC,
                    context=f"outer={outer_fold}/candidate={candidate_index}/inner={inner_fold}",
                )
                primary_scores.append(primary_score)
                tie_scores.append(tie_score)
                candidate_model_bytes = _serialize_model(pipeline)
                candidate_fit_rows.append(
                    {
                        **identity,
                        "policy": primary_policy,
                        "policy_role": primary_role,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "candidate_index": candidate_index,
                        "candidate_parameters_json": _json_mapping(candidate),
                        "fixed_parameters_json": _json_mapping(fixed_parameters),
                        "n_inner_development": len(development_ids),
                        "n_inner_validation": len(validation_ids),
                        "n_outer_test": len(outer_test_ids),
                        "inner_development_sample_sha256": _sample_set_sha256(development_ids),
                        "inner_validation_sample_sha256": _sample_set_sha256(validation_ids),
                        "outer_test_sample_sha256": _sample_set_sha256(outer_test_ids),
                        "raw_feature_order_sha256": _feature_order_sha256(primary_features.columns),
                        "candidate_model_sha256": _sha256_bytes(candidate_model_bytes),
                        "inner_validation_probability_sha256": _array_sha256(probability, dtype="<f8"),
                        PRIMARY_SELECTION_METRIC: primary_score,
                        SELECTION_TIE_BREAK_METRIC: tie_score,
                        "preprocessing_fit_scope": "inner_development_partition_only",
                        "outer_test_used_for_selection": False,
                        "outer_test_used_for_fit": False,
                        "threadpool_limit": FIT_THREAD_LIMIT,
                        "warning_count": 0,
                    }
                )
            primary_mean = float(np.mean(primary_scores))
            tie_mean = float(np.mean(tie_scores))
            candidate_primary_means.append(primary_mean)
            candidate_tie_means.append(tie_mean)
            evaluated_candidate_indices.append(candidate_index)
            outer_candidate_rows.append(
                {
                    **identity,
                    "policy": primary_policy,
                    "policy_role": primary_role,
                    "outer_fold": outer_fold,
                    "candidate_index": candidate_index,
                    "candidate_parameters_json": _json_mapping(candidate),
                    "selection_metric": PRIMARY_SELECTION_METRIC,
                    "selection_inner_fold_scores_json": _canonical_json(primary_scores).decode("utf-8"),
                    "selection_inner_mean": primary_mean,
                    "selection_inner_std": float(np.std(primary_scores, ddof=1)),
                    "selection_tie_break_metric": SELECTION_TIE_BREAK_METRIC,
                    "tie_break_inner_fold_scores_json": _canonical_json(tie_scores).decode("utf-8"),
                    "tie_break_inner_mean": tie_mean,
                    "tie_break_inner_std": float(np.std(tie_scores, ddof=1)),
                    "primary_practical_tie_tolerance": PRIMARY_PRACTICAL_TIE_TOLERANCE,
                    "n_inner_folds": PRODUCTION_INNER_SPLITS,
                    "candidate_status": "complete",
                    "outer_test_used_for_selection": False,
                }
            )
        selected_local_index = select_candidate_index(
            candidate_primary_means,
            candidate_tie_means,
            better_direction="higher",
            practical_tie_tolerance=PRIMARY_PRACTICAL_TIE_TOLERANCE,
        )
        selected_candidate_index = evaluated_candidate_indices[selected_local_index]
        selected_candidate = candidates[selected_candidate_index]
        best_primary = max(candidate_primary_means)
        tie_pool = [
            evaluated_candidate_indices[index]
            for index, score in enumerate(candidate_primary_means)
            if best_primary - score <= PRIMARY_PRACTICAL_TIE_TOLERANCE
        ]
        for row in outer_candidate_rows:
            local_position = evaluated_candidate_indices.index(int(row["candidate_index"]))
            row["primary_gap_from_best"] = float(
                best_primary - candidate_primary_means[local_position]
            )
            row["within_primary_practical_tie"] = int(row["candidate_index"]) in tie_pool
            row["selected_by_protocol"] = int(row["candidate_index"]) == selected_candidate_index
            candidate_search_rows.append(row)
        selected_rows.append(
            {
                **identity,
                "policy": primary_policy,
                "policy_role": primary_role,
                "outer_fold": outer_fold,
                "model": MODEL_NAME,
                "selected_candidate_index": selected_candidate_index,
                "selected_candidate_parameters_json": _json_mapping(selected_candidate),
                "fixed_parameters_json": _json_mapping(fixed_parameters),
                "selection_metric": PRIMARY_SELECTION_METRIC,
                "selected_inner_mean": candidate_primary_means[selected_local_index],
                "selection_tie_break_metric": SELECTION_TIE_BREAK_METRIC,
                "selected_tie_break_inner_mean": candidate_tie_means[selected_local_index],
                "primary_practical_tie_tolerance": PRIMARY_PRACTICAL_TIE_TOLERANCE,
                "primary_tie_candidate_indices_json": _canonical_json(tie_pool).decode("utf-8"),
                "tie_breaking": settings["tie_breaking"],
                "outer_test_used_for_selection": False,
            }
        )

        for policy, features in frames.items():
            policy_role = str(policy_roles[policy])
            pipeline = build_model_pipeline(
                MODEL_NAME,
                features.loc[outer_train_ids],
                fixed_parameters=fixed_parameters,
                candidate_parameters=selected_candidate,
                random_state=protocol.model_seed,
                forbidden_features=forbidden_contracts[policy],
            )
            _fit_pipeline(
                pipeline,
                features.loc[outer_train_ids],
                target.loc[outer_train_ids],
                context=f"outer model policy={policy}, outer={outer_fold}",
            )
            probability = _predict_pipeline(
                pipeline,
                features.loc[outer_test_ids],
                labels=EXPECTED_LABELS,
                context=f"outer model policy={policy}, outer={outer_fold}",
            )
            prediction = _prediction_labels(probability, EXPECTED_LABELS)
            model_bytes = _serialize_model(pipeline)
            model_hash = _sha256_bytes(model_bytes)
            key = (policy, outer_fold)
            fitted_outer_models[key] = pipeline
            serialized_outer_models[key] = model_bytes
            model_fit_contract = {
                "policy": policy,
                "policy_role": policy_role,
                "outer_fold": outer_fold,
                "selected_candidate_index": selected_candidate_index,
                "fixed_parameters": fixed_parameters,
                "selected_candidate_parameters": selected_candidate,
                "outer_train_sample_sha256": _sample_set_sha256(outer_train_ids),
                "outer_test_sample_sha256": _sample_set_sha256(outer_test_ids),
                "raw_feature_order_sha256": _feature_order_sha256(features.columns),
                "model_sha256": model_hash,
                "outer_test_probability_sha256": _array_sha256(probability, dtype="<f8"),
            }
            outer_receipts.append(
                {
                    **identity,
                    "policy": policy,
                    "policy_role": policy_role,
                    "outer_fold": outer_fold,
                    "model": MODEL_NAME,
                    "selected_candidate_source_policy": primary_policy,
                    "selected_candidate_index": selected_candidate_index,
                    "selected_candidate_parameters_json": _json_mapping(selected_candidate),
                    "fixed_parameters_json": _json_mapping(fixed_parameters),
                    "n_train": len(outer_train_ids),
                    "n_test": len(outer_test_ids),
                    "outer_train_sample_sha256": model_fit_contract["outer_train_sample_sha256"],
                    "outer_test_sample_sha256": model_fit_contract["outer_test_sample_sha256"],
                    "raw_feature_order_sha256": model_fit_contract["raw_feature_order_sha256"],
                    "raw_feature_count": features.shape[1],
                    "model_sha256": model_hash,
                    "model_size_bytes": len(model_bytes),
                    "outer_test_probability_sha256": model_fit_contract[
                        "outer_test_probability_sha256"
                    ],
                    "model_fit_contract_sha256": _sha256_bytes(_canonical_json(model_fit_contract)),
                    "preprocessing_fit_scope": "outer_training_partition_only",
                    "outer_test_used_for_fit": False,
                    "selected_primary_parameters_reused": True,
                    "threadpool_limit": FIT_THREAD_LIMIT,
                    "warning_count": 0,
                }
            )
            lineage_rows.extend(
                _model_lineage_rows(
                    pipeline,
                    identity=identity,
                    policy=policy,
                    policy_role=policy_role,
                    outer_fold=outer_fold,
                    model_sha256=model_hash,
                )
            )
            metrics = classification_metrics(
                target.loc[outer_test_ids],
                prediction,
                probability,
                list(EXPECTED_LABELS),
                task_type=ORDINAL_MULTICLASS_PERFORMANCE,
            )
            fold_metric_rows.append(
                {
                    **identity,
                    "policy": policy,
                    "policy_role": policy_role,
                    "model": MODEL_NAME,
                    "probability_method": "raw",
                    "outer_fold": outer_fold,
                    "n_train": len(outer_train_ids),
                    "n_test": len(outer_test_ids),
                    "selected_candidate_index": selected_candidate_index,
                    "source_outer_model_sha256": model_hash,
                    **metrics,
                    "population_confidence_interval_applicable": False,
                    "interpretation": FOLD_DESCRIPTIVE_NOTE,
                }
            )
            for position, sample_index in enumerate(outer_test_ids):
                row = {
                    **identity,
                    "system_id": policy,
                    "policy": policy,
                    "policy_role": policy_role,
                    "model": MODEL_NAME,
                    "probability_method": "raw",
                    "outer_fold": outer_fold,
                    "sample_index": int(sample_index),
                    "y_true": int(target.loc[sample_index]),
                    "y_pred": int(prediction[position]),
                    "selected_candidate_index": selected_candidate_index,
                    "source_outer_model_sha256": model_hash,
                    "outer_test_probability_sha256": model_fit_contract[
                        "outer_test_probability_sha256"
                    ],
                }
                row.update(
                    {
                        f"prob_class_{label}": float(probability[position, column])
                        for column, label in enumerate(EXPECTED_LABELS)
                    }
                )
                prediction_rows.append(row)

    candidate_fit_receipts = pd.DataFrame(candidate_fit_rows)
    candidate_search_results = pd.DataFrame(candidate_search_rows)
    selected_hyperparameters = pd.DataFrame(selected_rows)
    outer_model_receipts = pd.DataFrame(outer_receipts)
    transformed_feature_lineage = pd.DataFrame(lineage_rows)
    raw_oof = pd.DataFrame(prediction_rows).sort_values(["policy", "sample_index"]).reset_index(drop=True)
    fold_metrics = pd.DataFrame(fold_metric_rows).sort_values(["policy", "outer_fold"]).reset_index(drop=True)
    validate_consumer_fold_assignments(folds, raw_oof, group_columns=("system_id",))
    validate_aligned_oof_predictions(
        raw_oof,
        labels=EXPECTED_LABELS,
        task_type=ORDINAL_MULTICLASS_PERFORMANCE,
        metrics=REPLICATION_METRICS,
    )
    expected_raw_rows = len(target) * len(frames)
    if len(raw_oof) != expected_raw_rows:
        raise HRDatasetReplicationError(
            f"Raw OOF row count mismatch: expected={expected_raw_rows}, observed={len(raw_oof)}."
        )
    for policy in frames:
        scoped = raw_oof[raw_oof["policy"] == policy]
        if scoped["sample_index"].duplicated().any() or set(scoped["sample_index"].astype(int)) != set(
            target.index.astype(int)
        ):
            raise HRDatasetReplicationError(f"Policy {policy!r} is not exactly-once OOF.")

    (
        calibrated_oof,
        calibration_training_oof,
        calibration_fit_receipts,
        calibrator_parameters,
        calibrator_model_relationships,
        calibrators,
    ) = (
        _calibration_evidence(
            primary_features=primary_features,
            target=target,
            folds=folds,
            raw_primary=raw_oof[raw_oof["policy"] == primary_policy].copy(),
            selected_rows=selected_hyperparameters,
            fixed_parameters=fixed_parameters,
            candidates=candidates,
            primary_forbidden=forbidden_contracts[primary_policy],
            primary_policy=primary_policy,
            primary_role=primary_role,
            outer_models=fitted_outer_models,
            outer_model_bytes=serialized_outer_models,
            labels=EXPECTED_LABELS,
            model_seed=protocol.model_seed,
            calibration_seed=protocol.calibration_seed,
            identity=identity,
        )
    )

    n_resamples = (
        protocol.bootstrap_resamples
        if test_only_overrides is None
        else test_only_overrides.bootstrap_resamples
    )
    bootstrap_protocol = BootstrapProtocol(
        n_resamples=n_resamples,
        confidence_level=protocol.confidence_level,
        seed=protocol.bootstrap_seed,
        strata_columns=protocol.bootstrap_strata,
        method=protocol.bootstrap_method,
        quantile_method=protocol.quantile_method,
    )
    raw_comparisons = tuple(
        ComparisonSpec(
            comparison_id=f"{policy}_minus_{primary_policy}",
            system_a=policy,
            system_b=primary_policy,
            primary_gate=False,
        )
        for policy in frames
        if policy != primary_policy
    )
    raw_bootstrap = compute_paired_oof_bootstrap(
        raw_oof,
        labels=EXPECTED_LABELS,
        task_type=ORDINAL_MULTICLASS_PERFORMANCE,
        metrics=REPLICATION_METRICS,
        comparisons=raw_comparisons,
        protocol=bootstrap_protocol,
    )
    raw_primary = raw_oof[raw_oof["policy"] == primary_policy].copy()
    raw_primary["system_id"] = "raw"
    calibration_comparison_input = pd.concat(
        [raw_primary, calibrated_oof], ignore_index=True, sort=False
    )
    calibration_bootstrap = compute_paired_oof_bootstrap(
        calibration_comparison_input,
        labels=EXPECTED_LABELS,
        task_type=ORDINAL_MULTICLASS_PERFORMANCE,
        metrics=REPLICATION_METRICS,
        comparisons=(
            ComparisonSpec(
                comparison_id="sigmoid_minus_raw",
                system_a="sigmoid",
                system_b="raw",
                primary_gate=False,
            ),
        ),
        protocol=bootstrap_protocol,
    )
    raw_resample_hash = str(raw_bootstrap.metadata["resample_hash"])
    calibration_resample_hash = str(calibration_bootstrap.metadata["resample_hash"])
    if raw_resample_hash != calibration_resample_hash:
        raise HRDatasetReplicationError(
            "Policy and calibration bootstrap computations did not reuse one sample resample plan."
        )
    if (
        raw_bootstrap.resample_plan.sorted_sample_ids
        != calibration_bootstrap.resample_plan.sorted_sample_ids
        or not np.array_equal(
            raw_bootstrap.resample_plan.indices,
            calibration_bootstrap.resample_plan.indices,
        )
    ):
        raise HRDatasetReplicationError(
            "Policy and calibration bootstrap computations did not reuse identical sample indices."
        )
    resample_plan_evidence = _persistable_resample_plan(
        base_predictions=raw_oof[raw_oof["policy"] == primary_policy],
        plan=raw_bootstrap.resample_plan,
        identity=identity,
    )
    raw_metric_intervals = _add_identity(raw_bootstrap.metric_intervals, identity)
    raw_policy_differences = _add_identity(raw_bootstrap.paired_differences, identity)
    calibration_metric_intervals = _add_identity(
        calibration_bootstrap.metric_intervals, identity
    )
    calibration_differences = _add_identity(
        calibration_bootstrap.paired_differences, identity
    )
    fold_summary = _fold_descriptive_summary(fold_metrics)
    canonical_eligible = test_only_overrides is None
    protocol_metadata = MappingProxyType(
        {
            **identity,
            "task_type": ORDINAL_MULTICLASS_PERFORMANCE,
            "labels": list(EXPECTED_LABELS),
            "primary_policy": primary_policy,
            "policy_order": list(frames),
            "policy_roles": dict(policy_roles),
            "outer_splits": PRODUCTION_OUTER_SPLITS,
            "inner_splits": PRODUCTION_INNER_SPLITS,
            "selection_metric": PRIMARY_SELECTION_METRIC,
            "selection_tie_break_metric": SELECTION_TIE_BREAK_METRIC,
            "primary_practical_tie_tolerance": PRIMARY_PRACTICAL_TIE_TOLERANCE,
            "candidate_indices_evaluated": list(candidate_indices),
            "production_candidate_count": PRODUCTION_CANDIDATE_COUNT,
            "preprocessing": "canonical_training_partition_only",
            "estimator_threads": FIT_THREAD_LIMIT,
            "calibration_method": "predeclared_cross_fitted_sigmoid",
            "calibration_inner_folds": PRODUCTION_INNER_SPLITS,
            "outer_test_use": "evaluation_only",
            "bootstrap_n_resamples": n_resamples,
            "bootstrap_method": protocol.bootstrap_method,
            "bootstrap_strata": list(protocol.bootstrap_strata),
            "bootstrap_resample_hash": raw_resample_hash,
            "bootstrap_sample_order_sha256": resample_plan_evidence.receipt[
                "sample_order_sha256"
            ],
            "bootstrap_compressed_indices_sha256": resample_plan_evidence.receipt[
                "compressed_indices_sha256"
            ],
            "bootstrap_resample_plan_format": resample_plan_evidence.receipt["format"],
            "conditional_inference_note": CONDITIONAL_INFERENCE_NOTE,
            "fold_summary_scope": FOLD_DESCRIPTIVE_NOTE,
            "canonical_eligible": canonical_eligible,
            "test_only_reduction": None
            if test_only_overrides is None
            else {
                "candidate_indices": list(test_only_overrides.candidate_indices),
                "bootstrap_resamples": test_only_overrides.bootstrap_resamples,
            },
            "paid_api_calls": 0,
            "network_calls": 0,
        }
    )
    return HRDatasetReplicationResult(
        folds=folds,
        candidate_fit_receipts=candidate_fit_receipts,
        candidate_search_results=candidate_search_results,
        selected_hyperparameters=selected_hyperparameters,
        outer_model_receipts=outer_model_receipts,
        transformed_feature_lineage=transformed_feature_lineage,
        raw_oof_predictions=raw_oof,
        calibrated_oof_predictions=calibrated_oof,
        calibration_training_oof=calibration_training_oof,
        calibration_fit_receipts=calibration_fit_receipts,
        calibrator_parameters=calibrator_parameters,
        calibrator_model_relationships=calibrator_model_relationships,
        fold_metrics=fold_metrics,
        fold_descriptive_summary=fold_summary,
        raw_metric_intervals=raw_metric_intervals,
        raw_policy_differences=raw_policy_differences,
        calibration_metric_intervals=calibration_metric_intervals,
        calibration_differences=calibration_differences,
        bootstrap_resample_plan=resample_plan_evidence,
        fitted_outer_models=MappingProxyType(fitted_outer_models),
        serialized_outer_models=MappingProxyType(serialized_outer_models),
        calibrators=calibrators,
        protocol_metadata=protocol_metadata,
        canonical_eligible=canonical_eligible,
    )


__all__ = [
    "COMMON_IDENTITY_FIELDS",
    "CONDITIONAL_INFERENCE_NOTE",
    "DATASET_KEY",
    "EXPECTED_LABELS",
    "FOLD_DESCRIPTIVE_NOTE",
    "HRDatasetReplicationError",
    "HRDatasetReplicationProtocol",
    "HRDatasetReplicationResult",
    "HRDatasetTestOnlyOverrides",
    "PersistableResamplePlanEvidence",
    "PRIMARY_PRACTICAL_TIE_TOLERANCE",
    "PRIMARY_SELECTION_METRIC",
    "PRODUCTION_BOOTSTRAP_RESAMPLES",
    "PRODUCTION_INNER_SPLITS",
    "PRODUCTION_OUTER_SPLITS",
    "REPLICATION_METRICS",
    "SELECTION_TIE_BREAK_METRIC",
    "evaluate_hrdataset_replication",
]
