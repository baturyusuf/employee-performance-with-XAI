"""Predeclared cross-fitted sigmoid calibration for the canonical XGBoost model.

The stage consumes the exact shared 10x5 folds and prediction-producing
outer-fold XGBoost artifacts.  For every outer fold it refits the selected
XGBoost candidate on four inner partitions, predicts the held-out fifth, and
uses the resulting outer-training OOF probabilities to fit one-vs-rest Platt
sigmoids.  The calibrator is then applied to the untouched probabilities from
the exact persisted benchmark model for that outer test fold.

Outer-test observations are evaluation-only.  They never participate in model
tuning, inner refitting, calibrator fitting, method selection, or threshold
selection.  Sigmoid is predeclared; raw probabilities are a comparator and no
outcome-ranked calibration-method selection exists in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from src.core.atomic_publish import atomic_replace_directory, cleanup_temporary_directory
from src.core.io_utils import write_json
from src.data.canonical_loader import load_canonical_dataset
from src.experiments.benchmark_artifact_contract import (
    BenchmarkArtifactContractError,
    PERSISTED_PROBABILITY_ATOL,
    XGBoostOOFArtifacts,
    read_xgboost_oof_artifacts,
    validate_xgboost_oof_replay,
)
from src.experiments.manuscript_policy_ablation import exact_policy_frame
from src.experiments.shared_folds import (
    SharedFoldContractError,
    validate_consumer_fold_assignments,
)
from src.governance.manuscript_contract import (
    canonical_config_hash,
    load_manuscript_config,
    primary_excluded_features,
    sha256_file,
)
from src.models.canonical_models import (
    CanonicalModelError,
    aligned_predict_proba,
    build_model_pipeline,
    merge_model_parameters,
)
from src.models.evaluate import classification_metrics
from src.models.oof_bootstrap import (
    BootstrapProtocol,
    BootstrapResult,
    ComparisonSpec,
    OOFBootstrapError,
    compute_paired_oof_bootstrap,
    validate_aligned_oof_predictions,
)
from src.models.task_schema import get_task_schema


DEFAULT_CONFIG = Path("configs/manuscript_final.yaml")
PRIMARY_TASK = "ordinal_multiclass_performance"
PRIMARY_METHOD = "sigmoid"
SYSTEM_ORDER = ("raw", "sigmoid")
REQUIRED_OUTER_FOLDS = 10
REQUIRED_INNER_FOLDS = 5
REQUIRED_BOOTSTRAP_RESAMPLES = 5000
METRICS = (
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
IDENTITY_FIELDS = (
    "run_id",
    "config_hash",
    "scientific_input_hash",
    "fold_contract_hash",
    "xgboost_model_set_sha256",
    "dataset_sha256",
    "calibration_protocol_sha256",
)
_SHA256_HEX = frozenset("0123456789abcdef")
_DEFAULT_SIGMOID = {
    "algorithm": "one_vs_rest_platt_logit_then_row_renormalize",
    "implementation_dependency": "scikit-learn>=1.8,<1.9",
    "solver": "lbfgs",
    "regularization": "l2_via_l1_ratio_zero",
    "l1_ratio": 0.0,
    "C": 1.0,
    "fit_intercept": True,
    "max_iter": 1000,
    "tol": 1e-10,
    "probability_clip": 1e-6,
    "solver_threadpool_limit": 1,
}


class CalibrationContractError(RuntimeError):
    """Raised when canonical cross-fitted calibration cannot be trusted."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    observed = str(value)
    if len(observed) != 64 or any(character not in _SHA256_HEX for character in observed):
        raise CalibrationContractError(f"{name} must be a lowercase SHA-256.")
    return observed


def _array_sha256(values: np.ndarray, *, dtype: str) -> str:
    array = np.asarray(values, dtype=np.dtype(dtype))
    contiguous = np.ascontiguousarray(array)
    header = _canonical_json_bytes({"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)})
    return _sha256_bytes(header + contiguous.tobytes(order="C"))


def _sample_set_sha256(values: Iterable[int]) -> str:
    ordered = sorted(int(value) for value in values)
    if len(ordered) != len(set(ordered)):
        raise CalibrationContractError("Sample-set hashes require unique sample indices.")
    return _sha256_bytes(_canonical_json_bytes(ordered))


def _frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise CalibrationContractError(f"Cannot hash frame; columns are missing: {missing}.")
    payload = frame.loc[:, list(columns)].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return _sha256_bytes(payload)


def _json_mapping(value: Any, *, context: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CalibrationContractError(f"{context} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise CalibrationContractError(f"{context} must contain a JSON object.")
    return dict(parsed)


def _probability_columns(labels: Sequence[int]) -> list[str]:
    return [f"prob_class_{int(label)}" for label in labels]


def _validate_probability_matrix(
    probabilities: Any,
    *,
    n_rows: int,
    labels: Sequence[int],
    context: str,
    normalize: bool = True,
) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    expected_shape = (int(n_rows), len(tuple(labels)))
    if array.ndim != 2 or array.shape != expected_shape:
        raise CalibrationContractError(
            f"{context} probability shape {array.shape} does not equal {expected_shape}."
        )
    if not np.all(np.isfinite(array)):
        raise CalibrationContractError(f"{context} probabilities contain non-finite values.")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise CalibrationContractError(f"{context} probabilities fall outside [0,1].")
    row_sums = array.sum(axis=1, dtype=np.float64)
    if np.any(row_sums <= 0.0) or not np.all(np.isfinite(row_sums)):
        raise CalibrationContractError(f"{context} probabilities cannot be normalized.")
    if normalize:
        array = array / row_sums[:, np.newaxis]
        tolerance = np.finfo(np.float64).eps * max(2, len(tuple(labels)))
    else:
        # Persisted XGBoost probabilities originate as float32 and can retain a
        # small simplex residual.  Validation must accept the upstream contract
        # without silently renormalizing the scientific evidence.
        tolerance = PERSISTED_PROBABILITY_ATOL
    if not np.allclose(array.sum(axis=1), 1.0, rtol=0.0, atol=tolerance):
        raise CalibrationContractError(f"{context} probabilities do not satisfy the simplex.")
    return array


def _predict_labels(probabilities: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    label_array = np.asarray([int(value) for value in labels], dtype=int)
    return label_array[np.argmax(probabilities, axis=1)]


@dataclass(frozen=True)
class SigmoidClassParameters:
    class_label: int
    coefficient: float
    intercept: float
    n_positive: int
    n_negative: int
    n_iter: int


@dataclass(frozen=True)
class SigmoidCalibrator:
    """Replayable one-vs-rest logit sigmoid followed by row renormalization."""

    labels: tuple[int, ...]
    class_parameters: tuple[SigmoidClassParameters, ...]
    seed: int
    solver: str
    regularization: str
    l1_ratio: float
    c_value: float
    fit_intercept: bool
    max_iter: int
    tolerance: float
    probability_clip: float
    threadpool_limit: int
    training_probability_sha256: str
    training_labels_sha256: str

    def __post_init__(self) -> None:
        if tuple(item.class_label for item in self.class_parameters) != self.labels:
            raise CalibrationContractError("Sigmoid class-parameter order must equal label order.")
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            raise CalibrationContractError("Sigmoid labels must be unique and non-empty.")
        _require_sha256("training_probability_sha256", self.training_probability_sha256)
        _require_sha256("training_labels_sha256", self.training_labels_sha256)
        expected_solver_contract = {
            "solver": "lbfgs",
            "regularization": "l2_via_l1_ratio_zero",
            "l1_ratio": 0.0,
            "C": 1.0,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 1e-10,
            "threadpool_limit": 1,
        }
        observed_solver_contract = {
            "solver": self.solver,
            "regularization": self.regularization,
            "l1_ratio": self.l1_ratio,
            "C": self.c_value,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
            "tol": self.tolerance,
            "threadpool_limit": self.threadpool_limit,
        }
        if observed_solver_contract != expected_solver_contract:
            raise CalibrationContractError(
                "Sigmoid calibrator parameters differ from the frozen L2 Platt contract."
            )
        if not 0.0 < float(self.probability_clip) < 0.5:
            raise CalibrationContractError("Sigmoid probability_clip must lie in (0, 0.5).")
        for item in self.class_parameters:
            if (
                item.n_positive <= 0
                or item.n_negative <= 0
                or item.n_iter <= 0
                or not math.isfinite(item.coefficient)
                or not math.isfinite(item.intercept)
            ):
                raise CalibrationContractError(
                    f"Sigmoid parameters for class {item.class_label} are invalid."
                )

    @property
    def parameter_sha256(self) -> str:
        return _sha256_bytes(
            _canonical_json_bytes(
                {
                    "labels": list(self.labels),
                    "class_parameters": [
                        {
                            "class_label": item.class_label,
                            "coefficient": item.coefficient,
                            "intercept": item.intercept,
                            "n_positive": item.n_positive,
                            "n_negative": item.n_negative,
                            "n_iter": item.n_iter,
                        }
                        for item in self.class_parameters
                    ],
                    "seed": self.seed,
                    "solver": self.solver,
                    "regularization": self.regularization,
                    "l1_ratio": self.l1_ratio,
                    "C": self.c_value,
                    "fit_intercept": self.fit_intercept,
                    "max_iter": self.max_iter,
                    "tol": self.tolerance,
                    "probability_clip": self.probability_clip,
                    "threadpool_limit": self.threadpool_limit,
                    "training_probability_sha256": self.training_probability_sha256,
                    "training_labels_sha256": self.training_labels_sha256,
                }
            )
        )

    def transform(self, probabilities: Any) -> np.ndarray:
        raw = _validate_probability_matrix(
            probabilities,
            n_rows=len(probabilities),
            labels=self.labels,
            context="Sigmoid application",
            normalize=False,
        )
        calibrated = np.empty_like(raw, dtype=np.float64)
        for column, parameters in enumerate(self.class_parameters):
            clipped = np.clip(
                raw[:, column], self.probability_clip, 1.0 - self.probability_clip
            )
            logits = np.log(clipped / (1.0 - clipped))
            linear = parameters.coefficient * logits + parameters.intercept
            positive = linear >= 0.0
            values = np.empty_like(linear, dtype=np.float64)
            values[positive] = 1.0 / (1.0 + np.exp(-linear[positive]))
            exp_value = np.exp(linear[~positive])
            values[~positive] = exp_value / (1.0 + exp_value)
            calibrated[:, column] = values
        return _validate_probability_matrix(
            calibrated,
            n_rows=len(raw),
            labels=self.labels,
            context="Sigmoid calibrated",
        )


def fit_sigmoid_calibrator(
    probabilities: Any,
    y_true: Iterable[int],
    labels: Sequence[int],
    *,
    seed: int,
    settings: Mapping[str, Any] | None = None,
) -> SigmoidCalibrator:
    """Fit only the predeclared one-vs-rest Platt sigmoid contract."""

    labels_tuple = tuple(int(value) for value in labels)
    y_array = np.asarray(list(y_true), dtype=int)
    raw = _validate_probability_matrix(
        probabilities,
        n_rows=len(y_array),
        labels=labels_tuple,
        context="Sigmoid training",
        normalize=False,
    )
    if len(y_array) == 0:
        raise CalibrationContractError("Sigmoid training evidence is empty.")
    observed = set(int(value) for value in np.unique(y_array))
    if observed != set(labels_tuple):
        raise CalibrationContractError(
            f"Sigmoid training labels {sorted(observed)} do not cover {list(labels_tuple)}."
        )
    protocol = dict(_DEFAULT_SIGMOID if settings is None else settings)
    if protocol != _DEFAULT_SIGMOID:
        raise CalibrationContractError("Sigmoid settings differ from the predeclared contract.")
    class_parameters: list[SigmoidClassParameters] = []
    for column, label in enumerate(labels_tuple):
        binary = (y_array == label).astype(int)
        positives = int(binary.sum())
        negatives = int(len(binary) - positives)
        if positives <= 0 or negatives <= 0:
            raise CalibrationContractError(
                f"Class {label} has degenerate sigmoid training support: +{positives}/-{negatives}."
            )
        clipped = np.clip(
            raw[:, column],
            float(protocol["probability_clip"]),
            1.0 - float(protocol["probability_clip"]),
        )
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        estimator = LogisticRegression(
            solver=str(protocol["solver"]),
            l1_ratio=float(protocol["l1_ratio"]),
            C=float(protocol["C"]),
            fit_intercept=bool(protocol["fit_intercept"]),
            max_iter=int(protocol["max_iter"]),
            tol=float(protocol["tol"]),
            random_state=int(seed),
        )
        try:
            with threadpool_limits(limits=int(protocol["solver_threadpool_limit"])):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    estimator.fit(logits, binary)
        except (Warning, ValueError) as exc:
            raise CalibrationContractError(
                f"Class {label} sigmoid fitting failed: {type(exc).__name__}: {exc}"
            ) from exc
        coefficient = float(estimator.coef_[0, 0])
        intercept = float(estimator.intercept_[0])
        n_iter = int(np.max(estimator.n_iter_))
        if not math.isfinite(coefficient) or not math.isfinite(intercept) or n_iter <= 0:
            raise CalibrationContractError(f"Class {label} sigmoid parameters are invalid.")
        class_parameters.append(
            SigmoidClassParameters(
                class_label=label,
                coefficient=coefficient,
                intercept=intercept,
                n_positive=positives,
                n_negative=negatives,
                n_iter=n_iter,
            )
        )
    return SigmoidCalibrator(
        labels=labels_tuple,
        class_parameters=tuple(class_parameters),
        seed=int(seed),
        solver=str(protocol["solver"]),
        regularization=str(protocol["regularization"]),
        l1_ratio=float(protocol["l1_ratio"]),
        c_value=float(protocol["C"]),
        fit_intercept=bool(protocol["fit_intercept"]),
        max_iter=int(protocol["max_iter"]),
        tolerance=float(protocol["tol"]),
        probability_clip=float(protocol["probability_clip"]),
        threadpool_limit=int(protocol["solver_threadpool_limit"]),
        training_probability_sha256=_array_sha256(raw, dtype="<f8"),
        training_labels_sha256=_array_sha256(y_array, dtype="<i8"),
    )


def apply_sigmoid_calibrator(
    calibrator: SigmoidCalibrator, probabilities: Any
) -> np.ndarray:
    if not isinstance(calibrator, SigmoidCalibrator):
        raise CalibrationContractError("Expected a SigmoidCalibrator instance.")
    return calibrator.transform(probabilities)


def calibrator_from_parameter_rows(frame: pd.DataFrame) -> SigmoidCalibrator:
    """Reconstruct one fold calibrator from its auditable parameter rows."""

    required = {
        "class_label",
        "coefficient",
        "intercept",
        "n_positive",
        "n_negative",
        "n_iter",
        "calibration_seed",
        "solver",
        "regularization",
        "l1_ratio",
        "C",
        "fit_intercept",
        "max_iter",
        "tol",
        "probability_clip",
        "threadpool_limit",
        "training_probability_sha256",
        "training_labels_sha256",
        "sigmoid_parameter_sha256",
    }
    missing = sorted(required.difference(frame.columns))
    if missing or frame.empty:
        raise CalibrationContractError(
            f"Calibrator parameter rows are empty or missing fields: {missing}."
        )
    shared_columns = [
        "calibration_seed",
        "solver",
        "regularization",
        "l1_ratio",
        "C",
        "fit_intercept",
        "max_iter",
        "tol",
        "probability_clip",
        "threadpool_limit",
        "training_probability_sha256",
        "training_labels_sha256",
        "sigmoid_parameter_sha256",
    ]
    for column in shared_columns:
        if frame[column].nunique(dropna=False) != 1:
            raise CalibrationContractError(f"Calibrator parameter field {column} is inconsistent.")
    ordered = frame.sort_values("class_label")
    calibrator = SigmoidCalibrator(
        labels=tuple(int(value) for value in ordered["class_label"]),
        class_parameters=tuple(
            SigmoidClassParameters(
                class_label=int(row.class_label),
                coefficient=float(row.coefficient),
                intercept=float(row.intercept),
                n_positive=int(row.n_positive),
                n_negative=int(row.n_negative),
                n_iter=int(row.n_iter),
            )
            for row in ordered.itertuples(index=False)
        ),
        seed=int(ordered["calibration_seed"].iloc[0]),
        solver=str(ordered["solver"].iloc[0]),
        regularization=str(ordered["regularization"].iloc[0]),
        l1_ratio=float(ordered["l1_ratio"].iloc[0]),
        c_value=float(ordered["C"].iloc[0]),
        fit_intercept=str(ordered["fit_intercept"].iloc[0]).strip().lower()
        in {"true", "1"},
        max_iter=int(ordered["max_iter"].iloc[0]),
        tolerance=float(ordered["tol"].iloc[0]),
        probability_clip=float(ordered["probability_clip"].iloc[0]),
        threadpool_limit=int(ordered["threadpool_limit"].iloc[0]),
        training_probability_sha256=str(ordered["training_probability_sha256"].iloc[0]),
        training_labels_sha256=str(ordered["training_labels_sha256"].iloc[0]),
    )
    expected_hash = str(ordered["sigmoid_parameter_sha256"].iloc[0])
    if calibrator.parameter_sha256 != expected_hash:
        raise CalibrationContractError("Reconstructed sigmoid parameter hash does not match evidence.")
    return calibrator


def calibration_bin_rows(
    y_true: Sequence[int],
    probabilities: np.ndarray,
    labels: Sequence[int],
    *,
    run_id: str,
    config_hash: str,
    method: str,
    n_bins: int,
    identity: Mapping[str, Any] | None = None,
) -> list[Dict[str, Any]]:
    """Return every declared class/bin, including empty bins with explicit zero support."""

    if int(n_bins) != 10:
        raise CalibrationContractError("Canonical reliability evidence requires exactly 10 bins.")
    labels_tuple = tuple(int(value) for value in labels)
    y_array = np.asarray(y_true, dtype=int)
    probability = _validate_probability_matrix(
        probabilities,
        n_rows=len(y_array),
        labels=labels_tuple,
        context=f"Reliability {method}",
        normalize=False,
    )
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    base_identity = dict(identity or {})
    base_identity.setdefault("run_id", run_id)
    base_identity.setdefault("config_hash", config_hash)
    rows: list[Dict[str, Any]] = []
    for label_index, label in enumerate(labels_tuple):
        scores = probability[:, label_index]
        outcomes = (y_array == label).astype(int)
        assignments = np.digitize(scores, edges[1:-1], right=True) + 1
        for bin_index in range(1, int(n_bins) + 1):
            mask = assignments == bin_index
            count = int(mask.sum())
            positives = int(outcomes[mask].sum()) if count else 0
            predicted = float(scores[mask].mean()) if count else None
            observed = float(positives / count) if count else None
            if count:
                z = 1.959963984540054
                denominator = 1.0 + (z * z / count)
                centre = (observed + z * z / (2.0 * count)) / denominator
                half_width = (
                    z
                    * math.sqrt(
                        observed * (1.0 - observed) / count
                        + z * z / (4.0 * count * count)
                    )
                    / denominator
                )
                observed_low = max(0.0, centre - half_width)
                observed_high = min(1.0, centre + half_width)
            else:
                observed_low = None
                observed_high = None
            rows.append(
                {
                    **base_identity,
                    "method": method,
                    "primary_method": method == PRIMARY_METHOD,
                    "class_label": int(label),
                    "bin": int(bin_index),
                    "bin_low": float(edges[bin_index - 1]),
                    "bin_high": float(edges[bin_index]),
                    "n_samples": count,
                    "n_positive": positives,
                    "bin_status": "observed" if count else "empty",
                    "mean_predicted_probability": predicted,
                    "observed_frequency": observed,
                    "observed_frequency_ci_low": observed_low,
                    "observed_frequency_ci_high": observed_high,
                    "absolute_gap": (
                        abs(float(predicted) - float(observed)) if count else None
                    ),
                }
            )
    return rows


def validate_calibration_protocol(settings: Mapping[str, Any]) -> dict[str, Any]:
    protocol = settings.get("calibration")
    if not isinstance(protocol, Mapping):
        raise CalibrationContractError("Canonical calibration protocol is missing.")
    if protocol.get("primary_method") != PRIMARY_METHOD:
        raise CalibrationContractError("Primary calibration method must remain sigmoid.")
    if tuple(protocol.get("comparison_systems", ())) != SYSTEM_ORDER:
        raise CalibrationContractError("Calibration comparison order must be raw then sigmoid.")
    required = {
        "method_selection": "predeclared_not_outer_test_selected",
        "selection_performed": False,
        "training_protocol": "five_fold_cross_fitted_outer_training_only",
        "outer_folds_source": "shared_folds.outer_fold_assignments",
        "inner_folds_source": "shared_folds.inner_fold_assignments",
        "inner_splits": REQUIRED_INNER_FOLDS,
        "inner_model_parameters_source": (
            "model_benchmarks.xgboost_selected_candidate_by_outer_fold"
        ),
        "inner_model_seed_source": "seeds.model",
        "preprocessing_fit_scope": "inner_development_partition_only",
        "calibrator_fit_scope": "outer_training_cross_fitted_probabilities_only",
        "outer_model_source": (
            "model_benchmarks.persisted_selected_xgboost_outer_fold_pipeline"
        ),
        "outer_model_refit_in_calibration_stage": False,
        "outer_test_probability_source": "model_benchmarks.exact_xgboost_oof_predictions",
        "outer_test_usage": "evaluation_only",
        "outer_test_used_for_tuning_fitting_selection_or_thresholds": False,
        "label_decision_rule": "argmax_fixed_label_order_2_3_4",
        "threshold_selection": "none",
        "n_bins": 10,
        "uncertainty_source": "evaluation.bootstrap",
        "fold_summary_scope": "descriptive_variability_only_no_population_ci",
    }
    differences = {
        key: {"expected": expected, "observed": protocol.get(key)}
        for key, expected in required.items()
        if protocol.get(key) != expected
    }
    if differences or dict(protocol.get("sigmoid", {})) != _DEFAULT_SIGMOID:
        raise CalibrationContractError(
            "Calibration protocol drifted: "
            + json.dumps(differences, sort_keys=True, ensure_ascii=True)
        )
    return dict(protocol)


def _configured_metrics(settings: Mapping[str, Any], task_type: str) -> tuple[str, ...]:
    applicability = settings.get("evaluation", {}).get("metric_applicability", {})
    task = applicability.get(task_type, {})
    metrics = tuple(str(value) for value in task.get("applicable", ()))
    if len(set(metrics)) != len(metrics):
        raise CalibrationContractError(
            f"Metric applicability for {task_type!r} contains duplicates."
        )
    schema = get_task_schema(task_type)
    invalid = sorted(metric for metric in metrics if metric not in schema.applicable_metrics)
    missing = sorted(set(METRICS).difference(metrics))
    if invalid or missing:
        raise CalibrationContractError(
            "Calibration report metrics must be an applicable subset of the complete task registry; "
            f"invalid={invalid}, missing_report_metrics={missing}."
        )
    return METRICS


def _resolve_seed(settings: Mapping[str, Any], key: str) -> int:
    value = settings.get("seeds", {}).get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationContractError(f"Seed {key!r} must be an explicit integer.")
    return int(value)


def _bootstrap_protocol(settings: Mapping[str, Any]) -> BootstrapProtocol:
    bootstrap = settings.get("evaluation", {}).get("bootstrap", {})
    if bootstrap.get("seed") != "bootstrap":
        raise CalibrationContractError("Bootstrap seed must reference seeds.bootstrap.")
    strata = tuple(str(value) for value in bootstrap.get("stratify_by", ()))
    try:
        protocol = BootstrapProtocol(
            n_resamples=int(bootstrap.get("n_resamples", -1)),
            confidence_level=float(bootstrap.get("confidence_level", float("nan"))),
            seed=_resolve_seed(settings, "bootstrap"),
            strata_columns=strata,
            method=str(bootstrap.get("method", "")),
            quantile_method=str(bootstrap.get("quantile_method", "")),
        )
    except (OOFBootstrapError, TypeError, ValueError) as exc:
        raise CalibrationContractError(
            "Calibration bootstrap differs from the frozen paired stratified 95% "
            f"sample-level percentile contract: {exc}"
        ) from exc
    if protocol.n_resamples != REQUIRED_BOOTSTRAP_RESAMPLES:
        raise CalibrationContractError("Calibration requires exactly 5,000 bootstrap draws.")
    expected = {
        "confidence_level": 0.95,
        "strata_columns": ("outer_fold", "y_true"),
        "method": "paired_stratified_percentile",
        "quantile_method": "linear",
    }
    observed = {
        "confidence_level": protocol.confidence_level,
        "strata_columns": protocol.strata_columns,
        "method": protocol.method,
        "quantile_method": protocol.quantile_method,
    }
    if observed != expected:
        raise CalibrationContractError(
            "Calibration bootstrap differs from the frozen paired stratified 95% "
            "sample-level percentile contract."
        )
    return protocol


def _calibration_protocol_sha256(
    *,
    protocol: Mapping[str, Any],
    labels: Sequence[int],
    task_type: str,
    policy: str,
    excluded_features: Sequence[str],
    feature_columns: Sequence[str],
    metrics: Sequence[str],
    bootstrap: BootstrapProtocol,
    model_seed: int,
    calibration_seed: int,
) -> str:
    payload = {
        "calibration": dict(protocol),
        "labels": [int(value) for value in labels],
        "task_type": task_type,
        "primary_policy": policy,
        "excluded_features": list(excluded_features),
        "feature_columns": list(feature_columns),
        "metrics": list(metrics),
        "bootstrap": {
            "n_resamples": bootstrap.n_resamples,
            "confidence_level": bootstrap.confidence_level,
            "seed": bootstrap.seed,
            "strata_columns": list(bootstrap.strata_columns),
            "method": bootstrap.method,
            "quantile_method": bootstrap.quantile_method,
        },
        "model_seed": model_seed,
        "calibration_seed": calibration_seed,
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _selected_parameters_for_fold(
    bundle: XGBoostOOFArtifacts, outer_fold: int
) -> tuple[dict[str, Any], dict[str, Any], int]:
    selected = bundle.selected_hyperparameters
    rows = selected[pd.to_numeric(selected["outer_fold"], errors="raise").astype(int) == outer_fold]
    if len(rows) != 1:
        raise CalibrationContractError(
            f"Outer fold {outer_fold} requires one selected XGBoost parameter row."
        )
    row = rows.iloc[0]
    fixed = _json_mapping(row["fixed_parameters_json"], context="fixed_parameters_json")
    candidate = _json_mapping(
        row["selected_candidate_parameters_json"],
        context="selected_candidate_parameters_json",
    )
    candidate_index = int(row["selected_candidate_index"])
    if bundle.fold_models[outer_fold].selected_candidate_index != candidate_index:
        raise CalibrationContractError(
            f"Outer fold {outer_fold} selected candidate differs from its exact benchmark model."
        )
    return fixed, candidate, candidate_index


def _validate_inner_pipeline(
    pipeline: Any,
    *,
    feature_columns: Sequence[str],
    fixed: Mapping[str, Any],
    candidate: Mapping[str, Any],
    model_seed: int,
    outer_fold: int,
    inner_fold: int,
) -> None:
    expected_features = tuple(str(value) for value in feature_columns)
    observed_pipeline = tuple(str(value) for value in getattr(pipeline, "feature_names_in_", ()))
    preprocessor = pipeline.named_steps.get("preprocessor")
    observed_preprocessor = tuple(
        str(value) for value in getattr(preprocessor, "feature_names_in_", ())
    )
    if observed_pipeline != expected_features or observed_preprocessor != expected_features:
        raise CalibrationContractError(
            f"Outer {outer_fold}/inner {inner_fold} fitted feature lineage drifted."
        )
    expected_parameters = merge_model_parameters(fixed, candidate)
    expected_parameters["random_state"] = int(model_seed)
    expected_parameters["n_jobs"] = 1
    model = pipeline.named_steps.get("model")
    observed_parameters = model.get_params(deep=False)
    drift = {
        key: {"expected": value, "observed": observed_parameters.get(key)}
        for key, value in expected_parameters.items()
        if observed_parameters.get(key) != value
    }
    if drift:
        raise CalibrationContractError(
            f"Outer {outer_fold}/inner {inner_fold} fitted parameters drifted: "
            + json.dumps(drift, sort_keys=True, ensure_ascii=True)
        )


def cross_fit_outer_training(
    *,
    features: pd.DataFrame,
    target: pd.Series,
    bundle: XGBoostOOFArtifacts,
    outer_fold: int,
    forbidden_features: Sequence[str],
    model_seed: int,
    identity: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one OOF probability per outer-training sample using persisted inner folds."""

    if outer_fold not in range(1, REQUIRED_OUTER_FOLDS + 1):
        raise CalibrationContractError("outer_fold must lie in 1..10.")
    outer = bundle.folds.outer_assignments.copy()
    inner = bundle.folds.inner_assignments.copy()
    outer["outer_fold"] = pd.to_numeric(outer["outer_fold"], errors="raise").astype(int)
    inner["outer_fold"] = pd.to_numeric(inner["outer_fold"], errors="raise").astype(int)
    inner["inner_fold"] = pd.to_numeric(inner["inner_fold"], errors="raise").astype(int)
    outer_test_ids = set(
        outer.loc[outer["outer_fold"] == outer_fold, "sample_index"].astype(int)
    )
    outer_train_ids = set(
        outer.loc[outer["outer_fold"] != outer_fold, "sample_index"].astype(int)
    )
    inner_for_outer = inner[inner["outer_fold"] == outer_fold].copy()
    inner_ids = inner_for_outer["sample_index"].astype(int)
    if (
        len(outer_test_ids) == 0
        or len(outer_train_ids) == 0
        or set(inner_ids) != outer_train_ids
        or inner_ids.duplicated().any()
        or outer_test_ids.intersection(outer_train_ids)
    ):
        raise CalibrationContractError(
            f"Outer fold {outer_fold} inner membership does not equal the outer-training partition."
        )
    if sorted(inner_for_outer["inner_fold"].unique()) != list(
        range(1, REQUIRED_INNER_FOLDS + 1)
    ):
        raise CalibrationContractError(f"Outer fold {outer_fold} does not contain five inner folds.")
    observed_y = inner_for_outer.set_index("sample_index")["y_true"].astype(int).sort_index()
    expected_y = target.loc[observed_y.index].astype(int).sort_index()
    if not observed_y.equals(expected_y):
        raise CalibrationContractError(f"Outer fold {outer_fold} inner labels drifted from data.")

    fixed, candidate, candidate_index = _selected_parameters_for_fold(bundle, outer_fold)
    source_model = bundle.fold_models[outer_fold]
    probability_columns = _probability_columns(bundle.labels)
    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for inner_fold in range(1, REQUIRED_INNER_FOLDS + 1):
        validation_ids = inner_for_outer.loc[
            inner_for_outer["inner_fold"] == inner_fold, "sample_index"
        ].astype(int).tolist()
        development_ids = inner_for_outer.loc[
            inner_for_outer["inner_fold"] != inner_fold, "sample_index"
        ].astype(int).tolist()
        if (
            not validation_ids
            or not development_ids
            or set(validation_ids).intersection(development_ids)
            or set(validation_ids).intersection(outer_test_ids)
            or set(development_ids).intersection(outer_test_ids)
            or set(validation_ids).union(development_ids) != outer_train_ids
        ):
            raise CalibrationContractError(
                f"Outer {outer_fold}/inner {inner_fold} partition isolation failed."
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                pipeline = build_model_pipeline(
                    "xgboost",
                    features.loc[development_ids],
                    fixed_parameters=fixed,
                    candidate_parameters=candidate,
                    random_state=int(model_seed),
                    forbidden_features=forbidden_features,
                )
                with threadpool_limits(limits=1):
                    pipeline.fit(
                        features.loc[development_ids], target.loc[development_ids]
                    )
                _validate_inner_pipeline(
                    pipeline,
                    feature_columns=features.columns,
                    fixed=fixed,
                    candidate=candidate,
                    model_seed=model_seed,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                )
                probability = aligned_predict_proba(
                    pipeline,
                    features.loc[validation_ids],
                    labels=bundle.labels,
                )
        except (CanonicalModelError, KeyError, TypeError, ValueError, Warning) as exc:
            raise CalibrationContractError(
                f"Outer {outer_fold}/inner {inner_fold} cross-fit failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        probability = _validate_probability_matrix(
            probability,
            n_rows=len(validation_ids),
            labels=bundle.labels,
            context=f"Outer {outer_fold}/inner {inner_fold}",
            normalize=False,
        )
        contract_payload = {
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "selected_candidate_index": candidate_index,
            "fixed_parameters": fixed,
            "candidate_parameters": candidate,
            "model_seed": model_seed,
            "feature_columns": list(features.columns),
            "forbidden_features": list(forbidden_features),
            "development_sample_sha256": _sample_set_sha256(development_ids),
            "validation_sample_sha256": _sample_set_sha256(validation_ids),
            "outer_test_sample_sha256": _sample_set_sha256(outer_test_ids),
            "source_outer_model_sha256": source_model.sha256,
            "fold_contract_hash": bundle.identity.fold_contract_hash,
        }
        fit_contract_sha256 = _sha256_bytes(_canonical_json_bytes(contract_payload))
        probability_sha256 = _array_sha256(probability, dtype="<f8")
        receipts.append(
            {
                **identity,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "selected_candidate_index": candidate_index,
                "fixed_parameters_json": json.dumps(fixed, sort_keys=True, separators=(",", ":")),
                "selected_candidate_parameters_json": json.dumps(
                    candidate, sort_keys=True, separators=(",", ":")
                ),
                "model_seed": int(model_seed),
                "n_inner_development": len(development_ids),
                "n_inner_validation": len(validation_ids),
                "n_outer_test": len(outer_test_ids),
                "inner_development_sample_sha256": contract_payload[
                    "development_sample_sha256"
                ],
                "inner_validation_sample_sha256": contract_payload[
                    "validation_sample_sha256"
                ],
                "outer_test_sample_sha256": contract_payload["outer_test_sample_sha256"],
                "inner_validation_probability_sha256": probability_sha256,
                "feature_order_sha256": _sha256_bytes(
                    _canonical_json_bytes(list(features.columns))
                ),
                "source_outer_model_sha256": source_model.sha256,
                "crossfit_fit_contract_sha256": fit_contract_sha256,
                "preprocessing_fit_scope": "inner_development_partition_only",
                "outer_test_used_for_fit": False,
                "threadpool_limit": 1,
                "warning_count": 0,
                "model_persisted": False,
            }
        )
        predictions = _predict_labels(probability, bundle.labels)
        for position, sample_index in enumerate(validation_ids):
            row = {
                **identity,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "sample_index": int(sample_index),
                "y_true": int(target.loc[sample_index]),
                "y_pred": int(predictions[position]),
                "selected_candidate_index": candidate_index,
                "source_outer_model_sha256": source_model.sha256,
                "crossfit_fit_contract_sha256": fit_contract_sha256,
            }
            row.update(
                {
                    column: float(probability[position, index])
                    for index, column in enumerate(probability_columns)
                }
            )
            rows.append(row)
    prediction_frame = pd.DataFrame(rows).sort_values("sample_index").reset_index(drop=True)
    if (
        len(prediction_frame) != len(outer_train_ids)
        or prediction_frame["sample_index"].duplicated().any()
        or set(prediction_frame["sample_index"].astype(int)) != outer_train_ids
        or set(prediction_frame["sample_index"].astype(int)).intersection(outer_test_ids)
    ):
        raise CalibrationContractError(
            f"Outer fold {outer_fold} cross-fit predictions are not exactly-once outer-training OOF."
        )
    if len(receipts) != REQUIRED_INNER_FOLDS:
        raise CalibrationContractError(f"Outer fold {outer_fold} did not produce five fit receipts.")
    return prediction_frame, pd.DataFrame(receipts)


def _calibrator_parameter_rows(
    calibrator: SigmoidCalibrator,
    *,
    outer_fold: int,
    candidate_index: int,
    source_outer_model_sha256: str,
    calibrator_contract_sha256: str,
    identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            **identity,
            "outer_fold": int(outer_fold),
            "class_label": item.class_label,
            "coefficient": item.coefficient,
            "intercept": item.intercept,
            "n_positive": item.n_positive,
            "n_negative": item.n_negative,
            "n_iter": item.n_iter,
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
            "training_probability_sha256": calibrator.training_probability_sha256,
            "training_labels_sha256": calibrator.training_labels_sha256,
            "sigmoid_parameter_sha256": calibrator.parameter_sha256,
            "calibrator_contract_sha256": calibrator_contract_sha256,
            "selected_candidate_index": int(candidate_index),
            "source_outer_model_sha256": source_outer_model_sha256,
            "outer_test_used_for_calibrator_fit": False,
        }
        for item in calibrator.class_parameters
    ]


def _calibration_evidence(
    *,
    features: pd.DataFrame,
    target: pd.Series,
    bundle: XGBoostOOFArtifacts,
    forbidden_features: Sequence[str],
    model_seed: int,
    calibration_seed: int,
    sigmoid_settings: Mapping[str, Any],
    primary_policy: str,
    identity: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training_frames: list[pd.DataFrame] = []
    receipt_frames: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []
    relationship_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    probability_columns = _probability_columns(bundle.labels)
    benchmark_oof = bundle.oof_predictions.copy()
    for outer_fold in range(1, REQUIRED_OUTER_FOLDS + 1):
        crossfit, receipts = cross_fit_outer_training(
            features=features,
            target=target,
            bundle=bundle,
            outer_fold=outer_fold,
            forbidden_features=forbidden_features,
            model_seed=model_seed,
            identity=identity,
        )
        training_frames.append(crossfit)
        receipt_frames.append(receipts)
        calibration_probability = crossfit[probability_columns].to_numpy(dtype=np.float64)
        calibration_labels = crossfit["y_true"].to_numpy(dtype=int)
        calibrator = fit_sigmoid_calibrator(
            calibration_probability,
            calibration_labels,
            bundle.labels,
            seed=calibration_seed,
            settings=sigmoid_settings,
        )
        source_model = bundle.fold_models[outer_fold]
        selected_candidate = source_model.selected_candidate_index
        training_evidence_sha256 = _frame_sha256(
            crossfit,
            ["sample_index", "inner_fold", "y_true", *probability_columns],
        )
        calibrator_contract = {
            **identity,
            "outer_fold": outer_fold,
            "source_outer_model_sha256": source_model.sha256,
            "selected_candidate_index": selected_candidate,
            "calibration_protocol_sha256": identity["calibration_protocol_sha256"],
            "fold_contract_hash": identity["fold_contract_hash"],
            "training_evidence_sha256": training_evidence_sha256,
            "sigmoid_parameter_sha256": calibrator.parameter_sha256,
            "n_crossfit_rows": len(crossfit),
            "outer_test_used_for_fit": False,
        }
        calibrator_contract_sha256 = _sha256_bytes(
            _canonical_json_bytes(calibrator_contract)
        )
        parameter_rows.extend(
            _calibrator_parameter_rows(
                calibrator,
                outer_fold=outer_fold,
                candidate_index=selected_candidate,
                source_outer_model_sha256=source_model.sha256,
                calibrator_contract_sha256=calibrator_contract_sha256,
                identity=identity,
            )
        )
        raw_rows = benchmark_oof[
            pd.to_numeric(benchmark_oof["outer_fold"], errors="raise").astype(int)
            == outer_fold
        ].sort_values("sample_index")
        expected_test_ids = tuple(
            sorted(int(value) for value in source_model.test_sample_indices)
        )
        if tuple(raw_rows["sample_index"].astype(int)) != expected_test_ids:
            raise CalibrationContractError(
                f"Outer fold {outer_fold} raw rows differ from the exact benchmark test membership."
            )
        if not raw_rows["y_true"].astype(int).reset_index(drop=True).equals(
            target.loc[list(expected_test_ids)].astype(int).reset_index(drop=True)
        ):
            raise CalibrationContractError(
                f"Outer fold {outer_fold} raw benchmark labels drifted from canonical data."
            )
        raw_probability = _validate_probability_matrix(
            raw_rows[probability_columns].to_numpy(dtype=np.float64),
            n_rows=len(raw_rows),
            labels=bundle.labels,
            context=f"Outer fold {outer_fold} exact raw benchmark",
            normalize=False,
        )
        raw_prediction = _predict_labels(raw_probability, bundle.labels)
        if not np.array_equal(raw_prediction, raw_rows["y_pred"].to_numpy(dtype=int)):
            raise CalibrationContractError(
                f"Outer fold {outer_fold} benchmark labels are not the declared argmax rule."
            )
        sigmoid_probability = apply_sigmoid_calibrator(calibrator, raw_probability)
        sigmoid_prediction = _predict_labels(sigmoid_probability, bundle.labels)
        relationship_rows.append(
            {
                **identity,
                "outer_fold": outer_fold,
                "selected_candidate_index": selected_candidate,
                "source_outer_model_sha256": source_model.sha256,
                "source_outer_model_size_bytes": source_model.size_bytes,
                "n_crossfit_training_rows": len(crossfit),
                "n_outer_test_rows": len(raw_rows),
                "crossfit_training_evidence_sha256": training_evidence_sha256,
                "sigmoid_parameter_sha256": calibrator.parameter_sha256,
                "calibrator_contract_sha256": calibrator_contract_sha256,
                "raw_outer_test_probability_sha256": _array_sha256(
                    raw_probability, dtype="<f8"
                ),
                "sigmoid_outer_test_probability_sha256": _array_sha256(
                    sigmoid_probability, dtype="<f8"
                ),
                "outer_model_refit_in_calibration_stage": False,
                "outer_test_used_for_calibrator_fit": False,
                "method_selection_performed": False,
                "threshold_selection_performed": False,
            }
        )
        for method, probability, prediction in (
            ("raw", raw_probability, raw_prediction),
            ("sigmoid", sigmoid_probability, sigmoid_prediction),
        ):
            for position, source in enumerate(raw_rows.itertuples(index=False)):
                row: dict[str, Any] = {
                    **identity,
                    "system_id": method,
                    "method": method,
                    "primary_method": method == PRIMARY_METHOD,
                    "model": "xgboost",
                    "policy": primary_policy,
                    "sample_index": int(source.sample_index),
                    "outer_fold": outer_fold,
                    "y_true": int(source.y_true),
                    "y_pred": int(prediction[position]),
                    "raw_y_pred": int(raw_prediction[position]),
                    "decision_changed_from_raw": bool(
                        int(prediction[position]) != int(raw_prediction[position])
                    ),
                    "selected_candidate_index": selected_candidate,
                    "source_outer_model_sha256": source_model.sha256,
                    "calibrator_contract_sha256": calibrator_contract_sha256,
                    "probability_source": (
                        "exact_benchmark_oof"
                        if method == "raw"
                        else "sigmoid_applied_to_exact_benchmark_oof"
                    ),
                    "outer_test_usage": "evaluation_only",
                }
                row.update(
                    {
                        column: float(probability[position, index])
                        for index, column in enumerate(probability_columns)
                    }
                )
                evaluation_rows.append(row)
    return (
        pd.concat(training_frames, ignore_index=True).sort_values(
            ["outer_fold", "sample_index"]
        ).reset_index(drop=True),
        pd.concat(receipt_frames, ignore_index=True).sort_values(
            ["outer_fold", "inner_fold"]
        ).reset_index(drop=True),
        pd.DataFrame(parameter_rows).sort_values(["outer_fold", "class_label"]).reset_index(
            drop=True
        ),
        pd.DataFrame(relationship_rows).sort_values("outer_fold").reset_index(drop=True),
        pd.DataFrame(evaluation_rows).sort_values(
            ["system_id", "sample_index"]
        ).reset_index(drop=True),
    )


def _fold_metrics(
    predictions: pd.DataFrame,
    *,
    labels: Sequence[int],
    task_type: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in SYSTEM_ORDER:
        method_rows = predictions[predictions["system_id"] == method]
        for outer_fold in range(1, REQUIRED_OUTER_FOLDS + 1):
            group = method_rows[method_rows["outer_fold"] == outer_fold].sort_values(
                "sample_index"
            )
            if group.empty:
                raise CalibrationContractError(
                    f"Method {method} outer fold {outer_fold} has no evaluation rows."
                )
            probability = group[_probability_columns(labels)].to_numpy(dtype=np.float64)
            calculated = classification_metrics(
                group["y_true"].to_numpy(dtype=int),
                group["y_pred"].to_numpy(dtype=int),
                probability,
                list(labels),
                task_type=task_type,
            )
            row = {
                **{field: group[field].iloc[0] for field in IDENTITY_FIELDS},
                "method": method,
                "primary_method": method == PRIMARY_METHOD,
                "outer_fold": outer_fold,
                "n_outer_train": len(predictions[predictions["system_id"] == method])
                - len(group),
                "n_outer_test": len(group),
                "selected_candidate_index": int(group["selected_candidate_index"].iloc[0]),
                "source_outer_model_sha256": group["source_outer_model_sha256"].iloc[0],
                "calibrator_contract_sha256": group["calibrator_contract_sha256"].iloc[0],
                "n_decision_changes_from_raw": int(
                    group["decision_changed_from_raw"].astype(bool).sum()
                ),
                "fold_variability_inference": "descriptive_only_not_population_ci",
            }
            row.update({metric: float(calculated[metric]) for metric in metrics})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_calibration_methods(
    fold_metrics: pd.DataFrame,
    metric_intervals: pd.DataFrame,
) -> pd.DataFrame:
    """Combine OOF sample intervals with descriptive fold variability."""

    rows: list[dict[str, Any]] = []
    for method in SYSTEM_ORDER:
        folds = fold_metrics[fold_metrics["method"] == method]
        intervals = metric_intervals[metric_intervals["system_id"] == method]
        if len(folds) != REQUIRED_OUTER_FOLDS or folds["outer_fold"].nunique() != 10:
            raise CalibrationContractError(f"Method {method} requires ten fold rows.")
        row: dict[str, Any] = {
            **{field: folds[field].iloc[0] for field in IDENTITY_FIELDS},
            "method": method,
            "primary_method": method == PRIMARY_METHOD,
            "selection_source": "predeclared_config",
            "selection_performed": False,
            "outer_test_used_for_selection": False,
            "n_folds": REQUIRED_OUTER_FOLDS,
            "n_samples": int(folds["n_outer_test"].sum()),
            "point_estimate_unit": "all_exactly_once_oof_samples",
            "confidence_interval_method": (
                "paired_stratified_sample_level_percentile_bootstrap"
            ),
            "fold_variability_status": "descriptive_only_not_population_ci",
        }
        for metric in METRICS:
            interval = intervals[intervals["metric"] == metric]
            if len(interval) != 1:
                raise CalibrationContractError(
                    f"Method {method} metric {metric} requires one interval row."
                )
            values = folds[metric].astype(float).to_numpy()
            observed = interval.iloc[0]
            row[f"{metric}_oof"] = float(observed["point_estimate"])
            row[f"{metric}_ci_low"] = float(observed["ci_low"])
            row[f"{metric}_ci_high"] = float(observed["ci_high"])
            row[f"{metric}_bootstrap_std"] = float(observed["bootstrap_std"])
            row[f"{metric}_fold_mean"] = float(values.mean())
            row[f"{metric}_fold_std"] = float(values.std(ddof=1))
            row[f"{metric}_fold_min"] = float(values.min())
            row[f"{metric}_fold_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_single_reliability(
    bins: pd.DataFrame,
    class_label: int,
    output_dir: Path,
    *,
    identity: Mapping[str, Any],
) -> Dict[str, Path]:
    fig, axis = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    colors = {"raw": "#7F8C8D", "sigmoid": "#176B87"}
    for method in SYSTEM_ORDER:
        rows = bins[
            (bins["method"] == method)
            & (bins["class_label"] == int(class_label))
            & (bins["n_samples"] > 0)
        ].sort_values("bin")
        axis.plot(
            rows["mean_predicted_probability"],
            rows["observed_frequency"],
            marker="o",
            linewidth=1.5,
            label=method,
            color=colors[method],
        )
    axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="ideal")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed frequency",
        title=f"Reliability for performance class {class_label}",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    description = "; ".join(f"{field}={identity[field]}" for field in IDENTITY_FIELDS)
    png = output_dir / f"reliability_class_{class_label}.png"
    svg = output_dir / f"reliability_class_{class_label}.svg"
    fig.savefig(png, dpi=300, metadata={"Description": description})
    fig.savefig(
        svg,
        format="svg",
        metadata={"Title": f"Class {class_label} reliability", "Description": description},
    )
    plt.close(fig)
    return {"png": png, "svg": svg}


def _metric_panel(
    axis: Any,
    summary: pd.DataFrame,
    metrics: Sequence[str],
    title: str,
) -> None:
    methods = list(SYSTEM_ORDER)
    colors = {"raw": "#7F8C8D", "sigmoid": "#176B87"}
    width = 0.34
    positions = np.arange(len(metrics), dtype=float)
    for method_index, method in enumerate(methods):
        row = summary[summary["method"] == method].iloc[0]
        values = np.asarray([float(row[f"{metric}_oof"]) for metric in metrics])
        lows = np.asarray([float(row[f"{metric}_ci_low"]) for metric in metrics])
        highs = np.asarray([float(row[f"{metric}_ci_high"]) for metric in metrics])
        if (
            not np.all(np.isfinite(values))
            or not np.all(np.isfinite(lows))
            or not np.all(np.isfinite(highs))
            or np.any(lows > highs)
        ):
            raise CalibrationContractError("Calibration figure intervals are invalid.")
        offset = (method_index - 0.5) * width
        x_positions = positions + offset
        axis.bar(
            x_positions,
            values,
            width,
            label=method,
            color=colors[method],
        )
        # Draw percentile endpoints directly.  A valid percentile interval need
        # not contain the observed point, whereas Matplotlib's ``yerr`` API
        # requires non-negative distances from that point.
        axis.vlines(x_positions, lows, highs, color="black", linewidth=1.1)
        cap_width = width * 0.18
        axis.hlines(lows, x_positions - cap_width, x_positions + cap_width, color="black")
        axis.hlines(highs, x_positions - cap_width, x_positions + cap_width, color="black")
    axis.set_xticks(positions, [metric.replace("_", " ") for metric in metrics])
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)


def write_calibration_summary_figure(
    bins: pd.DataFrame,
    summary: pd.DataFrame,
    labels: Sequence[int],
    output_dir: Path,
    *,
    identity: Mapping[str, Any],
) -> Dict[str, Path]:
    """Generate a reproducible calibration summary; final paper numbering is assigned later."""

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.4), constrained_layout=True)
    colors = {"raw": "#7F8C8D", "sigmoid": "#176B87"}
    for axis, label in zip(axes[0], labels):
        for method in SYSTEM_ORDER:
            rows = bins[
                (bins["method"] == method)
                & (bins["class_label"] == int(label))
                & (bins["n_samples"] > 0)
            ].sort_values("bin")
            axis.plot(
                rows["mean_predicted_probability"],
                rows["observed_frequency"],
                marker="o",
                label=method,
                color=colors[method],
            )
        axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Mean predicted probability",
            ylabel="Observed frequency",
            title=f"Class {label} reliability",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    _metric_panel(
        axes[1, 0],
        summary,
        ("ordinal_mae", "severe_error_rate"),
        "Ordinal error (95% sample bootstrap CI)",
    )
    _metric_panel(
        axes[1, 1],
        summary,
        ("nll_log_loss", "multiclass_brier"),
        "Probability loss (95% sample bootstrap CI)",
    )
    _metric_panel(
        axes[1, 2],
        summary,
        ("ece_confidence",),
        "Expected calibration error (95% sample bootstrap CI)",
    )
    fig.suptitle("Predeclared five-fold cross-fitted sigmoid calibration", fontsize=14)
    description = "; ".join(f"{field}={identity[field]}" for field in IDENTITY_FIELDS)
    png = output_dir / "calibration_summary.png"
    svg = output_dir / "calibration_summary.svg"
    fig.savefig(png, dpi=300, metadata={"Description": description})
    fig.savefig(
        svg,
        format="svg",
        metadata={"Title": "Cross-fitted sigmoid calibration", "Description": description},
    )
    plt.close(fig)
    return {"png": png, "svg": svg}


def _write_rationale(
    path: Path,
    *,
    identity: Mapping[str, Any],
    bootstrap_resample_sha256: str,
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Predeclared Sigmoid Calibration Rationale",
                "",
                f"Run ID: `{identity['run_id']}`  ",
                f"Config hash: `{identity['config_hash']}`  ",
                f"Calibration protocol hash: `{identity['calibration_protocol_sha256']}`  ",
                f"Paired-bootstrap resample hash: `{bootstrap_resample_sha256}`",
                "",
                "Sigmoid calibration was selected before outer-test evaluation. Raw probabilities "
                "are retained only as a comparator; no method ranking or threshold selection is "
                "performed.",
                "",
                "For each outer fold, five inner-fold refits generate one cross-fitted probability "
                "row for every outer-training observation. The fold-specific one-vs-rest Platt "
                "sigmoid is fitted only on those rows and is applied to the exact persisted "
                "benchmark model's untouched outer-test probabilities.",
                "",
                "## Probability-Use Warning",
                "",
                "These are uncertain model confidence estimates, not objective employee-performance "
                "probabilities. They must not be converted into autonomous HR thresholds, rankings, "
                "or employment decisions.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_persisted_outputs(
    *,
    paths: Mapping[str, Path],
    staging: Path,
    n_samples: int,
    labels: Sequence[int],
    identity: Mapping[str, Any],
    expected_raw_oof: pd.DataFrame,
    expected_model_hashes: Mapping[int, str],
) -> None:
    csv_expectations = {
        "training_oof_predictions": n_samples * (REQUIRED_OUTER_FOLDS - 1),
        "fit_receipts": REQUIRED_OUTER_FOLDS * REQUIRED_INNER_FOLDS,
        "calibrator_parameters": REQUIRED_OUTER_FOLDS * len(tuple(labels)),
        "calibrator_model_relationships": REQUIRED_OUTER_FOLDS,
        "predictions": n_samples * len(SYSTEM_ORDER),
        "fold_metrics": REQUIRED_OUTER_FOLDS * len(SYSTEM_ORDER),
        "bins": len(SYSTEM_ORDER) * len(tuple(labels)) * 10,
        "method_comparison": len(SYSTEM_ORDER),
        "uncertainty": len(SYSTEM_ORDER) * len(METRICS),
        "paired_differences": len(METRICS),
    }
    loaded: dict[str, pd.DataFrame] = {}
    for key, expected_rows in csv_expectations.items():
        path = paths[key]
        if not path.is_file() or path.stat().st_size <= 0:
            raise CalibrationContractError(f"Required calibration output is missing/empty: {key}.")
        # Exact persisted-probability and calibrator-parameter replay requires
        # pandas' round-trip float parser; the default fast parser can differ by
        # one ULP from values just written by DataFrame.to_csv.
        frame = pd.read_csv(path, float_precision="round_trip")
        if len(frame) != expected_rows:
            raise CalibrationContractError(
                f"Calibration output {key} has {len(frame)} rows; expected {expected_rows}."
            )
        for field in IDENTITY_FIELDS:
            if field not in frame.columns or set(frame[field].astype(str)) != {str(identity[field])}:
                raise CalibrationContractError(f"Calibration output {key} identity {field} drifted.")
        loaded[key] = frame
    predictions = loaded["predictions"]
    for method in SYSTEM_ORDER:
        rows = predictions[predictions["system_id"] == method]
        if len(rows) != n_samples or rows["sample_index"].duplicated().any():
            raise CalibrationContractError(f"Method {method} is not exactly-once OOF.")
    raw = predictions[predictions["system_id"] == "raw"].sort_values(
        "sample_index"
    ).reset_index(drop=True)
    expected_raw = expected_raw_oof.sort_values("sample_index").reset_index(drop=True)
    exact_columns = [
        "sample_index",
        "outer_fold",
        "y_true",
        "y_pred",
        "selected_candidate_index",
    ]
    if not raw[exact_columns].astype(int).equals(expected_raw[exact_columns].astype(int)):
        raise CalibrationContractError(
            "Persisted raw calibration rows differ from exact benchmark OOF membership or labels."
        )
    probability_columns = _probability_columns(labels)
    if not np.array_equal(
        raw[probability_columns].to_numpy(dtype=np.float64),
        expected_raw[probability_columns].to_numpy(dtype=np.float64),
    ):
        raise CalibrationContractError(
            "Persisted raw calibration probabilities are not bit-exact benchmark OOF values."
        )
    expected_hash_series = raw["outer_fold"].astype(int).map(
        {int(key): str(value) for key, value in expected_model_hashes.items()}
    )
    if expected_hash_series.isna().any() or not raw["source_outer_model_sha256"].astype(
        str
    ).equals(expected_hash_series.astype(str)):
        raise CalibrationContractError(
            "Persisted raw rows are not bound to their exact benchmark fold models."
        )
    bins = loaded["bins"]
    bin_counts = bins.groupby(["method", "class_label"])["n_samples"].sum()
    if not (bin_counts == n_samples).all():
        raise CalibrationContractError("Reliability-bin denominators do not cover all OOF cases.")
    uncertainty = loaded["uncertainty"]
    paired = loaded["paired_differences"]
    for frame in (uncertainty, paired):
        if not (
            (frame["n_samples"] == n_samples).all()
            and (frame["n_resamples"] == REQUIRED_BOOTSTRAP_RESAMPLES).all()
            and (frame["n_valid"] == REQUIRED_BOOTSTRAP_RESAMPLES).all()
        ):
            raise CalibrationContractError("Calibration bootstrap denominators are incomplete.")
    parameters = loaded["calibrator_parameters"]
    for outer_fold in range(1, REQUIRED_OUTER_FOLDS + 1):
        calibrator = calibrator_from_parameter_rows(
            parameters[parameters["outer_fold"] == outer_fold]
        )
        raw = predictions[
            (predictions["system_id"] == "raw")
            & (predictions["outer_fold"] == outer_fold)
        ].sort_values("sample_index")
        sigmoid = predictions[
            (predictions["system_id"] == "sigmoid")
            & (predictions["outer_fold"] == outer_fold)
        ].sort_values("sample_index")
        replay = calibrator.transform(raw[probability_columns].to_numpy(dtype=float))
        if not np.allclose(
            replay,
            sigmoid[probability_columns].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise CalibrationContractError(
                f"Outer fold {outer_fold} persisted sigmoid probabilities do not replay."
            )
    for key, path in paths.items():
        if key == "metadata":
            continue
        if not path.is_file() or path.stat().st_size <= 0:
            raise CalibrationContractError(f"Calibration output {key} is missing or empty.")
        try:
            path.relative_to(staging)
        except ValueError as exc:
            raise CalibrationContractError(f"Calibration output {key} escapes staging.") from exc


def _revalidate_upstreams_before_publish(
    *,
    config_path: str | Path,
    shared_folds_dir: str | Path,
    model_benchmarks_dir: str | Path,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    dataset_sha256: str,
    features: pd.DataFrame,
    target: pd.Series,
    labels: Sequence[int],
    original_bundle: XGBoostOOFArtifacts,
) -> None:
    """Fail if any scientific upstream changes during the long calibration run."""

    try:
        refreshed_config = load_manuscript_config(config_path)
        if canonical_config_hash(refreshed_config) != config_hash:
            raise CalibrationContractError("Canonical config changed during calibration.")
        refreshed_dataset = load_canonical_dataset(config_path, "inx_primary")
        if str(refreshed_dataset.receipt.get("actual_sha256")) != dataset_sha256:
            raise CalibrationContractError("Canonical dataset changed during calibration.")
        refreshed_bundle = read_xgboost_oof_artifacts(
            shared_folds_dir,
            model_benchmarks_dir,
            expected_run_id=run_id,
            expected_config_hash=config_hash,
            expected_scientific_input_hash=scientific_input_hash,
            expected_feature_columns=features.columns.tolist(),
            expected_labels=labels,
        )
        validate_xgboost_oof_replay(
            refreshed_bundle,
            features,
            target,
            labels=labels,
            probability_atol=1e-12,
        )
    except CalibrationContractError:
        raise
    except Exception as exc:
        raise CalibrationContractError(
            f"Scientific upstream revalidation failed before publication: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if (
        refreshed_bundle.identity != original_bundle.identity
        or refreshed_bundle.model_set_sha256 != original_bundle.model_set_sha256
        or dict(refreshed_bundle.upstream_file_hashes)
        != dict(original_bundle.upstream_file_hashes)
    ):
        raise CalibrationContractError(
            "Benchmark/shared-fold evidence changed during calibration; publication is blocked."
        )


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    shared_folds_dir: str | Path,
    model_benchmarks_dir: str | Path,
    output_dir: str | Path,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
) -> Dict[str, Path]:
    """Generate the complete cross-fitted sigmoid calibration evidence package."""

    raw_config = load_manuscript_config(config_path)
    settings = raw_config["manuscript_final"]
    observed_config_hash = canonical_config_hash(raw_config)
    if str(config_hash) != observed_config_hash:
        raise CalibrationContractError("Supplied config_hash differs from canonical config.")
    _require_sha256("config_hash", config_hash)
    _require_sha256("scientific_input_hash", scientific_input_hash)
    if not isinstance(run_id, str) or not run_id.strip():
        raise CalibrationContractError("run_id must be a non-blank string.")
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise CalibrationContractError(
            f"Calibration output must be absent or an empty builder-owned directory: {output}"
        )

    protocol = validate_calibration_protocol(settings)
    target_config = settings["target"]
    target_column = str(target_config["column"])
    labels = tuple(int(value) for value in target_config["labels"])
    task_type = str(target_config["problem_type"])
    if labels != (2, 3, 4) or task_type != PRIMARY_TASK:
        raise CalibrationContractError("Calibration requires the canonical 2/3/4 ordinal task.")
    identifier_fields = settings["governance_fields"]["identifier_fields"]
    if not isinstance(identifier_fields, list) or len(identifier_fields) != 1:
        raise CalibrationContractError("Calibration requires one canonical identifier field.")
    id_column = str(identifier_fields[0])
    primary_policy = str(settings["feature_policies"]["primary_policy"])
    if primary_policy != "no_salary_hike_no_attrition_no_department":
        raise CalibrationContractError("Canonical calibration primary policy changed.")
    definition = settings["feature_policies"]["definitions"][primary_policy]
    metrics = _configured_metrics(settings, task_type)
    bootstrap_protocol = _bootstrap_protocol(settings)
    model_seed = _resolve_seed(settings, "model")
    calibration_seed = _resolve_seed(settings, "calibration")

    canonical = load_canonical_dataset(config_path, "inx_primary")
    data = canonical.frame
    if not data.index.is_unique:
        raise CalibrationContractError("Canonical calibration data index must be unique.")
    integer_index = pd.Index([int(value) for value in data.index])
    if not integer_index.equals(pd.Index(data.index)):
        data = data.copy()
        data.index = integer_index
    features, excluded = exact_policy_frame(
        data,
        primary_policy,
        definition,
        target_column=target_column,
        id_column=id_column,
    )
    if tuple(excluded) != tuple(primary_excluded_features(raw_config)):
        raise CalibrationContractError("Calibration primary exclusions drifted from canonical policy.")
    target = data[target_column].astype(int)
    if set(target.unique()) != set(labels):
        raise CalibrationContractError("Calibration target support differs from canonical labels.")
    dataset_sha256 = _require_sha256(
        "dataset_sha256", canonical.receipt.get("actual_sha256")
    )

    try:
        bundle = read_xgboost_oof_artifacts(
            shared_folds_dir,
            model_benchmarks_dir,
            expected_run_id=run_id,
            expected_config_hash=config_hash,
            expected_scientific_input_hash=scientific_input_hash,
            expected_feature_columns=features.columns.tolist(),
            expected_labels=labels,
        )
        validate_xgboost_oof_replay(
            bundle,
            features,
            target,
            labels=labels,
            probability_atol=1e-12,
        )
    except BenchmarkArtifactContractError as exc:
        raise CalibrationContractError(f"Benchmark evidence is incompatible: {exc}") from exc
    if (
        int(bundle.folds.contract.get("outer_splits", -1)) != REQUIRED_OUTER_FOLDS
        or int(bundle.folds.contract.get("inner_splits", -1)) != REQUIRED_INNER_FOLDS
        or str(bundle.folds.contract.get("dataset_sha256")) != dataset_sha256
    ):
        raise CalibrationContractError("Calibration shared-fold contract is not canonical 10x5 INX.")

    protocol_sha256 = _calibration_protocol_sha256(
        protocol=protocol,
        labels=labels,
        task_type=task_type,
        policy=primary_policy,
        excluded_features=excluded,
        feature_columns=features.columns,
        metrics=metrics,
        bootstrap=bootstrap_protocol,
        model_seed=model_seed,
        calibration_seed=calibration_seed,
    )
    identity: dict[str, Any] = {
        "run_id": run_id,
        "config_hash": config_hash,
        "scientific_input_hash": scientific_input_hash,
        "fold_contract_hash": bundle.identity.fold_contract_hash,
        "xgboost_model_set_sha256": bundle.model_set_sha256,
        "dataset_sha256": dataset_sha256,
        "calibration_protocol_sha256": protocol_sha256,
    }
    for field in IDENTITY_FIELDS[1:]:
        _require_sha256(field, identity[field])

    training_oof, fit_receipts, calibrator_parameters, relationships, predictions = (
        _calibration_evidence(
            features=features,
            target=target,
            bundle=bundle,
            forbidden_features=excluded,
            model_seed=model_seed,
            calibration_seed=calibration_seed,
            sigmoid_settings=protocol["sigmoid"],
            primary_policy=primary_policy,
            identity=identity,
        )
    )
    try:
        validate_consumer_fold_assignments(
            bundle.folds, predictions, group_columns=("system_id",)
        )
        alignment = validate_aligned_oof_predictions(
            predictions,
            labels=labels,
            task_type=task_type,
            metrics=metrics,
        )
    except (SharedFoldContractError, OOFBootstrapError) as exc:
        raise CalibrationContractError(f"Calibration OOF alignment failed: {exc}") from exc
    if alignment["n_systems"] != 2 or alignment["n_samples"] != len(data):
        raise CalibrationContractError("Calibration evaluation denominators are incomplete.")

    fold_metrics = _fold_metrics(
        predictions,
        labels=labels,
        task_type=task_type,
        metrics=metrics,
    )
    try:
        bootstrap: BootstrapResult = compute_paired_oof_bootstrap(
            predictions,
            labels=labels,
            task_type=task_type,
            metrics=metrics,
            comparisons=(
                ComparisonSpec(
                    comparison_id="sigmoid_minus_raw",
                    system_a="sigmoid",
                    system_b="raw",
                    primary_gate=False,
                ),
            ),
            primary_metric=None,
            protocol=bootstrap_protocol,
            n_bins=int(protocol["n_bins"]),
        )
    except OOFBootstrapError as exc:
        raise CalibrationContractError(f"Calibration bootstrap failed: {exc}") from exc
    resample_hash = _require_sha256(
        "calibration resample_hash", bootstrap.metadata.get("resample_hash")
    )
    if (
        resample_hash != bundle.baseline_gate.get("resample_hash")
        or int(bootstrap.metadata.get("n_resamples", -1)) != REQUIRED_BOOTSTRAP_RESAMPLES
        or bootstrap.metadata.get("primary_metric") is not None
    ):
        raise CalibrationContractError(
            "Calibration bootstrap does not match the benchmark paired resample contract."
        )
    uncertainty = bootstrap.metric_intervals.copy()
    paired = bootstrap.paired_differences.copy()
    for frame in (uncertainty, paired):
        for field, value in reversed(list(identity.items())):
            frame.insert(0, field, value)
    method_comparison = summarize_calibration_methods(fold_metrics, uncertainty)

    bin_rows: list[dict[str, Any]] = []
    probability_columns = _probability_columns(labels)
    for method in SYSTEM_ORDER:
        method_rows = predictions[predictions["system_id"] == method].sort_values(
            "sample_index"
        )
        bin_rows.extend(
            calibration_bin_rows(
                method_rows["y_true"].to_numpy(dtype=int),
                method_rows[probability_columns].to_numpy(dtype=np.float64),
                labels,
                run_id=run_id,
                config_hash=config_hash,
                method=method,
                n_bins=int(protocol["n_bins"]),
                identity=identity,
            )
        )
    bins = pd.DataFrame(bin_rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent)
    staging = Path(temporary.name)
    figures = staging / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {
        "training_oof_predictions": staging / "calibration_training_oof_predictions.csv",
        "fit_receipts": staging / "calibration_fit_receipts.csv",
        "calibrator_parameters": staging / "sigmoid_calibrator_parameters.csv",
        "calibrator_model_relationships": staging / "calibrator_model_relationships.csv",
        "predictions": staging / "calibration_predictions.csv",
        "fold_metrics": staging / "calibration_fold_metrics.csv",
        "bins": staging / "calibration_bins.csv",
        "method_comparison": staging / "calibration_method_comparison.csv",
        "uncertainty": staging / "calibration_metric_intervals.csv",
        "paired_differences": staging / "calibration_paired_differences.csv",
        "bootstrap_metadata": staging / "bootstrap_metadata.json",
        "protocol": staging / "calibration_protocol.json",
        "rationale": staging / "predeclared_method_rationale.md",
        "figure_source": staging / "calibration_figure_source.json",
        "metadata": staging / "calibration_metadata.json",
    }
    try:
        training_oof.to_csv(paths["training_oof_predictions"], index=False)
        fit_receipts.to_csv(paths["fit_receipts"], index=False)
        calibrator_parameters.to_csv(paths["calibrator_parameters"], index=False)
        relationships.to_csv(paths["calibrator_model_relationships"], index=False)
        predictions.to_csv(paths["predictions"], index=False)
        fold_metrics.to_csv(paths["fold_metrics"], index=False)
        bins.to_csv(paths["bins"], index=False)
        method_comparison.to_csv(paths["method_comparison"], index=False)
        uncertainty.to_csv(paths["uncertainty"], index=False)
        paired.to_csv(paths["paired_differences"], index=False)
        write_json(
            paths["bootstrap_metadata"],
            {
                **dict(bootstrap.metadata),
                **identity,
                "comparison": "sigmoid_minus_raw",
                "primary_gate_applicable": False,
                "multiplicity_adjustment": "none",
            },
        )
        write_json(
            paths["protocol"],
            {
                "stage": "sigmoid_calibration",
                "status": "complete",
                **identity,
                "primary_policy": primary_policy,
                "labels": list(labels),
                "task_type": task_type,
                "protocol": protocol,
                "primary_method": PRIMARY_METHOD,
                "raw_comparator": True,
                "selection_performed": False,
                "outer_test_used_for_fit_or_selection": False,
                "expected_inner_model_fits": 50,
                "expected_fold_calibrators": 10,
            },
        )
        _write_rationale(
            paths["rationale"],
            identity=identity,
            bootstrap_resample_sha256=resample_hash,
        )
        for label in labels:
            outputs = _plot_single_reliability(
                bins,
                label,
                figures,
                identity=identity,
            )
            paths[f"reliability_class_{label}_png"] = outputs["png"]
            paths[f"reliability_class_{label}_svg"] = outputs["svg"]
        summary_figure = write_calibration_summary_figure(
            bins,
            method_comparison,
            labels,
            staging,
            identity=identity,
        )
        paths["calibration_summary_png"] = summary_figure["png"]
        paths["calibration_summary_svg"] = summary_figure["svg"]
        write_json(
            paths["figure_source"],
            {
                **identity,
                "final_manuscript_figure_number_assigned": False,
                "method_order": list(SYSTEM_ORDER),
                "primary_method": PRIMARY_METHOD,
                "class_order": list(labels),
                "n_bins": 10,
                "panel_metric_order": [
                    "ordinal_mae",
                    "severe_error_rate",
                    "nll_log_loss",
                    "multiclass_brier",
                    "ece_confidence",
                ],
                "sources": {
                    "reliability_bins": {
                        "path": paths["bins"].relative_to(staging).as_posix(),
                        "sha256": sha256_file(paths["bins"]),
                    },
                    "metric_intervals": {
                        "path": paths["uncertainty"].relative_to(staging).as_posix(),
                        "sha256": sha256_file(paths["uncertainty"]),
                    },
                    "paired_differences": {
                        "path": paths["paired_differences"].relative_to(staging).as_posix(),
                        "sha256": sha256_file(paths["paired_differences"]),
                    },
                },
                "caption_warning": protocol["probability_warning"],
            },
        )
        _validate_persisted_outputs(
            paths=paths,
            staging=staging,
            n_samples=len(data),
            labels=labels,
            identity=identity,
            expected_raw_oof=bundle.oof_predictions,
            expected_model_hashes={
                fold: model.sha256 for fold, model in bundle.fold_models.items()
            },
        )
        paths["validation"] = staging / "calibration_validation.json"
        write_json(
            paths["validation"],
            {
                "stage": "sigmoid_calibration",
                "status": "validated_complete",
                **identity,
                "counts": {
                    "inner_model_fits": len(fit_receipts),
                    "crossfit_training_rows": len(training_oof),
                    "fold_calibrators": relationships["outer_fold"].nunique(),
                    "calibrator_parameter_rows": len(calibrator_parameters),
                    "outer_oof_cases": len(data),
                    "evaluation_methods": len(SYSTEM_ORDER),
                    "evaluation_rows": len(predictions),
                    "bootstrap_resamples": REQUIRED_BOOTSTRAP_RESAMPLES,
                },
                "raw_benchmark_probability_source": "exact_replayed_model_benchmarks_oof",
                "sigmoid_parameter_replay_atol": 1e-12,
                "outer_model_refits": 0,
                "outer_test_used_for_fit_or_selection": False,
                "method_selection_performed": False,
                "threshold_selection_performed": False,
            },
        )
        if paths["validation"].stat().st_size <= 0:
            raise CalibrationContractError("Calibration validation receipt is empty.")
        output_records = {
            key: {
                "path": value.relative_to(staging).as_posix(),
                "sha256": sha256_file(value),
                "size_bytes": value.stat().st_size,
            }
            for key, value in paths.items()
            if key != "metadata"
        }
        write_json(
            paths["metadata"],
            {
                "stage": "sigmoid_calibration",
                "status": "complete",
                **identity,
                "primary_policy": primary_policy,
                "excluded_features": list(excluded),
                "feature_columns": list(features.columns),
                "labels": list(labels),
                "task_type": task_type,
                "metrics": list(metrics),
                "primary_method": PRIMARY_METHOD,
                "selection_performed": False,
                "outer_test_used_for_tuning_fitting_selection_or_thresholds": False,
                "outer_model_refit_in_calibration_stage": False,
                "bootstrap_resample_sha256": resample_hash,
                "upstream_file_hashes": dict(sorted(bundle.upstream_file_hashes.items())),
                "outputs": output_records,
                "probability_warning": protocol["probability_warning"],
                "claim_boundaries": [
                    "calibration is evaluated on fixed outer-test folds and does not prove future calibration",
                    "probabilities are uncertain model confidence estimates, not objective employee probabilities",
                    "no calibration threshold was selected or validated for HR decisions",
                    "research evidence must not drive autonomous HR decisions",
                ],
            },
        )
        _revalidate_upstreams_before_publish(
            config_path=config_path,
            shared_folds_dir=shared_folds_dir,
            model_benchmarks_dir=model_benchmarks_dir,
            run_id=run_id,
            config_hash=config_hash,
            scientific_input_hash=scientific_input_hash,
            dataset_sha256=dataset_sha256,
            features=features,
            target=target,
            labels=labels,
            original_bundle=bundle,
        )
        relative_paths = {key: path.relative_to(staging) for key, path in paths.items()}
        if output.exists():
            output.rmdir()
        atomic_replace_directory(staging, output)
        cleanup_temporary_directory(temporary)
    except Exception as error:
        cleanup_temporary_directory(temporary, primary_error=error)
        raise
    return {key: output / relative for key, relative in relative_paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predeclared five-inner-fold cross-fitted sigmoid calibration."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--shared-folds-dir", required=True)
    parser.add_argument("--model-benchmarks-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--scientific-input-hash", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            {
                key: str(value)
                for key, value in run(
                    arguments.config,
                    shared_folds_dir=arguments.shared_folds_dir,
                    model_benchmarks_dir=arguments.model_benchmarks_dir,
                    output_dir=arguments.output_dir,
                    run_id=arguments.run_id,
                    config_hash=arguments.config_hash,
                    scientific_input_hash=arguments.scientific_input_hash,
                ).items()
            },
            indent=2,
            sort_keys=True,
        )
    )


__all__ = [
    "CalibrationContractError",
    "SigmoidCalibrator",
    "SigmoidClassParameters",
    "apply_sigmoid_calibrator",
    "calibration_bin_rows",
    "calibrator_from_parameter_rows",
    "cross_fit_outer_training",
    "fit_sigmoid_calibrator",
    "run",
    "summarize_calibration_methods",
    "validate_calibration_protocol",
    "write_calibration_summary_figure",
]
