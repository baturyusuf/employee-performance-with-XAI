"""Deterministic paired sample-level bootstrap for aligned OOF predictions.

This module is deliberately independent of report roots and historical
artifacts.  It treats the observed samples (conditional on one fixed outer-fold
assignment) as the bootstrap units, preserves every outer-fold/target stratum,
and uses the same resampled positions for all compared systems.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from src.models.evaluate import classification_metrics
from src.models.task_schema import get_task_schema


class OOFBootstrapError(RuntimeError):
    """Raised when OOF inputs or bootstrap results violate the contract."""


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    better_direction: Literal["higher", "lower"]
    lower_bound: float
    upper_bound: float
    requires_probabilities: bool = False


_UNIT_HIGHER = {
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "weighted_precision",
    "macro_recall",
    "weighted_recall",
    "adjacent_accuracy",
}

METRIC_DEFINITIONS: Mapping[str, MetricDefinition] = MappingProxyType(
    {
        **{
            name: MetricDefinition(name, "higher", 0.0, 1.0)
            for name in sorted(_UNIT_HIGHER)
        },
        "quadratic_weighted_kappa": MetricDefinition(
            "quadratic_weighted_kappa", "higher", -1.0, 1.0
        ),
        "ordinal_mae": MetricDefinition("ordinal_mae", "lower", 0.0, 2.0),
        "severe_error_rate": MetricDefinition("severe_error_rate", "lower", 0.0, 1.0),
        "nll_log_loss": MetricDefinition(
            "nll_log_loss", "lower", 0.0, math.inf, requires_probabilities=True
        ),
        # The repository definition is mean(sum((p_k-y_k)^2)); it is not
        # divided by the number of classes and therefore has range [0, 2].
        "multiclass_brier": MetricDefinition(
            "multiclass_brier", "lower", 0.0, 2.0, requires_probabilities=True
        ),
        "binary_brier": MetricDefinition(
            "binary_brier", "lower", 0.0, 1.0, requires_probabilities=True
        ),
        "ece_confidence": MetricDefinition(
            "ece_confidence", "lower", 0.0, 1.0, requires_probabilities=True
        ),
        "roc_auc": MetricDefinition(
            "roc_auc", "higher", 0.0, 1.0, requires_probabilities=True
        ),
        "average_precision": MetricDefinition(
            "average_precision", "higher", 0.0, 1.0, requires_probabilities=True
        ),
    }
)


@dataclass(frozen=True)
class BootstrapProtocol:
    n_resamples: int = 5000
    confidence_level: float = 0.95
    seed: int = 42
    strata_columns: tuple[str, ...] = ("outer_fold", "y_true")
    method: str = "paired_stratified_percentile"
    quantile_method: str = "linear"

    def __post_init__(self) -> None:
        if not isinstance(self.n_resamples, int) or self.n_resamples < 2:
            raise OOFBootstrapError("n_resamples must be an integer of at least two.")
        if not 0.0 < float(self.confidence_level) < 1.0:
            raise OOFBootstrapError("confidence_level must lie strictly between zero and one.")
        if not isinstance(self.seed, int):
            raise OOFBootstrapError("seed must be an integer.")
        if not self.strata_columns or "y_true" not in self.strata_columns:
            raise OOFBootstrapError("strata_columns must be non-empty and include y_true.")
        if len(set(self.strata_columns)) != len(self.strata_columns):
            raise OOFBootstrapError("strata_columns must not contain duplicates.")
        if self.method != "paired_stratified_percentile":
            raise OOFBootstrapError("Only paired_stratified_percentile is supported.")
        if self.quantile_method != "linear":
            raise OOFBootstrapError("The predeclared percentile quantile method is linear.")


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_id: str
    system_a: str
    system_b: str
    primary_gate: bool = False

    def __post_init__(self) -> None:
        if not self.comparison_id.strip():
            raise OOFBootstrapError("comparison_id must be non-empty.")
        if not self.system_a.strip() or not self.system_b.strip():
            raise OOFBootstrapError("Comparison systems must be non-empty.")
        if self.system_a == self.system_b:
            raise OOFBootstrapError("A paired comparison requires two distinct system IDs.")


@dataclass(frozen=True)
class ResamplePlan:
    sorted_sample_ids: tuple[Any, ...]
    indices: np.ndarray
    resample_hash: str
    stratum_counts: Mapping[str, int]


@dataclass(frozen=True)
class BootstrapResult:
    metric_intervals: pd.DataFrame
    paired_differences: pd.DataFrame
    metadata: Mapping[str, Any]
    resample_plan: ResamplePlan


def metric_definition(metric: str) -> MetricDefinition:
    try:
        return METRIC_DEFINITIONS[metric]
    except KeyError as exc:
        raise OOFBootstrapError(
            f"Metric {metric!r} has no uncertainty direction/domain contract; "
            f"allowed={sorted(METRIC_DEFINITIONS)}."
        ) from exc


def validate_metric_value(metric: str, value: Any, *, tolerance: float = 1e-12) -> float:
    definition = metric_definition(metric)
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise OOFBootstrapError(f"Metric {metric!r} is not numeric: {value!r}.") from exc
    if not math.isfinite(observed):
        raise OOFBootstrapError(f"Metric {metric!r} is non-finite: {observed!r}.")
    if observed < definition.lower_bound - tolerance:
        raise OOFBootstrapError(
            f"Metric {metric!r}={observed} is below its domain lower bound {definition.lower_bound}."
        )
    if math.isfinite(definition.upper_bound) and observed > definition.upper_bound + tolerance:
        raise OOFBootstrapError(
            f"Metric {metric!r}={observed} is above its domain upper bound {definition.upper_bound}."
        )
    return observed


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sample_sort_key(value: Any) -> tuple[int, Any]:
    """Natural deterministic ordering without conflating unlike ID types."""

    if isinstance(value, (bool, np.bool_)):
        return 0, int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 1, float(value)
    if isinstance(value, str):
        return 2, value
    if isinstance(value, pd.Timestamp):
        return 3, value.isoformat()
    return 4, f"{type(value).__name__}:{value}"


def generate_stratified_resample_indices(
    base_samples: pd.DataFrame,
    protocol: BootstrapProtocol = BootstrapProtocol(),
    *,
    sample_id_column: str = "sample_index",
) -> ResamplePlan:
    """Create indices into sample-ID-sorted rows, preserving every stratum size."""

    required = {sample_id_column, *protocol.strata_columns}
    missing = sorted(required.difference(base_samples.columns))
    if missing:
        raise OOFBootstrapError(f"Base samples are missing resampling columns: {missing}.")
    if base_samples.empty:
        raise OOFBootstrapError("Cannot bootstrap an empty OOF sample table.")
    if base_samples[list(required)].isna().any().any():
        raise OOFBootstrapError("Sample IDs and bootstrap strata must not contain missing values.")
    if base_samples[sample_id_column].duplicated().any():
        duplicates = base_samples.loc[
            base_samples[sample_id_column].duplicated(keep=False), sample_id_column
        ].tolist()
        raise OOFBootstrapError(f"Base sample IDs must be unique; duplicates={duplicates[:10]}.")

    ordered_ids = sorted(base_samples[sample_id_column].tolist(), key=_sample_sort_key)
    ordered = base_samples.set_index(sample_id_column, drop=False).loc[ordered_ids].reset_index(drop=True)
    grouped = ordered.groupby(list(protocol.strata_columns), sort=True, dropna=False).indices
    if not grouped:
        raise OOFBootstrapError("No bootstrap strata were produced.")
    stratum_positions = [np.asarray(grouped[key], dtype=np.int64) for key in grouped]
    if any(len(positions) == 0 for positions in stratum_positions):
        raise OOFBootstrapError("Bootstrap strata must be non-empty.")

    rng = np.random.default_rng(protocol.seed)
    indices = np.empty((protocol.n_resamples, len(ordered)), dtype=np.int64)
    for iteration in range(protocol.n_resamples):
        offset = 0
        for positions in stratum_positions:
            drawn = rng.choice(positions, size=len(positions), replace=True)
            indices[iteration, offset : offset + len(drawn)] = drawn
            offset += len(drawn)

    hash_payload = {
        "sample_ids": [_json_scalar(value) for value in ordered_ids],
        "n_resamples": protocol.n_resamples,
        "confidence_level": protocol.confidence_level,
        "seed": protocol.seed,
        "strata_columns": list(protocol.strata_columns),
        "method": protocol.method,
        "quantile_method": protocol.quantile_method,
    }
    digest = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + indices.tobytes(order="C")
    ).hexdigest()
    counts = MappingProxyType(
        {
            json.dumps([_json_scalar(value) for value in (key if isinstance(key, tuple) else (key,))]): int(
                len(grouped[key])
            )
            for key in grouped
        }
    )
    indices.setflags(write=False)
    return ResamplePlan(
        sorted_sample_ids=tuple(ordered_ids),
        indices=indices,
        resample_hash=digest,
        stratum_counts=counts,
    )


def _validate_requested_metrics(metrics: Sequence[str], task_type: str) -> tuple[str, ...]:
    if not metrics:
        raise OOFBootstrapError("At least one metric is required.")
    if len(set(metrics)) != len(metrics):
        raise OOFBootstrapError("Requested metrics must not contain duplicates.")
    schema = get_task_schema(task_type)
    validated: list[str] = []
    for metric in metrics:
        metric_definition(metric)
        try:
            applicable = schema.is_metric_applicable(metric)
        except ValueError as exc:
            raise OOFBootstrapError(str(exc)) from exc
        if not applicable:
            raise OOFBootstrapError(f"Metric {metric!r} is inapplicable to task {schema.name!r}.")
        validated.append(metric)
    return tuple(validated)


def _aligned_system_frames(
    predictions: pd.DataFrame,
    *,
    labels: Sequence[int],
    system_column: str,
    sample_id_column: str,
    fold_column: str,
    y_true_column: str,
    y_pred_column: str,
    probability_columns: Mapping[int, str],
) -> tuple[pd.DataFrame, Mapping[str, pd.DataFrame]]:
    required = {
        system_column,
        sample_id_column,
        fold_column,
        y_true_column,
        y_pred_column,
        *probability_columns.values(),
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise OOFBootstrapError(f"OOF predictions are missing required columns: {missing}.")
    if predictions.empty:
        raise OOFBootstrapError("OOF predictions must not be empty.")
    if predictions[list(required)].isna().any().any():
        raise OOFBootstrapError("Required OOF identity, label and probability fields must not be missing.")

    systems = predictions[system_column].astype(str)
    if systems.str.strip().eq("").any():
        raise OOFBootstrapError("OOF system IDs must be non-empty.")
    frame = predictions.copy()
    frame[system_column] = systems
    system_ids = sorted(frame[system_column].unique().tolist())
    if not system_ids:
        raise OOFBootstrapError("No OOF systems were found.")

    label_values = tuple(int(label) for label in labels)
    if len(label_values) < 2 or len(set(label_values)) != len(label_values):
        raise OOFBootstrapError("labels must contain at least two unique values.")
    if set(probability_columns) != set(label_values):
        raise OOFBootstrapError(
            "Probability-column labels must match the declared labels exactly: "
            f"labels={label_values}, probability_labels={tuple(probability_columns)}."
        )

    aligned: dict[str, pd.DataFrame] = {}
    reference_ids: tuple[Any, ...] | None = None
    reference_truth: pd.DataFrame | None = None
    for system_id in system_ids:
        group = frame[frame[system_column] == system_id].copy()
        if group[sample_id_column].duplicated().any():
            duplicates = group.loc[
                group[sample_id_column].duplicated(keep=False), sample_id_column
            ].tolist()
            raise OOFBootstrapError(
                f"System {system_id!r} has duplicate OOF sample IDs: {duplicates[:10]}."
            )
        ordered_ids = tuple(sorted(group[sample_id_column].tolist(), key=_sample_sort_key))
        if reference_ids is None:
            reference_ids = ordered_ids
        elif ordered_ids != reference_ids:
            missing_ids = sorted(set(reference_ids).difference(ordered_ids), key=_sample_sort_key)
            extra_ids = sorted(set(ordered_ids).difference(reference_ids), key=_sample_sort_key)
            raise OOFBootstrapError(
                f"OOF sample coverage differs for system {system_id!r}: "
                f"missing={missing_ids[:10]}, extra={extra_ids[:10]}."
            )
        ordered = group.set_index(sample_id_column, drop=False).loc[list(reference_ids)].reset_index(drop=True)
        try:
            y_true_numeric = pd.to_numeric(ordered[y_true_column], errors="raise").to_numpy(dtype=float)
            y_pred_numeric = pd.to_numeric(ordered[y_pred_column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise OOFBootstrapError(f"System {system_id!r} contains non-integer labels.") from exc
        if (
            not np.isfinite(y_true_numeric).all()
            or not np.isfinite(y_pred_numeric).all()
            or not np.equal(y_true_numeric, np.round(y_true_numeric)).all()
            or not np.equal(y_pred_numeric, np.round(y_pred_numeric)).all()
        ):
            raise OOFBootstrapError(f"System {system_id!r} contains non-integer labels.")
        y_true = pd.Series(y_true_numeric.astype(int), index=ordered.index)
        y_pred = pd.Series(y_pred_numeric.astype(int), index=ordered.index)
        if set(y_true.unique()) != set(label_values):
            raise OOFBootstrapError(
                f"System {system_id!r} true-label support does not equal declared labels: "
                f"observed={sorted(y_true.unique())}, labels={sorted(label_values)}."
            )
        if not set(y_pred.unique()).issubset(label_values):
            raise OOFBootstrapError(f"System {system_id!r} predicts labels outside {label_values}.")
        ordered[y_true_column] = y_true
        ordered[y_pred_column] = y_pred

        proba_names = [probability_columns[label] for label in label_values]
        try:
            proba = ordered[proba_names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise OOFBootstrapError(f"System {system_id!r} has non-numeric probabilities.") from exc
        if not np.isfinite(proba).all():
            raise OOFBootstrapError(f"System {system_id!r} has non-finite probabilities.")
        if np.any(proba < -1e-12) or np.any(proba > 1.0 + 1e-12):
            raise OOFBootstrapError(f"System {system_id!r} has probabilities outside [0,1].")
        if not np.allclose(proba.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
            raise OOFBootstrapError(f"System {system_id!r} probability rows do not sum to one.")

        identity = ordered[[sample_id_column, fold_column, y_true_column]].copy()
        if reference_truth is None:
            reference_truth = identity
        else:
            for column in (fold_column, y_true_column):
                mismatch = identity[column].to_numpy() != reference_truth[column].to_numpy()
                if bool(np.any(mismatch)):
                    samples = identity.loc[mismatch, sample_id_column].tolist()
                    raise OOFBootstrapError(
                        f"OOF {column} differs across systems for samples {samples[:10]}."
                    )
        aligned[system_id] = ordered

    assert reference_truth is not None
    return reference_truth, MappingProxyType(aligned)


def validate_aligned_oof_predictions(
    predictions: pd.DataFrame,
    *,
    labels: Sequence[int],
    task_type: str,
    metrics: Sequence[str],
    system_column: str = "system_id",
    sample_id_column: str = "sample_index",
    fold_column: str = "outer_fold",
    y_true_column: str = "y_true",
    y_pred_column: str = "y_pred",
    probability_columns: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Validate paired exactly-once OOF coverage without computing uncertainty."""

    requested_metrics = _validate_requested_metrics(metrics, task_type)
    probability_columns = probability_columns or {
        int(label): f"prob_class_{int(label)}" for label in labels
    }
    base, systems = _aligned_system_frames(
        predictions,
        labels=labels,
        system_column=system_column,
        sample_id_column=sample_id_column,
        fold_column=fold_column,
        y_true_column=y_true_column,
        y_pred_column=y_pred_column,
        probability_columns=probability_columns,
    )
    return {
        "status": "passed",
        "n_samples": int(len(base)),
        "n_systems": int(len(systems)),
        "system_ids": list(systems),
        "metrics": list(requested_metrics),
        "labels": [int(label) for label in labels],
        "task_type": get_task_schema(task_type).name,
    }


def _percentile_interval(values: np.ndarray, protocol: BootstrapProtocol) -> tuple[float, float]:
    alpha = 1.0 - protocol.confidence_level
    low, high = np.quantile(
        values,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method=protocol.quantile_method,
    )
    return float(low), float(high)


def compute_paired_oof_bootstrap(
    predictions: pd.DataFrame,
    *,
    labels: Sequence[int],
    task_type: str,
    metrics: Sequence[str],
    comparisons: Sequence[ComparisonSpec] = (),
    primary_metric: str | None = None,
    protocol: BootstrapProtocol = BootstrapProtocol(),
    system_column: str = "system_id",
    sample_id_column: str = "sample_index",
    fold_column: str = "outer_fold",
    y_true_column: str = "y_true",
    y_pred_column: str = "y_pred",
    probability_columns: Mapping[int, str] | None = None,
    n_bins: int = 10,
) -> BootstrapResult:
    """Compute percentile intervals and exact paired OOF metric differences."""

    if not isinstance(n_bins, int) or n_bins < 2:
        raise OOFBootstrapError("n_bins must be an integer of at least two.")
    requested_metrics = _validate_requested_metrics(metrics, task_type)
    if primary_metric is not None and primary_metric not in requested_metrics:
        raise OOFBootstrapError("primary_metric must be explicitly included in metrics.")
    if any(comparison.primary_gate for comparison in comparisons) and primary_metric is None:
        raise OOFBootstrapError(
            "A primary baseline gate requires an explicitly supplied primary_metric."
        )
    comparison_ids = [comparison.comparison_id for comparison in comparisons]
    if len(set(comparison_ids)) != len(comparison_ids):
        raise OOFBootstrapError("comparison_id values must be unique.")

    probability_columns = probability_columns or {
        int(label): f"prob_class_{int(label)}" for label in labels
    }
    base, systems = _aligned_system_frames(
        predictions,
        labels=labels,
        system_column=system_column,
        sample_id_column=sample_id_column,
        fold_column=fold_column,
        y_true_column=y_true_column,
        y_pred_column=y_pred_column,
        probability_columns=probability_columns,
    )
    system_ids = tuple(systems)
    unknown_comparison_systems = sorted(
        {
            value
            for comparison in comparisons
            for value in (comparison.system_a, comparison.system_b)
            if value not in systems
        }
    )
    if unknown_comparison_systems:
        raise OOFBootstrapError(
            f"Comparisons reference unknown OOF systems: {unknown_comparison_systems}."
        )

    resample_base = base.rename(columns={fold_column: "outer_fold", y_true_column: "y_true"})
    if protocol.strata_columns != ("outer_fold", "y_true"):
        # Custom reusable strata may refer to original base columns; retain
        # canonical aliases while copying any additional requested columns.
        for column in protocol.strata_columns:
            if column in base.columns and column not in resample_base.columns:
                resample_base[column] = base[column]
    if sample_id_column != "sample_index":
        resample_base = resample_base.rename(columns={sample_id_column: "sample_index"})
    plan = generate_stratified_resample_indices(
        resample_base,
        protocol,
        sample_id_column="sample_index",
    )

    labels_list = [int(label) for label in labels]
    probability_names = [probability_columns[label] for label in labels_list]
    draws: dict[str, dict[str, np.ndarray]] = {}
    points: dict[str, dict[str, float]] = {}
    interval_rows: list[dict[str, Any]] = []
    for system_id in system_ids:
        system = systems[system_id]
        y_true = system[y_true_column].to_numpy(dtype=int)
        y_pred = system[y_pred_column].to_numpy(dtype=int)
        proba = system[probability_names].to_numpy(dtype=float)
        point_metrics = classification_metrics(
            y_true,
            y_pred,
            proba,
            labels_list,
            n_bins=n_bins,
            task_type=task_type,
        )
        points[system_id] = {}
        draws[system_id] = {
            metric: np.empty(protocol.n_resamples, dtype=float) for metric in requested_metrics
        }
        for metric in requested_metrics:
            points[system_id][metric] = validate_metric_value(metric, point_metrics.get(metric))
        for iteration, sample_positions in enumerate(plan.indices):
            boot_metrics = classification_metrics(
                y_true[sample_positions],
                y_pred[sample_positions],
                proba[sample_positions],
                labels_list,
                n_bins=n_bins,
                task_type=task_type,
            )
            for metric in requested_metrics:
                draws[system_id][metric][iteration] = validate_metric_value(
                    metric,
                    boot_metrics.get(metric),
                )
        for metric in requested_metrics:
            metric_draws = draws[system_id][metric]
            if not np.isfinite(metric_draws).all():
                raise OOFBootstrapError(
                    f"Metric {metric!r} produced invalid bootstrap draws for {system_id!r}; "
                    "draws are never dropped."
                )
            low, high = _percentile_interval(metric_draws, protocol)
            validate_metric_value(metric, low)
            validate_metric_value(metric, high)
            definition = metric_definition(metric)
            interval_rows.append(
                {
                    "system_id": system_id,
                    "task_type": get_task_schema(task_type).name,
                    "metric": metric,
                    "point_estimate": points[system_id][metric],
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_std": float(np.std(metric_draws, ddof=1)),
                    "n_samples": int(len(base)),
                    "n_resamples": protocol.n_resamples,
                    "n_valid": protocol.n_resamples,
                    "confidence_level": protocol.confidence_level,
                    "method": protocol.method,
                    "strata": ";".join(protocol.strata_columns),
                    "seed": protocol.seed,
                    "resample_hash": plan.resample_hash,
                    "better_direction": definition.better_direction,
                    "domain_low": definition.lower_bound,
                    "domain_high": definition.upper_bound,
                }
            )

    difference_rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        for metric in requested_metrics:
            definition = metric_definition(metric)
            raw_draws = draws[comparison.system_a][metric] - draws[comparison.system_b][metric]
            oriented_draws = raw_draws if definition.better_direction == "higher" else -raw_draws
            if not np.isfinite(oriented_draws).all():
                raise OOFBootstrapError(
                    f"Paired difference {comparison.comparison_id!r}/{metric!r} has invalid draws."
                )
            raw_low, raw_high = _percentile_interval(raw_draws, protocol)
            oriented_low, oriented_high = _percentile_interval(oriented_draws, protocol)
            raw_point = points[comparison.system_a][metric] - points[comparison.system_b][metric]
            oriented_point = raw_point if definition.better_direction == "higher" else -raw_point
            gate_eligible = bool(comparison.primary_gate and metric == primary_metric)
            difference_rows.append(
                {
                    "comparison_id": comparison.comparison_id,
                    "system_a": comparison.system_a,
                    "system_b": comparison.system_b,
                    "task_type": get_task_schema(task_type).name,
                    "metric": metric,
                    "estimate_a": points[comparison.system_a][metric],
                    "estimate_b": points[comparison.system_b][metric],
                    "raw_difference_a_minus_b": raw_point,
                    "raw_difference_ci_low": raw_low,
                    "raw_difference_ci_high": raw_high,
                    "improvement_oriented_difference": oriented_point,
                    "improvement_ci_low": oriented_low,
                    "improvement_ci_high": oriented_high,
                    "bootstrap_std": float(np.std(oriented_draws, ddof=1)),
                    "n_samples": int(len(base)),
                    "n_resamples": protocol.n_resamples,
                    "n_valid": protocol.n_resamples,
                    "confidence_level": protocol.confidence_level,
                    "method": protocol.method,
                    "strata": ";".join(protocol.strata_columns),
                    "seed": protocol.seed,
                    "resample_hash": plan.resample_hash,
                    "better_direction": definition.better_direction,
                    "primary_metric": primary_metric,
                    "primary_gate_comparison": bool(comparison.primary_gate),
                    "gate_eligible": gate_eligible,
                    "gate_triggered": bool(
                        gate_eligible and oriented_point > 0.0 and oriented_low > 0.0
                    ),
                }
            )

    metadata = MappingProxyType(
        {
            "task_type": get_task_schema(task_type).name,
            "labels": labels_list,
            "systems": list(system_ids),
            "metrics": list(requested_metrics),
            "comparison_ids": comparison_ids,
            "primary_metric": primary_metric,
            "n_samples": int(len(base)),
            "n_resamples": protocol.n_resamples,
            "confidence_level": protocol.confidence_level,
            "seed": protocol.seed,
            "strata_columns": list(protocol.strata_columns),
            "method": protocol.method,
            "quantile_method": protocol.quantile_method,
            "resample_hash": plan.resample_hash,
            "valid_draw_policy": "fail_on_any_invalid_draw",
            "inference_scope": (
                "sample_level_uncertainty_conditional_on_fixed_outer_fold_assignment_and_fitted_models"
            ),
        }
    )
    return BootstrapResult(
        metric_intervals=pd.DataFrame(interval_rows),
        paired_differences=pd.DataFrame(difference_rows),
        metadata=metadata,
        resample_plan=plan,
    )
