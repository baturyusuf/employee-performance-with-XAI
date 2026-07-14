from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional


ORDINAL_MULTICLASS_PERFORMANCE = "ordinal_multiclass_performance"
RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS = "restricted_target_performance_robustness"
NOMINAL_MULTICLASS_PROXY_DIAGNOSTIC = "nominal_multiclass_proxy_diagnostic"
BINARY_ATTRITION_TRANSFER = "binary_attrition_transfer"
BINARY_TURNOVER_TRANSFER = "binary_turnover_transfer"

COMMON_CLASSIFICATION_METRICS = frozenset(
    {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "weighted_precision",
        "macro_recall",
        "weighted_recall",
    }
)
PROBABILITY_METRICS = frozenset(
    {
        "nll_log_loss",
        "multiclass_brier",
        "binary_brier",
        "roc_auc",
        "average_precision",
        "ece_confidence",
    }
)
ORDINAL_METRICS = frozenset(
    {
        "ordinal_mae",
        "quadratic_weighted_kappa",
        "adjacent_accuracy",
        "severe_error_rate",
    }
)
KNOWN_METRICS = COMMON_CLASSIFICATION_METRICS | PROBABILITY_METRICS | ORDINAL_METRICS


@dataclass(frozen=True)
class MetricDefinition:
    """One authoritative scientific definition for a reported metric."""

    name: str
    display_name: str
    family: str
    better_direction: Literal["higher", "lower"]
    lower_bound: float
    upper_bound: float
    unit: str
    estimand: str
    aggregation: str = "all_exactly_once_out_of_fold_samples"
    denominator: str = "complete_task_oof_sample_count"
    uncertainty_method: str = "paired_stratified_percentile_bootstrap"
    uncertainty_unit: str = "sample"
    requires_probabilities: bool = False


def _metric(
    name: str,
    display_name: str,
    family: str,
    direction: Literal["higher", "lower"],
    lower: float,
    upper: float,
    unit: str,
    estimand: str,
    *,
    probabilities: bool = False,
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        display_name=display_name,
        family=family,
        better_direction=direction,
        lower_bound=lower,
        upper_bound=upper,
        unit=unit,
        estimand=estimand,
        requires_probabilities=probabilities,
    )


METRIC_DEFINITIONS: Mapping[str, MetricDefinition] = MappingProxyType(
    {
        "accuracy": _metric("accuracy", "Accuracy", "classification", "higher", 0.0, 1.0, "proportion", "fraction of samples classified correctly"),
        "balanced_accuracy": _metric("balanced_accuracy", "Balanced accuracy", "classification", "higher", 0.0, 1.0, "proportion", "unweighted mean recall across observed classes"),
        "macro_f1": _metric("macro_f1", "Macro-F1", "classification", "higher", 0.0, 1.0, "proportion", "unweighted mean class-specific F1"),
        "weighted_f1": _metric("weighted_f1", "Weighted F1", "classification", "higher", 0.0, 1.0, "proportion", "support-weighted mean class-specific F1"),
        "macro_precision": _metric("macro_precision", "Macro precision", "classification", "higher", 0.0, 1.0, "proportion", "unweighted mean class-specific precision"),
        "weighted_precision": _metric("weighted_precision", "Weighted precision", "classification", "higher", 0.0, 1.0, "proportion", "support-weighted mean class-specific precision"),
        "macro_recall": _metric("macro_recall", "Macro recall", "classification", "higher", 0.0, 1.0, "proportion", "unweighted mean class-specific recall"),
        "weighted_recall": _metric("weighted_recall", "Weighted recall", "classification", "higher", 0.0, 1.0, "proportion", "support-weighted mean class-specific recall"),
        "quadratic_weighted_kappa": _metric("quadratic_weighted_kappa", "Quadratic weighted kappa", "ordinal", "higher", -1.0, 1.0, "agreement_coefficient", "quadratically weighted agreement beyond chance"),
        "ordinal_mae": _metric("ordinal_mae", "Ordinal MAE", "ordinal", "lower", 0.0, 2.0, "target_class_steps", "mean absolute ordered-class error"),
        "adjacent_accuracy": _metric("adjacent_accuracy", "Adjacent accuracy", "ordinal", "higher", 0.0, 1.0, "proportion", "fraction of predictions within one ordered class"),
        "severe_error_rate": _metric("severe_error_rate", "Severe-error rate", "ordinal", "lower", 0.0, 1.0, "proportion", "fraction of predictions more than one ordered class away"),
        "nll_log_loss": _metric("nll_log_loss", "Log loss", "probability", "lower", 0.0, math.inf, "nats_per_sample", "mean negative log-likelihood of the true class", probabilities=True),
        "multiclass_brier": _metric("multiclass_brier", "Multiclass Brier score", "probability", "lower", 0.0, 2.0, "squared_probability_error", "mean summed squared class-probability error", probabilities=True),
        "binary_brier": _metric("binary_brier", "Binary Brier score", "probability", "lower", 0.0, 1.0, "squared_probability_error", "mean squared positive-class probability error", probabilities=True),
        "ece_confidence": _metric("ece_confidence", "Confidence ECE", "calibration", "lower", 0.0, 1.0, "absolute_probability_gap", "support-weighted absolute confidence-accuracy gap across fixed bins", probabilities=True),
        "roc_auc": _metric("roc_auc", "ROC AUC", "ranking", "higher", 0.0, 1.0, "probability", "positive-negative ranking probability", probabilities=True),
        "average_precision": _metric("average_precision", "Average precision", "ranking", "higher", 0.0, 1.0, "proportion", "positive-class precision-recall summary", probabilities=True),
    }
)

if frozenset(METRIC_DEFINITIONS) != KNOWN_METRICS:
    raise RuntimeError("Metric definition registry and applicability registry differ.")


def metric_definition(metric: str) -> MetricDefinition:
    try:
        return METRIC_DEFINITIONS[metric]
    except KeyError as exc:
        raise ValueError(
            f"Metric {metric!r} has no scientific definition; allowed={sorted(METRIC_DEFINITIONS)}."
        ) from exc


@dataclass(frozen=True)
class TaskSchema:
    """Scientific task contract used to gate metrics and cross-task comparisons."""

    name: str
    comparison_group: str
    applicable_metrics: frozenset[str]
    ordinal_metrics_comparable: bool
    applicability_note: str
    selection_primary_metric: str = "macro_f1"
    selection_tie_break_metric: str = "balanced_accuracy"

    def is_metric_applicable(self, metric: str) -> bool:
        if metric not in KNOWN_METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Known metrics: {sorted(KNOWN_METRICS)}")
        return metric in self.applicable_metrics


_COMMON_METRICS = COMMON_CLASSIFICATION_METRICS
_COMMON_CALIBRATION_METRICS = frozenset({"nll_log_loss", "ece_confidence"})

TASK_SCHEMAS: Mapping[str, TaskSchema] = {
    ORDINAL_MULTICLASS_PERFORMANCE: TaskSchema(
        name=ORDINAL_MULTICLASS_PERFORMANCE,
        comparison_group="three_class_ordinal_performance",
        applicable_metrics=(
            _COMMON_METRICS
            | ORDINAL_METRICS
            | _COMMON_CALIBRATION_METRICS
            | {"multiclass_brier"}
        ),
        ordinal_metrics_comparable=True,
        applicability_note=(
            "Ordinal and severe-error metrics apply to the ordered 2/3/4 target. "
            "Independent replication remains distinct from locked-model transport."
        ),
        selection_tie_break_metric="quadratic_weighted_kappa",
    ),
    RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS: TaskSchema(
        name=RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS,
        comparison_group="restricted_target_performance_robustness",
        applicable_metrics=(
            _COMMON_METRICS
            | _COMMON_CALIBRATION_METRICS
            | {"binary_brier", "roc_auc", "average_precision"}
        ),
        ordinal_metrics_comparable=False,
        applicability_note=(
            "The observed 3/4-only target is evaluated as a restricted binary robustness task "
            "with class 4 as the predeclared positive class. Binary probability and ranking "
            "metrics apply, but ordinal distance, adjacency, severe-error, and multiclass Brier "
            "metrics are N/A because this task is not comparable with the primary 2/3/4 task."
        ),
    ),
    NOMINAL_MULTICLASS_PROXY_DIAGNOSTIC: TaskSchema(
        name=NOMINAL_MULTICLASS_PROXY_DIAGNOSTIC,
        comparison_group="nominal_proxy_risk_diagnostic",
        applicable_metrics=(
            _COMMON_METRICS | _COMMON_CALIBRATION_METRICS | {"multiclass_brier"}
        ),
        ordinal_metrics_comparable=False,
        applicability_note=(
            "Department reconstructability is a nominal multiclass proxy-risk diagnostic. "
            "It is not an employee-performance task and ordinal metrics are inapplicable."
        ),
    ),
    BINARY_ATTRITION_TRANSFER: TaskSchema(
        name=BINARY_ATTRITION_TRANSFER,
        comparison_group="related_binary_task_transfer",
        applicable_metrics=(
            _COMMON_METRICS
            | _COMMON_CALIBRATION_METRICS
            | {"binary_brier", "roc_auc", "average_precision"}
        ),
        ordinal_metrics_comparable=False,
        applicability_note=(
            "Ordinal metrics are N/A for binary attrition transfer; in particular, "
            "|y_true-y_pred|>1 is structurally impossible and must not be reported as zero risk."
        ),
    ),
    BINARY_TURNOVER_TRANSFER: TaskSchema(
        name=BINARY_TURNOVER_TRANSFER,
        comparison_group="related_binary_task_transfer",
        applicable_metrics=(
            _COMMON_METRICS
            | _COMMON_CALIBRATION_METRICS
            | {"binary_brier", "roc_auc", "average_precision"}
        ),
        ordinal_metrics_comparable=False,
        applicability_note=(
            "Ordinal metrics are N/A for binary turnover transfer; in particular, "
            "|y_true-y_pred|>1 is structurally impossible and must not be reported as zero risk."
        ),
    ),
}


TASK_TYPE_ALIASES: Mapping[str, str] = {
    ORDINAL_MULTICLASS_PERFORMANCE: ORDINAL_MULTICLASS_PERFORMANCE,
    "restricted_ordinal_performance": RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS,
    "restricted_performance": RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS,
    RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS: RESTRICTED_TARGET_PERFORMANCE_ROBUSTNESS,
    NOMINAL_MULTICLASS_PROXY_DIAGNOSTIC: NOMINAL_MULTICLASS_PROXY_DIAGNOSTIC,
    "nominal_proxy_diagnostic": NOMINAL_MULTICLASS_PROXY_DIAGNOSTIC,
    "binary_attrition": BINARY_ATTRITION_TRANSFER,
    BINARY_ATTRITION_TRANSFER: BINARY_ATTRITION_TRANSFER,
    "binary_turnover": BINARY_TURNOVER_TRANSFER,
    BINARY_TURNOVER_TRANSFER: BINARY_TURNOVER_TRANSFER,
}


def canonical_task_type(task_type: str) -> str:
    normalized = str(task_type).strip().lower()
    try:
        return TASK_TYPE_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown task type '{task_type}'. Allowed task types/aliases: {sorted(TASK_TYPE_ALIASES)}"
        ) from exc


def get_task_schema(task_type: str) -> TaskSchema:
    return TASK_SCHEMAS[canonical_task_type(task_type)]


def apply_metric_applicability(
    metrics: Mapping[str, Any],
    task_type: str,
    *,
    include_inapplicable: bool = True,
) -> dict[str, Any]:
    """Return metrics with scientifically inapplicable values omitted or set to N/A (`None`)."""

    schema = get_task_schema(task_type)
    output: dict[str, Any] = {}
    for name, value in metrics.items():
        if name not in KNOWN_METRICS or name in schema.applicable_metrics:
            output[name] = value
        elif include_inapplicable:
            output[name] = None

    if include_inapplicable:
        for name in KNOWN_METRICS - schema.applicable_metrics:
            output.setdefault(name, None)
    return output


def metric_applicability(task_type: str, metric: str) -> tuple[bool, Optional[str]]:
    schema = get_task_schema(task_type)
    applicable = schema.is_metric_applicable(metric)
    return applicable, None if applicable else schema.applicability_note


def metric_schema_records() -> tuple[dict[str, Any], ...]:
    """Return the complete task-by-metric registry in publication-safe row form."""

    records: list[dict[str, Any]] = []
    for task_name in sorted(TASK_SCHEMAS):
        task = TASK_SCHEMAS[task_name]
        for metric_name in sorted(METRIC_DEFINITIONS):
            definition = METRIC_DEFINITIONS[metric_name]
            applicable = metric_name in task.applicable_metrics
            if metric_name == task.selection_primary_metric:
                selection_role = "primary"
            elif metric_name == task.selection_tie_break_metric:
                selection_role = "tie_break"
            else:
                selection_role = "not_used_for_selection"
            records.append(
                {
                    "task_type": task.name,
                    "comparison_group": task.comparison_group,
                    "metric": definition.name,
                    "display_name": definition.display_name,
                    "family": definition.family,
                    "applicable": applicable,
                    "not_applicable_reason": "" if applicable else task.applicability_note,
                    "better_direction": definition.better_direction,
                    "domain_lower_bound": definition.lower_bound,
                    "domain_upper_bound": (
                        None if math.isinf(definition.upper_bound) else definition.upper_bound
                    ),
                    "unit": definition.unit,
                    "estimand": definition.estimand,
                    "aggregation": definition.aggregation,
                    "denominator": definition.denominator,
                    "requires_probabilities": definition.requires_probabilities,
                    "uncertainty_method": definition.uncertainty_method,
                    "uncertainty_unit": definition.uncertainty_unit,
                    "uncertainty_resamples": 5000,
                    "confidence_level": 0.95,
                    "selection_role": selection_role,
                    "cross_task_comparability": task.applicability_note,
                }
            )
    return tuple(records)


def metric_schema_hash() -> str:
    payload = json.dumps(
        metric_schema_records(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_metric_applicability_projection(rules: Mapping[str, Any]) -> None:
    """Reject configuration that disagrees with the authoritative task registry."""

    if set(rules) != set(TASK_SCHEMAS):
        raise ValueError(
            "Configured metric-applicability tasks differ from the authoritative registry: "
            f"configured={sorted(rules)}, expected={sorted(TASK_SCHEMAS)}."
        )
    for task_name, task in TASK_SCHEMAS.items():
        raw = rules.get(task_name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Metric-applicability rule for {task_name!r} must be a mapping.")
        applicable = raw.get("applicable")
        not_applicable = raw.get("not_applicable")
        if not isinstance(applicable, list) or not all(isinstance(value, str) for value in applicable):
            raise ValueError(f"Configured applicable metrics for {task_name!r} must be strings.")
        if not isinstance(not_applicable, list) or not all(
            isinstance(value, str) for value in not_applicable
        ):
            raise ValueError(f"Configured N/A metrics for {task_name!r} must be strings.")
        expected_applicable = task.applicable_metrics
        expected_not_applicable = KNOWN_METRICS - expected_applicable
        if set(applicable) != expected_applicable or len(applicable) != len(expected_applicable):
            raise ValueError(
                f"Configured applicable metrics for {task_name!r} differ from the registry."
            )
        if set(not_applicable) != expected_not_applicable or len(not_applicable) != len(
            expected_not_applicable
        ):
            raise ValueError(
                f"Configured N/A metrics for {task_name!r} differ from the registry."
            )
        comparability = raw.get("cross_task_comparability")
        if not isinstance(comparability, str) or not comparability.strip():
            raise ValueError(
                f"Configured cross-task comparability for {task_name!r} must be non-empty."
            )
