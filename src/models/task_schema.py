from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


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
class TaskSchema:
    """Scientific task contract used to gate metrics and cross-task comparisons."""

    name: str
    comparison_group: str
    applicable_metrics: frozenset[str]
    ordinal_metrics_comparable: bool
    applicability_note: str

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
