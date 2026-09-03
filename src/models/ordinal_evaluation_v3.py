"""Ordinal probability and class-level evaluation contracts for v3 evidence."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.models.evaluate import classification_metrics
from src.models.task_schema import ORDINAL_MULTICLASS_PERFORMANCE


class OrdinalEvaluationContractError(ValueError):
    """Raised when labels, predictions, or probabilities violate the v3 contract."""


def _validated_inputs(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    y_proba: np.ndarray,
    labels: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[Any, ...]]:
    true = np.asarray(list(y_true))
    predicted = np.asarray(list(y_pred))
    probabilities = np.asarray(y_proba, dtype=np.float64)
    ordered_labels = tuple(labels)
    if len(ordered_labels) < 3 or len(set(ordered_labels)) != len(ordered_labels):
        raise OrdinalEvaluationContractError(
            "Ordinal evaluation requires at least three unique labels in declared order."
        )
    if true.ndim != 1 or predicted.ndim != 1 or len(true) == 0 or len(predicted) != len(true):
        raise OrdinalEvaluationContractError(
            "y_true and y_pred must be equally sized non-empty vectors."
        )
    if probabilities.shape != (len(true), len(ordered_labels)):
        raise OrdinalEvaluationContractError(
            "Probability shape must equal (sample_count, ordered_label_count)."
        )
    if not np.all(np.isfinite(probabilities)):
        raise OrdinalEvaluationContractError("Probabilities must be finite.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise OrdinalEvaluationContractError("Probabilities must fall inside [0, 1].")
    if not np.allclose(
        probabilities.sum(axis=1, dtype=np.float64), 1.0, rtol=0.0, atol=1e-9
    ):
        raise OrdinalEvaluationContractError("Probability rows must sum to one.")
    allowed = set(ordered_labels)
    unknown_true = set(true.tolist()) - allowed
    unknown_predicted = set(predicted.tolist()) - allowed
    if unknown_true or unknown_predicted:
        raise OrdinalEvaluationContractError(
            "Targets and predictions must use only declared ordered labels; "
            f"unknown_true={sorted(map(str, unknown_true))}, "
            f"unknown_predicted={sorted(map(str, unknown_predicted))}."
        )
    return true, predicted, probabilities, ordered_labels


def ranked_probability_score(
    y_true: Iterable[Any],
    y_proba: np.ndarray,
    *,
    labels: Sequence[Any],
) -> float:
    """Return the normalized multiclass RPS over the declared label order.

    The score averages squared cumulative-probability errors over the K-1
    nontrivial thresholds, so its domain is [0, 1] and lower is better.
    """

    true = np.asarray(list(y_true))
    probabilities = np.asarray(y_proba, dtype=np.float64)
    ordered_labels = tuple(labels)
    if len(true) == 0:
        raise OrdinalEvaluationContractError("RPS requires at least one sample.")
    if len(ordered_labels) < 3 or len(set(ordered_labels)) != len(ordered_labels):
        raise OrdinalEvaluationContractError(
            "RPS requires at least three unique labels in declared order."
        )
    if probabilities.shape != (len(true), len(ordered_labels)):
        raise OrdinalEvaluationContractError(
            "RPS probability shape must equal (sample_count, ordered_label_count)."
        )
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise OrdinalEvaluationContractError("RPS probabilities must be finite and inside [0, 1].")
    if not np.allclose(
        probabilities.sum(axis=1, dtype=np.float64), 1.0, rtol=0.0, atol=1e-9
    ):
        raise OrdinalEvaluationContractError("RPS probability rows must sum to one.")
    label_position = {label: position for position, label in enumerate(ordered_labels)}
    try:
        encoded = np.asarray([label_position[value] for value in true], dtype=np.int64)
    except KeyError as exc:
        raise OrdinalEvaluationContractError(
            f"RPS target contains undeclared label {exc.args[0]!r}."
        ) from exc
    predicted_cumulative = np.cumsum(probabilities, axis=1)[:, :-1]
    thresholds = np.arange(len(ordered_labels) - 1, dtype=np.int64)
    observed_cumulative = encoded[:, np.newaxis] <= thresholds[np.newaxis, :]
    per_sample = np.mean(
        (predicted_cumulative - observed_cumulative.astype(np.float64)) ** 2,
        axis=1,
    )
    score = float(np.mean(per_sample))
    if score < -1e-12 or score > 1.0 + 1e-12:
        raise OrdinalEvaluationContractError(f"RPS escaped its [0, 1] domain: {score}.")
    return float(np.clip(score, 0.0, 1.0))


def two_level_reversal_rate(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    labels: Sequence[Any],
) -> float:
    """Fraction of predictions at least two positions from the true ordinal class."""

    true = np.asarray(list(y_true))
    predicted = np.asarray(list(y_pred))
    ordered_labels = tuple(labels)
    if len(true) == 0 or len(predicted) != len(true):
        raise OrdinalEvaluationContractError(
            "Two-level reversal requires equally sized non-empty target vectors."
        )
    if len(ordered_labels) < 3 or len(set(ordered_labels)) != len(ordered_labels):
        raise OrdinalEvaluationContractError(
            "Two-level reversal requires at least three unique ordered labels."
        )
    label_position = {label: position for position, label in enumerate(ordered_labels)}
    try:
        true_position = np.asarray([label_position[value] for value in true], dtype=np.int64)
        predicted_position = np.asarray(
            [label_position[value] for value in predicted], dtype=np.int64
        )
    except KeyError as exc:
        raise OrdinalEvaluationContractError(
            f"Two-level reversal input contains undeclared label {exc.args[0]!r}."
        ) from exc
    return float(np.mean(np.abs(true_position - predicted_position) >= 2))


def ordinal_evaluation_bundle_v3(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    y_proba: np.ndarray,
    *,
    labels: Sequence[Any],
    dataset_key: str,
    model_name: str,
) -> dict[str, Any]:
    """Return aggregate, per-class, and long-form confusion evidence."""

    true, predicted, probabilities, ordered_labels = _validated_inputs(
        y_true, y_pred, y_proba, labels
    )
    if not str(dataset_key).strip() or not str(model_name).strip():
        raise OrdinalEvaluationContractError(
            "dataset_key and model_name must be non-empty evidence identifiers."
        )
    aggregate = classification_metrics(
        true.astype(int),
        predicted.astype(int),
        probabilities,
        labels=[int(label) for label in ordered_labels],
        task_type=ORDINAL_MULTICLASS_PERFORMANCE,
    )
    aggregate = {name: value for name, value in aggregate.items() if value is not None}
    legacy_severe = aggregate.pop("severe_error_rate")
    two_level = two_level_reversal_rate(true, predicted, labels=ordered_labels)
    if legacy_severe is not None and not np.isclose(legacy_severe, two_level):
        raise OrdinalEvaluationContractError(
            "Legacy severe-error and position-based two-level reversal definitions disagree."
        )
    aggregate["two_level_reversal_rate"] = two_level
    aggregate["ranked_probability_score"] = ranked_probability_score(
        true, probabilities, labels=ordered_labels
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        predicted,
        labels=list(ordered_labels),
        zero_division=0,
    )
    class_rows = []
    for index, label in enumerate(ordered_labels):
        class_rows.append(
            {
                "dataset_key": str(dataset_key),
                "model_name": str(model_name),
                "class_label": label.item() if hasattr(label, "item") else label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )

    matrix = confusion_matrix(true, predicted, labels=list(ordered_labels))
    confusion_rows = []
    for true_index, true_label in enumerate(ordered_labels):
        for predicted_index, predicted_label in enumerate(ordered_labels):
            confusion_rows.append(
                {
                    "dataset_key": str(dataset_key),
                    "model_name": str(model_name),
                    "true_label": (
                        true_label.item() if hasattr(true_label, "item") else true_label
                    ),
                    "predicted_label": (
                        predicted_label.item()
                        if hasattr(predicted_label, "item")
                        else predicted_label
                    ),
                    "count": int(matrix[true_index, predicted_index]),
                }
            )
    return {
        "dataset_key": str(dataset_key),
        "model_name": str(model_name),
        "ordered_labels": list(ordered_labels),
        "aggregate_metrics": aggregate,
        "per_class_metrics": class_rows,
        "confusion_matrix": confusion_rows,
        "rps_definition": (
            "mean_over_samples_and_K_minus_1_thresholds_of_squared_cumulative_error"
        ),
        "two_level_reversal_definition": (
            "absolute_declared_class_position_error_greater_than_or_equal_to_two"
        ),
    }


__all__ = [
    "OrdinalEvaluationContractError",
    "ordinal_evaluation_bundle_v3",
    "ranked_probability_score",
    "two_level_reversal_rate",
]
