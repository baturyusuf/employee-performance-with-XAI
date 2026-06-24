from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def multiclass_brier(y_true: Iterable[int], y_proba: np.ndarray, labels: list[int]) -> float:
    y_bin = label_binarize(list(y_true), classes=labels)
    if len(labels) == 2 and y_bin.ndim == 2 and y_bin.shape[1] == 1:
        y_bin = np.column_stack([1 - y_bin[:, 0], y_bin[:, 0]])
    return float(np.mean(np.sum((np.asarray(y_proba) - y_bin) ** 2, axis=1)))


def expected_calibration_error(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)
    confidence = np.max(np.asarray(y_proba), axis=1)
    correct = (y_true_arr == y_pred_arr).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        if i == 0:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence > low) & (confidence <= high)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
    return float(ece)


def severe_error_rate(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    return float(np.mean(np.abs(yt - yp) > 1))


def adjacent_accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    return float(np.mean(np.abs(yt - yp) <= 1))


def classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_proba: np.ndarray | None = None,
    labels: list[int] | None = None,
    n_bins: int = 10,
) -> Dict[str, Optional[float]]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)
    labels = labels or sorted(np.unique(y_true_arr).astype(int).tolist())

    metrics: Dict[str, Optional[float]] = {
        "accuracy": safe_float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": safe_float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": safe_float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_f1": safe_float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "macro_precision": safe_float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_precision": safe_float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "macro_recall": safe_float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_recall": safe_float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "ordinal_mae": safe_float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "quadratic_weighted_kappa": safe_float(cohen_kappa_score(y_true_arr, y_pred_arr, weights="quadratic")),
        "adjacent_accuracy": safe_float(adjacent_accuracy(y_true_arr, y_pred_arr)),
        "severe_error_rate": safe_float(severe_error_rate(y_true_arr, y_pred_arr)),
    }

    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        metrics["nll_log_loss"] = safe_float(log_loss(y_true_arr, proba, labels=labels))
        metrics["multiclass_brier"] = safe_float(multiclass_brier(y_true_arr, proba, labels))
        metrics["ece_confidence"] = safe_float(expected_calibration_error(y_true_arr, y_pred_arr, proba, n_bins=n_bins))

    return metrics


def leakage_sensitivity_index(full_feature_score: float, leakage_safe_score: float) -> Dict[str, Optional[float]]:
    """Compute leakage sensitivity for metrics where higher is better."""
    full = safe_float(full_feature_score)
    safe = safe_float(leakage_safe_score)
    if full is None or safe is None:
        return {"absolute_drop": None, "lsi": None}
    absolute_drop = full - safe
    if abs(full) == 0:
        lsi = None
    else:
        lsi = absolute_drop / abs(full)
    return {"absolute_drop": safe_float(absolute_drop), "lsi": safe_float(lsi)}
