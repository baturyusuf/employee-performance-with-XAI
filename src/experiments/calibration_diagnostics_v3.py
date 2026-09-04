"""Extend exact canonical OOF calibration evidence with explicit v3 diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.canonical_loader import sha256_file
from src.governance.calibration_diagnostics_contract_v3 import (
    DEFAULT_CALIBRATION_DIAGNOSTICS_CONTRACT,
    validate_calibration_diagnostics_contract_v3,
)
from src.governance.manuscript_contract import source_tree_hash
from src.governance.offline_runtime import enforce_offline_runtime
from src.models.ordinal_evaluation_v3 import ranked_probability_score
from src.utils.config_loader import PROJECT_ROOT


matplotlib.rcParams["svg.hashsalt"] = "calibration_diagnostics_v3"

DEFAULT_LOCAL_RUN_ROOT = Path("reports/major_revision_v3_runs")
LABELS = (2, 3, 4)
METHODS = ("raw", "sigmoid")
PROBABILITY_COLUMNS = ("prob_class_2", "prob_class_3", "prob_class_4")
EXPECTED_LOCAL_FILES = frozenset(
    {
        "calibration_metric_summary.csv",
        "classwise_calibration_metrics.csv",
        "cumulative_calibration_metrics.csv",
        "extended_reliability_bins.csv",
        "method_comparison.csv",
        "classwise_reliability.png",
        "classwise_reliability.svg",
        "cumulative_reliability.png",
        "cumulative_reliability.svg",
        "diagnostic_receipt.json",
        "stage_metadata.json",
    }
)


class CalibrationDiagnosticsV3Error(RuntimeError):
    """Raised when Phase 2B execution violates its frozen diagnostic contract."""


@dataclass(frozen=True)
class Phase2BResult:
    metric_summary: pd.DataFrame
    classwise_metrics: pd.DataFrame
    cumulative_metrics: pd.DataFrame
    reliability_bins: pd.DataFrame
    method_comparison: pd.DataFrame
    diagnostic_receipt: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationDiagnosticsV3Error(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationDiagnosticsV3Error(
            f"Could not read {path.as_posix()}: {exc}"
        ) from exc
    _require(isinstance(payload, dict), f"{path.name} must contain a JSON object.")
    return payload


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clean_git_identity() -> dict[str, str]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CalibrationDiagnosticsV3Error(
            f"Could not establish Git identity: {exc}"
        ) from exc
    _require(
        len(head) == 40 and all(character in "0123456789abcdef" for character in head),
        "Git HEAD digest is invalid.",
    )
    _require(
        not status,
        f"Scientific execution requires a clean worktree; status={status.splitlines()[:10]}.",
    )
    return {"commit": head, "branch": branch}


def _validated_predictions(contract: Mapping[str, Any]) -> pd.DataFrame:
    source = contract["source_contracts"]["calibration_predictions"]
    frame = pd.read_csv(PROJECT_ROOT / str(source["path"]))
    required = {
        "method",
        "sample_index",
        "outer_fold",
        "y_true",
        "y_pred",
        *PROBABILITY_COLUMNS,
    }
    _require(required.issubset(frame.columns), "Calibration prediction schema drifted.")
    _require(len(frame) == 2400, "Calibration prediction row count drifted.")
    for method, rows in frame.groupby("method", sort=True):
        _require(method in METHODS and len(rows) == 1200, f"Calibration method coverage drifted for {method}.")
        _require(set(rows["sample_index"].astype(int)) == set(range(1200)), f"Sample coverage drifted for {method}.")
        probability = rows.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
        _require(np.isfinite(probability).all(), f"Non-finite probability for {method}.")
        _require(np.all((0.0 <= probability) & (probability <= 1.0)), f"Probability outside [0,1] for {method}.")
        _require(np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-9), f"Probability simplex drifted for {method}.")
        predicted = np.asarray(LABELS)[np.argmax(probability, axis=1)]
        _require(np.array_equal(predicted, rows["y_pred"].to_numpy(int)), f"Argmax labels drifted for {method}.")
    raw = frame[frame["method"] == "raw"].sort_values("sample_index")
    sigmoid = frame[frame["method"] == "sigmoid"].sort_values("sample_index")
    for column in ("sample_index", "outer_fold", "y_true"):
        _require(np.array_equal(raw[column].to_numpy(int), sigmoid[column].to_numpy(int)), f"Raw/sigmoid identity drifted for {column}.")
    return frame.sort_values(["method", "sample_index"]).reset_index(drop=True)


def reliability_bin_rows_v3(
    outcomes: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    method: str,
    reliability_scope: str,
    target_id: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return all fixed equal-width reliability bins, including empty bins."""

    target = np.asarray(outcomes, dtype=int)
    probability = np.asarray(scores, dtype=float)
    _require(len(target) > 0 and target.shape == probability.shape, "Reliability vectors must be equal and non-empty.")
    _require(set(np.unique(target)).issubset({0, 1}), "Reliability outcomes must be binary.")
    _require(np.isfinite(probability).all(), "Reliability probabilities must be finite.")
    _require(np.all((0.0 <= probability) & (probability <= 1.0)), "Reliability probabilities escaped [0,1].")
    _require(n_bins == 10, "Phase 2B reliability requires ten bins.")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for position, (low, high) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        mask = (
            (probability >= low) & (probability <= high)
            if position == 1
            else (probability > low) & (probability <= high)
        )
        count = int(mask.sum())
        positives = int(target[mask].sum()) if count else 0
        predicted = float(probability[mask].mean()) if count else np.nan
        observed = float(positives / count) if count else np.nan
        rows.append(
            {
                "method": str(method),
                "reliability_scope": str(reliability_scope),
                "target_id": str(target_id),
                "bin": position,
                "bin_low": float(low),
                "bin_high": float(high),
                "n_samples": count,
                "n_positive": positives,
                "bin_status": "observed" if count else "empty",
                "mean_predicted_probability": predicted,
                "observed_frequency": observed,
                "absolute_gap": abs(predicted - observed) if count else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    _require(int(result["n_samples"].sum()) == len(target), "Reliability bins do not partition all samples.")
    return result


def expected_calibration_error_v3(bins: pd.DataFrame) -> float:
    """Calculate support-weighted absolute-gap ECE from an explicit bin table."""

    _require(len(bins) == 10 and set(bins["bin"].astype(int)) == set(range(1, 11)), "ECE requires one complete ten-bin target grid.")
    total = int(bins["n_samples"].sum())
    _require(total > 0, "ECE denominator is empty.")
    observed = bins[bins["n_samples"].astype(int) > 0]
    return float(
        np.sum(
            observed["n_samples"].to_numpy(float)
            / total
            * observed["absolute_gap"].to_numpy(float)
        )
    )


def _binary_log_loss(outcomes: np.ndarray, scores: np.ndarray) -> float:
    clipped = np.clip(np.asarray(scores, dtype=float), 1e-15, 1.0 - 1e-15)
    target = np.asarray(outcomes, dtype=float)
    return float(-np.mean(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped)))


def calibration_intercept_slope_v3(
    outcomes: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    probability_clip: float = 1e-6,
    maximum_iterations: int = 100,
    step_tolerance: float = 1e-10,
    hessian_ridge: float = 1e-12,
) -> dict[str, Any]:
    """Jointly estimate unpenalized logistic calibration intercept and slope."""

    target = np.asarray(outcomes, dtype=float)
    probability = np.asarray(scores, dtype=float)
    _require(target.shape == probability.shape and len(target) > 0, "Calibration regression vectors drifted.")
    _require(set(np.unique(target)).issubset({0.0, 1.0}), "Calibration regression outcome must be binary.")
    _require(0.0 < probability_clip < 0.5, "Calibration regression clip is invalid.")
    clipped = np.clip(probability, probability_clip, 1.0 - probability_clip)
    logit = np.log(clipped / (1.0 - clipped))
    design = np.column_stack([np.ones(len(logit), dtype=float), logit])
    beta = np.asarray([0.0, 1.0], dtype=float)
    converged = False
    maximum_step = float("inf")
    condition_number = float("nan")
    for iteration in range(1, maximum_iterations + 1):
        linear = design @ beta
        fitted = np.empty_like(linear)
        positive = linear >= 0.0
        fitted[positive] = 1.0 / (1.0 + np.exp(-linear[positive]))
        exponential = np.exp(linear[~positive])
        fitted[~positive] = exponential / (1.0 + exponential)
        weights = np.maximum(fitted * (1.0 - fitted), np.finfo(float).eps)
        information = design.T @ (weights[:, np.newaxis] * design)
        condition_number = float(np.linalg.cond(information))
        gradient = design.T @ (target - fitted)
        try:
            step = np.linalg.solve(
                information + hessian_ridge * np.eye(2, dtype=float), gradient
            )
        except np.linalg.LinAlgError as exc:
            raise CalibrationDiagnosticsV3Error(
                f"Calibration regression Hessian is singular: {exc}"
            ) from exc
        beta = beta + step
        maximum_step = float(np.max(np.abs(step)))
        if maximum_step <= step_tolerance:
            converged = True
            break
    _require(converged, "Calibration intercept/slope regression did not converge.")
    _require(np.isfinite(beta).all(), "Calibration intercept/slope is non-finite.")
    return {
        "calibration_intercept": float(beta[0]),
        "calibration_slope": float(beta[1]),
        "regression_converged": True,
        "regression_iterations": iteration,
        "maximum_final_step": maximum_step,
        "information_condition_number": condition_number,
    }


def _binary_diagnostic_row(
    outcomes: np.ndarray,
    scores: np.ndarray,
    *,
    method: str,
    target_id: str,
    target_type: str,
    ece: float,
) -> dict[str, Any]:
    return {
        "method": method,
        "target_type": target_type,
        "target_id": target_id,
        "n_samples": len(outcomes),
        "n_positive": int(outcomes.sum()),
        "n_negative": int(len(outcomes) - outcomes.sum()),
        "prevalence": float(np.mean(outcomes)),
        "expected_calibration_error": ece,
        "binary_brier": float(np.mean((scores - outcomes) ** 2)),
        "binary_log_loss": _binary_log_loss(outcomes, scores),
        **calibration_intercept_slope_v3(outcomes, scores),
        "regression_scope": "pooled_exactly_once_oof_descriptive_diagnostic",
        "confidence_interval_applicable": False,
    }


def _multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    positions = {label: index for index, label in enumerate(LABELS)}
    encoded = np.zeros_like(probabilities, dtype=float)
    encoded[np.arange(len(y_true)), [positions[int(value)] for value in y_true]] = 1.0
    return float(np.mean(np.sum((probabilities - encoded) ** 2, axis=1)))


def _multiclass_log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    positions = {label: index for index, label in enumerate(LABELS)}
    true_probability = probabilities[
        np.arange(len(y_true)), [positions[int(value)] for value in y_true]
    ]
    return float(-np.mean(np.log(np.clip(true_probability, 1e-15, 1.0))))


def evaluate_calibration_diagnostics_v3(
    contract: Mapping[str, Any], predictions: pd.DataFrame
) -> Phase2BResult:
    """Compute all frozen Phase 2B metrics from exact persisted OOF probabilities."""

    all_bins: list[pd.DataFrame] = []
    class_rows: list[dict[str, Any]] = []
    cumulative_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        rows = predictions[predictions["method"] == method].sort_values("sample_index")
        y_true = rows["y_true"].to_numpy(int)
        y_pred = rows["y_pred"].to_numpy(int)
        probabilities = rows.loc[:, PROBABILITY_COLUMNS].to_numpy(float)

        top_scores = probabilities.max(axis=1)
        top_outcomes = (y_true == y_pred).astype(int)
        top_bins = reliability_bin_rows_v3(
            top_outcomes,
            top_scores,
            method=method,
            reliability_scope="top_label",
            target_id="argmax_correct",
        )
        all_bins.append(top_bins)
        top_ece = expected_calibration_error_v3(top_bins)

        class_ece_values: list[float] = []
        for position, label in enumerate(LABELS):
            outcomes = (y_true == label).astype(int)
            scores = probabilities[:, position]
            bins = reliability_bin_rows_v3(
                outcomes,
                scores,
                method=method,
                reliability_scope="one_vs_rest_class",
                target_id=f"class_{label}",
            )
            all_bins.append(bins)
            ece = expected_calibration_error_v3(bins)
            class_ece_values.append(ece)
            class_rows.append(
                {
                    "class_label": label,
                    **_binary_diagnostic_row(
                        outcomes,
                        scores,
                        method=method,
                        target_id=f"class_{label}",
                        target_type="one_vs_rest_class",
                        ece=ece,
                    ),
                }
            )

        cumulative_ece_values: list[float] = []
        cumulative_brier_values: list[float] = []
        for position, threshold in enumerate(LABELS[:-1]):
            outcomes = (y_true <= threshold).astype(int)
            scores = probabilities[:, : position + 1].sum(axis=1)
            target_id = f"Y_le_{threshold}"
            bins = reliability_bin_rows_v3(
                outcomes,
                scores,
                method=method,
                reliability_scope="cumulative_threshold",
                target_id=target_id,
            )
            all_bins.append(bins)
            ece = expected_calibration_error_v3(bins)
            diagnostic = _binary_diagnostic_row(
                outcomes,
                scores,
                method=method,
                target_id=target_id,
                target_type="cumulative_threshold",
                ece=ece,
            )
            cumulative_rows.append(
                {
                    "threshold_label": threshold,
                    "cumulative_event": f"Y<={threshold}",
                    **diagnostic,
                }
            )
            cumulative_ece_values.append(ece)
            cumulative_brier_values.append(float(diagnostic["binary_brier"]))

        rps = ranked_probability_score(y_true, probabilities, labels=LABELS)
        mean_cumulative_brier = float(np.mean(cumulative_brier_values))
        _require(math.isclose(rps, mean_cumulative_brier, rel_tol=0.0, abs_tol=1e-15), "RPS/cumulative-Brier identity drifted.")
        method_bins = pd.concat(all_bins, ignore_index=True)
        method_bins = method_bins[method_bins["method"] == method]
        summary_rows.append(
            {
                "method": method,
                "primary_method": method == "sigmoid",
                "n_samples": len(rows),
                "nll_log_loss": _multiclass_log_loss(y_true, probabilities),
                "multiclass_brier": _multiclass_brier(y_true, probabilities),
                "top_label_ece": top_ece,
                "macro_classwise_ece": float(np.mean(class_ece_values)),
                "mean_cumulative_ece": float(np.mean(cumulative_ece_values)),
                "ranked_probability_score": rps,
                "mean_cumulative_binary_brier": mean_cumulative_brier,
                "empty_reliability_bin_count": int(
                    (method_bins["n_samples"].astype(int) == 0).sum()
                ),
                "calibration_method_selection_performed": False,
            }
        )

    reliability = pd.concat(all_bins, ignore_index=True).sort_values(
        ["method", "reliability_scope", "target_id", "bin"]
    ).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows)
    classwise = pd.DataFrame(class_rows).sort_values(
        ["method", "class_label"]
    ).reset_index(drop=True)
    cumulative = pd.DataFrame(cumulative_rows).sort_values(
        ["method", "threshold_label"]
    ).reset_index(drop=True)
    summary_lookup = summary.set_index("method")
    paired_source = contract["source_contracts"]["calibration_paired_differences"]
    paired = pd.read_csv(PROJECT_ROOT / str(paired_source["path"]))
    source_metric = {
        "nll_log_loss": "nll_log_loss",
        "multiclass_brier": "multiclass_brier",
        "top_label_ece": "ece_confidence",
    }
    comparison_rows: list[dict[str, Any]] = []
    for metric in contract["comparison"]["metrics"]:
        raw_value = float(summary_lookup.loc["raw", metric])
        sigmoid_value = float(summary_lookup.loc["sigmoid", metric])
        source_name = source_metric.get(metric)
        source = paired[paired["metric"] == source_name] if source_name else pd.DataFrame()
        interval_available = len(source) == 1
        comparison_rows.append(
            {
                "comparison_id": "sigmoid_minus_raw",
                "metric": metric,
                "better_direction": "lower",
                "raw_value": raw_value,
                "sigmoid_value": sigmoid_value,
                "raw_difference_sigmoid_minus_raw": sigmoid_value - raw_value,
                "direction_aligned_improvement": raw_value - sigmoid_value,
                "bootstrap_interval_available": interval_available,
                "raw_difference_ci_low": (
                    float(source.iloc[0]["raw_difference_ci_low"])
                    if interval_available
                    else np.nan
                ),
                "raw_difference_ci_high": (
                    float(source.iloc[0]["raw_difference_ci_high"])
                    if interval_available
                    else np.nan
                ),
                "improvement_ci_low": (
                    float(source.iloc[0]["improvement_ci_low"])
                    if interval_available
                    else np.nan
                ),
                "improvement_ci_high": (
                    float(source.iloc[0]["improvement_ci_high"])
                    if interval_available
                    else np.nan
                ),
                "interval_source": (
                    "canonical_v2_paired_stratified_sample_bootstrap"
                    if interval_available
                    else "not_estimated_for_new_diagnostic"
                ),
                "test_set_method_selection_performed": False,
                "all_metrics_improved_claim_allowed": False,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    diagnostic_receipt = {
        "schema_version": 1,
        "source_methods": list(METHODS),
        "sample_count_per_method": 1200,
        "ordered_labels": list(LABELS),
        "sigmoid_algorithm": "one_vs_rest_platt_on_class_probability_logit_then_row_renormalize",
        "sigmoid_predeclared": True,
        "calibration_method_selection_performed": False,
        "outer_test_used_for_calibrator_fit_or_selection": False,
        "new_model_fit_calls": 0,
        "new_calibrator_fit_calls": 0,
        "diagnostic_calibration_regression_fit_count": len(classwise) + len(cumulative),
        "reliability_bin_count": 10,
        "reliability_binning": "fixed_equal_width",
        "empty_bins_persisted": True,
        "top_label_ece_definition": "support_weighted_absolute_confidence_accuracy_gap",
        "classwise_ece_definition": "one_vs_rest_support_weighted_absolute_probability_frequency_gap",
        "cumulative_ece_definition": "ordered_threshold_support_weighted_absolute_probability_frequency_gap",
        "rps_normalized_by_threshold_count": True,
        "rps_equals_mean_cumulative_binary_brier": True,
        "calibration_regression_scope": "pooled_exactly_once_oof_descriptive_diagnostic",
        "future_calibration_validation_claim_allowed": False,
        "all_metrics_improved_claim_allowed": False,
    }
    return Phase2BResult(
        metric_summary=summary,
        classwise_metrics=classwise,
        cumulative_metrics=cumulative,
        reliability_bins=reliability,
        method_comparison=comparison,
        diagnostic_receipt=diagnostic_receipt,
    )


def _plot_reliability(
    bins: pd.DataFrame,
    *,
    scopes: Sequence[tuple[str, str, str]],
    title: str,
    output_stem: Path,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(1, len(scopes), figsize=(5.2 * len(scopes), 4.8), constrained_layout=True)
    axes_values = np.atleast_1d(axes)
    colors = {"raw": "#6C757D", "sigmoid": "#176B87"}
    for axis, (scope, target_id, panel_title) in zip(axes_values, scopes):
        for method in METHODS:
            rows = bins[
                (bins["method"] == method)
                & (bins["reliability_scope"] == scope)
                & (bins["target_id"] == target_id)
                & (bins["n_samples"].astype(int) > 0)
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
            title=panel_title,
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(title, fontsize=13)
    png = output_stem.with_suffix(".png")
    svg = output_stem.with_suffix(".svg")
    description = "Ten fixed equal-width bins; empty bins omitted visually but retained in source CSV."
    figure.savefig(
        png,
        dpi=300,
        metadata={"Software": "employee-performance-xai-v3", "Description": description},
    )
    figure.savefig(
        svg,
        format="svg",
        metadata={"Title": title, "Description": description, "Date": None},
    )
    plt.close(figure)
    return png, svg


def preflight_calibration_diagnostics_v3(
    contract_path: Path | str = DEFAULT_CALIBRATION_DIAGNOSTICS_CONTRACT,
) -> dict[str, Any]:
    """Validate all Phase 2B sources and scope without writing or fitting."""

    path = Path(contract_path)
    contract_receipt = validate_calibration_diagnostics_contract_v3(path)
    contract = _load_json(PROJECT_ROOT / path)
    predictions = _validated_predictions(contract)
    return {
        "status": "preflight_passed",
        "contract_sha256": contract_receipt["contract_sha256"],
        "prediction_rows": len(predictions),
        "sample_count_per_method": 1200,
        "planned_new_model_fit_calls": 0,
        "planned_new_calibrator_fit_calls": 0,
        "planned_diagnostic_regression_fit_calls": 10,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def run_calibration_diagnostics_v3(
    *,
    contract_path: Path | str,
    output_dir: Path | str,
    run_id: str,
) -> dict[str, Any]:
    """Atomically publish complete Phase 2B evidence from a clean exact commit."""

    contract_path = Path(contract_path)
    output_dir = Path(output_dir)
    _require(bool(str(run_id).strip()), "run_id must be non-empty.")
    with enforce_offline_runtime() as offline_state:
        git_identity = _clean_git_identity()
        contract_receipt = validate_calibration_diagnostics_contract_v3(contract_path)
        contract = _load_json(PROJECT_ROOT / contract_path)
        predictions = _validated_predictions(contract)
        _require(not output_dir.exists(), f"Output destination already exists: {output_dir}.")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = output_dir.parent / f".{output_dir.name}.staging.{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            implementation_paths = (
                Path("src/experiments/calibration_diagnostics_v3.py"),
                Path("src/governance/calibration_diagnostics_contract_v3.py"),
                Path("src/models/ordinal_evaluation_v3.py"),
            )
            scientific_inputs = {
                "git_identity": git_identity,
                "source_tree_hash": source_tree_hash(PROJECT_ROOT),
                "contract_sha256": contract_receipt["contract_sha256"],
                "bound_source_hashes": {
                    name: record["sha256"]
                    for name, record in contract["source_contracts"].items()
                },
                "implementation_hashes": {
                    path.as_posix(): sha256_file(PROJECT_ROOT / path)
                    for path in implementation_paths
                },
            }
            scientific_input_sha256 = _canonical_json_sha256(scientific_inputs)
            result = evaluate_calibration_diagnostics_v3(contract, predictions)
            for observed, expected, label in (
                (len(result.metric_summary), 2, "metric summary"),
                (len(result.classwise_metrics), 6, "classwise metrics"),
                (len(result.cumulative_metrics), 4, "cumulative metrics"),
                (len(result.reliability_bins), 120, "reliability bins"),
                (len(result.method_comparison), 6, "method comparison"),
            ):
                _require(observed == expected, f"Phase 2B {label} row count drifted.")
            frames = {
                "calibration_metric_summary.csv": result.metric_summary,
                "classwise_calibration_metrics.csv": result.classwise_metrics,
                "cumulative_calibration_metrics.csv": result.cumulative_metrics,
                "extended_reliability_bins.csv": result.reliability_bins,
                "method_comparison.csv": result.method_comparison,
            }
            for filename, frame in frames.items():
                frame.to_csv(staging / filename, index=False)
            _plot_reliability(
                result.reliability_bins,
                scopes=[
                    ("one_vs_rest_class", "class_2", "Class 2 one-vs-rest"),
                    ("one_vs_rest_class", "class_3", "Class 3 one-vs-rest"),
                    ("one_vs_rest_class", "class_4", "Class 4 one-vs-rest"),
                ],
                title="Classwise OOF reliability",
                output_stem=staging / "classwise_reliability",
            )
            _plot_reliability(
                result.reliability_bins,
                scopes=[
                    ("cumulative_threshold", "Y_le_2", "Cumulative event Y≤2"),
                    ("cumulative_threshold", "Y_le_3", "Cumulative event Y≤3"),
                ],
                title="Ordinal cumulative OOF reliability",
                output_stem=staging / "cumulative_reliability",
            )
            diagnostic_receipt = {
                **result.diagnostic_receipt,
                "run_id": run_id,
                "contract_sha256": contract_receipt["contract_sha256"],
                "scientific_input_sha256": scientific_input_sha256,
                "source_calibration_identity": contract["canonical_identity"],
            }
            _write_json(staging / "diagnostic_receipt.json", diagnostic_receipt)
            output_hashes = {
                path.name: sha256_file(path)
                for path in sorted(staging.iterdir())
                if path.is_file()
            }
            metadata = {
                "schema_version": 1,
                "stage": "calibration_diagnostics_v3",
                "status": "complete",
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract_sha256": contract_receipt["contract_sha256"],
                "scientific_input_sha256": scientific_input_sha256,
                "scientific_inputs": scientific_inputs,
                "git_identity": git_identity,
                "source_calibration_identity": contract["canonical_identity"],
                "prediction_row_count": len(predictions),
                "sample_count_per_method": 1200,
                "method_count": 2,
                "new_model_fit_calls": 0,
                "new_calibrator_fit_calls": 0,
                "diagnostic_calibration_regression_fit_count": 10,
                "metric_summary_row_count": len(result.metric_summary),
                "classwise_metric_row_count": len(result.classwise_metrics),
                "cumulative_metric_row_count": len(result.cumulative_metrics),
                "reliability_bin_row_count": len(result.reliability_bins),
                "method_comparison_row_count": len(result.method_comparison),
                "same_dataset_oof_diagnostic_not_future_validation": True,
                "test_set_method_selection_performed": False,
                "all_metrics_improved_claim_allowed": False,
                "runtime_policy": offline_state.receipt(),
                "network_calls": 0,
                "paid_api_calls": 0,
                "output_hashes": output_hashes,
            }
            _require(_clean_git_identity() == git_identity, "Git identity changed during Phase 2B execution.")
            _require(source_tree_hash(PROJECT_ROOT) == scientific_inputs["source_tree_hash"], "Source tree changed during Phase 2B execution.")
            repeated = validate_calibration_diagnostics_contract_v3(contract_path)
            _require(repeated["contract_sha256"] == contract_receipt["contract_sha256"], "Phase 2B contract changed during execution.")
            for name, record in contract["source_contracts"].items():
                _require(sha256_file(PROJECT_ROOT / str(record["path"])) == record["sha256"], f"Phase 2B source changed during execution: {name}.")
            _write_json(staging / "stage_metadata.json", metadata)
            _require({path.name for path in staging.iterdir() if path.is_file()} == EXPECTED_LOCAL_FILES, "Phase 2B output inventory drifted.")
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
        "contract_sha256": contract_receipt["contract_sha256"],
        "scientific_input_sha256": scientific_input_sha256,
        "new_model_fit_calls": 0,
        "new_calibrator_fit_calls": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract", type=Path, default=DEFAULT_CALIBRATION_DIAGNOSTICS_CONTRACT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_LOCAL_RUN_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.preflight_only:
        receipt = preflight_calibration_diagnostics_v3(args.contract)
    else:
        _require(bool(args.run_id), "Full Phase 2B execution requires --run-id.")
        output = args.output_root / str(args.run_id) / "calibration_diagnostics"
        receipt = run_calibration_diagnostics_v3(
            contract_path=args.contract,
            output_dir=output,
            run_id=str(args.run_id),
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
