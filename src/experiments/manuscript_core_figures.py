"""Generate the seven frozen, source-bound v2 core figures.

The generator consumes only exact artifacts beneath the active immutable run
root.  It never searches historical report trees and it publishes one atomic
``core_figures`` directory containing PNG/SVG pairs, machine-readable source
tables, technical captions, and an identity-bound figure manifest.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.governance.core_figure_contract import (
    CORE_FIGURE_IDENTITY_FIELDS,
    CORE_FIGURE_KEYS,
    CORE_FIGURE_PLAN_VERSION,
    expected_core_figure_plan,
)
from src.governance.manuscript_contract import (
    canonical_config_hash,
    manuscript_settings,
    sha256_file,
)
from src.utils.config_loader import load_config


class CoreFigureGenerationError(RuntimeError):
    """Raised when a declared source or rendered figure violates the contract."""


_MODEL_ORDER = ("xgboost", "logistic_regression", "random_forest", "lightgbm")
_MODEL_LABELS = {
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
}
_COLORS = {
    "xgboost": "#176B87",
    "logistic_regression": "#7A5195",
    "random_forest": "#2F7D32",
    "lightgbm": "#D17A22",
    "raw": "#6B7280",
    "sigmoid": "#176B87",
}
_SVG_NUMBER = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)(?:px|pt|pc|mm|cm|in)?\s*$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CoreFigureGenerationError(message)


def _is_linklike(path: Path) -> bool:
    try:
        details = path.lstat()
    except OSError:
        return False
    return path.is_symlink() or bool(getattr(details, "st_file_attributes", 0) & 0x400)


def _portable_path(value: Any, *, context: str) -> str:
    _require(
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and "\\" not in value,
        f"{context} must be a non-empty portable relative path.",
    )
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and not value.startswith("./")
        and all(part not in {"", ".", ".."} for part in path.parts)
        and ":" not in path.parts[0],
        f"{context} must be a contained portable relative path.",
    )
    return value


def _source_path(run_root: Path, configured: str) -> Path:
    relative = _portable_path(configured, context="Figure source")
    path = run_root / Path(relative)
    _require(path.is_file() and not _is_linklike(path), f"Figure source is missing or link-like: {relative}")
    resolved_run = run_root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_run)
    except ValueError as exc:
        raise CoreFigureGenerationError(f"Figure source escapes the active run root: {relative}") from exc
    _require(resolved.stat().st_size > 0, f"Figure source is empty: {relative}")
    return resolved


def _read_json(path: Path, *, context: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CoreFigureGenerationError(f"Cannot parse {context}: {exc}") from exc
    _require(isinstance(payload, Mapping), f"{context} must be a JSON object.")
    return payload


def _read_csv(path: Path, *, context: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise CoreFigureGenerationError(f"Cannot parse {context}: {exc}") from exc
    _require(not frame.empty, f"{context} has no rows.")
    return frame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, context: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    _require(not missing, f"{context} omits required columns: {missing}")


def _check_frame_identity(
    frame: pd.DataFrame,
    identity: Mapping[str, str],
    *,
    context: str,
) -> None:
    for field in ("run_id", "config_hash", "scientific_input_hash"):
        _require(field in frame.columns, f"{context} omits {field}.")
        observed = set(frame[field].astype(str))
        _require(observed == {identity[field]}, f"{context} has mixed or wrong {field}: {sorted(observed)}")


def _check_json_identity(
    payload: Mapping[str, Any],
    identity: Mapping[str, str],
    *,
    context: str,
) -> None:
    for field in ("run_id", "config_hash", "scientific_input_hash"):
        _require(payload.get(field) == identity[field], f"{context} has the wrong {field}.")


def _numeric(frame: pd.DataFrame, columns: Sequence[str], *, context: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise CoreFigureGenerationError(f"{context} has a nonnumeric {column} value.") from exc
        values = result[column].to_numpy(dtype=float)
        _require(np.isfinite(values).all(), f"{context} has a nonfinite {column} value.")
    return result


def _records(
    frame: pd.DataFrame,
    *,
    record_type: str,
    source_path: str,
    identity: Mapping[str, str],
) -> pd.DataFrame:
    result = frame.copy()
    for field in CORE_FIGURE_IDENTITY_FIELDS:
        result[field] = identity[field]
    result["record_type"] = record_type
    result["source_path"] = source_path
    leading = [*CORE_FIGURE_IDENTITY_FIELDS, "record_type", "source_path"]
    return result.loc[:, [*leading, *(column for column in result.columns if column not in leading)]]


def _metadata_record(
    payload: Mapping[str, Any],
    *,
    fields: Sequence[str],
    record_type: str,
    source_path: str,
    identity: Mapping[str, str],
) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for field in fields:
        value = payload.get(field)
        if isinstance(value, (Mapping, list, tuple)):
            value = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        row[field] = value
    return _records(
        pd.DataFrame([row]),
        record_type=record_type,
        source_path=source_path,
        identity=identity,
    )


def _combine_records(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    _require(bool(frames), "Figure source-data assembly received no frames.")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    _require(not combined.empty, "Figure source-data assembly produced no rows.")
    return combined


def _figure_1(
    sources: Mapping[str, Path],
    identity: Mapping[str, str],
) -> tuple[pd.DataFrame, plt.Figure]:
    fold_path = "shared_folds/fold_contract.json"
    folds = _read_json(sources[fold_path], context="shared-fold contract")
    _check_json_identity(folds, identity, context="shared-fold contract")
    outer = folds.get("outer_splits")
    inner = folds.get("inner_splits")
    _require(outer == 10 and inner == 5, "Figure 1 requires the frozen 10 outer x 5 inner fold contract.")
    rows = [
        (1, "inputs", "Immutable inputs", "Raw bytes, canonical content, schema, policy, and side-input receipts"),
        (2, "folds", "Shared nested resampling", f"{outer} outer folds x {inner} inner folds; macro-F1 selection, QWK tie-break"),
        (3, "benchmark", "Predeclared benchmark", "XGBoost reference plus logistic regression, random forest, and LightGBM"),
        (4, "uncertainty", "Paired OOF uncertainty", "Sample-level stratified bootstrap with 5,000 shared draws"),
        (5, "calibration", "Cross-fitted calibration", "Predeclared sigmoid calibration bound to exact outer-fold models"),
        (6, "xai", "Exact-fold OOF XAI", "Grouped SHAP from the same persisted outer-fold model as each prediction"),
        (7, "diagnostics", "Bounded diagnostics", "Support-aware subgroup/proxy summaries and independent mapped-target replication"),
    ]
    source = pd.DataFrame(rows, columns=["step_order", "step_id", "label", "detail"])
    source = _records(
        source,
        record_type="protocol_step",
        source_path=";".join(sources),
        identity=identity,
    )

    fig, axis = plt.subplots(figsize=(12, 7.2), constrained_layout=True)
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 8)
    axis.axis("off")
    positions = [(2.1, 6.8), (6.0, 6.8), (9.9, 6.8), (9.9, 3.9), (6.0, 3.9), (2.1, 3.9), (6.0, 1.0)]
    colors = ("#DCEAF1", "#CFE8DD", "#F7E3C3", "#EEE0F4", "#D7E6F5", "#F3DDD6", "#E5E7EB")
    for index, row in enumerate(source.itertuples(index=False)):
        x, y = positions[index]
        axis.text(
            x,
            y,
            f"{row.step_order}. {row.label}\n{row.detail}",
            ha="center",
            va="center",
            fontsize=9.2,
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.65", "facecolor": colors[index], "edgecolor": "#344054", "linewidth": 1.1},
        )
    for first, second in zip(positions[:-1], positions[1:]):
        axis.annotate("", xy=second, xytext=first, arrowprops={"arrowstyle": "->", "color": "#475467", "lw": 1.5, "shrinkA": 68, "shrinkB": 68})
    axis.set_title("Leakage-aware audit protocol and evidence flow", fontsize=15, weight="bold", pad=14)
    axis.text(6, 0.15, "Research evidence only - no autonomous HR decision support", ha="center", fontsize=9, color="#7A271A")
    return source, fig


def _figure_2(
    sources: Mapping[str, Path],
    identity: Mapping[str, str],
) -> tuple[pd.DataFrame, plt.Figure]:
    configured = "policy_ablation/figure_leakage_policy_tradeoff_source.csv"
    frame = _read_csv(sources[configured], context="feature-policy trade-off source")
    _check_frame_identity(frame, identity, context="feature-policy trade-off source")
    required = (
        "policy",
        "policy_order",
        "role",
        "audit_only",
        "macro_f1_oof",
        "macro_f1_ci_low",
        "macro_f1_ci_high",
        "quadratic_weighted_kappa_oof",
        "ordinal_mae_oof",
    )
    _require_columns(frame, required, context="feature-policy trade-off source")
    frame = _numeric(
        frame,
        ("policy_order", "macro_f1_oof", "macro_f1_ci_low", "macro_f1_ci_high", "quadratic_weighted_kappa_oof", "ordinal_mae_oof"),
        context="feature-policy trade-off source",
    ).sort_values("policy_order")
    _require(frame["policy"].astype(str).is_unique, "Feature-policy source contains duplicate policies.")
    source = _records(frame, record_type="policy_interval", source_path=configured, identity=identity)

    labels = [value.replace("_", " ") for value in frame["policy"].astype(str)]
    positions = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), constrained_layout=True)
    for position, row in enumerate(frame.itertuples(index=False)):
        policy = str(row.policy)
        color = "#B33A3A" if policy == "full_feature_upper_bound" else ("#176B87" if str(row.role) == "canonical_primary" else "#D17A22" if bool(row.audit_only) else "#3A7D44")
        axes[0].errorbar(
            float(row.macro_f1_oof),
            position,
            xerr=[[float(row.macro_f1_oof - row.macro_f1_ci_low)], [float(row.macro_f1_ci_high - row.macro_f1_oof)]],
            fmt="o",
            color=color,
            capsize=3,
        )
        axes[1].scatter(float(row.ordinal_mae_oof), float(row.quadratic_weighted_kappa_oof), color=color, s=52)
        axes[1].annotate(labels[position], (float(row.ordinal_mae_oof), float(row.quadratic_weighted_kappa_oof)), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("OOF macro-F1 (paired-bootstrap 95% CI)")
    axes[0].set_title("Predictive interval by policy")
    axes[1].set_xlabel("Ordinal MAE (lower is better)")
    axes[1].set_ylabel("Quadratic weighted kappa")
    axes[1].set_title("Ordinal trade-off")
    fig.suptitle("Feature-policy and leakage-risk sensitivity", fontsize=15, weight="bold")
    return source, fig


def _figure_3(
    sources: Mapping[str, Path],
    identity: Mapping[str, str],
) -> tuple[pd.DataFrame, plt.Figure]:
    summary_path = "model_benchmarks/model_summary.csv"
    paired_path = "model_benchmarks/paired_model_differences.csv"
    gate_path = "model_benchmarks/baseline_xgboost_gate.json"
    summary = _read_csv(sources[summary_path], context="model summary")
    paired = _read_csv(sources[paired_path], context="paired model differences")
    gate = _read_json(sources[gate_path], context="baseline XGBoost gate")
    _check_frame_identity(summary, identity, context="model summary")
    _check_frame_identity(paired, identity, context="paired model differences")
    _check_json_identity(gate, identity, context="baseline XGBoost gate")
    _require_columns(summary, ("system_id", "metric", "point_estimate", "ci_low", "ci_high"), context="model summary")
    _require_columns(paired, ("comparison_id", "metric", "improvement_oriented_difference", "improvement_ci_low", "improvement_ci_high"), context="paired model differences")
    summary = _numeric(summary, ("point_estimate", "ci_low", "ci_high"), context="model summary")
    selected = summary[summary["metric"].isin(("macro_f1", "quadratic_weighted_kappa"))].copy()
    expected_pairs = {(model, metric) for model in _MODEL_ORDER for metric in ("macro_f1", "quadratic_weighted_kappa")}
    observed_pairs = set(zip(selected["system_id"].astype(str), selected["metric"].astype(str)))
    _require(observed_pairs == expected_pairs and len(selected) == 8, "Model summary must contain exactly four models for macro-F1 and QWK.")
    gate_row = _metadata_record(
        gate,
        fields=("gate_metric", "comparison_direction", "trigger_rule", "gate_triggered", "triggered_comparisons", "n_resamples", "resample_hash"),
        record_type="baseline_gate",
        source_path=gate_path,
        identity=identity,
    )
    source = _combine_records(
        (
            _records(selected, record_type="model_interval", source_path=summary_path, identity=identity),
            _records(paired, record_type="paired_difference", source_path=paired_path, identity=identity),
            gate_row,
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True, sharey=True)
    positions = np.arange(len(_MODEL_ORDER))
    for axis, metric, title in zip(axes, ("macro_f1", "quadratic_weighted_kappa"), ("Macro-F1", "Quadratic weighted kappa")):
        metric_rows = selected[selected["metric"] == metric].set_index("system_id").loc[list(_MODEL_ORDER)]
        for position, (model, row) in enumerate(metric_rows.iterrows()):
            axis.errorbar(
                float(row["point_estimate"]),
                position,
                xerr=[[float(row["point_estimate"] - row["ci_low"])], [float(row["ci_high"] - row["point_estimate"])]],
                fmt="o",
                color=_COLORS[model],
                capsize=4,
                markersize=7,
            )
        axis.set_yticks(positions, [_MODEL_LABELS[value] for value in _MODEL_ORDER])
        axis.invert_yaxis()
        axis.set_xlabel("OOF point estimate (paired-bootstrap 95% CI)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("XGBoost reference versus predeclared baselines", fontsize=15, weight="bold")
    return source, fig


def _figure_4(
    sources: Mapping[str, Path],
    identity: Mapping[str, str],
) -> tuple[pd.DataFrame, plt.Figure]:
    bins_path = "sigmoid_calibration/calibration_bins.csv"
    comparison_path = "sigmoid_calibration/calibration_method_comparison.csv"
    intervals_path = "sigmoid_calibration/calibration_metric_intervals.csv"
    paired_path = "sigmoid_calibration/calibration_paired_differences.csv"
    metadata_path = "sigmoid_calibration/calibration_figure_source.json"
    bins = _read_csv(sources[bins_path], context="calibration bins")
    comparison = _read_csv(sources[comparison_path], context="calibration method comparison")
    intervals = _read_csv(sources[intervals_path], context="calibration metric intervals")
    paired = _read_csv(sources[paired_path], context="calibration paired differences")
    metadata = _read_json(sources[metadata_path], context="calibration figure source")
    for frame, context in ((bins, "calibration bins"), (comparison, "calibration method comparison"), (intervals, "calibration metric intervals"), (paired, "calibration paired differences")):
        _check_frame_identity(frame, identity, context=context)
    _check_json_identity(metadata, identity, context="calibration figure source")
    _require_columns(bins, ("method", "class_label", "mean_predicted_probability", "observed_frequency", "n_samples"), context="calibration bins")
    _require_columns(intervals, ("system_id", "metric", "point_estimate", "ci_low", "ci_high"), context="calibration metric intervals")
    bins = _numeric(bins, ("n_samples",), context="calibration bins")
    observed_bins = bins[bins["n_samples"].astype(float) > 0].copy()
    _require(not observed_bins.empty, "Calibration bins contain no observed rows.")
    observed_bins = _numeric(
        observed_bins,
        ("mean_predicted_probability", "observed_frequency"),
        context="observed calibration bins",
    )
    intervals = _numeric(intervals, ("point_estimate", "ci_low", "ci_high"), context="calibration metric intervals")
    _require(set(bins["method"].astype(str)) == {"raw", "sigmoid"}, "Calibration bins must contain exactly raw and sigmoid methods.")
    probability_metrics = ("nll_log_loss", "multiclass_brier", "ece_confidence")
    selected = intervals[intervals["metric"].isin(probability_metrics)].copy()
    expected = {(system, metric) for system in ("raw", "sigmoid") for metric in probability_metrics}
    observed = set(zip(selected["system_id"].astype(str), selected["metric"].astype(str)))
    _require(observed == expected and len(selected) == 6, "Calibration intervals must contain raw and sigmoid NLL, Brier, and ECE rows.")
    source = _combine_records(
        (
            _records(bins, record_type="reliability_bin", source_path=bins_path, identity=identity),
            _records(comparison, record_type="method_comparison", source_path=comparison_path, identity=identity),
            _records(intervals, record_type="metric_interval", source_path=intervals_path, identity=identity),
            _records(paired, record_type="paired_difference", source_path=paired_path, identity=identity),
            _metadata_record(metadata, fields=("primary_method", "method_order", "class_order", "n_bins", "panel_metric_order", "caption_warning"), record_type="figure_metadata", source_path=metadata_path, identity=identity),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.9), constrained_layout=True)
    axes[0].plot((0, 1), (0, 1), color="#98A2B3", linestyle="--", linewidth=1)
    for method in ("raw", "sigmoid"):
        rows = observed_bins[observed_bins["method"].astype(str) == method]
        for class_index, (_, class_rows) in enumerate(rows.groupby("class_label", sort=True)):
            ordered = class_rows.sort_values("mean_predicted_probability")
            axes[0].plot(
                ordered["mean_predicted_probability"],
                ordered["observed_frequency"],
                marker="o",
                markersize=3,
                linewidth=1,
                alpha=0.45,
                color=_COLORS[method],
                label=method.capitalize() if class_index == 0 else None,
            )
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed frequency", title="Classwise reliability")
    axes[0].legend(frameon=False)
    metric_labels = {"nll_log_loss": "NLL", "multiclass_brier": "Brier", "ece_confidence": "ECE"}
    positions = np.arange(len(probability_metrics), dtype=float)
    offsets = {"raw": -0.12, "sigmoid": 0.12}
    for method in ("raw", "sigmoid"):
        rows = selected[selected["system_id"].astype(str) == method].set_index("metric").loc[list(probability_metrics)]
        axes[1].errorbar(
            rows["point_estimate"].to_numpy(float),
            positions + offsets[method],
            xerr=np.vstack((rows["point_estimate"].to_numpy(float) - rows["ci_low"].to_numpy(float), rows["ci_high"].to_numpy(float) - rows["point_estimate"].to_numpy(float))),
            fmt="o",
            capsize=3,
            color=_COLORS[method],
            label=method.capitalize(),
        )
    axes[1].set_yticks(positions, [metric_labels[value] for value in probability_metrics])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Metric value (paired-bootstrap 95% CI; lower is better)")
    axes[1].set_title("Probability-quality metrics")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("Predeclared cross-fitted sigmoid calibration", fontsize=15, weight="bold")
    return source, fig


def _figure_5(
    sources: Mapping[str, Path],
    identity: Mapping[str, str],
) -> tuple[pd.DataFrame, plt.Figure]:
    importance_path = "oof_shap/global_grouped_shap_importance.csv"
    metadata_path = "oof_shap/shap_metadata.json"
    importance = _read_csv(sources[importance_path], context="global grouped OOF SHAP")
    metadata = _read_json(sources[metadata_path], context="SHAP metadata")
    _check_frame_identity(importance, identity, context="global grouped OOF SHAP")
    _check_json_identity(metadata, identity, context="SHAP metadata")
    _require_columns(importance, ("feature", "mean_abs_grouped_shap", "rank"), context="global grouped OOF SHAP")
    importance = _numeric(importance, ("mean_abs_grouped_shap", "rank"), context="global grouped OOF SHAP").sort_values("rank")
    _require(importance["feature"].astype(str).is_unique, "Global grouped OOF SHAP has duplicate features.")
    source = _combine_records(
        (
            _records(importance, record_type="global_grouped_importance", source_path=importance_path, identity=identity),
            _metadata_record(metadata, fields=("n_samples", "n_outer_folds", "n_raw_features", "model_set_sha256", "attribution_warning", "temporality_warning"), record_type="shap_metadata", source_path=metadata_path, identity=identity),
        )
    )
    top = importance.head(15).sort_values("mean_abs_grouped_shap")
    fig, axis = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
    axis.barh(top["feature"].astype(str), top["mean_abs_grouped_shap"].astype(float), color="#176B87")
    axis.set_xlabel("Mean absolute grouped SHAP value across exact-fold OOF cases")
    axis.set_ylabel("Canonical feature group")
    axis.set_title("Global grouped out-of-fold SHAP attribution", fontsize=15, weight="bold")
    axis.grid(axis="x", alpha=0.2)
    return source, fig


def _figure_6(
    sources: Mapping[str, Path],
    identity: Mapping[str, str],
) -> tuple[pd.DataFrame, plt.Figure]:
    rankings_path = "oof_shap/fold_feature_rankings.csv"
    pairwise_path = "oof_shap/shap_stability_pairwise.csv"
    summary_path = "oof_shap/shap_stability_summary.csv"
    metadata_path = "oof_shap/shap_metadata.json"
    rankings = _read_csv(sources[rankings_path], context="fold SHAP rankings")
    pairwise = _read_csv(sources[pairwise_path], context="pairwise SHAP stability")
    summary = _read_csv(sources[summary_path], context="SHAP stability summary")
    metadata = _read_json(sources[metadata_path], context="SHAP metadata")
    for frame, context in ((rankings, "fold SHAP rankings"), (pairwise, "pairwise SHAP stability"), (summary, "SHAP stability summary")):
        _check_frame_identity(frame, identity, context=context)
    _check_json_identity(metadata, identity, context="SHAP metadata")
    _require_columns(rankings, ("outer_fold", "feature", "rank"), context="fold SHAP rankings")
    _require_columns(pairwise, ("outer_fold_a", "outer_fold_b", "top_k", "top_k_jaccard", "spearman_all_features"), context="pairwise SHAP stability")
    _require_columns(
        summary,
        (
            "top_k",
            "n_fold_pairs",
            "jaccard_mean",
            "jaccard_min",
            "jaccard_max",
            "spearman_mean",
            "spearman_min",
            "spearman_max",
            "confidence_interval_applicable",
        ),
        context="SHAP stability summary",
    )
    rankings = _numeric(rankings, ("outer_fold", "rank"), context="fold SHAP rankings")
    pairwise = _numeric(pairwise, ("outer_fold_a", "outer_fold_b", "top_k", "top_k_jaccard", "spearman_all_features"), context="pairwise SHAP stability")
    summary = _numeric(
        summary,
        ("top_k", "n_fold_pairs", "jaccard_mean", "jaccard_min", "jaccard_max", "spearman_mean", "spearman_min", "spearman_max"),
        context="SHAP stability summary",
    )
    _require(set(rankings["outer_fold"].astype(int)) == set(range(1, 11)), "SHAP rankings must contain all ten outer folds.")
    _require((summary["n_fold_pairs"].astype(int) == 45).all(), "SHAP stability summaries require all 45 dependent fold pairs.")
    _require(not summary["confidence_interval_applicable"].astype(str).str.casefold().isin(("true", "1")).any(), "Dependent fold-pair stability may not claim confidence intervals.")
    source = _combine_records(
        (
            _records(rankings, record_type="fold_ranking", source_path=rankings_path, identity=identity),
            _records(pairwise, record_type="dependent_fold_pair", source_path=pairwise_path, identity=identity),
            _records(summary, record_type="descriptive_summary", source_path=summary_path, identity=identity),
            _metadata_record(metadata, fields=("n_samples", "n_outer_folds", "model_set_sha256", "confidence_interval_for_fold_pairs", "attribution_warning"), record_type="shap_metadata", source_path=metadata_path, identity=identity),
        )
    )

    feature_order = rankings.groupby("feature", sort=False)["rank"].median().sort_values().head(12).index.tolist()
    pivot = rankings[rankings["feature"].isin(feature_order)].pivot(index="feature", columns="outer_fold", values="rank").reindex(feature_order)
    _require(not pivot.isna().any().any(), "SHAP fold-ranking matrix is incomplete for selected features.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), constrained_layout=True)
    image = axes[0].imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis_r")
    axes[0].set_yticks(np.arange(len(feature_order)), feature_order)
    axes[0].set_xticks(np.arange(10), [str(value) for value in range(1, 11)])
    axes[0].set_xlabel("Outer fold")
    axes[0].set_title("Feature rank by exact outer-fold model")
    fig.colorbar(image, ax=axes[0], label="Rank (lower is more important)")
    display_rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        display_rows.extend(
            (
                {
                    "label": f"top-k Jaccard\n(k={int(row.top_k)})",
                    "mean": float(row.jaccard_mean),
                    "minimum": float(row.jaccard_min),
                    "maximum": float(row.jaccard_max),
                },
                {
                    "label": f"Spearman all features\n(k={int(row.top_k)})",
                    "mean": float(row.spearman_mean),
                    "minimum": float(row.spearman_min),
                    "maximum": float(row.spearman_max),
                },
            )
        )
    display = pd.DataFrame(display_rows)
    y = np.arange(len(display))
    axes[1].errorbar(
        display["mean"].to_numpy(float),
        y,
        xerr=np.vstack((display["mean"].to_numpy(float) - display["minimum"].to_numpy(float), display["maximum"].to_numpy(float) - display["mean"].to_numpy(float))),
        fmt="o",
        color="#7A5195",
        capsize=3,
    )
    axes[1].set_yticks(y, display["label"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_xlabel("Descriptive fold-pair mean and observed range")
    axes[1].set_title("Dependent fold-pair stability (no CI)")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("Grouped OOF SHAP stability across outer folds", fontsize=15, weight="bold")
    return source, fig


def _figure_7(
    sources: Mapping[str, Path],
    identity: Mapping[str, str],
) -> tuple[pd.DataFrame, plt.Figure]:
    support_path = "external_replication/target_support.csv"
    raw_path = "external_replication/raw_metric_intervals.csv"
    calibration_path = "external_replication/calibration_metric_intervals.csv"
    calibration_diff_path = "external_replication/calibration_paired_differences.csv"
    policy_diff_path = "external_replication/policy_pairwise_differences.csv"
    metadata_path = "external_replication/external_replication_metadata.json"
    support = _read_csv(sources[support_path], context="external target support")
    raw = _read_csv(sources[raw_path], context="external raw intervals")
    calibration = _read_csv(sources[calibration_path], context="external calibration intervals")
    calibration_diff = _read_csv(sources[calibration_diff_path], context="external calibration differences")
    policy_diff = _read_csv(sources[policy_diff_path], context="external policy differences")
    metadata = _read_json(sources[metadata_path], context="external replication metadata")
    for frame, context in ((support, "external target support"), (raw, "external raw intervals"), (calibration, "external calibration intervals"), (calibration_diff, "external calibration differences"), (policy_diff, "external policy differences")):
        _check_frame_identity(frame, identity, context=context)
    _check_json_identity(metadata, identity, context="external replication metadata")
    _require_columns(support, ("support_scale", "target_column", "target_value", "count", "proportion", "n_total"), context="external target support")
    for frame, context in ((raw, "external raw intervals"), (calibration, "external calibration intervals")):
        _require_columns(frame, ("system_id", "metric", "point_estimate", "ci_low", "ci_high", "n_samples", "n_resamples"), context=context)
    _require_columns(policy_diff, ("comparison_id", "metric", "improvement_oriented_difference", "improvement_ci_low", "improvement_ci_high"), context="external policy differences")
    support = _numeric(support, ("count", "proportion", "n_total"), context="external target support")
    raw = _numeric(raw, ("point_estimate", "ci_low", "ci_high", "n_samples", "n_resamples"), context="external raw intervals")
    calibration = _numeric(calibration, ("point_estimate", "ci_low", "ci_high", "n_samples", "n_resamples"), context="external calibration intervals")
    policy_diff = _numeric(policy_diff, ("improvement_oriented_difference", "improvement_ci_low", "improvement_ci_high"), context="external policy differences")
    mapped = support[support["support_scale"].astype(str) == "mapped"].sort_values("target_value")
    _require(set(mapped["target_value"].astype(str)) == {"2", "3", "4"} and len(mapped) == 3, "External mapped-target support must contain exactly labels 2, 3, and 4.")
    _require(mapped["n_total"].nunique() == 1 and int(mapped["count"].sum()) == int(mapped.iloc[0]["n_total"]), "External mapped-target support denominator is inconsistent.")
    metrics = ("macro_f1", "quadratic_weighted_kappa")
    primary = raw[(raw["system_id"].astype(str) == "conservative_primary") & raw["metric"].isin(metrics)].copy()
    calibrated = calibration[calibration["metric"].isin(metrics)].copy()
    _require(set(primary["metric"].astype(str)) == set(metrics) and len(primary) == 2, "External primary result omits macro-F1 or QWK.")
    _require(set(calibrated["system_id"].astype(str)) == {"raw", "sigmoid"} and len(calibrated) == 4, "External calibration result must contain raw and sigmoid macro-F1/QWK rows.")
    policy_macro = policy_diff[policy_diff["metric"].astype(str) == "macro_f1"].copy()
    _require(not policy_macro.empty, "External policy differences omit macro-F1.")
    source = _combine_records(
        (
            _records(support, record_type="target_support", source_path=support_path, identity=identity),
            _records(raw, record_type="raw_metric_interval", source_path=raw_path, identity=identity),
            _records(calibration, record_type="calibration_metric_interval", source_path=calibration_path, identity=identity),
            _records(calibration_diff, record_type="calibration_paired_difference", source_path=calibration_diff_path, identity=identity),
            _records(policy_diff, record_type="policy_paired_difference", source_path=policy_diff_path, identity=identity),
            _metadata_record(metadata, fields=("scope", "role", "task_type", "labels", "primary_policy", "claim_boundary", "network_calls", "paid_api_calls"), record_type="external_metadata", source_path=metadata_path, identity=identity),
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.8), constrained_layout=True)
    axes[0].bar(mapped["target_value"].astype(str), mapped["count"].astype(float), color=("#7A5195", "#176B87", "#D17A22"))
    axes[0].set_xlabel("Mapped performance rating")
    axes[0].set_ylabel("Employees")
    axes[0].set_title(f"Mapped-target support (n={int(mapped.iloc[0]['n_total'])})")
    positions = np.arange(len(metrics))
    offsets = {"raw": -0.12, "sigmoid": 0.12}
    for method in ("raw", "sigmoid"):
        rows = calibrated[calibrated["system_id"].astype(str) == method].set_index("metric").loc[list(metrics)]
        axes[1].errorbar(
            rows["point_estimate"].to_numpy(float),
            positions + offsets[method],
            xerr=np.vstack((rows["point_estimate"].to_numpy(float) - rows["ci_low"].to_numpy(float), rows["ci_high"].to_numpy(float) - rows["point_estimate"].to_numpy(float))),
            fmt="o",
            capsize=3,
            color=_COLORS[method],
            label=method.capitalize(),
        )
    axes[1].set_yticks(positions, ("Macro-F1", "QWK"))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("OOF point estimate (paired-bootstrap 95% CI)")
    axes[1].set_title("Conservative-primary calibration")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="x", alpha=0.2)
    policy_macro = policy_macro.sort_values("comparison_id")
    labels = [str(value).replace("_minus_conservative_primary", "").replace("_", " ") for value in policy_macro["comparison_id"]]
    y = np.arange(len(policy_macro))
    axes[2].axvline(0, color="#98A2B3", linestyle="--", linewidth=1)
    axes[2].errorbar(
        policy_macro["improvement_oriented_difference"].to_numpy(float),
        y,
        xerr=np.vstack((policy_macro["improvement_oriented_difference"].to_numpy(float) - policy_macro["improvement_ci_low"].to_numpy(float), policy_macro["improvement_ci_high"].to_numpy(float) - policy_macro["improvement_oriented_difference"].to_numpy(float))),
        fmt="o",
        capsize=3,
        color="#D17A22",
    )
    axes[2].set_yticks(y, labels)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Macro-F1 difference versus conservative primary")
    axes[2].set_title("Audit-policy sensitivity")
    axes[2].grid(axis="x", alpha=0.2)
    fig.suptitle("HRDataset_v14 independent mapped-target replication", fontsize=15, weight="bold")
    return source, fig


_BUILDERS: Mapping[str, Callable[[Mapping[str, Path], Mapping[str, str]], tuple[pd.DataFrame, plt.Figure]]] = {
    "figure_1": _figure_1,
    "figure_2": _figure_2,
    "figure_3": _figure_3,
    "figure_4": _figure_4,
    "figure_5": _figure_5,
    "figure_6": _figure_6,
    "figure_7": _figure_7,
}


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    _require(len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR", f"Rendered PNG is malformed: {path.name}")
    return struct.unpack(">II", header[16:24])


def _svg_dimensions(path: Path) -> tuple[float, float]:
    try:
        root = ElementTree.fromstring(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ElementTree.ParseError) as exc:
        raise CoreFigureGenerationError(f"Rendered SVG is malformed: {path.name}: {exc}") from exc
    values: list[float] = []
    for field in ("width", "height"):
        match = _SVG_NUMBER.fullmatch(str(root.attrib.get(field, "")))
        _require(match is not None, f"Rendered SVG omits a numeric {field}: {path.name}")
        values.append(float(match.group(1)))
    return values[0], values[1]


def _save_figure(
    figure: plt.Figure,
    root: Path,
    stem: str,
    *,
    dpi: int,
) -> tuple[Path, Path, tuple[int, int], tuple[float, float]]:
    png = root / f"{stem}.png"
    svg = root / f"{stem}.svg"
    try:
        figure.savefig(
            png,
            dpi=dpi,
            facecolor="white",
            metadata={"Software": "employee-performance-with-XAI production core-figure generator"},
        )
        figure.savefig(
            svg,
            format="svg",
            facecolor="white",
            metadata={"Creator": "employee-performance-with-XAI production core-figure generator", "Date": None},
        )
    finally:
        plt.close(figure)
    svg_text = svg.read_text(encoding="utf-8")
    sanitized_svg = re.sub(
        r"<!DOCTYPE[^>]*>\s*",
        "",
        svg_text,
        count=1,
        flags=re.IGNORECASE,
    )
    _require(
        "<!DOCTYPE" not in sanitized_svg.upper() and "<!ENTITY" not in sanitized_svg.upper(),
        f"Rendered SVG contains a prohibited declaration: {stem}",
    )
    svg.write_text(sanitized_svg, encoding="utf-8", newline="\n")
    _require(png.is_file() and png.stat().st_size > 0, f"PNG rendering failed: {stem}")
    _require(svg.is_file() and svg.stat().st_size > 0, f"SVG rendering failed: {stem}")
    return png, svg, _png_dimensions(png), _svg_dimensions(svg)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _source_records(
    definition: Mapping[str, Any],
    sources: Mapping[str, Path],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for declared in definition["sources"]:
        configured = str(declared["path"])
        path = sources[configured]
        records.append(
            {
                "stage": str(declared["stage"]),
                "path": configured,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def run(
    config_path: str | Path,
    *,
    run_root: str | Path,
    output_dir: str | Path,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    source_tree_hash: str,
) -> Mapping[str, Any]:
    """Render and atomically publish the exact seven-figure v2 package."""

    config = load_config(config_path)
    _require(canonical_config_hash(config) == config_hash, "Core-figure config hash differs from the active run identity.")
    settings = manuscript_settings(config)
    plan = settings.get("figures")
    _require(isinstance(plan, Mapping) and dict(plan) == expected_core_figure_plan(), "Configured core-figure plan differs from the frozen v2 contract.")
    identity = {
        "run_id": run_id,
        "config_hash": config_hash,
        "scientific_input_hash": scientific_input_hash,
        "source_tree_hash": source_tree_hash,
    }
    _require(isinstance(run_id, str) and bool(run_id), "run_id must be non-empty.")
    for field in ("config_hash", "scientific_input_hash", "source_tree_hash"):
        value = identity[field]
        _require(isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value), f"{field} must be a lowercase SHA-256 digest.")

    run_path = Path(run_root)
    output = Path(output_dir)
    _require(run_path.is_dir() and not _is_linklike(run_path), "Active run root is absent or link-like.")
    _require(output.resolve() == (run_path / "core_figures").resolve(), "Core figures must publish to the exact active-run core_figures directory.")
    _require(not output.exists(), f"Core-figure output already exists and may not be overwritten: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    definitions = plan["definitions"]
    _require(tuple(definitions) == CORE_FIGURE_KEYS and set(_BUILDERS) == set(CORE_FIGURE_KEYS), "Core-figure definitions or builders are incomplete.")
    all_sources: dict[str, Path] = {}
    for definition in definitions.values():
        for declared in definition["sources"]:
            configured = str(declared["path"])
            all_sources.setdefault(configured, _source_path(run_path, configured))

    staging = Path(tempfile.mkdtemp(prefix=f"{output.name}.__staging__.", dir=output.parent))
    try:
        (staging / str(plan["source_data_subdirectory"])).mkdir()
        (staging / str(plan["caption_subdirectory"])).mkdir()
        manifest_rows: list[dict[str, Any]] = []
        with matplotlib.rc_context(
            {
                "font.family": str(plan["font_family"]),
                "font.size": 9.5,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "svg.hashsalt": "leakage-aware-core-figures-v2",
            }
        ):
            for key in CORE_FIGURE_KEYS:
                definition = definitions[key]
                declared_sources = {
                    str(source["path"]): all_sources[str(source["path"])]
                    for source in definition["sources"]
                }
                source_frame, figure = _BUILDERS[key](declared_sources, identity)
                source_path = staging / str(plan["source_data_subdirectory"]) / str(definition["source_data_filename"])
                source_frame.to_csv(source_path, index=False, lineterminator="\n")
                _require(source_path.stat().st_size > 0, f"{key} source table is empty.")
                caption_path = staging / str(plan["caption_subdirectory"]) / str(definition["caption_filename"])
                caption_path.write_text(
                    f"Technical evidence caption for Figure {definition['number']}: {definition['title']}.\n\n"
                    f"Claim boundary: {definition['claim_boundary']}\n\n"
                    + "; ".join(f"{field}={identity[field]}" for field in CORE_FIGURE_IDENTITY_FIELDS)
                    + "\n\nSources: "
                    + "; ".join(declared_sources)
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                stem = str(definition["output_stem"])
                png, svg, png_dimensions, svg_dimensions = _save_figure(
                    figure,
                    staging,
                    stem,
                    dpi=int(plan["publication_dpi"]),
                )
                manifest_rows.append(
                    {
                        "figure_key": key,
                        "number": int(definition["number"]),
                        "figure_id": str(definition["figure_id"]),
                        "output_stem": stem,
                        "png_path": png.relative_to(staging).as_posix(),
                        "png_sha256": sha256_file(png),
                        "png_size_bytes": png.stat().st_size,
                        "png_width_px": png_dimensions[0],
                        "png_height_px": png_dimensions[1],
                        "svg_path": svg.relative_to(staging).as_posix(),
                        "svg_sha256": sha256_file(svg),
                        "svg_size_bytes": svg.stat().st_size,
                        "svg_width_px": svg_dimensions[0],
                        "svg_height_px": svg_dimensions[1],
                        "source_data_path": source_path.relative_to(staging).as_posix(),
                        "source_data_sha256": sha256_file(source_path),
                        "source_data_size_bytes": source_path.stat().st_size,
                        "caption_path": caption_path.relative_to(staging).as_posix(),
                        "caption_sha256": sha256_file(caption_path),
                        "caption_size_bytes": caption_path.stat().st_size,
                        "sources": _source_records(definition, declared_sources),
                    }
                )
        manifest_path = staging / "figure_manifest.json"
        _write_json(
            manifest_path,
            {
                "schema_version": 1,
                "manifest_kind": "core_figure_package",
                "status": "complete",
                "inventory_mode": "closed_world",
                "path_basis": "core_figures_relative",
                "plan_version": CORE_FIGURE_PLAN_VERSION,
                "stage": "core_figures",
                "scope": "core",
                "hash_algorithm": "sha256",
                "n_figures": 7,
                **identity,
                "figures": manifest_rows,
            },
        )
        files = [path for path in staging.rglob("*") if path.is_file()]
        _require(len(files) == 29, f"Core-figure stage must contain exactly 29 runner-owned artifacts, found {len(files)}.")
        _require(not output.exists(), "Core-figure output appeared during generation; refusing to overwrite it.")
        os.replace(staging, output)
    except Exception:
        # Preserve failed staging output for forensic recovery.  The builder
        # refuses to run while a matching orphan directory remains.
        raise

    published = sorted((path for path in output.rglob("*") if path.is_file()), key=lambda value: value.relative_to(output).as_posix())
    return {
        "output": output,
        "manifest": output / "figure_manifest.json",
        "files": published,
    }


__all__ = ["CoreFigureGenerationError", "run"]
