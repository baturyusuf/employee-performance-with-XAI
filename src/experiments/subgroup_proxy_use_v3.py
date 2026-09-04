"""Run support-sensitive subgroup and performance-model proxy-use diagnostics."""

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

import numpy as np
import pandas as pd

from src.data.canonical_loader import load_canonical_dataset, sha256_file
from src.experiments.benchmark_artifact_contract import read_xgboost_oof_artifacts
from src.governance.manuscript_contract import source_tree_hash
from src.governance.offline_runtime import enforce_offline_runtime
from src.governance.subgroup_proxy_use_contract_v3 import (
    ATTRIBUTES,
    DEFAULT_SUBGROUP_PROXY_USE_CONTRACT,
    LABELS,
    METRICS,
    PRIMARY_SYSTEM,
    PROXY_REDUCED_SYSTEM,
    SYSTEMS,
    validate_subgroup_proxy_use_contract_v3,
)
from src.utils.config_loader import PROJECT_ROOT


DEFAULT_LOCAL_RUN_ROOT = Path("reports/major_revision_v3_runs")
PROBABILITY_COLUMNS = tuple(f"prob_class_{label}" for label in LABELS)
CANONICAL_ROOT = Path(
    "reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f/core"
)
EXPECTED_LOCAL_FILES = frozenset(
    {
        "subgroup_metric_grid.csv",
        "subgroup_gap_sensitivity.csv",
        "primary_gap_bootstrap_intervals.csv",
        "proxy_prediction_change_sample.csv",
        "proxy_prediction_change_by_department.csv",
        "jobrole_permutation_sample.csv",
        "jobrole_permutation_repetition.csv",
        "jobrole_permutation_summary.csv",
        "department_reconstructability_metrics.csv",
        "department_reconstructability_differences.csv",
        "diagnostic_receipt.json",
        "stage_metadata.json",
    }
)
HIGHER_IS_BETTER = {
    "macro_f1": True,
    "balanced_accuracy": True,
    "quadratic_weighted_kappa": True,
    "ordinal_mae": False,
    "recall_class_2": True,
    "recall_class_3": True,
    "recall_class_4": True,
    "multiclass_brier": False,
    "log_loss": False,
}
CLASS_SENSITIVE_SUPPORT_METRICS = {
    "balanced_accuracy",
    "quadratic_weighted_kappa",
}


class SubgroupProxyUseV3Error(RuntimeError):
    """Raised when Phase 2C violates its frozen diagnostic contract."""


@dataclass(frozen=True)
class Phase2CInputs:
    contract: Mapping[str, Any]
    contract_receipt: Mapping[str, Any]
    frame: pd.DataFrame
    features: pd.DataFrame
    oof: pd.DataFrame
    artifacts: Any


@dataclass(frozen=True)
class Phase2CResult:
    subgroup_metric_grid: pd.DataFrame
    subgroup_gap_sensitivity: pd.DataFrame
    primary_gap_bootstrap_intervals: pd.DataFrame
    proxy_prediction_change_sample: pd.DataFrame
    proxy_prediction_change_by_department: pd.DataFrame
    jobrole_permutation_sample: pd.DataFrame
    jobrole_permutation_repetition: pd.DataFrame
    jobrole_permutation_summary: pd.DataFrame
    department_reconstructability_metrics: pd.DataFrame
    department_reconstructability_differences: pd.DataFrame
    diagnostic_receipt: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SubgroupProxyUseV3Error(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SubgroupProxyUseV3Error(f"Could not read {path.as_posix()}: {exc}") from exc
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
        raise SubgroupProxyUseV3Error(f"Could not establish Git identity: {exc}") from exc
    _require(
        len(head) == 40 and all(character in "0123456789abcdef" for character in head),
        "Git HEAD digest is invalid.",
    )
    _require(
        not status,
        f"Scientific execution requires a clean worktree; status={status.splitlines()[:10]}.",
    )
    return {"commit": head, "branch": branch}


def _p3_features(frame: pd.DataFrame, feature_contract: Mapping[str, Any]) -> pd.DataFrame:
    all_features = [str(record["feature_name"]) for record in feature_contract["features"]]
    _require(set(all_features) == set(frame.columns), "Feature contract/source schema drifted.")
    policies = {str(record["policy_id"]): record for record in feature_contract["policies"]}
    excluded = set(map(str, policies["P3"]["excluded_features"]))
    retained = [feature for feature in all_features if feature not in excluded]
    _require(
        len(retained) == 20
        and "EmpJobRole" in retained
        and "EmpDepartment" not in retained
        and "PerformanceRating" not in retained,
        "P3 feature set drifted.",
    )
    return frame.loc[:, retained].copy()


def _prepare_inputs(contract_path: Path | str) -> Phase2CInputs:
    path = Path(contract_path)
    contract_receipt = validate_subgroup_proxy_use_contract_v3(path)
    full_path = path if path.is_absolute() else PROJECT_ROOT / path
    contract = _load_json(full_path)
    sources = contract["source_contracts"]
    canonical = load_canonical_dataset(
        sources["canonical_loader_config"]["path"],
        contract["dataset_key"],
        sources["acquisition_manifest"]["path"],
        allow_download=False,
    )
    _require(
        canonical.receipt["actual_sha256"] == contract["canonical_identity"]["dataset_sha256"],
        "Canonical dataset digest drifted.",
    )
    frame = canonical.frame.copy()
    _require(list(frame.index) == list(range(1200)), "Canonical sample index drifted.")
    feature_contract = _load_json(PROJECT_ROOT / sources["feature_availability"]["path"])
    features = _p3_features(frame, feature_contract)
    oof = pd.read_csv(PROJECT_ROOT / sources["fairness_oof_predictions"]["path"])
    identity = contract["canonical_identity"]
    artifacts = read_xgboost_oof_artifacts(
        (PROJECT_ROOT / sources["shared_fold_contract"]["path"]).parent,
        (PROJECT_ROOT / sources["benchmark_stage_metadata"]["path"]).parent,
        expected_run_id=identity["run_id"],
        expected_config_hash=identity["config_hash"],
        expected_scientific_input_hash=identity["scientific_input_hash"],
        expected_feature_columns=features.columns,
        expected_labels=LABELS,
    )
    _require(
        artifacts.model_set_sha256 == identity["xgboost_model_set_sha256"],
        "XGBoost model-set identity drifted.",
    )
    primary = oof[oof["system_id"] == PRIMARY_SYSTEM].sort_values("sample_index")
    benchmark = artifacts.oof_predictions.sort_values("sample_index")
    for column in ("sample_index", "outer_fold", "y_true", "y_pred"):
        _require(
            np.array_equal(primary[column].to_numpy(int), benchmark[column].to_numpy(int)),
            f"Primary fairness/benchmark OOF alignment drifted for {column}.",
        )
    _require(
        np.array_equal(
            primary[list(PROBABILITY_COLUMNS)].to_numpy(float),
            benchmark[list(PROBABILITY_COLUMNS)].to_numpy(float),
        ),
        "Primary fairness probabilities differ from exact benchmark OOF probabilities.",
    )
    return Phase2CInputs(
        contract=contract,
        contract_receipt=contract_receipt,
        frame=frame,
        features=features,
        oof=oof,
        artifacts=artifacts,
    )


def _group_series(frame: pd.DataFrame, attribute: str, contract: Mapping[str, Any]) -> pd.Series:
    if attribute == "Age":
        bins = contract["subgroup_audit"]["age_bins"]
        values = pd.cut(
            pd.to_numeric(frame["Age"], errors="raise"),
            bins=bins["edges"],
            labels=bins["labels"],
            right=bool(bins["right_closed"]),
            include_lowest=bool(bins["include_lowest"]),
        )
        _require(values.notna().all(), "One or more ages escaped the declared bins.")
        return values.astype("string").astype(str)
    return frame[attribute].astype("string").fillna("__MISSING__").astype(str)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_index = np.searchsorted(np.asarray(LABELS), y_true)
    pred_index = np.searchsorted(np.asarray(LABELS), y_pred)
    _require(
        np.all(np.asarray(LABELS)[true_index] == y_true)
        and np.all(np.asarray(LABELS)[pred_index] == y_pred),
        "Metric calculation received an undeclared label.",
    )
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(matrix, (true_index, pred_index), 1)
    return matrix


def metric_bundle_v3(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: np.ndarray,
) -> tuple[dict[str, float], dict[str, int]]:
    """Calculate the frozen ordinal/multiclass metric bundle and denominators."""

    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    _require(len(y) > 0 and pred.shape == y.shape, "Metric arrays must be aligned and nonempty.")
    _require(probs.shape == (len(y), len(LABELS)), "Probability matrix shape drifted.")
    _require(np.isfinite(probs).all(), "Probability matrix contains non-finite values.")
    confusion = _confusion(y, pred)
    actual = confusion.sum(axis=1).astype(float)
    predicted = confusion.sum(axis=0).astype(float)
    true_positive = np.diag(confusion).astype(float)
    recall = np.divide(
        true_positive,
        actual,
        out=np.full(len(LABELS), np.nan, dtype=float),
        where=actual > 0,
    )
    f1_denominator = 2.0 * true_positive + (predicted - true_positive) + (actual - true_positive)
    f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.zeros(len(LABELS), dtype=float),
        where=f1_denominator > 0,
    )
    balanced = float(np.mean(recall)) if np.isfinite(recall).all() else math.nan
    weights = np.square(
        np.arange(len(LABELS), dtype=float)[:, None]
        - np.arange(len(LABELS), dtype=float)[None, :]
    ) / float((len(LABELS) - 1) ** 2)
    expected = np.outer(actual, predicted) / float(len(y))
    observed_disagreement = float(np.sum(weights * confusion))
    expected_disagreement = float(np.sum(weights * expected))
    qwk = (
        float(1.0 - observed_disagreement / expected_disagreement)
        if expected_disagreement > 0.0
        else math.nan
    )
    true_indices = np.searchsorted(np.asarray(LABELS), y)
    clipped = np.clip(probs[np.arange(len(y)), true_indices], 1e-15, 1.0)
    one_hot = np.eye(len(LABELS), dtype=float)[true_indices]
    metrics = {
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": balanced,
        "quadratic_weighted_kappa": qwk,
        "ordinal_mae": float(np.mean(np.abs(pred - y))),
        "recall_class_2": float(recall[0]),
        "recall_class_3": float(recall[1]),
        "recall_class_4": float(recall[2]),
        "multiclass_brier": float(np.mean(np.sum(np.square(probs - one_hot), axis=1))),
        "log_loss": float(-np.mean(np.log(clipped))),
    }
    denominators = {
        "macro_f1": int(len(y)),
        "balanced_accuracy": int(np.min(actual)),
        "quadratic_weighted_kappa": int(np.min(actual)),
        "ordinal_mae": int(len(y)),
        "recall_class_2": int(actual[0]),
        "recall_class_3": int(actual[1]),
        "recall_class_4": int(actual[2]),
        "multiclass_brier": int(len(y)),
        "log_loss": int(len(y)),
    }
    return metrics, denominators


def _metric_support_status(
    *,
    metric: str,
    group_n: int,
    metric_denominator: int,
    point_estimate: float,
    support_threshold: int,
    minimum_true_class_denominator: int,
) -> tuple[bool, str]:
    if group_n < support_threshold:
        return False, "insufficient_group_support"
    if metric.startswith("recall_class_") or metric in CLASS_SENSITIVE_SUPPORT_METRICS:
        if metric_denominator < minimum_true_class_denominator:
            return False, "insufficient_true_class_support"
    if not math.isfinite(point_estimate):
        return False, "undefined_metric"
    return True, "eligible_descriptive_estimate"


def compute_subgroup_metric_grid_v3(
    contract: Mapping[str, Any], frame: pd.DataFrame, oof: pd.DataFrame
) -> pd.DataFrame:
    """Return every declared system/threshold/attribute/group/metric row."""

    settings = contract["subgroup_audit"]
    system_labels = {row["system_id"]: row["v3_label"] for row in settings["systems"]}
    group_values = {attribute: _group_series(frame, attribute, contract) for attribute in ATTRIBUTES}
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        predictions = oof[oof["system_id"] == system].sort_values("sample_index")
        _require(len(predictions) == len(frame), f"OOF coverage drifted for {system}.")
        y_true = predictions["y_true"].to_numpy(int)
        y_pred = predictions["y_pred"].to_numpy(int)
        probabilities = predictions[list(PROBABILITY_COLUMNS)].to_numpy(float)
        for threshold in settings["support_thresholds"]:
            for attribute in ATTRIBUTES:
                series = group_values[attribute]
                categories = (
                    list(settings["age_bins"]["labels"])
                    if attribute == "Age"
                    else sorted(series.unique())
                )
                for group in categories:
                    mask = series.to_numpy() == group
                    group_n = int(mask.sum())
                    metrics, denominators = metric_bundle_v3(
                        y_true[mask], y_pred[mask], probabilities[mask]
                    )
                    for metric in METRICS:
                        eligible, status = _metric_support_status(
                            metric=metric,
                            group_n=group_n,
                            metric_denominator=denominators[metric],
                            point_estimate=metrics[metric],
                            support_threshold=int(threshold),
                            minimum_true_class_denominator=int(
                                settings["minimum_true_class_denominator"]
                            ),
                        )
                        rows.append(
                            {
                                "system_id": system,
                                "v3_system_label": system_labels[system],
                                "support_threshold": int(threshold),
                                "attribute": attribute,
                                "group": group,
                                "group_n": group_n,
                                "metric": metric,
                                "metric_denominator": denominators[metric],
                                "point_estimate": metrics[metric] if eligible else math.nan,
                                "eligible_for_gap": eligible,
                                "support_status": status,
                                "higher_value_is_better": HIGHER_IS_BETTER[metric],
                                "probability_source": settings["probability_source"],
                                "inference_scope": "descriptive_exactly_once_oof",
                                "claim_boundary": "exploratory_subgroup_audit_not_fairness_certification_or_discrimination_evidence",
                            }
                        )
    result = pd.DataFrame(rows)
    expected_groups = sum(group_values[attribute].nunique() for attribute in ATTRIBUTES)
    expected_rows = len(SYSTEMS) * 3 * expected_groups * len(METRICS)
    _require(len(result) == expected_rows == 2025, "Subgroup metric-grid row count drifted.")
    return result


def compute_gap_sensitivity_v3(metric_grid: pd.DataFrame) -> pd.DataFrame:
    """Calculate every declared max-minus-min subgroup gap without selecting one headline."""

    rows: list[dict[str, Any]] = []
    keys = ["system_id", "v3_system_label", "support_threshold", "attribute", "metric"]
    for key, group in metric_grid.groupby(keys, sort=False, dropna=False):
        eligible = group[group["eligible_for_gap"].astype(bool)].copy()
        if len(eligible) >= 2:
            ordered = eligible.sort_values(["point_estimate", "group"], ascending=[True, True])
            minimum = ordered.iloc[0]
            maximum = ordered.iloc[-1]
            gap = float(maximum["point_estimate"] - minimum["point_estimate"])
            status = "estimable_descriptive_gap"
            min_group = str(minimum["group"])
            max_group = str(maximum["group"])
            min_value = float(minimum["point_estimate"])
            max_value = float(maximum["point_estimate"])
        else:
            gap = min_value = max_value = math.nan
            min_group = max_group = ""
            status = "fewer_than_two_eligible_groups"
        rows.append(
            {
                "system_id": key[0],
                "v3_system_label": key[1],
                "support_threshold": int(key[2]),
                "attribute": key[3],
                "metric": key[4],
                "gap_max_minus_min": gap,
                "minimum_group": min_group,
                "minimum_value": min_value,
                "maximum_group": max_group,
                "maximum_value": max_value,
                "eligible_group_count": len(eligible),
                "declared_group_count": len(group),
                "status": status,
                "higher_value_is_better": HIGHER_IS_BETTER[key[4]],
                "maximum_gap_selection_scope": "exploratory_all_cells_reported_no_selection_adjusted_single_winner_claim",
                "inference_scope": "descriptive_unless_primary_simultaneous_interval_is_joined",
            }
        )
    result = pd.DataFrame(rows)
    _require(len(result) == len(SYSTEMS) * 3 * len(ATTRIBUTES) * len(METRICS) == 486, "Gap grid row count drifted.")
    return result


def _stratified_resample_indices(
    outer_fold: np.ndarray, y_true: np.ndarray, *, n_resamples: int, seed: int
) -> tuple[list[np.ndarray], str]:
    strata = [
        np.flatnonzero((outer_fold == fold) & (y_true == label))
        for fold in sorted(np.unique(outer_fold))
        for label in LABELS
    ]
    _require(all(len(indices) > 0 for indices in strata), "Bootstrap contains an empty fold/class stratum.")
    rng = np.random.default_rng(int(seed))
    digest = hashlib.sha256(b"phase2c_stratified_outer_fold_y_true_indices_v1\0")
    draws: list[np.ndarray] = []
    for _ in range(int(n_resamples)):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
        ).astype(np.int64, copy=False)
        _require(len(sampled) == len(outer_fold), "Bootstrap sample size drifted.")
        digest.update(sampled.astype("<i8", copy=False).tobytes(order="C"))
        draws.append(sampled)
    return draws, digest.hexdigest()


def compute_primary_gap_bootstrap_v3(
    contract: Mapping[str, Any],
    frame: pd.DataFrame,
    oof: pd.DataFrame,
    metric_grid: pd.DataFrame,
    gaps: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add pointwise and familywise exploratory intervals to all P3 gap cells."""

    settings = contract["simultaneous_bootstrap"]
    primary = oof[oof["system_id"] == PRIMARY_SYSTEM].sort_values("sample_index")
    y_true = primary["y_true"].to_numpy(int)
    y_pred = primary["y_pred"].to_numpy(int)
    probabilities = primary[list(PROBABILITY_COLUMNS)].to_numpy(float)
    outer_fold = primary["outer_fold"].to_numpy(int)
    group_values = {
        attribute: _group_series(frame, attribute, contract).to_numpy()
        for attribute in ATTRIBUTES
    }
    primary_gaps = gaps[gaps["system_id"] == PRIMARY_SYSTEM].reset_index(drop=True)
    _require(len(primary_gaps) == 162, "Primary gap family size drifted.")
    eligibility: list[list[str]] = []
    primary_grid = metric_grid[metric_grid["system_id"] == PRIMARY_SYSTEM]
    for record in primary_gaps.itertuples(index=False):
        selected = primary_grid[
            (primary_grid["support_threshold"] == record.support_threshold)
            & (primary_grid["attribute"] == record.attribute)
            & (primary_grid["metric"] == record.metric)
            & primary_grid["eligible_for_gap"].astype(bool)
        ]
        eligibility.append(selected["group"].astype(str).tolist())

    draws, resample_hash = _stratified_resample_indices(
        outer_fold,
        y_true,
        n_resamples=int(settings["n_resamples"]),
        seed=int(settings["seed"]),
    )
    bootstrap_gaps = np.full((len(draws), len(primary_gaps)), np.nan, dtype=float)
    for draw_index, sample_indices in enumerate(draws):
        metric_lookup: dict[tuple[str, str], Mapping[str, float]] = {}
        for attribute in ATTRIBUTES:
            sampled_groups = group_values[attribute][sample_indices]
            for group_name in sorted(set(group_values[attribute].astype(str))):
                selected = sampled_groups.astype(str) == group_name
                if not selected.any():
                    continue
                metrics, _ = metric_bundle_v3(
                    y_true[sample_indices][selected],
                    y_pred[sample_indices][selected],
                    probabilities[sample_indices][selected],
                )
                metric_lookup[(attribute, group_name)] = metrics
        for column, record in enumerate(primary_gaps.itertuples(index=False)):
            eligible_groups = eligibility[column]
            if len(eligible_groups) < 2:
                continue
            values = np.asarray(
                [metric_lookup[(record.attribute, name)][record.metric] for name in eligible_groups],
                dtype=float,
            )
            if np.isfinite(values).all():
                bootstrap_gaps[draw_index, column] = float(np.max(values) - np.min(values))

    point = primary_gaps["gap_max_minus_min"].to_numpy(float)
    standard_errors = np.nanstd(bootstrap_gaps, axis=0, ddof=1)
    estimable = np.isfinite(point) & np.isfinite(standard_errors)
    complete_draws = np.isfinite(bootstrap_gaps[:, estimable]).all(axis=1)
    n_complete = int(complete_draws.sum())
    _require(n_complete >= int(0.95 * len(draws)), "Too few complete bootstrap draws for the simultaneous family.")
    standardized_columns = estimable & (standard_errors > 0.0)
    if standardized_columns.any():
        deviations = np.abs(
            (bootstrap_gaps[complete_draws][:, standardized_columns] - point[standardized_columns])
            / standard_errors[standardized_columns]
        )
        maximum_deviation = np.max(deviations, axis=1)
        critical_value = float(np.quantile(maximum_deviation, 0.95, method="linear"))
    else:
        critical_value = 0.0

    rows: list[dict[str, Any]] = []
    for column, record in enumerate(primary_gaps.itertuples(index=False)):
        values = bootstrap_gaps[:, column]
        valid = values[np.isfinite(values)]
        if not math.isfinite(record.gap_max_minus_min) or len(valid) == 0:
            point_low = point_high = simultaneous_low = simultaneous_high = math.nan
            status = "not_estimable_from_fixed_support_grid"
        else:
            point_low, point_high = np.quantile(valid, [0.025, 0.975], method="linear")
            half_width = critical_value * standard_errors[column]
            simultaneous_low = max(0.0, float(record.gap_max_minus_min - half_width))
            simultaneous_high = float(record.gap_max_minus_min + half_width)
            status = "exploratory_pointwise_and_familywise_intervals_available"
        rows.append(
            {
                "system_id": PRIMARY_SYSTEM,
                "support_threshold": int(record.support_threshold),
                "attribute": record.attribute,
                "metric": record.metric,
                "gap_max_minus_min": record.gap_max_minus_min,
                "pointwise_ci_low": point_low,
                "pointwise_ci_high": point_high,
                "simultaneous_ci_low": simultaneous_low,
                "simultaneous_ci_high": simultaneous_high,
                "bootstrap_std": standard_errors[column],
                "n_resamples": len(draws),
                "n_valid_cell_draws": len(valid),
                "n_complete_familywise_draws": n_complete,
                "familywise_critical_value": critical_value,
                "resample_hash": resample_hash,
                "status": status,
                "interval_scope": settings["multiplicity_scope"],
                "eligibility_scope": settings["eligibility_scope"],
                "model_training_variability_included": settings[
                    "model_training_variability_included"
                ],
                "claim_boundary": "exploratory_simultaneous_interval_not_confirmatory_fairness_inference",
            }
        )
    result = pd.DataFrame(rows)
    _require(len(result) == 162, "Primary bootstrap interval grid drifted.")
    return result, {
        "n_resamples": len(draws),
        "resample_hash": resample_hash,
        "family_size": int(estimable.sum()),
        "declared_family_rows": len(result),
        "n_complete_familywise_draws": n_complete,
        "familywise_critical_value": critical_value,
    }


def compute_proxy_prediction_change_v3(
    contract: Mapping[str, Any], frame: pd.DataFrame, oof: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare exact P3 and P3-minus-JobRole predictions overall and by department."""

    primary = oof[oof["system_id"] == PRIMARY_SYSTEM].sort_values("sample_index")
    reduced = oof[oof["system_id"] == PROXY_REDUCED_SYSTEM].sort_values("sample_index")
    for column in ("sample_index", "outer_fold", "y_true"):
        _require(np.array_equal(primary[column].to_numpy(), reduced[column].to_numpy()), f"Proxy comparator alignment drifted for {column}.")
    primary_probs = primary[list(PROBABILITY_COLUMNS)].to_numpy(float)
    reduced_probs = reduced[list(PROBABILITY_COLUMNS)].to_numpy(float)
    delta = reduced_probs - primary_probs
    primary_pred = primary["y_pred"].to_numpy(int)
    reduced_pred = reduced["y_pred"].to_numpy(int)
    true_indices = np.searchsorted(np.asarray(LABELS), primary["y_true"].to_numpy(int))
    sorted_primary = np.sort(primary_probs, axis=1)
    sorted_reduced = np.sort(reduced_probs, axis=1)
    epsilon = 1e-15
    sample = pd.DataFrame(
        {
            "sample_index": primary["sample_index"].to_numpy(int),
            "outer_fold": primary["outer_fold"].to_numpy(int),
            "department": frame.loc[primary["sample_index"].to_numpy(int), "EmpDepartment"].astype(str).to_numpy(),
            "y_true": primary["y_true"].to_numpy(int),
            "primary_y_pred": primary_pred,
            "proxy_reduced_y_pred": reduced_pred,
            "delta_prob_class_2": delta[:, 0],
            "delta_prob_class_3": delta[:, 1],
            "delta_prob_class_4": delta[:, 2],
            "total_variation": 0.5 * np.abs(delta).sum(axis=1),
            "max_absolute_probability_change": np.abs(delta).max(axis=1),
            "delta_true_class_probability": delta[np.arange(len(delta)), true_indices],
            "primary_confidence_margin": sorted_primary[:, -1] - sorted_primary[:, -2],
            "proxy_reduced_confidence_margin": sorted_reduced[:, -1] - sorted_reduced[:, -2],
            "delta_confidence_margin": (sorted_reduced[:, -1] - sorted_reduced[:, -2]) - (sorted_primary[:, -1] - sorted_primary[:, -2]),
            "primary_ordinal_log_odds_margin": np.log((primary_probs[:, 2] + epsilon) / (primary_probs[:, 0] + epsilon)),
            "proxy_reduced_ordinal_log_odds_margin": np.log((reduced_probs[:, 2] + epsilon) / (reduced_probs[:, 0] + epsilon)),
            "prediction_changed": primary_pred != reduced_pred,
            "signed_ordinal_prediction_shift": reduced_pred - primary_pred,
            "absolute_ordinal_prediction_shift": np.abs(reduced_pred - primary_pred),
        }
    )
    sample["delta_ordinal_log_odds_margin"] = (
        sample["proxy_reduced_ordinal_log_odds_margin"]
        - sample["primary_ordinal_log_odds_margin"]
    )
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", sample)]
    scopes.extend(
        ("department", str(department), rows_for_department)
        for department, rows_for_department in sample.groupby("department", sort=True)
    )
    for scope, department, selected in scopes:
        ids = selected["sample_index"].to_numpy(int)
        primary_selected = primary.set_index("sample_index").loc[ids]
        reduced_selected = reduced.set_index("sample_index").loc[ids]
        primary_metrics, _ = metric_bundle_v3(
            primary_selected["y_true"],
            primary_selected["y_pred"],
            primary_selected[list(PROBABILITY_COLUMNS)].to_numpy(float),
        )
        reduced_metrics, _ = metric_bundle_v3(
            reduced_selected["y_true"],
            reduced_selected["y_pred"],
            reduced_selected[list(PROBABILITY_COLUMNS)].to_numpy(float),
        )
        rows.append(
            {
                "scope": scope,
                "department": department,
                "n_samples": len(selected),
                "mean_total_variation": selected["total_variation"].mean(),
                "median_total_variation": selected["total_variation"].median(),
                "p90_total_variation": selected["total_variation"].quantile(0.9),
                "mean_max_absolute_probability_change": selected[
                    "max_absolute_probability_change"
                ].mean(),
                "mean_delta_prob_class_2": selected["delta_prob_class_2"].mean(),
                "mean_delta_prob_class_3": selected["delta_prob_class_3"].mean(),
                "mean_delta_prob_class_4": selected["delta_prob_class_4"].mean(),
                "mean_delta_true_class_probability": selected[
                    "delta_true_class_probability"
                ].mean(),
                "mean_delta_confidence_margin": selected["delta_confidence_margin"].mean(),
                "mean_delta_ordinal_log_odds_margin": selected[
                    "delta_ordinal_log_odds_margin"
                ].mean(),
                "prediction_change_rate": selected["prediction_changed"].mean(),
                "mean_absolute_ordinal_prediction_shift": selected[
                    "absolute_ordinal_prediction_shift"
                ].mean(),
                "primary_macro_f1": primary_metrics["macro_f1"],
                "proxy_reduced_macro_f1": reduced_metrics["macro_f1"],
                "delta_macro_f1": reduced_metrics["macro_f1"] - primary_metrics["macro_f1"],
                "primary_accuracy": float(
                    np.mean(primary_selected["y_true"].to_numpy(int) == primary_selected["y_pred"].to_numpy(int))
                ),
                "proxy_reduced_accuracy": float(
                    np.mean(reduced_selected["y_true"].to_numpy(int) == reduced_selected["y_pred"].to_numpy(int))
                ),
                "comparison_direction": contract["proxy_prediction_comparison"]["comparison_direction"],
                "inference_scope": "descriptive_paired_exact_oof_prediction_change",
                "claim_boundary": "performance_model_dependence_sensitivity_not_causality_or_discrimination_evidence",
            }
        )
    aggregate = pd.DataFrame(rows)
    _require(len(sample) == 1200 and len(aggregate) == 7, "Proxy prediction-change grid drifted.")
    return sample, aggregate


def _aligned_model_outputs(pipeline: Any, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    classifier = pipeline.named_steps["model"]
    classes = np.asarray(classifier.classes_, dtype=int)
    _require(set(classes) == set(LABELS), "Persisted model class set drifted.")
    order = [int(np.flatnonzero(classes == label)[0]) for label in LABELS]
    probabilities = np.asarray(pipeline.predict_proba(features), dtype=float)[:, order]
    transformed = pipeline.named_steps["preprocessor"].transform(features)
    margins = np.asarray(classifier.model_.predict(transformed, output_margin=True), dtype=float)
    _require(margins.shape == probabilities.shape, "Raw-margin shape drifted.")
    margins = margins[:, order]
    return probabilities, margins


def compute_jobrole_permutation_v3(
    contract: Mapping[str, Any],
    frame: pd.DataFrame,
    features: pd.DataFrame,
    oof: pd.DataFrame,
    artifacts: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply outcome-blind JobRole perturbations to exact persisted P3 outer models."""

    settings = contract["job_role_permutation"]
    primary = oof[oof["system_id"] == PRIMARY_SYSTEM].sort_values("sample_index")
    y_true = primary["y_true"].to_numpy(int)
    original_probabilities = primary[list(PROBABILITY_COLUMNS)].to_numpy(float)
    original_predictions = primary["y_pred"].to_numpy(int)
    original_margins = np.full_like(original_probabilities, np.nan)
    fold_inputs: list[tuple[int, np.ndarray, Any, np.ndarray, np.ndarray]] = []
    for outer_fold, fold_model in sorted(artifacts.fold_models.items()):
        ids = np.asarray(fold_model.test_sample_indices, dtype=int)
        pipeline = fold_model.pipeline
        replay_probabilities, replay_margins = _aligned_model_outputs(pipeline, features.loc[ids])
        _require(
            np.allclose(replay_probabilities, original_probabilities[ids], rtol=0.0, atol=1e-6),
            f"Persisted model replay drifted in outer fold {outer_fold}.",
        )
        original_margins[ids] = replay_margins
        fold_inputs.append(
            (
                int(outer_fold),
                ids,
                pipeline,
                frame.loc[ids, "EmpDepartment"].astype(str).to_numpy(),
                features.loc[ids, "EmpJobRole"].astype(str).to_numpy(),
            )
        )
    _require(np.isfinite(original_margins).all(), "Original raw-margin OOF replay is incomplete.")
    original_metrics, _ = metric_bundle_v3(
        y_true, original_predictions, original_probabilities
    )
    schemes = list(settings["schemes"])
    sample_frames: list[pd.DataFrame] = []
    repetition_rows: list[dict[str, Any]] = []
    original_predicted_index = np.argmax(original_probabilities, axis=1)
    for scheme_index, scheme in enumerate(schemes):
        for seed in settings["seeds"]:
            permuted_probabilities = np.full_like(original_probabilities, np.nan)
            permuted_margins = np.full_like(original_margins, np.nan)
            role_changed = np.zeros(len(features), dtype=bool)
            for outer_fold, ids, pipeline, departments, original_roles in fold_inputs:
                permuted_features = features.loc[ids].copy()
                permuted_roles = original_roles.copy()
                rng = np.random.default_rng(
                    np.random.SeedSequence([int(seed), int(outer_fold), int(scheme_index)])
                )
                if scheme == "marginal_within_outer_test_fold":
                    permuted_roles = permuted_roles[rng.permutation(len(permuted_roles))]
                elif scheme == "department_conditional_within_outer_test_fold_and_department":
                    for department in sorted(set(departments)):
                        positions = np.flatnonzero(departments == department)
                        permuted_roles[positions] = permuted_roles[positions][
                            rng.permutation(len(positions))
                        ]
                else:
                    raise SubgroupProxyUseV3Error(f"Unsupported permutation scheme: {scheme}.")
                permuted_features.loc[:, "EmpJobRole"] = permuted_roles
                probabilities, margins = _aligned_model_outputs(pipeline, permuted_features)
                permuted_probabilities[ids] = probabilities
                permuted_margins[ids] = margins
                role_changed[ids] = permuted_roles != original_roles
            _require(np.isfinite(permuted_probabilities).all(), "Permutation probability OOF is incomplete.")
            _require(np.isfinite(permuted_margins).all(), "Permutation margin OOF is incomplete.")
            permuted_predictions = np.asarray(LABELS)[np.argmax(permuted_probabilities, axis=1)]
            total_variation = 0.5 * np.abs(permuted_probabilities - original_probabilities).sum(axis=1)
            prediction_changed = permuted_predictions != original_predictions
            probability_drop = (
                original_probabilities[np.arange(len(y_true)), original_predicted_index]
                - permuted_probabilities[np.arange(len(y_true)), original_predicted_index]
            )
            margin_drop = (
                original_margins[np.arange(len(y_true)), original_predicted_index]
                - permuted_margins[np.arange(len(y_true)), original_predicted_index]
            )
            sample_frames.append(
                pd.DataFrame(
                    {
                        "scheme": scheme,
                        "seed": int(seed),
                        "sample_index": np.arange(len(y_true), dtype=int),
                        "outer_fold": primary["outer_fold"].to_numpy(int),
                        "department": frame["EmpDepartment"].astype(str).to_numpy(),
                        "y_true": y_true,
                        "original_y_pred": original_predictions,
                        "permuted_y_pred": permuted_predictions,
                        "job_role_value_changed": role_changed,
                        "total_variation": total_variation,
                        "prediction_changed": prediction_changed,
                        "absolute_ordinal_prediction_shift": np.abs(
                            permuted_predictions - original_predictions
                        ),
                        "original_predicted_class_probability_drop": probability_drop,
                        "original_predicted_class_raw_margin_drop": margin_drop,
                    }
                )
            )
            perturbed_metrics, _ = metric_bundle_v3(
                y_true, permuted_predictions, permuted_probabilities
            )
            repetition_rows.append(
                {
                    "scheme": scheme,
                    "seed": int(seed),
                    "n_samples": len(y_true),
                    "job_role_value_changed_fraction": role_changed.mean(),
                    "mean_total_variation": total_variation.mean(),
                    "prediction_change_rate": prediction_changed.mean(),
                    "mean_absolute_ordinal_prediction_shift": np.abs(
                        permuted_predictions - original_predictions
                    ).mean(),
                    "mean_original_predicted_class_probability_drop": probability_drop.mean(),
                    "mean_original_predicted_class_raw_margin_drop": margin_drop.mean(),
                    **{f"original_{metric}": original_metrics[metric] for metric in METRICS},
                    **{f"permuted_{metric}": perturbed_metrics[metric] for metric in METRICS},
                    **{
                        f"delta_{metric}": perturbed_metrics[metric] - original_metrics[metric]
                        for metric in METRICS
                    },
                    "inference_scope": "dependent_random_perturbation_repetitions_descriptive_not_confidence_intervals",
                    "claim_boundary": "model_dependence_sensitivity_not_causal_effect_or_discrimination_evidence",
                }
            )
    sample = pd.concat(sample_frames, ignore_index=True)
    repetitions = pd.DataFrame(repetition_rows)
    _require(len(sample) == 48000 and len(repetitions) == 40, "JobRole permutation grid drifted.")
    summary_rows: list[dict[str, Any]] = []
    summary_metrics = [
        "job_role_value_changed_fraction",
        "mean_total_variation",
        "prediction_change_rate",
        "mean_absolute_ordinal_prediction_shift",
        "mean_original_predicted_class_probability_drop",
        "mean_original_predicted_class_raw_margin_drop",
        *[f"delta_{metric}" for metric in METRICS],
    ]
    for scheme, selected in repetitions.groupby("scheme", sort=False):
        row: dict[str, Any] = {
            "scheme": scheme,
            "n_repetitions": len(selected),
            "n_samples_per_repetition": int(selected["n_samples"].iloc[0]),
        }
        for metric in summary_metrics:
            row[f"{metric}_mean"] = selected[metric].mean()
            row[f"{metric}_std"] = selected[metric].std(ddof=1)
            row[f"{metric}_min"] = selected[metric].min()
            row[f"{metric}_max"] = selected[metric].max()
        row["inference_scope"] = "descriptive_perturbation_variability_not_sampling_uncertainty"
        row["claim_boundary"] = "JobRole_permutation_sensitivity_not_causality_fairness_or_discrimination_evidence"
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    _require(len(summary) == 2, "JobRole permutation scheme summary drifted.")
    return sample, repetitions, summary


def _reconstructability_sources(
    contract: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = contract["source_contracts"]
    metrics = pd.read_csv(PROJECT_ROOT / sources["proxy_metric_intervals"]["path"])
    differences = pd.read_csv(
        PROJECT_ROOT / sources["proxy_policy_paired_differences"]["path"]
    )
    metrics["v3_interpretation"] = (
        "department_reconstructability_from_feature_space_not_performance_model_department_use"
    )
    differences["v3_interpretation"] = (
        "JobRole_contribution_to_department_reconstructability_not_performance_model_dependence"
    )
    metrics["performance_dependence_claim_allowed"] = False
    differences["performance_dependence_claim_allowed"] = False
    return metrics, differences


def evaluate_subgroup_proxy_use_v3(inputs: Phase2CInputs) -> Phase2CResult:
    """Evaluate the complete in-memory Phase 2C grid from validated exact sources."""

    subgroup_grid = compute_subgroup_metric_grid_v3(
        inputs.contract, inputs.frame, inputs.oof
    )
    gaps = compute_gap_sensitivity_v3(subgroup_grid)
    bootstrap, bootstrap_receipt = compute_primary_gap_bootstrap_v3(
        inputs.contract, inputs.frame, inputs.oof, subgroup_grid, gaps
    )
    proxy_sample, proxy_department = compute_proxy_prediction_change_v3(
        inputs.contract, inputs.frame, inputs.oof
    )
    permutation_sample, permutation_repetitions, permutation_summary = (
        compute_jobrole_permutation_v3(
            inputs.contract,
            inputs.frame,
            inputs.features,
            inputs.oof,
            inputs.artifacts,
        )
    )
    reconstructability_metrics, reconstructability_differences = (
        _reconstructability_sources(inputs.contract)
    )
    status_counts = {
        str(key): int(value)
        for key, value in subgroup_grid["support_status"].value_counts().sort_index().items()
    }
    diagnostic = {
        "schema_version": 1,
        "stage": "subgroup_proxy_use_v3",
        "status": "complete",
        "subgroup_metric_rows": len(subgroup_grid),
        "subgroup_gap_rows": len(gaps),
        "subgroup_support_status_counts": status_counts,
        "primary_gap_interval_rows": len(bootstrap),
        "bootstrap": bootstrap_receipt,
        "proxy_prediction_change_sample_rows": len(proxy_sample),
        "proxy_department_aggregate_rows": len(proxy_department),
        "jobrole_permutation_sample_rows": len(permutation_sample),
        "jobrole_permutation_repetition_rows": len(permutation_repetitions),
        "jobrole_permutation_scheme_rows": len(permutation_summary),
        "department_reconstructability_metric_rows": len(reconstructability_metrics),
        "department_reconstructability_difference_rows": len(
            reconstructability_differences
        ),
        "new_performance_model_fit_calls": 0,
        "new_proxy_reconstruction_model_fit_calls": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
        "fairness_certification_allowed": False,
        "formal_discrimination_claim_allowed": False,
        "department_reconstructability_proves_performance_use": False,
        "permutation_supports_causal_claim": False,
    }
    return Phase2CResult(
        subgroup_metric_grid=subgroup_grid,
        subgroup_gap_sensitivity=gaps,
        primary_gap_bootstrap_intervals=bootstrap,
        proxy_prediction_change_sample=proxy_sample,
        proxy_prediction_change_by_department=proxy_department,
        jobrole_permutation_sample=permutation_sample,
        jobrole_permutation_repetition=permutation_repetitions,
        jobrole_permutation_summary=permutation_summary,
        department_reconstructability_metrics=reconstructability_metrics,
        department_reconstructability_differences=reconstructability_differences,
        diagnostic_receipt=diagnostic,
    )


def preflight_subgroup_proxy_use_v3(
    contract_path: Path | str = DEFAULT_SUBGROUP_PROXY_USE_CONTRACT,
) -> dict[str, Any]:
    """Validate every Phase 2C source and fit-free scope without writing outputs."""

    inputs = _prepare_inputs(contract_path)
    return {
        "status": "preflight_passed",
        "contract_sha256": inputs.contract_receipt["contract_sha256"],
        "sample_count": len(inputs.frame),
        "oof_rows": len(inputs.oof),
        "systems": len(SYSTEMS),
        "subgroup_attributes": len(ATTRIBUTES),
        "support_thresholds": inputs.contract["subgroup_audit"]["support_thresholds"],
        "metrics": len(METRICS),
        "planned_bootstrap_resamples": inputs.contract["simultaneous_bootstrap"][
            "n_resamples"
        ],
        "planned_permutation_repetitions": len(
            inputs.contract["job_role_permutation"]["seeds"]
        )
        * len(inputs.contract["job_role_permutation"]["schemes"]),
        "persisted_outer_model_count": len(inputs.artifacts.fold_models),
        "planned_new_model_fit_calls": 0,
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


def run_subgroup_proxy_use_v3(
    *, contract_path: Path | str, output_dir: Path | str, run_id: str
) -> dict[str, Any]:
    """Atomically publish a complete Phase 2C package from one clean exact commit."""

    contract_path = Path(contract_path)
    output_dir = Path(output_dir)
    _require(bool(str(run_id).strip()), "run_id must be non-empty.")
    with enforce_offline_runtime() as offline_state:
        git_identity = _clean_git_identity()
        inputs = _prepare_inputs(contract_path)
        _require(not output_dir.exists(), f"Output destination already exists: {output_dir}.")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = output_dir.parent / f".{output_dir.name}.staging.{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            implementation_paths = (
                Path("src/experiments/subgroup_proxy_use_v3.py"),
                Path("src/governance/subgroup_proxy_use_contract_v3.py"),
                Path("src/experiments/benchmark_artifact_contract.py"),
            )
            scientific_inputs = {
                "git_identity": git_identity,
                "source_tree_hash": source_tree_hash(PROJECT_ROOT),
                "contract_sha256": inputs.contract_receipt["contract_sha256"],
                "bound_source_hashes": {
                    name: record["sha256"]
                    for name, record in inputs.contract["source_contracts"].items()
                },
                "implementation_hashes": {
                    path.as_posix(): sha256_file(PROJECT_ROOT / path)
                    for path in implementation_paths
                },
            }
            scientific_input_sha256 = _canonical_json_sha256(scientific_inputs)
            result = evaluate_subgroup_proxy_use_v3(inputs)
            frames = {
                "subgroup_metric_grid.csv": result.subgroup_metric_grid,
                "subgroup_gap_sensitivity.csv": result.subgroup_gap_sensitivity,
                "primary_gap_bootstrap_intervals.csv": result.primary_gap_bootstrap_intervals,
                "proxy_prediction_change_sample.csv": result.proxy_prediction_change_sample,
                "proxy_prediction_change_by_department.csv": result.proxy_prediction_change_by_department,
                "jobrole_permutation_sample.csv": result.jobrole_permutation_sample,
                "jobrole_permutation_repetition.csv": result.jobrole_permutation_repetition,
                "jobrole_permutation_summary.csv": result.jobrole_permutation_summary,
                "department_reconstructability_metrics.csv": result.department_reconstructability_metrics,
                "department_reconstructability_differences.csv": result.department_reconstructability_differences,
            }
            for filename, frame in frames.items():
                frame.to_csv(staging / filename, index=False, lineterminator="\n")
            diagnostic = {
                **result.diagnostic_receipt,
                "run_id": run_id,
                "contract_sha256": inputs.contract_receipt["contract_sha256"],
                "scientific_input_sha256": scientific_input_sha256,
                "canonical_identity": inputs.contract["canonical_identity"],
            }
            _write_json(staging / "diagnostic_receipt.json", diagnostic)
            output_hashes = {
                path.name: sha256_file(path)
                for path in sorted(staging.iterdir())
                if path.is_file()
            }
            metadata = {
                "schema_version": 1,
                "stage": "subgroup_proxy_use_v3",
                "status": "complete",
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract_sha256": inputs.contract_receipt["contract_sha256"],
                "scientific_input_sha256": scientific_input_sha256,
                "scientific_inputs": scientific_inputs,
                "git_identity": git_identity,
                "canonical_identity": inputs.contract["canonical_identity"],
                **result.diagnostic_receipt,
                "runtime_policy": offline_state.receipt(),
                "output_hashes": output_hashes,
            }
            _require(_clean_git_identity() == git_identity, "Git identity changed during Phase 2C execution.")
            _require(
                source_tree_hash(PROJECT_ROOT) == scientific_inputs["source_tree_hash"],
                "Source tree changed during Phase 2C execution.",
            )
            repeated = validate_subgroup_proxy_use_contract_v3(contract_path)
            _require(
                repeated["contract_sha256"] == inputs.contract_receipt["contract_sha256"],
                "Phase 2C contract changed during execution.",
            )
            for name, record in inputs.contract["source_contracts"].items():
                _require(
                    sha256_file(PROJECT_ROOT / record["path"]) == record["sha256"],
                    f"Phase 2C source changed during execution: {name}.",
                )
            _write_json(staging / "stage_metadata.json", metadata)
            _require(
                {path.name for path in staging.iterdir() if path.is_file()}
                == EXPECTED_LOCAL_FILES,
                "Phase 2C output inventory drifted.",
            )
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
        "contract_sha256": inputs.contract_receipt["contract_sha256"],
        "scientific_input_sha256": scientific_input_sha256,
        "new_model_fit_calls": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract", type=Path, default=DEFAULT_SUBGROUP_PROXY_USE_CONTRACT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_LOCAL_RUN_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.preflight_only:
        receipt = preflight_subgroup_proxy_use_v3(args.contract)
    else:
        _require(bool(args.run_id), "Full Phase 2C execution requires --run-id.")
        receipt = run_subgroup_proxy_use_v3(
            contract_path=args.contract,
            output_dir=args.output_root / str(args.run_id) / "subgroup_proxy_use",
            run_id=str(args.run_id),
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_LOCAL_FILES",
    "Phase2CInputs",
    "Phase2CResult",
    "SubgroupProxyUseV3Error",
    "compute_gap_sensitivity_v3",
    "compute_jobrole_permutation_v3",
    "compute_primary_gap_bootstrap_v3",
    "compute_proxy_prediction_change_v3",
    "compute_subgroup_metric_grid_v3",
    "evaluate_subgroup_proxy_use_v3",
    "metric_bundle_v3",
    "preflight_subgroup_proxy_use_v3",
    "run_subgroup_proxy_use_v3",
]
