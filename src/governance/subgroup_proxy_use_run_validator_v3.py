"""Independent closed-world validator for the complete v3 Phase 2C run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.canonical_loader import load_canonical_dataset, sha256_file
from src.experiments.benchmark_artifact_contract import read_xgboost_oof_artifacts
from src.governance.offline_runtime import validate_policy_receipt
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


DEFAULT_SUBGROUP_PROXY_USE_RUN = Path(
    "reports/major_revision_v3_runs/"
    "phase2c_v3_20260907T070905Z_e314bb5/subgroup_proxy_use"
)
EXPECTED_GENERATION_COMMIT = "e314bb5d1935da6a23e26b2085330dacb468fe32"
PROBABILITY_COLUMNS = tuple(f"prob_class_{label}" for label in LABELS)
EXPECTED_FILES = frozenset(
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
OUTPUT_HASH_FILES = EXPECTED_FILES - {"stage_metadata.json"}
EXPECTED_IMPLEMENTATIONS = frozenset(
    {
        "src/experiments/subgroup_proxy_use_v3.py",
        "src/governance/subgroup_proxy_use_contract_v3.py",
        "src/experiments/benchmark_artifact_contract.py",
    }
)
EXPECTED_ROW_COUNTS = {
    "subgroup_metric_grid.csv": 2025,
    "subgroup_gap_sensitivity.csv": 486,
    "primary_gap_bootstrap_intervals.csv": 162,
    "proxy_prediction_change_sample.csv": 1200,
    "proxy_prediction_change_by_department.csv": 7,
    "jobrole_permutation_sample.csv": 48000,
    "jobrole_permutation_repetition.csv": 40,
    "jobrole_permutation_summary.csv": 2,
    "department_reconstructability_metrics.csv": 6,
    "department_reconstructability_differences.csv": 3,
}
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
CLASS_SENSITIVE = {"balanced_accuracy", "quadratic_weighted_kappa"}


class V3SubgroupProxyUseRunValidationError(RuntimeError):
    """Raised when persisted Phase 2C evidence is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3SubgroupProxyUseRunValidationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V3SubgroupProxyUseRunValidationError(
            f"Could not read {path.as_posix()}: {exc}"
        ) from exc
    _require(isinstance(payload, dict), f"{path.name} must contain a JSON object.")
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, float_precision="round_trip")
    except Exception as exc:
        raise V3SubgroupProxyUseRunValidationError(
            f"Could not parse {path.name}: {exc}"
        ) from exc


def _canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _git_blob(commit: str, relative_path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise V3SubgroupProxyUseRunValidationError(
            f"Could not resolve generation blob {commit}:{relative_path}: {exc}"
        ) from exc


def _assert_frame_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_columns: Sequence[str],
    context: str,
    tolerance: float = 1e-12,
) -> None:
    _require(set(observed.columns) == set(expected.columns), f"{context} schema drifted.")
    columns = list(expected.columns)
    try:
        pd.testing.assert_frame_equal(
            observed.loc[:, columns].sort_values(list(sort_columns)).reset_index(drop=True),
            expected.loc[:, columns].sort_values(list(sort_columns)).reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=tolerance,
        )
    except AssertionError as exc:
        raise V3SubgroupProxyUseRunValidationError(
            f"{context} does not match independent recomputation: {exc}"
        ) from exc


def _bool_values(series: pd.Series, *, context: str) -> pd.Series:
    values = series.map(
        {
            True: True,
            False: False,
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            1: True,
            0: False,
            "1": True,
            "0": False,
        }
    )
    _require(values.notna().all(), f"{context} contains an invalid Boolean.")
    return values.astype(bool)


def _group_series(frame: pd.DataFrame, attribute: str, contract: Mapping[str, Any]) -> pd.Series:
    if attribute == "Age":
        age = contract["subgroup_audit"]["age_bins"]
        values = pd.cut(
            pd.to_numeric(frame["Age"], errors="raise"),
            bins=age["edges"],
            labels=age["labels"],
            right=True,
            include_lowest=True,
        )
        _require(values.notna().all(), "Independent age binning lost a sample.")
        return values.astype("string").astype(str)
    return frame[attribute].astype("string").fillna("__MISSING__").astype(str)


def _metric_bundle(
    y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray
) -> tuple[dict[str, float], dict[str, int]]:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    _require(len(y) > 0 and pred.shape == y.shape, "Independent metric arrays are invalid.")
    label_array = np.asarray(LABELS)
    true_index = np.searchsorted(label_array, y)
    pred_index = np.searchsorted(label_array, pred)
    _require(
        np.all(label_array[true_index] == y) and np.all(label_array[pred_index] == pred),
        "Independent metric input contains an unknown label.",
    )
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (true_index, pred_index), 1)
    actual = confusion.sum(axis=1).astype(float)
    predicted = confusion.sum(axis=0).astype(float)
    true_positive = np.diag(confusion).astype(float)
    recalls = np.divide(
        true_positive,
        actual,
        out=np.full(3, np.nan, dtype=float),
        where=actual > 0,
    )
    f1_denominator = true_positive * 2.0 + predicted - true_positive + actual - true_positive
    f1_values = np.divide(
        true_positive * 2.0,
        f1_denominator,
        out=np.zeros(3, dtype=float),
        where=f1_denominator > 0,
    )
    coordinates = np.arange(3, dtype=float)
    weights = np.square(coordinates[:, None] - coordinates[None, :]) / 4.0
    expected = np.outer(actual, predicted) / float(len(y))
    expected_disagreement = float(np.sum(weights * expected))
    qwk = (
        float(1.0 - np.sum(weights * confusion) / expected_disagreement)
        if expected_disagreement > 0.0
        else math.nan
    )
    true_probability = np.clip(probs[np.arange(len(y)), true_index], 1e-15, 1.0)
    one_hot = np.eye(3, dtype=float)[true_index]
    metrics = {
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recalls)) if np.isfinite(recalls).all() else math.nan,
        "quadratic_weighted_kappa": qwk,
        "ordinal_mae": float(np.mean(np.abs(pred - y))),
        "recall_class_2": float(recalls[0]),
        "recall_class_3": float(recalls[1]),
        "recall_class_4": float(recalls[2]),
        "multiclass_brier": float(np.mean(np.sum(np.square(probs - one_hot), axis=1))),
        "log_loss": float(-np.mean(np.log(true_probability))),
    }
    denominators = {
        "macro_f1": len(y),
        "balanced_accuracy": int(np.min(actual)),
        "quadratic_weighted_kappa": int(np.min(actual)),
        "ordinal_mae": len(y),
        "recall_class_2": int(actual[0]),
        "recall_class_3": int(actual[1]),
        "recall_class_4": int(actual[2]),
        "multiclass_brier": len(y),
        "log_loss": len(y),
    }
    return metrics, denominators


def _eligible_status(
    metric: str,
    group_n: int,
    denominator: int,
    value: float,
    threshold: int,
    minimum_class_support: int,
) -> tuple[bool, str]:
    if group_n < threshold:
        return False, "insufficient_group_support"
    if (metric.startswith("recall_class_") or metric in CLASS_SENSITIVE) and denominator < minimum_class_support:
        return False, "insufficient_true_class_support"
    if not math.isfinite(value):
        return False, "undefined_metric"
    return True, "eligible_descriptive_estimate"


def _subgroup_grid(
    contract: Mapping[str, Any], frame: pd.DataFrame, oof: pd.DataFrame
) -> pd.DataFrame:
    settings = contract["subgroup_audit"]
    labels = {row["system_id"]: row["v3_label"] for row in settings["systems"]}
    group_values = {attribute: _group_series(frame, attribute, contract) for attribute in ATTRIBUTES}
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        source = oof[oof["system_id"] == system].sort_values("sample_index")
        y_true = source["y_true"].to_numpy(int)
        y_pred = source["y_pred"].to_numpy(int)
        probabilities = source.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
        for threshold in (20, 30, 50):
            for attribute in ATTRIBUTES:
                values = group_values[attribute]
                categories = (
                    list(settings["age_bins"]["labels"])
                    if attribute == "Age"
                    else sorted(values.unique())
                )
                for category in categories:
                    selected = values.to_numpy() == category
                    metrics, denominators = _metric_bundle(
                        y_true[selected], y_pred[selected], probabilities[selected]
                    )
                    for metric in METRICS:
                        eligible, status = _eligible_status(
                            metric,
                            int(selected.sum()),
                            denominators[metric],
                            metrics[metric],
                            threshold,
                            int(settings["minimum_true_class_denominator"]),
                        )
                        rows.append(
                            {
                                "system_id": system,
                                "v3_system_label": labels[system],
                                "support_threshold": threshold,
                                "attribute": attribute,
                                "group": category,
                                "group_n": int(selected.sum()),
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
    return pd.DataFrame(rows)


def _gap_grid(group_grid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["system_id", "v3_system_label", "support_threshold", "attribute", "metric"]
    for key, group in group_grid.groupby(keys, sort=False, dropna=False):
        eligible = group[_bool_values(group["eligible_for_gap"], context="gap eligibility")]
        if len(eligible) >= 2:
            ordered = eligible.sort_values(["point_estimate", "group"])
            low, high = ordered.iloc[0], ordered.iloc[-1]
            values = {
                "gap_max_minus_min": float(high["point_estimate"] - low["point_estimate"]),
                "minimum_group": str(low["group"]),
                "minimum_value": float(low["point_estimate"]),
                "maximum_group": str(high["group"]),
                "maximum_value": float(high["point_estimate"]),
                "status": "estimable_descriptive_gap",
            }
        else:
            values = {
                "gap_max_minus_min": math.nan,
                "minimum_group": "",
                "minimum_value": math.nan,
                "maximum_group": "",
                "maximum_value": math.nan,
                "status": "fewer_than_two_eligible_groups",
            }
        rows.append(
            {
                "system_id": key[0],
                "v3_system_label": key[1],
                "support_threshold": int(key[2]),
                "attribute": key[3],
                "metric": key[4],
                **values,
                "eligible_group_count": len(eligible),
                "declared_group_count": len(group),
                "higher_value_is_better": HIGHER_IS_BETTER[key[4]],
                "maximum_gap_selection_scope": "exploratory_all_cells_reported_no_selection_adjusted_single_winner_claim",
                "inference_scope": "descriptive_unless_primary_simultaneous_interval_is_joined",
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_intervals(
    contract: Mapping[str, Any],
    frame: pd.DataFrame,
    oof: pd.DataFrame,
    group_grid: pd.DataFrame,
    gap_grid: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    primary = oof[oof["system_id"] == PRIMARY_SYSTEM].sort_values("sample_index")
    y_true = primary["y_true"].to_numpy(int)
    y_pred = primary["y_pred"].to_numpy(int)
    probabilities = primary.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    outer_fold = primary["outer_fold"].to_numpy(int)
    groups = {attribute: _group_series(frame, attribute, contract).to_numpy() for attribute in ATTRIBUTES}
    cells = gap_grid[gap_grid["system_id"] == PRIMARY_SYSTEM].reset_index(drop=True)
    primary_grid = group_grid[group_grid["system_id"] == PRIMARY_SYSTEM]
    eligible_groups: list[list[str]] = []
    for cell in cells.itertuples(index=False):
        rows = primary_grid[
            (primary_grid["support_threshold"] == cell.support_threshold)
            & (primary_grid["attribute"] == cell.attribute)
            & (primary_grid["metric"] == cell.metric)
            & _bool_values(primary_grid["eligible_for_gap"], context="bootstrap eligibility")
        ]
        eligible_groups.append(rows["group"].astype(str).tolist())
    strata = [
        np.flatnonzero((outer_fold == fold) & (y_true == label))
        for fold in range(1, 11)
        for label in LABELS
    ]
    rng = np.random.default_rng(31041)
    digest = hashlib.sha256(b"phase2c_stratified_outer_fold_y_true_indices_v1\0")
    values = np.full((5000, len(cells)), np.nan, dtype=float)
    for draw in range(5000):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
        ).astype(np.int64, copy=False)
        digest.update(sampled.astype("<i8", copy=False).tobytes(order="C"))
        metric_lookup: dict[tuple[str, str], Mapping[str, float]] = {}
        for attribute in ATTRIBUTES:
            sampled_group = groups[attribute][sampled].astype(str)
            for category in sorted(set(groups[attribute].astype(str))):
                selected = sampled_group == category
                if selected.any():
                    metrics, _ = _metric_bundle(
                        y_true[sampled][selected],
                        y_pred[sampled][selected],
                        probabilities[sampled][selected],
                    )
                    metric_lookup[(attribute, category)] = metrics
        for column, cell in enumerate(cells.itertuples(index=False)):
            names = eligible_groups[column]
            if len(names) < 2:
                continue
            cell_values = np.asarray(
                [metric_lookup[(cell.attribute, name)][cell.metric] for name in names]
            )
            if np.isfinite(cell_values).all():
                values[draw, column] = float(np.max(cell_values) - np.min(cell_values))
    point = cells["gap_max_minus_min"].to_numpy(float)
    standard_error = np.nanstd(values, axis=0, ddof=1)
    estimable = np.isfinite(point) & np.isfinite(standard_error)
    complete = np.isfinite(values[:, estimable]).all(axis=1)
    standardized = estimable & (standard_error > 0.0)
    maximum = np.max(
        np.abs(
            (values[complete][:, standardized] - point[standardized])
            / standard_error[standardized]
        ),
        axis=1,
    )
    critical = float(np.quantile(maximum, 0.95, method="linear"))
    resample_hash = digest.hexdigest()
    rows: list[dict[str, Any]] = []
    for column, cell in enumerate(cells.itertuples(index=False)):
        valid = values[np.isfinite(values[:, column]), column]
        if math.isfinite(cell.gap_max_minus_min) and len(valid):
            point_low, point_high = np.quantile(valid, [0.025, 0.975], method="linear")
            half_width = critical * standard_error[column]
            simultaneous_low = max(0.0, float(cell.gap_max_minus_min - half_width))
            simultaneous_high = float(cell.gap_max_minus_min + half_width)
            status = "exploratory_pointwise_and_familywise_intervals_available"
        else:
            point_low = point_high = simultaneous_low = simultaneous_high = math.nan
            status = "not_estimable_from_fixed_support_grid"
        rows.append(
            {
                "system_id": PRIMARY_SYSTEM,
                "support_threshold": int(cell.support_threshold),
                "attribute": cell.attribute,
                "metric": cell.metric,
                "gap_max_minus_min": cell.gap_max_minus_min,
                "pointwise_ci_low": point_low,
                "pointwise_ci_high": point_high,
                "simultaneous_ci_low": simultaneous_low,
                "simultaneous_ci_high": simultaneous_high,
                "bootstrap_std": standard_error[column],
                "n_resamples": 5000,
                "n_valid_cell_draws": len(valid),
                "n_complete_familywise_draws": int(complete.sum()),
                "familywise_critical_value": critical,
                "resample_hash": resample_hash,
                "status": status,
                "interval_scope": contract["simultaneous_bootstrap"]["multiplicity_scope"],
                "eligibility_scope": contract["simultaneous_bootstrap"]["eligibility_scope"],
                "model_training_variability_included": False,
                "claim_boundary": "exploratory_simultaneous_interval_not_confirmatory_fairness_inference",
            }
        )
    receipt = {
        "n_resamples": 5000,
        "resample_hash": resample_hash,
        "family_size": int(estimable.sum()),
        "declared_family_rows": len(cells),
        "n_complete_familywise_draws": int(complete.sum()),
        "familywise_critical_value": critical,
    }
    return pd.DataFrame(rows), receipt


def _proxy_change(
    contract: Mapping[str, Any], frame: pd.DataFrame, oof: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = oof[oof["system_id"] == PRIMARY_SYSTEM].sort_values("sample_index")
    reduced = oof[oof["system_id"] == PROXY_REDUCED_SYSTEM].sort_values("sample_index")
    p_primary = primary.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    p_reduced = reduced.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    delta = p_reduced - p_primary
    y_true = primary["y_true"].to_numpy(int)
    y_primary = primary["y_pred"].to_numpy(int)
    y_reduced = reduced["y_pred"].to_numpy(int)
    true_index = np.searchsorted(np.asarray(LABELS), y_true)
    sorted_primary = np.sort(p_primary, axis=1)
    sorted_reduced = np.sort(p_reduced, axis=1)
    epsilon = 1e-15
    sample = pd.DataFrame(
        {
            "sample_index": primary["sample_index"].to_numpy(int),
            "outer_fold": primary["outer_fold"].to_numpy(int),
            "department": frame.loc[primary["sample_index"].to_numpy(int), "EmpDepartment"].astype(str).to_numpy(),
            "y_true": y_true,
            "primary_y_pred": y_primary,
            "proxy_reduced_y_pred": y_reduced,
            "delta_prob_class_2": delta[:, 0],
            "delta_prob_class_3": delta[:, 1],
            "delta_prob_class_4": delta[:, 2],
            "total_variation": np.abs(delta).sum(axis=1) / 2.0,
            "max_absolute_probability_change": np.abs(delta).max(axis=1),
            "delta_true_class_probability": delta[np.arange(len(delta)), true_index],
            "primary_confidence_margin": sorted_primary[:, 2] - sorted_primary[:, 1],
            "proxy_reduced_confidence_margin": sorted_reduced[:, 2] - sorted_reduced[:, 1],
            "delta_confidence_margin": (sorted_reduced[:, 2] - sorted_reduced[:, 1]) - (sorted_primary[:, 2] - sorted_primary[:, 1]),
            "primary_ordinal_log_odds_margin": np.log((p_primary[:, 2] + epsilon) / (p_primary[:, 0] + epsilon)),
            "proxy_reduced_ordinal_log_odds_margin": np.log((p_reduced[:, 2] + epsilon) / (p_reduced[:, 0] + epsilon)),
            "prediction_changed": y_primary != y_reduced,
            "signed_ordinal_prediction_shift": y_reduced - y_primary,
            "absolute_ordinal_prediction_shift": np.abs(y_reduced - y_primary),
        }
    )
    sample["delta_ordinal_log_odds_margin"] = sample["proxy_reduced_ordinal_log_odds_margin"] - sample["primary_ordinal_log_odds_margin"]
    source_primary = primary.set_index("sample_index")
    source_reduced = reduced.set_index("sample_index")
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", sample)]
    scopes.extend(
        ("department", str(department), selected)
        for department, selected in sample.groupby("department", sort=True)
    )
    aggregate_rows: list[dict[str, Any]] = []
    for scope, department, selected in scopes:
        ids = selected["sample_index"].to_numpy(int)
        p_rows = source_primary.loc[ids]
        r_rows = source_reduced.loc[ids]
        p_metrics, _ = _metric_bundle(
            p_rows["y_true"].to_numpy(int),
            p_rows["y_pred"].to_numpy(int),
            p_rows.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
        )
        r_metrics, _ = _metric_bundle(
            r_rows["y_true"].to_numpy(int),
            r_rows["y_pred"].to_numpy(int),
            r_rows.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
        )
        aggregate_rows.append(
            {
                "scope": scope,
                "department": department,
                "n_samples": len(selected),
                "mean_total_variation": selected["total_variation"].mean(),
                "median_total_variation": selected["total_variation"].median(),
                "p90_total_variation": selected["total_variation"].quantile(0.9),
                "mean_max_absolute_probability_change": selected["max_absolute_probability_change"].mean(),
                "mean_delta_prob_class_2": selected["delta_prob_class_2"].mean(),
                "mean_delta_prob_class_3": selected["delta_prob_class_3"].mean(),
                "mean_delta_prob_class_4": selected["delta_prob_class_4"].mean(),
                "mean_delta_true_class_probability": selected["delta_true_class_probability"].mean(),
                "mean_delta_confidence_margin": selected["delta_confidence_margin"].mean(),
                "mean_delta_ordinal_log_odds_margin": selected["delta_ordinal_log_odds_margin"].mean(),
                "prediction_change_rate": selected["prediction_changed"].mean(),
                "mean_absolute_ordinal_prediction_shift": selected["absolute_ordinal_prediction_shift"].mean(),
                "primary_macro_f1": p_metrics["macro_f1"],
                "proxy_reduced_macro_f1": r_metrics["macro_f1"],
                "delta_macro_f1": r_metrics["macro_f1"] - p_metrics["macro_f1"],
                "primary_accuracy": float(np.mean(p_rows["y_true"].to_numpy(int) == p_rows["y_pred"].to_numpy(int))),
                "proxy_reduced_accuracy": float(np.mean(r_rows["y_true"].to_numpy(int) == r_rows["y_pred"].to_numpy(int))),
                "comparison_direction": contract["proxy_prediction_comparison"]["comparison_direction"],
                "inference_scope": "descriptive_paired_exact_oof_prediction_change",
                "claim_boundary": "performance_model_dependence_sensitivity_not_causality_or_discrimination_evidence",
            }
        )
    return sample, pd.DataFrame(aggregate_rows)


def _p3_features(frame: pd.DataFrame, contract: Mapping[str, Any]) -> pd.DataFrame:
    source = contract["source_contracts"]["feature_availability"]["path"]
    feature_contract = _load_json(PROJECT_ROOT / source)
    feature_names = [str(row["feature_name"]) for row in feature_contract["features"]]
    policies = {row["policy_id"]: row for row in feature_contract["policies"]}
    excluded = set(policies["P3"]["excluded_features"])
    retained = [name for name in feature_names if name not in excluded]
    _require(len(retained) == 20 and "EmpJobRole" in retained, "Independent P3 features drifted.")
    return frame.loc[:, retained].copy()


def _aligned_outputs(pipeline: Any, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    classifier = pipeline.named_steps["model"]
    classes = np.asarray(classifier.classes_, dtype=int)
    order = [int(np.flatnonzero(classes == label)[0]) for label in LABELS]
    probability = np.asarray(pipeline.predict_proba(features), dtype=float)[:, order]
    transformed = pipeline.named_steps["preprocessor"].transform(features)
    margin = np.asarray(classifier.model_.predict(transformed, output_margin=True), dtype=float)[:, order]
    return probability, margin


def _permutation(
    contract: Mapping[str, Any],
    frame: pd.DataFrame,
    features: pd.DataFrame,
    oof: pd.DataFrame,
    artifacts: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = oof[oof["system_id"] == PRIMARY_SYSTEM].sort_values("sample_index")
    y_true = primary["y_true"].to_numpy(int)
    original_probability = primary.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    original_prediction = primary["y_pred"].to_numpy(int)
    original_margin = np.full_like(original_probability, np.nan)
    fold_records: list[tuple[int, np.ndarray, Any, np.ndarray, np.ndarray]] = []
    for fold, model in sorted(artifacts.fold_models.items()):
        ids = np.asarray(model.test_sample_indices, dtype=int)
        probability, margin = _aligned_outputs(model.pipeline, features.loc[ids])
        _require(np.allclose(probability, original_probability[ids], rtol=0.0, atol=1e-6), f"Independent fold-{fold} replay drifted.")
        original_margin[ids] = margin
        fold_records.append(
            (
                int(fold),
                ids,
                model.pipeline,
                frame.loc[ids, "EmpDepartment"].astype(str).to_numpy(),
                features.loc[ids, "EmpJobRole"].astype(str).to_numpy(),
            )
        )
    original_metrics, _ = _metric_bundle(y_true, original_prediction, original_probability)
    original_argmax = np.argmax(original_probability, axis=1)
    sample_frames: list[pd.DataFrame] = []
    repetition_rows: list[dict[str, Any]] = []
    schemes = contract["job_role_permutation"]["schemes"]
    for scheme_index, scheme in enumerate(schemes):
        for seed in contract["job_role_permutation"]["seeds"]:
            probability = np.full_like(original_probability, np.nan)
            margin = np.full_like(original_margin, np.nan)
            changed = np.zeros(len(features), dtype=bool)
            for fold, ids, pipeline, departments, original_roles in fold_records:
                perturbed = features.loc[ids].copy()
                roles = original_roles.copy()
                rng = np.random.default_rng(np.random.SeedSequence([seed, fold, scheme_index]))
                if scheme == "marginal_within_outer_test_fold":
                    roles = roles[rng.permutation(len(roles))]
                else:
                    for department in sorted(set(departments)):
                        positions = np.flatnonzero(departments == department)
                        roles[positions] = roles[positions][rng.permutation(len(positions))]
                perturbed.loc[:, "EmpJobRole"] = roles
                probability[ids], margin[ids] = _aligned_outputs(pipeline, perturbed)
                changed[ids] = roles != original_roles
            prediction = np.asarray(LABELS)[np.argmax(probability, axis=1)]
            total_variation = np.abs(probability - original_probability).sum(axis=1) / 2.0
            prediction_changed = prediction != original_prediction
            probability_drop = original_probability[np.arange(1200), original_argmax] - probability[np.arange(1200), original_argmax]
            margin_drop = original_margin[np.arange(1200), original_argmax] - margin[np.arange(1200), original_argmax]
            sample_frames.append(
                pd.DataFrame(
                    {
                        "scheme": scheme,
                        "seed": seed,
                        "sample_index": np.arange(1200, dtype=int),
                        "outer_fold": primary["outer_fold"].to_numpy(int),
                        "department": frame["EmpDepartment"].astype(str).to_numpy(),
                        "y_true": y_true,
                        "original_y_pred": original_prediction,
                        "permuted_y_pred": prediction,
                        "job_role_value_changed": changed,
                        "total_variation": total_variation,
                        "prediction_changed": prediction_changed,
                        "absolute_ordinal_prediction_shift": np.abs(prediction - original_prediction),
                        "original_predicted_class_probability_drop": probability_drop,
                        "original_predicted_class_raw_margin_drop": margin_drop,
                    }
                )
            )
            perturbed_metrics, _ = _metric_bundle(y_true, prediction, probability)
            repetition_rows.append(
                {
                    "scheme": scheme,
                    "seed": seed,
                    "n_samples": 1200,
                    "job_role_value_changed_fraction": changed.mean(),
                    "mean_total_variation": total_variation.mean(),
                    "prediction_change_rate": prediction_changed.mean(),
                    "mean_absolute_ordinal_prediction_shift": np.abs(prediction - original_prediction).mean(),
                    "mean_original_predicted_class_probability_drop": probability_drop.mean(),
                    "mean_original_predicted_class_raw_margin_drop": margin_drop.mean(),
                    **{f"original_{metric}": original_metrics[metric] for metric in METRICS},
                    **{f"permuted_{metric}": perturbed_metrics[metric] for metric in METRICS},
                    **{f"delta_{metric}": perturbed_metrics[metric] - original_metrics[metric] for metric in METRICS},
                    "inference_scope": "dependent_random_perturbation_repetitions_descriptive_not_confidence_intervals",
                    "claim_boundary": "model_dependence_sensitivity_not_causal_effect_or_discrimination_evidence",
                }
            )
    sample = pd.concat(sample_frames, ignore_index=True)
    repetitions = pd.DataFrame(repetition_rows)
    summary_metrics = [
        "job_role_value_changed_fraction",
        "mean_total_variation",
        "prediction_change_rate",
        "mean_absolute_ordinal_prediction_shift",
        "mean_original_predicted_class_probability_drop",
        "mean_original_predicted_class_raw_margin_drop",
        *[f"delta_{metric}" for metric in METRICS],
    ]
    summary_rows: list[dict[str, Any]] = []
    for scheme, selected in repetitions.groupby("scheme", sort=False):
        row: dict[str, Any] = {
            "scheme": scheme,
            "n_repetitions": len(selected),
            "n_samples_per_repetition": 1200,
        }
        for metric in summary_metrics:
            row[f"{metric}_mean"] = selected[metric].mean()
            row[f"{metric}_std"] = selected[metric].std(ddof=1)
            row[f"{metric}_min"] = selected[metric].min()
            row[f"{metric}_max"] = selected[metric].max()
        row["inference_scope"] = "descriptive_perturbation_variability_not_sampling_uncertainty"
        row["claim_boundary"] = "JobRole_permutation_sensitivity_not_causality_fairness_or_discrimination_evidence"
        summary_rows.append(row)
    return sample, repetitions, pd.DataFrame(summary_rows)


def _load_independent_sources(contract: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    sources = contract["source_contracts"]
    canonical = load_canonical_dataset(
        sources["canonical_loader_config"]["path"],
        contract["dataset_key"],
        sources["acquisition_manifest"]["path"],
        allow_download=False,
    )
    frame = canonical.frame.copy()
    oof = _read_csv(PROJECT_ROOT / sources["fairness_oof_predictions"]["path"])
    features = _p3_features(frame, contract)
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
    return frame, oof, artifacts


def validate_subgroup_proxy_use_run_v3(
    run_dir: Path | str = DEFAULT_SUBGROUP_PROXY_USE_RUN,
) -> dict[str, Any]:
    """Validate a complete Phase 2C run without importing runner calculations."""

    root = Path(run_dir)
    _require(root.is_dir(), f"Phase 2C run directory is absent: {root.as_posix()}.")
    inventory = {path.name for path in root.iterdir() if path.is_file()}
    _require(inventory == EXPECTED_FILES, f"Phase 2C closed-world inventory drifted: {sorted(inventory ^ EXPECTED_FILES)}.")
    _require(not any(path.is_dir() for path in root.iterdir()), "Phase 2C run contains an unexpected directory.")
    metadata = _load_json(root / "stage_metadata.json")
    _require(metadata.get("status") == "complete", "Phase 2C metadata is incomplete.")
    _require(metadata.get("stage") == "subgroup_proxy_use_v3", "Phase 2C stage drifted.")
    _require(metadata.get("run_id") == root.parent.name, "Phase 2C run-id/path drifted.")
    _require(str(metadata["run_id"]).endswith("_e314bb5"), "Phase 2C run-id suffix drifted.")
    generation_commit = str(metadata.get("git_identity", {}).get("commit"))
    _require(generation_commit == EXPECTED_GENERATION_COMMIT, "Phase 2C generation commit drifted.")
    expected_metadata = {
        "subgroup_metric_rows": 2025,
        "subgroup_gap_rows": 486,
        "primary_gap_interval_rows": 162,
        "proxy_prediction_change_sample_rows": 1200,
        "proxy_department_aggregate_rows": 7,
        "jobrole_permutation_sample_rows": 48000,
        "jobrole_permutation_repetition_rows": 40,
        "jobrole_permutation_scheme_rows": 2,
        "department_reconstructability_metric_rows": 6,
        "department_reconstructability_difference_rows": 3,
        "new_performance_model_fit_calls": 0,
        "new_proxy_reconstruction_model_fit_calls": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
        "fairness_certification_allowed": False,
        "formal_discrimination_claim_allowed": False,
        "department_reconstructability_proves_performance_use": False,
        "permutation_supports_causal_claim": False,
    }
    for field, expected in expected_metadata.items():
        _require(metadata.get(field) == expected, f"Phase 2C metadata {field} drifted.")
    validate_policy_receipt(metadata["runtime_policy"])
    output_hashes = metadata.get("output_hashes")
    _require(isinstance(output_hashes, Mapping) and set(output_hashes) == OUTPUT_HASH_FILES, "Phase 2C output-hash inventory drifted.")
    for filename, expected_hash in output_hashes.items():
        _require(sha256_file(root / filename) == expected_hash, f"Phase 2C output hash drifted for {filename}.")

    receipt = validate_subgroup_proxy_use_contract_v3()
    _require(receipt["contract_sha256"] == metadata["contract_sha256"], "Phase 2C contract hash drifted.")
    contract = _load_json(PROJECT_ROOT / DEFAULT_SUBGROUP_PROXY_USE_CONTRACT)
    scientific_inputs = metadata.get("scientific_inputs")
    _require(isinstance(scientific_inputs, Mapping), "Phase 2C scientific inputs are absent.")
    _require(_canonical_json_sha256(scientific_inputs) == metadata["scientific_input_sha256"], "Phase 2C scientific-input hash drifted.")
    _require(scientific_inputs.get("git_identity") == metadata.get("git_identity"), "Phase 2C Git identity drifted.")
    sources = scientific_inputs.get("bound_source_hashes")
    _require(isinstance(sources, Mapping) and set(sources) == set(contract["source_contracts"]), "Phase 2C bound-source inventory drifted.")
    for name, record in contract["source_contracts"].items():
        _require(sources[name] == record["sha256"], f"Phase 2C source receipt drifted for {name}.")
        _require(sha256_file(PROJECT_ROOT / record["path"]) == record["sha256"], f"Phase 2C source bytes drifted for {name}.")
    implementations = scientific_inputs.get("implementation_hashes")
    _require(isinstance(implementations, Mapping) and set(implementations) == EXPECTED_IMPLEMENTATIONS, "Phase 2C implementation inventory drifted.")
    for relative_path, expected_hash in implementations.items():
        blob_hash = hashlib.sha256(_git_blob(generation_commit, relative_path)).hexdigest()
        _require(blob_hash == expected_hash, f"Phase 2C generation blob drifted for {relative_path}.")
    contract_blob = hashlib.sha256(
        _git_blob(generation_commit, DEFAULT_SUBGROUP_PROXY_USE_CONTRACT.as_posix())
    ).hexdigest()
    _require(contract_blob == metadata["contract_sha256"], "Generation Phase 2C contract blob drifted.")

    observed = {filename: _read_csv(root / filename) for filename in EXPECTED_ROW_COUNTS}
    for filename, count in EXPECTED_ROW_COUNTS.items():
        _require(len(observed[filename]) == count, f"{filename} row count drifted.")
    frame, oof, artifacts = _load_independent_sources(contract)
    group_grid = _subgroup_grid(contract, frame, oof)
    gap_grid = _gap_grid(group_grid)
    intervals, bootstrap_receipt = _bootstrap_intervals(
        contract, frame, oof, group_grid, gap_grid
    )
    proxy_sample, proxy_aggregate = _proxy_change(contract, frame, oof)
    features = _p3_features(frame, contract)
    permutation_sample, permutation_repetitions, permutation_summary = _permutation(
        contract, frame, features, oof, artifacts
    )
    reconstruct_metrics = _read_csv(
        PROJECT_ROOT / contract["source_contracts"]["proxy_metric_intervals"]["path"]
    )
    reconstruct_metrics["v3_interpretation"] = "department_reconstructability_from_feature_space_not_performance_model_department_use"
    reconstruct_metrics["performance_dependence_claim_allowed"] = False
    reconstruct_differences = _read_csv(
        PROJECT_ROOT
        / contract["source_contracts"]["proxy_policy_paired_differences"]["path"]
    )
    reconstruct_differences["v3_interpretation"] = "JobRole_contribution_to_department_reconstructability_not_performance_model_dependence"
    reconstruct_differences["performance_dependence_claim_allowed"] = False
    expected = {
        "subgroup_metric_grid.csv": group_grid,
        "subgroup_gap_sensitivity.csv": gap_grid,
        "primary_gap_bootstrap_intervals.csv": intervals,
        "proxy_prediction_change_sample.csv": proxy_sample,
        "proxy_prediction_change_by_department.csv": proxy_aggregate,
        "jobrole_permutation_sample.csv": permutation_sample,
        "jobrole_permutation_repetition.csv": permutation_repetitions,
        "jobrole_permutation_summary.csv": permutation_summary,
        "department_reconstructability_metrics.csv": reconstruct_metrics,
        "department_reconstructability_differences.csv": reconstruct_differences,
    }
    sort_columns = {
        "subgroup_metric_grid.csv": ["system_id", "support_threshold", "attribute", "group", "metric"],
        "subgroup_gap_sensitivity.csv": ["system_id", "support_threshold", "attribute", "metric"],
        "primary_gap_bootstrap_intervals.csv": ["support_threshold", "attribute", "metric"],
        "proxy_prediction_change_sample.csv": ["sample_index"],
        "proxy_prediction_change_by_department.csv": ["scope", "department"],
        "jobrole_permutation_sample.csv": ["scheme", "seed", "sample_index"],
        "jobrole_permutation_repetition.csv": ["scheme", "seed"],
        "jobrole_permutation_summary.csv": ["scheme"],
        "department_reconstructability_metrics.csv": ["system_id", "metric"],
        "department_reconstructability_differences.csv": ["comparison_id", "metric"],
    }
    for filename, expected_frame in expected.items():
        _assert_frame_equal(
            observed[filename],
            expected_frame,
            sort_columns=sort_columns[filename],
            context=filename,
            tolerance=1e-11 if "permutation" in filename else 1e-12,
        )

    diagnostic = _load_json(root / "diagnostic_receipt.json")
    for field, expected_value in {
        "run_id": metadata["run_id"],
        "contract_sha256": metadata["contract_sha256"],
        "scientific_input_sha256": metadata["scientific_input_sha256"],
        **expected_metadata,
    }.items():
        _require(diagnostic.get(field) == expected_value, f"Phase 2C diagnostic {field} drifted.")
    _require(diagnostic.get("bootstrap") == bootstrap_receipt, "Phase 2C bootstrap receipt drifted.")
    status_counts = {
        str(key): int(value)
        for key, value in group_grid["support_status"].value_counts().sort_index().items()
    }
    _require(diagnostic.get("subgroup_support_status_counts") == status_counts, "Phase 2C support-status receipt drifted.")
    interval_frame = observed["primary_gap_bootstrap_intervals.csv"]
    _require(not _bool_values(interval_frame["model_training_variability_included"], context="bootstrap training variability").any(), "Bootstrap incorrectly includes training variability.")
    _require(interval_frame["n_complete_familywise_draws"].eq(5000).all(), "Incomplete simultaneous bootstrap draws were retained.")
    _require(not _bool_values(observed["department_reconstructability_metrics.csv"]["performance_dependence_claim_allowed"], context="reconstructability use claim").any(), "Reconstructability incorrectly permits a performance-use claim.")

    overall = observed["proxy_prediction_change_by_department.csv"].query("scope == 'overall'").iloc[0]
    schemes = observed["jobrole_permutation_summary.csv"].set_index("scheme")
    return {
        "status": "passed",
        "run_id": metadata["run_id"],
        "generation_commit": generation_commit,
        "contract_sha256": metadata["contract_sha256"],
        "scientific_input_sha256": metadata["scientific_input_sha256"],
        "file_count": len(EXPECTED_FILES),
        "subgroup_metric_rows": len(group_grid),
        "subgroup_gap_rows": len(gap_grid),
        "primary_gap_interval_rows": len(intervals),
        "bootstrap_resample_hash": bootstrap_receipt["resample_hash"],
        "familywise_critical_value": bootstrap_receipt["familywise_critical_value"],
        "proxy_overall_mean_total_variation": float(overall["mean_total_variation"]),
        "proxy_overall_prediction_change_rate": float(overall["prediction_change_rate"]),
        "proxy_overall_delta_macro_f1": float(overall["delta_macro_f1"]),
        "marginal_permutation_mean_total_variation": float(
            schemes.loc["marginal_within_outer_test_fold", "mean_total_variation_mean"]
        ),
        "conditional_permutation_mean_total_variation": float(
            schemes.loc[
                "department_conditional_within_outer_test_fold_and_department",
                "mean_total_variation_mean",
            ]
        ),
        "new_model_fit_calls": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_SUBGROUP_PROXY_USE_RUN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    print(
        json.dumps(
            validate_subgroup_proxy_use_run_v3(args.run_dir),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SUBGROUP_PROXY_USE_RUN",
    "EXPECTED_FILES",
    "V3SubgroupProxyUseRunValidationError",
    "validate_subgroup_proxy_use_run_v3",
]
