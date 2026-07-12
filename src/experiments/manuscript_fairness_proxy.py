from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.preprocess import load_validated_or_raw_data
from src.experiments.leakage_safe_cv import LabelEncodedXGBClassifier, make_preprocessor
from src.experiments.manuscript_policy_ablation import exact_policy_frame, resolve_seed
from src.experiments.proxy_analysis import feature_proxy_associations
from src.governance.manuscript_contract import canonical_config_hash
from src.utils.config_loader import load_config


DEFAULT_CONFIG = Path("configs/manuscript_final.yaml")
REQUIRED_POLICY_COMPARISONS = (
    "no_salary_hike_no_attrition",
    "no_salary_hike_no_attrition_no_department",
    "no_salary_hike_no_attrition_no_department_no_job_role",
)
OVERALL_METRICS = ("accuracy", "macro_f1")
CLASS_METRICS = (
    "positive_prediction_rate",
    "true_positive_rate",
    "false_positive_rate",
    "precision",
    "mean_predicted_probability",
)


class FairnessProxyError(RuntimeError):
    """Raised when the canonical fairness/proxy contract cannot be evaluated."""


def _settings(config_path: str | Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    raw = load_config(config_path)
    settings = raw.get("manuscript_final", raw)
    if not isinstance(settings, dict):
        raise FairnessProxyError("Canonical config must contain a manuscript_final mapping.")
    return raw, settings


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - old scikit-learn compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _model_parameters(settings: Mapping[str, Any]) -> Dict[str, Any]:
    model = settings.get("model", {})
    source = model.get("xgboost", {}) if isinstance(model, Mapping) else {}
    if not isinstance(source, Mapping):
        raise FairnessProxyError("model.xgboost must be a mapping.")
    allowed = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "objective",
        "eval_metric",
        "random_state",
        "n_jobs",
    }
    parameters = {str(key): value for key, value in source.items() if key != "random_state_seed"}
    unexpected = sorted(set(parameters).difference(allowed))
    if unexpected:
        raise FairnessProxyError(f"Unsupported canonical XGBoost parameters: {unexpected}")
    return parameters


def _audit_category(attribute: str, sensitive: set[str]) -> str:
    if attribute in sensitive:
        return "protected_or_sensitive_descriptive_audit"
    return "exploratory_operational_subgroup_diagnostic"


def transform_audit_attribute(
    values: pd.Series,
    attribute: str,
    transform: Mapping[str, Any] | None,
) -> pd.Series:
    """Apply the predeclared audit transform without learning from outcomes."""

    if not transform:
        return values.astype("string").fillna("__MISSING__").astype(str)
    if transform.get("type") != "numeric_bins":
        raise FairnessProxyError(f"Unsupported transform for {attribute}: {transform}")
    edges = [float(value) for value in transform.get("edges", [])]
    labels = [str(value) for value in transform.get("labels", [])]
    if len(edges) != len(labels) + 1 or len(labels) < 2:
        raise FairnessProxyError(f"Invalid numeric-bin definition for {attribute}.")
    numeric = pd.to_numeric(values, errors="coerce")
    binned = pd.cut(numeric, bins=edges, labels=labels, right=True, include_lowest=True)
    return binned.astype("string").fillna("__MISSING__").astype(str)


def generate_common_fold_oof_predictions(
    data: pd.DataFrame,
    settings: Mapping[str, Any],
    *,
    run_id: str,
    config_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = settings.get("target", {})
    target_column = str(target.get("column", "PerformanceRating"))
    labels = [int(value) for value in target.get("labels", [2, 3, 4])]
    definitions = settings.get("feature_policies", {}).get("definitions", {})
    missing = [policy for policy in REQUIRED_POLICY_COMPARISONS if policy not in definitions]
    if missing:
        raise FairnessProxyError(f"Missing policy comparisons: {missing}")

    identifier_fields = settings.get("governance_fields", {}).get("identifier_fields", ["EmpNumber"])
    id_column = str(identifier_fields[0] if identifier_fields else "EmpNumber")
    y = data[target_column].astype(int).reset_index(drop=True)
    cv = settings.get("evaluation", {}).get("cv", {})
    n_splits = int(cv.get("n_splits", 10))
    seed = resolve_seed(settings, cv.get("seed", "cv"))
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=bool(cv.get("shuffle", True)),
        random_state=seed,
    )
    folds = list(splitter.split(np.zeros(len(y)), y))
    fold_assignment = np.zeros(len(y), dtype=int)
    for fold, (_, test_positions) in enumerate(folds, start=1):
        fold_assignment[test_positions] = fold
    assignment = pd.DataFrame(
        {
            "run_id": run_id,
            "config_hash": config_hash,
            "sample_index": np.arange(len(data), dtype=int),
            "fold": fold_assignment,
            "y_true": y,
        }
    )
    assignment["fold_assignment_sha256"] = hashlib.sha256(
        assignment[["sample_index", "fold", "y_true"]].to_csv(index=False).encode("utf-8")
    ).hexdigest()

    model_seed = resolve_seed(settings, settings.get("seeds", {}).get("model", "model"))
    parameters = _model_parameters(settings)
    parameters["random_state"] = model_seed
    rows: list[Dict[str, Any]] = []
    reset = data.reset_index(drop=True)
    for policy in REQUIRED_POLICY_COMPARISONS:
        X, excluded = exact_policy_frame(
            reset,
            policy,
            definitions[policy],
            target_column=target_column,
            id_column=id_column,
        )
        for fold, (train_positions, test_positions) in enumerate(folds, start=1):
            pipeline = Pipeline(
                [
                    ("preprocessor", make_preprocessor(X.iloc[train_positions])),
                    ("model", LabelEncodedXGBClassifier(**parameters)),
                ]
            )
            pipeline.fit(X.iloc[train_positions], y.iloc[train_positions])
            predictions = pipeline.predict(X.iloc[test_positions])
            probabilities = pipeline.predict_proba(X.iloc[test_positions])
            model_classes = [int(value) for value in pipeline.named_steps["model"].classes_]
            class_positions = {label: model_classes.index(label) for label in labels}
            for row_position, sample_index in enumerate(test_positions):
                record: Dict[str, Any] = {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "policy": policy,
                    "fold": fold,
                    "sample_index": int(sample_index),
                    "y_true": int(y.iloc[sample_index]),
                    "y_pred": int(predictions[row_position]),
                    "n_features": int(X.shape[1]),
                    "excluded_features": json.dumps(excluded),
                    "fold_assignment_sha256": assignment["fold_assignment_sha256"].iloc[0],
                }
                for label in labels:
                    record[f"prob_class_{label}"] = float(probabilities[row_position, class_positions[label]])
                rows.append(record)
    predictions = pd.DataFrame(rows).sort_values(["policy", "sample_index"]).reset_index(drop=True)
    expected = len(data) * len(REQUIRED_POLICY_COMPARISONS)
    if len(predictions) != expected or predictions.groupby("policy")["sample_index"].nunique().min() != len(data):
        raise FairnessProxyError("OOF predictions do not cover every case exactly once per policy.")
    return predictions, assignment


def compute_group_metric_rows(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    *,
    labels: Sequence[int],
    attributes: Sequence[str],
    transforms: Mapping[str, Mapping[str, Any]],
    sensitive_attributes: set[str],
    minimum_group_support: int,
    minimum_class_denominator: int,
) -> pd.DataFrame:
    """Return long-form subgroup metrics with their actual applicability denominators."""

    audit_values: Dict[str, pd.Series] = {}
    reset = data.reset_index(drop=True)
    for attribute in attributes:
        if attribute not in reset:
            continue
        audit_values[attribute] = transform_audit_attribute(reset[attribute], attribute, transforms.get(attribute))

    rows: list[Dict[str, Any]] = []
    for policy, policy_predictions in predictions.groupby("policy", sort=False):
        ordered = policy_predictions.sort_values("sample_index").reset_index(drop=True)
        sample_indices = ordered["sample_index"].astype(int).to_numpy()
        y_true = ordered["y_true"].astype(int).to_numpy()
        y_pred = ordered["y_pred"].astype(int).to_numpy()
        for attribute, full_values in audit_values.items():
            values = full_values.iloc[sample_indices].to_numpy(dtype=str)
            for group_value in sorted(pd.unique(values).tolist()):
                mask = values == group_value
                group_true = y_true[mask]
                group_pred = y_pred[mask]
                n_samples = int(mask.sum())
                support_ok = n_samples >= minimum_group_support
                common = {
                    "run_id": ordered["run_id"].iloc[0],
                    "config_hash": ordered["config_hash"].iloc[0],
                    "policy": policy,
                    "attribute": attribute,
                    "group_value": str(group_value),
                    "interpretation_category": _audit_category(attribute, sensitive_attributes),
                    "n_samples": n_samples,
                    "minimum_group_support_threshold": minimum_group_support,
                    "group_support_eligible": support_ok,
                }
                overall = {
                    "accuracy": float(accuracy_score(group_true, group_pred)),
                    "macro_f1": float(
                        f1_score(group_true, group_pred, labels=list(labels), average="macro", zero_division=0)
                    ),
                }
                for metric, value in overall.items():
                    rows.append(
                        {
                            **common,
                            "metric": metric,
                            "class_label": np.nan,
                            "metric_value": value,
                            "metric_denominator": n_samples,
                            "minimum_metric_denominator_threshold": minimum_group_support,
                            "metric_denominator_eligible": support_ok,
                            "eligible_for_gap": support_ok,
                        }
                    )

                for label in labels:
                    true_positive = int(np.sum((group_true == label) & (group_pred == label)))
                    false_positive = int(np.sum((group_true != label) & (group_pred == label)))
                    actual_support = int(np.sum(group_true == label))
                    predicted_support = int(np.sum(group_pred == label))
                    negative_support = n_samples - actual_support
                    probability = ordered.loc[mask, f"prob_class_{label}"].to_numpy(dtype=float)
                    definitions = {
                        "positive_prediction_rate": (predicted_support / n_samples if n_samples else math.nan, n_samples),
                        "true_positive_rate": (
                            true_positive / actual_support if actual_support else math.nan,
                            actual_support,
                        ),
                        "false_positive_rate": (
                            false_positive / negative_support if negative_support else math.nan,
                            negative_support,
                        ),
                        "precision": (
                            true_positive / predicted_support if predicted_support else math.nan,
                            predicted_support,
                        ),
                        "mean_predicted_probability": (
                            float(probability.mean()) if len(probability) else math.nan,
                            n_samples,
                        ),
                    }
                    for metric, (value, denominator) in definitions.items():
                        denominator_threshold = (
                            minimum_group_support
                            if metric in {"positive_prediction_rate", "mean_predicted_probability"}
                            else minimum_class_denominator
                        )
                        denominator_ok = denominator >= denominator_threshold
                        rows.append(
                            {
                                **common,
                                "metric": metric,
                                "class_label": int(label),
                                "metric_value": value,
                                "metric_denominator": int(denominator),
                                "minimum_metric_denominator_threshold": int(denominator_threshold),
                                "metric_denominator_eligible": bool(denominator_ok),
                                "eligible_for_gap": bool(support_ok and denominator_ok and np.isfinite(value)),
                            }
                        )
    return pd.DataFrame(rows)


def _stratified_bootstrap_indices(
    fold: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sampled_strata: list[np.ndarray] = []
    strata = pd.DataFrame({"position": np.arange(len(y_true)), "fold": fold, "y_true": y_true})
    for _, stratum in strata.groupby(["fold", "y_true"], sort=True):
        positions = stratum["position"].to_numpy(dtype=int)
        sampled_strata.append(rng.choice(positions, size=(n_bootstrap, len(positions)), replace=True))
    return np.concatenate(sampled_strata, axis=1)


def _bootstrap_attribute_gaps(
    predictions: pd.DataFrame,
    group_values: np.ndarray,
    *,
    labels: Sequence[int],
    bootstrap_indices: np.ndarray,
    minimum_group_support: int,
    minimum_class_denominator: int,
) -> Dict[tuple[str, int | None], np.ndarray]:
    original_y_true = predictions["y_true"].astype(int).to_numpy()
    original_y_pred = predictions["y_pred"].astype(int).to_numpy()
    y_true = original_y_true[bootstrap_indices]
    y_pred = original_y_pred[bootstrap_indices]
    groups, group_codes = np.unique(group_values.astype(str), return_inverse=True)
    sampled_groups = group_codes[bootstrap_indices]
    n_bootstrap = bootstrap_indices.shape[0]
    values: Dict[tuple[str, int | None], list[np.ndarray]] = {
        (metric, None): [] for metric in OVERALL_METRICS
    }
    values.update({(metric, int(label)): [] for metric in CLASS_METRICS for label in labels})

    for group_code in range(len(groups)):
        original_member = group_codes == group_code
        original_group_n = int(original_member.sum())
        original_group_ok = original_group_n >= minimum_group_support
        member = sampled_groups == group_code
        group_n = member.sum(axis=1)
        group_ok = group_n >= minimum_group_support
        correct = ((y_true == y_pred) & member).sum(axis=1)
        accuracy = np.divide(correct, group_n, out=np.full(n_bootstrap, np.nan), where=group_n > 0)
        values[("accuracy", None)].append(
            np.where(group_ok & original_group_ok, accuracy, np.nan)
        )

        f1_by_class: list[np.ndarray] = []
        for label in labels:
            true_label = y_true == label
            pred_label = y_pred == label
            tp = (member & true_label & pred_label).sum(axis=1)
            fp = (member & ~true_label & pred_label).sum(axis=1)
            fn = (member & true_label & ~pred_label).sum(axis=1)
            denominator = 2 * tp + fp + fn
            f1_by_class.append(
                np.divide(2 * tp, denominator, out=np.zeros(n_bootstrap, dtype=float), where=denominator > 0)
            )
        macro_f1 = np.mean(np.vstack(f1_by_class), axis=0)
        values[("macro_f1", None)].append(
            np.where(group_ok & original_group_ok, macro_f1, np.nan)
        )

        for label in labels:
            true_label = y_true == label
            pred_label = y_pred == label
            tp = (member & true_label & pred_label).sum(axis=1)
            fp = (member & ~true_label & pred_label).sum(axis=1)
            actual = (member & true_label).sum(axis=1)
            predicted = (member & pred_label).sum(axis=1)
            negative = group_n - actual
            probability = predictions[f"prob_class_{label}"].to_numpy(dtype=float)[bootstrap_indices]
            probability_sum = np.where(member, probability, 0.0).sum(axis=1)
            original_actual = int(np.sum(original_member & (original_y_true == label)))
            original_predicted = int(np.sum(original_member & (original_y_pred == label)))
            original_negative = original_group_n - original_actual
            metric_values = {
                "positive_prediction_rate": (
                    np.divide(predicted, group_n, out=np.full(n_bootstrap, np.nan), where=group_n > 0),
                    group_n,
                    minimum_group_support,
                    original_group_n,
                ),
                "true_positive_rate": (
                    np.divide(tp, actual, out=np.full(n_bootstrap, np.nan), where=actual > 0),
                    actual,
                    minimum_class_denominator,
                    original_actual,
                ),
                "false_positive_rate": (
                    np.divide(fp, negative, out=np.full(n_bootstrap, np.nan), where=negative > 0),
                    negative,
                    minimum_class_denominator,
                    original_negative,
                ),
                "precision": (
                    np.divide(tp, predicted, out=np.full(n_bootstrap, np.nan), where=predicted > 0),
                    predicted,
                    minimum_class_denominator,
                    original_predicted,
                ),
                "mean_predicted_probability": (
                    np.divide(probability_sum, group_n, out=np.full(n_bootstrap, np.nan), where=group_n > 0),
                    group_n,
                    minimum_group_support,
                    original_group_n,
                ),
            }
            for metric, (metric_value, denominator, threshold, original_denominator) in metric_values.items():
                eligible = (
                    group_ok
                    & original_group_ok
                    & (denominator >= threshold)
                    & (original_denominator >= threshold)
                    & np.isfinite(metric_value)
                )
                values[(metric, int(label))].append(np.where(eligible, metric_value, np.nan))

    gaps: Dict[tuple[str, int | None], np.ndarray] = {}
    for key, group_metric_values in values.items():
        matrix = np.column_stack(group_metric_values)
        finite = np.isfinite(matrix)
        valid_groups = finite.sum(axis=1)
        minimum = np.where(finite, matrix, np.inf).min(axis=1)
        maximum = np.where(finite, matrix, -np.inf).max(axis=1)
        gaps[key] = np.where(valid_groups >= 2, maximum - minimum, np.nan)
    return gaps


def summarize_disparities_with_bootstrap(
    group_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    *,
    labels: Sequence[int],
    attributes: Sequence[str],
    transforms: Mapping[str, Mapping[str, Any]],
    sensitive_attributes: set[str],
    minimum_group_support: int,
    minimum_class_denominator: int,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
    minimum_valid_fraction: float,
    wide_interval_threshold: float,
) -> pd.DataFrame:
    alpha = 1.0 - confidence_level
    rows: list[Dict[str, Any]] = []
    reset = data.reset_index(drop=True)
    first_policy = predictions["policy"].drop_duplicates().iloc[0]
    base = predictions[predictions["policy"] == first_policy].sort_values("sample_index")
    bootstrap_indices = _stratified_bootstrap_indices(
        base["fold"].to_numpy(dtype=int),
        base["y_true"].to_numpy(dtype=int),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    for policy, policy_predictions in predictions.groupby("policy", sort=False):
        ordered = policy_predictions.sort_values("sample_index").reset_index(drop=True)
        sample_indices = ordered["sample_index"].astype(int).to_numpy()
        for attribute in attributes:
            if attribute not in reset:
                continue
            attribute_values = transform_audit_attribute(
                reset[attribute], attribute, transforms.get(attribute)
            ).iloc[sample_indices].to_numpy(dtype=str)
            bootstrap_gaps = _bootstrap_attribute_gaps(
                ordered,
                attribute_values,
                labels=labels,
                bootstrap_indices=bootstrap_indices,
                minimum_group_support=minimum_group_support,
                minimum_class_denominator=minimum_class_denominator,
            )
            attribute_metrics = group_metrics[
                (group_metrics["policy"] == policy) & (group_metrics["attribute"] == attribute)
            ]
            for (metric, class_label), metric_rows in attribute_metrics.groupby(
                ["metric", "class_label"], dropna=False, sort=False
            ):
                normalized_class = None if pd.isna(class_label) else int(class_label)
                eligible = metric_rows[metric_rows["eligible_for_gap"].astype(bool)].copy()
                point_values = pd.to_numeric(eligible["metric_value"], errors="coerce").dropna()
                gap = float(point_values.max() - point_values.min()) if len(point_values) >= 2 else math.nan
                samples = bootstrap_gaps[(str(metric), normalized_class)]
                finite_samples = samples[np.isfinite(samples)]
                valid = int(len(finite_samples))
                if valid:
                    ci_low = float(np.quantile(finite_samples, alpha / 2.0))
                    ci_high = float(np.quantile(finite_samples, 1.0 - alpha / 2.0))
                else:
                    ci_low = ci_high = math.nan
                valid_fraction = valid / n_bootstrap if n_bootstrap else 0.0
                if len(point_values) < 2:
                    status = "insufficient_subgroup_or_metric_support"
                elif valid_fraction < minimum_valid_fraction:
                    status = "unstable_insufficient_valid_bootstrap_replicates"
                elif np.isfinite(ci_high - ci_low) and ci_high - ci_low > wide_interval_threshold:
                    status = "support_sufficient_but_interval_wide"
                else:
                    status = "support_sufficient_descriptive_estimate"
                limitations = (
                    "Descriptive OOF subgroup audit only; gaps do not establish discrimination, fairness, or causality. "
                    "Class-specific metrics require the configured metric denominator in every included subgroup."
                )
                rows.append(
                    {
                        "run_id": ordered["run_id"].iloc[0],
                        "config_hash": ordered["config_hash"].iloc[0],
                        "analysis_type": "subgroup_disparity",
                        "policy": policy,
                        "attribute": attribute,
                        "metric": str(metric),
                        "class_label": normalized_class,
                        "gap": gap,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "confidence_level": confidence_level,
                        "n_groups_total": int(metric_rows["group_value"].nunique()),
                        "n_groups_included": int(eligible["group_value"].nunique()),
                        "minimum_subgroup_support": (
                            int(eligible["n_samples"].min()) if not eligible.empty else 0
                        ),
                        "minimum_group_support_threshold": minimum_group_support,
                        "minimum_metric_denominator": (
                            int(eligible["metric_denominator"].min()) if not eligible.empty else 0
                        ),
                        "minimum_metric_denominator_threshold": int(
                            metric_rows["minimum_metric_denominator_threshold"].max()
                        ),
                        "bootstrap_samples_requested": n_bootstrap,
                        "valid_bootstrap_samples": valid,
                        "valid_bootstrap_fraction": valid_fraction,
                        "estimate_status": status,
                        "interpretation_category": _audit_category(attribute, sensitive_attributes),
                        "limitations": limitations,
                    }
                )
    return pd.DataFrame(rows).sort_values(["policy", "attribute", "metric", "class_label"], na_position="first")


def _proxy_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame if column not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def proxy_predictor_frames(
    data: pd.DataFrame,
    settings: Mapping[str, Any],
) -> Dict[str, tuple[pd.DataFrame, list[str], bool]]:
    """Apply performance policies, always removing the proxy target from proxy predictors."""

    target_column = str(settings.get("target", {}).get("column", "PerformanceRating"))
    id_fields = settings.get("governance_fields", {}).get("identifier_fields", ["EmpNumber"])
    id_column = str(id_fields[0] if id_fields else "EmpNumber")
    definitions = settings.get("feature_policies", {}).get("definitions", {})
    proxy_target = str(settings.get("proxy_analysis", {}).get("target", "EmpDepartment"))
    frames: Dict[str, tuple[pd.DataFrame, list[str], bool]] = {}
    for policy in REQUIRED_POLICY_COMPARISONS:
        X, _ = exact_policy_frame(
            data,
            policy,
            definitions[policy],
            target_column=target_column,
            id_column=id_column,
        )
        removed_by_proxy_safeguard = proxy_target in X
        X = X.drop(columns=[proxy_target], errors="ignore")
        if proxy_target in X:
            raise FairnessProxyError("Proxy target remained in proxy predictors.")
        frames[policy] = (X, list(X.columns), removed_by_proxy_safeguard)
    return frames


def _mean_t_interval(values: Iterable[float], confidence_level: float) -> tuple[float, float, float, float]:
    array = np.asarray(list(values), dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, 0.0, mean, mean
    standard_deviation = float(array.std(ddof=1))
    critical = float(student_t.ppf(0.5 + confidence_level / 2.0, len(array) - 1))
    half_width = critical * standard_deviation / math.sqrt(len(array))
    return mean, standard_deviation, mean - half_width, mean + half_width


def run_proxy_policy_comparison(
    data: pd.DataFrame,
    settings: Mapping[str, Any],
    *,
    run_id: str,
    config_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proxy = settings.get("proxy_analysis", {})
    target_column = str(proxy.get("target", "EmpDepartment"))
    if target_column not in data:
        raise FairnessProxyError(f"Proxy target is absent: {target_column}")
    y = data[target_column].astype("string").fillna("__MISSING__").astype(str).reset_index(drop=True)
    frames = proxy_predictor_frames(data.reset_index(drop=True), settings)
    cv = settings.get("evaluation", {}).get("cv", {})
    configured_splits = int(cv.get("n_splits", 10))
    n_splits = min(configured_splits, int(y.value_counts().min()))
    if n_splits < 2:
        raise FairnessProxyError("Proxy target lacks sufficient support for stratified CV.")
    seed = resolve_seed(settings, cv.get("seed", "cv"))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(y)), y))
    confidence = float(settings.get("evaluation", {}).get("bootstrap", {}).get("confidence_level", 0.95))
    fold_rows: list[Dict[str, Any]] = []
    association_frames: list[pd.DataFrame] = []
    for policy, (X, feature_names, removed_by_proxy_safeguard) in frames.items():
        for fold, (train_positions, test_positions) in enumerate(folds, start=1):
            classifier = Pipeline(
                [
                    ("preprocessor", _proxy_preprocessor(X.iloc[train_positions])),
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=3000,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            classifier.fit(X.iloc[train_positions], y.iloc[train_positions])
            prediction = classifier.predict(X.iloc[test_positions])
            fold_rows.append(
                {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "policy": policy,
                    "fold": fold,
                    "n_train": len(train_positions),
                    "n_test": len(test_positions),
                    "n_features": len(feature_names),
                    "proxy_target_absent_from_predictors": target_column not in X.columns,
                    "proxy_target_removed_by_proxy_safeguard": removed_by_proxy_safeguard,
                    "proxy_target_removal_method": (
                        "explicit_proxy_safeguard"
                        if removed_by_proxy_safeguard
                        else "already_excluded_by_performance_policy"
                    ),
                    "accuracy": float(accuracy_score(y.iloc[test_positions], prediction)),
                    "balanced_accuracy": float(balanced_accuracy_score(y.iloc[test_positions], prediction)),
                    "macro_f1": float(f1_score(y.iloc[test_positions], prediction, average="macro", zero_division=0)),
                }
            )
        associations = feature_proxy_associations(X, y, random_state=seed).copy()
        associations["proxy_watchlist"] = associations["feature"].isin(
            [str(value) for value in proxy.get("watchlist", [])]
        )
        associations.insert(0, "policy", policy)
        associations.insert(0, "config_hash", config_hash)
        associations.insert(0, "run_id", run_id)
        association_frames.append(associations)

    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows: list[Dict[str, Any]] = []
    for policy, group in fold_metrics.groupby("policy", sort=False):
        row: Dict[str, Any] = {
            "run_id": run_id,
            "config_hash": config_hash,
            "analysis_type": "department_reconstructability_proxy_risk",
            "policy": policy,
            "n_folds": int(group["fold"].nunique()),
            "n_features": int(group["n_features"].iloc[0]),
            "proxy_target_absent_from_predictors": bool(
                group["proxy_target_absent_from_predictors"].iloc[0]
            ),
            "proxy_target_removed_by_proxy_safeguard": bool(
                group["proxy_target_removed_by_proxy_safeguard"].iloc[0]
            ),
            "proxy_target_removal_method": str(group["proxy_target_removal_method"].iloc[0]),
            "interpretation_category": "proxy_risk_reconstructability_not_causal_use",
            "limitations": (
                "Department reconstructability is proxy-risk evidence only; it does not show that the performance "
                "model uses department causally or discriminatorily, and it does not establish unfairness."
            ),
        }
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            mean, std, low, high = _mean_t_interval(group[metric], confidence)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        summary_rows.append(row)
    associations = pd.concat(association_frames, ignore_index=True)
    return fold_metrics, pd.DataFrame(summary_rows), associations


def manuscript_fairness_proxy_table(
    disparity: pd.DataFrame,
    proxy_summary: pd.DataFrame,
) -> pd.DataFrame:
    fairness = disparity.copy()
    fairness["estimate"] = fairness["gap"]
    fairness["proxy_policy"] = ""
    proxy_rows: list[Dict[str, Any]] = []
    for row in proxy_summary.itertuples(index=False):
        for metric in ("balanced_accuracy", "macro_f1"):
            proxy_rows.append(
                {
                    "run_id": row.run_id,
                    "config_hash": row.config_hash,
                    "analysis_type": "department_reconstructability_proxy_risk",
                    "policy": row.policy,
                    "proxy_policy": row.policy,
                    "attribute": "EmpDepartment",
                    "metric": f"proxy_{metric}",
                    "class_label": np.nan,
                    "estimate": getattr(row, f"{metric}_mean"),
                    "gap": np.nan,
                    "ci_low": getattr(row, f"{metric}_ci_low"),
                    "ci_high": getattr(row, f"{metric}_ci_high"),
                    "confidence_level": 0.95,
                    "n_groups_total": np.nan,
                    "n_groups_included": np.nan,
                    "minimum_subgroup_support": np.nan,
                    "minimum_group_support_threshold": np.nan,
                    "minimum_metric_denominator": np.nan,
                    "minimum_metric_denominator_threshold": np.nan,
                    "bootstrap_samples_requested": np.nan,
                    "valid_bootstrap_samples": np.nan,
                    "valid_bootstrap_fraction": np.nan,
                    "estimate_status": "cross_validated_proxy_risk_estimate",
                    "interpretation_category": row.interpretation_category,
                    "limitations": row.limitations,
                }
            )
    return pd.concat([fairness, pd.DataFrame(proxy_rows)], ignore_index=True, sort=False)


def _write_interpretation(
    disparity: pd.DataFrame,
    proxy_summary: pd.DataFrame,
    path: Path,
    *,
    run_id: str,
    config_hash: str,
) -> None:
    primary = "no_salary_hike_no_attrition_no_department"
    stable = disparity[
        (disparity["policy"] == primary)
        & disparity["gap"].notna()
        & disparity["estimate_status"].str.startswith("support_sufficient")
    ].sort_values("gap", ascending=False)
    lines = [
        "# Canonical Fairness and Proxy Audit",
        "",
        f"Run ID: `{run_id}`  ",
        f"Config hash: `{config_hash}`",
        "",
        "Subgroup gaps use out-of-fold predictions, declared minimum support, metric-specific class denominators, and stratified bootstrap uncertainty. Sensitive audits and exploratory operational subgroup diagnostics are labelled separately.",
        "",
        "## Largest support-qualified primary-policy gaps",
        "",
    ]
    if stable.empty:
        lines.append("No primary-policy gap satisfied the configured support and bootstrap-stability requirements.")
    else:
        for row in stable.head(10).itertuples(index=False):
            class_text = "" if pd.isna(row.class_label) else f", class {int(row.class_label)}"
            lines.append(
                f"- {row.attribute}, {row.metric}{class_text}: gap={row.gap:.4f}, "
                f"95% CI [{row.ci_low:.4f}, {row.ci_high:.4f}], minimum subgroup n={row.minimum_subgroup_support}, "
                f"valid bootstrap replicates={row.valid_bootstrap_samples}/{row.bootstrap_samples_requested}."
            )
    lines.extend(["", "## Department reconstructability", ""])
    for row in proxy_summary.itertuples(index=False):
        lines.append(
            f"- `{row.policy}`: proxy macro-F1={row.macro_f1_mean:.4f} "
            f"(95% CI {row.macro_f1_ci_low:.4f}-{row.macro_f1_ci_high:.4f}); "
            f"proxy target absent from predictors={bool(row.proxy_target_absent_from_predictors)} "
            f"({row.proxy_target_removal_method})."
        )
    lines.extend(
        [
            "",
            "## Claim boundaries",
            "",
            "- Subgroup differences are descriptive audit evidence, not legal findings or proof of discrimination, fairness, or causality.",
            "- Groups or class-specific denominators below threshold remain visible in the group-support file but are excluded from gap estimates.",
            "- Department reconstructability is proxy-risk evidence; it is not proof that the performance model uses department causally or discriminatorily.",
            "- Removing sensitive, department, or job-role fields does not establish fairness or eliminate indirect proxy information.",
            "- The system is research-grade decision support only and must not make autonomous HR decisions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_dir: str | Path,
    run_id: str,
    config_hash: str | None = None,
    n_bootstrap_override: int | None = None,
) -> Dict[str, Path]:
    raw_config, settings = _settings(config_path)
    config_hash = config_hash or canonical_config_hash(raw_config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = load_validated_or_raw_data().reset_index(drop=True)
    labels = [int(value) for value in settings.get("target", {}).get("labels", [2, 3, 4])]
    fairness = settings.get("fairness", {})
    task_type = str(settings.get("target", {}).get("problem_type", ""))
    task_metrics = fairness.get("metrics_by_task", {}).get(task_type)
    if not isinstance(task_metrics, Mapping):
        raise FairnessProxyError(
            f"No fairness metric applicability schema is declared for task type {task_type!r}."
        )
    declared_overall = {str(value) for value in task_metrics.get("overall", [])}
    declared_class = {str(value) for value in task_metrics.get("class_specific", [])}
    if declared_overall != set(OVERALL_METRICS) or declared_class != set(CLASS_METRICS):
        raise FairnessProxyError(
            "Canonical fairness metric schema does not match the implemented task-aware metric contract."
        )
    governance = settings.get("governance_fields", {})
    sensitive = set(str(value) for value in governance.get("fairness_sensitive_fields", []))
    attributes = [str(value) for value in governance.get("fairness_audit_fields", [])]
    transforms = fairness.get("attribute_transforms", {})
    minimum_support = int(fairness.get("minimum_group_support", 30))
    minimum_class_denominator = int(fairness.get("minimum_class_metric_denominator", 10))
    bootstrap = settings.get("evaluation", {}).get("bootstrap", {})
    n_bootstrap = int(n_bootstrap_override or bootstrap.get("n_resamples", 5000))
    confidence_level = float(bootstrap.get("confidence_level", 0.95))
    bootstrap_seed = resolve_seed(settings, bootstrap.get("seed", "bootstrap"))
    stability = fairness.get("stability", {})

    predictions, fold_assignment = generate_common_fold_oof_predictions(
        data, settings, run_id=run_id, config_hash=config_hash
    )
    group_metrics = compute_group_metric_rows(
        predictions,
        data,
        labels=labels,
        attributes=attributes,
        transforms=transforms,
        sensitive_attributes=sensitive,
        minimum_group_support=minimum_support,
        minimum_class_denominator=minimum_class_denominator,
    )
    disparity = summarize_disparities_with_bootstrap(
        group_metrics,
        predictions,
        data,
        labels=labels,
        attributes=attributes,
        transforms=transforms,
        sensitive_attributes=sensitive,
        minimum_group_support=minimum_support,
        minimum_class_denominator=minimum_class_denominator,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
        minimum_valid_fraction=float(stability.get("minimum_valid_bootstrap_fraction", 0.8)),
        wide_interval_threshold=float(stability.get("wide_interval_threshold", 0.25)),
    )
    proxy_folds, proxy_summary, associations = run_proxy_policy_comparison(
        data, settings, run_id=run_id, config_hash=config_hash
    )
    manuscript_table = manuscript_fairness_proxy_table(disparity, proxy_summary)

    outputs = {
        "oof_predictions": output / "fairness_oof_predictions.csv",
        "fold_assignment": output / "common_fold_assignment.csv",
        "group_support_and_metrics": output / "fairness_group_support_and_metrics.csv",
        "disparity_uncertainty": output / "fairness_disparity_uncertainty.csv",
        "proxy_fold_metrics": output / "proxy_fold_metrics.csv",
        "proxy_policy_comparison": output / "proxy_policy_comparison.csv",
        "proxy_watchlist_associations": output / "proxy_watchlist_associations.csv",
        "manuscript_table": output / "manuscript_fairness_proxy_table.csv",
        "interpretation": output / "fairness_proxy_interpretation.md",
        "metadata": output / "metadata.json",
    }
    predictions.to_csv(outputs["oof_predictions"], index=False)
    fold_assignment.to_csv(outputs["fold_assignment"], index=False)
    group_metrics.to_csv(outputs["group_support_and_metrics"], index=False)
    disparity.to_csv(outputs["disparity_uncertainty"], index=False)
    proxy_folds.to_csv(outputs["proxy_fold_metrics"], index=False)
    proxy_summary.to_csv(outputs["proxy_policy_comparison"], index=False)
    associations.to_csv(outputs["proxy_watchlist_associations"], index=False)
    manuscript_table.to_csv(outputs["manuscript_table"], index=False)
    _write_interpretation(
        disparity, proxy_summary, outputs["interpretation"], run_id=run_id, config_hash=config_hash
    )
    _write_json(
        {
            "run_id": run_id,
            "config_hash": config_hash,
            "task_type": task_type,
            "policies": list(REQUIRED_POLICY_COMPARISONS),
            "common_fold_assignment": True,
            "audit_attributes": attributes,
            "sensitive_attributes": sorted(sensitive),
            "minimum_group_support": minimum_support,
            "minimum_class_metric_denominator": minimum_class_denominator,
            "bootstrap_samples": n_bootstrap,
            "bootstrap_stratification": fairness.get("bootstrap_stratify_by", ["fold", "y_true"]),
            "confidence_level": confidence_level,
            "claim_boundary": fairness.get("claim_boundary"),
            "proxy_claim_boundary": settings.get("proxy_analysis", {}).get("interpretation"),
            "outputs": {name: str(path) for name, path in outputs.items() if name != "metadata"},
        },
        outputs["metadata"],
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canonical support-aware fairness and proxy evidence.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash")
    parser.add_argument("--bootstrap-iterations", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        run(
            arguments.config,
            output_dir=arguments.output_dir,
            run_id=arguments.run_id,
            config_hash=arguments.config_hash,
            n_bootstrap_override=arguments.bootstrap_iterations,
        )
    )
