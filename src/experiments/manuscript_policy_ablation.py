"""Shared-fold, paired-OOF feature-policy sensitivity evidence.

The canonical policy stage is a matched sensitivity analysis.  It consumes the
persisted shared-fold and nested-benchmark contracts, reuses the exact primary
XGBoost OOF predictions, and applies each outer fold's primary-policy-selected
parameters to every non-primary policy.  Policies are deliberately *not*
independently tuned.

Population-style uncertainty is computed from one paired sample-level OOF
bootstrap.  Outer-fold summaries are retained only as descriptive variability;
folds are not treated as independent inferential units.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.core.io_utils import ensure_dir, write_json
from src.data.canonical_loader import load_canonical_dataset
from src.experiments.benchmark_artifact_contract import (
    BenchmarkArtifactContractError,
    XGBoostOOFArtifacts,
    read_xgboost_oof_artifacts,
    validate_xgboost_oof_replay,
)
from src.experiments.shared_folds import (
    SharedFoldContractError,
    validate_consumer_fold_assignments,
)
from src.governance.manuscript_contract import (
    canonical_config_hash,
    primary_excluded_features,
    repository_feature_policy_projection,
    validate_policy_consistency,
)
from src.models.canonical_models import (
    CanonicalModelError,
    aligned_predict_proba,
    build_model_pipeline,
)
from src.models.evaluate import classification_metrics
from src.models.oof_bootstrap import (
    BootstrapProtocol,
    BootstrapResult,
    ComparisonSpec,
    OOFBootstrapError,
    compute_paired_oof_bootstrap,
    metric_definition,
    validate_aligned_oof_predictions,
)
from src.utils.config_loader import load_config


DEFAULT_CONFIG = Path("configs/manuscript_final.yaml")
PRIMARY_TASK = "ordinal_multiclass_performance"
REQUIRED_OUTER_FOLDS = 10
REQUIRED_BOOTSTRAP_RESAMPLES = 5000
REQUIRED_POLICIES = (
    "full_feature_upper_bound",
    "no_salary_hike",
    "no_salary_hike_no_attrition",
    "no_salary_hike_no_attrition_no_department",
    "no_salary_hike_no_attrition_no_department_no_job_role",
)
SENSITIVE_RETAINING_AUDIT = "no_salary_hike_no_attrition_sensitive_retaining_audit"
CANONICAL_POLICY_ORDER = (
    "full_feature_upper_bound",
    "no_salary_hike",
    SENSITIVE_RETAINING_AUDIT,
    "no_salary_hike_no_attrition",
    "no_salary_hike_no_attrition_no_department",
    "no_salary_hike_no_attrition_no_department_no_job_role",
)
PRIMARY_METRIC_ORDER = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "quadratic_weighted_kappa",
    "ordinal_mae",
    "severe_error_rate",
    "nll_log_loss",
    "multiclass_brier",
    "ece_confidence",
)
IDENTITY_FIELDS = (
    "run_id",
    "config_hash",
    "scientific_input_hash",
    "fold_contract_hash",
    "xgboost_model_set_sha256",
    "dataset_sha256",
)


class PolicyAblationError(RuntimeError):
    """Raised when matched policy evidence violates the canonical contract."""


def _settings(config_path: str | Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    raw = load_config(config_path)
    settings = raw.get("manuscript_final", raw)
    if not isinstance(settings, dict):
        raise PolicyAblationError("Canonical config must contain a manuscript_final mapping.")
    try:
        validate_policy_consistency(
            raw,
            {
                "configs/feature_sets.yaml legacy projection": (
                    repository_feature_policy_projection()
                )
            },
        )
    except ValueError as exc:
        raise PolicyAblationError(f"Repository feature-policy projection is incompatible: {exc}") from exc
    return raw, settings


def _policy_definitions(settings: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    feature_policies = settings.get("feature_policies", {})
    definitions = feature_policies.get("definitions", {}) if isinstance(feature_policies, Mapping) else {}
    if not isinstance(definitions, dict):
        raise PolicyAblationError("feature_policies.definitions must be a mapping.")
    missing = [name for name in CANONICAL_POLICY_ORDER if name not in definitions]
    if missing:
        raise PolicyAblationError(f"Canonical config is missing required policies: {missing}")
    unexpected = sorted(set(definitions).difference(CANONICAL_POLICY_ORDER))
    if unexpected:
        raise PolicyAblationError(
            "Canonical matched policy sensitivity must define exactly the frozen six-policy "
            f"scope; unexpected policies={unexpected}."
        )
    return definitions


def _selected_policies(definitions: Mapping[str, Mapping[str, Any]]) -> list[str]:
    selected = list(CANONICAL_POLICY_ORDER)
    selected.extend(
        name
        for name, definition in definitions.items()
        if name not in selected and bool(definition.get("audit_only", False))
    )
    if len(selected) != len(set(selected)):
        raise PolicyAblationError("Selected policy names must be unique.")
    return selected


def exact_policy_frame(
    frame: pd.DataFrame,
    policy_name: str,
    definition: Mapping[str, Any],
    *,
    target_column: str,
    id_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply exactly one declared policy without implicit feature removal."""

    excluded = [str(value) for value in definition.get("excluded_features", [])]
    if len(excluded) != len(set(excluded)):
        raise PolicyAblationError(f"Policy {policy_name!r} repeats an excluded feature.")
    required_exclusions = {target_column, id_column}
    if not required_exclusions.issubset(excluded):
        raise PolicyAblationError(
            f"Policy {policy_name!r} must explicitly exclude target and identifier: "
            f"{sorted(required_exclusions)}"
        )
    unknown = sorted(set(excluded).difference(frame.columns))
    if unknown:
        raise PolicyAblationError(f"Policy {policy_name!r} excludes unknown columns: {unknown}")
    features = [column for column in frame.columns if column not in set(excluded)]
    if not features:
        raise PolicyAblationError(f"Policy {policy_name!r} leaves no model features.")
    if target_column in features or id_column in features:
        raise PolicyAblationError(f"Policy {policy_name!r} leaks target or identifier into features.")
    return frame.loc[:, features].copy(), excluded


def resolve_seed(settings: Mapping[str, Any], value_or_name: Any, *, default: int = 42) -> int:
    if isinstance(value_or_name, (int, np.integer)) and not isinstance(value_or_name, bool):
        return int(value_or_name)
    if isinstance(value_or_name, str):
        seeds = settings.get("seeds", {})
        if isinstance(seeds, Mapping) and value_or_name in seeds:
            return int(seeds[value_or_name])
        try:
            return int(value_or_name)
        except ValueError as exc:
            raise PolicyAblationError(f"Unknown seed reference: {value_or_name!r}") from exc
    return default


# Deprecated legacy-only compatibility helpers.  The canonical policy ``run``
# path below never calls either helper.  Pending calibration/counterfactual
# modules still import them and will be migrated in their own bounded units.
def _model_parameters(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the obsolete fixed XGBoost block for legacy consumers only."""

    model = settings.get("model", {})
    params: Dict[str, Any] = {}
    if isinstance(model, Mapping):
        source = model.get("hyperparameters", model.get("xgboost", {}))
        if isinstance(source, Mapping):
            params = dict(source)
    params.pop("random_state_seed", None)
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
    unexpected = sorted(set(params).difference(allowed))
    if unexpected:
        raise PolicyAblationError(f"Unsupported legacy XGBoost parameters: {unexpected}")
    return params


def _mean_ci(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float, float, float]:
    """Deprecated fold-t helper retained only until legacy calibration migration."""

    from scipy.stats import t as student_t

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(array.mean())
    if len(array) == 1:
        return mean, 0.0, mean, mean
    std = float(array.std(ddof=1))
    alpha = 1.0 - confidence
    critical = float(student_t.ppf(1.0 - alpha / 2.0, df=len(array) - 1))
    half_width = critical * std / math.sqrt(len(array))
    return mean, std, mean - half_width, mean + half_width


def _require_sha256(name: str, value: Any) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise PolicyAblationError(f"{name} must be a lowercase SHA-256 digest.")
    return text


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _json_mapping(value: Any, *, context: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PolicyAblationError(f"{context} is not valid JSON.") from exc
    if not isinstance(parsed, dict) or any(not isinstance(key, str) for key in parsed):
        raise PolicyAblationError(f"{context} must encode an object with string keys.")
    return parsed


def _parameter_contract_sha256(
    *,
    policy: str,
    outer_fold: int,
    feature_columns: Sequence[str],
    fixed_parameters: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
    selected_candidate_index: int,
    random_state: int,
    fold_contract_hash: str,
    source_model_set_sha256: str,
) -> str:
    payload = {
        "schema_version": 1,
        "policy": policy,
        "outer_fold": int(outer_fold),
        "feature_columns": list(feature_columns),
        "fixed_parameters": dict(fixed_parameters),
        "candidate_parameters": dict(candidate_parameters),
        "selected_candidate_index": int(selected_candidate_index),
        "random_state": int(random_state),
        "fold_contract_hash": fold_contract_hash,
        "parameter_source": "primary_policy_nested_selection_same_outer_fold",
        "source_model_set_sha256": source_model_set_sha256,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _configured_metrics(settings: Mapping[str, Any]) -> tuple[str, ...]:
    applicability = settings.get("evaluation", {}).get("metric_applicability", {})
    task = applicability.get(PRIMARY_TASK, {}) if isinstance(applicability, Mapping) else {}
    metrics = tuple(str(value) for value in task.get("applicable", ())) if isinstance(task, Mapping) else ()
    if metrics != PRIMARY_METRIC_ORDER:
        raise PolicyAblationError(
            "Primary policy metrics must match the predeclared ordinal metric order exactly; "
            f"observed={metrics}, expected={PRIMARY_METRIC_ORDER}."
        )
    if "weighted_f1" in metrics:
        raise PolicyAblationError("weighted_f1 is not a canonical policy-ablation metric.")
    for metric in metrics:
        metric_definition(metric)
    return metrics


def _validate_comparison_protocol(settings: Mapping[str, Any]) -> Mapping[str, Any]:
    feature_policies = settings.get("feature_policies", {})
    protocol = (
        feature_policies.get("comparison_protocol", {})
        if isinstance(feature_policies, Mapping)
        else {}
    )
    expected: Mapping[str, Any] = {
        "evaluation_type": "matched_oof_feature_access_sensitivity",
        "outer_folds_source": "shared_folds.outer_fold_assignments",
        "primary_oof_source": "model_benchmarks.exact_xgboost_oof_predictions",
        "primary_oof_replay_probability_atol": 1e-12,
        "non_primary_hyperparameters_source": (
            "model_benchmarks.xgboost_selected_candidate_by_outer_fold"
        ),
        "independent_policy_tuning": False,
        "preprocessing_fit_scope": "outer_training_partition_only",
        "uncertainty_source": "evaluation.bootstrap",
        "fold_summary_scope": "descriptive_variability_only_no_population_ci",
        "pairwise_inference": (
            "pointwise_paired_bootstrap_intervals_no_multiplicity_adjusted_rejection_claim"
        ),
        "full_feature_comparator_boundary": (
            "diagnostic_information_rich_comparator_not_guaranteed_optimized_upper_bound"
        ),
    }
    if not isinstance(protocol, Mapping):
        raise PolicyAblationError("feature_policies.comparison_protocol must be a mapping.")
    if dict(protocol) != dict(expected):
        raise PolicyAblationError(
            "feature_policies.comparison_protocol differs from the frozen matched-policy contract."
        )
    return protocol


def _bootstrap_protocol(settings: Mapping[str, Any]) -> BootstrapProtocol:
    configured = settings.get("evaluation", {}).get("bootstrap", {})
    if not isinstance(configured, Mapping):
        raise PolicyAblationError("evaluation.bootstrap must be a mapping.")
    protocol = BootstrapProtocol(
        n_resamples=int(configured.get("n_resamples", -1)),
        confidence_level=float(configured.get("confidence_level", float("nan"))),
        seed=resolve_seed(settings, configured.get("seed", "bootstrap")),
        strata_columns=tuple(str(value) for value in configured.get("stratify_by", ())),
        method=str(configured.get("method", "")),
        quantile_method=str(configured.get("quantile_method", "")),
    )
    if (
        protocol.n_resamples != REQUIRED_BOOTSTRAP_RESAMPLES
        or protocol.confidence_level != 0.95
        or protocol.strata_columns != ("outer_fold", "y_true")
        or protocol.method != "paired_stratified_percentile"
        or protocol.quantile_method != "linear"
    ):
        raise PolicyAblationError(
            "Policy uncertainty requires the frozen 5,000-draw paired stratified percentile protocol."
        )
    return protocol


def _validate_resample_binding(
    bootstrap_metadata: Mapping[str, Any],
    baseline_gate: Mapping[str, Any],
) -> str:
    policy_hash = _require_sha256(
        "policy resample_hash", bootstrap_metadata.get("resample_hash")
    )
    benchmark_hash = _require_sha256(
        "benchmark baseline-gate resample_hash", baseline_gate.get("resample_hash")
    )
    if policy_hash != benchmark_hash:
        raise PolicyAblationError(
            "Policy bootstrap resample plan differs from the benchmark paired-bootstrap plan."
        )
    return policy_hash


def _predeclared_policy_pairs(policies: Sequence[str]) -> set[tuple[str, str]]:
    pairs = {
        (CANONICAL_POLICY_ORDER[index], CANONICAL_POLICY_ORDER[index + 1])
        for index in range(len(CANONICAL_POLICY_ORDER) - 1)
    }
    if not set(CANONICAL_POLICY_ORDER).issubset(policies):
        raise PolicyAblationError("The canonical matched-contrast policy sequence is incomplete.")
    return pairs


def _comparison_specs(policies: Sequence[str]) -> tuple[tuple[ComparisonSpec, ...], set[tuple[str, str]]]:
    predeclared = _predeclared_policy_pairs(policies)
    specifications = tuple(
        ComparisonSpec(
            comparison_id=f"{policy_a}__minus__{policy_b}",
            system_a=policy_a,
            system_b=policy_b,
            primary_gate=False,
        )
        for policy_a, policy_b in itertools.combinations(policies, 2)
    )
    return specifications, predeclared


def _feature_contract_frame(
    *,
    data: pd.DataFrame,
    policies: Sequence[str],
    definitions: Mapping[str, Mapping[str, Any]],
    target_column: str,
    id_column: str,
    identity: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    features_by_policy: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for policy_order, policy in enumerate(policies):
        definition = definitions[policy]
        features, excluded = exact_policy_frame(
            data,
            policy,
            definition,
            target_column=target_column,
            id_column=id_column,
        )
        features_by_policy[policy] = features
        rows.append(
            {
                **identity,
                "policy": policy,
                "policy_order": policy_order,
                "role": str(definition.get("role", "unspecified")),
                "audit_only": bool(definition.get("audit_only", False)),
                "n_features": int(features.shape[1]),
                "excluded_features_json": _canonical_json(excluded),
                "feature_columns_json": _canonical_json(features.columns.tolist()),
            }
        )
    return features_by_policy, pd.DataFrame(rows)


def _selected_parameters_by_fold(
    bundle: XGBoostOOFArtifacts,
    *,
    policies: Sequence[str],
    features_by_policy: Mapping[str, pd.DataFrame],
    random_state: int,
    identity: Mapping[str, Any],
) -> tuple[dict[int, tuple[dict[str, Any], dict[str, Any], int]], pd.DataFrame]:
    selected = bundle.selected_hyperparameters.copy()
    if len(selected) != REQUIRED_OUTER_FOLDS or set(selected["outer_fold"].astype(int)) != set(
        range(1, REQUIRED_OUTER_FOLDS + 1)
    ):
        raise PolicyAblationError("Selected XGBoost parameters must cover outer folds 1..10 exactly.")
    parameters: dict[int, tuple[dict[str, Any], dict[str, Any], int]] = {}
    rows: list[dict[str, Any]] = []
    for selected_row in selected.sort_values("outer_fold").itertuples(index=False):
        outer_fold = int(selected_row.outer_fold)
        fixed = _json_mapping(
            selected_row.fixed_parameters_json,
            context=f"outer fold {outer_fold} fixed_parameters_json",
        )
        candidate = _json_mapping(
            selected_row.selected_candidate_parameters_json,
            context=f"outer fold {outer_fold} selected_candidate_parameters_json",
        )
        overlap = sorted(set(fixed).intersection(candidate))
        if overlap:
            raise PolicyAblationError(
                f"Outer fold {outer_fold} fixed/candidate parameters overlap: {overlap}."
            )
        candidate_index = int(selected_row.selected_candidate_index)
        fold_model = bundle.fold_models.get(outer_fold)
        if fold_model is None or fold_model.selected_candidate_index != candidate_index:
            raise PolicyAblationError(
                f"Outer fold {outer_fold} selected candidate drifts from benchmark model evidence."
            )
        parameters[outer_fold] = (fixed, candidate, candidate_index)
        for policy in policies:
            contract_hash = _parameter_contract_sha256(
                policy=policy,
                outer_fold=outer_fold,
                feature_columns=features_by_policy[policy].columns,
                fixed_parameters=fixed,
                candidate_parameters=candidate,
                selected_candidate_index=candidate_index,
                random_state=random_state,
                fold_contract_hash=bundle.identity.fold_contract_hash,
                source_model_set_sha256=bundle.model_set_sha256,
            )
            rows.append(
                {
                    **identity,
                    "policy": policy,
                    "outer_fold": outer_fold,
                    "model": "xgboost",
                    "selected_candidate_index": candidate_index,
                    "fixed_parameters_json": _canonical_json(fixed),
                    "selected_candidate_parameters_json": _canonical_json(candidate),
                    "parameter_source": "primary_policy_nested_selection_same_outer_fold",
                    "outer_test_used_for_parameter_selection": False,
                    "policy_independently_tuned": False,
                    "planned_fit_threadpool_limit": 1,
                    "policy_model_contract_sha256": contract_hash,
                    "source_primary_model_sha256": fold_model.sha256,
                    "source_primary_model_persisted": True,
                }
            )
    return parameters, pd.DataFrame(rows)


def _fold_metric_rows(
    oof: pd.DataFrame,
    *,
    metrics: Sequence[str],
    labels: Sequence[int],
    task_type: str,
    feature_contract: pd.DataFrame,
    parameter_contract: pd.DataFrame,
) -> pd.DataFrame:
    feature_by_policy = feature_contract.set_index("policy")
    parameter_by_key = parameter_contract.set_index(["policy", "outer_fold"])
    rows: list[dict[str, Any]] = []
    for policy in feature_contract.sort_values("policy_order")["policy"]:
        policy_rows = oof[oof["system_id"].astype(str) == str(policy)]
        for outer_fold in range(1, REQUIRED_OUTER_FOLDS + 1):
            group = policy_rows[policy_rows["outer_fold"].astype(int) == outer_fold].sort_values(
                "sample_index"
            )
            if group.empty:
                raise PolicyAblationError(f"Policy {policy!r} outer fold {outer_fold} has no OOF rows.")
            probability = group[[f"prob_class_{label}" for label in labels]].to_numpy(dtype=float)
            calculated = classification_metrics(
                group["y_true"].to_numpy(dtype=int),
                group["y_pred"].to_numpy(dtype=int),
                probability,
                list(labels),
                task_type=task_type,
            )
            feature_row = feature_by_policy.loc[policy]
            parameter_row = parameter_by_key.loc[(policy, outer_fold)]
            row = {
                **{field: group[field].iloc[0] for field in IDENTITY_FIELDS},
                "policy": policy,
                "role": feature_row["role"],
                "audit_only": bool(feature_row["audit_only"]),
                "model": "xgboost",
                "outer_fold": outer_fold,
                "n_train": int(len(oof[oof["system_id"].astype(str) == str(policy)]) - len(group)),
                "n_test": int(len(group)),
                "n_features": int(feature_row["n_features"]),
                "excluded_features_json": feature_row["excluded_features_json"],
                "selected_candidate_index": int(parameter_row["selected_candidate_index"]),
                "policy_model_contract_sha256": parameter_row["policy_model_contract_sha256"],
                "model_fit_mode": group["model_fit_mode"].iloc[0],
                "fold_variability_inference": "descriptive_only_not_population_ci",
            }
            row.update({metric: float(calculated[metric]) for metric in metrics})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_policies(
    fold_metrics: pd.DataFrame,
    metric_intervals: pd.DataFrame,
    feature_contract: pd.DataFrame,
) -> pd.DataFrame:
    """Combine OOF bootstrap intervals with non-inferential fold variability."""

    rows: list[dict[str, Any]] = []
    ordered_features = feature_contract.sort_values("policy_order")
    for feature_row in ordered_features.itertuples(index=False):
        policy = str(feature_row.policy)
        folds = fold_metrics[fold_metrics["policy"].astype(str) == policy]
        intervals = metric_intervals[metric_intervals["system_id"].astype(str) == policy]
        if len(folds) != REQUIRED_OUTER_FOLDS or folds["outer_fold"].nunique() != REQUIRED_OUTER_FOLDS:
            raise PolicyAblationError(f"Policy {policy!r} must have ten descriptive fold rows.")
        row: dict[str, Any] = {
            **{field: getattr(feature_row, field) for field in IDENTITY_FIELDS},
            "policy": policy,
            "policy_order": int(feature_row.policy_order),
            "role": str(feature_row.role),
            "audit_only": bool(feature_row.audit_only),
            "n_folds": REQUIRED_OUTER_FOLDS,
            "n_samples": int(folds["n_test"].sum()),
            "n_features": int(feature_row.n_features),
            "excluded_features_json": str(feature_row.excluded_features_json),
            "feature_columns_json": str(feature_row.feature_columns_json),
            "point_estimate_unit": "all_exactly_once_oof_samples",
            "confidence_interval_method": "paired_stratified_sample_level_percentile_bootstrap",
            "fold_variability_status": "descriptive_only_not_population_ci",
        }
        for metric in PRIMARY_METRIC_ORDER:
            interval = intervals[intervals["metric"].astype(str) == metric]
            if len(interval) != 1:
                raise PolicyAblationError(
                    f"Policy {policy!r} metric {metric!r} requires one bootstrap interval row."
                )
            interval_row = interval.iloc[0]
            values = pd.to_numeric(folds[metric], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise PolicyAblationError(f"Policy {policy!r} fold metric {metric!r} is non-finite.")
            point = float(interval_row["point_estimate"])
            row[f"{metric}_oof"] = point
            row[f"{metric}_ci_low"] = float(interval_row["ci_low"])
            row[f"{metric}_ci_high"] = float(interval_row["ci_high"])
            row[f"{metric}_bootstrap_std"] = float(interval_row["bootstrap_std"])
            row[f"{metric}_fold_mean"] = float(values.mean())
            row[f"{metric}_fold_std"] = float(values.std(ddof=1))
            row[f"{metric}_fold_min"] = float(values.min())
            row[f"{metric}_fold_max"] = float(values.max())
        unique_resamples = set(intervals["resample_hash"].astype(str))
        if len(unique_resamples) != 1:
            raise PolicyAblationError(f"Policy {policy!r} bootstrap intervals have identity drift.")
        row["n_resamples"] = int(intervals["n_resamples"].iloc[0])
        row["n_valid_bootstrap"] = int(intervals["n_valid"].min())
        row["resample_hash"] = next(iter(unique_resamples))
        rows.append(row)
    return pd.DataFrame(rows)


def annotate_pairwise_differences(
    paired: pd.DataFrame,
    *,
    policies: Sequence[str],
    predeclared_pairs: set[tuple[str, str]],
) -> pd.DataFrame:
    order = {policy: index for index, policy in enumerate(policies)}
    result = paired.copy()
    result["adjacent_policy_step"] = [
        order[str(row.system_b)] == order[str(row.system_a)] + 1
        for row in result.itertuples(index=False)
    ]
    result["predeclared_comparison"] = [
        (str(row.system_a), str(row.system_b)) in predeclared_pairs
        or (str(row.system_b), str(row.system_a)) in predeclared_pairs
        for row in result.itertuples(index=False)
    ]
    result["multiplicity_adjustment"] = "none"
    result["inference_scope"] = (
        "pointwise_paired_oof_bootstrap_no_familywise_or_false_discovery_claim"
    )
    if result["primary_gate_comparison"].astype(bool).any() or result["gate_eligible"].astype(bool).any():
        raise PolicyAblationError("Policy comparisons must not participate in the model superiority gate.")
    if result["gate_triggered"].astype(bool).any():
        raise PolicyAblationError("A policy comparison was incorrectly marked as a model gate.")
    return result


def leakage_sensitivity_indices(
    pairwise: pd.DataFrame,
    *,
    policies: Sequence[str],
    metrics: Sequence[str],
    identity: Mapping[str, Any],
    resample_hash: str,
    n_samples: int,
    n_resamples: int,
) -> pd.DataFrame:
    """Express full-policy degradation in metric units and finite-domain units."""

    reference = CANONICAL_POLICY_ORDER[0]
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for metric in metrics:
            definition = metric_definition(metric)
            if policy == reference:
                degradation = low = high = 0.0
                n_valid = int(n_resamples)
            else:
                match = pairwise[
                    (pairwise["system_a"].astype(str) == reference)
                    & (pairwise["system_b"].astype(str) == policy)
                    & (pairwise["metric"].astype(str) == metric)
                ]
                if len(match) != 1:
                    raise PolicyAblationError(
                        f"Missing full-policy paired comparison for {policy!r}/{metric!r}."
                    )
                comparison = match.iloc[0]
                degradation = float(comparison["improvement_oriented_difference"])
                low = float(comparison["improvement_ci_low"])
                high = float(comparison["improvement_ci_high"])
                try:
                    n_valid = int(comparison["n_valid"])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise PolicyAblationError(
                        f"Paired comparison for {policy!r}/{metric!r} lacks n_valid."
                    ) from exc
                if n_valid != int(n_resamples):
                    raise PolicyAblationError(
                        f"Paired comparison for {policy!r}/{metric!r} has "
                        f"n_valid={n_valid}, expected={n_resamples}."
                    )
            finite_domain = math.isfinite(definition.lower_bound) and math.isfinite(
                definition.upper_bound
            )
            span = float(definition.upper_bound - definition.lower_bound) if finite_domain else math.nan
            normalizable = finite_domain and span > 0.0
            rows.append(
                {
                    **identity,
                    "reference_policy": reference,
                    "policy": policy,
                    "metric": metric,
                    "better_direction": definition.better_direction,
                    "definition": (
                        "positive values indicate degradation relative to the diagnostic "
                        "information-rich comparator, which is not guaranteed optimized; "
                        "negative values indicate improvement"
                    ),
                    "absolute_metric_degradation": degradation,
                    "absolute_degradation_ci_low": low,
                    "absolute_degradation_ci_high": high,
                    "domain_low": definition.lower_bound,
                    "domain_high": definition.upper_bound,
                    "domain_span": span,
                    "domain_normalized_degradation": degradation / span if normalizable else math.nan,
                    "domain_normalized_ci_low": low / span if normalizable else math.nan,
                    "domain_normalized_ci_high": high / span if normalizable else math.nan,
                    "normalization_status": (
                        "finite_metric_domain_span" if normalizable else "not_applicable_unbounded_domain"
                    ),
                    "n_samples": int(n_samples),
                    "n_resamples": int(n_resamples),
                    "n_valid_bootstrap": n_valid,
                    "resample_hash": resample_hash,
                    "uncertainty_method": "paired_stratified_sample_level_percentile_bootstrap",
                }
            )
    return pd.DataFrame(rows)


def manuscript_policy_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *IDENTITY_FIELDS,
        "policy",
        "role",
        "audit_only",
        "n_samples",
        "n_folds",
        "n_features",
        "macro_f1_oof",
        "macro_f1_ci_low",
        "macro_f1_ci_high",
        "quadratic_weighted_kappa_oof",
        "quadratic_weighted_kappa_ci_low",
        "quadratic_weighted_kappa_ci_high",
        "ordinal_mae_oof",
        "ordinal_mae_ci_low",
        "ordinal_mae_ci_high",
        "severe_error_rate_oof",
        "severe_error_rate_ci_low",
        "severe_error_rate_ci_high",
        "resample_hash",
        "excluded_features_json",
        "point_estimate_unit",
        "confidence_interval_method",
    ]
    missing = sorted(set(columns).difference(summary.columns))
    if missing:
        raise PolicyAblationError(f"Manuscript policy summary is missing columns: {missing}.")
    return summary.loc[:, columns].copy()


def write_interpretation(summary: pd.DataFrame, path: Path) -> None:
    full = summary[summary["policy"] == "full_feature_upper_bound"]
    if len(full) != 1:
        raise PolicyAblationError("Interpretation requires one diagnostic full-feature row.")
    full_row = full.iloc[0]
    lines = [
        "# Matched Feature-Policy Sensitivity Interpretation",
        "",
        f"Run ID: `{full_row['run_id']}`  ",
        f"Config hash: `{full_row['config_hash']}`  ",
        f"Scientific input hash: `{full_row['scientific_input_hash']}`  ",
        f"Shared-fold contract: `{full_row['fold_contract_hash']}`",
        "",
        "Each non-primary policy uses the primary policy's already-selected XGBoost candidate for the same outer fold. Policies were not independently tuned. The canonical primary row reuses the exact benchmark OOF predictions rather than refitting another model.",
        "",
        "Confidence intervals are paired, sample-level, outer-fold-and-class-stratified percentile bootstrap intervals with 5,000 draws. Fold mean/SD/min/max fields are descriptive only and are not population confidence intervals.",
        "",
        "## Policy results",
        "",
    ]
    for row in summary.sort_values("policy_order").itertuples(index=False):
        lines.append(
            f"- `{row.policy}` ({row.role}; audit_only={bool(row.audit_only)}): "
            f"OOF macro-F1 {row.macro_f1_oof:.4f} "
            f"(95% CI {row.macro_f1_ci_low:.4f}-{row.macro_f1_ci_high:.4f}); "
            f"QWK {row.quadratic_weighted_kappa_oof:.4f}; "
            f"ordinal MAE {row.ordinal_mae_oof:.4f}; features {row.n_features}."
        )
    lines.extend(
        [
            "",
            "## Claim boundaries",
            "",
            "- `full_feature_upper_bound` is a diagnostic information-rich comparator, is not guaranteed optimized, and is never deployable evidence.",
            "- Pairwise intervals are pointwise; no multiplicity-adjusted familywise or false-discovery claim is made.",
            "- Performance changes across compound policies cannot be attributed to a single removed field unless the predeclared contrast isolates that field.",
            "- Sensitive-retaining policies are audit-only. Removing sensitive or organisational fields does not prove fairness or eliminate proxy risk.",
            "- The results describe model sensitivity, not causal feature effects or autonomous HR decision suitability.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_tradeoff_figure(
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    identity: Mapping[str, str],
) -> Dict[str, Path]:
    source_columns = [
        *IDENTITY_FIELDS,
        "policy",
        "policy_order",
        "role",
        "audit_only",
        "macro_f1_oof",
        "macro_f1_ci_low",
        "macro_f1_ci_high",
        "quadratic_weighted_kappa_oof",
        "ordinal_mae_oof",
        "resample_hash",
    ]
    source = summary.loc[:, source_columns].sort_values("policy_order").reset_index(drop=True)
    source_path = output_dir / "figure_leakage_policy_tradeoff_source.csv"
    source.to_csv(source_path, index=False)

    labels = [value.replace("_", " ") for value in source["policy"]]
    positions = np.arange(len(source))
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), constrained_layout=True)
    for position, row in source.iterrows():
        policy = str(row["policy"])
        if policy == "full_feature_upper_bound":
            color, marker, size, label = "#B33A3A", "D", 8, "information-rich comparator"
        elif str(row["role"]) == "canonical_primary":
            color, marker, size, label = "#176B87", "*", 12, "canonical primary"
        elif bool(row["audit_only"]):
            color, marker, size, label = "#D17A22", "x", 9, "audit-only"
        else:
            color, marker, size, label = "#3A7D44", "o", 8, "governed sensitivity"
        axes[0].errorbar(
            float(row["macro_f1_oof"]),
            position,
            xerr=np.asarray(
                [[float(row["macro_f1_oof"] - row["macro_f1_ci_low"])],
                 [float(row["macro_f1_ci_high"] - row["macro_f1_oof"])]]
            ),
            fmt=marker,
            markersize=size,
            capsize=4,
            color=color,
            label=label,
        )
        axes[1].scatter(
            float(row["ordinal_mae_oof"]),
            float(row["quadratic_weighted_kappa_oof"]),
            color=color,
            marker=marker,
            s=size * 9,
        )
        axes[1].annotate(
            labels[position],
            (float(row["ordinal_mae_oof"]), float(row["quadratic_weighted_kappa_oof"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.5,
        )
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Macro-F1 (OOF point estimate, paired-bootstrap 95% CI)")
    axes[0].set_title("Leakage-aware feature-policy sensitivity")
    axes[0].grid(axis="x", alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    axes[0].legend(unique.values(), unique.keys(), loc="best", fontsize=8)
    axes[1].set_xlabel("Ordinal MAE (lower is better)")
    axes[1].set_ylabel("Quadratic weighted kappa (higher is better)")
    axes[1].set_title("OOF ordinal error-agreement trade-off")
    axes[1].grid(alpha=0.25)
    fig.suptitle(
        "Matched XGBoost policies (red diagnostic and orange audit-only variants are non-primary)",
        fontsize=13,
    )

    png = output_dir / "figure_leakage_policy_tradeoff.png"
    svg = output_dir / "figure_leakage_policy_tradeoff.svg"
    description = "; ".join(f"{field}={identity[field]}" for field in IDENTITY_FIELDS)
    fig.savefig(png, dpi=300, metadata={"Description": description})
    fig.savefig(
        svg,
        format="svg",
        metadata={"Title": "Matched feature-policy sensitivity", "Description": description},
    )
    plt.close(fig)
    return {"png": png, "svg": svg, "source": source_path}


def _primary_oof_rows(
    bundle: XGBoostOOFArtifacts,
    *,
    primary_policy: str,
    feature_row: pd.Series,
    parameter_contract: pd.DataFrame,
    identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    parameter_by_fold = parameter_contract[
        parameter_contract["policy"].astype(str) == primary_policy
    ].set_index("outer_fold")
    rows: list[dict[str, Any]] = []
    for source in bundle.oof_predictions.sort_values("sample_index").itertuples(index=False):
        outer_fold = int(source.outer_fold)
        parameter_row = parameter_by_fold.loc[outer_fold]
        row = {
            **identity,
            "system_id": primary_policy,
            "policy": primary_policy,
            "role": feature_row["role"],
            "audit_only": bool(feature_row["audit_only"]),
            "model": "xgboost",
            "sample_index": int(source.sample_index),
            "outer_fold": outer_fold,
            "y_true": int(source.y_true),
            "y_pred": int(source.y_pred),
            "selected_candidate_index": int(source.selected_candidate_index),
            "policy_model_contract_sha256": parameter_row["policy_model_contract_sha256"],
            "source_primary_model_sha256": parameter_row["source_primary_model_sha256"],
            "model_fit_mode": "exact_benchmark_oof_reuse_no_refit",
        }
        for label in bundle.labels:
            row[f"prob_class_{label}"] = float(getattr(source, f"prob_class_{label}"))
        rows.append(row)
    return rows


def _validate_nonprimary_fitted_pipeline(
    pipeline: Any,
    *,
    feature_columns: Sequence[str],
    fixed_parameters: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
    policy: str,
    outer_fold: int,
) -> None:
    named_steps = getattr(pipeline, "named_steps", None)
    if not isinstance(named_steps, Mapping) or set(named_steps) != {"preprocessor", "model"}:
        raise PolicyAblationError(
            f"Policy {policy!r} outer-fold {outer_fold} did not produce the canonical pipeline steps."
        )
    expected_features = tuple(str(value) for value in feature_columns)
    pipeline_features = tuple(str(value) for value in getattr(pipeline, "feature_names_in_", ()))
    preprocessor_features = tuple(
        str(value) for value in getattr(named_steps["preprocessor"], "feature_names_in_", ())
    )
    if pipeline_features != expected_features or preprocessor_features != expected_features:
        raise PolicyAblationError(
            f"Policy {policy!r} outer-fold {outer_fold} preprocessing feature lineage drifted."
        )
    estimator = named_steps["model"]
    try:
        observed_parameters = estimator.get_params(deep=False)
    except (AttributeError, TypeError, ValueError) as exc:
        raise PolicyAblationError(
            f"Policy {policy!r} outer-fold {outer_fold} estimator parameters cannot be verified."
        ) from exc
    for parameter, expected in {**dict(fixed_parameters), **dict(candidate_parameters)}.items():
        if parameter not in observed_parameters or observed_parameters[parameter] != expected:
            raise PolicyAblationError(
                f"Policy {policy!r} outer-fold {outer_fold} fitted parameter {parameter!r} "
                "differs from the primary-selected schedule."
            )


def _fit_non_primary_oof_rows(
    *,
    policies: Sequence[str],
    primary_policy: str,
    features_by_policy: Mapping[str, pd.DataFrame],
    target: pd.Series,
    bundle: XGBoostOOFArtifacts,
    parameters_by_fold: Mapping[int, tuple[dict[str, Any], dict[str, Any], int]],
    feature_contract: pd.DataFrame,
    parameter_contract: pd.DataFrame,
    identity: Mapping[str, Any],
    random_state: int,
    target_column: str,
    id_column: str,
) -> list[dict[str, Any]]:
    feature_by_policy = feature_contract.set_index("policy")
    parameter_by_key = parameter_contract.set_index(["policy", "outer_fold"])
    outer = bundle.folds.outer_assignments.copy()
    rows: list[dict[str, Any]] = []
    for policy in policies:
        if policy == primary_policy:
            continue
        features = features_by_policy[policy]
        feature_row = feature_by_policy.loc[policy]
        for outer_fold in range(1, REQUIRED_OUTER_FOLDS + 1):
            test_ids = outer.loc[
                outer["outer_fold"].astype(int) == outer_fold, "sample_index"
            ].astype(int).tolist()
            train_ids = outer.loc[
                outer["outer_fold"].astype(int) != outer_fold, "sample_index"
            ].astype(int).tolist()
            if not train_ids or not test_ids or set(train_ids).intersection(test_ids):
                raise PolicyAblationError(f"Invalid outer partition for fold {outer_fold}.")
            fixed, candidate, candidate_index = parameters_by_fold[outer_fold]
            excluded_features = tuple(
                json.loads(str(feature_row["excluded_features_json"]))
            )
            if target_column not in excluded_features or id_column not in excluded_features:
                raise PolicyAblationError(
                    f"Policy {policy!r} fit receipt lacks target/identifier exclusions."
                )
            try:
                pipeline = build_model_pipeline(
                    "xgboost",
                    features.loc[train_ids],
                    fixed_parameters=fixed,
                    candidate_parameters=candidate,
                    random_state=random_state,
                    forbidden_features=excluded_features,
                )
                with threadpool_limits(limits=1):
                    pipeline.fit(features.loc[train_ids], target.loc[train_ids])
                _validate_nonprimary_fitted_pipeline(
                    pipeline,
                    feature_columns=features.columns,
                    fixed_parameters=fixed,
                    candidate_parameters=candidate,
                    policy=policy,
                    outer_fold=outer_fold,
                )
                prediction = np.asarray(pipeline.predict(features.loc[test_ids]), dtype=int)
                probability = aligned_predict_proba(
                    pipeline,
                    features.loc[test_ids],
                    labels=bundle.labels,
                )
            except (CanonicalModelError, TypeError, ValueError) as exc:
                raise PolicyAblationError(
                    f"Policy {policy!r} outer-fold {outer_fold} model evaluation failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if prediction.shape != (len(test_ids),) or probability.shape != (
                len(test_ids),
                len(bundle.labels),
            ):
                raise PolicyAblationError(
                    f"Policy {policy!r} outer-fold {outer_fold} returned invalid prediction shapes."
                )
            parameter_row = parameter_by_key.loc[(policy, outer_fold)]
            if int(parameter_row["selected_candidate_index"]) != candidate_index:
                raise PolicyAblationError(
                    f"Policy {policy!r} outer-fold {outer_fold} parameter identity drifted."
                )
            for position, sample_index in enumerate(test_ids):
                row = {
                    **identity,
                    "system_id": policy,
                    "policy": policy,
                    "role": feature_row["role"],
                    "audit_only": bool(feature_row["audit_only"]),
                    "model": "xgboost",
                    "sample_index": sample_index,
                    "outer_fold": outer_fold,
                    "y_true": int(target.loc[sample_index]),
                    "y_pred": int(prediction[position]),
                    "selected_candidate_index": candidate_index,
                    "policy_model_contract_sha256": parameter_row[
                        "policy_model_contract_sha256"
                    ],
                    "source_primary_model_sha256": parameter_row["source_primary_model_sha256"],
                    "model_fit_mode": "outer_train_refit_with_primary_selected_parameters",
                }
                row.update(
                    {
                        f"prob_class_{label}": float(probability[position, column])
                        for column, label in enumerate(bundle.labels)
                    }
                )
                rows.append(row)
    return rows


def _fit_receipt_frame(
    fold_metrics: pd.DataFrame,
    parameter_contract: pd.DataFrame,
) -> pd.DataFrame:
    if parameter_contract.duplicated(["policy", "outer_fold"]).any():
        raise PolicyAblationError("Policy hyperparameter schedule repeats a policy/fold key.")
    parameters = parameter_contract.set_index(["policy", "outer_fold"])
    rows: list[dict[str, Any]] = []
    for metric_row in fold_metrics.itertuples(index=False):
        key = (str(metric_row.policy), int(metric_row.outer_fold))
        parameter = parameters.loc[key]
        primary_reuse = str(metric_row.model_fit_mode) == "exact_benchmark_oof_reuse_no_refit"
        rows.append(
            {
                **{field: getattr(metric_row, field) for field in IDENTITY_FIELDS},
                "policy": key[0],
                "outer_fold": key[1],
                "execution_mode": str(metric_row.model_fit_mode),
                "execution_status": "complete",
                "stage_fit_performed": not primary_reuse,
                "upstream_primary_fit_complete": True,
                "primary_benchmark_oof_reused": primary_reuse,
                "n_train": int(metric_row.n_train),
                "n_test": int(metric_row.n_test),
                "selected_candidate_index": int(parameter["selected_candidate_index"]),
                "parameter_source": str(parameter["parameter_source"]),
                "parameter_source_sha256": str(parameter["policy_model_contract_sha256"]),
                "source_primary_model_sha256": str(parameter["source_primary_model_sha256"]),
                "preprocessing_fit_scope": (
                    "upstream_outer_training_partition_only"
                    if primary_reuse
                    else "current_outer_training_partition_only"
                ),
                "threadpool_limit": 1,
                "policy_independently_tuned": False,
            }
        )
    return pd.DataFrame(rows)


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    shared_folds_dir: str | Path,
    model_benchmarks_dir: str | Path,
    output_dir: str | Path,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
) -> Dict[str, Path]:
    """Generate matched policy sensitivity evidence from current-run inputs."""

    raw_config, settings = _settings(config_path)
    observed_config_hash = canonical_config_hash(raw_config)
    if str(config_hash) != observed_config_hash:
        raise PolicyAblationError(
            "Supplied config_hash does not match the canonical manuscript configuration."
        )
    _require_sha256("config_hash", config_hash)
    _require_sha256("scientific_input_hash", scientific_input_hash)
    if not isinstance(run_id, str) or not run_id.strip():
        raise PolicyAblationError("run_id must be a non-blank string.")
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise PolicyAblationError(
            f"Policy output directory must be absent or an empty builder-owned directory: {output}"
        )

    definitions = _policy_definitions(settings)
    policies = _selected_policies(definitions)
    comparison_protocol = _validate_comparison_protocol(settings)
    feature_policy_config = settings.get("feature_policies", {})
    primary_policy = str(feature_policy_config.get("primary_policy", ""))
    if primary_policy != "no_salary_hike_no_attrition_no_department":
        raise PolicyAblationError("The canonical primary feature policy changed unexpectedly.")
    if primary_policy not in policies:
        raise PolicyAblationError("The primary feature policy is absent from policy sensitivity.")

    target_config = settings.get("target", {})
    target_column = str(target_config.get("column", "PerformanceRating"))
    labels = tuple(int(value) for value in target_config.get("labels", ()))
    task_type = str(target_config.get("problem_type", ""))
    if labels != (2, 3, 4) or task_type != PRIMARY_TASK:
        raise PolicyAblationError("Policy sensitivity requires the canonical 2/3/4 ordinal task.")
    identifier_fields = settings.get("governance_fields", {}).get("identifier_fields", ())
    if not isinstance(identifier_fields, list) or len(identifier_fields) != 1:
        raise PolicyAblationError("Policy sensitivity requires exactly one declared identifier field.")
    id_column = str(identifier_fields[0])
    metrics = _configured_metrics(settings)
    bootstrap_protocol = _bootstrap_protocol(settings)
    random_state = resolve_seed(settings, "model")

    canonical = load_canonical_dataset(config_path, "inx_primary")
    data = canonical.frame
    if not data.index.is_unique:
        raise PolicyAblationError("Canonical policy data requires unique sample indices.")
    try:
        integer_index = pd.Index([int(value) for value in data.index])
    except (TypeError, ValueError, OverflowError) as exc:
        raise PolicyAblationError("Canonical sample indices must be integers.") from exc
    if not integer_index.equals(pd.Index(data.index)):
        data = data.copy()
        data.index = integer_index
    if target_column not in data or id_column not in data:
        raise PolicyAblationError("Canonical target or identifier column is missing.")
    target = data[target_column].astype(int)
    if set(target.unique()) != set(labels):
        raise PolicyAblationError(
            f"Observed labels {sorted(target.unique())} do not match canonical labels {list(labels)}."
        )
    dataset_sha256 = _require_sha256(
        "canonical dataset actual_sha256", canonical.receipt.get("actual_sha256")
    )

    primary_definition = definitions[primary_policy]
    primary_features, primary_excluded = exact_policy_frame(
        data,
        primary_policy,
        primary_definition,
        target_column=target_column,
        id_column=id_column,
    )
    resolved_primary_exclusions = tuple(primary_excluded_features(raw_config))
    if tuple(primary_excluded) != resolved_primary_exclusions:
        raise PolicyAblationError(
            "Primary policy exclusions differ from the resolved canonical feature-policy contract."
        )

    try:
        bundle = read_xgboost_oof_artifacts(
            shared_folds_dir,
            model_benchmarks_dir,
            expected_run_id=run_id,
            expected_config_hash=config_hash,
            expected_scientific_input_hash=scientific_input_hash,
            expected_feature_columns=primary_features.columns.tolist(),
            expected_labels=labels,
        )
    except BenchmarkArtifactContractError as exc:
        raise PolicyAblationError(f"Benchmark evidence is incompatible: {exc}") from exc
    try:
        validate_xgboost_oof_replay(
            bundle,
            primary_features,
            target,
            labels=labels,
            probability_atol=float(comparison_protocol["primary_oof_replay_probability_atol"]),
        )
    except BenchmarkArtifactContractError as exc:
        raise PolicyAblationError(f"Exact primary XGBoost OOF replay failed: {exc}") from exc
    if bundle.identity.run_id != run_id or bundle.identity.config_hash != config_hash:
        raise PolicyAblationError("Benchmark identity drifted after validation.")
    if str(bundle.folds.contract.get("scientific_input_hash")) != scientific_input_hash:
        raise PolicyAblationError("Shared-fold scientific-input identity is incompatible.")
    if str(bundle.folds.contract.get("dataset_sha256")) != dataset_sha256:
        raise PolicyAblationError("Shared folds do not bind the canonical dataset bytes.")
    if int(bundle.folds.contract.get("outer_splits", -1)) != REQUIRED_OUTER_FOLDS:
        raise PolicyAblationError("Policy sensitivity requires exactly ten shared outer folds.")
    if tuple(bundle.raw_feature_order) != tuple(primary_features.columns):
        raise PolicyAblationError("Benchmark primary feature order differs from policy evidence.")

    identity: dict[str, Any] = {
        "run_id": run_id,
        "config_hash": config_hash,
        "scientific_input_hash": scientific_input_hash,
        "fold_contract_hash": bundle.identity.fold_contract_hash,
        "xgboost_model_set_sha256": bundle.model_set_sha256,
        "dataset_sha256": dataset_sha256,
    }
    for field in ("fold_contract_hash", "xgboost_model_set_sha256"):
        _require_sha256(field, identity[field])

    features_by_policy, feature_contract = _feature_contract_frame(
        data=data,
        policies=policies,
        definitions=definitions,
        target_column=target_column,
        id_column=id_column,
        identity=identity,
    )
    if tuple(features_by_policy[primary_policy].columns) != tuple(primary_features.columns):
        raise PolicyAblationError("Primary feature construction is not deterministic.")
    parameters_by_fold, parameter_contract = _selected_parameters_by_fold(
        bundle,
        policies=policies,
        features_by_policy=features_by_policy,
        random_state=random_state,
        identity=identity,
    )

    primary_feature_row = feature_contract.set_index("policy").loc[primary_policy]
    oof_rows = _primary_oof_rows(
        bundle,
        primary_policy=primary_policy,
        feature_row=primary_feature_row,
        parameter_contract=parameter_contract,
        identity=identity,
    )
    oof_rows.extend(
        _fit_non_primary_oof_rows(
            policies=policies,
            primary_policy=primary_policy,
            features_by_policy=features_by_policy,
            target=target,
            bundle=bundle,
            parameters_by_fold=parameters_by_fold,
            feature_contract=feature_contract,
            parameter_contract=parameter_contract,
            identity=identity,
            random_state=random_state,
            target_column=target_column,
            id_column=id_column,
        )
    )
    oof = pd.DataFrame(oof_rows).sort_values(["system_id", "sample_index"]).reset_index(drop=True)
    try:
        validate_consumer_fold_assignments(bundle.folds, oof, group_columns=("policy",))
        alignment = validate_aligned_oof_predictions(
            oof,
            labels=labels,
            task_type=task_type,
            metrics=metrics,
        )
    except (SharedFoldContractError, OOFBootstrapError) as exc:
        raise PolicyAblationError(f"Policy OOF alignment failed: {exc}") from exc
    if alignment["n_systems"] != len(policies) or alignment["n_samples"] != len(data):
        raise PolicyAblationError("Policy OOF alignment has the wrong system or sample denominator.")

    fold_metrics = _fold_metric_rows(
        oof,
        metrics=metrics,
        labels=labels,
        task_type=task_type,
        feature_contract=feature_contract,
        parameter_contract=parameter_contract,
    )
    fit_receipts = _fit_receipt_frame(fold_metrics, parameter_contract)
    comparisons, predeclared_pairs = _comparison_specs(policies)
    try:
        bootstrap: BootstrapResult = compute_paired_oof_bootstrap(
            oof,
            labels=labels,
            task_type=task_type,
            metrics=metrics,
            comparisons=comparisons,
            primary_metric=None,
            protocol=bootstrap_protocol,
        )
    except OOFBootstrapError as exc:
        raise PolicyAblationError(f"Policy paired OOF bootstrap failed: {exc}") from exc
    if int(bootstrap.metadata.get("n_resamples", -1)) != REQUIRED_BOOTSTRAP_RESAMPLES:
        raise PolicyAblationError("Policy bootstrap returned the wrong number of resamples.")
    if bootstrap.metadata.get("primary_metric") is not None:
        raise PolicyAblationError("Policy bootstrap must not declare a model-selection primary metric.")
    resample_hash = _validate_resample_binding(bootstrap.metadata, bundle.baseline_gate)

    intervals = bootstrap.metric_intervals.copy()
    pairwise = annotate_pairwise_differences(
        bootstrap.paired_differences,
        policies=policies,
        predeclared_pairs=predeclared_pairs,
    )
    for frame in (intervals, pairwise):
        for offset, (field, value) in enumerate(reversed(list(identity.items()))):
            frame.insert(0, field, value)
    summary = summarize_policies(fold_metrics, intervals, feature_contract)
    sensitivity = leakage_sensitivity_indices(
        pairwise,
        policies=policies,
        metrics=metrics,
        identity=identity,
        resample_hash=resample_hash,
        n_samples=len(data),
        n_resamples=REQUIRED_BOOTSTRAP_RESAMPLES,
    )
    manuscript = manuscript_policy_table(summary)

    ensure_dir(output.parent)
    temporary = tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent)
    staging = Path(temporary.name)
    paths: Dict[str, Path] = {
        "oof_predictions": staging / "oof_predictions.csv",
        "fold_metrics": staging / "fold_metrics.csv",
        "policy_metric_intervals": staging / "policy_metric_intervals.csv",
        "policy_summary": staging / "policy_summary.csv",
        "policy_pairwise_tests": staging / "policy_pairwise_tests.csv",
        "leakage_sensitivity_index": staging / "leakage_sensitivity_index.csv",
        "policy_feature_contract": staging / "policy_feature_contract.csv",
        "policy_hyperparameter_schedule": staging / "policy_hyperparameter_schedule.csv",
        "policy_fit_receipts": staging / "policy_fit_receipts.csv",
        "bootstrap_metadata": staging / "bootstrap_metadata.json",
        "policy_interpretation": staging / "policy_interpretation.md",
        "manuscript_policy_table": staging / "manuscript_policy_table.csv",
        "metadata": staging / "stage_metadata.json",
    }
    try:
        oof.to_csv(paths["oof_predictions"], index=False)
        fold_metrics.to_csv(paths["fold_metrics"], index=False)
        intervals.to_csv(paths["policy_metric_intervals"], index=False)
        summary.to_csv(paths["policy_summary"], index=False)
        pairwise.to_csv(paths["policy_pairwise_tests"], index=False)
        sensitivity.to_csv(paths["leakage_sensitivity_index"], index=False)
        feature_contract.to_csv(paths["policy_feature_contract"], index=False)
        parameter_contract.to_csv(paths["policy_hyperparameter_schedule"], index=False)
        fit_receipts.to_csv(paths["policy_fit_receipts"], index=False)
        manuscript.to_csv(paths["manuscript_policy_table"], index=False)
        write_interpretation(summary, paths["policy_interpretation"])
        write_json(
            paths["bootstrap_metadata"],
            {
                **dict(bootstrap.metadata),
                **identity,
                "policy_system_order": policies,
                "number_of_pairwise_comparisons": len(comparisons),
                "predeclared_comparisons": [list(pair) for pair in sorted(predeclared_pairs)],
                "multiplicity_adjustment": "none",
                "policy_model_gate_applicable": False,
            },
        )
        figure_paths = write_tradeoff_figure(summary, staging, identity=identity)
        paths.update({f"figure_{key}": value for key, value in figure_paths.items()})
        write_json(
            paths["metadata"],
            {
                "stage": "policy_ablation",
                "status": "complete",
                **identity,
                "primary_policy": primary_policy,
                "policies": policies,
                "labels": list(labels),
                "task_type": task_type,
                "metrics": list(metrics),
                "protocol": {
                    "comparison_protocol": dict(comparison_protocol),
                    "outer_folds_source": "shared_folds/fold_assignments.csv",
                    "n_outer_folds": REQUIRED_OUTER_FOLDS,
                    "primary_oof_source": "exact model_benchmarks/oof_predictions.csv rows",
                    "non_primary_fit_scope": "outer_train_only",
                    "non_primary_fit_threadpool_limit": 1,
                    "hyperparameter_source": "primary policy selected candidate for same outer fold",
                    "policy_independently_tuned": False,
                    "bootstrap": dict(bootstrap.metadata),
                    "fold_variability": "descriptive_mean_std_min_max_only",
                    "pairwise_inference": "pointwise_no_multiplicity_adjustment",
                },
                "upstream_file_hashes": dict(sorted(bundle.upstream_file_hashes.items())),
                "outputs": {
                    key: value.relative_to(staging).as_posix()
                    for key, value in paths.items()
                    if key != "metadata"
                },
                "claim_boundaries": [
                    "full_feature_upper_bound is an information-rich comparator not guaranteed optimized",
                    "audit-only policies are not deployable candidates",
                    "feature-policy differences are predictive sensitivity, not causality",
                    "field exclusion does not establish fairness or eliminate proxy risk",
                    "research evidence must not drive autonomous HR decisions",
                ],
            },
        )
        relative_paths = {key: path.relative_to(staging) for key, path in paths.items()}
        if output.exists():
            output.rmdir()
        staging.replace(output)
        temporary.cleanup()
    except Exception:
        temporary.cleanup()
        raise
    return {key: output / relative for key, relative in relative_paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched shared-fold XGBoost feature-policy sensitivity.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--shared-folds-dir", required=True)
    parser.add_argument("--model-benchmarks-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--scientific-input-hash", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            {
                key: str(value)
                for key, value in run(
                    arguments.config,
                    shared_folds_dir=arguments.shared_folds_dir,
                    model_benchmarks_dir=arguments.model_benchmarks_dir,
                    output_dir=arguments.output_dir,
                    run_id=arguments.run_id,
                    config_hash=arguments.config_hash,
                    scientific_input_hash=arguments.scientific_input_hash,
                ).items()
            },
            indent=2,
            sort_keys=True,
        )
    )


__all__ = [
    "PolicyAblationError",
    "exact_policy_frame",
    "leakage_sensitivity_indices",
    "manuscript_policy_table",
    "run",
    "summarize_policies",
]
