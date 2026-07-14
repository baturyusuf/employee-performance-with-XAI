"""Canonical support-aware subgroup and department proxy-risk evidence.

This stage is deliberately a consumer of current-run evidence.  Performance
subgroup diagnostics use the exact raw OOF rows produced by policy ablation;
they never fit or replay a performance model.  Department reconstructability
uses the persisted shared outer folds and two unique, target-free predictor
contracts.  All population-style intervals are paired sample-level stratified
bootstrap intervals.  Fold summaries are descriptive only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.core.atomic_publish import atomic_replace_directory, cleanup_temporary_directory
from src.data.canonical_loader import load_canonical_dataset
from src.experiments.benchmark_artifact_contract import (
    BenchmarkArtifactContractError,
    XGBoostOOFArtifacts,
    read_xgboost_oof_artifacts,
    validate_xgboost_oof_replay,
)
from src.experiments.manuscript_policy_ablation import exact_policy_frame, resolve_seed
from src.experiments.shared_folds import (
    SharedFoldContractError,
    read_shared_folds,
    validate_consumer_fold_assignments,
)
from src.governance.manuscript_contract import (
    canonical_config_hash,
    load_manuscript_config,
    primary_excluded_features,
    sha256_file,
)
from src.models.oof_bootstrap import (
    BootstrapProtocol,
    OOFBootstrapError,
    ResamplePlan,
    generate_stratified_resample_indices,
    validate_aligned_oof_predictions,
)
from src.models.task_schema import NOMINAL_MULTICLASS_PROXY_DIAGNOSTIC


DEFAULT_CONFIG = Path("configs/manuscript_final.yaml")
PRIMARY_TASK = "ordinal_multiclass_performance"
PRIMARY_POLICY = "no_salary_hike_no_attrition_no_department"
STRICT_POLICY = "no_salary_hike_no_attrition_no_department_no_job_role"
ALIAS_POLICY = "no_salary_hike_no_attrition"
REQUIRED_POLICY_COMPARISONS = (ALIAS_POLICY, PRIMARY_POLICY, STRICT_POLICY)
UNIQUE_PROXY_POLICIES = (PRIMARY_POLICY, STRICT_POLICY)
REQUIRED_BOOTSTRAP_RESAMPLES = 5000
CONDITIONAL_INFERENCE_NOTE = (
    "Intervals condition on the observed employees and fixed fold/model-training protocol; "
    "they do not estimate model-training instability."
)
IDENTITY_FIELDS = (
    "run_id",
    "config_hash",
    "scientific_input_hash",
    "fold_contract_hash",
    "xgboost_model_set_sha256",
    "dataset_sha256",
)
OVERALL_METRICS = ("accuracy", "macro_f1")
CLASS_METRICS = (
    "positive_prediction_rate",
    "true_positive_rate",
    "false_positive_rate",
    "precision",
    "mean_predicted_probability",
)
PROXY_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


class FairnessProxyError(RuntimeError):
    """Raised when canonical subgroup/proxy evidence cannot be produced."""


@dataclass(frozen=True)
class PolicyEvidence:
    """Validated policy-stage evidence consumed by subgroup diagnostics."""

    oof_predictions: pd.DataFrame = field(repr=False, compare=False)
    feature_contract: pd.DataFrame = field(repr=False, compare=False)
    stage_metadata: Mapping[str, Any]
    bootstrap_metadata: Mapping[str, Any]
    upstream_file_hashes: Mapping[str, str]
    performance_resample_hash: str


@dataclass(frozen=True)
class SubgroupBootstrapEvidence:
    intervals: pd.DataFrame = field(repr=False, compare=False)
    paired_differences: pd.DataFrame = field(repr=False, compare=False)
    draw_map: Mapping[tuple[str, str, str, int | None], np.ndarray] = field(
        repr=False, compare=False
    )
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ProxyEvidence:
    fold_assignments: pd.DataFrame = field(repr=False, compare=False)
    feature_contracts: pd.DataFrame = field(repr=False, compare=False)
    equivalence: pd.DataFrame = field(repr=False, compare=False)
    oof_predictions: pd.DataFrame = field(repr=False, compare=False)
    fold_metrics: pd.DataFrame = field(repr=False, compare=False)
    descriptive_summary: pd.DataFrame = field(repr=False, compare=False)
    metric_intervals: pd.DataFrame = field(repr=False, compare=False)
    paired_differences: pd.DataFrame = field(repr=False, compare=False)
    associations: pd.DataFrame = field(repr=False, compare=False)
    label_mapping: Mapping[str, Any]
    bootstrap_metadata: Mapping[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _identity_bound_mapping(
    mapping: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    for field_name, expected in identity.items():
        if field_name in mapping and mapping[field_name] != expected:
            raise FairnessProxyError(
                f"Mapping {field_name} conflicts with the current scientific identity."
            )
    return {**identity, **dict(mapping)}


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FairnessProxyError(f"Required {name} is missing or empty: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FairnessProxyError(f"Cannot read {name}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, dict):
        raise FairnessProxyError(f"{name} must contain a JSON object.")
    return value


def _read_csv(path: Path, *, name: str, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FairnessProxyError(f"Required {name} is missing or empty: {path}")
    try:
        frame = pd.read_csv(path, float_precision="round_trip")
    except Exception as exc:
        raise FairnessProxyError(f"Cannot read {name}: {type(exc).__name__}: {exc}") from exc
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise FairnessProxyError(f"{name} is missing required columns: {missing}")
    if frame.empty:
        raise FairnessProxyError(f"{name} must not be empty.")
    return frame


def _require_sha256(name: str, value: Any) -> str:
    observed = str(value)
    if len(observed) != 64 or any(character not in "0123456789abcdef" for character in observed):
        raise FairnessProxyError(f"{name} must be a lowercase SHA-256 digest.")
    return observed


def _snapshot(paths: Mapping[str, Path]) -> Mapping[str, str]:
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FairnessProxyError(f"Required upstream {name} is missing or empty: {path}")
        hashes[name] = sha256_file(path)
    return MappingProxyType(hashes)


def _settings(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = load_manuscript_config(config_path)
    settings = raw.get("manuscript_final")
    if not isinstance(settings, dict):
        raise FairnessProxyError("Canonical config must contain manuscript_final.")
    return raw, settings


def _audit_category(attribute: str, sensitive: set[str]) -> str:
    return (
        "protected_or_sensitive_descriptive_audit"
        if attribute in sensitive
        else "exploratory_operational_subgroup_diagnostic"
    )


def _cramers_v(feature: pd.Series, target: pd.Series) -> float:
    table = pd.crosstab(feature, target)
    if table.empty or min(table.shape) <= 1:
        return 0.0
    chi2 = float(chi2_contingency(table, correction=False)[0])
    n = int(table.to_numpy().sum())
    denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi2 / denominator)) if denominator > 0 else 0.0


def feature_proxy_associations(
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    """Compute deterministic exploratory associations without legacy-stage imports."""

    encoded_target, _ = pd.factorize(target.astype("string").fillna("__MISSING__"))
    rows: list[dict[str, Any]] = []
    for feature_name in features.columns:
        series = features[feature_name]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                numeric = numeric.fillna(float(numeric.median()))
                mutual_information = float(
                    mutual_info_classif(
                        numeric.to_numpy().reshape(-1, 1),
                        encoded_target,
                        random_state=int(random_state),
                        discrete_features=False,
                    )[0]
                )
            else:
                mutual_information = 0.0
            association_type = "mutual_info_numeric"
            cramers = math.nan
        else:
            categorical = series.astype("string").fillna("__MISSING__").astype(str)
            encoded_feature, _ = pd.factorize(categorical)
            mutual_information = float(
                mutual_info_classif(
                    encoded_feature.reshape(-1, 1),
                    encoded_target,
                    random_state=int(random_state),
                    discrete_features=True,
                )[0]
            )
            association_type = "categorical_mi_and_cramers_v"
            cramers = _cramers_v(categorical, target.astype(str))
        rows.append(
            {
                "feature": str(feature_name),
                "association_type": association_type,
                "mutual_info": mutual_information,
                "cramers_v": cramers,
                "proxy_watchlist": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mutual_info", "cramers_v", "feature"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def _validate_protocol(settings: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    fairness = settings.get("fairness")
    proxy = settings.get("proxy_analysis")
    if not isinstance(fairness, Mapping) or not isinstance(proxy, Mapping):
        raise FairnessProxyError("Canonical fairness and proxy_analysis mappings are required.")
    if proxy.get("task_type") != NOMINAL_MULTICLASS_PROXY_DIAGNOSTIC:
        raise FairnessProxyError("Department reconstructability requires the nominal proxy task schema.")

    expected_prediction = {
        "required_upstream_stages": ["shared_folds", "model_benchmarks", "policy_ablation"],
        "outer_folds_source": "shared_folds.outer_fold_assignments",
        "oof_predictions_source": "policy_ablation.exact_oof_predictions",
        "primary_model_provenance_source": "model_benchmarks.xgboost_selected_candidate_by_outer_fold",
        "model_refit_in_stage": False,
        "probability_source": "policy_ablation.raw_uncalibrated_oof_probabilities",
        "probability_semantics": "raw_uncalibrated_for_matched_cross_policy_audit",
    }
    if dict(fairness.get("prediction_contract", {})) != expected_prediction:
        raise FairnessProxyError("fairness.prediction_contract differs from the frozen OOF contract.")
    if fairness.get("policy_comparisons_source") != "policy_ablation.exact_oof_predictions":
        raise FairnessProxyError("Fairness policy comparisons must consume exact policy OOF evidence.")
    if list(fairness.get("bootstrap_stratify_by", ())) != ["outer_fold", "y_true"]:
        raise FairnessProxyError("Fairness bootstrap strata must be outer_fold plus y_true.")
    expected_bootstrap_contract = {
        "n_resamples_source": "evaluation.bootstrap.n_resamples",
        "method_source": "evaluation.bootstrap.method",
        "confidence_level_source": "evaluation.bootstrap.confidence_level",
        "quantile_method_source": "evaluation.bootstrap.quantile_method",
        "seed_source": "seeds.bootstrap",
        "same_resamples_across_policies": True,
        "resample_hash_required": True,
        "resample_hash_source": "policy_ablation.bootstrap_metadata.resample_hash",
        "resample_hash_equality_required_with": (
            "model_benchmarks.baseline_xgboost_gate.resample_hash"
        ),
    }
    if dict(fairness.get("bootstrap_contract", {})) != expected_bootstrap_contract:
        raise FairnessProxyError("fairness.bootstrap_contract differs from the frozen contract.")
    expected_statuses = [
        "insufficient_subgroup_or_metric_support",
        "unstable_insufficient_valid_bootstrap_replicates",
        "support_sufficient_but_interval_wide",
        "support_sufficient_descriptive_estimate",
    ]
    support_rules = fairness.get("support_status_rules", {})
    if not isinstance(support_rules, Mapping) or list(support_rules.get("status_values", ())) != expected_statuses:
        raise FairnessProxyError("Fairness support status values differ from the frozen contract.")
    expected_paired_statuses = [
        "insufficient_common_subgroup_or_metric_support",
        "unstable_insufficient_valid_bootstrap_replicates",
        "support_sufficient_but_interval_wide",
        "support_sufficient_descriptive_estimate",
    ]
    if list(support_rules.get("paired_status_values", ())) != expected_paired_statuses:
        raise FairnessProxyError("Fairness paired status values differ from the frozen contract.")
    required_support_values = {
        "below_threshold_rows_retained": True,
        "below_threshold_rows_eligible_for_gap": False,
        "class_specific_metrics_use_metric_denominator": True,
        "minimum_two_eligible_groups_for_gap": True,
        "eligibility_scope": "fixed_from_complete_oof_before_resampling",
        "paired_policy_common_group_scope": (
            "intersection_of_complete_oof_eligible_groups_per_pair_attribute_metric_class"
        ),
        "paired_policy_minimum_common_groups": 2,
    }
    if any(support_rules.get(key) != value for key, value in required_support_values.items()):
        raise FairnessProxyError("Fairness support eligibility rules differ from the frozen contract.")
    headline = fairness.get("headline_rules", {})
    if not isinstance(headline, Mapping) or list(headline.get("eligible_statuses", ())) != [
        "support_sufficient_descriptive_estimate"
    ]:
        raise FairnessProxyError("Only stable support-sufficient subgroup estimates may be headlined.")
    for key in (
        "wide_interval_rows_headline_eligible",
        "unstable_rows_headline_eligible",
        "insufficient_support_rows_headline_eligible",
        "paired_policy_rows_headline_eligible",
    ):
        if headline.get(key) is not False:
            raise FairnessProxyError(f"fairness.headline_rules.{key} must be false.")
    for key in (
        "require_minimum_subgroup_support_context",
        "require_minimum_metric_denominator_context",
        "require_valid_bootstrap_context",
        "boundary_value_one_requires_explicit_support_context",
    ):
        if headline.get(key) is not True:
            raise FairnessProxyError(f"fairness.headline_rules.{key} must be true.")
    expected_inference_scope = {
        "intervals": "pointwise_descriptive",
        "multiplicity_adjustment": "none",
        "simultaneous_or_familywise_claims_allowed": False,
        "observed_gap_ranking": "descriptive_only_no_selection_adjusted_inference",
    }
    if dict(fairness.get("inference_scope", {})) != expected_inference_scope:
        raise FairnessProxyError("fairness.inference_scope differs from the frozen contract.")
    batch_size = fairness.get("bootstrap_batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise FairnessProxyError("fairness.bootstrap_batch_size must be a positive integer.")

    expected_classifier = {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "solver": "lbfgs",
        "regularization": "l2_via_l1_ratio_zero",
        "l1_ratio": 0.0,
        "C": 1.0,
        "class_weight": "balanced",
        "fit_intercept": True,
        "max_iter": 5000,
        "tol": 0.0001,
        "multiclass_mode": "native_multinomial_for_three_or_more_classes",
        "random_state_source": "seeds.fairness",
        "estimator_threads": 1,
    }
    if dict(proxy.get("classifier", {})) != expected_classifier:
        raise FairnessProxyError("proxy_analysis.classifier differs from the frozen estimator contract.")
    expected_preprocessing = {
        "numeric": "median_imputation_then_standard_scaling",
        "categorical": "most_frequent_imputation_then_dense_one_hot_handle_unknown_ignore",
        "fit_scope": "outer_training_partition_only",
    }
    if dict(proxy.get("preprocessing", {})) != expected_preprocessing:
        raise FairnessProxyError("proxy_analysis.preprocessing differs from the frozen contract.")
    expected_unique = {
        PRIMARY_POLICY: {
            "source_policy": PRIMARY_POLICY,
            "job_role_retained": True,
            "proxy_target_removed": True,
        },
        STRICT_POLICY: {
            "source_policy": STRICT_POLICY,
            "job_role_retained": False,
            "proxy_target_removed": True,
        },
    }
    if dict(proxy.get("unique_predictor_contracts", {})) != expected_unique:
        raise FairnessProxyError("proxy_analysis.unique_predictor_contracts differs from the frozen contract.")
    if dict(proxy.get("policy_aliases", {})) != {ALIAS_POLICY: PRIMARY_POLICY}:
        raise FairnessProxyError("The department-including proxy policy must be an explicit alias.")
    if list(proxy.get("reported_policy_order", ())) != list(REQUIRED_POLICY_COMPARISONS):
        raise FairnessProxyError("Proxy reported policy order differs from the frozen contract.")
    expected_proxy_bootstrap = {
        "n_resamples": 5000,
        "method_source": "evaluation.bootstrap.method",
        "confidence_level_source": "evaluation.bootstrap.confidence_level",
        "quantile_method_source": "evaluation.bootstrap.quantile_method",
        "seed_source": "seeds.fairness",
        "stratify_by": ["outer_fold", "proxy_target"],
        "paired_across_unique_predictor_contracts": True,
        "resample_hash_required": True,
        "resample_hash_scope": "proxy_target_oof_bootstrap_indices",
        "separate_from_performance_policy_bootstrap": True,
        "batch_size_source": "fairness.bootstrap_batch_size",
        "semantic_strata_adapter": {
            "semantic_columns": ["outer_fold", "proxy_target"],
            "internal_columns": ["outer_fold", "y_true"],
            "internal_y_true_semantics": (
                "deterministic_sorted_proxy_target_class_codes"
            ),
            "performance_target_used": False,
            "adapter_receipt_required": True,
            "adapter_hash_required": True,
        },
    }
    if dict(proxy.get("bootstrap", {})) != expected_proxy_bootstrap:
        raise FairnessProxyError("proxy_analysis.bootstrap differs from the frozen target-specific contract.")
    if proxy.get("outer_folds_source") != "shared_folds.outer_fold_assignments":
        raise FairnessProxyError("Proxy models must use the persisted shared outer folds.")
    if proxy.get("target_removed_from_all_proxy_predictors") is not True:
        raise FairnessProxyError("The proxy target must be removed from every predictor contract.")
    if list(proxy.get("metrics", ())) != list(PROXY_METRICS):
        raise FairnessProxyError("Proxy metric order differs from the frozen contract.")
    expected_oof_contract = {
        "exactly_once_per_sample_per_unique_predictor_contract": True,
        "fold_assignment_source": "shared_folds.outer_fold_assignments",
        "proxy_target_absent_from_predictors": True,
    }
    if dict(proxy.get("oof_contract", {})) != expected_oof_contract:
        raise FairnessProxyError("proxy_analysis.oof_contract differs from the frozen contract.")
    if proxy.get("primary_uncertainty") != "paired_sample_level_stratified_percentile_bootstrap":
        raise FairnessProxyError("Proxy primary uncertainty must be paired sample-level bootstrap.")
    if proxy.get("fold_summary_scope") != "descriptive_mean_std_min_max_only_no_population_ci":
        raise FairnessProxyError("Proxy fold summaries must be descriptive and contain no population CI.")
    expected_proxy_inference_scope = {
        "intervals": "pointwise_descriptive",
        "multiplicity_adjustment": "none",
        "simultaneous_or_familywise_claims_allowed": False,
        "paired_rows_headline_eligible": False,
    }
    if dict(proxy.get("inference_scope", {})) != expected_proxy_inference_scope:
        raise FairnessProxyError("proxy_analysis.inference_scope differs from the frozen contract.")
    return fairness, proxy


def transform_audit_attribute(
    values: pd.Series,
    attribute: str,
    transform: Mapping[str, Any] | None,
) -> pd.Series:
    """Apply one predeclared audit transform without outcome learning."""

    if not transform:
        return values.astype("string").fillna("__MISSING__").astype(str)
    if transform.get("type") != "numeric_bins":
        raise FairnessProxyError(f"Unsupported transform for {attribute}: {transform}")
    edges = [float(value) for value in transform.get("edges", ())]
    labels = [str(value) for value in transform.get("labels", ())]
    if len(edges) != len(labels) + 1 or len(labels) < 2:
        raise FairnessProxyError(f"Invalid numeric-bin definition for {attribute}.")
    numeric = pd.to_numeric(values, errors="coerce")
    binned = pd.cut(numeric, bins=edges, labels=labels, right=True, include_lowest=True)
    return binned.astype("string").fillna("__MISSING__").astype(str)


def _sample_indexed_data(data: pd.DataFrame) -> pd.DataFrame:
    """Key audit fields by explicit integer sample identity, never row position."""

    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in data.index
    ):
        raise FairnessProxyError("Canonical audit data index is not an integer sample identity.")
    sample_index = pd.Index([int(value) for value in data.index], name="sample_index")
    if sample_index.has_duplicates:
        raise FairnessProxyError("Canonical audit data repeats sample identities.")
    indexed = data.copy()
    indexed.index = sample_index
    return indexed


def _validate_mapping_identity(value: Mapping[str, Any], identity: Mapping[str, Any], name: str) -> None:
    for field_name, expected in identity.items():
        if value.get(field_name) != expected:
            raise FairnessProxyError(
                f"{name} {field_name} differs from the current run identity: "
                f"observed={value.get(field_name)!r}, expected={expected!r}."
            )


def _validate_frame_identity(frame: pd.DataFrame, identity: Mapping[str, Any], name: str) -> None:
    missing = sorted(set(identity).difference(frame.columns))
    if missing:
        raise FairnessProxyError(f"{name} is missing identity columns: {missing}")
    for field_name, expected in identity.items():
        if set(frame[field_name].astype(str)) != {str(expected)}:
            raise FairnessProxyError(f"{name} {field_name} is incompatible with the current run.")


def _boolean_column(frame: pd.DataFrame, column: str, *, name: str) -> pd.Series:
    normalized = frame[column].astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise FairnessProxyError(f"{name} {column} must contain only explicit booleans.")
    return normalized.eq("true")


def read_policy_evidence(
    policy_ablation_dir: str | Path,
    *,
    bundle: XGBoostOOFArtifacts,
    data: pd.DataFrame,
    settings: Mapping[str, Any],
    identity: Mapping[str, Any],
    labels: Sequence[int],
) -> PolicyEvidence:
    """Read and cross-validate the exact three policy OOF systems."""

    root = Path(policy_ablation_dir).resolve()
    paths = {
        "stage_metadata": root / "stage_metadata.json",
        "oof_predictions": root / "oof_predictions.csv",
        "bootstrap_metadata": root / "bootstrap_metadata.json",
        "policy_feature_contract": root / "policy_feature_contract.csv",
        "policy_hyperparameter_schedule": root / "policy_hyperparameter_schedule.csv",
        "policy_fit_receipts": root / "policy_fit_receipts.csv",
    }
    snapshot = _snapshot(paths)
    stage = _read_json(paths["stage_metadata"], name="policy stage metadata")
    bootstrap = _read_json(paths["bootstrap_metadata"], name="policy bootstrap metadata")
    _validate_mapping_identity(stage, identity, "Policy stage metadata")
    _validate_mapping_identity(bootstrap, identity, "Policy bootstrap metadata")
    if stage.get("stage") != "policy_ablation" or stage.get("status") != "complete":
        raise FairnessProxyError("Policy stage is not complete current-run evidence.")
    if int(bootstrap.get("n_resamples", -1)) != REQUIRED_BOOTSTRAP_RESAMPLES:
        raise FairnessProxyError("Policy bootstrap must contain exactly 5,000 resamples.")
    if list(bootstrap.get("strata_columns", ())) != ["outer_fold", "y_true"]:
        raise FairnessProxyError("Policy bootstrap strata differ from outer_fold plus y_true.")
    evaluation_bootstrap = settings["evaluation"]["bootstrap"]
    for field_name in ("method", "confidence_level", "quantile_method"):
        if bootstrap.get(field_name) != evaluation_bootstrap[field_name]:
            raise FairnessProxyError(
                f"Policy bootstrap {field_name} differs from the canonical evaluation contract."
            )
    performance_hash = _require_sha256("policy resample_hash", bootstrap.get("resample_hash"))
    if performance_hash != _require_sha256(
        "benchmark resample_hash", bundle.baseline_gate.get("resample_hash")
    ):
        raise FairnessProxyError("Policy and benchmark performance resample hashes differ.")

    required_oof = {
        *IDENTITY_FIELDS,
        "system_id",
        "policy",
        "model",
        "sample_index",
        "outer_fold",
        "y_true",
        "y_pred",
        "selected_candidate_index",
        "policy_model_contract_sha256",
        "model_fit_mode",
        *(f"prob_class_{int(label)}" for label in labels),
    }
    all_oof = _read_csv(
        paths["oof_predictions"], name="policy OOF predictions", required=required_oof
    )
    selected = all_oof[
        all_oof["policy"].astype(str).isin(REQUIRED_POLICY_COMPARISONS)
    ].copy()
    selected["task_type"] = PRIMARY_TASK
    if set(selected["policy"].astype(str)) != set(REQUIRED_POLICY_COMPARISONS):
        raise FairnessProxyError("Policy OOF evidence lacks one or more required subgroup systems.")
    if not selected["system_id"].astype(str).equals(selected["policy"].astype(str)):
        raise FairnessProxyError("Policy OOF system_id must equal policy for every selected row.")
    _validate_frame_identity(selected, identity, "Policy OOF predictions")
    try:
        validate_consumer_fold_assignments(bundle.folds, selected, group_columns=("policy",))
        alignment = validate_aligned_oof_predictions(
            selected,
            labels=labels,
            task_type=PRIMARY_TASK,
            metrics=("macro_f1",),
        )
    except (SharedFoldContractError, OOFBootstrapError) as exc:
        raise FairnessProxyError(f"Policy OOF alignment is invalid: {exc}") from exc
    if alignment["n_systems"] != 3 or alignment["n_samples"] != len(data):
        raise FairnessProxyError("Policy OOF evidence must contain exactly three full-coverage systems.")
    for label in labels:
        selected[f"prob_class_{int(label)}"] = pd.to_numeric(
            selected[f"prob_class_{int(label)}"], errors="raise"
        )
    probability = selected[[f"prob_class_{int(label)}" for label in labels]].to_numpy(float)
    argmax = np.asarray(labels, dtype=int)[np.argmax(probability, axis=1)]
    if not np.array_equal(argmax, selected["y_pred"].astype(int).to_numpy()):
        raise FairnessProxyError("Policy predictions disagree with raw probability argmax.")
    primary = selected[selected["policy"].astype(str) == PRIMARY_POLICY].sort_values("sample_index")
    benchmark = bundle.oof_predictions.sort_values("sample_index")
    if not np.array_equal(primary["sample_index"].astype(int), benchmark["sample_index"].astype(int)):
        raise FairnessProxyError("Primary policy sample order differs from exact benchmark OOF evidence.")
    for column in ("outer_fold", "y_true", "y_pred"):
        if not np.array_equal(primary[column].astype(int), benchmark[column].astype(int)):
            raise FairnessProxyError(f"Primary policy {column} differs from exact benchmark OOF evidence.")
    for label in labels:
        column = f"prob_class_{int(label)}"
        if not np.allclose(
            primary[column].to_numpy(float), benchmark[column].to_numpy(float), rtol=0.0, atol=1e-12
        ):
            raise FairnessProxyError("Primary policy raw probabilities differ from benchmark OOF evidence.")

    feature_contract = _read_csv(
        paths["policy_feature_contract"],
        name="policy feature contract",
        required={*IDENTITY_FIELDS, "policy", "n_features", "excluded_features_json", "feature_columns_json"},
    )
    feature_contract = feature_contract[
        feature_contract["policy"].astype(str).isin(REQUIRED_POLICY_COMPARISONS)
    ].copy()
    _validate_frame_identity(feature_contract, identity, "Policy feature contract")
    if len(feature_contract) != 3 or feature_contract["policy"].astype(str).duplicated().any():
        raise FairnessProxyError("Policy feature contract must contain exactly the three selected policies.")
    definitions = settings["feature_policies"]["definitions"]
    target_column = str(settings["target"]["column"])
    id_column = str(settings["governance_fields"]["identifier_fields"][0])
    for row in feature_contract.itertuples(index=False):
        features, excluded = exact_policy_frame(
            data,
            str(row.policy),
            definitions[str(row.policy)],
            target_column=target_column,
            id_column=id_column,
        )
        if json.loads(str(row.feature_columns_json)) != features.columns.tolist():
            raise FairnessProxyError(f"Policy {row.policy!r} feature order differs from current config/data.")
        if int(row.n_features) != int(features.shape[1]):
            raise FairnessProxyError(f"Policy {row.policy!r} feature count differs from current config/data.")
        if json.loads(str(row.excluded_features_json)) != excluded:
            raise FairnessProxyError(f"Policy {row.policy!r} exclusions differ from current config/data.")

    schedule = _read_csv(
        paths["policy_hyperparameter_schedule"],
        name="policy hyperparameter schedule",
        required={*IDENTITY_FIELDS, "policy", "outer_fold", "policy_model_contract_sha256"},
    )
    schedule = schedule[schedule["policy"].astype(str).isin(REQUIRED_POLICY_COMPARISONS)].copy()
    _validate_frame_identity(schedule, identity, "Policy hyperparameter schedule")
    if len(schedule) != 30 or schedule.duplicated(["policy", "outer_fold"]).any():
        raise FairnessProxyError("Selected policy schedule must contain one row per policy and outer fold.")
    keyed = schedule.set_index(["policy", "outer_fold"])["policy_model_contract_sha256"].astype(str)
    observed = selected[["policy", "outer_fold", "policy_model_contract_sha256"]].drop_duplicates()
    for row in observed.itertuples(index=False):
        if keyed.loc[(str(row.policy), int(row.outer_fold))] != str(row.policy_model_contract_sha256):
            raise FairnessProxyError("Policy OOF model-contract identity differs from its schedule.")

    required_schedule = {
        "model",
        "selected_candidate_index",
        "fixed_parameters_json",
        "selected_candidate_parameters_json",
        "parameter_source",
        "outer_test_used_for_parameter_selection",
        "policy_independently_tuned",
        "planned_fit_threadpool_limit",
        "source_primary_model_sha256",
        "source_primary_model_persisted",
    }
    missing_schedule = sorted(required_schedule.difference(schedule.columns))
    if missing_schedule:
        raise FairnessProxyError(
            f"Policy hyperparameter schedule lacks benchmark-binding fields: {missing_schedule}."
        )
    selected_parameters = bundle.selected_hyperparameters.set_index("outer_fold")
    for row in schedule.itertuples(index=False):
        outer_fold = int(row.outer_fold)
        benchmark_row = selected_parameters.loc[outer_fold]
        fold_model = bundle.fold_models.get(outer_fold)
        if fold_model is None:
            raise FairnessProxyError(f"Benchmark model is missing for outer fold {outer_fold}.")
        if (
            str(row.model) != "xgboost"
            or int(row.selected_candidate_index)
            != int(benchmark_row["selected_candidate_index"])
            or json.loads(str(row.fixed_parameters_json))
            != json.loads(str(benchmark_row["fixed_parameters_json"]))
            or json.loads(str(row.selected_candidate_parameters_json))
            != json.loads(str(benchmark_row["selected_candidate_parameters_json"]))
            or str(row.source_primary_model_sha256) != str(fold_model.sha256)
            or int(row.planned_fit_threadpool_limit) != 1
            or str(row.parameter_source)
            != "primary_policy_nested_selection_same_outer_fold"
        ):
            raise FairnessProxyError(
                f"Policy schedule drifts from the selected benchmark model in outer fold {outer_fold}."
            )
    if _boolean_column(
        schedule,
        "outer_test_used_for_parameter_selection",
        name="Policy schedule",
    ).any():
        raise FairnessProxyError("Policy schedule used outer-test evidence for parameter selection.")
    if _boolean_column(
        schedule, "policy_independently_tuned", name="Policy schedule"
    ).any():
        raise FairnessProxyError("Policy sensitivity systems must not be independently tuned.")
    if not _boolean_column(
        schedule, "source_primary_model_persisted", name="Policy schedule"
    ).all():
        raise FairnessProxyError("Policy schedule lacks a persisted source benchmark model.")
    selected_schedule = schedule.set_index(["policy", "outer_fold"])
    for row in selected.itertuples(index=False):
        schedule_row = selected_schedule.loc[(str(row.policy), int(row.outer_fold))]
        if int(row.selected_candidate_index) != int(schedule_row["selected_candidate_index"]):
            raise FairnessProxyError("Policy OOF selected-candidate identity differs from its schedule.")

    receipts = _read_csv(
        paths["policy_fit_receipts"],
        name="policy fit receipts",
        required={
            *IDENTITY_FIELDS,
            "policy",
            "outer_fold",
            "execution_status",
            "stage_fit_performed",
            "upstream_primary_fit_complete",
            "primary_benchmark_oof_reused",
            "selected_candidate_index",
            "parameter_source_sha256",
            "source_primary_model_sha256",
            "policy_independently_tuned",
        },
    )
    receipts = receipts[
        receipts["policy"].astype(str).isin(REQUIRED_POLICY_COMPARISONS)
    ].copy()
    _validate_frame_identity(receipts, identity, "Policy fit receipts")
    if len(receipts) != 30 or receipts.duplicated(["policy", "outer_fold"]).any():
        raise FairnessProxyError("Policy fit receipts must contain one row per policy and outer fold.")
    if set(receipts["execution_status"].astype(str)) != {"complete"}:
        raise FairnessProxyError("Policy fit receipts contain an incomplete execution.")
    if not _boolean_column(
        receipts, "upstream_primary_fit_complete", name="Policy fit receipts"
    ).all():
        raise FairnessProxyError("Policy fit receipts do not bind a complete upstream fit.")
    if _boolean_column(
        receipts, "policy_independently_tuned", name="Policy fit receipts"
    ).any():
        raise FairnessProxyError("Policy fit receipts declare independent tuning.")
    receipts["_stage_fit_performed_bool"] = _boolean_column(
        receipts, "stage_fit_performed", name="Policy fit receipts"
    )
    receipts["_primary_benchmark_oof_reused_bool"] = _boolean_column(
        receipts, "primary_benchmark_oof_reused", name="Policy fit receipts"
    )
    receipt_lookup = receipts.set_index(["policy", "outer_fold"])
    for row in schedule.itertuples(index=False):
        key = (str(row.policy), int(row.outer_fold))
        receipt = receipt_lookup.loc[key]
        stage_fit = bool(receipt["_stage_fit_performed_bool"])
        benchmark_reuse = bool(receipt["_primary_benchmark_oof_reused_bool"])
        primary = key[0] == PRIMARY_POLICY
        if (
            int(receipt["selected_candidate_index"]) != int(row.selected_candidate_index)
            or str(receipt["parameter_source_sha256"])
            != str(row.policy_model_contract_sha256)
            or str(receipt["source_primary_model_sha256"])
            != str(row.source_primary_model_sha256)
            or stage_fit == primary
            or benchmark_reuse != primary
        ):
            raise FairnessProxyError(f"Policy fit receipt drifts from its schedule for {key}.")

    return PolicyEvidence(
        oof_predictions=selected.sort_values(["policy", "sample_index"]).reset_index(drop=True),
        feature_contract=feature_contract.sort_values("policy").reset_index(drop=True),
        stage_metadata=MappingProxyType(stage),
        bootstrap_metadata=MappingProxyType(bootstrap),
        upstream_file_hashes=snapshot,
        performance_resample_hash=performance_hash,
    )


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
    """Return every subgroup row, including support-ineligible rows."""

    if minimum_group_support < 1 or minimum_class_denominator < 1:
        raise FairnessProxyError("Subgroup and metric support thresholds must be positive.")
    indexed = _sample_indexed_data(data)
    missing_attributes = [attribute for attribute in attributes if attribute not in indexed]
    if missing_attributes:
        raise FairnessProxyError(f"Configured audit attribute is missing: {missing_attributes}")
    audit_values = {
        attribute: transform_audit_attribute(
            indexed[attribute], attribute, transforms.get(attribute)
        ).set_axis(indexed.index)
        for attribute in attributes
    }
    probability_columns = [f"prob_class_{int(label)}" for label in labels]
    missing_prediction = sorted(
        {"policy", "sample_index", "y_true", "y_pred", *probability_columns}.difference(predictions.columns)
    )
    if missing_prediction:
        raise FairnessProxyError(f"Subgroup OOF predictions are missing columns: {missing_prediction}")

    rows: list[dict[str, Any]] = []
    identity_columns = [column for column in IDENTITY_FIELDS if column in predictions]
    for policy, scoped in predictions.groupby("policy", sort=False):
        ordered = scoped.sort_values("sample_index").reset_index(drop=True)
        if ordered["sample_index"].duplicated().any():
            raise FairnessProxyError(f"Policy {policy!r} repeats subgroup OOF sample rows.")
        sample_indices = ordered["sample_index"].astype(int).to_numpy()
        if not set(sample_indices).issubset(indexed.index):
            raise FairnessProxyError("Subgroup OOF sample identities are absent from canonical data.")
        y_true = ordered["y_true"].astype(int).to_numpy()
        y_pred = ordered["y_pred"].astype(int).to_numpy()
        identity = {column: ordered[column].iloc[0] for column in identity_columns}
        for attribute, full_values in audit_values.items():
            values = full_values.loc[sample_indices].to_numpy(dtype=str)
            for group_value in sorted(pd.unique(values).tolist()):
                mask = values == group_value
                group_true = y_true[mask]
                group_pred = y_pred[mask]
                n_samples = int(mask.sum())
                support_ok = n_samples >= minimum_group_support
                common = {
                    **identity,
                    "task_type": PRIMARY_TASK,
                    "policy": str(policy),
                    "attribute": attribute,
                    "group_value": str(group_value),
                    "interpretation_category": _audit_category(attribute, sensitive_attributes),
                    "n_samples": n_samples,
                    "minimum_group_support_threshold": int(minimum_group_support),
                    "group_support_eligible": bool(support_ok),
                    "probability_source": "policy_ablation_raw_uncalibrated_oof_probabilities",
                }
                overall = {
                    "accuracy": (
                        float(accuracy_score(group_true, group_pred)),
                        int(np.sum(group_true == group_pred)),
                    ),
                    "macro_f1": (
                        float(
                            f1_score(
                                group_true,
                                group_pred,
                                labels=list(labels),
                                average="macro",
                                zero_division=0,
                            )
                        ),
                        math.nan,
                    ),
                }
                for metric, (value, numerator) in overall.items():
                    rows.append(
                        {
                            **common,
                            "metric": metric,
                            "class_label": np.nan,
                            "metric_value": value,
                            "metric_numerator": numerator,
                            "metric_denominator": n_samples,
                            "minimum_metric_denominator_threshold": int(minimum_group_support),
                            "metric_denominator_eligible": bool(support_ok),
                            "eligible_for_gap": bool(support_ok),
                        }
                    )
                for label in labels:
                    label = int(label)
                    true_positive = int(np.sum((group_true == label) & (group_pred == label)))
                    false_positive = int(np.sum((group_true != label) & (group_pred == label)))
                    actual_support = int(np.sum(group_true == label))
                    predicted_support = int(np.sum(group_pred == label))
                    negative_support = n_samples - actual_support
                    probability = ordered.loc[mask, f"prob_class_{label}"].to_numpy(float)
                    definitions = {
                        "positive_prediction_rate": (
                            predicted_support / n_samples if n_samples else math.nan,
                            predicted_support,
                            n_samples,
                            minimum_group_support,
                        ),
                        "true_positive_rate": (
                            true_positive / actual_support if actual_support else math.nan,
                            true_positive,
                            actual_support,
                            minimum_class_denominator,
                        ),
                        "false_positive_rate": (
                            false_positive / negative_support if negative_support else math.nan,
                            false_positive,
                            negative_support,
                            minimum_class_denominator,
                        ),
                        "precision": (
                            true_positive / predicted_support if predicted_support else math.nan,
                            true_positive,
                            predicted_support,
                            minimum_class_denominator,
                        ),
                        "mean_predicted_probability": (
                            float(probability.mean()) if len(probability) else math.nan,
                            float(probability.sum()),
                            n_samples,
                            minimum_group_support,
                        ),
                    }
                    for metric, (value, numerator, denominator, threshold) in definitions.items():
                        denominator_ok = denominator >= threshold
                        rows.append(
                            {
                                **common,
                                "metric": metric,
                                "class_label": label,
                                "metric_value": value,
                                "metric_numerator": numerator,
                                "metric_denominator": int(denominator),
                                "minimum_metric_denominator_threshold": int(threshold),
                                "metric_denominator_eligible": bool(denominator_ok),
                                "eligible_for_gap": bool(
                                    support_ok and denominator_ok and np.isfinite(value)
                                ),
                            }
                        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise FairnessProxyError("No subgroup metric rows were produced.")
    return result


def _fallback_performance_plan(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> ResamplePlan:
    """Construct a strict canonical outer-fold plan for pure-helper tests."""

    first_policy = str(predictions["policy"].drop_duplicates().iloc[0])
    base = predictions[predictions["policy"].astype(str) == first_policy].copy()
    if "outer_fold" not in base:
        raise FairnessProxyError("Subgroup predictions require canonical outer_fold values.")
    protocol = BootstrapProtocol(
        n_resamples=int(n_bootstrap),
        confidence_level=float(confidence_level),
        seed=int(seed),
        strata_columns=("outer_fold", "y_true"),
        method="paired_stratified_percentile",
        quantile_method="linear",
    )
    try:
        return generate_stratified_resample_indices(
            base[["sample_index", "outer_fold", "y_true"]], protocol
        )
    except OOFBootstrapError as exc:
        raise FairnessProxyError(f"Cannot create subgroup bootstrap plan: {exc}") from exc


def _fixed_group_gap_draws(
    predictions: pd.DataFrame,
    group_values: np.ndarray,
    group_metrics: pd.DataFrame,
    *,
    labels: Sequence[int],
    plan: ResamplePlan,
    batch_size: int,
    eligible_groups_by_metric: Mapping[tuple[str, int | None], Sequence[str]] | None = None,
) -> dict[tuple[str, int | None], np.ndarray]:
    """Compute fixed-eligibility gap draws in deterministic memory-bounded batches."""

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise FairnessProxyError("Subgroup bootstrap batch_size must be a positive integer.")
    ordered = predictions.set_index("sample_index", drop=False).loc[
        list(plan.sorted_sample_ids)
    ]
    group_series = pd.Series(group_values, index=predictions["sample_index"].astype(int))
    groups = group_series.loc[list(plan.sorted_sample_ids)].astype(str).to_numpy()
    observed_groups = sorted(set(groups.tolist()))
    group_code_lookup = {value: index for index, value in enumerate(observed_groups)}
    group_codes = np.asarray([group_code_lookup[value] for value in groups], dtype=np.int16)
    y_true_base = ordered["y_true"].astype(int).to_numpy()
    y_pred_base = ordered["y_pred"].astype(int).to_numpy()
    n_draws = plan.indices.shape[0]
    gap_draws: dict[tuple[str, int | None], np.ndarray] = {}

    grouped_rows = group_metrics.groupby(["metric", "class_label"], dropna=False, sort=False)
    for (metric_value, class_value), metric_rows in grouped_rows:
        metric = str(metric_value)
        class_label = None if pd.isna(class_value) else int(class_value)
        key = (metric, class_label)
        if eligible_groups_by_metric is None:
            eligible_groups = sorted(
                metric_rows.loc[
                    metric_rows["eligible_for_gap"].astype(bool), "group_value"
                ]
                .astype(str)
                .unique()
                .tolist()
            )
        else:
            eligible_groups = sorted(
                str(value) for value in eligible_groups_by_metric.get(key, ())
            )
        unknown_groups = sorted(set(eligible_groups).difference(group_code_lookup))
        if unknown_groups:
            raise FairnessProxyError(
                f"Fixed subgroup estimand contains unknown groups: {unknown_groups}."
            )
        if len(eligible_groups) < 2:
            gap_draws[key] = np.full(n_draws, np.nan)
            continue
        gap = np.full(n_draws, np.nan)
        probability_base = (
            ordered[f"prob_class_{class_label}"].to_numpy(float)
            if metric == "mean_predicted_probability" and class_label is not None
            else None
        )
        for start in range(0, n_draws, batch_size):
            stop = min(start + batch_size, n_draws)
            indices = plan.indices[start:stop]
            y_true = y_true_base[indices]
            y_pred = y_pred_base[indices]
            sampled_group_codes = group_codes[indices]
            batch_draws = stop - start
            values: list[np.ndarray] = []
            for group in eligible_groups:
                member = sampled_group_codes == group_code_lookup[group]
                group_n = member.sum(axis=1)
                if class_label is None and metric == "accuracy":
                    numerator = (member & (y_true == y_pred)).sum(axis=1)
                    value = np.divide(
                        numerator,
                        group_n,
                        out=np.full(batch_draws, np.nan),
                        where=group_n > 0,
                    )
                elif class_label is None and metric == "macro_f1":
                    f1_parts: list[np.ndarray] = []
                    for label in labels:
                        true_label = y_true == int(label)
                        pred_label = y_pred == int(label)
                        tp = (member & true_label & pred_label).sum(axis=1)
                        fp = (member & ~true_label & pred_label).sum(axis=1)
                        fn = (member & true_label & ~pred_label).sum(axis=1)
                        denominator = 2 * tp + fp + fn
                        f1_parts.append(
                            np.divide(
                                2 * tp,
                                denominator,
                                out=np.zeros(batch_draws, dtype=float),
                                where=denominator > 0,
                            )
                        )
                    value = np.mean(np.vstack(f1_parts), axis=0)
                    value[group_n == 0] = np.nan
                else:
                    assert class_label is not None
                    true_label = y_true == class_label
                    pred_label = y_pred == class_label
                    tp = (member & true_label & pred_label).sum(axis=1)
                    fp = (member & ~true_label & pred_label).sum(axis=1)
                    actual = (member & true_label).sum(axis=1)
                    predicted = (member & pred_label).sum(axis=1)
                    negative = group_n - actual
                    if metric == "positive_prediction_rate":
                        value = np.divide(
                            predicted,
                            group_n,
                            out=np.full(batch_draws, np.nan),
                            where=group_n > 0,
                        )
                    elif metric == "true_positive_rate":
                        value = np.divide(
                            tp,
                            actual,
                            out=np.full(batch_draws, np.nan),
                            where=actual > 0,
                        )
                    elif metric == "false_positive_rate":
                        value = np.divide(
                            fp,
                            negative,
                            out=np.full(batch_draws, np.nan),
                            where=negative > 0,
                        )
                    elif metric == "precision":
                        value = np.divide(
                            tp,
                            predicted,
                            out=np.full(batch_draws, np.nan),
                            where=predicted > 0,
                        )
                    elif metric == "mean_predicted_probability":
                        assert probability_base is not None
                        probability = probability_base[indices]
                        probability_sum = np.where(member, probability, 0.0).sum(axis=1)
                        value = np.divide(
                            probability_sum,
                            group_n,
                            out=np.full(batch_draws, np.nan),
                            where=group_n > 0,
                        )
                    else:
                        raise FairnessProxyError(f"Unsupported subgroup metric: {metric}")
                values.append(value)
            matrix = np.column_stack(values)
            valid = np.isfinite(matrix).all(axis=1)
            batch_gap = np.full(batch_draws, np.nan)
            batch_gap[valid] = matrix[valid].max(axis=1) - matrix[valid].min(axis=1)
            gap[start:stop] = batch_gap
        gap_draws[key] = gap
    return gap_draws


def _status(
    *,
    n_groups: int,
    valid_fraction: float,
    ci_low: float,
    ci_high: float,
    minimum_valid_fraction: float,
    wide_interval_threshold: float,
) -> str:
    if n_groups < 2:
        return "insufficient_subgroup_or_metric_support"
    if valid_fraction < minimum_valid_fraction:
        return "unstable_insufficient_valid_bootstrap_replicates"
    if np.isfinite(ci_low) and np.isfinite(ci_high) and ci_high - ci_low > wide_interval_threshold:
        return "support_sufficient_but_interval_wide"
    return "support_sufficient_descriptive_estimate"


def compute_subgroup_bootstrap_evidence(
    group_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    *,
    labels: Sequence[int],
    attributes: Sequence[str],
    transforms: Mapping[str, Mapping[str, Any]],
    sensitive_attributes: set[str],
    plan: ResamplePlan,
    confidence_level: float,
    minimum_valid_fraction: float,
    wide_interval_threshold: float,
    batch_size: int = 200,
    conditional_inference_note: str = CONDITIONAL_INFERENCE_NOTE,
) -> SubgroupBootstrapEvidence:
    """Create support-aware gaps and paired policy gap differences."""

    if conditional_inference_note != CONDITIONAL_INFERENCE_NOTE:
        raise FairnessProxyError("Subgroup conditional-inference scope differs from the contract.")
    alpha = 1.0 - float(confidence_level)
    indexed = _sample_indexed_data(data)
    missing = [attribute for attribute in attributes if attribute not in indexed]
    if missing:
        raise FairnessProxyError(f"Configured audit attribute is missing: {missing}")
    draw_map: dict[tuple[str, str, str, int | None], np.ndarray] = {}
    context_map: dict[tuple[str, str], tuple[pd.DataFrame, np.ndarray, pd.DataFrame]] = {}
    interval_rows: list[dict[str, Any]] = []
    identity_columns = [column for column in IDENTITY_FIELDS if column in predictions]
    for policy, scoped in predictions.groupby("policy", sort=False):
        ordered = scoped.sort_values("sample_index").reset_index(drop=True)
        if tuple(ordered["sample_index"].astype(int)) != tuple(plan.sorted_sample_ids):
            raise FairnessProxyError(f"Policy {policy!r} sample order differs from bootstrap plan.")
        sample_indices = ordered["sample_index"].astype(int).to_numpy()
        identity = {column: ordered[column].iloc[0] for column in identity_columns}
        for attribute in attributes:
            values = transform_audit_attribute(
                indexed[attribute], attribute, transforms.get(attribute)
            ).set_axis(indexed.index).loc[sample_indices].to_numpy(dtype=str)
            scoped_metrics = group_metrics[
                (group_metrics["policy"].astype(str) == str(policy))
                & (group_metrics["attribute"].astype(str) == attribute)
            ]
            context_map[(str(policy), attribute)] = (ordered, values, scoped_metrics)
            draws = _fixed_group_gap_draws(
                ordered,
                values,
                scoped_metrics,
                labels=labels,
                plan=plan,
                batch_size=batch_size,
            )
            for (metric, class_label), metric_rows in scoped_metrics.groupby(
                ["metric", "class_label"], dropna=False, sort=False
            ):
                normalized_class = None if pd.isna(class_label) else int(class_label)
                key = (str(policy), attribute, str(metric), normalized_class)
                sample_draws = draws[(str(metric), normalized_class)]
                draw_map[key] = sample_draws
                eligible = metric_rows[metric_rows["eligible_for_gap"].astype(bool)]
                point_values = pd.to_numeric(eligible["metric_value"], errors="coerce").dropna()
                point = (
                    float(point_values.max() - point_values.min())
                    if len(point_values) >= 2
                    else math.nan
                )
                finite = sample_draws[np.isfinite(sample_draws)]
                valid = int(len(finite))
                if valid:
                    low, high = np.quantile(
                        finite,
                        [alpha / 2.0, 1.0 - alpha / 2.0],
                        method="linear",
                    )
                    ci_low, ci_high = float(low), float(high)
                else:
                    ci_low = ci_high = math.nan
                valid_fraction = valid / plan.indices.shape[0]
                estimate_status = _status(
                    n_groups=int(eligible["group_value"].nunique()),
                    valid_fraction=valid_fraction,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    minimum_valid_fraction=minimum_valid_fraction,
                    wide_interval_threshold=wide_interval_threshold,
                )
                interval_rows.append(
                    {
                        **identity,
                        "task_type": PRIMARY_TASK,
                        "analysis_type": "support_aware_subgroup_disparity",
                        "policy": str(policy),
                        "attribute": attribute,
                        "metric": str(metric),
                        "class_label": normalized_class,
                        "gap": point,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "confidence_level": float(confidence_level),
                        "n_groups_total": int(metric_rows["group_value"].nunique()),
                        "n_groups_included": int(eligible["group_value"].nunique()),
                        "included_groups_json": _canonical_json(
                            sorted(eligible["group_value"].astype(str).unique().tolist())
                        ),
                        "minimum_subgroup_support": (
                            int(eligible["n_samples"].min()) if not eligible.empty else 0
                        ),
                        "minimum_group_support_threshold": int(
                            metric_rows["minimum_group_support_threshold"].max()
                        ),
                        "minimum_metric_denominator": (
                            int(eligible["metric_denominator"].min()) if not eligible.empty else 0
                        ),
                        "minimum_metric_denominator_threshold": int(
                            metric_rows["minimum_metric_denominator_threshold"].max()
                        ),
                        "bootstrap_samples_requested": int(plan.indices.shape[0]),
                        "valid_bootstrap_samples": valid,
                        "valid_bootstrap_fraction": valid_fraction,
                        "resample_hash": plan.resample_hash,
                        "bootstrap_method": "paired_stratified_sample_level_percentile",
                        "eligibility_scope": "fixed_from_complete_oof_before_resampling",
                        "estimate_status": estimate_status,
                        "headline_eligible": estimate_status
                        == "support_sufficient_descriptive_estimate",
                        "interpretation_category": _audit_category(attribute, sensitive_attributes),
                        "probability_source": "policy_ablation_raw_uncalibrated_oof_probabilities",
                        "inference_scope": "pointwise_descriptive",
                        "multiplicity_adjustment": "none",
                        "bootstrap_batch_size": int(batch_size),
                        "conditional_inference_note": conditional_inference_note,
                        "limitations": (
                            "Descriptive OOF subgroup audit only; gaps do not establish discrimination, "
                            "fairness, or causality. Class-specific rows use their reported denominators. "
                            "Intervals are pointwise; no multiplicity-adjusted or simultaneous inference."
                        ),
                    }
                )

    intervals = pd.DataFrame(interval_rows).sort_values(
        ["policy", "attribute", "metric", "class_label"], na_position="first"
    ).reset_index(drop=True)
    paired_rows: list[dict[str, Any]] = []

    def metric_rows_for(
        frame: pd.DataFrame, metric: str, class_label: int | None
    ) -> pd.DataFrame:
        class_mask = (
            frame["class_label"].isna()
            if class_label is None
            else pd.to_numeric(frame["class_label"], errors="coerce").eq(class_label)
        )
        return frame[(frame["metric"].astype(str) == metric) & class_mask]

    for policy_a, policy_b in itertools.combinations(REQUIRED_POLICY_COMPARISONS, 2):
        for attribute in attributes:
            context_a = context_map.get((policy_a, attribute))
            context_b = context_map.get((policy_b, attribute))
            if context_a is None or context_b is None:
                continue
            ordered_a, values_a, metrics_a = context_a
            ordered_b, values_b, metrics_b = context_b
            metric_keys = sorted(
                {
                    (
                        str(row.metric),
                        None if pd.isna(row.class_label) else int(row.class_label),
                    )
                    for row in pd.concat([metrics_a, metrics_b], ignore_index=True).itertuples(
                        index=False
                    )
                },
                key=lambda value: (value[0], -1 if value[1] is None else value[1]),
            )
            common_by_metric: dict[tuple[str, int | None], list[str]] = {}
            rows_by_metric: dict[
                tuple[str, int | None], tuple[pd.DataFrame, pd.DataFrame]
            ] = {}
            for metric, class_label in metric_keys:
                rows_a = metric_rows_for(metrics_a, metric, class_label)
                rows_b = metric_rows_for(metrics_b, metric, class_label)
                eligible_a = set(
                    rows_a.loc[rows_a["eligible_for_gap"].astype(bool), "group_value"].astype(str)
                )
                eligible_b = set(
                    rows_b.loc[rows_b["eligible_for_gap"].astype(bool), "group_value"].astype(str)
                )
                common_by_metric[(metric, class_label)] = sorted(eligible_a.intersection(eligible_b))
                rows_by_metric[(metric, class_label)] = (rows_a, rows_b)
            common_draws_a = _fixed_group_gap_draws(
                ordered_a,
                values_a,
                metrics_a,
                labels=labels,
                plan=plan,
                batch_size=batch_size,
                eligible_groups_by_metric=common_by_metric,
            )
            common_draws_b = _fixed_group_gap_draws(
                ordered_b,
                values_b,
                metrics_b,
                labels=labels,
                plan=plan,
                batch_size=batch_size,
                eligible_groups_by_metric=common_by_metric,
            )
            for metric, class_label in metric_keys:
                rows_a, rows_b = rows_by_metric[(metric, class_label)]
                common_groups = common_by_metric[(metric, class_label)]
                common_a = rows_a[rows_a["group_value"].astype(str).isin(common_groups)]
                common_b = rows_b[rows_b["group_value"].astype(str).isin(common_groups)]
                point_a_values = pd.to_numeric(common_a["metric_value"], errors="coerce")
                point_b_values = pd.to_numeric(common_b["metric_value"], errors="coerce")
                if (
                    len(common_groups) >= 2
                    and point_a_values.notna().all()
                    and point_b_values.notna().all()
                ):
                    point_a = float(point_a_values.max() - point_a_values.min())
                    point_b = float(point_b_values.max() - point_b_values.min())
                    point = point_a - point_b
                else:
                    point = math.nan
                draws = common_draws_a[(metric, class_label)] - common_draws_b[
                    (metric, class_label)
                ]
                finite = draws[np.isfinite(draws)]
                valid = int(len(finite))
                low = high = math.nan
                if valid:
                    low, high = np.quantile(
                        finite,
                        [alpha / 2.0, 1.0 - alpha / 2.0],
                        method="linear",
                    )
                valid_fraction = valid / plan.indices.shape[0]
                if len(common_groups) < 2:
                    paired_status = "insufficient_common_subgroup_or_metric_support"
                elif valid_fraction < minimum_valid_fraction:
                    paired_status = "unstable_insufficient_valid_bootstrap_replicates"
                elif np.isfinite(low) and np.isfinite(high) and high - low > wide_interval_threshold:
                    paired_status = "support_sufficient_but_interval_wide"
                else:
                    paired_status = "support_sufficient_descriptive_estimate"
                combined_common = pd.concat([common_a, common_b], ignore_index=True)
                identity = {
                    column: ordered_a[column].iloc[0]
                    for column in identity_columns
                }
                paired_rows.append(
                    {
                        **identity,
                        "task_type": PRIMARY_TASK,
                        "comparison_id": f"{policy_a}__minus__{policy_b}",
                        "policy_a": policy_a,
                        "policy_b": policy_b,
                        "attribute": attribute,
                        "metric": metric,
                        "class_label": class_label,
                        "gap_difference": point,
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "confidence_level": float(confidence_level),
                        "common_eligible_groups_json": _canonical_json(common_groups),
                        "n_common_groups": len(common_groups),
                        "minimum_common_subgroup_support": (
                            int(combined_common["n_samples"].min())
                            if not combined_common.empty
                            else 0
                        ),
                        "minimum_common_metric_denominator": (
                            int(combined_common["metric_denominator"].min())
                            if not combined_common.empty
                            else 0
                        ),
                        "bootstrap_samples_requested": int(plan.indices.shape[0]),
                        "valid_bootstrap_samples": valid,
                        "valid_bootstrap_fraction": valid_fraction,
                        "resample_hash": plan.resample_hash,
                        "bootstrap_method": "paired_stratified_sample_level_percentile",
                        "bootstrap_batch_size": int(batch_size),
                        "eligibility_scope": (
                            "intersection_of_complete_oof_eligible_groups_per_pair_"
                            "attribute_metric_class"
                        ),
                        "paired_estimate_status": paired_status,
                        "estimate_status": paired_status,
                        "headline_eligible": False,
                        "inference_scope": "pointwise_descriptive",
                        "multiplicity_adjustment": "none",
                        "conditional_inference_note": conditional_inference_note,
                        "limitations": (
                            "Pointwise descriptive paired difference over the same fixed common "
                            "group set; no multiplicity-adjusted, causal, discrimination, or "
                            "fairness-guarantee interpretation."
                        ),
                    }
                )
    metadata = MappingProxyType(
        {
            "n_resamples": int(plan.indices.shape[0]),
            "confidence_level": float(confidence_level),
            "strata_columns": ["outer_fold", "y_true"],
            "method": "paired_stratified_percentile",
            "quantile_method": "linear",
            "resample_hash": plan.resample_hash,
            "resample_hash_source": "policy_ablation.bootstrap_metadata.resample_hash",
            "resample_hash_equal_to_benchmark_gate": True,
            "stratum_counts": dict(plan.stratum_counts),
            "eligibility_scope": "fixed_from_complete_oof_before_resampling",
            "paired_policy_common_group_scope": (
                "intersection_of_complete_oof_eligible_groups_per_pair_attribute_metric_class"
            ),
            "bootstrap_batch_size": int(batch_size),
            "inference_scope": "pointwise_descriptive",
            "multiplicity_adjustment": "none",
            "simultaneous_or_familywise_claims_allowed": False,
            "conditional_inference_note": conditional_inference_note,
            "task_type": PRIMARY_TASK,
        }
    )
    return SubgroupBootstrapEvidence(
        intervals=intervals,
        paired_differences=pd.DataFrame(paired_rows),
        draw_map=MappingProxyType(draw_map),
        metadata=metadata,
    )


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
    """Compatibility wrapper returning the canonical interval table."""

    del minimum_group_support, minimum_class_denominator
    plan = _fallback_performance_plan(
        predictions,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
    return compute_subgroup_bootstrap_evidence(
        group_metrics,
        predictions,
        data,
        labels=labels,
        attributes=attributes,
        transforms=transforms,
        sensitive_attributes=sensitive_attributes,
        plan=plan,
        confidence_level=confidence_level,
        minimum_valid_fraction=minimum_valid_fraction,
        wide_interval_threshold=wide_interval_threshold,
        batch_size=200,
    ).intervals


def _one_hot_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def _proxy_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = [column for column in frame if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame if column not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
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
    if not transformers:
        raise FairnessProxyError("Proxy predictor contract contains no features.")
    return ColumnTransformer(transformers, remainder="drop")


def proxy_predictor_frames(
    data: pd.DataFrame,
    settings: Mapping[str, Any],
) -> Dict[str, tuple[pd.DataFrame, list[str], bool]]:
    """Apply all reported policies and always remove the proxy target."""

    target_column = str(settings.get("target", {}).get("column", "PerformanceRating"))
    id_fields = settings.get("governance_fields", {}).get("identifier_fields", ["EmpNumber"])
    id_column = str(id_fields[0])
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
        removed_by_safeguard = proxy_target in X.columns
        X = X.drop(columns=[proxy_target], errors="ignore")
        if proxy_target in X.columns or target_column in X.columns or id_column in X.columns:
            raise FairnessProxyError("Target or identifier remained in proxy predictors.")
        frames[policy] = (X, X.columns.tolist(), removed_by_safeguard)
    alias_frame = frames[ALIAS_POLICY][0]
    primary_frame = frames[PRIMARY_POLICY][0]
    if alias_frame.columns.tolist() != primary_frame.columns.tolist() or not alias_frame.equals(primary_frame):
        raise FairnessProxyError("Declared department-including alias is not equivalent after target removal.")
    return frames


def _proxy_contract_hash(
    *,
    system_id: str,
    source_policy: str,
    frame: pd.DataFrame,
    proxy_target: str,
    dataset_sha256: str,
) -> str:
    payload = {
        "system_id": system_id,
        "source_policy": source_policy,
        "feature_columns": frame.columns.tolist(),
        "feature_dtypes": {column: str(frame[column].dtype) for column in frame},
        "proxy_target": proxy_target,
        "proxy_target_absent": proxy_target not in frame,
        "dataset_sha256": dataset_sha256,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _proxy_pipeline(frame: pd.DataFrame, proxy: Mapping[str, Any], seed: int) -> Pipeline:
    classifier = proxy["classifier"]
    estimator = LogisticRegression(
        solver=str(classifier["solver"]),
        l1_ratio=float(classifier["l1_ratio"]),
        C=float(classifier["C"]),
        class_weight=str(classifier["class_weight"]),
        fit_intercept=bool(classifier["fit_intercept"]),
        max_iter=int(classifier["max_iter"]),
        tol=float(classifier["tol"]),
        random_state=int(seed),
    )
    return Pipeline([("preprocessor", _proxy_preprocessor(frame)), ("classifier", estimator)])


def _proxy_metric_values(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> dict[str, float]:
    recalls = []
    for label in labels:
        actual = y_true == int(label)
        if bool(np.any(actual)):
            recalls.append(float(np.mean(y_pred[actual] == int(label))))
    if not recalls:
        raise FairnessProxyError("Proxy metric evaluation has no observed target classes.")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
        ),
    }


def _proxy_metric_draws(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int],
    indices: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise FairnessProxyError("Proxy bootstrap batch_size must be a positive integer.")
    n_draws = len(indices)
    output = {
        "accuracy": np.empty(n_draws, dtype=float),
        "balanced_accuracy": np.empty(n_draws, dtype=float),
        "macro_f1": np.empty(n_draws, dtype=float),
    }
    for start in range(0, n_draws, batch_size):
        stop = min(start + batch_size, n_draws)
        batch_indices = indices[start:stop]
        truth = y_true[batch_indices]
        prediction = y_pred[batch_indices]
        output["accuracy"][start:stop] = np.mean(truth == prediction, axis=1)
        recalls: list[np.ndarray] = []
        f1s: list[np.ndarray] = []
        for label in labels:
            true_label = truth == int(label)
            pred_label = prediction == int(label)
            tp = np.sum(true_label & pred_label, axis=1)
            actual = np.sum(true_label, axis=1)
            predicted = np.sum(pred_label, axis=1)
            recall = np.divide(
                tp,
                actual,
                out=np.full(stop - start, np.nan),
                where=actual > 0,
            )
            denominator = actual + predicted
            f1 = np.divide(
                2 * tp,
                denominator,
                out=np.zeros(stop - start),
                where=denominator > 0,
            )
            recalls.append(recall)
            f1s.append(f1)
        output["balanced_accuracy"][start:stop] = np.mean(np.vstack(recalls), axis=0)
        output["macro_f1"][start:stop] = np.mean(np.vstack(f1s), axis=0)
    return output


def generate_proxy_oof_evidence(
    data: pd.DataFrame,
    settings: Mapping[str, Any],
    *,
    bundle: XGBoostOOFArtifacts,
    identity: Mapping[str, Any],
    proxy: Mapping[str, Any],
) -> ProxyEvidence:
    """Fit two unique proxy contracts on shared outer folds and bootstrap them pairwise."""

    proxy_task_type = str(proxy["task_type"])
    if proxy_task_type != NOMINAL_MULTICLASS_PROXY_DIAGNOSTIC:
        raise FairnessProxyError("Proxy evidence requires the nominal multiclass task schema.")
    proxy_target = str(proxy["target"])
    performance_target = str(settings["target"]["column"])
    identifier = str(settings["governance_fields"]["identifier_fields"][0])
    forbidden_proxy_inputs = {proxy_target, performance_target, identifier}
    if proxy_target not in data:
        raise FairnessProxyError(f"Proxy target is absent: {proxy_target}")
    frames = proxy_predictor_frames(data, settings)
    for system_id, (frame, _, _) in frames.items():
        leaked = sorted(forbidden_proxy_inputs.intersection(frame.columns))
        if leaked:
            raise FairnessProxyError(
                f"Proxy predictor contract {system_id!r} contains forbidden fields: {leaked}."
            )
    seed = resolve_seed(settings, "fairness")
    batch_size = int(settings["fairness"]["bootstrap_batch_size"])
    target_text = data[proxy_target].astype("string").fillna("__MISSING__").astype(str)
    target_classes = sorted(target_text.unique().tolist())
    if len(target_classes) < 3:
        raise FairnessProxyError("Department proxy target requires at least three observed classes.")
    target_class_counts = {
        value: int((target_text == value).sum()) for value in target_classes
    }
    minimum_target_class_support = min(target_class_counts.values())
    target_lookup = {value: index for index, value in enumerate(target_classes)}
    target = target_text.map(target_lookup).astype(int)
    outer = bundle.folds.outer_assignments.sort_values("sample_index").copy()
    if set(outer["sample_index"].astype(int)) != set(data.index.astype(int)):
        raise FairnessProxyError("Proxy data samples differ from shared-fold samples.")
    outer_support = outer[["sample_index", "outer_fold"]].copy()
    outer_support["proxy_target"] = target_text.loc[
        outer_support["sample_index"].astype(int)
    ].to_numpy()
    outer_support_grid = (
        outer_support.groupby(["outer_fold", "proxy_target"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(1, 11), columns=target_classes, fill_value=0)
    )
    nonzero_outer_support = outer_support_grid.to_numpy(int)
    minimum_nonzero_outer_test_class_support = int(
        nonzero_outer_support[nonzero_outer_support > 0].min()
    )
    zero_support_outer_test_cells = int((nonzero_outer_support == 0).sum())

    contracts: list[dict[str, Any]] = []
    contract_hashes: dict[str, str] = {}
    for system_id in UNIQUE_PROXY_POLICIES:
        frame = frames[system_id][0]
        contract_hash = _proxy_contract_hash(
            system_id=system_id,
            source_policy=system_id,
            frame=frame,
            proxy_target=proxy_target,
            dataset_sha256=str(identity["dataset_sha256"]),
        )
        contract_hashes[system_id] = contract_hash
        contracts.append(
            {
                **identity,
                "task_type": proxy_task_type,
                "system_id": system_id,
                "source_policy": system_id,
                "job_role_retained": system_id == PRIMARY_POLICY,
                "proxy_target": proxy_target,
                "proxy_target_absent_from_predictors": proxy_target not in frame,
                "performance_target_absent_from_predictors": performance_target not in frame,
                "identifier_absent_from_predictors": identifier not in frame,
                "n_features": int(frame.shape[1]),
                "feature_columns_json": _canonical_json(frame.columns.tolist()),
                "predictor_contract_sha256": contract_hash,
            }
        )
    equivalence = pd.DataFrame(
        [
            {
                **identity,
                "task_type": proxy_task_type,
                "reported_policy": ALIAS_POLICY,
                "effective_system_id": PRIMARY_POLICY,
                "fit_performed": False,
                "equivalence_reason": "proxy_target_removed_before_predictor_contract_hashing",
                "predictor_contract_sha256": contract_hashes[PRIMARY_POLICY],
            },
            {
                **identity,
                "task_type": proxy_task_type,
                "reported_policy": PRIMARY_POLICY,
                "effective_system_id": PRIMARY_POLICY,
                "fit_performed": True,
                "equivalence_reason": "unique_job_role_retained_predictor_contract",
                "predictor_contract_sha256": contract_hashes[PRIMARY_POLICY],
            },
            {
                **identity,
                "task_type": proxy_task_type,
                "reported_policy": STRICT_POLICY,
                "effective_system_id": STRICT_POLICY,
                "fit_performed": True,
                "equivalence_reason": "unique_job_role_removed_predictor_contract",
                "predictor_contract_sha256": contract_hashes[STRICT_POLICY],
            },
        ]
    )

    oof_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    associations: list[pd.DataFrame] = []
    for system_id in UNIQUE_PROXY_POLICIES:
        features = frames[system_id][0]
        for outer_fold in range(1, 11):
            test_ids = outer.loc[outer["outer_fold"].astype(int) == outer_fold, "sample_index"].astype(int).tolist()
            train_ids = outer.loc[outer["outer_fold"].astype(int) != outer_fold, "sample_index"].astype(int).tolist()
            if not train_ids or not test_ids or set(train_ids).intersection(test_ids):
                raise FairnessProxyError(f"Invalid shared proxy partition for outer fold {outer_fold}.")
            test_target_counts = {
                value: int((target_text.loc[test_ids] == value).sum())
                for value in target_classes
            }
            pipeline = _proxy_pipeline(features.loc[train_ids], proxy, seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with threadpool_limits(limits=1):
                    pipeline.fit(features.loc[train_ids], target.loc[train_ids])
                    prediction = np.asarray(pipeline.predict(features.loc[test_ids]), dtype=int)
                    unaligned = np.asarray(pipeline.predict_proba(features.loc[test_ids]), dtype=float)
            if caught:
                messages = [f"{item.category.__name__}: {item.message}" for item in caught]
                raise FairnessProxyError(
                    f"Proxy fit emitted warnings for {system_id} fold {outer_fold}: {messages[:3]}"
                )
            if tuple(str(value) for value in pipeline.feature_names_in_) != tuple(features.columns):
                raise FairnessProxyError("Proxy fitted feature order differs from its predictor contract.")
            transformed_names = tuple(
                str(value)
                for value in pipeline.named_steps["preprocessor"].get_feature_names_out()
            )
            if not transformed_names or any(
                forbidden in value
                for value in transformed_names
                for forbidden in forbidden_proxy_inputs
            ):
                raise FairnessProxyError(
                    "Proxy transformed lineage is empty or contains proxy target, performance "
                    "target, or identifier fields."
                )
            transformed_lineage_sha256 = hashlib.sha256(
                _canonical_json(list(transformed_names)).encode("utf-8")
            ).hexdigest()
            classes = [int(value) for value in pipeline.named_steps["classifier"].classes_]
            if set(classes) != set(range(len(target_classes))):
                raise FairnessProxyError("A proxy outer-training fold lacks a target class.")
            probability = np.column_stack(
                [unaligned[:, classes.index(label)] for label in range(len(target_classes))]
            )
            if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
                raise FairnessProxyError("Proxy probabilities are not normalized.")
            if not np.array_equal(np.argmax(probability, axis=1), prediction):
                raise FairnessProxyError("Proxy predictions disagree with probability argmax.")
            metrics = _proxy_metric_values(
                target.loc[test_ids].to_numpy(int), prediction, range(len(target_classes))
            )
            fold_rows.append(
                {
                    **identity,
                    "task_type": proxy_task_type,
                    "system_id": system_id,
                    "outer_fold": outer_fold,
                    "n_train": len(train_ids),
                    "n_test": len(test_ids),
                    "proxy_target_class_counts_json": _canonical_json(test_target_counts),
                    "minimum_proxy_target_class_support_in_test": min(
                        test_target_counts.values()
                    ),
                    "zero_support_proxy_target_classes_in_test": sum(
                        value == 0 for value in test_target_counts.values()
                    ),
                    "predictor_contract_sha256": contract_hashes[system_id],
                    "transformed_feature_count": len(transformed_names),
                    "transformed_lineage_sha256": transformed_lineage_sha256,
                    "proxy_target_absent_from_raw_predictors": proxy_target not in features.columns,
                    "proxy_target_absent_from_transformed_lineage": True,
                    "performance_target_absent_from_transformed_lineage": True,
                    "identifier_absent_from_transformed_lineage": True,
                    "warning_count": 0,
                    **metrics,
                }
            )
            for position, sample_index in enumerate(test_ids):
                row = {
                    **identity,
                    "task_type": proxy_task_type,
                    "system_id": system_id,
                    "source_policy": system_id,
                    "sample_index": sample_index,
                    "outer_fold": outer_fold,
                    "proxy_target": str(target_text.loc[sample_index]),
                    "y_true": int(target.loc[sample_index]),
                    "y_pred": int(prediction[position]),
                    "predictor_contract_sha256": contract_hashes[system_id],
                    "proxy_target_absent_from_predictors": True,
                }
                row.update(
                    {
                        f"prob_class_{label}": float(probability[position, label])
                        for label in range(len(target_classes))
                    }
                )
                oof_rows.append(row)
        association = feature_proxy_associations(features, target_text, random_state=seed).copy()
        configured_watchlist = set(str(value) for value in proxy["watchlist"])
        association["proxy_watchlist"] = association["feature"].astype(str).isin(
            configured_watchlist
        )
        association["watchlist_source"] = "proxy_analysis.watchlist"
        association.insert(0, "system_id", system_id)
        association.insert(0, "task_type", proxy_task_type)
        for offset, (key, value) in enumerate(reversed(list(identity.items()))):
            association.insert(0, key, value)
        associations.append(association)

    oof = pd.DataFrame(oof_rows).sort_values(["system_id", "sample_index"]).reset_index(drop=True)
    probability_columns = {label: f"prob_class_{label}" for label in range(len(target_classes))}
    try:
        alignment = validate_aligned_oof_predictions(
            oof,
            labels=range(len(target_classes)),
            task_type=NOMINAL_MULTICLASS_PROXY_DIAGNOSTIC,
            metrics=PROXY_METRICS,
            probability_columns=probability_columns,
        )
        # Shared-fold validation concerns sample membership and outer-fold
        # ownership.  Proxy ``y_true`` is an encoded EmpDepartment value, not
        # PerformanceRating, so it must not be presented to the shared-fold
        # validator as the performance target.
        validate_consumer_fold_assignments(
            bundle.folds,
            oof.drop(columns=["y_true"]),
            group_columns=("system_id",),
        )
    except (OOFBootstrapError, SharedFoldContractError) as exc:
        raise FairnessProxyError(f"Proxy OOF alignment is invalid: {exc}") from exc
    if alignment["n_systems"] != 2 or alignment["n_samples"] != len(data):
        raise FairnessProxyError("Proxy OOF evidence must contain two exactly-once systems.")
    expected_proxy_target = target.sort_index()
    for system_id, scoped in oof.groupby("system_id", sort=False):
        observed_proxy_target = (
            scoped.set_index("sample_index")["y_true"].astype(int).sort_index()
        )
        if not observed_proxy_target.equals(expected_proxy_target):
            raise FairnessProxyError(
                f"Proxy OOF target for {system_id!r} differs from encoded EmpDepartment."
            )

    # The generic bootstrap utility requires a literal y_true column.  For this
    # task it is an explicit semantic adapter for EmpDepartment, never the
    # performance target.
    base = oof[oof["system_id"].astype(str) == PRIMARY_POLICY][
        ["sample_index", "outer_fold", "y_true", "proxy_target"]
    ].copy()
    evaluation_bootstrap = settings["evaluation"]["bootstrap"]
    conditional_inference_note = str(evaluation_bootstrap["conditional_inference_note"])
    if conditional_inference_note != CONDITIONAL_INFERENCE_NOTE:
        raise FairnessProxyError("Proxy conditional-inference scope differs from the contract.")
    proxy_protocol = BootstrapProtocol(
        n_resamples=REQUIRED_BOOTSTRAP_RESAMPLES,
        confidence_level=float(evaluation_bootstrap["confidence_level"]),
        seed=seed,
        strata_columns=("outer_fold", "y_true"),
        method=str(evaluation_bootstrap["method"]),
        quantile_method=str(evaluation_bootstrap["quantile_method"]),
    )
    try:
        plan = generate_stratified_resample_indices(base, proxy_protocol)
    except OOFBootstrapError as exc:
        raise FairnessProxyError(f"Cannot create proxy bootstrap plan: {exc}") from exc
    alpha = 1.0 - proxy_protocol.confidence_level
    interval_rows: list[dict[str, Any]] = []
    draw_by_system: dict[str, dict[str, np.ndarray]] = {}
    for system_id in UNIQUE_PROXY_POLICIES:
        scoped = oof[oof["system_id"].astype(str) == system_id].set_index("sample_index").loc[
            list(plan.sorted_sample_ids)
        ]
        point = _proxy_metric_values(
            scoped["y_true"].to_numpy(int),
            scoped["y_pred"].to_numpy(int),
            range(len(target_classes)),
        )
        draws = _proxy_metric_draws(
            scoped["y_true"].to_numpy(int),
            scoped["y_pred"].to_numpy(int),
            range(len(target_classes)),
            plan.indices,
            batch_size=batch_size,
        )
        draw_by_system[system_id] = draws
        for metric in PROXY_METRICS:
            values = draws[metric]
            if not np.isfinite(values).all():
                raise FairnessProxyError(f"Proxy bootstrap metric {metric} contains invalid draws.")
            low, high = np.quantile(
                values, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear"
            )
            interval_rows.append(
                    {
                        **identity,
                        "task_type": proxy_task_type,
                        "analysis_type": "department_reconstructability_proxy_risk",
                        "system_id": system_id,
                    "metric": metric,
                    "point_estimate": point[metric],
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "bootstrap_std": float(np.std(values, ddof=1)),
                    "n_samples": len(scoped),
                    "n_resamples": REQUIRED_BOOTSTRAP_RESAMPLES,
                    "n_valid": REQUIRED_BOOTSTRAP_RESAMPLES,
                    "proxy_target_n_classes": len(target_classes),
                    "proxy_target_class_counts_json": _canonical_json(target_class_counts),
                    "minimum_proxy_target_class_support": minimum_target_class_support,
                    "minimum_nonzero_outer_test_class_support": (
                        minimum_nonzero_outer_test_class_support
                    ),
                    "zero_support_outer_test_cells": zero_support_outer_test_cells,
                    "confidence_level": float(proxy_protocol.confidence_level),
                    "resample_hash": plan.resample_hash,
                    "uncertainty_method": "paired_stratified_sample_level_percentile_bootstrap",
                    "quantile_method": proxy_protocol.quantile_method,
                    "bootstrap_batch_size": batch_size,
                    "inference_scope": "pointwise_descriptive",
                    "multiplicity_adjustment": "none",
                    "headline_eligible": False,
                    "conditional_inference_note": conditional_inference_note,
                    "interpretation_category": "proxy_risk_reconstructability_not_causal_use",
                    "limitations": (
                        "Department reconstructability is proxy-risk evidence only; it does not "
                        "establish causal use, discrimination, or fairness. Shared folds are "
                        "performance-stratified, so rare department classes can be absent from "
                        "individual proxy test folds; overall target support is reported."
                    ),
                }
            )
    paired_rows: list[dict[str, Any]] = []
    for metric in PROXY_METRICS:
        difference = draw_by_system[PRIMARY_POLICY][metric] - draw_by_system[STRICT_POLICY][metric]
        low, high = np.quantile(
            difference, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear"
        )
        point_a = next(
            row["point_estimate"]
            for row in interval_rows
            if row["system_id"] == PRIMARY_POLICY and row["metric"] == metric
        )
        point_b = next(
            row["point_estimate"]
            for row in interval_rows
            if row["system_id"] == STRICT_POLICY and row["metric"] == metric
        )
        paired_rows.append(
            {
                **identity,
                "task_type": proxy_task_type,
                "comparison_id": f"{PRIMARY_POLICY}__minus__{STRICT_POLICY}",
                "system_a": PRIMARY_POLICY,
                "system_b": STRICT_POLICY,
                "metric": metric,
                "difference": float(point_a - point_b),
                "ci_low": float(low),
                "ci_high": float(high),
                "n_resamples": REQUIRED_BOOTSTRAP_RESAMPLES,
                "n_valid": REQUIRED_BOOTSTRAP_RESAMPLES,
                "proxy_target_n_classes": len(target_classes),
                "proxy_target_class_counts_json": _canonical_json(target_class_counts),
                "minimum_proxy_target_class_support": minimum_target_class_support,
                "minimum_nonzero_outer_test_class_support": (
                    minimum_nonzero_outer_test_class_support
                ),
                "zero_support_outer_test_cells": zero_support_outer_test_cells,
                "confidence_level": float(proxy_protocol.confidence_level),
                "resample_hash": plan.resample_hash,
                "uncertainty_method": "paired_stratified_sample_level_percentile_bootstrap",
                "quantile_method": proxy_protocol.quantile_method,
                "bootstrap_batch_size": batch_size,
                "inference_scope": "pointwise_descriptive",
                "multiplicity_adjustment": "none",
                "headline_eligible": False,
                "conditional_inference_note": conditional_inference_note,
                "limitations": (
                    "Pointwise paired OOF proxy-risk difference only; no multiplicity-adjusted, "
                    "causal, discrimination, or fairness-guarantee interpretation."
                ),
            }
        )

    fold_metrics = pd.DataFrame(fold_rows)
    descriptive_rows: list[dict[str, Any]] = []
    for system_id, scoped in fold_metrics.groupby("system_id", sort=False):
        for metric in PROXY_METRICS:
            values = scoped[metric].to_numpy(float)
            descriptive_rows.append(
                {
                    **identity,
                    "task_type": proxy_task_type,
                    "system_id": system_id,
                    "metric": metric,
                    "fold_mean": float(values.mean()),
                    "fold_std": float(values.std(ddof=1)),
                    "fold_min": float(values.min()),
                    "fold_max": float(values.max()),
                    "n_folds": len(values),
                    "summary_scope": "descriptive_only_no_population_ci",
                }
            )
    fold_assignment = outer[["sample_index", "outer_fold"]].copy()
    fold_assignment["task_type"] = proxy_task_type
    fold_assignment["proxy_target"] = target_text.loc[fold_assignment["sample_index"]].to_numpy()
    fold_assignment["proxy_target_code"] = target.loc[fold_assignment["sample_index"]].to_numpy()
    for offset, (key, value) in enumerate(reversed(list(identity.items()))):
        fold_assignment.insert(0, key, value)
    fold_assignment_hash = hashlib.sha256(
        fold_assignment[["sample_index", "outer_fold", "proxy_target"]]
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    ).hexdigest()
    fold_assignment["proxy_fold_assignment_sha256"] = fold_assignment_hash
    semantic_adapter = {
        **dict(proxy["bootstrap"]["semantic_strata_adapter"]),
        "proxy_target": proxy_target,
        "target_label_mapping": [
            {"code": code, "value": value} for value, code in target_lookup.items()
        ],
    }
    semantic_adapter_sha256 = hashlib.sha256(
        _canonical_json(semantic_adapter).encode("utf-8")
    ).hexdigest()
    semantic_strata_sha256 = hashlib.sha256(
        base[["sample_index", "outer_fold", "proxy_target", "y_true"]]
        .sort_values("sample_index")
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    ).hexdigest()
    bootstrap_metadata = MappingProxyType(
        {
            "n_resamples": REQUIRED_BOOTSTRAP_RESAMPLES,
            "confidence_level": proxy_protocol.confidence_level,
            "method": proxy_protocol.method,
            "quantile_method": proxy_protocol.quantile_method,
            "semantic_strata_columns": ["outer_fold", "proxy_target"],
            "internal_bootstrap_columns": ["outer_fold", "y_true"],
            "semantic_adapter": semantic_adapter,
            "semantic_adapter_sha256": semantic_adapter_sha256,
            "semantic_strata_sha256": semantic_strata_sha256,
            "resample_hash": plan.resample_hash,
            "stratum_counts": dict(plan.stratum_counts),
            "separate_from_performance_policy_bootstrap": True,
            "proxy_fold_assignment_sha256": fold_assignment_hash,
            "proxy_target_n_classes": len(target_classes),
            "proxy_target_class_counts": target_class_counts,
            "minimum_proxy_target_class_support": minimum_target_class_support,
            "minimum_nonzero_outer_test_class_support": (
                minimum_nonzero_outer_test_class_support
            ),
            "zero_support_outer_test_cells": zero_support_outer_test_cells,
            "bootstrap_batch_size": batch_size,
            "inference_scope": "pointwise_descriptive",
            "multiplicity_adjustment": "none",
            "simultaneous_or_familywise_claims_allowed": False,
            "conditional_inference_note": conditional_inference_note,
            "task_type": proxy_task_type,
        }
    )
    interval_frame = pd.DataFrame(interval_rows)
    paired_frame = pd.DataFrame(paired_rows)
    for column in ("point_estimate", "ci_low", "ci_high"):
        values = interval_frame[column].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
            raise FairnessProxyError(f"Proxy {column} lies outside the metric domain [0, 1].")
    if (interval_frame["ci_low"] > interval_frame["ci_high"]).any():
        raise FairnessProxyError("Proxy interval bounds are reversed.")
    for column in ("difference", "ci_low", "ci_high"):
        values = paired_frame[column].to_numpy(float)
        if not np.isfinite(values).all() or (values < -1.0).any() or (values > 1.0).any():
            raise FairnessProxyError(
                f"Proxy paired {column} lies outside the difference domain [-1, 1]."
            )
    if (paired_frame["ci_low"] > paired_frame["ci_high"]).any():
        raise FairnessProxyError("Proxy paired interval bounds are reversed.")
    return ProxyEvidence(
        fold_assignments=fold_assignment,
        feature_contracts=pd.DataFrame(contracts),
        equivalence=equivalence,
        oof_predictions=oof,
        fold_metrics=fold_metrics,
        descriptive_summary=pd.DataFrame(descriptive_rows),
        metric_intervals=interval_frame,
        paired_differences=paired_frame,
        associations=pd.concat(associations, ignore_index=True),
        label_mapping=MappingProxyType(
            {
                "proxy_target": proxy_target,
                "task_type": proxy_task_type,
                "encoding": "sorted_string_labels_zero_based",
                "labels": [
                    {"code": code, "value": value} for value, code in target_lookup.items()
                ],
                "class_counts": target_class_counts,
                "minimum_class_support": minimum_target_class_support,
            }
        ),
        bootstrap_metadata=bootstrap_metadata,
    )


def _reported_proxy_comparison(
    intervals: pd.DataFrame,
    equivalence: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for receipt in equivalence.itertuples(index=False):
        scoped = intervals[
            intervals["system_id"].astype(str) == str(receipt.effective_system_id)
        ]
        for metric in scoped.itertuples(index=False):
            row = metric._asdict()
            row["reported_policy"] = str(receipt.reported_policy)
            row["effective_system_id"] = str(receipt.effective_system_id)
            row["fit_performed_for_reported_policy"] = bool(receipt.fit_performed)
            row["equivalence_reason"] = str(receipt.equivalence_reason)
            rows.append(row)
    return pd.DataFrame(rows)


def manuscript_fairness_proxy_table(
    disparity: pd.DataFrame,
    proxy_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create one long manuscript-facing table with explicit analysis boundaries."""

    fairness = disparity.copy()
    fairness["estimate"] = fairness["gap"]
    fairness["reported_policy"] = fairness["policy"]
    proxy_rows: list[dict[str, Any]] = []
    if not proxy_summary.empty and {"reported_policy", "point_estimate"}.issubset(proxy_summary.columns):
        for row in proxy_summary.itertuples(index=False):
            proxy_rows.append(
                {
                    "run_id": row.run_id,
                    "config_hash": row.config_hash,
                    "scientific_input_hash": row.scientific_input_hash,
                    "fold_contract_hash": row.fold_contract_hash,
                    "xgboost_model_set_sha256": row.xgboost_model_set_sha256,
                    "dataset_sha256": row.dataset_sha256,
                    "task_type": row.task_type,
                    "analysis_type": "department_reconstructability_proxy_risk",
                    "policy": row.reported_policy,
                    "reported_policy": row.reported_policy,
                    "attribute": "EmpDepartment",
                    "metric": f"proxy_{row.metric}",
                    "class_label": np.nan,
                    "estimate": row.point_estimate,
                    "gap": np.nan,
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                    "confidence_level": row.confidence_level,
                    "bootstrap_samples_requested": row.n_resamples,
                    "valid_bootstrap_samples": row.n_valid,
                    "n_groups_total": row.proxy_target_n_classes,
                    "n_groups_included": row.proxy_target_n_classes,
                    "included_groups_json": _canonical_json(
                        sorted(json.loads(row.proxy_target_class_counts_json))
                    ),
                    "minimum_subgroup_support": row.minimum_proxy_target_class_support,
                    "minimum_metric_denominator": row.minimum_proxy_target_class_support,
                    "proxy_target_class_counts_json": row.proxy_target_class_counts_json,
                    "minimum_nonzero_outer_test_class_support": (
                        row.minimum_nonzero_outer_test_class_support
                    ),
                    "zero_support_outer_test_cells": row.zero_support_outer_test_cells,
                    "resample_hash": row.resample_hash,
                    "estimate_status": "paired_oof_bootstrap_proxy_risk_estimate",
                    "headline_eligible": False,
                    "inference_scope": row.inference_scope,
                    "multiplicity_adjustment": row.multiplicity_adjustment,
                    "conditional_inference_note": row.conditional_inference_note,
                    "interpretation_category": row.interpretation_category,
                    "limitations": row.limitations,
                    "effective_system_id": row.effective_system_id,
                    "fit_performed_for_reported_policy": row.fit_performed_for_reported_policy,
                    "equivalence_reason": row.equivalence_reason,
                }
            )
    return pd.concat([fairness, pd.DataFrame(proxy_rows)], ignore_index=True, sort=False)


def _write_interpretation(
    disparity: pd.DataFrame,
    reported_proxy: pd.DataFrame,
    path: Path,
    *,
    identity: Mapping[str, Any],
) -> None:
    stable = disparity[
        (disparity["policy"].astype(str) == PRIMARY_POLICY)
        & disparity["headline_eligible"].astype(bool)
    ].sort_values("gap", ascending=False)
    lines = [
        "# Canonical Support-Aware Subgroup and Proxy-Risk Diagnostics",
        "",
        f"Run ID: `{identity['run_id']}`  ",
        f"Config hash: `{identity['config_hash']}`  ",
        f"Scientific input hash: `{identity['scientific_input_hash']}`  ",
        f"Shared-fold contract: `{identity['fold_contract_hash']}`",
        "",
        "Performance subgroup rows consume exact raw policy-ablation OOF predictions. This stage fits no performance model. Eligibility is fixed from the complete OOF sample; every interval reports its support, metric denominator, and valid bootstrap count.",
        "",
        CONDITIONAL_INFERENCE_NOTE,
        "",
        "All intervals are pointwise and descriptive. No multiplicity adjustment, familywise control, or simultaneous inference is claimed. The ordering below is an observed-gap ranking selected after estimation and carries no selection-adjusted inference.",
        "",
        "## Descriptive support-qualified primary-policy observed-gap ranking",
        "",
    ]
    if stable.empty:
        lines.append("No primary-policy gap met every support, stability, and interval-width rule.")
    else:
        for row in stable.head(10).itertuples(index=False):
            class_text = "" if pd.isna(row.class_label) else f", class {int(row.class_label)}"
            lines.append(
                f"- {row.attribute}, {row.metric}{class_text}: gap={row.gap:.4f}; "
                f"95% CI [{row.ci_low:.4f}, {row.ci_high:.4f}]; minimum subgroup n="
                f"{row.minimum_subgroup_support}; minimum metric denominator="
                f"{row.minimum_metric_denominator}; valid bootstrap draws="
                f"{row.valid_bootstrap_samples}/{row.bootstrap_samples_requested}."
            )
    lines.extend(["", "## Department reconstructability", ""])
    for row in reported_proxy[
        reported_proxy["metric"].astype(str) == "macro_f1"
    ].itertuples(index=False):
        alias = "alias; no separate fit" if not bool(row.fit_performed_for_reported_policy) else "unique fitted contract"
        lines.append(
            f"- `{row.reported_policy}` ({alias}; effective `{row.effective_system_id}`): "
            f"OOF proxy macro-F1={row.point_estimate:.4f}, 95% paired-bootstrap CI "
            f"[{row.ci_low:.4f}, {row.ci_high:.4f}]; minimum overall department-class "
            f"support={int(row.minimum_proxy_target_class_support)}; zero-support "
            f"outer-test class cells={int(row.zero_support_outer_test_cells)}."
        )
    lines.extend(
        [
            "",
            "## Claim boundaries",
            "",
            "- Subgroup differences are descriptive audit evidence, not proof of discrimination, fairness, legal compliance, or causality.",
            "- Department reconstructability is proxy-risk evidence, not proof that the performance model uses department causally or discriminatorily.",
            "- Removing sensitive, department, or job-role fields does not establish fairness or eliminate indirect proxy information.",
            "- Displayed rankings and all intervals are descriptive and pointwise; they do not support multiplicity-adjusted significance claims.",
            "- The package is research evidence only and must not make autonomous HR decisions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _output_records(paths: Mapping[str, Path], staging: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, path in sorted(paths.items()):
        if name == "metadata":
            continue
        if not path.is_file() or path.stat().st_size <= 0:
            raise FairnessProxyError(f"Generated output {name} is missing or empty.")
        try:
            relative = path.relative_to(staging).as_posix()
        except ValueError as exc:
            raise FairnessProxyError(f"Generated output {name} escapes staging.") from exc
        records.append(
            {
                "name": name,
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return records


def _revalidate_upstreams_before_publish(
    *,
    config_path: str | Path,
    shared_folds_dir: str | Path,
    model_benchmarks_dir: str | Path,
    policy_ablation_dir: str | Path,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    data: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    labels: Sequence[int],
    dataset_sha256: str,
    original_bundle: XGBoostOOFArtifacts,
    original_policy: PolicyEvidence,
) -> None:
    """Block publication if config, data, folds, benchmark or policy evidence changed."""

    try:
        refreshed_config = load_manuscript_config(config_path)
        if canonical_config_hash(refreshed_config) != config_hash:
            raise FairnessProxyError("Canonical config changed during subgroup/proxy evaluation.")
        refreshed_dataset = load_canonical_dataset(config_path, "inx_primary")
        if str(refreshed_dataset.receipt.get("actual_sha256")) != dataset_sha256:
            raise FairnessProxyError("Canonical dataset changed during subgroup/proxy evaluation.")
        refreshed_bundle = read_xgboost_oof_artifacts(
            shared_folds_dir,
            model_benchmarks_dir,
            expected_run_id=run_id,
            expected_config_hash=config_hash,
            expected_scientific_input_hash=scientific_input_hash,
            expected_feature_columns=features.columns.tolist(),
            expected_labels=labels,
        )
        refreshed_identity = {
            "run_id": run_id,
            "config_hash": config_hash,
            "scientific_input_hash": scientific_input_hash,
            "fold_contract_hash": refreshed_bundle.identity.fold_contract_hash,
            "xgboost_model_set_sha256": refreshed_bundle.model_set_sha256,
            "dataset_sha256": dataset_sha256,
        }
        refreshed_policy = read_policy_evidence(
            policy_ablation_dir,
            bundle=refreshed_bundle,
            data=data,
            settings=refreshed_config["manuscript_final"],
            identity=refreshed_identity,
            labels=labels,
        )
    except FairnessProxyError:
        raise
    except Exception as exc:
        raise FairnessProxyError(
            f"Scientific upstream revalidation failed before publication: {type(exc).__name__}: {exc}"
        ) from exc
    if (
        refreshed_bundle.identity != original_bundle.identity
        or refreshed_bundle.model_set_sha256 != original_bundle.model_set_sha256
        or dict(refreshed_bundle.upstream_file_hashes)
        != dict(original_bundle.upstream_file_hashes)
        or dict(refreshed_policy.upstream_file_hashes)
        != dict(original_policy.upstream_file_hashes)
        or refreshed_policy.performance_resample_hash != original_policy.performance_resample_hash
    ):
        raise FairnessProxyError("A scientific upstream changed before subgroup/proxy publication.")


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    shared_folds_dir: str | Path,
    model_benchmarks_dir: str | Path,
    policy_ablation_dir: str | Path,
    output_dir: str | Path,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
) -> Dict[str, Path]:
    """Build the complete canonical subgroup/proxy evidence package."""

    raw_config, settings = _settings(config_path)
    if canonical_config_hash(raw_config) != str(config_hash):
        raise FairnessProxyError("Supplied config_hash differs from canonical config.")
    _require_sha256("config_hash", config_hash)
    _require_sha256("scientific_input_hash", scientific_input_hash)
    if not isinstance(run_id, str) or not run_id.strip():
        raise FairnessProxyError("run_id must be a non-blank string.")
    fairness, proxy = _validate_protocol(settings)
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FairnessProxyError(
            f"Subgroup/proxy output must be absent or an empty builder-owned directory: {output}"
        )

    target_config = settings["target"]
    labels = tuple(int(value) for value in target_config["labels"])
    if labels != (2, 3, 4) or str(target_config["problem_type"]) != PRIMARY_TASK:
        raise FairnessProxyError("Subgroup diagnostics require the canonical 2/3/4 ordinal task.")
    canonical = load_canonical_dataset(config_path, "inx_primary")
    data = _sample_indexed_data(canonical.frame)
    dataset_sha256 = _require_sha256(
        "dataset_sha256", canonical.receipt.get("actual_sha256")
    )
    target_column = str(target_config["column"])
    id_column = str(settings["governance_fields"]["identifier_fields"][0])
    primary_features, excluded = exact_policy_frame(
        data,
        PRIMARY_POLICY,
        settings["feature_policies"]["definitions"][PRIMARY_POLICY],
        target_column=target_column,
        id_column=id_column,
    )
    if tuple(excluded) != tuple(primary_excluded_features(raw_config)):
        raise FairnessProxyError("Primary exclusions drifted from the canonical policy.")
    target = data[target_column].astype(int)
    try:
        declared_folds = read_shared_folds(shared_folds_dir)
        bundle = read_xgboost_oof_artifacts(
            shared_folds_dir,
            model_benchmarks_dir,
            expected_run_id=run_id,
            expected_config_hash=config_hash,
            expected_scientific_input_hash=scientific_input_hash,
            expected_feature_columns=primary_features.columns.tolist(),
            expected_labels=labels,
        )
        validate_xgboost_oof_replay(
            bundle, primary_features, target, labels=labels, probability_atol=1e-12
        )
    except BenchmarkArtifactContractError as exc:
        raise FairnessProxyError(f"Benchmark evidence is incompatible: {exc}") from exc
    except SharedFoldContractError as exc:
        raise FairnessProxyError(f"Shared-fold evidence is incompatible: {exc}") from exc
    if declared_folds.contract != bundle.folds.contract:
        raise FairnessProxyError("Independently read shared-fold evidence differs from benchmark binding.")
    if str(bundle.folds.contract.get("dataset_sha256")) != dataset_sha256:
        raise FairnessProxyError("Shared folds do not bind the canonical dataset bytes.")
    identity: dict[str, Any] = {
        "run_id": run_id,
        "config_hash": config_hash,
        "scientific_input_hash": scientific_input_hash,
        "fold_contract_hash": bundle.identity.fold_contract_hash,
        "xgboost_model_set_sha256": bundle.model_set_sha256,
        "dataset_sha256": dataset_sha256,
    }
    for field_name in IDENTITY_FIELDS[1:]:
        _require_sha256(field_name, identity[field_name])
    policy_evidence = read_policy_evidence(
        policy_ablation_dir,
        bundle=bundle,
        data=data,
        settings=settings,
        identity=identity,
        labels=labels,
    )

    evaluation_bootstrap = settings["evaluation"]["bootstrap"]
    performance_protocol = BootstrapProtocol(
        n_resamples=int(evaluation_bootstrap["n_resamples"]),
        confidence_level=float(evaluation_bootstrap["confidence_level"]),
        seed=resolve_seed(settings, "bootstrap"),
        strata_columns=("outer_fold", "y_true"),
        method=str(evaluation_bootstrap["method"]),
        quantile_method=str(evaluation_bootstrap["quantile_method"]),
    )
    if performance_protocol.n_resamples != REQUIRED_BOOTSTRAP_RESAMPLES:
        raise FairnessProxyError("Performance subgroup bootstrap must use exactly 5,000 draws.")
    base = policy_evidence.oof_predictions[
        policy_evidence.oof_predictions["policy"].astype(str) == ALIAS_POLICY
    ][["sample_index", "outer_fold", "y_true"]]
    try:
        performance_plan = generate_stratified_resample_indices(base, performance_protocol)
    except OOFBootstrapError as exc:
        raise FairnessProxyError(f"Cannot reconstruct performance bootstrap: {exc}") from exc
    if performance_plan.resample_hash != policy_evidence.performance_resample_hash:
        raise FairnessProxyError(
            "Reconstructed subgroup plan differs from policy and benchmark resample evidence."
        )

    governance = settings["governance_fields"]
    sensitive = set(str(value) for value in governance["fairness_sensitive_fields"])
    attributes = [str(value) for value in governance["fairness_audit_fields"]]
    transforms = fairness["attribute_transforms"]
    minimum_support = int(fairness["minimum_group_support"])
    minimum_class_denominator = int(fairness["minimum_class_metric_denominator"])
    group_metrics = compute_group_metric_rows(
        policy_evidence.oof_predictions,
        data,
        labels=labels,
        attributes=attributes,
        transforms=transforms,
        sensitive_attributes=sensitive,
        minimum_group_support=minimum_support,
        minimum_class_denominator=minimum_class_denominator,
    )
    stability = fairness["stability"]
    subgroup = compute_subgroup_bootstrap_evidence(
        group_metrics,
        policy_evidence.oof_predictions,
        data,
        labels=labels,
        attributes=attributes,
        transforms=transforms,
        sensitive_attributes=sensitive,
        plan=performance_plan,
        confidence_level=performance_protocol.confidence_level,
        minimum_valid_fraction=float(stability["minimum_valid_bootstrap_fraction"]),
        wide_interval_threshold=float(stability["wide_interval_threshold"]),
        batch_size=int(fairness["bootstrap_batch_size"]),
        conditional_inference_note=str(
            evaluation_bootstrap["conditional_inference_note"]
        ),
    )
    proxy_evidence = generate_proxy_oof_evidence(
        data, settings, bundle=bundle, identity=identity, proxy=proxy
    )
    if proxy_evidence.bootstrap_metadata["resample_hash"] == policy_evidence.performance_resample_hash:
        raise FairnessProxyError(
            "Proxy-target bootstrap must remain distinct from the performance-task resample plan."
        )
    reported_proxy = _reported_proxy_comparison(
        proxy_evidence.metric_intervals, proxy_evidence.equivalence
    )
    manuscript_table = manuscript_fairness_proxy_table(subgroup.intervals, reported_proxy)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent)
    staging = Path(temporary.name)
    paths: Dict[str, Path] = {
        "oof_predictions": staging / "fairness_oof_predictions.csv",
        "group_support_and_metrics": staging / "fairness_group_support_and_metrics.csv",
        "disparity_uncertainty": staging / "fairness_disparity_uncertainty.csv",
        "subgroup_paired_differences": staging / "fairness_policy_paired_gap_differences.csv",
        "proxy_fold_assignments": staging / "proxy_fold_assignments.csv",
        "proxy_feature_contracts": staging / "proxy_feature_contracts.csv",
        "proxy_equivalence": staging / "proxy_equivalence.csv",
        "proxy_oof_predictions": staging / "proxy_oof_predictions.csv",
        "proxy_fold_metrics": staging / "proxy_fold_metrics.csv",
        "proxy_descriptive_summary": staging / "proxy_descriptive_fold_summary.csv",
        "proxy_metric_intervals": staging / "proxy_metric_intervals.csv",
        "proxy_paired_differences": staging / "proxy_policy_paired_differences.csv",
        "proxy_policy_comparison": staging / "proxy_policy_comparison.csv",
        "proxy_watchlist_associations": staging / "proxy_watchlist_associations.csv",
        "proxy_label_mapping": staging / "proxy_label_mapping.json",
        "performance_bootstrap_metadata": staging / "performance_subgroup_bootstrap_metadata.json",
        "proxy_bootstrap_metadata": staging / "proxy_bootstrap_metadata.json",
        "upstream_input_contract": staging / "upstream_input_contract.json",
        "manuscript_table": staging / "manuscript_fairness_proxy_table.csv",
        "interpretation": staging / "fairness_proxy_interpretation.md",
        "metadata": staging / "stage_metadata.json",
    }
    try:
        policy_evidence.oof_predictions.to_csv(paths["oof_predictions"], index=False, lineterminator="\n")
        group_metrics.to_csv(paths["group_support_and_metrics"], index=False, lineterminator="\n")
        subgroup.intervals.to_csv(paths["disparity_uncertainty"], index=False, lineterminator="\n")
        subgroup.paired_differences.to_csv(paths["subgroup_paired_differences"], index=False, lineterminator="\n")
        proxy_evidence.fold_assignments.to_csv(paths["proxy_fold_assignments"], index=False, lineterminator="\n")
        proxy_evidence.feature_contracts.to_csv(paths["proxy_feature_contracts"], index=False, lineterminator="\n")
        proxy_evidence.equivalence.to_csv(paths["proxy_equivalence"], index=False, lineterminator="\n")
        proxy_evidence.oof_predictions.to_csv(paths["proxy_oof_predictions"], index=False, lineterminator="\n")
        proxy_evidence.fold_metrics.to_csv(paths["proxy_fold_metrics"], index=False, lineterminator="\n")
        proxy_evidence.descriptive_summary.to_csv(paths["proxy_descriptive_summary"], index=False, lineterminator="\n")
        proxy_evidence.metric_intervals.to_csv(paths["proxy_metric_intervals"], index=False, lineterminator="\n")
        proxy_evidence.paired_differences.to_csv(paths["proxy_paired_differences"], index=False, lineterminator="\n")
        reported_proxy.to_csv(paths["proxy_policy_comparison"], index=False, lineterminator="\n")
        proxy_evidence.associations.to_csv(paths["proxy_watchlist_associations"], index=False, lineterminator="\n")
        manuscript_table.to_csv(paths["manuscript_table"], index=False, lineterminator="\n")
        _write_json(
            _identity_bound_mapping(proxy_evidence.label_mapping, identity),
            paths["proxy_label_mapping"],
        )
        _write_json({**dict(subgroup.metadata), **identity}, paths["performance_bootstrap_metadata"])
        _write_json({**dict(proxy_evidence.bootstrap_metadata), **identity}, paths["proxy_bootstrap_metadata"])
        _write_json(
            {
                **identity,
                "performance_resample_hash": policy_evidence.performance_resample_hash,
                "performance_task_type": PRIMARY_TASK,
                "proxy_task_type": NOMINAL_MULTICLASS_PROXY_DIAGNOSTIC,
                "benchmark_upstream_file_hashes": dict(sorted(bundle.upstream_file_hashes.items())),
                "policy_upstream_file_hashes": dict(sorted(policy_evidence.upstream_file_hashes.items())),
            },
            paths["upstream_input_contract"],
        )
        _write_interpretation(
            subgroup.intervals, reported_proxy, paths["interpretation"], identity=identity
        )
        output_records = _output_records(paths, staging)
        _write_json(
            {
                "stage": "subgroup_proxy",
                "status": "complete",
                **identity,
                "performance_task_type": PRIMARY_TASK,
                "proxy_task_type": NOMINAL_MULTICLASS_PROXY_DIAGNOSTIC,
                "policies": list(REQUIRED_POLICY_COMPARISONS),
                "performance_model_refit_in_stage": False,
                "performance_oof_source": "policy_ablation/oof_predictions.csv",
                "performance_probability_source": "raw_uncalibrated_policy_oof",
                "performance_resample_hash": policy_evidence.performance_resample_hash,
                "proxy_unique_fitted_contracts": list(UNIQUE_PROXY_POLICIES),
                "proxy_aliases": {ALIAS_POLICY: PRIMARY_POLICY},
                "proxy_oof_exactly_once_per_sample_per_unique_contract": True,
                "proxy_minimum_target_class_support": int(
                    proxy_evidence.bootstrap_metadata["minimum_proxy_target_class_support"]
                ),
                "conditional_inference_note": CONDITIONAL_INFERENCE_NOTE,
                "proxy_bootstrap": dict(proxy_evidence.bootstrap_metadata),
                "outputs": output_records,
                "claim_boundaries": [
                    "subgroup gaps do not establish discrimination, fairness, or causality",
                    "reconstructability is proxy-risk evidence, not proof of causal or discriminatory use",
                    "sensitive-field removal does not establish fairness",
                    "research evidence must not drive autonomous HR decisions",
                ],
            },
            paths["metadata"],
        )
        _revalidate_upstreams_before_publish(
            config_path=config_path,
            shared_folds_dir=shared_folds_dir,
            model_benchmarks_dir=model_benchmarks_dir,
            policy_ablation_dir=policy_ablation_dir,
            run_id=run_id,
            config_hash=config_hash,
            scientific_input_hash=scientific_input_hash,
            data=data,
            features=primary_features,
            target=target,
            labels=labels,
            dataset_sha256=dataset_sha256,
            original_bundle=bundle,
            original_policy=policy_evidence,
        )
        relative_paths = {name: path.relative_to(staging) for name, path in paths.items()}
        if output.exists():
            output.rmdir()
        atomic_replace_directory(staging, output)
        cleanup_temporary_directory(temporary)
    except Exception as error:
        cleanup_temporary_directory(temporary, primary_error=error)
        raise
    return {name: output / relative for name, relative in relative_paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical support-aware subgroup and proxy-risk evidence."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shared-folds-dir", type=Path, required=True)
    parser.add_argument("--model-benchmarks-dir", type=Path, required=True)
    parser.add_argument("--policy-ablation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--scientific-input-hash", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        run(
            arguments.config,
            shared_folds_dir=arguments.shared_folds_dir,
            model_benchmarks_dir=arguments.model_benchmarks_dir,
            policy_ablation_dir=arguments.policy_ablation_dir,
            output_dir=arguments.output_dir,
            run_id=arguments.run_id,
            config_hash=arguments.config_hash,
            scientific_input_hash=arguments.scientific_input_hash,
        )
    )


__all__ = [
    "FairnessProxyError",
    "PolicyEvidence",
    "ProxyEvidence",
    "SubgroupBootstrapEvidence",
    "compute_group_metric_rows",
    "compute_subgroup_bootstrap_evidence",
    "generate_proxy_oof_evidence",
    "manuscript_fairness_proxy_table",
    "parse_args",
    "proxy_predictor_frames",
    "read_policy_evidence",
    "run",
    "summarize_disparities_with_bootstrap",
    "transform_audit_attribute",
]
