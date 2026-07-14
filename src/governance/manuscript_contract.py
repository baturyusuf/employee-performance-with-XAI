"""Canonical configuration and provenance contract for manuscript evidence.

This module deliberately contains no experiment logic.  It provides the shared
boundary that experiment stages use to resolve feature policies, bind outputs to
one run/configuration identity, and reject stale or incompatible artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.governance.core_figure_contract import (
    CoreFigureContractError,
    validate_core_figure_plan,
)
from src.governance.external_replication_contract import (
    ExternalReplicationContractError,
    validate_external_replication_settings,
    validate_external_replication_side_inputs,
)
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "manuscript_final.yaml"
LEGACY_FEATURE_POLICY_PROJECTION_PATH = PROJECT_ROOT / "configs" / "feature_sets.yaml"
MANIFEST_SCHEMA_VERSION = 3

_PORTABLE_RUN_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,158}[A-Za-z0-9])?\Z")
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)

EXPECTED_EVIDENCE_SCOPE_DATASETS: Mapping[str, frozenset[str]] = {
    "core": frozenset({"inx_primary", "hrdataset_v14"}),
    "supplementary": frozenset(
        {
            "inx_primary",
            "ibm_hr_analytics",
            "ibm_hr_analytics_attrition",
            "employee_turnover",
        }
    ),
}

CORE_PROHIBITED_STAGE_TOKENS = frozenset(
    {"llm", "chatbot", "counterfactual", "ibm", "turnover"}
)

ACTUAL_INPUT_IDENTITY_FIELDS = (
    "dataset_key",
    "physical_dataset_id",
    "actual_path",
    "actual_sha256",
    "row_count",
    "column_count",
    "schema_status",
    "schema_columns",
    "target_column",
    "target_distribution",
    "acquisition_manifest_path",
    "acquisition_manifest_sha256",
    "automatic_download_allowed",
    "source_authenticity_status",
    "licence_verification_status",
)

REQUIRED_POLICY_NAMES = frozenset(
    {
        "full_feature_upper_bound",
        "no_salary_hike",
        "no_salary_hike_no_attrition",
        "no_salary_hike_no_attrition_no_department",
        "no_salary_hike_no_attrition_no_department_no_job_role",
        "no_salary_hike_no_attrition_sensitive_retaining_audit",
    }
)

STRUCTURED_FEATURE_FIELDS = frozenset(
    {
        "feature",
        "feature_name",
        "feature_family",
        "raw_feature",
        "raw_feature_name",
        "grouped_feature",
        "grouped_feature_name",
    }
)


class ManuscriptContractError(ValueError):
    """Base class for canonical manuscript-contract failures."""


class ManuscriptConfigError(ManuscriptContractError):
    """Raised when the canonical configuration is incomplete or inconsistent."""


class FeaturePolicyConsistencyError(ManuscriptContractError):
    """Raised when a module-local feature policy conflicts with the canonical one."""


class ForbiddenFeatureError(ManuscriptContractError):
    """Raised when a primary-model input or artifact names an excluded feature."""


class RunManifestError(ManuscriptContractError):
    """Raised when a run manifest or a referenced artifact is incompatible."""


def validate_portable_run_id(value: Any) -> str:
    """Return one portable, single-component run identifier or fail closed."""

    if (
        not isinstance(value, str)
        or value != value.strip()
        or not _PORTABLE_RUN_ID.fullmatch(value)
    ):
        raise RunManifestError(
            "run_id must be 1-160 portable ASCII characters, start and end with an "
            "alphanumeric character, and contain only alphanumerics, '.', '_' or '-'."
        )
    if value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise RunManifestError("run_id must be one portable path component.")
    if value.split(".", 1)[0].upper() in _WINDOWS_RESERVED_BASENAMES:
        raise RunManifestError("run_id uses a reserved Windows device basename.")
    return value


def utc_now_iso() -> str:
    """Return a second-resolution UTC timestamp with an explicit timezone."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path) -> str:
    """Hash a regular file without loading it into memory."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot hash missing or non-file path: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_document(config: Mapping[str, Any]) -> dict[str, Any]:
    if "manuscript_final" in config:
        return dict(config)
    return {"manuscript_final": dict(config)}


def canonical_config_hash(config_or_path: Mapping[str, Any] | str | Path) -> str:
    """Return a semantic SHA-256 hash of the parsed canonical configuration.

    Hashing canonical JSON rather than source bytes means comments, indentation,
    and mapping order cannot create false run incompatibilities.
    """

    if isinstance(config_or_path, (str, Path)):
        config = load_config(config_or_path)
    elif isinstance(config_or_path, Mapping):
        config = _canonical_document(config_or_path)
    else:  # pragma: no cover - protected by the type contract
        raise TypeError("config_or_path must be a mapping or path")
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manuscript_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the validated inner ``manuscript_final`` settings mapping."""

    settings = config.get("manuscript_final")
    if not isinstance(settings, Mapping):
        raise ManuscriptConfigError("Config must contain a top-level 'manuscript_final' mapping.")
    return settings


def _require_mapping(parent: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ManuscriptConfigError(f"{context}.{key} must be a mapping.")
    return value


def _string_list(value: Any, context: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ManuscriptConfigError(f"{context} must be {qualifier} of strings.")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ManuscriptConfigError(f"{context} must contain only non-empty strings.")
    if len(value) != len(set(value)):
        raise ManuscriptConfigError(f"{context} contains duplicate values.")
    return list(value)


def evidence_scope_contract(
    config: Mapping[str, Any],
    scope_name: str,
) -> dict[str, list[str]]:
    """Return one immutable, config-declared scientific evidence scope.

    Only the fixed ``core`` and ``supplementary`` scopes are accepted.  Dataset
    membership is hard-checked against the author-approved study boundary so a
    caller cannot silently manufacture an arbitrary input subset while using a
    canonical scope name.
    """

    if not isinstance(scope_name, str) or not scope_name.strip():
        raise ManuscriptConfigError("evidence_scope must be a non-empty string.")
    if scope_name not in EXPECTED_EVIDENCE_SCOPE_DATASETS:
        raise ManuscriptConfigError(
            f"Unknown evidence scope {scope_name!r}; allowed scopes are "
            f"{sorted(EXPECTED_EVIDENCE_SCOPE_DATASETS)}."
        )

    settings = manuscript_settings(config)
    scopes = _require_mapping(settings, "evidence_scopes", "manuscript_final")
    if set(scopes) != set(EXPECTED_EVIDENCE_SCOPE_DATASETS):
        raise ManuscriptConfigError(
            "manuscript_final.evidence_scopes must define exactly the immutable "
            f"scopes {sorted(EXPECTED_EVIDENCE_SCOPE_DATASETS)}."
        )
    raw_contract = scopes.get(scope_name)
    if not isinstance(raw_contract, Mapping):
        raise ManuscriptConfigError(f"Evidence scope {scope_name!r} must be a mapping.")

    required_fields = {"dataset_keys", "side_input_keys", "stages"}
    missing = sorted(required_fields - set(raw_contract))
    if missing:
        raise ManuscriptConfigError(
            f"Evidence scope {scope_name!r} is missing required fields: {missing}."
        )
    contract = {
        field: _string_list(
            raw_contract.get(field),
            f"manuscript_final.evidence_scopes.{scope_name}.{field}",
        )
        for field in ("dataset_keys", "side_input_keys", "stages")
    }

    observed_datasets = set(contract["dataset_keys"])
    expected_datasets = set(EXPECTED_EVIDENCE_SCOPE_DATASETS[scope_name])
    if observed_datasets != expected_datasets:
        raise ManuscriptConfigError(
            f"Evidence scope {scope_name!r} has a non-canonical dataset set; "
            f"missing={sorted(expected_datasets - observed_datasets)}, "
            f"extra={sorted(observed_datasets - expected_datasets)}."
        )

    datasets = _require_mapping(settings, "datasets", "manuscript_final")
    unknown_datasets = sorted(observed_datasets - set(datasets))
    if unknown_datasets:
        raise ManuscriptConfigError(
            f"Evidence scope {scope_name!r} references unknown datasets: {unknown_datasets}."
        )
    provenance = _require_mapping(settings, "provenance", "manuscript_final")
    declared_side_inputs = provenance.get("scientific_side_inputs")
    if not isinstance(declared_side_inputs, Mapping) or not declared_side_inputs:
        raise ManuscriptConfigError(
            "manuscript_final.provenance.scientific_side_inputs must be a non-empty mapping."
        )
    unknown_side_inputs = sorted(set(contract["side_input_keys"]) - set(declared_side_inputs))
    if unknown_side_inputs:
        raise ManuscriptConfigError(
            f"Evidence scope {scope_name!r} references unknown side inputs: {unknown_side_inputs}."
        )

    acquisition_manifest = provenance.get("data_acquisition_manifest")
    selected_side_paths = {
        declared_side_inputs[key] for key in contract["side_input_keys"]
    }
    if acquisition_manifest not in selected_side_paths:
        raise ManuscriptConfigError(
            f"Evidence scope {scope_name!r} must include the configured data acquisition manifest."
        )
    missing_schema_mappings: list[str] = []
    for dataset_key in contract["dataset_keys"]:
        definition = datasets.get(dataset_key)
        if not isinstance(definition, Mapping):
            continue
        mapping_path = definition.get("schema_mapping_path")
        if isinstance(mapping_path, str) and mapping_path and mapping_path not in selected_side_paths:
            missing_schema_mappings.append(dataset_key)
    if missing_schema_mappings:
        raise ManuscriptConfigError(
            f"Evidence scope {scope_name!r} omits schema-mapping side inputs for datasets: "
            f"{sorted(missing_schema_mappings)}."
        )

    if scope_name == "core":
        prohibited = {
            stage
            for stage in contract["stages"]
            if any(token in stage.casefold() for token in CORE_PROHIBITED_STAGE_TOKENS)
        }
        if prohibited:
            raise ManuscriptConfigError(
                "Core evidence stages contain prohibited scope dependencies: "
                f"{sorted(prohibited)}."
            )
    return contract


def evidence_scope_contract_hash(
    config: Mapping[str, Any],
    scope_name: str,
) -> str:
    """Return the canonical SHA-256 identity of a validated evidence scope."""

    return _sha256_canonical_json(evidence_scope_contract(config, scope_name))


def feature_policy_definitions(config: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    settings = manuscript_settings(config)
    policies = _require_mapping(settings, "feature_policies", "manuscript_final")
    definitions = _require_mapping(policies, "definitions", "manuscript_final.feature_policies")
    return definitions  # type: ignore[return-value]


def primary_policy_name(config: Mapping[str, Any]) -> str:
    policies = _require_mapping(manuscript_settings(config), "feature_policies", "manuscript_final")
    name = policies.get("primary_policy")
    if not isinstance(name, str) or not name:
        raise ManuscriptConfigError("manuscript_final.feature_policies.primary_policy must be a name.")
    return name


def primary_policy_definition(config: Mapping[str, Any]) -> Mapping[str, Any]:
    name = primary_policy_name(config)
    definitions = feature_policy_definitions(config)
    definition = definitions.get(name)
    if not isinstance(definition, Mapping):
        raise ManuscriptConfigError(f"Primary feature policy is not defined: {name}")
    return definition


def primary_excluded_features(config: Mapping[str, Any]) -> tuple[str, ...]:
    definition = primary_policy_definition(config)
    return tuple(_string_list(definition.get("excluded_features"), "primary excluded_features"))


def _policy_exclusions(definition: Any, context: str) -> set[str]:
    if isinstance(definition, Mapping):
        raw = definition.get("excluded_features", definition.get("drop"))
    else:
        raw = definition
    if not isinstance(raw, (list, tuple, set, frozenset)):
        raise FeaturePolicyConsistencyError(
            f"{context} must be an exclusion sequence or a mapping containing "
            "'excluded_features' (legacy 'drop' is also recognized)."
        )
    values = list(raw)
    if any(not isinstance(value, str) or not value for value in values):
        raise FeaturePolicyConsistencyError(f"{context} contains a non-string feature name.")
    return set(values)


def canonical_policy_mapping(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return all canonical policy exclusions in deterministic order."""

    return {
        name: tuple(definition["excluded_features"])
        for name, definition in sorted(feature_policy_definitions(config).items())
    }


def validate_policy_consistency(
    config: Mapping[str, Any],
    policy_sources: Mapping[str, Mapping[str, Any]],
    *,
    reject_unknown_policies: bool = False,
) -> None:
    """Reject module-local definitions that disagree with canonical policies.

    ``policy_sources`` maps a source/module label to its named policy mapping.
    Values may be exclusion sequences, canonical ``excluded_features`` mappings,
    or legacy ``drop`` mappings.  Scientific stages should call this during
    preflight if they still accept externally supplied policy definitions.
    """

    canonical = {
        name: set(definition["excluded_features"])
        for name, definition in feature_policy_definitions(config).items()
    }
    for source_name, source_policies in policy_sources.items():
        if not isinstance(source_policies, Mapping):
            raise FeaturePolicyConsistencyError(f"Policy source {source_name!r} is not a mapping.")
        for policy_name, source_definition in source_policies.items():
            if policy_name not in canonical:
                if reject_unknown_policies:
                    raise FeaturePolicyConsistencyError(
                        f"Policy source {source_name!r} defines unknown policy {policy_name!r}."
                    )
                continue
            observed = _policy_exclusions(
                source_definition,
                f"policy {policy_name!r} from source {source_name!r}",
            )
            expected = canonical[policy_name]
            if observed != expected:
                missing = sorted(expected - observed)
                extra = sorted(observed - expected)
                raise FeaturePolicyConsistencyError(
                    f"Policy {policy_name!r} from source {source_name!r} conflicts with the "
                    f"canonical definition; missing={missing}, extra={extra}."
                )


def repository_feature_policy_projection(
    path: str | Path = LEGACY_FEATURE_POLICY_PROJECTION_PATH,
) -> Mapping[str, Any]:
    """Load the explicitly labelled legacy policy projection or fail closed.

    The manuscript configuration is the sole canonical policy source.  Older
    repository modules still read ``feature_sets.yaml`` during their staged
    retirement, so any policy name shared with the canonical source must remain
    an exact compatibility projection rather than a second interpretation.
    """

    projection = load_config(path)
    metadata = projection.get("policy_source")
    if not isinstance(metadata, Mapping):
        raise FeaturePolicyConsistencyError(
            "configs/feature_sets.yaml must declare its legacy policy-source status."
        )
    expected_metadata = {
        "status": "legacy_compatibility_projection",
        "canonical_source": "configs/manuscript_final.yaml",
        "same_named_policy_exclusions_must_match": True,
    }
    differences = {
        key: {"expected": expected, "observed": metadata.get(key)}
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if differences:
        raise FeaturePolicyConsistencyError(
            "configs/feature_sets.yaml has invalid legacy projection metadata: "
            + json.dumps(differences, sort_keys=True, ensure_ascii=True)
        )
    policies = projection.get("feature_sets")
    if not isinstance(policies, Mapping):
        raise FeaturePolicyConsistencyError(
            "configs/feature_sets.yaml must define a feature_sets mapping."
        )
    return policies


def validate_manuscript_config(config: Mapping[str, Any]) -> None:
    """Validate required sections and cross-section scientific invariants."""

    settings = manuscript_settings(config)
    if settings.get("schema_version") != 1:
        raise ManuscriptConfigError("manuscript_final.schema_version must be 1.")

    required_sections = {
        "package",
        "datasets",
        "evidence_scopes",
        "external_replication",
        "target",
        "model",
        "feature_policies",
        "governance_fields",
        "fairness",
        "proxy_analysis",
        "evaluation",
        "calibration",
        "shap",
        "counterfactuals",
        "output",
        "seeds",
        "figures",
        "provenance",
    }
    missing_sections = sorted(required_sections - set(settings))
    if missing_sections:
        raise ManuscriptConfigError(f"Canonical config is missing sections: {missing_sections}")
    for section in required_sections:
        _require_mapping(settings, section, "manuscript_final")

    seeds = _require_mapping(settings, "seeds", "manuscript_final")
    if not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds.values()
    ):
        raise ManuscriptConfigError("All canonical protocol seeds must be explicit integers.")
    missing_required_seeds = sorted({"bootstrap", "fairness"}.difference(seeds))
    if missing_required_seeds:
        raise ManuscriptConfigError(
            "Canonical subgroup/proxy seed references are missing: "
            f"{missing_required_seeds}."
        )

    try:
        validate_external_replication_settings(settings)
    except ExternalReplicationContractError as exc:
        raise ManuscriptConfigError(str(exc)) from exc

    package = _require_mapping(settings, "package", "manuscript_final")
    if package.get("autonomous_hr_decisions_allowed") is not False:
        raise ManuscriptConfigError("Canonical package must prohibit autonomous HR decisions.")

    target = _require_mapping(settings, "target", "manuscript_final")
    target_column = target.get("column")
    labels = target.get("labels")
    if not isinstance(target_column, str) or not target_column:
        raise ManuscriptConfigError("manuscript_final.target.column must be a non-empty string.")
    if labels != [2, 3, 4] or target.get("ordering") != [2, 3, 4]:
        raise ManuscriptConfigError("Primary target labels and ordering must be exactly [2, 3, 4].")

    datasets = _require_mapping(settings, "datasets", "manuscript_final")
    if target.get("primary_dataset") not in datasets:
        raise ManuscriptConfigError("target.primary_dataset must reference a configured dataset.")
    for name, definition in datasets.items():
        if not isinstance(definition, Mapping):
            raise ManuscriptConfigError(f"Dataset {name!r} must be a mapping.")
        for required in ("path", "role", "task_type", "target", "allowed_claim"):
            if not isinstance(definition.get(required), str) or not definition.get(required):
                raise ManuscriptConfigError(f"Dataset {name!r} requires non-empty {required!r}.")

    definitions = feature_policy_definitions(config)
    missing_policies = sorted(REQUIRED_POLICY_NAMES - set(definitions))
    if missing_policies:
        raise ManuscriptConfigError(f"Canonical config is missing feature policies: {missing_policies}")
    for name, definition in definitions.items():
        if not isinstance(definition, Mapping):
            raise ManuscriptConfigError(f"Feature policy {name!r} must be a mapping.")
        _string_list(definition.get("excluded_features"), f"feature policy {name!r}.excluded_features")
        if not isinstance(definition.get("role"), str) or not definition.get("role"):
            raise ManuscriptConfigError(f"Feature policy {name!r} requires a non-empty role.")
        if not isinstance(definition.get("audit_only"), bool):
            raise ManuscriptConfigError(f"Feature policy {name!r}.audit_only must be boolean.")

    primary_name = primary_policy_name(config)
    primary = primary_policy_definition(config)
    if primary.get("role") != "canonical_primary" or primary.get("audit_only") is not False:
        raise ManuscriptConfigError("The primary policy must be non-audit and have role 'canonical_primary'.")

    governance = _require_mapping(settings, "governance_fields", "manuscript_final")
    sensitive = set(_string_list(governance.get("fairness_sensitive_fields"), "fairness_sensitive_fields"))
    identifiers = set(_string_list(governance.get("identifier_fields"), "identifier_fields"))
    outcome_fields = set(
        _string_list(governance.get("outcome_or_post_outcome_fields"), "outcome_or_post_outcome_fields")
    )
    proxy = _require_mapping(settings, "proxy_analysis", "manuscript_final")
    proxy_target = proxy.get("target")
    if not isinstance(proxy_target, str) or not proxy_target:
        raise ManuscriptConfigError("proxy_analysis.target must be a non-empty feature name.")

    expected_primary = sensitive | identifiers | outcome_fields | {proxy_target}
    actual_primary = set(primary["excluded_features"])
    if actual_primary != expected_primary:
        raise ManuscriptConfigError(
            f"Primary policy {primary_name!r} must be the single exact union of sensitive, "
            f"identifier, outcome/post-outcome, and proxy-target exclusions; "
            f"missing={sorted(expected_primary - actual_primary)}, "
            f"extra={sorted(actual_primary - expected_primary)}."
        )
    if target_column not in actual_primary:
        raise ManuscriptConfigError("The primary target must be excluded from model features.")

    audit_name = "no_salary_hike_no_attrition_sensitive_retaining_audit"
    audit_definition = definitions[audit_name]
    expected_audit = identifiers | outcome_fields
    if set(audit_definition["excluded_features"]) != expected_audit or audit_definition["audit_only"] is not True:
        raise ManuscriptConfigError(
            f"{audit_name} must be audit-only and drop only identifier, target, salary-hike, and attrition fields."
        )

    governed_base = definitions["no_salary_hike_no_attrition"]
    if set(governed_base["excluded_features"]) != expected_audit | sensitive:
        raise ManuscriptConfigError(
            "no_salary_hike_no_attrition must add all sensitive exclusions to the pure leakage audit policy."
        )
    department_free = definitions["no_salary_hike_no_attrition_no_department"]
    if set(department_free["excluded_features"]) != set(governed_base["excluded_features"]) | {proxy_target}:
        raise ManuscriptConfigError("The canonical department-free policy must add only the proxy target.")
    strict = definitions["no_salary_hike_no_attrition_no_department_no_job_role"]
    strict_extra = set(strict["excluded_features"]) - set(department_free["excluded_features"])
    if strict_extra != {"EmpJobRole"}:
        raise ManuscriptConfigError("The strict policy must add exactly EmpJobRole to the primary exclusions.")

    policy_section = _require_mapping(settings, "feature_policies", "manuscript_final")
    comparison_protocol = _require_mapping(
        policy_section,
        "comparison_protocol",
        "manuscript_final.feature_policies",
    )
    expected_comparison_protocol = {
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
    if dict(comparison_protocol) != expected_comparison_protocol:
        raise ManuscriptConfigError(
            "feature_policies.comparison_protocol differs from the frozen matched-OOF "
            "feature-access sensitivity contract."
        )

    fairness = _require_mapping(settings, "fairness", "manuscript_final")
    expected_fairness = {
        "scope": "support_aware_primary_task_oof_audit",
        "prediction_contract": {
            "required_upstream_stages": [
                "shared_folds",
                "model_benchmarks",
                "policy_ablation",
            ],
            "outer_folds_source": "shared_folds.outer_fold_assignments",
            "oof_predictions_source": "policy_ablation.exact_oof_predictions",
            "primary_model_provenance_source": (
                "model_benchmarks.xgboost_selected_candidate_by_outer_fold"
            ),
            "model_refit_in_stage": False,
            "probability_source": "policy_ablation.raw_uncalibrated_oof_probabilities",
            "probability_semantics": "raw_uncalibrated_for_matched_cross_policy_audit",
        },
        "policy_comparisons_source": "policy_ablation.exact_oof_predictions",
        "protected_or_sensitive_source": "governance_fields.fairness_sensitive_fields",
        "exploratory_operational_source": (
            "governance_fields.fairness_audit_fields_minus_sensitive"
        ),
        "minimum_group_support": 30,
        "minimum_class_metric_denominator": 10,
        "attribute_transforms": {
            "Age": {
                "type": "numeric_bins",
                "edges": [17, 29, 39, 49, 59, 200],
                "labels": ["18-29", "30-39", "40-49", "50-59", "60+"],
            }
        },
        "metrics_by_task": {
            "ordinal_multiclass_performance": {
                "overall": ["accuracy", "macro_f1"],
                "class_specific": [
                    "positive_prediction_rate",
                    "true_positive_rate",
                    "false_positive_rate",
                    "precision",
                    "mean_predicted_probability",
                ],
            }
        },
        "bootstrap_source": "evaluation.bootstrap",
        "bootstrap_stratify_by": ["outer_fold", "y_true"],
        "bootstrap_batch_size": 200,
        "bootstrap_contract": {
            "n_resamples_source": "evaluation.bootstrap.n_resamples",
            "method_source": "evaluation.bootstrap.method",
            "confidence_level_source": "evaluation.bootstrap.confidence_level",
            "quantile_method_source": "evaluation.bootstrap.quantile_method",
            "seed_source": "seeds.bootstrap",
            "same_resamples_across_policies": True,
            "resample_hash_required": True,
            "resample_hash_source": (
                "policy_ablation.bootstrap_metadata.resample_hash"
            ),
            "resample_hash_equality_required_with": (
                "model_benchmarks.baseline_xgboost_gate.resample_hash"
            ),
        },
        "stability": {
            "minimum_valid_bootstrap_fraction": 0.8,
            "wide_interval_threshold": 0.25,
        },
        "support_status_rules": {
            "below_threshold_rows_retained": True,
            "below_threshold_rows_eligible_for_gap": False,
            "class_specific_metrics_use_metric_denominator": True,
            "minimum_two_eligible_groups_for_gap": True,
            "eligibility_scope": "fixed_from_complete_oof_before_resampling",
            "paired_policy_common_group_scope": (
                "intersection_of_complete_oof_eligible_groups_per_pair_"
                "attribute_metric_class"
            ),
            "paired_policy_minimum_common_groups": 2,
            "paired_status_values": [
                "insufficient_common_subgroup_or_metric_support",
                "unstable_insufficient_valid_bootstrap_replicates",
                "support_sufficient_but_interval_wide",
                "support_sufficient_descriptive_estimate",
            ],
            "status_values": [
                "insufficient_subgroup_or_metric_support",
                "unstable_insufficient_valid_bootstrap_replicates",
                "support_sufficient_but_interval_wide",
                "support_sufficient_descriptive_estimate",
            ],
        },
        "headline_rules": {
            "eligible_statuses": ["support_sufficient_descriptive_estimate"],
            "wide_interval_rows_headline_eligible": False,
            "unstable_rows_headline_eligible": False,
            "insufficient_support_rows_headline_eligible": False,
            "paired_policy_rows_headline_eligible": False,
            "require_minimum_subgroup_support_context": True,
            "require_minimum_metric_denominator_context": True,
            "require_valid_bootstrap_context": True,
            "boundary_value_one_requires_explicit_support_context": True,
        },
        "inference_scope": {
            "intervals": "pointwise_descriptive",
            "multiplicity_adjustment": "none",
            "simultaneous_or_familywise_claims_allowed": False,
            "observed_gap_ranking": (
                "descriptive_only_no_selection_adjusted_inference"
            ),
        },
        "claim_boundary": (
            "Subgroup gaps are descriptive OOF audit evidence with support and uncertainty "
            "constraints; they do not establish discrimination, fairness, or causality."
        ),
    }
    if dict(fairness) != expected_fairness:
        raise ManuscriptConfigError(
            "fairness differs from the frozen exact-OOF support/status/headline contract."
        )

    expected_proxy = {
        "task_type": "nominal_multiclass_proxy_diagnostic",
        "target": "EmpDepartment",
        "watchlist": ["EmpJobRole", "EducationBackground", "BusinessTravelFrequency"],
        "interpretation": (
            "Reconstructability is proxy-risk evidence, not proof of causal or "
            "discriminatory use by the performance model."
        ),
        "outer_folds_source": "shared_folds.outer_fold_assignments",
        "classifier": {
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
        },
        "preprocessing": {
            "numeric": "median_imputation_then_standard_scaling",
            "categorical": (
                "most_frequent_imputation_then_dense_one_hot_handle_unknown_ignore"
            ),
            "fit_scope": "outer_training_partition_only",
        },
        "target_removed_from_all_proxy_predictors": True,
        "metrics": ["accuracy", "balanced_accuracy", "macro_f1"],
        "unique_predictor_contracts": {
            "no_salary_hike_no_attrition_no_department": {
                "source_policy": "no_salary_hike_no_attrition_no_department",
                "job_role_retained": True,
                "proxy_target_removed": True,
            },
            "no_salary_hike_no_attrition_no_department_no_job_role": {
                "source_policy": (
                    "no_salary_hike_no_attrition_no_department_no_job_role"
                ),
                "job_role_retained": False,
                "proxy_target_removed": True,
            },
        },
        "policy_aliases": {
            "no_salary_hike_no_attrition": (
                "no_salary_hike_no_attrition_no_department"
            )
        },
        "reported_policy_order": [
            "no_salary_hike_no_attrition",
            "no_salary_hike_no_attrition_no_department",
            "no_salary_hike_no_attrition_no_department_no_job_role",
        ],
        "oof_contract": {
            "exactly_once_per_sample_per_unique_predictor_contract": True,
            "fold_assignment_source": "shared_folds.outer_fold_assignments",
            "proxy_target_absent_from_predictors": True,
        },
        "bootstrap": {
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
        },
        "primary_uncertainty": "paired_sample_level_stratified_percentile_bootstrap",
        "fold_summary_scope": "descriptive_mean_std_min_max_only_no_population_ci",
        "inference_scope": {
            "intervals": "pointwise_descriptive",
            "multiplicity_adjustment": "none",
            "simultaneous_or_familywise_claims_allowed": False,
            "paired_rows_headline_eligible": False,
        },
    }
    if dict(proxy) != expected_proxy:
        raise ManuscriptConfigError(
            "proxy_analysis differs from the frozen shared-fold predictor/bootstrap contract."
        )

    calibration = _require_mapping(settings, "calibration", "manuscript_final")
    expected_calibration = {
        "scope": "canonical_primary_policy_exact_benchmark_model",
        "primary_method": "sigmoid",
        "comparison_systems": ["raw", "sigmoid"],
        "method_selection": "predeclared_not_outer_test_selected",
        "selection_performed": False,
        "training_protocol": "five_fold_cross_fitted_outer_training_only",
        "outer_folds_source": "shared_folds.outer_fold_assignments",
        "inner_folds_source": "shared_folds.inner_fold_assignments",
        "inner_splits": 5,
        "inner_model_parameters_source": (
            "model_benchmarks.xgboost_selected_candidate_by_outer_fold"
        ),
        "inner_model_seed_source": "seeds.model",
        "preprocessing_fit_scope": "inner_development_partition_only",
        "calibrator_fit_scope": "outer_training_cross_fitted_probabilities_only",
        "outer_model_source": (
            "model_benchmarks.persisted_selected_xgboost_outer_fold_pipeline"
        ),
        "outer_model_refit_in_calibration_stage": False,
        "outer_test_probability_source": "model_benchmarks.exact_xgboost_oof_predictions",
        "outer_test_usage": "evaluation_only",
        "outer_test_used_for_tuning_fitting_selection_or_thresholds": False,
        "sigmoid": {
            "algorithm": "one_vs_rest_platt_logit_then_row_renormalize",
            "implementation_dependency": "scikit-learn>=1.8,<1.9",
            "solver": "lbfgs",
            "regularization": "l2_via_l1_ratio_zero",
            "l1_ratio": 0.0,
            "C": 1.0,
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 1e-10,
            "probability_clip": 1e-6,
            "solver_threadpool_limit": 1,
        },
        "label_decision_rule": "argmax_fixed_label_order_2_3_4",
        "threshold_selection": "none",
        "n_bins": 10,
        "uncertainty_source": "evaluation.bootstrap",
        "fold_summary_scope": "descriptive_variability_only_no_population_ci",
        "probability_warning": (
            "Calibrated probabilities are uncertain research outputs and must not be used "
            "as autonomous HR decision thresholds."
        ),
    }
    if dict(calibration) != expected_calibration:
        raise ManuscriptConfigError(
            "calibration differs from the frozen five-inner-fold cross-fitted sigmoid contract."
        )

    evaluation = _require_mapping(settings, "evaluation", "manuscript_final")
    evaluation_bootstrap = _require_mapping(
        evaluation,
        "bootstrap",
        "manuscript_final.evaluation",
    )
    expected_bootstrap_fields = {
        "n_resamples": 5000,
        "confidence_level": 0.95,
        "seed": "bootstrap",
        "method": "paired_stratified_percentile",
        "stratify_by": ["outer_fold", "y_true"],
        "quantile_method": "linear",
        "conditional_inference_note": (
            "Intervals condition on the observed employees and fixed fold/model-training "
            "protocol; they do not estimate model-training instability."
        ),
    }
    observed_bootstrap_fields = {
        key: evaluation_bootstrap.get(key) for key in expected_bootstrap_fields
    }
    if observed_bootstrap_fields != expected_bootstrap_fields:
        raise ManuscriptConfigError(
            "evaluation.bootstrap differs from the frozen 5000-resample paired OOF contract."
        )

    metric_rules = _require_mapping(
        evaluation,
        "metric_applicability",
        "manuscript_final.evaluation",
    )
    for task in (
        "binary_attrition_transfer",
        "binary_turnover_transfer",
        "nominal_multiclass_proxy_diagnostic",
    ):
        task_rules = _require_mapping(metric_rules, task, "metric_applicability")
        not_applicable = set(
            _string_list(task_rules.get("not_applicable"), f"metric_applicability.{task}.not_applicable")
        )
        if "severe_error_rate" not in not_applicable:
            raise ManuscriptConfigError(f"severe_error_rate must be N/A for {task}.")

    output = _require_mapping(settings, "output", "manuscript_final")
    raw_output_root = output.get("root")
    if (
        not isinstance(raw_output_root, str)
        or not raw_output_root
        or raw_output_root != raw_output_root.strip()
        or "\\" in raw_output_root
    ):
        raise ManuscriptConfigError("output.root must be a portable repository-relative path.")
    output_root_path = Path(raw_output_root)
    if (
        output_root_path.is_absolute()
        or output_root_path.drive
        or raw_output_root in {".", ".."}
        or any(part in {".", ".."} for part in output_root_path.parts)
    ):
        raise ManuscriptConfigError("output.root must remain below the repository root.")
    if output.get("manifest_filename") != "run_manifest.json":
        raise ManuscriptConfigError(
            "output.manifest_filename is fixed to the portable leaf 'run_manifest.json'."
        )
    if output.get("run_directory_template") != "{root}/{run_id}/{evidence_scope}":
        raise ManuscriptConfigError(
            "output.run_directory_template must bind root, run_id, and evidence_scope exactly."
        )
    if output.get("latest_pointer") != f"{raw_output_root}/latest":
        raise ManuscriptConfigError(
            "output.latest_pointer must be the pointer-only '<output.root>/latest' path."
        )

    scope_contracts = {
        scope_name: evidence_scope_contract(config, scope_name)
        for scope_name in EXPECTED_EVIDENCE_SCOPE_DATASETS
    }
    raw_scopes = _require_mapping(settings, "evidence_scopes", "manuscript_final")
    raw_core_scope = _require_mapping(
        raw_scopes,
        "core",
        "manuscript_final.evidence_scopes",
    )
    try:
        validate_core_figure_plan(
            _require_mapping(settings, "figures", "manuscript_final"),
            core_stages=scope_contracts["core"]["stages"],
            core_scope_release_ready=raw_core_scope.get("release_ready"),
            core_scope_blocking_reason=raw_core_scope.get("blocking_reason"),
        )
    except CoreFigureContractError as exc:
        raise ManuscriptConfigError(str(exc)) from exc


def load_manuscript_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    policy_sources: Mapping[str, Mapping[str, Any]] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the canonical manuscript configuration."""

    resolved_path = Path(path).resolve()
    if project_root is not None:
        validation_root = Path(project_root).resolve()
    else:
        try:
            resolved_path.relative_to(PROJECT_ROOT.resolve())
            validation_root = PROJECT_ROOT.resolve()
        except ValueError:
            validation_root = (
                resolved_path.parent.parent
                if resolved_path.parent.name.casefold() == "configs"
                else resolved_path.parent
            )
    data = load_config(resolved_path)
    validate_manuscript_config(data)
    raw_output_root = manuscript_settings(data)["output"]["root"]
    try:
        _resolve_portable_reference(
            raw_output_root,
            validation_root,
            context="manuscript_final.output.root",
        )
    except RunManifestError as exc:
        raise ManuscriptConfigError(str(exc)) from exc
    validate_policy_consistency(
        data,
        {"configs/feature_sets.yaml legacy projection": repository_feature_policy_projection()},
    )
    if policy_sources:
        validate_policy_consistency(data, policy_sources)
    try:
        validate_external_replication_side_inputs(
            manuscript_settings(data),
            project_root=validation_root,
        )
    except ExternalReplicationContractError as exc:
        raise ManuscriptConfigError(str(exc)) from exc
    return data


def _candidate_matches_raw_feature(candidate: str, raw_feature: str) -> bool:
    text = candidate.strip().strip("'\"")
    # sklearn commonly prefixes transformed columns with ``transformer__``.
    base = text.rsplit("__", 1)[-1]
    base_folded = base.casefold()
    raw_folded = raw_feature.casefold()
    if base_folded == raw_folded:
        return True
    return any(base_folded.startswith(raw_folded + separator) for separator in ("_", "=", "[", ":", "-", " "))


def forbidden_feature_mentions(
    feature_names: Iterable[Any],
    config: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Map forbidden raw feature families to observed raw/encoded names."""

    forbidden = primary_excluded_features(config)
    found: dict[str, set[str]] = {}
    for value in feature_names:
        if value is None:
            continue
        candidate = str(value).strip()
        if not candidate:
            continue
        for raw_feature in forbidden:
            if _candidate_matches_raw_feature(candidate, raw_feature):
                found.setdefault(raw_feature, set()).add(candidate)
    return {name: sorted(values) for name, values in sorted(found.items())}


def validate_primary_feature_names(
    feature_names: Iterable[Any],
    config: Mapping[str, Any],
    *,
    context: str = "primary model input",
) -> None:
    mentions = forbidden_feature_mentions(feature_names, config)
    if mentions:
        raise ForbiddenFeatureError(f"Forbidden feature families appear in {context}: {mentions}")


def _collect_json_feature_values(value: Any, feature_fields: set[str]) -> list[Any]:
    collected: list[Any] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in feature_fields:
                if isinstance(nested, (list, tuple, set)):
                    collected.extend(nested)
                elif not isinstance(nested, Mapping):
                    collected.append(nested)
            collected.extend(_collect_json_feature_values(nested, feature_fields))
    elif isinstance(value, list):
        for nested in value:
            collected.extend(_collect_json_feature_values(nested, feature_fields))
    return collected


def artifact_feature_names(
    artifact_path: str | Path,
    *,
    feature_fields: Iterable[str] = STRUCTURED_FEATURE_FIELDS,
    scan_text: bool = False,
    forbidden_candidates: Iterable[str] = (),
) -> list[Any]:
    """Extract structured feature names (or explicit text mentions) from an artifact."""

    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Primary artifact is missing: {path}")
    fields = {field.casefold() for field in feature_fields}
    suffix = path.suffix.casefold()
    values: list[Any] = []

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames:
                selected = [name for name in reader.fieldnames if name.casefold() in fields]
                for row in reader:
                    values.extend(row.get(name) for name in selected)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        values.extend(_collect_json_feature_values(payload, fields))
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ForbiddenFeatureError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
                values.extend(_collect_json_feature_values(payload, fields))
    elif not scan_text:
        raise ForbiddenFeatureError(
            f"Cannot structurally inspect {path.suffix or 'extensionless'} artifact {path}; "
            "set scan_text=True for a text artifact."
        )

    if scan_text:
        text = path.read_text(encoding="utf-8")
        for candidate in forbidden_candidates:
            pattern = rf"(?<![A-Za-z0-9]){re.escape(candidate)}(?![A-Za-z0-9])"
            if re.search(pattern, text, flags=re.IGNORECASE):
                values.append(candidate)
    return values


def validate_artifact_forbidden_features(
    artifact_path: str | Path,
    config: Mapping[str, Any],
    *,
    feature_fields: Iterable[str] = STRUCTURED_FEATURE_FIELDS,
    scan_text: bool = False,
) -> None:
    """Reject a structured primary-model artifact containing excluded features."""

    forbidden = primary_excluded_features(config)
    names = artifact_feature_names(
        artifact_path,
        feature_fields=feature_fields,
        scan_text=scan_text,
        forbidden_candidates=forbidden,
    )
    validate_primary_feature_names(names, config, context=str(artifact_path))


def _run_git(project_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _scoped_git_status(
    project_root: Path,
    allowed_untracked_roots: Sequence[str | Path],
) -> tuple[str, list[str]]:
    """Return tracked/disallowed-untracked status plus declared evidence exclusions."""

    root = project_root.resolve()
    allowed: list[Path] = []
    portable_allowed: list[str] = []
    for raw in allowed_untracked_roots:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise RunManifestError("Allowed untracked evidence root escapes the repository.") from exc
        if not relative.parts:
            raise RunManifestError("The repository root cannot be an allowed untracked evidence root.")
        allowed.append(candidate)
        portable_allowed.append(relative.as_posix())
    if len(set(portable_allowed)) != len(portable_allowed):
        raise RunManifestError("Allowed untracked evidence roots contain duplicates.")

    tracked = _run_git(root, "status", "--porcelain", "--untracked-files=no")
    if tracked == "unavailable":
        return "unavailable", sorted(portable_allowed)
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", sorted(portable_allowed)
    disallowed: list[str] = []
    for relative in (value for value in completed.stdout.split("\0") if value):
        candidate = (root / relative).resolve()
        if not any(
            candidate == allowed_root or allowed_root in candidate.parents
            for allowed_root in allowed
        ):
            disallowed.append(relative.replace("\\", "/"))
    status_rows = [value for value in tracked.splitlines() if value]
    status_rows.extend(f"?? {value}" for value in sorted(disallowed))
    return "\n".join(status_rows), sorted(portable_allowed)


def source_tree_hash(
    project_root: str | Path,
    *,
    roots: Sequence[str] = ("src", "configs"),
    files: Sequence[str] = ("requirements.txt", "requirements-dev.txt"),
) -> str:
    """Hash experiment/config source content independently of Git state."""

    root = Path(project_root).resolve()
    candidates: set[Path] = set()
    for relative_root in roots:
        directory = root / relative_root
        if directory.is_dir():
            candidates.update(
                path
                for path in directory.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
            )
    for relative_file in files:
        path = root / relative_file
        if path.is_file():
            candidates.add(path)

    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _package_versions(package_names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    for package_name in package_names:
        if package_name.casefold() == "python":
            continue
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = "not_installed"
    return versions


def _resolve_from_root(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise RunManifestError(
            f"Scientific manifest paths must remain inside the project root: {resolved}"
        ) from exc


def _resolve_portable_reference(
    raw_path: Any,
    project_root: Path,
    *,
    context: str,
) -> Path:
    """Resolve a manifest/config reference only when it is repository-relative."""

    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RunManifestError(f"{context} must be a non-empty repository-relative path.")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise RunManifestError(f"{context} must not be absolute: {raw_path!r}")
    resolved = (project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise RunManifestError(f"{context} escapes the project root: {raw_path!r}") from exc
    return resolved


def _sha256_canonical_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def declared_side_input_hashes(
    config: Mapping[str, Any],
    *,
    side_input_keys: Sequence[str] | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, dict[str, Any]]:
    """Hash every explicitly declared non-dataset scientific input.

    Side inputs are declared as ``logical_name: repository/relative/path`` in
    ``manuscript_final.provenance.scientific_side_inputs``.  Requiring an
    explicit non-empty mapping prevents configuration, schema mapping, feature
    taxonomy, or search-space changes from bypassing the run/cache identity.
    """

    root = Path(project_root).resolve()
    provenance = _require_mapping(manuscript_settings(config), "provenance", "manuscript_final")
    declared = provenance.get("scientific_side_inputs")
    if not isinstance(declared, Mapping) or not declared:
        raise ManuscriptConfigError(
            "manuscript_final.provenance.scientific_side_inputs must be a non-empty "
            "logical-name to repository-relative path mapping."
        )

    selected = list(side_input_keys) if side_input_keys is not None else list(declared)
    if not selected or any(not isinstance(name, str) or not name for name in selected):
        raise ManuscriptConfigError("side_input_keys must be a non-empty sequence of names.")
    if len(selected) != len(set(selected)):
        raise ManuscriptConfigError("side_input_keys contains duplicate names.")
    unknown = sorted(set(selected) - set(declared))
    if unknown:
        raise ManuscriptConfigError(f"Unknown declared scientific side inputs: {unknown}.")

    records: dict[str, dict[str, Any]] = {}
    for raw_name in sorted(selected):
        raw_path = declared[raw_name]
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ManuscriptConfigError("Scientific side-input names must be non-empty strings.")
        try:
            path = _resolve_portable_reference(
                raw_path,
                root,
                context=f"scientific side input {raw_name!r}",
            )
        except RunManifestError as exc:
            raise ManuscriptConfigError(str(exc)) from exc
        if not path.is_file():
            raise RunManifestError(f"Declared scientific side input is missing for {raw_name!r}: {path}")
        records[raw_name] = {
            "path": _portable_path(path, root),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return records


def scientific_input_hash(
    *,
    config_hash: str,
    scope_contract_hash: str,
    dataset_hashes: Mapping[str, Any],
    side_input_hashes: Mapping[str, Any],
) -> str:
    """Bind config, actual datasets, and declared side inputs to one identity."""

    if not re.fullmatch(r"[0-9a-f]{64}", config_hash):
        raise RunManifestError("config_hash must be a lowercase SHA-256 digest.")
    if not re.fullmatch(r"[0-9a-f]{64}", scope_contract_hash):
        raise RunManifestError("scope_contract_hash must be a lowercase SHA-256 digest.")
    if not isinstance(dataset_hashes, Mapping) or not dataset_hashes:
        raise RunManifestError("dataset_hashes must be a non-empty mapping.")
    if not isinstance(side_input_hashes, Mapping) or not side_input_hashes:
        raise RunManifestError("side_input_hashes must be a non-empty mapping.")
    return _sha256_canonical_json(
        {
            "config_hash": config_hash,
            "scope_contract_hash": scope_contract_hash,
            "dataset_hashes": dict(dataset_hashes),
            "side_input_hashes": dict(side_input_hashes),
        }
    )


def _actual_receipt_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable scientific identity subset of a loader receipt."""

    return {field: receipt.get(field) for field in ACTUAL_INPUT_IDENTITY_FIELDS}


def make_run_id(config: Mapping[str, Any], config_hash: str, *, timestamp: datetime | None = None) -> str:
    settings = manuscript_settings(config)
    package = _require_mapping(settings, "package", "manuscript_final")
    prefix = str(package.get("run_id_prefix", "manuscript_final"))
    safe_prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", prefix).strip("_") or "manuscript_final"
    when = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return validate_portable_run_id(
        f"{safe_prefix}_{when.strftime('%Y%m%dT%H%M%SZ')}_{config_hash[:12]}"
    )


def create_run_manifest(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    evidence_scope: str,
    project_root: str | Path = PROJECT_ROOT,
    run_id: str | None = None,
    dataset_paths: Mapping[str, str | Path] | None = None,
    allow_dataset_download: bool = False,
    initial_command: str | None = None,
    allowed_untracked_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Create a schema-v2 manifest from verified actual and side inputs.

    ``dataset_paths`` remains as a compatibility assertion only: callers may
    provide it, but it must exactly match every path in the named immutable
    evidence scope.  It cannot override the canonical loader contract.
    """

    root = Path(project_root).resolve()
    if run_id is not None:
        run_id = validate_portable_run_id(run_id)
    raw_config_path = Path(config_path)
    rooted_config_path = root / raw_config_path if not raw_config_path.is_absolute() else raw_config_path
    resolved_config_path = rooted_config_path.resolve()
    if not resolved_config_path.is_file():
        raise RunManifestError(f"Canonical config is missing: {resolved_config_path}")
    _portable_path(resolved_config_path, root)
    config = load_manuscript_config(resolved_config_path, project_root=root)
    config_hash = canonical_config_hash(config)
    settings = manuscript_settings(config)
    scope_contract = evidence_scope_contract(config, evidence_scope)
    scope_contract_hash = evidence_scope_contract_hash(config, evidence_scope)

    datasets = _require_mapping(settings, "datasets", "manuscript_final")
    configured_paths = {
        name: str(definition["path"])
        for name in scope_contract["dataset_keys"]
        if isinstance((definition := datasets.get(name)), Mapping)
    }
    if set(configured_paths) != set(scope_contract["dataset_keys"]):
        raise ManuscriptConfigError("Every scoped dataset must define an explicit path.")
    if dataset_paths is not None:
        if set(dataset_paths) != set(configured_paths):
            raise RunManifestError(
                "dataset_paths must name exactly every dataset in the evidence scope."
            )
        mismatches = {
            name: {"configured": configured_paths[name], "supplied": str(dataset_paths[name])}
            for name in configured_paths
            if _resolve_from_root(configured_paths[name], root)
            != _resolve_from_root(dataset_paths[name], root)
        }
        if mismatches:
            raise RunManifestError(
                "dataset_paths cannot override canonical configured paths: "
                f"{mismatches}"
            )

    provenance = _require_mapping(settings, "provenance", "manuscript_final")
    acquisition_manifest_path = provenance.get("data_acquisition_manifest")
    if not isinstance(acquisition_manifest_path, str) or not acquisition_manifest_path:
        raise ManuscriptConfigError(
            "manuscript_final.provenance.data_acquisition_manifest must be a non-empty path."
        )
    side_input_hashes = declared_side_input_hashes(
        config,
        side_input_keys=scope_contract["side_input_keys"],
        project_root=root,
    )

    try:
        from src.data.canonical_loader import verify_configured_datasets

        verified = verify_configured_datasets(
            resolved_config_path,
            acquisition_manifest_path,
            dataset_keys=scope_contract["dataset_keys"],
            allow_download=allow_dataset_download,
            project_root=root,
        )
    except Exception as exc:
        raise RunManifestError(f"Canonical dataset verification failed: {exc}") from exc

    actual_input_receipts: dict[str, dict[str, Any]] = {}
    dataset_hashes: dict[str, dict[str, Any]] = {}
    for name in scope_contract["dataset_keys"]:
        loaded = verified.get(name)
        if loaded is None or not isinstance(loaded.receipt, Mapping):
            raise RunManifestError(f"Canonical loader returned no receipt for dataset {name!r}.")
        receipt = dict(loaded.receipt)
        raw_actual_path = receipt.get("actual_path")
        path = _resolve_portable_reference(
            raw_actual_path,
            root,
            context=f"actual input receipt {name!r}.actual_path",
        )
        if not path.is_file():
            raise RunManifestError(f"Verified actual dataset disappeared for {name!r}: {path}")
        actual_hash = sha256_file(path)
        if receipt.get("actual_sha256") != actual_hash:
            raise RunManifestError(
                f"Loader receipt hash mismatch for dataset {name!r}: "
                f"receipt={receipt.get('actual_sha256')}, actual={actual_hash}"
            )
        receipt["actual_path"] = _portable_path(path, root)
        receipt["size_bytes"] = path.stat().st_size
        actual_input_receipts[name] = receipt
        definition = datasets.get(name, {})
        dataset_hashes[name] = {
            "path": _portable_path(path, root),
            "sha256": actual_hash,
            "size_bytes": path.stat().st_size,
            "row_count": receipt.get("row_count"),
            "column_count": receipt.get("column_count"),
            "schema_status": receipt.get("schema_status"),
            "target_column": receipt.get("target_column"),
            "target_distribution": receipt.get("target_distribution"),
            "role": definition.get("role", "") if isinstance(definition, Mapping) else "",
            "task_type": definition.get("task_type", "") if isinstance(definition, Mapping) else "",
        }

    package_names = provenance.get("package_names", [])
    if not isinstance(package_names, list):
        raise ManuscriptConfigError("provenance.package_names must be a list.")

    git_commit = _run_git(root, "rev-parse", "HEAD")
    if allowed_untracked_roots:
        git_status, portable_allowed_untracked = _scoped_git_status(
            root,
            allowed_untracked_roots,
        )
    else:
        git_status = _run_git(root, "status", "--porcelain", "--untracked-files=all")
        portable_allowed_untracked = []
    if provenance.get("git_commit_required") is True and git_commit == "unavailable" and root == PROJECT_ROOT.resolve():
        raise RunManifestError("A Git commit is required but could not be resolved.")

    manifest: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id or make_run_id(config, config_hash),
        "git_commit": git_commit,
        "git_worktree_dirty": bool(git_status and git_status != "unavailable"),
        "git_status_sha256": (
            hashlib.sha256(git_status.encode("utf-8")).hexdigest()
            if git_status != "unavailable"
            else "unavailable"
        ),
        "preexisting_evidence_roots_ignored_for_clean_start": portable_allowed_untracked,
        "source_tree_hash": source_tree_hash(root),
        "config_path": _portable_path(resolved_config_path, root),
        "config_hash": config_hash,
        "evidence_scope": evidence_scope,
        "scope_contract": scope_contract,
        "scope_contract_hash": scope_contract_hash,
        "actual_input_receipts": actual_input_receipts,
        "dataset_hashes": dataset_hashes,
        "side_input_hashes": side_input_hashes,
        "scientific_input_hash": scientific_input_hash(
            config_hash=config_hash,
            scope_contract_hash=scope_contract_hash,
            dataset_hashes=dataset_hashes,
            side_input_hashes=side_input_hashes,
        ),
        "code_package_versions": _package_versions(package_names),
        "start_timestamp": utc_now_iso(),
        "end_timestamp": None,
        "random_seeds": dict(_require_mapping(settings, "seeds", "manuscript_final")),
        "commands": [],
        "output_files": [],
        "status": "running",
        "failure_information": [],
    }
    if initial_command:
        record_command(manifest, initial_command, stage="entrypoint", status="started")
    return manifest


def record_command(
    manifest: MutableMapping[str, Any],
    command: str,
    *,
    stage: str,
    status: str,
    started_at: str | None = None,
    ended_at: str | None = None,
    return_code: int | None = None,
) -> dict[str, Any]:
    if not isinstance(command, str) or not command.strip():
        raise RunManifestError("A recorded command must be a non-empty string.")
    commands = manifest.setdefault("commands", [])
    if not isinstance(commands, list):
        raise RunManifestError("manifest.commands must be a list.")
    record = {
        "command": command,
        "stage": stage,
        "status": status,
        "started_at": started_at or utc_now_iso(),
        "ended_at": ended_at,
        "return_code": return_code,
    }
    commands.append(record)
    return record


def register_artifact(
    manifest: MutableMapping[str, Any],
    artifact_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    stage: str,
    artifact_type: str,
    artifact_run_id: str | None = None,
    artifact_config_hash: str | None = None,
) -> dict[str, Any]:
    """Hash and register one artifact while enforcing its run/config identity."""

    root = Path(project_root).resolve()
    path = _resolve_from_root(artifact_path, root)
    if not path.is_file():
        raise RunManifestError(f"Cannot register missing artifact: {path}")

    expected_run_id = manifest.get("run_id")
    expected_config_hash = manifest.get("config_hash")
    observed_run_id = artifact_run_id or expected_run_id
    observed_config_hash = artifact_config_hash or expected_config_hash
    if observed_run_id != expected_run_id:
        raise RunManifestError(
            f"Artifact {path} has run_id={observed_run_id!r}; expected {expected_run_id!r}."
        )
    if observed_config_hash != expected_config_hash:
        raise RunManifestError(
            f"Artifact {path} has config_hash={observed_config_hash!r}; expected {expected_config_hash!r}."
        )

    outputs = manifest.setdefault("output_files", [])
    if not isinstance(outputs, list):
        raise RunManifestError("manifest.output_files must be a list.")
    portable = _portable_path(path, root)
    if any(isinstance(item, Mapping) and item.get("path") == portable for item in outputs):
        raise RunManifestError(f"Artifact is already registered: {portable}")

    record = {
        "path": portable,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "run_id": observed_run_id,
        "config_hash": observed_config_hash,
        "stage": stage,
        "artifact_type": artifact_type,
        "registered_at": utc_now_iso(),
    }
    outputs.append(record)
    return record


def record_failure(
    manifest: MutableMapping[str, Any],
    *,
    stage: str,
    error_type: str,
    message: str,
) -> dict[str, str]:
    failures = manifest.setdefault("failure_information", [])
    if not isinstance(failures, list):
        raise RunManifestError("manifest.failure_information must be a list.")
    record = {
        "timestamp": utc_now_iso(),
        "stage": stage,
        "error_type": error_type,
        "message": message,
    }
    failures.append(record)
    return record


def finalize_run_manifest(
    manifest: MutableMapping[str, Any],
    *,
    status: str,
) -> MutableMapping[str, Any]:
    if status not in {"complete", "failed"}:
        raise RunManifestError("Final manifest status must be 'complete' or 'failed'.")
    failures = manifest.get("failure_information", [])
    if status == "complete" and failures:
        raise RunManifestError("A run with recorded failures cannot be finalized as complete.")
    manifest["status"] = status
    manifest["end_timestamp"] = utc_now_iso()
    return manifest


def _load_manifest(manifest_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(manifest_or_path, Mapping):
        return dict(manifest_or_path)
    path = Path(manifest_or_path)
    if not path.is_file():
        raise RunManifestError(f"Run manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunManifestError(f"Invalid run manifest JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunManifestError("Run manifest root must be an object.")
    return payload


def validate_run_manifest(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    expected_config_hash: str | None = None,
    expected_evidence_scope: str | None = None,
    require_complete: bool = False,
    verify_source_tree: bool = False,
) -> dict[str, Any]:
    """Validate identity, hashes, and existence for all manifest references."""

    manifest = _load_manifest(manifest_or_path)
    root = Path(project_root).resolve()
    errors: list[str] = []

    required_fields = {
        "manifest_schema_version",
        "run_id",
        "git_commit",
        "config_path",
        "config_hash",
        "evidence_scope",
        "scope_contract",
        "scope_contract_hash",
        "actual_input_receipts",
        "dataset_hashes",
        "side_input_hashes",
        "scientific_input_hash",
        "code_package_versions",
        "start_timestamp",
        "end_timestamp",
        "random_seeds",
        "commands",
        "output_files",
        "status",
        "failure_information",
        "source_tree_hash",
        "git_worktree_dirty",
        "git_status_sha256",
    }
    missing_fields = sorted(required_fields - set(manifest))
    if missing_fields:
        errors.append(f"missing required fields: {missing_fields}")

    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(f"unsupported manifest schema version: {manifest.get('manifest_schema_version')!r}")
    run_id = manifest.get("run_id")
    config_hash = manifest.get("config_hash")
    try:
        validate_portable_run_id(run_id)
    except RunManifestError as exc:
        errors.append(str(exc))
    if not isinstance(config_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", config_hash):
        errors.append("config_hash must be a lowercase SHA-256 digest")
    if expected_config_hash is not None and config_hash != expected_config_hash:
        errors.append(f"config_hash {config_hash!r} does not equal expected {expected_config_hash!r}")

    worktree_dirty = manifest.get("git_worktree_dirty")
    git_status_hash = manifest.get("git_status_sha256")
    if not isinstance(worktree_dirty, bool):
        errors.append("git_worktree_dirty must be boolean")
    if git_status_hash != "unavailable" and not (
        isinstance(git_status_hash, str)
        and re.fullmatch(r"[0-9a-f]{64}", git_status_hash)
    ):
        errors.append("git_status_sha256 must be a lowercase SHA-256 digest or 'unavailable'")
    clean_status_hash = hashlib.sha256(b"").hexdigest()
    if (
        worktree_dirty is False
        and manifest.get("git_commit") != "unavailable"
        and git_status_hash != clean_status_hash
    ):
        errors.append(
            "git_worktree_dirty=False must be bound to the SHA-256 of an empty Git status"
        )
    preexisting_roots = manifest.get("preexisting_evidence_roots_ignored_for_clean_start", [])
    if not isinstance(preexisting_roots, list) or any(
        not isinstance(value, str) or not value or "\\" in value
        for value in preexisting_roots
    ):
        errors.append("preexisting evidence roots must be a list of portable paths")
    elif len(preexisting_roots) != len(set(preexisting_roots)):
        errors.append("preexisting evidence roots contain duplicates")
    else:
        for value in preexisting_roots:
            try:
                _resolve_portable_reference(value, root, context="preexisting evidence root")
            except RunManifestError as exc:
                errors.append(str(exc))

    observed_scope = manifest.get("evidence_scope")
    if observed_scope not in EXPECTED_EVIDENCE_SCOPE_DATASETS:
        errors.append(
            f"evidence_scope must be one of {sorted(EXPECTED_EVIDENCE_SCOPE_DATASETS)}"
        )
    if (
        expected_evidence_scope is not None
        and expected_evidence_scope not in EXPECTED_EVIDENCE_SCOPE_DATASETS
    ):
        errors.append(
            f"expected_evidence_scope must be one of {sorted(EXPECTED_EVIDENCE_SCOPE_DATASETS)}"
        )
    if expected_evidence_scope is not None and observed_scope != expected_evidence_scope:
        errors.append(
            f"evidence_scope {observed_scope!r} does not equal expected "
            f"{expected_evidence_scope!r}"
        )

    loaded_config: dict[str, Any] | None = None
    config_path: Path | None = None
    raw_config_path = manifest.get("config_path")
    try:
        config_path = _resolve_portable_reference(
            raw_config_path,
            root,
            context="config_path",
        )
    except RunManifestError as exc:
        errors.append(str(exc))
    else:
        if not config_path.is_file():
            errors.append(f"config file is missing: {config_path}")
        else:
            try:
                loaded_config = load_manuscript_config(config_path, project_root=root)
                actual_config_hash = canonical_config_hash(loaded_config)
            except Exception as exc:  # validation reports all manifest defects together
                errors.append(f"config cannot be loaded or hashed: {exc}")
            else:
                if actual_config_hash != config_hash:
                    errors.append(
                        f"config hash mismatch for {config_path}: manifest={config_hash}, actual={actual_config_hash}"
                    )

    current_scope_contract: dict[str, list[str]] | None = None
    observed_scope_contract = manifest.get("scope_contract")
    observed_scope_contract_hash = manifest.get("scope_contract_hash")
    if not isinstance(observed_scope_contract, Mapping):
        errors.append("scope_contract must be a mapping")
    if not isinstance(observed_scope_contract_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", observed_scope_contract_hash
    ):
        errors.append("scope_contract_hash must be a lowercase SHA-256 digest")
    elif isinstance(observed_scope_contract, Mapping):
        actual_scope_hash = _sha256_canonical_json(dict(observed_scope_contract))
        if observed_scope_contract_hash != actual_scope_hash:
            errors.append("scope_contract_hash does not bind the recorded scope_contract")
    if (
        loaded_config is not None
        and isinstance(observed_scope, str)
        and isinstance(observed_scope_contract, Mapping)
    ):
        try:
            current_scope_contract = evidence_scope_contract(loaded_config, observed_scope)
        except Exception as exc:
            errors.append(f"current evidence scope cannot be validated: {exc}")
        else:
            if dict(observed_scope_contract) != current_scope_contract:
                errors.append("scope_contract does not match the current canonical evidence scope")
    if loaded_config is not None:
        try:
            loaded_settings = manuscript_settings(loaded_config)
            expected_seeds = dict(_require_mapping(loaded_settings, "seeds", "manuscript_final"))
            provenance_settings = _require_mapping(
                loaded_settings,
                "provenance",
                "manuscript_final",
            )
            package_names = provenance_settings.get("package_names", [])
            if not isinstance(package_names, list):
                raise ManuscriptConfigError("provenance.package_names must be a list")
            expected_versions = _package_versions(package_names)
        except Exception as exc:
            errors.append(f"runtime identity cannot be reconstructed: {exc}")
        else:
            if manifest.get("random_seeds") != expected_seeds:
                errors.append("random_seeds do not match the canonical config")
            if manifest.get("code_package_versions") != expected_versions:
                errors.append("code_package_versions do not match the current runtime")

    side_input_hashes = manifest.get("side_input_hashes")
    if not isinstance(side_input_hashes, Mapping) or not side_input_hashes:
        errors.append("side_input_hashes must be a non-empty mapping")
    else:
        for side_name, record in side_input_hashes.items():
            label = f"side input {side_name!r}"
            if not isinstance(side_name, str) or not side_name:
                errors.append("side-input names must be non-empty strings")
            if not isinstance(record, Mapping):
                errors.append(f"{label} record is not a mapping")
                continue
            try:
                path = _resolve_portable_reference(
                    record.get("path"),
                    root,
                    context=f"{label}.path",
                )
            except RunManifestError as exc:
                errors.append(str(exc))
                continue
            if not path.is_file():
                errors.append(f"{label} is missing: {path}")
                continue
            actual_hash = sha256_file(path)
            if record.get("sha256") != actual_hash:
                errors.append(
                    f"{label} hash mismatch: manifest={record.get('sha256')}, actual={actual_hash}"
                )
            if record.get("size_bytes") != path.stat().st_size:
                errors.append(f"{label} size mismatch")

        if loaded_config is not None and current_scope_contract is not None:
            try:
                current_side_inputs = declared_side_input_hashes(
                    loaded_config,
                    side_input_keys=current_scope_contract["side_input_keys"],
                    project_root=root,
                )
            except Exception as exc:
                errors.append(f"declared side inputs cannot be verified: {exc}")
            else:
                if dict(side_input_hashes) != current_side_inputs:
                    errors.append(
                        "side_input_hashes do not match the current canonical declarations/content"
                    )

                provenance = _require_mapping(
                    manuscript_settings(loaded_config),
                    "provenance",
                    "manuscript_final",
                )
                acquisition_path = provenance.get("data_acquisition_manifest")
                acquisition_records = [
                    record
                    for record in current_side_inputs.values()
                    if isinstance(record, Mapping) and record.get("path") == acquisition_path
                ]
                if not acquisition_records:
                    errors.append(
                        "the configured data acquisition manifest must be declared as a scientific side input"
                    )

    actual_input_receipts = manifest.get("actual_input_receipts")
    if not isinstance(actual_input_receipts, Mapping) or not actual_input_receipts:
        errors.append("actual_input_receipts must be a non-empty mapping")

    dataset_hashes = manifest.get("dataset_hashes")
    if not isinstance(dataset_hashes, Mapping) or not dataset_hashes:
        errors.append("dataset_hashes must be a non-empty mapping")
    elif isinstance(actual_input_receipts, Mapping):
        if set(dataset_hashes) != set(actual_input_receipts):
            errors.append("dataset_hashes and actual_input_receipts must name exactly the same datasets")
        if current_scope_contract is not None:
            if set(dataset_hashes) != set(current_scope_contract["dataset_keys"]):
                errors.append(
                    "manifest dataset identities do not match the exact canonical evidence scope"
                )

        for dataset_name, record in dataset_hashes.items():
            if not isinstance(record, Mapping):
                errors.append(f"dataset {dataset_name!r} record is not a mapping")
                continue
            receipt = actual_input_receipts.get(dataset_name)
            if not isinstance(receipt, Mapping):
                errors.append(f"actual input receipt {dataset_name!r} is not a mapping")
                continue
            missing_receipt_fields = [
                field
                for field in (*ACTUAL_INPUT_IDENTITY_FIELDS, "size_bytes")
                if field not in receipt
            ]
            if missing_receipt_fields:
                errors.append(
                    f"actual input receipt {dataset_name!r} is missing fields: {missing_receipt_fields}"
                )
            if receipt.get("dataset_key") != dataset_name:
                errors.append(f"actual input receipt {dataset_name!r} has a mismatched dataset_key")
            try:
                path = _resolve_portable_reference(
                    record.get("path"),
                    root,
                    context=f"dataset {dataset_name!r}.path",
                )
                receipt_path = _resolve_portable_reference(
                    receipt.get("actual_path"),
                    root,
                    context=f"actual input receipt {dataset_name!r}.actual_path",
                )
            except RunManifestError as exc:
                errors.append(str(exc))
                continue
            if path != receipt_path:
                errors.append(f"dataset {dataset_name!r} path does not match its actual input receipt")
            if not path.is_file():
                errors.append(f"dataset {dataset_name!r} is missing: {path}")
                continue
            actual_hash = sha256_file(path)
            actual_size = path.stat().st_size
            if record.get("sha256") != actual_hash:
                errors.append(
                    f"dataset {dataset_name!r} hash mismatch: manifest={record.get('sha256')}, actual={actual_hash}"
                )
            if receipt.get("actual_sha256") != actual_hash:
                errors.append(
                    f"actual input receipt {dataset_name!r} hash mismatch: "
                    f"receipt={receipt.get('actual_sha256')}, actual={actual_hash}"
                )
            if record.get("size_bytes") != actual_size:
                errors.append(f"dataset {dataset_name!r} size mismatch")
            if receipt.get("size_bytes") != actual_size:
                errors.append(f"actual input receipt {dataset_name!r} size mismatch")
            receipt_links = {
                "path": "actual_path",
                "sha256": "actual_sha256",
                "size_bytes": "size_bytes",
                "row_count": "row_count",
                "column_count": "column_count",
                "schema_status": "schema_status",
                "target_column": "target_column",
                "target_distribution": "target_distribution",
            }
            for dataset_field, receipt_field in receipt_links.items():
                if record.get(dataset_field) != receipt.get(receipt_field):
                    errors.append(
                        f"dataset {dataset_name!r}.{dataset_field} does not match "
                        f"actual_input_receipts.{dataset_name}.{receipt_field}"
                    )

            try:
                acquisition_path = _resolve_portable_reference(
                    receipt.get("acquisition_manifest_path"),
                    root,
                    context=f"actual input receipt {dataset_name!r}.acquisition_manifest_path",
                )
            except RunManifestError as exc:
                errors.append(str(exc))
            else:
                if not acquisition_path.is_file():
                    errors.append(
                        f"actual input receipt {dataset_name!r} acquisition manifest is missing: "
                        f"{acquisition_path}"
                    )
                elif receipt.get("acquisition_manifest_sha256") != sha256_file(acquisition_path):
                    errors.append(
                        f"actual input receipt {dataset_name!r} acquisition-manifest hash mismatch"
                    )

        if (
            loaded_config is not None
            and config_path is not None
            and current_scope_contract is not None
        ):
            try:
                from src.data.canonical_loader import verify_configured_datasets

                provenance = _require_mapping(
                    manuscript_settings(loaded_config),
                    "provenance",
                    "manuscript_final",
                )
                current_verified = verify_configured_datasets(
                    config_path,
                    provenance.get("data_acquisition_manifest"),
                    dataset_keys=current_scope_contract["dataset_keys"],
                    allow_download=False,
                    project_root=root,
                )
            except Exception as exc:
                errors.append(f"current canonical datasets cannot be reverified: {exc}")
            else:
                for dataset_name, loaded in current_verified.items():
                    recorded_receipt = actual_input_receipts.get(dataset_name)
                    if not isinstance(recorded_receipt, Mapping):
                        continue
                    if _actual_receipt_identity(recorded_receipt) != _actual_receipt_identity(
                        loaded.receipt
                    ):
                        errors.append(
                            f"actual input receipt {dataset_name!r} does not match current "
                            "canonical-loader verification"
                        )

    observed_scientific_hash = manifest.get("scientific_input_hash")
    if not isinstance(observed_scientific_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", observed_scientific_hash
    ):
        errors.append("scientific_input_hash must be a lowercase SHA-256 digest")
    elif (
        isinstance(config_hash, str)
        and re.fullmatch(r"[0-9a-f]{64}", config_hash)
        and isinstance(observed_scope_contract_hash, str)
        and re.fullmatch(r"[0-9a-f]{64}", observed_scope_contract_hash)
        and isinstance(dataset_hashes, Mapping)
        and dataset_hashes
        and isinstance(side_input_hashes, Mapping)
        and side_input_hashes
    ):
        try:
            expected_scientific_hash = scientific_input_hash(
                config_hash=config_hash,
                scope_contract_hash=observed_scope_contract_hash,
                dataset_hashes=dataset_hashes,
                side_input_hashes=side_input_hashes,
            )
        except Exception as exc:
            errors.append(f"scientific input identity cannot be recomputed: {exc}")
        else:
            if observed_scientific_hash != expected_scientific_hash:
                errors.append(
                    "scientific_input_hash does not bind the recorded config, datasets, and side inputs"
                )

    commands = manifest.get("commands")
    command_records: list[Mapping[str, Any]] = []
    if not isinstance(commands, list):
        errors.append("commands must be a list")
    else:
        for index, record in enumerate(commands):
            label = f"commands[{index}]"
            if not isinstance(record, Mapping):
                errors.append(f"{label} must be a mapping")
                continue
            command_records.append(record)
            for field in ("command", "stage", "started_at"):
                if not isinstance(record.get(field), str) or not str(record.get(field)).strip():
                    errors.append(f"{label}.{field} must be a non-empty string")
            command_status = record.get("status")
            ended_at = record.get("ended_at")
            return_code = record.get("return_code")
            if command_status not in {"started", "complete", "failed", "interrupted"}:
                errors.append(f"{label} has invalid command status {command_status!r}")
            elif command_status == "started":
                if ended_at is not None or return_code is not None:
                    errors.append(
                        f"{label} started command must have null ended_at and return_code"
                    )
            else:
                if not isinstance(ended_at, str) or not ended_at.strip():
                    errors.append(f"{label} terminal command requires ended_at")
                if command_status == "complete" and return_code != 0:
                    errors.append(f"{label} complete command requires return_code=0")
                elif command_status == "failed" and (
                    not isinstance(return_code, int)
                    or isinstance(return_code, bool)
                    or return_code == 0
                ):
                    errors.append(f"{label} failed command requires a nonzero return_code")
                elif command_status == "interrupted" and return_code is not None and (
                    not isinstance(return_code, int)
                    or isinstance(return_code, bool)
                    or return_code == 0
                ):
                    errors.append(
                        f"{label} interrupted command return_code must be null or nonzero"
                    )

    outputs = manifest.get("output_files")
    if not isinstance(outputs, list):
        errors.append("output_files must be a list")
    else:
        seen_paths: set[str] = set()
        for index, record in enumerate(outputs):
            label = f"output_files[{index}]"
            if not isinstance(record, Mapping):
                errors.append(f"{label} is not a mapping")
                continue
            raw_path = record.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                errors.append(f"{label} has no path")
                continue
            if raw_path in seen_paths:
                errors.append(f"duplicate artifact path in manifest: {raw_path}")
            seen_paths.add(raw_path)
            if record.get("run_id") != run_id:
                errors.append(f"{label} run_id does not match manifest run_id")
            if record.get("config_hash") != config_hash:
                errors.append(f"{label} config_hash does not match manifest config_hash")
            try:
                path = _resolve_portable_reference(
                    raw_path,
                    root,
                    context=f"{label}.path",
                )
            except RunManifestError as exc:
                errors.append(str(exc))
                continue
            if not path.is_file():
                errors.append(f"manifest-referenced artifact is missing: {path}")
                continue
            actual_hash = sha256_file(path)
            if record.get("sha256") != actual_hash:
                errors.append(
                    f"artifact hash mismatch for {path}: manifest={record.get('sha256')}, actual={actual_hash}"
                )
            if record.get("size_bytes") != path.stat().st_size:
                errors.append(f"artifact size mismatch for {path}")

    status = manifest.get("status")
    if status not in {"running", "complete", "failed"}:
        errors.append(f"invalid run status: {status!r}")
    if require_complete and status != "complete":
        errors.append(f"run is not complete: status={status!r}")
    if status in {"complete", "failed"} and not manifest.get("end_timestamp"):
        errors.append("a finalized run requires end_timestamp")
    if status == "complete" and manifest.get("failure_information"):
        errors.append("a complete run cannot contain failure_information")
    failures = manifest.get("failure_information")
    if not isinstance(failures, list):
        errors.append("failure_information must be a list")
    if status == "complete":
        if worktree_dirty is not False:
            errors.append("a complete run must originate from a clean git_worktree_dirty=False state")
        if not isinstance(outputs, list) or not outputs:
            errors.append("a complete run requires at least one registered output artifact")
        entrypoints = [record for record in command_records if record.get("stage") == "entrypoint"]
        successful_entrypoints = [
            record
            for record in entrypoints
            if record.get("status") == "complete" and record.get("return_code") == 0
        ]
        if len(successful_entrypoints) != 1:
            errors.append("a complete run requires exactly one successful terminal entrypoint command")
        unfinished = [
            str(record.get("stage"))
            for record in command_records
            if record.get("status") in {"started", "failed"}
        ]
        if unfinished:
            errors.append(
                f"a complete run contains unfinished or failed command records: {unfinished}"
            )
    if status == "failed":
        if not isinstance(failures, list) or not failures:
            errors.append("a failed run requires non-empty failure_information")
        entrypoint_records = [
            record for record in command_records if record.get("stage") == "entrypoint"
        ]
        failed_entrypoints = [
            record
            for record in entrypoint_records
            if record.get("status") == "failed"
            and isinstance(record.get("return_code"), int)
            and not isinstance(record.get("return_code"), bool)
            and record.get("return_code") != 0
        ]
        if len(failed_entrypoints) != 1:
            errors.append("a failed run requires exactly one terminal failed entrypoint command")
        noninterrupted_prior_entrypoints = [
            str(record.get("status"))
            for record in entrypoint_records
            if record.get("status") in {"complete", "started"}
        ]
        if noninterrupted_prior_entrypoints:
            errors.append(
                "a failed run cannot contain complete or started entrypoint commands; "
                "prior entrypoint attempts may only be interrupted"
            )
        started_commands = [
            str(record.get("stage"))
            for record in command_records
            if record.get("status") == "started"
        ]
        if started_commands:
            errors.append(f"a failed run contains started command records: {started_commands}")

    if verify_source_tree:
        actual_source_hash = source_tree_hash(root)
        if manifest.get("source_tree_hash") != actual_source_hash:
            errors.append(
                "source tree hash mismatch: the experiment/config source changed after the run manifest was created"
            )
        current_commit = _run_git(root, "rev-parse", "HEAD")
        if manifest.get("git_commit") != current_commit:
            errors.append(
                "git commit mismatch: HEAD changed after the run manifest was created"
            )
        tracked_status = _run_git(root, "status", "--porcelain", "--untracked-files=no")
        if tracked_status == "unavailable":
            errors.append("tracked worktree status could not be verified")
        elif tracked_status:
            errors.append(
                "tracked worktree changed after the run started: "
                + hashlib.sha256(tracked_status.encode("utf-8")).hexdigest()
            )

    if errors:
        raise RunManifestError("Invalid manuscript run manifest:\n- " + "\n- ".join(errors))
    return manifest


def write_run_manifest(
    manifest: Mapping[str, Any],
    path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    validate: bool = True,
    require_complete: bool = False,
) -> Path:
    """Atomically write a manifest after optional integrity validation."""

    if validate:
        validate_run_manifest(
            manifest,
            project_root=project_root,
            require_complete=require_complete,
            verify_source_tree=(
                require_complete
                and Path(project_root).resolve() == PROJECT_ROOT.resolve()
            ),
        )
    root = Path(project_root).resolve()
    destination = _resolve_from_root(path, root)
    _portable_path(destination, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


__all__ = [
    "ACTUAL_INPUT_IDENTITY_FIELDS",
    "DEFAULT_CONFIG_PATH",
    "EXPECTED_EVIDENCE_SCOPE_DATASETS",
    "LEGACY_FEATURE_POLICY_PROJECTION_PATH",
    "FeaturePolicyConsistencyError",
    "ForbiddenFeatureError",
    "ManuscriptConfigError",
    "RunManifestError",
    "artifact_feature_names",
    "canonical_config_hash",
    "canonical_policy_mapping",
    "create_run_manifest",
    "declared_side_input_hashes",
    "evidence_scope_contract",
    "evidence_scope_contract_hash",
    "feature_policy_definitions",
    "finalize_run_manifest",
    "forbidden_feature_mentions",
    "load_manuscript_config",
    "make_run_id",
    "manuscript_settings",
    "primary_excluded_features",
    "primary_policy_definition",
    "primary_policy_name",
    "record_command",
    "record_failure",
    "register_artifact",
    "repository_feature_policy_projection",
    "sha256_file",
    "scientific_input_hash",
    "source_tree_hash",
    "validate_artifact_forbidden_features",
    "validate_manuscript_config",
    "validate_policy_consistency",
    "validate_portable_run_id",
    "validate_primary_feature_names",
    "validate_run_manifest",
    "write_run_manifest",
]
