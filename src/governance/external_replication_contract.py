"""Frozen HRDataset_v14 external-replication configuration contract.

This module contains configuration validation only.  It deliberately performs
no model fitting and does not import the external experiment runner.  The
schema mapping and provenance binding are duplicated scientific side inputs,
so the canonical loader validates their semantic equality before a default
manuscript configuration is admitted.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


class ExternalReplicationContractError(ValueError):
    """Raised when the frozen external-replication contract drifts."""


POLICY_ORDER = (
    "conservative_primary",
    "department_including_audit",
    "job_role_free_audit",
    "proxy_rich_audit",
    "temporality_restricted_audit",
)

ALWAYS_FORBIDDEN_FEATURE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "identifiers": (
        "Employee_Name",
        "EmpID",
        "EmpNumber",
        "ExternalSampleId",
        "ManagerName",
        "ManagerID",
    ),
    "targets": ("PerformanceScore", "PerformanceRating", "PerfScoreID"),
    "leakage_or_post_outcome": (
        "Termd",
        "EmpStatusID",
        "EmploymentStatus",
        "DateofTermination",
        "TermReason",
        "LastPerformanceReview_Date",
    ),
    "sensitive_or_sensitive_proxy": (
        "DOB",
        "Gender",
        "Sex",
        "GenderID",
        "MarriedID",
        "MaritalStatus",
        "MaritalDesc",
        "MaritalStatusID",
        "RaceEthnicity",
        "RaceDesc",
        "HispanicLatino",
        "CitizenDesc",
        "FromDiversityJobFairID",
    ),
    # The verified raw schema spells this field ``DateofHire``.  Feature-name
    # enforcement is case-insensitive, so listing ``DateOfHire`` as a second
    # alias would create an internally contradictory exclusion contract.
    "encoded_or_raw_aliases": ("DeptID", "PositionID", "DateofHire"),
}

PRIMARY_GOVERNANCE_EXCLUSIONS = (
    "EmpDepartment",
    "Salary",
    "State",
    "Zip",
    "RecruitmentSource",
)

CONSERVATIVE_PRIMARY_FEATURES = (
    "EmpJobRole",
    "EngagementSurvey",
    "EmpJobSatisfaction",
    "SpecialProjectsCount",
    "DaysLateLast30",
    "Absences",
    "ExperienceYearsAtThisCompany",
)

PRIMARY_FEATURE_GOVERNANCE: Mapping[str, Mapping[str, str]] = {
    "EmpJobRole": {
        "category": "operational_proxy_context",
        "temporality_status": "recorded_role_context_timing_unverified",
    },
    "EngagementSurvey": {
        "category": "employee_reported_context",
        "temporality_status": "timing_unverified_contemporaneous",
    },
    "EmpJobSatisfaction": {
        "category": "employee_reported_context",
        "temporality_status": "timing_unverified_contemporaneous",
    },
    "SpecialProjectsCount": {
        "category": "operational_history_or_window",
        "temporality_status": "timing_unverified_history_or_window",
    },
    "DaysLateLast30": {
        "category": "operational_history_or_window",
        "temporality_status": "timing_unverified_history_or_window",
    },
    "Absences": {
        "category": "operational_history_or_window",
        "temporality_status": "timing_unverified_history_or_window",
    },
    "ExperienceYearsAtThisCompany": {
        "category": "derived_tenure_context",
        "temporality_status": "derived_at_last_review_timing_unverified_negative_durations_set_missing",
    },
}

TARGET_MAPPING: Mapping[str, int] = {
    "PIP": 2,
    "Needs Improvement": 2,
    "Fully Meets": 3,
    "Exceeds": 4,
    "Exceptional": 4,
}

TARGET_MAPPING_RATIONALE = (
    "Predeclared mapped-target replication merges PIP and Needs Improvement as label 2, "
    "maps Fully Meets to label 3, and maps Exceeds or Exceptional to label 4 so the "
    "observed ordered performance strata can be evaluated on the canonical 2/3/4 scale."
)
TARGET_MAPPING_LIMITATION = (
    "The mapping is dataset-specific and does not prove semantic, measurement, or "
    "prevalence equivalence with the INX PerformanceRating target; observed raw and "
    "mapped support must be reported."
)

EXPECTED_EXTERNAL_SEEDS: Mapping[str, int] = {
    "external_replication": 42,
    "inner_cv": 43,
    "model": 42,
    "calibration": 42,
    "bootstrap": 42,
    "fairness": 42,
}

POLICY_METADATA: Mapping[str, Mapping[str, Any]] = {
    "conservative_primary": {
        "role": "canonical_external_primary",
        "audit_only": False,
        "description": (
            "Conservative leakage-aware external primary policy excluding direct and "
            "encoded identifiers, targets, leakage/post-outcome fields, sensitive fields, "
            "department, and the declared high-risk salary/location/recruitment proxies "
            "while retaining EmpJobRole."
        ),
        "restored_features": [],
        "additional_excluded_features": [],
    },
    "department_including_audit": {
        "role": "department_text_including_sensitivity_audit",
        "audit_only": True,
        "description": (
            "Audit-only sensitivity variant restoring EmpDepartment text while DeptID "
            "and every other always-forbidden alias remain excluded."
        ),
        "restored_features": ["EmpDepartment"],
        "additional_excluded_features": [],
    },
    "job_role_free_audit": {
        "role": "job_role_removed_proxy_sensitivity_audit",
        "audit_only": True,
        "description": (
            "Audit-only proxy sensitivity variant adding EmpJobRole to the conservative "
            "primary exclusions while PositionID remains forbidden."
        ),
        "restored_features": [],
        "additional_excluded_features": ["EmpJobRole"],
    },
    "proxy_rich_audit": {
        "role": "proxy_rich_sensitivity_audit",
        "audit_only": True,
        "description": (
            "Audit-only sensitivity variant restoring Salary, State, and RecruitmentSource; "
            "Zip and all always-forbidden aliases remain excluded."
        ),
        "restored_features": ["Salary", "State", "RecruitmentSource"],
        "additional_excluded_features": [],
    },
    "temporality_restricted_audit": {
        "role": "temporality_restricted_robustness_audit",
        "audit_only": True,
        "description": (
            "Audit-only temporal-availability sensitivity variant additionally excluding "
            "survey, satisfaction, project, lateness, and absence fields from the "
            "conservative primary policy."
        ),
        "restored_features": [],
        "additional_excluded_features": [
            "EngagementSurvey",
            "EmpJobSatisfaction",
            "SpecialProjectsCount",
            "DaysLateLast30",
            "Absences",
        ],
    },
}


def _base_exclusions() -> list[str]:
    return [
        feature
        for category in ALWAYS_FORBIDDEN_FEATURE_ALIASES.values()
        for feature in category
    ] + list(PRIMARY_GOVERNANCE_EXCLUSIONS)


def policy_exclusion_list(policy_name: str) -> list[str]:
    """Return the exact ordered raw/canonical exclusion list for one policy."""

    try:
        metadata = POLICY_METADATA[policy_name]
    except KeyError as exc:  # pragma: no cover - constant-backed public guard
        raise ExternalReplicationContractError(f"Unknown external policy {policy_name!r}.") from exc
    restored = set(metadata["restored_features"])
    exclusions = [feature for feature in _base_exclusions() if feature not in restored]
    exclusions.extend(metadata["additional_excluded_features"])
    casefolded = [feature.casefold() for feature in exclusions]
    if len(casefolded) != len(set(casefolded)):
        raise ExternalReplicationContractError(
            f"External policy {policy_name!r} contains case-insensitive duplicate exclusions."
        )
    return exclusions


def expected_schema_policy_variants() -> dict[str, dict[str, Any]]:
    """Return schema-mapping policy rows including exact exclusion lists."""

    result: dict[str, dict[str, Any]] = {}
    for name in POLICY_ORDER:
        row = copy.deepcopy(dict(POLICY_METADATA[name]))
        row["exclude_columns"] = policy_exclusion_list(name)
        result[name] = row
    return result


def expected_external_replication_contract() -> dict[str, Any]:
    """Return the complete author-approved external-replication config section."""

    return {
        "scope": "core_independent_mapped_target_replication",
        "dataset_key": "hrdataset_v14",
        "task_type": "ordinal_multiclass_performance",
        "role": "independent_external_performance_target_replication",
        "locked_inx_model_transport": False,
        "target": {
            "raw_column": "PerformanceScore",
            "canonical_column": "PerformanceRating",
            "labels": [2, 3, 4],
            "ordering": [2, 3, 4],
            "mapping": dict(TARGET_MAPPING),
            "mapping_rationale": TARGET_MAPPING_RATIONALE,
            "mapping_limitation": TARGET_MAPPING_LIMITATION,
        },
        "claim_boundary": (
            "Independently trained mapped-target replication of the leakage-aware protocol; "
            "not locked-model transport, universal external validation, causal evidence, "
            "fairness proof, deployment readiness, or autonomous HR decision support."
        ),
        "feature_policy_contract": {
            "schema_mapping_side_input": "external_hrdataset_v14_schema_mapping",
            "primary_policy": "conservative_primary",
            "reported_policy_order": list(POLICY_ORDER),
            "always_forbidden_feature_aliases": {
                name: list(values) for name, values in ALWAYS_FORBIDDEN_FEATURE_ALIASES.items()
            },
            "primary_governance_exclusions": list(PRIMARY_GOVERNANCE_EXCLUSIONS),
            "policies": copy.deepcopy(dict(POLICY_METADATA)),
            "exact_exclusion_lists_required": True,
            "forbidden_alias_scan_required_for_raw_and_transformed_features": True,
            "zip_may_never_be_restored": True,
        },
        "feature_governance": {
            "scope": "conservative_primary_exact_feature_families",
            "feature_list_source": (
                "schema_adapted_conservative_primary_after_exact_exclusions"
            ),
            "exact_primary_feature_families": list(CONSERVATIVE_PRIMARY_FEATURES),
            "metadata_key_equality_required": True,
            "shap_metadata_required": True,
            "model_scenario_only_warning": (
                "Feature metadata governs model attribution only; it does not make any "
                "feature causal, actionable, prescriptive, or suitable for employee advice."
            ),
            "features": copy.deepcopy(dict(PRIMARY_FEATURE_GOVERNANCE)),
        },
        "model_protocol": {
            "model": "xgboost",
            "search_space_source": "provenance.scientific_side_inputs.model_search_space",
            "selection_protocol": "nested_inner_cv_within_each_external_outer_training_partition",
            "selection_primary_metric": "macro_f1",
            "selection_tie_break_metric": "quadratic_weighted_kappa",
            "macro_f1_tie_tolerance": 0.001,
            "outer_test_used_for_tuning_or_selection": False,
            "preprocessing_fit_scope": "development_partition_only",
            "selected_parameters_and_fit_receipts_required_per_outer_fold": True,
        },
        "cv": {
            "outer_strategy": "StratifiedKFold",
            "outer_splits": 10,
            "outer_shuffle": True,
            "outer_seed_source": "seeds.external_replication",
            "inner_strategy": "StratifiedKFold",
            "inner_splits": 5,
            "inner_shuffle": True,
            "inner_seed_source": "seeds.inner_cv",
            "same_outer_folds_across_policies": True,
            "fold_assignments_persisted_and_hashed": True,
            "oof_prediction_exactly_once_per_sample_per_policy": True,
            "target_and_identifiers_absent_from_preprocessing": True,
        },
        "calibration": {
            "primary_method": "sigmoid",
            "method_selection": "predeclared_not_outer_test_selected",
            "training_protocol": "five_fold_cross_fitted_outer_training_only",
            "inner_splits": 5,
            "calibrator_fit_input": "inner_oof_probabilities_and_labels_only",
            "outer_model_source": "exact_selected_external_xgboost_outer_fold_model",
            "outer_model_refit_in_calibration_stage": False,
            "outer_test_usage": "evaluation_only_after_calibrator_fit",
            "threshold_selection": "none",
            "n_bins": 10,
            "raw_and_sigmoid_oof_probabilities_reported": True,
        },
        "uncertainty": {
            "method": "paired_stratified_percentile_bootstrap",
            "n_resamples": 5000,
            "confidence_level": 0.95,
            "quantile_method": "linear",
            "seed_source": "seeds.bootstrap",
            "stratify_by": ["outer_fold", "y_true"],
            "same_resamples_across_policies": True,
            "paired_policy_differences_required": True,
            "resample_indices_persisted_and_hashed": True,
            "fold_summaries": "descriptive_variability_only_no_population_ci",
            "conditional_inference_note": (
                "Intervals condition on the observed HRDataset_v14 employees and fixed "
                "nested model-training protocol; they do not estimate dataset-source, "
                "target-mapping, or model-training instability."
            ),
        },
        "shap": {
            "scope": "conservative_primary_oof_xgboost_only",
            "model_source": "exact_prediction_producing_external_outer_fold_model",
            "attribution_unit": "xgboost_raw_margin_score",
            "additivity_output_space": "xgboost_raw_margin",
            "model_refit_in_shap_stage": False,
            "oof_prediction_replay_required": True,
            "group_one_hot_to_raw_feature_families": True,
            "global_mean_absolute": True,
            "class_specific_global": True,
            "local_oof_values": True,
            "fold_rankings": True,
            "stability_top_k": 5,
            "stability_metrics": ["top_k_jaccard", "spearman_rank_correlation"],
            "fold_pair_inference": "descriptive_distribution_only_no_independence_ci",
            "governance_metadata_required": True,
            "non_causality_warning_required": True,
            "forbidden_feature_source": "external_replication.feature_policy_contract",
        },
        "subgroup_diagnostics": {
            "prediction_source": "conservative_primary_exact_oof_predictions",
            "probability_method": "raw",
            "attributes": {
                "protected_sensitive": {
                    "type": "categorical",
                    "features": [
                        "Gender",
                        "RaceEthnicity",
                        "HispanicLatino",
                        "MaritalStatus",
                    ],
                },
                "exploratory_operational": {
                    "type": "categorical",
                    "features": ["EmpDepartment"],
                },
            },
            "all_specified_attributes_absent_from_conservative_primary": True,
            "minimum_group_support": 30,
            "minimum_class_metric_denominator": 10,
            "uncertainty_source": "external_replication.uncertainty",
            "insufficient_support_retained_and_flagged": True,
            "headline_requires_support_and_valid_bootstrap_context": True,
            "intervals": "pointwise_descriptive",
            "multiplicity_adjustment": "none",
            "legacy_fairness_helper_dependency_allowed": False,
            "claim_boundary": (
                "Support-aware subgroup diagnostics are descriptive and do not establish "
                "fairness, discrimination, legal compliance, or causality."
            ),
        },
        "proxy_diagnostics": {
            "task_type": "nominal_multiclass_proxy_diagnostic",
            "target": "EmpDepartment",
            "target_aliases": ["Department", "DeptID", "EmpDepartment"],
            "target_aliases_absent_from_all_predictors": True,
            "predictor_policy_sources": [
                "conservative_primary",
                "job_role_free_audit",
            ],
            "predictor_features_source": (
                "external_replication.feature_policy_contract."
                "exact_schema_adapted_policy_features"
            ),
            "classifier_source": "proxy_analysis.classifier",
            "preprocessing_source": "proxy_analysis.preprocessing",
            "metrics_source": "proxy_analysis.metrics",
            "outer_folds_source": (
                "external_replication.cv.persisted_outer_fold_assignments"
            ),
            "same_exact_outer_folds_as_external_performance_models": True,
            "oof_prediction_exactly_once_per_sample_per_predictor_system": True,
            "bootstrap": {
                "method": "paired_stratified_percentile_bootstrap",
                "n_resamples": 5000,
                "confidence_level": 0.95,
                "quantile_method": "linear",
                "seed_source": "seeds.fairness",
                "stratify_by": ["outer_fold", "proxy_target"],
                "paired_across_predictor_systems": True,
                "resample_indices_persisted_and_hashed": True,
            },
            "class_support": {
                "required_outer_training_class_set": (
                    "complete_observed_proxy_target_class_set"
                ),
                "merge_classes_allowed": False,
                "drop_classes_allowed": False,
                "insufficient_support_status": (
                    "not_estimated_insufficient_outer_training_class_support"
                ),
                "class_counts_and_zero_support_cells_reported": True,
            },
            "claim_boundary": (
                "Department reconstructability is descriptive proxy-risk evidence; it does "
                "not prove that the performance model uses department causally or "
                "discriminatorily."
            ),
        },
        "provenance": {
            "actual_input_receipt_required": True,
            "actual_input_receipt_must_equal_scoped_run_manifest": True,
            "schema_mapping_hash_must_equal_scoped_side_input": True,
            "target_mapping_must_equal_dataset_provenance": True,
            "scientific_input_hash_required_in_all_artifacts": True,
            "fold_model_preprocessor_calibrator_hashes_required": True,
            "prepublication_input_revalidation_required": True,
            "cache_key_includes_dataset_and_all_scientific_side_input_hashes": True,
            "historical_artifact_reuse_allowed": False,
            "source_licence_authenticity_status": "manual_review_required",
        },
        "publication": {
            "staged_atomic_publication_required": True,
            "publish_only_after_all_validations_pass": True,
            "partial_or_fallback_artifacts_allowed": False,
            "output_paths_repo_or_run_relative": True,
            "stage_manifest_hashes_every_output": True,
            "core_output_contains_no_counterfactual_llm_chatbot_or_related_task_artifact": True,
        },
    }


def validate_external_replication_settings(settings: Mapping[str, Any]) -> None:
    """Fail closed unless the canonical external section is exactly frozen."""

    observed = settings.get("external_replication")
    if not isinstance(observed, Mapping):
        raise ExternalReplicationContractError(
            "manuscript_final.external_replication must be a mapping."
        )
    expected = expected_external_replication_contract()
    if dict(observed) != expected:
        raise ExternalReplicationContractError(
            "external_replication differs from the frozen HRDataset_v14 10x5 "
            "leakage-aware replication contract."
        )
    seeds = settings.get("seeds")
    if not isinstance(seeds, Mapping):
        raise ExternalReplicationContractError("Canonical seeds mapping is missing.")
    drift = {
        key: {"expected": expected_value, "observed": seeds.get(key)}
        for key, expected_value in EXPECTED_EXTERNAL_SEEDS.items()
        if seeds.get(key) != expected_value
    }
    if drift:
        raise ExternalReplicationContractError(
            "HRDataset external protocol seeds differ from the frozen computation contract: "
            + json.dumps(drift, sort_keys=True)
        )


def _resolve(reference: Any, project_root: Path, *, context: str) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise ExternalReplicationContractError(f"{context} must be a non-empty path.")
    path = Path(reference)
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_file():
        raise ExternalReplicationContractError(f"{context} is missing: {reference!r}.")
    return path


def _load_json_mapping(path: Path, *, context: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalReplicationContractError(f"Cannot load {context}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ExternalReplicationContractError(f"{context} must contain a mapping.")
    return payload


def _contains_leakage_safe(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_leakage_safe(key) or _contains_leakage_safe(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_leakage_safe(item) for item in value)
    return isinstance(value, str) and "leakage-safe" in value.casefold()


def validate_external_replication_side_inputs(
    settings: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> None:
    """Validate schema policy and target mapping equality across side inputs."""

    validate_external_replication_settings(settings)
    root = Path(project_root).resolve()
    datasets = settings.get("datasets")
    if not isinstance(datasets, Mapping) or not isinstance(datasets.get("hrdataset_v14"), Mapping):
        raise ExternalReplicationContractError("Canonical HRDataset_v14 declaration is missing.")
    dataset = datasets["hrdataset_v14"]
    schema_path = _resolve(
        dataset.get("schema_mapping_path"),
        root,
        context="HRDataset_v14 schema mapping",
    )
    schema = _load_json_mapping(schema_path, context="HRDataset_v14 schema mapping")

    expected_target = expected_external_replication_contract()["target"]
    if schema.get("dataset_name") != "hrdataset_v14":
        raise ExternalReplicationContractError("HRDataset schema mapping identity drifted.")
    schema_target = schema.get("target")
    expected_schema_target = {
        "raw_column": expected_target["raw_column"],
        "canonical_column": expected_target["canonical_column"],
        "task_type": "ordinal_multiclass_performance",
        "mapping": expected_target["mapping"],
        "mapping_rationale": expected_target["mapping_rationale"],
        "mapping_limitation": expected_target["mapping_limitation"],
    }
    if schema_target != expected_schema_target:
        raise ExternalReplicationContractError(
            "HRDataset schema target mapping/rationale differs from the canonical contract."
        )
    if schema.get("feature_policy_variants") != expected_schema_policy_variants():
        raise ExternalReplicationContractError(
            "HRDataset schema feature policies differ from the exact canonical exclusions/roles."
        )
    expected_derived = {
        "ExperienceYearsAtThisCompany": {
            "from": "DateofHire",
            "reference": "LastPerformanceReview_Date",
            "description": "Approximate tenure at last available performance review date.",
            "missing_or_invalid_date_policy": "preserve_missing_and_require_expected_counts",
            "expected_missing_or_invalid_source_count": 0,
            "expected_missing_or_invalid_reference_count": 0,
            "invalid_duration_policy": "set_negative_to_missing",
            "expected_invalid_negative_count": 2,
            "lineage_limitation": (
                "Derived only as a timing-unverified tenure context; raw date fields remain "
                "forbidden model inputs and the derived value is not causal or actionable."
            ),
        }
    }
    if schema.get("derived_features") != expected_derived:
        raise ExternalReplicationContractError(
            "HRDataset derived tenure lineage/data-quality policy differs from the frozen contract."
        )
    expected_role_lists = {
        "id_columns": ["EmpID", "Employee_Name"],
        "leakage_risk_columns": [
            "PerformanceScore",
            "PerfScoreID",
            "Termd",
            "EmpStatusID",
            "EmploymentStatus",
            "DateofTermination",
            "TermReason",
            "LastPerformanceReview_Date",
            "ManagerName",
            "ManagerID",
        ],
        "sensitive_audit_only_columns": [
            "DOB",
            "Gender",
            "Sex",
            "GenderID",
            "MarriedID",
            "MaritalStatus",
            "MaritalDesc",
            "MaritalStatusID",
            "RaceEthnicity",
            "RaceDesc",
            "HispanicLatino",
            "CitizenDesc",
            "FromDiversityJobFairID",
        ],
        "proxy_risk_columns": [
            "Salary",
            "EmpDepartment",
            "DeptID",
            "EmpJobRole",
            "PositionID",
            "RecruitmentSource",
            "State",
            "Zip",
            "ExperienceYearsAtThisCompany",
            "SpecialProjectsCount",
            "Absences",
            "DaysLateLast30",
        ],
    }
    for field, expected_values in expected_role_lists.items():
        if schema.get(field) != expected_values:
            raise ExternalReplicationContractError(
                f"HRDataset schema {field} differs from the frozen semantic-role list."
            )
    if _contains_leakage_safe(schema):
        raise ExternalReplicationContractError(
            "HRDataset schema mapping must use leakage-aware, never leakage-safe, terminology."
        )

    provenance = settings.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ExternalReplicationContractError("Canonical provenance section is missing.")
    provenance_path = _resolve(
        provenance.get("dataset_cards_config"),
        root,
        context="dataset provenance config",
    )
    provenance_document = _load_json_mapping(
        provenance_path,
        context="dataset provenance config",
    )
    provenance_root = provenance_document.get("dataset_provenance")
    if not isinstance(provenance_root, Mapping):
        raise ExternalReplicationContractError("dataset_provenance root mapping is missing.")
    bindings = provenance_root.get("dataset_bindings")
    binding = bindings.get("hrdataset_v14") if isinstance(bindings, Mapping) else None
    if not isinstance(binding, Mapping):
        raise ExternalReplicationContractError("HRDataset provenance binding is missing.")
    expected_binding = {
        "physical_source": "hrdataset_v14",
        "raw_target": expected_target["raw_column"],
        "target_mapping": expected_target["mapping"],
        "target_mapping_note": expected_target["mapping_rationale"],
        "target_mapping_limitation": expected_target["mapping_limitation"],
    }
    if dict(binding) != expected_binding:
        raise ExternalReplicationContractError(
            "HRDataset target mapping differs between schema, provenance, and canonical config."
        )
    physical_sources = provenance_root.get("physical_sources")
    physical = physical_sources.get("hrdataset_v14") if isinstance(physical_sources, Mapping) else None
    if not isinstance(physical, Mapping):
        raise ExternalReplicationContractError("HRDataset physical provenance record is missing.")
    if physical.get("source_authenticity_verification_status") != "manual_review_required":
        raise ExternalReplicationContractError(
            "HRDataset source authenticity may not be silently marked verified."
        )
    if physical.get("licence_verification_status") != "manual_review_required":
        raise ExternalReplicationContractError(
            "HRDataset licence may not be silently marked verified."
        )
    if schema.get("source_url") != physical.get("retrieval_url"):
        raise ExternalReplicationContractError(
            "HRDataset recorded schema/provenance source URLs differ."
        )


__all__ = [
    "ALWAYS_FORBIDDEN_FEATURE_ALIASES",
    "CONSERVATIVE_PRIMARY_FEATURES",
    "ExternalReplicationContractError",
    "EXPECTED_EXTERNAL_SEEDS",
    "POLICY_ORDER",
    "PRIMARY_GOVERNANCE_EXCLUSIONS",
    "PRIMARY_FEATURE_GOVERNANCE",
    "TARGET_MAPPING",
    "expected_external_replication_contract",
    "expected_schema_policy_variants",
    "policy_exclusion_list",
    "validate_external_replication_settings",
    "validate_external_replication_side_inputs",
]
