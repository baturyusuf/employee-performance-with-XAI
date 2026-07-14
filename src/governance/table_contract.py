"""Frozen source-bound table plan for the v2 evidence packages."""

from __future__ import annotations

import copy
from types import MappingProxyType
from typing import Any, Mapping


class TableContractError(ValueError):
    """Raised when the configured table plan differs from the frozen contract."""


TABLE_PLAN_VERSION = "v2.0.0"
TABLE_IDENTITY_FIELDS = (
    "run_id",
    "config_hash",
    "scientific_input_hash",
    "source_tree_hash",
)


def _definition(
    number: int,
    table_id: str,
    title: str,
    *,
    scope: str,
    sources: tuple[str, ...],
    evaluation_scope: str,
    dataset_scope: str,
    model_scope: str,
    uncertainty_method: str,
    claim_boundary: str,
    special_renderer: str = "source_records",
) -> dict[str, Any]:
    return {
        "number": number,
        "table_id": table_id,
        "filename": f"table_{number:02d}_{table_id}.csv",
        "title": title,
        "evidence_scope": scope,
        "sources": list(sources),
        "evaluation_scope": evaluation_scope,
        "dataset_scope": dataset_scope,
        "model_scope": model_scope,
        "uncertainty_method": uncertainty_method,
        "claim_boundary": claim_boundary,
        "renderer": special_renderer,
    }


_CORE = {
    "dataset_roles_target_mappings_support": _definition(
        1,
        "dataset_roles_target_mappings_support",
        "Dataset roles, target mappings, and support",
        scope="core",
        sources=("dataset_cards/dataset_cards.csv", "external_replication/target_support.csv"),
        evaluation_scope="dataset_identity_target_mapping_and_observed_support",
        dataset_scope="source_declared_core_dataset",
        model_scope="not_applicable_dataset_contract",
        uncertainty_method="not_applicable_observed_support",
        claim_boundary="Dataset role, mapping, and observed support only; no target equivalence, source-authenticity, licence, or transport claim.",
    ),
    "exact_primary_feature_policy": _definition(
        2,
        "exact_primary_feature_policy",
        "Exact primary feature policy",
        scope="core",
        sources=("policy_ablation/policy_feature_contract.csv",),
        evaluation_scope="predeclared_feature_inclusion_and_exclusion_contract",
        dataset_scope="inx_primary",
        model_scope="xgboost_policy_contract",
        uncertainty_method="not_applicable_configuration_contract",
        claim_boundary="Predeclared feature-governance evidence only; proxy-rich policies remain audit sensitivities and cannot replace the primary policy.",
    ),
    "four_model_nested_benchmark": _definition(
        3,
        "four_model_nested_benchmark",
        "Four-model nested benchmark",
        scope="core",
        sources=(
            "model_benchmarks/model_summary.csv",
            "model_benchmarks/paired_model_differences.csv",
            "model_benchmarks/selected_hyperparameters.csv",
        ),
        evaluation_scope="ten_outer_by_five_inner_fold_exactly_once_oof_benchmark",
        dataset_scope="inx_primary",
        model_scope="source_declared_benchmark_system_or_outer_fold_model",
        uncertainty_method="paired_stratified_percentile_bootstrap_5000_draws",
        claim_boundary="Within-dataset predictive benchmark only; paired intervals condition on the observed samples and fixed resampling protocol.",
    ),
    "leakage_policy_sensitivity": _definition(
        4,
        "leakage_policy_sensitivity",
        "Leakage-policy sensitivity",
        scope="core",
        sources=(
            "policy_ablation/manuscript_policy_table.csv",
            "policy_ablation/policy_pairwise_tests.csv",
            "policy_ablation/leakage_sensitivity_index.csv",
        ),
        evaluation_scope="same_fold_same_selected_candidate_policy_sensitivity",
        dataset_scope="inx_primary",
        model_scope="xgboost_exact_outer_fold_policy_models",
        uncertainty_method="paired_stratified_percentile_bootstrap_5000_draws",
        claim_boundary="Leakage-risk and feature-policy sensitivity only; audit policies are not alternate primary models and differences are not causal effects.",
    ),
    "cross_fitted_sigmoid_calibration": _definition(
        5,
        "cross_fitted_sigmoid_calibration",
        "Cross-fitted sigmoid calibration",
        scope="core",
        sources=(
            "sigmoid_calibration/calibration_method_comparison.csv",
            "sigmoid_calibration/calibration_metric_intervals.csv",
            "sigmoid_calibration/calibration_paired_differences.csv",
        ),
        evaluation_scope="predeclared_cross_fitted_sigmoid_on_exact_outer_fold_models",
        dataset_scope="inx_primary",
        model_scope="xgboost_primary_policy_exact_outer_fold_models",
        uncertainty_method="paired_stratified_percentile_bootstrap_5000_draws",
        claim_boundary="Calibration evidence for research probabilities only; no autonomous decision threshold or deployment claim.",
    ),
    "global_grouped_oof_shap": _definition(
        6,
        "global_grouped_oof_shap",
        "Global exact-fold OOF grouped SHAP attribution",
        scope="core",
        sources=("oof_shap/global_grouped_shap_importance.csv",),
        evaluation_scope="exact_fold_oof_grouped_shap_same_model_as_prediction",
        dataset_scope="inx_primary",
        model_scope="xgboost_primary_policy_exact_outer_fold_models",
        uncertainty_method="descriptive_complete_oof_attribution_aggregation",
        claim_boundary="Model attribution only; SHAP values do not establish causality, fairness, actionability, or employee-level advice.",
    ),
    "oof_shap_stability": _definition(
        7,
        "oof_shap_stability",
        "Descriptive OOF SHAP stability",
        scope="core",
        sources=("oof_shap/shap_stability_pairwise.csv", "oof_shap/shap_stability_summary.csv"),
        evaluation_scope="dependent_outer_fold_ranking_comparison",
        dataset_scope="inx_primary",
        model_scope="xgboost_primary_policy_exact_outer_fold_models",
        uncertainty_method="descriptive_dependent_fold_pairs_no_confidence_interval",
        claim_boundary="Descriptive fold-ranking stability only; dependent fold pairs do not support population confidence intervals.",
    ),
    "support_aware_subgroup_diagnostics": _definition(
        8,
        "support_aware_subgroup_diagnostics",
        "Support-aware subgroup diagnostics",
        scope="core",
        sources=(
            "subgroup_proxy/fairness_group_support_and_metrics.csv",
            "subgroup_proxy/fairness_disparity_uncertainty.csv",
        ),
        evaluation_scope="support_gated_descriptive_oof_subgroup_diagnostics",
        dataset_scope="inx_primary",
        model_scope="source_declared_xgboost_policy",
        uncertainty_method="pointwise_paired_stratified_percentile_bootstrap_5000_draws",
        claim_boundary="Support-aware descriptive subgroup diagnostics only; no discrimination, fairness, or causal conclusion.",
    ),
    "department_proxy_reconstructability": _definition(
        9,
        "department_proxy_reconstructability",
        "Department proxy reconstructability",
        scope="core",
        sources=(
            "subgroup_proxy/proxy_metric_intervals.csv",
            "subgroup_proxy/proxy_policy_paired_differences.csv",
            "subgroup_proxy/proxy_policy_comparison.csv",
            "subgroup_proxy/proxy_feature_contracts.csv",
        ),
        evaluation_scope="nominal_multiclass_proxy_risk_diagnostic",
        dataset_scope="inx_primary",
        model_scope="proxy_diagnostic_models_only",
        uncertainty_method="paired_stratified_percentile_bootstrap_5000_draws",
        claim_boundary="Department reconstructability is a proxy-risk diagnostic, not employee-performance validation, fairness proof, discrimination evidence, or causality.",
    ),
    "hrdataset_v14_mapped_target_replication": _definition(
        10,
        "hrdataset_v14_mapped_target_replication",
        "HRDataset_v14 independent mapped-target replication",
        scope="core",
        sources=(
            "external_replication/target_support.csv",
            "external_replication/raw_metric_intervals.csv",
            "external_replication/calibration_metric_intervals.csv",
            "external_replication/calibration_paired_differences.csv",
            "external_replication/policy_pairwise_differences.csv",
            "external_replication/external_replication_metadata.json",
        ),
        evaluation_scope="independent_nested_mapped_target_external_replication",
        dataset_scope="hrdataset_v14",
        model_scope="independently_trained_xgboost_exact_outer_fold_models",
        uncertainty_method="paired_stratified_percentile_bootstrap_5000_draws",
        claim_boundary="Independent mapped-target replication only; not locked-model transport, target equivalence, deployment validation, or causal evidence.",
    ),
    "reproducibility_and_claim_boundaries": _definition(
        13,
        "reproducibility_and_claim_boundaries",
        "Reproducibility and claim boundaries",
        scope="core",
        sources=("run_inputs/input_contract.json", "run_inputs/canonical_config_snapshot.yaml"),
        evaluation_scope="machine_readable_metric_and_claim_contract",
        dataset_scope="all_core_datasets_as_declared",
        model_scope="all_core_models_as_declared",
        uncertainty_method="registry_declared_per_metric",
        claim_boundary="Contract and provenance table only; it supplies no new scientific estimate or release-readiness claim.",
        special_renderer="metric_and_claim_registry",
    ),
}

_SUPPLEMENTARY = {
    "heuristic_counterfactual_search_success": _definition(
        11,
        "heuristic_counterfactual_search_success",
        "Supplementary heuristic counterfactual search success",
        scope="supplementary",
        sources=(
            "heuristic_counterfactual/heuristic_search_summary.csv",
            "heuristic_counterfactual/heuristic_search_uncertainty.csv",
            "heuristic_counterfactual/heuristic_search_failure_reasons.csv",
            "heuristic_counterfactual/heuristic_search_budget_sensitivity.csv",
        ),
        evaluation_scope="all_eligible_exact_fold_oof_heuristic_search_cases",
        dataset_scope="inx_primary",
        model_scope="xgboost_primary_policy_exact_outer_fold_models",
        uncertainty_method="wilson_95_interval_and_conditional_case_bootstrap_5000_draws",
        claim_boundary="Supplementary heuristic search-success evidence only; not causal recourse, feasibility, advice, recommendation, or deployment evidence.",
    ),
    "restricted_and_binary_task_evidence": _definition(
        12,
        "restricted_and_binary_task_evidence",
        "Supplementary restricted-target and binary-task evidence",
        scope="supplementary",
        sources=(
            "external_robustness/task_strata_index.csv",
            "external_robustness/metric_applicability.csv",
            "external_robustness/ibm_restricted_target_performance_robustness.csv",
            "external_robustness/ibm_attrition_task_transfer.csv",
            "external_robustness/employee_turnover_task_transfer.csv",
        ),
        evaluation_scope="task_bounded_independent_nested_robustness_and_transfer",
        dataset_scope="source_declared_supplementary_dataset",
        model_scope="independently_trained_task_specific_xgboost_models",
        uncertainty_method="paired_stratified_percentile_bootstrap_5000_draws_within_task",
        claim_boundary="Task-bounded robustness or related-task transfer only; no cross-task aggregation, primary-task validation, transportability, or deployment claim.",
    ),
    "supplementary_reproducibility_and_claim_boundaries": _definition(
        13,
        "supplementary_reproducibility_and_claim_boundaries",
        "Supplementary reproducibility and claim boundaries",
        scope="supplementary",
        sources=("run_inputs/input_contract.json", "run_inputs/canonical_config_snapshot.yaml"),
        evaluation_scope="machine_readable_metric_and_claim_contract",
        dataset_scope="all_supplementary_datasets_as_declared",
        model_scope="all_supplementary_models_as_declared",
        uncertainty_method="registry_declared_per_metric",
        claim_boundary="Contract and provenance table only; it supplies no new scientific estimate or release-readiness claim.",
        special_renderer="metric_and_claim_registry",
    ),
}

EXPECTED_TABLE_PLANS: Mapping[str, Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {"core": MappingProxyType(_CORE), "supplementary": MappingProxyType(_SUPPLEMENTARY)}
)


def expected_table_plan(scope: str) -> dict[str, dict[str, Any]]:
    try:
        return copy.deepcopy(dict(EXPECTED_TABLE_PLANS[scope]))
    except KeyError as exc:
        raise TableContractError(f"Unknown table evidence scope: {scope!r}.") from exc


def validate_table_plan(configured: Any, *, scope: str) -> dict[str, dict[str, Any]]:
    if not isinstance(configured, Mapping):
        raise TableContractError(f"Configured {scope} table definitions must be a mapping.")
    expected = expected_table_plan(scope)
    observed = copy.deepcopy(dict(configured))
    if observed != expected:
        raise TableContractError(f"Configured {scope} table definitions differ from the frozen v2 plan.")
    filenames = [str(value["filename"]) for value in observed.values()]
    if len(set(filenames)) != len(filenames):
        raise TableContractError(f"Configured {scope} table filenames are not unique.")
    return observed


def validate_table_plan_declaration(configured: Any) -> None:
    """Validate the compact config projection of the code-owned exact plan."""

    if not isinstance(configured, Mapping):
        raise TableContractError("manuscript_final.tables must be a mapping.")
    expected_top = {
        "plan_version": TABLE_PLAN_VERSION,
        "contract_source": "src.governance.table_contract.EXPECTED_TABLE_PLANS",
        "release_ready": True,
        "blocking_reason": "Production source-table generation and closed-world validation are implemented; canonical artifacts remain pending the final clean real-data builds.",
        "identity_fields": list(TABLE_IDENTITY_FIELDS),
        "source_hash_algorithm": "sha256",
    }
    for field, expected in expected_top.items():
        if configured.get(field) != expected:
            raise TableContractError(f"Configured table-plan field {field!r} differs from the frozen contract.")
    if set(configured) != {*expected_top, "scopes"}:
        raise TableContractError("Configured table-plan fields differ from the frozen declaration.")
    scopes = configured.get("scopes")
    if not isinstance(scopes, Mapping) or set(scopes) != set(EXPECTED_TABLE_PLANS):
        raise TableContractError("Configured table-plan scopes differ from the frozen contract.")
    for scope, definitions in EXPECTED_TABLE_PLANS.items():
        declaration = scopes.get(scope)
        if not isinstance(declaration, Mapping):
            raise TableContractError(f"Configured table scope {scope!r} must be a mapping.")
        expected = {
            "definition_keys": list(definitions),
            "table_numbers": [int(value["number"]) for value in definitions.values()],
            "filenames": [str(value["filename"]) for value in definitions.values()],
        }
        if dict(declaration) != expected:
            raise TableContractError(f"Configured table scope {scope!r} differs from the frozen plan.")
