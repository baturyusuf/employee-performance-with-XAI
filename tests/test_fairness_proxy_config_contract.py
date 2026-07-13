from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.governance.manuscript_contract import (
    ManuscriptConfigError,
    load_manuscript_config,
    validate_manuscript_config,
)


CONFIG_PATH = Path("configs/manuscript_final.yaml")


def test_fairness_contract_requires_exact_policy_oof_and_central_5000_draws() -> None:
    settings = load_manuscript_config(CONFIG_PATH)["manuscript_final"]
    fairness = settings["fairness"]

    assert fairness["prediction_contract"] == {
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
    }
    assert fairness["bootstrap_stratify_by"] == ["outer_fold", "y_true"]
    assert settings["evaluation"]["bootstrap"]["n_resamples"] == 5000
    assert settings["evaluation"]["bootstrap"]["conditional_inference_note"] == (
        "Intervals condition on the observed employees and fixed fold/model-training "
        "protocol; they do not estimate model-training instability."
    )
    assert fairness["bootstrap_contract"]["same_resamples_across_policies"] is True
    assert fairness["bootstrap_contract"]["resample_hash_required"] is True
    assert fairness["bootstrap_contract"]["resample_hash_source"] == (
        "policy_ablation.bootstrap_metadata.resample_hash"
    )
    assert fairness["bootstrap_contract"]["resample_hash_equality_required_with"] == (
        "model_benchmarks.baseline_xgboost_gate.resample_hash"
    )
    assert fairness["support_status_rules"]["eligibility_scope"] == (
        "fixed_from_complete_oof_before_resampling"
    )
    assert fairness["support_status_rules"]["paired_policy_common_group_scope"] == (
        "intersection_of_complete_oof_eligible_groups_per_pair_attribute_metric_class"
    )
    assert fairness["support_status_rules"]["paired_status_values"][0] == (
        "insufficient_common_subgroup_or_metric_support"
    )
    assert fairness["inference_scope"]["multiplicity_adjustment"] == "none"
    assert fairness["inference_scope"]["simultaneous_or_familywise_claims_allowed"] is False


@pytest.mark.parametrize(
    ("section", "path", "value"),
    [
        ("fairness", ("prediction_contract", "model_refit_in_stage"), True),
        ("fairness", ("bootstrap_stratify_by",), ["y_true"]),
        ("fairness", ("headline_rules", "wide_interval_rows_headline_eligible"), True),
        ("fairness", ("support_status_rules", "eligibility_scope"), "per_resample"),
        ("fairness", ("inference_scope", "multiplicity_adjustment"), "holm"),
        ("proxy_analysis", ("outer_folds_source",), "independent_proxy_splitter"),
        ("proxy_analysis", ("task_type",), "restricted_target_performance_robustness"),
        ("proxy_analysis", ("bootstrap", "n_resamples"), 1000),
        ("proxy_analysis", ("oof_contract", "proxy_target_absent_from_predictors"), False),
        (
            "proxy_analysis",
            ("bootstrap", "semantic_strata_adapter", "performance_target_used"),
            True,
        ),
    ],
)
def test_fairness_proxy_protocol_drift_fails_config_validation(
    section: str,
    path: tuple[str, ...],
    value: object,
) -> None:
    malformed = copy.deepcopy(load_manuscript_config(CONFIG_PATH))
    target = malformed["manuscript_final"][section]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ManuscriptConfigError, match="fairness differs|proxy_analysis differs"):
        validate_manuscript_config(malformed)


def test_proxy_contract_has_two_unique_fits_and_one_explicit_alias() -> None:
    proxy = load_manuscript_config(CONFIG_PATH)["manuscript_final"]["proxy_analysis"]

    assert proxy["task_type"] == "nominal_multiclass_proxy_diagnostic"
    assert set(proxy["unique_predictor_contracts"]) == {
        "no_salary_hike_no_attrition_no_department",
        "no_salary_hike_no_attrition_no_department_no_job_role",
    }
    assert proxy["policy_aliases"] == {
        "no_salary_hike_no_attrition": "no_salary_hike_no_attrition_no_department"
    }
    assert proxy["target_removed_from_all_proxy_predictors"] is True
    assert proxy["bootstrap"]["stratify_by"] == ["outer_fold", "proxy_target"]
    assert proxy["bootstrap"]["n_resamples"] == 5000
    assert proxy["bootstrap"]["separate_from_performance_policy_bootstrap"] is True
    assert proxy["bootstrap"]["batch_size_source"] == "fairness.bootstrap_batch_size"
    assert proxy["bootstrap"]["semantic_strata_adapter"]["performance_target_used"] is False
    assert proxy["inference_scope"]["paired_rows_headline_eligible"] is False
    assert proxy["fold_summary_scope"] == (
        "descriptive_mean_std_min_max_only_no_population_ci"
    )


@pytest.mark.parametrize("seed_name", ["bootstrap", "fairness"])
def test_required_subgroup_proxy_seed_reference_cannot_be_removed(seed_name: str) -> None:
    malformed = copy.deepcopy(load_manuscript_config(CONFIG_PATH))
    malformed["manuscript_final"]["seeds"].pop(seed_name)

    with pytest.raises(ManuscriptConfigError, match="seed references are missing"):
        validate_manuscript_config(malformed)


@pytest.mark.parametrize("seed_name", ["bootstrap", "fairness"])
def test_boolean_subgroup_proxy_seed_is_rejected(seed_name: str) -> None:
    malformed = copy.deepcopy(load_manuscript_config(CONFIG_PATH))
    malformed["manuscript_final"]["seeds"][seed_name] = True

    with pytest.raises(ManuscriptConfigError, match="explicit integers"):
        validate_manuscript_config(malformed)


def test_bootstrap_conditional_inference_scope_cannot_drift() -> None:
    malformed = copy.deepcopy(load_manuscript_config(CONFIG_PATH))
    malformed["manuscript_final"]["evaluation"]["bootstrap"][
        "conditional_inference_note"
    ] = "model training uncertainty included"

    with pytest.raises(ManuscriptConfigError, match="evaluation.bootstrap differs"):
        validate_manuscript_config(malformed)
