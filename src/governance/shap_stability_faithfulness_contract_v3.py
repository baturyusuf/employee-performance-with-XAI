"""Fail-closed validation for the v3 SHAP stability and faithfulness contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import shap

from src.data.canonical_loader import load_canonical_dataset, sha256_file
from src.experiments.benchmark_artifact_contract import read_xgboost_oof_artifacts
from src.utils.config_loader import PROJECT_ROOT


DEFAULT_SHAP_STABILITY_FAITHFULNESS_CONTRACT = Path(
    "configs/shap_stability_faithfulness_v3.json"
)
CANONICAL_V2_ROOT = Path(
    "reports/manuscript_final/canonical_v2_20260714T221501Z_483f96f"
)
POLICY_IDS = ("P3",)
P3_FEATURE_COUNT = 20
TOP_K_VALUES = (5, 10, 15)
DELETION_COUNTS = (1, 3, 5)
EXPECTED_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "contract_id",
        "dataset_key",
        "target",
        "ordered_labels",
        "policy_id",
        "policy_name",
        "model",
        "purpose",
        "source_contracts",
        "canonical_identity",
        "grouped_shap_implementation",
        "seed_stability",
        "resampling_stability",
        "stability_evaluation",
        "faithfulness",
        "computational_scope",
        "publication",
    }
)


class ShapStabilityFaithfulnessContractError(RuntimeError):
    """Raised when the Phase 2A contract drifts from the frozen design."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ShapStabilityFaithfulnessContractError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ShapStabilityFaithfulnessContractError(
            f"Could not read {path.as_posix()}: {exc}"
        ) from exc
    _require(isinstance(value, dict), "Phase 2A contract must be a JSON object.")
    return value


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _exact(mapping: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, value in expected.items():
        _require(mapping.get(key) == value, f"{label} drifted for {key}.")


def _policy_features(feature_contract: Mapping[str, Any]) -> list[str]:
    all_features = [str(record["feature_name"]) for record in feature_contract["features"]]
    policies = {str(record["policy_id"]): record for record in feature_contract["policies"]}
    _require("P3" in policies, "P3 feature policy is absent.")
    excluded = set(map(str, policies["P3"]["excluded_features"]))
    retained = [feature for feature in all_features if feature not in excluded]
    _require(len(retained) == P3_FEATURE_COUNT, "P3 feature count drifted.")
    _require("PerformanceRating" not in retained and "EmpNumber" not in retained, "P3 retains a forbidden field.")
    return retained


def _validate_local_sources(contract: Mapping[str, Any]) -> dict[str, Any]:
    sources = contract["source_contracts"]
    for name, record in sources.items():
        _require(set(record) == {"path", "sha256"}, f"Source record schema drifted for {name}.")
        _require(_digest(record["sha256"]), f"Source digest is invalid for {name}.")
        path = PROJECT_ROOT / str(record["path"])
        _require(path.is_file(), f"Required Phase 2A source is absent: {name}.")
        _require(sha256_file(path) == record["sha256"], f"Phase 2A source hash drifted: {name}.")

    feature_contract = _load(PROJECT_ROOT / sources["feature_availability"]["path"])
    retained = _policy_features(feature_contract)
    canonical = load_canonical_dataset(
        sources["canonical_loader_config"]["path"],
        "inx_primary",
        sources["acquisition_manifest"]["path"],
        allow_download=False,
    )
    identity = contract["canonical_identity"]
    _require(canonical.receipt["actual_sha256"] == identity["dataset_sha256"], "Canonical dataset identity drifted.")
    _require(len(canonical.frame) == 1200, "Canonical sample count drifted.")
    _require(set(canonical.frame["PerformanceRating"].astype(int)) == {2, 3, 4}, "Target support drifted.")

    artifacts = read_xgboost_oof_artifacts(
        CANONICAL_V2_ROOT / "core/shared_folds",
        CANONICAL_V2_ROOT / "core/model_benchmarks",
        expected_run_id=identity["run_id"],
        expected_config_hash=identity["config_hash"],
        expected_scientific_input_hash=identity["scientific_input_hash"],
        expected_feature_columns=retained,
        expected_labels=(2, 3, 4),
    )
    _require(artifacts.model_set_sha256 == identity["xgboost_model_set_sha256"], "Canonical XGBoost model-set identity drifted.")
    _require(artifacts.identity.fold_contract_hash == identity["fold_contract_hash"], "Canonical fold identity drifted.")
    _require(len(artifacts.fold_models) == 10 and len(artifacts.oof_predictions) == 1200, "Canonical OOF model coverage drifted.")

    shap_metadata = _load(PROJECT_ROOT / sources["canonical_v2_shap_metadata"]["path"])
    _require(shap_metadata["model_set_sha256"] == identity["xgboost_model_set_sha256"], "Canonical SHAP/model identity drifted.")
    _require(shap_metadata["protocol"]["evaluation"] == "exact_prediction_producing_outer_fold_models", "Canonical SHAP evaluation source drifted.")
    _require(shap_metadata["protocol"]["model_refit_in_shap_stage"] is False, "Canonical SHAP unexpectedly refit models.")
    _require(tuple(shap_metadata["protocol"]["top_k_values"]) == TOP_K_VALUES, "Canonical fold-pair top-k grid drifted.")
    return {
        "available": True,
        "validated": True,
        "sample_count": len(canonical.frame),
        "feature_count": len(retained),
        "outer_model_count": len(artifacts.fold_models),
        "canonical_model_set_sha256": artifacts.model_set_sha256,
    }


def validate_shap_stability_faithfulness_contract_v3(
    contract_path: Path | str = DEFAULT_SHAP_STABILITY_FAITHFULNESS_CONTRACT,
) -> dict[str, Any]:
    """Validate the exact Phase 2A design without fitting a model."""

    path = Path(contract_path)
    contract = _load(path)
    _require(set(contract) == EXPECTED_TOP_LEVEL, "Phase 2A top-level inventory drifted.")
    _exact(
        contract,
        {
            "schema_version": 1,
            "contract_id": "shap_stability_faithfulness_v3",
            "dataset_key": "inx_primary",
            "target": "PerformanceRating",
            "ordered_labels": [2, 3, 4],
            "policy_id": "P3",
            "policy_name": "PRIMARY_LEAKAGE_AWARE",
            "model": "xgboost",
            "purpose": "separate_grouped_shap_aggregation_validity_ranking_stability_and_model_level_deletion_faithfulness",
        },
        "Phase 2A identity",
    )
    identity = contract["canonical_identity"]
    _exact(
        identity,
        {
            "run_id": "canonical_v2_20260714T221501Z_483f96f",
            "generation_commit": "483f96fdbaab16cb0f32d03d9dbe676a759af44a",
            "config_hash": "51415c2ce68c89d9ce2b042b0a7a811fe3e98180b72460006f9d0465f6bf49b7",
            "scientific_input_hash": "06c507bee525ea1daca43b61249764007d4d8baaa05c9333f23446ea723ce160",
            "dataset_sha256": "b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a",
            "fold_contract_hash": "c1300316fe5baec24e789c06aec35dd4f283fa4843b71c7aab1edbf4818f8e91",
            "xgboost_model_set_sha256": "d483d8ddb93a47a99a2eab54fe3d138d8f614798dd418bca860ea3f20c813f51",
            "sample_count": 1200,
            "feature_count": 20,
            "outer_folds": 10,
        },
        "Canonical identity",
    )
    implementation = contract["grouped_shap_implementation"]
    _exact(
        implementation,
        {
            "library": "shap",
            "required_library_version": "0.51.0",
            "explainer": "shap.TreeExplainer",
            "background_data": "none",
            "feature_perturbation": "tree_path_dependent",
            "model_output": "raw_margin",
            "check_additivity": True,
            "maximum_additivity_absolute_error": 0.00001,
            "normalized_axis_order": "sample_class_transformed_feature",
            "transformed_to_raw_operation": "signed_sum_within_each_raw_feature_family",
            "class_operation_for_global_importance": "mean_of_absolute_grouped_values_across_classes",
            "sample_operation_for_global_importance": "mean_across_exactly_once_oof_samples",
            "absolute_before_transformed_grouping": False,
            "grouped_sum_preservation_required_per_sample_class": True,
            "probability_space_explanation_allowed": False,
        },
        "Grouped SHAP implementation",
    )
    _require(shap.__version__ == implementation["required_library_version"], "Installed SHAP version drifted.")

    seed = contract["seed_stability"]
    _exact(
        seed,
        {
            "reference_run": {"run_label": "canonical_seed_42", "model_seed": 42, "model_source": "reuse_exact_canonical_v2_outer_models"},
            "outer_assignments": "exact_canonical_v2",
            "training_rows": "complete_outer_training_partition",
            "candidate_schedule": "exact_canonical_v2_P3_fold_specific_xgboost_schedule",
            "only_intended_change": "xgboost_model_seed",
            "new_estimator_fit_calls": 50,
        },
        "Seed stability",
    )
    expected_seed_runs = [
        {"run_label": f"seed_{value}", "model_seed": value}
        for value in (1044, 2044, 3044, 4044, 5044)
    ]
    _require(seed.get("new_runs") == expected_seed_runs, "Seed-stability run registry drifted.")

    resampling = contract["resampling_stability"]
    _exact(
        resampling,
        {
            "method": "stratified_without_replacement_subsample_within_each_outer_training_partition",
            "training_fraction": 0.8,
            "subsample_seeds": [11042, 12042, 13042, 14042, 15042],
            "per_fold_seed_rule": "subsample_seed_plus_outer_fold",
            "model_seed": 42,
            "outer_test_rows": "exact_canonical_v2_outer_test_partition",
            "candidate_schedule": "exact_canonical_v2_P3_fold_specific_xgboost_schedule",
            "selection_repeated": False,
            "new_estimator_fit_calls": 50,
        },
        "Resampling stability",
    )
    stability = contract["stability_evaluation"]
    _exact(
        stability,
        {
            "ranking_unit": "global_mean_absolute_grouped_raw_feature_family_shap",
            "top_k_values": [5, 10, 15],
            "set_metric": "jaccard",
            "all_feature_rank_metric": "spearman",
            "pairwise_values_are_independent": False,
            "confidence_interval_applicable": False,
            "strong_robust_explanation_claim_allowed": False,
        },
        "Stability evaluation",
    )

    faithfulness = contract["faithfulness"]
    _exact(
        faithfulness,
        {
            "model_source": "exact_prediction_producing_canonical_v2_outer_fold_models",
            "sample_scope": "every_exactly_once_outer_test_sample",
            "ranking_scope": "absolute_grouped_shap_for_original_predicted_class_per_sample",
            "deletion_feature_counts": [1, 3, 5],
            "mask_reference": "outer_training_partition_numeric_median_and_categorical_mode",
            "categorical_mode_tie_break": "canonical_sorted_first",
            "random_baseline_repetitions": 20,
            "random_baseline_seeds": list(range(21042, 21062)),
            "random_feature_sampling": "without_replacement_same_feature_count_per_sample",
            "primary_output": "original_predicted_class_probability_drop",
            "secondary_output": "original_predicted_class_raw_margin_drop",
            "deletion_auc": "trapezoidal_probability_drop_over_deleted_fraction_of_five_using_counts_0_1_3_5",
            "guided_vs_random_comparison": "guided_mean_minus_distribution_of_random_repetition_means",
            "masking_creates_possible_out_of_distribution_hybrids": True,
            "human_usefulness_claim_allowed": False,
            "causal_feature_effect_claim_allowed": False,
            "prescriptive_hr_claim_allowed": False,
            "allowed_claim_scope": "model_level_explanation_faithfulness_only",
        },
        "Faithfulness",
    )
    computation = contract["computational_scope"]
    _exact(
        computation,
        {
            "seed_stability_new_fit_calls": 50,
            "resampling_stability_new_fit_calls": 50,
            "total_new_estimator_fit_calls": 100,
            "stability_model_fold_explanations_including_reference": 110,
            "guided_sample_perturbations": 3600,
            "random_sample_perturbations": 72000,
        },
        "Computational scope",
    )
    publication = contract["publication"]
    _require(publication.get("local_output_root") == "reports/major_revision_v3_runs", "Publication root drifted.")
    for field in (
        "publish_local_shap_rows",
        "publish_sample_level_faithfulness_rows",
        "publish_resample_memberships",
        "publish_fitted_models",
    ):
        _require(publication.get(field) is False, f"Publication control {field} drifted.")
    _require(
        publication.get("compact_outputs")
        == ["aggregation_receipt", "stability_summary", "faithfulness_summary", "faithfulness_contrasts", "deletion_auc_summary", "provenance_receipt"],
        "Compact-output allowlist drifted.",
    )
    required_limitations = {
        "fold_or_resample_pair_dependence",
        "masking_out_of_distribution_risk",
        "stability_is_not_faithfulness",
        "model_attribution_is_not_causal",
        "no_human_usefulness_evidence",
    }
    _require(set(publication.get("mandatory_limitations", [])) == required_limitations, "Mandatory limitations drifted.")
    prohibited = set(publication.get("prohibited_claims", []))
    _require({"causal_feature_importance", "human_explanation_usefulness", "fairness", "deployment_ready_hr_decision_system"}.issubset(prohibited), "Prohibited-claim registry drifted.")

    local = _validate_local_sources(contract)
    return {
        "status": "passed",
        "contract_sha256": sha256_file(path),
        "sample_count": 1200,
        "feature_count": 20,
        "seed_stability_run_count_including_reference": 6,
        "resampling_run_count": 5,
        "top_k_values": list(TOP_K_VALUES),
        "deletion_feature_counts": list(DELETION_COUNTS),
        "random_baseline_repetitions": 20,
        "planned_new_estimator_fit_calls": 100,
        "local_canonical_sources": local,
        "model_fit_count": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def main() -> int:
    print(json.dumps(validate_shap_stability_faithfulness_contract_v3(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
