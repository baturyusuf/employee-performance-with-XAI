"""Fail-closed validator for the v3 fixed-versus-retuned policy contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.experiments.manuscript_model_benchmark import validate_benchmark_config
from src.experiments.shared_folds import read_shared_folds
from src.governance.feature_availability_contract import (
    validate_feature_availability_contract,
)
from src.governance.ordinal_benchmark_contract_v3 import EXPECTED_AGGREGATE_METRICS
from src.utils.config_loader import load_config


DEFAULT_POLICY_RETUNING_CONTRACT_PATH = Path("configs/policy_retuning_v3.json")
CANONICAL_V2_RUN_ID = "canonical_v2_20260714T221501Z_483f96f"
CANONICAL_V2_ROOT = Path("reports/manuscript_final") / CANONICAL_V2_RUN_ID
POLICY_IDS = ("P0", "P1", "P2", "P3", "P4", "P5")
POLICY_NAMES = (
    "INFORMATION_RICH_DIAGNOSTIC",
    "LEAKAGE_CONTROLLED",
    "GOVERNANCE_CONTROLLED",
    "PRIMARY_LEAKAGE_AWARE",
    "STRICT_PROSPECTIVE",
    "STRICT_PROXY",
)
POLICY_FEATURE_COUNTS = (26, 24, 21, 20, 13, 6)
HEADLINE_METRICS = (
    "macro_f1",
    "quadratic_weighted_kappa",
    "balanced_accuracy",
    "ordinal_mae",
)
EXPECTED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "contract_id",
        "dataset_key",
        "task_type",
        "target",
        "ordered_labels",
        "purpose",
        "source_contracts",
        "information_contract",
        "fixed_evidence_crosswalk",
        "historical_policy_boundary",
        "design",
        "model",
        "fixed_hyperparameter_estimand",
        "independently_retuned_estimand",
        "computational_scope",
        "evaluation",
        "publication",
    }
)
EXPECTED_SOURCE_RECORDS = {
    "canonical_loader_config": (
        "configs/manuscript_final.yaml",
        "f8a6354ab67757c5d92c93cef2fe03ab88f4b16b10ce470f2db97ed4d0407779",
        True,
    ),
    "acquisition_manifest": (
        "configs/data_acquisition.yaml",
        "a5e820a7213967e494bc9238f104e478c5a65c0840d654397cc3063314262d97",
        True,
    ),
    "feature_availability": (
        "configs/feature_availability_v3.json",
        "6dd5fdde534e379cceacfaa01e865d1551310fb632b691f5b937ef39394e93cf",
        True,
    ),
    "xgboost_candidate_registry": (
        "configs/model_grid.yaml",
        "d8fb0584d7106f8941ca8e1b0e0a9c58e13f39519c6c75b383ff333d92d41617",
        True,
    ),
    "canonical_v2_receipt": (
        "reports/research_log/finalization_v2/15_canonical_evidence_receipt.json",
        "bd09523276dcbc54d036f8c9eb0c71ba62f8b0f913c495695a2259296101827c",
        True,
    ),
    "canonical_v2_fold_contract": (
        f"reports/manuscript_final/{CANONICAL_V2_RUN_ID}/core/shared_folds/fold_contract.json",
        "364b0fc0443dbe1d7eceeb82e383802aa4b1b91f268f09703aa8a9a3a7f41d28",
        False,
    ),
    "canonical_v2_outer_assignments": (
        f"reports/manuscript_final/{CANONICAL_V2_RUN_ID}/core/shared_folds/fold_assignments.csv",
        "6b22a827cdba12dff40162f784291399614fb2a9e9ffe1ac889545767129204f",
        False,
    ),
    "canonical_v2_inner_assignments": (
        f"reports/manuscript_final/{CANONICAL_V2_RUN_ID}/core/shared_folds/inner_fold_assignments.csv",
        "35c1afc7d2c8debb9ce54a4be3ac06d9830a2946c41fd756d3532a0f54d32e89",
        False,
    ),
    "canonical_v2_selected_hyperparameters": (
        f"reports/manuscript_final/{CANONICAL_V2_RUN_ID}/core/model_benchmarks/selected_hyperparameters.csv",
        "5ee6b2c3b43453aa3f54bbc79fa58fdf6f6aa4eae35d02174232cb25f51615f1",
        False,
    ),
    "canonical_v2_benchmark_oof": (
        f"reports/manuscript_final/{CANONICAL_V2_RUN_ID}/core/model_benchmarks/oof_predictions.csv",
        "9b769838b0a55feda162fec949ad17c9dc78141077cd6840598081fb77089104",
        False,
    ),
    "canonical_v2_fixed_policy_oof": (
        f"reports/manuscript_final/{CANONICAL_V2_RUN_ID}/core/policy_ablation/oof_predictions.csv",
        "299e1b8688b565daee71180e6197862788f027277d77957d7d523722e68a6a4d",
        False,
    ),
    "canonical_v2_policy_feature_contract": (
        f"reports/manuscript_final/{CANONICAL_V2_RUN_ID}/core/policy_ablation/policy_feature_contract.csv",
        "998cd252539ca546943957bbb9e1c7fde7a87e39fb2efb23a27520b16da97f09",
        False,
    ),
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class PolicyRetuningContractError(ValueError):
    """Raised when policy, source, estimand, or claim separation drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyRetuningContractError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyRetuningContractError(
            f"Could not read policy-retuning contract {path.as_posix()}: {exc}"
        ) from exc
    _require(isinstance(payload, dict), "Policy-retuning contract must be a JSON object.")
    return payload


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object.")
    return value


def _exact_list(value: Any, expected: Sequence[Any], *, name: str) -> None:
    _require(
        isinstance(value, list) and tuple(value) == tuple(expected),
        f"{name} must equal the exact ordered contract {list(expected)}.",
    )


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PolicyRetuningContractError(f"Could not hash {path.as_posix()}: {exc}") from exc


def _feature_sets() -> tuple[dict[str, list[str]], dict[str, Mapping[str, Any]]]:
    payload = _load_json(Path("configs/feature_availability_v3.json"))
    all_features = [str(record["feature_name"]) for record in payload["features"]]
    records = {str(record["policy_id"]): record for record in payload["policies"]}
    feature_sets = {
        policy_id: [
            feature
            for feature in all_features
            if feature not in set(map(str, records[policy_id]["excluded_features"]))
        ]
        for policy_id in POLICY_IDS
    }
    return feature_sets, records


def _validate_local_canonical_sources(
    sources: Mapping[str, Any],
    feature_sets: Mapping[str, Sequence[str]],
    crosswalk: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate ignored v2 inputs when present; execution always requires them."""

    if not CANONICAL_V2_ROOT.is_dir():
        return {"available": False, "validated": False}
    local_keys = [name for name, (_, _, tracked) in EXPECTED_SOURCE_RECORDS.items() if not tracked]
    for name in local_keys:
        record = _mapping(sources[name], name=f"source_contracts.{name}")
        _require(_sha256_file(Path(str(record["path"]))) == record["sha256"], f"Local canonical-v2 hash drifted for {name}.")
    folds = read_shared_folds(CANONICAL_V2_ROOT / "core/shared_folds")
    _require(folds.contract["fold_contract_hash"] == "c1300316fe5baec24e789c06aec35dd4f283fa4843b71c7aab1edbf4818f8e91", "Canonical-v2 fold identity drifted.")
    _require(len(folds.outer_assignments) == 1200, "Canonical-v2 fold sample count drifted.")
    _require(len(folds.inner_assignments) == 10 * 1080, "Canonical-v2 inner assignment count drifted.")

    feature_contract = pd.read_csv(sources["canonical_v2_policy_feature_contract"]["path"])
    fixed_oof = pd.read_csv(sources["canonical_v2_fixed_policy_oof"]["path"])
    benchmark_oof = pd.read_csv(sources["canonical_v2_benchmark_oof"]["path"])
    for record in crosswalk[:4]:
        policy_id = str(record["policy_id"])
        source_policy = str(record["source_policy"])
        feature_rows = feature_contract[feature_contract["policy"] == source_policy]
        _require(len(feature_rows) == 1, f"Canonical-v2 feature crosswalk is not unique for {policy_id}.")
        observed_features = json.loads(str(feature_rows.iloc[0]["feature_columns_json"]))
        _require(observed_features == list(feature_sets[policy_id]), f"Canonical-v2 feature crosswalk drifted for {policy_id}.")
        rows = fixed_oof[fixed_oof["system_id"] == source_policy]
        _require(len(rows) == 1200 and rows["sample_index"].nunique() == 1200, f"Fixed OOF coverage drifted for {policy_id}.")
        probability = rows[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float)
        _require(np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-9), f"Fixed OOF probability simplex drifted for {policy_id}.")

    fixed_primary = fixed_oof[fixed_oof["system_id"] == str(crosswalk[3]["source_policy"])]
    benchmark_primary = benchmark_oof[benchmark_oof["model"] == "xgboost"]
    columns = [
        "sample_index",
        "outer_fold",
        "y_true",
        "y_pred",
        "selected_candidate_index",
        "prob_class_2",
        "prob_class_3",
        "prob_class_4",
    ]
    try:
        pd.testing.assert_frame_equal(
            fixed_primary[columns].sort_values("sample_index").reset_index(drop=True),
            benchmark_primary[columns].sort_values("sample_index").reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise PolicyRetuningContractError(
            f"Canonical-v2 P3 fixed/benchmark replay drifted: {exc}"
        ) from exc
    return {
        "available": True,
        "validated": True,
        "sample_count": 1200,
        "outer_splits": 10,
        "inner_splits": 5,
        "exact_fixed_policy_crosswalk_count": 4,
        "primary_replay_maximum_tolerance": 1e-12,
    }


def validate_policy_retuning_contract_v3(
    contract_path: Path | str = DEFAULT_POLICY_RETUNING_CONTRACT_PATH,
) -> dict[str, Any]:
    """Validate the complete Phase 1D estimand and evidence-source separation."""

    path = Path(contract_path)
    contract = _load_json(path)
    _require(
        set(contract) == EXPECTED_TOP_LEVEL_KEYS,
        "Policy-retuning top-level inventory drifted: "
        f"missing={sorted(EXPECTED_TOP_LEVEL_KEYS - set(contract))}, "
        f"unexpected={sorted(set(contract) - EXPECTED_TOP_LEVEL_KEYS)}.",
    )
    exact_scalars = {
        "schema_version": 1,
        "contract_id": "policy_retuning_v3",
        "dataset_key": "inx_primary",
        "task_type": "ordinal_multiclass_performance",
        "target": "PerformanceRating",
        "purpose": "separate_matched_feature_access_sensitivity_from_independently_retuned_policy_performance",
    }
    for key, expected in exact_scalars.items():
        _require(contract.get(key) == expected, f"{key} drifted.")
    _exact_list(contract.get("ordered_labels"), (2, 3, 4), name="ordered_labels")

    sources = _mapping(contract.get("source_contracts"), name="source_contracts")
    _require(set(sources) == set(EXPECTED_SOURCE_RECORDS), "Policy-retuning source inventory drifted.")
    for name, (expected_path, expected_hash, tracked) in EXPECTED_SOURCE_RECORDS.items():
        record = _mapping(sources[name], name=f"source_contracts.{name}")
        expected_keys = {"path", "sha256", *( ["semantic_sha256"] if name == "feature_availability" else [])}
        _require(set(record) == expected_keys, f"Source-record field inventory drifted for {name}.")
        _require(record["path"] == expected_path, f"Source path drifted for {name}.")
        _require(record["sha256"] == expected_hash, f"Source hash drifted for {name}.")
        _require(SHA256_PATTERN.fullmatch(str(record["sha256"])) is not None, f"Source hash is invalid for {name}.")
        if tracked:
            _require(_sha256_file(Path(expected_path)) == expected_hash, f"Tracked source bytes drifted for {name}.")

    feature_receipt = validate_feature_availability_contract()
    feature_source = _mapping(sources["feature_availability"], name="feature source")
    _require(feature_source["semantic_sha256"] == feature_receipt["contract_semantic_sha256"], "Feature semantic identity drifted.")
    information = _mapping(contract.get("information_contract"), name="information_contract")
    _exact_list(information.get("policy_ids"), POLICY_IDS, name="policy_ids")
    _exact_list(information.get("policy_names"), POLICY_NAMES, name="policy_names")
    _exact_list(information.get("retained_feature_counts"), POLICY_FEATURE_COUNTS, name="retained_feature_counts")
    _require(feature_receipt["policy_feature_counts"] == dict(zip(POLICY_IDS, POLICY_FEATURE_COUNTS)), "Feature-policy counts drifted.")
    _require(information.get("primary_policy_id") == "P3", "Primary policy must remain P3.")
    _require(information.get("timing_status") == "timestamp_unverified_cross_sectional", "Timing limitation drifted.")
    for field in ("strict_prospective_is_sensitivity_not_validation",):
        _require(information.get(field) is True, f"information_contract.{field} must be true.")
    for field in ("leakage_free_claim_allowed", "prospective_validity_claim_allowed"):
        _require(information.get(field) is False, f"information_contract.{field} must be false.")

    feature_sets, feature_records = _feature_sets()
    for policy_id, policy_name, count in zip(POLICY_IDS, POLICY_NAMES, POLICY_FEATURE_COUNTS):
        _require(str(feature_records[policy_id]["name"]) == policy_name, f"Policy name drifted for {policy_id}.")
        _require(len(feature_sets[policy_id]) == count, f"Retained feature count drifted for {policy_id}.")

    crosswalk = contract.get("fixed_evidence_crosswalk")
    _require(isinstance(crosswalk, list) and len(crosswalk) == 6, "Fixed-evidence crosswalk must contain six rows.")
    _require([row.get("policy_id") for row in crosswalk] == list(POLICY_IDS), "Fixed-evidence crosswalk order drifted.")
    expected_aliases = (
        "full_feature_upper_bound",
        "no_salary_hike_no_attrition_sensitive_retaining_audit",
        "no_salary_hike_no_attrition",
        "no_salary_hike_no_attrition_no_department",
        None,
        None,
    )
    expected_statuses = (
        "exact_feature_set_reuse",
        "exact_feature_set_reuse",
        "exact_feature_set_reuse",
        "exact_feature_set_reuse_and_primary_benchmark_replay",
        "new_fixed_schedule_fit_required",
        "new_fixed_schedule_fit_required",
    )
    for row, alias, status in zip(crosswalk, expected_aliases, expected_statuses):
        _require(set(row) == {"policy_id", "source_policy", "source_status"}, "Crosswalk field inventory drifted.")
        _require(row["source_policy"] == alias and row["source_status"] == status, "Fixed-evidence crosswalk drifted.")
    historical = _mapping(contract.get("historical_policy_boundary"), name="historical_policy_boundary")
    _exact_list(
        historical.get("preserve_unmodified_v2_policies"),
        ("no_salary_hike", "no_salary_hike_no_attrition_no_department_no_job_role"),
        name="preserve_unmodified_v2_policies",
    )
    _require(historical.get("phase1d_exclusion_reason") == "not_exact_members_of_the_v3_P0_to_P5_information_contract", "Historical policy boundary drifted.")
    _require(historical.get("suppression_or_deletion_allowed") is False, "Historical evidence cannot be suppressed.")

    design = _mapping(contract.get("design"), name="design")
    expected_design = {
        "outer_strategy": "canonical_v2_persisted_StratifiedKFold",
        "outer_splits": 10,
        "inner_strategy": "canonical_v2_persisted_StratifiedKFold",
        "inner_splits": 5,
        "same_exact_outer_and_inner_assignments_for_both_estimands": True,
        "every_sample_exactly_once_per_policy_per_estimand": True,
        "outer_test_usage": "evaluation_only_never_tuning_selection_preprocessing_or_policy_choice",
        "seed_or_policy_selected_from_results": False,
    }
    _require(dict(design) == expected_design, "Policy-retuning fold design drifted.")
    model = _mapping(contract.get("model"), name="model")
    expected_model = {
        "model_name": "xgboost",
        "candidate_count": 8,
        "fixed_parameters_and_candidates": "exact_configs_model_grid_yaml_xgboost_registry",
        "preprocessing_factory": "src.models.canonical_models.build_common_preprocessor",
        "preprocessing_fit_scope": "current_inner_or_outer_training_partition_only",
        "selection_primary_metric": "macro_f1",
        "selection_tie_break_metric": "quadratic_weighted_kappa",
        "primary_tie_tolerance": 0.001,
        "estimator_threads": 1,
        "early_stopping": False,
    }
    _require(dict(model) == expected_model, "Policy-retuning model contract drifted.")
    model_config = validate_benchmark_config(load_config("configs/model_grid.yaml"))["models"]["xgboost"]
    _require(len(model_config["candidates"]) == 8, "XGBoost candidate registry must contain eight candidates.")

    fixed = _mapping(contract.get("fixed_hyperparameter_estimand"), name="fixed estimand")
    expected_fixed = {
        "estimand_id": "fixed_primary_schedule_feature_access_sensitivity",
        "question": "How do held-out predictions change when feature access changes while the primary-policy fold-specific candidate schedule is held fixed?",
        "fold_specific_schedule_source": "canonical_v2_selected_hyperparameters_xgboost",
        "P0_to_P3_evidence_source": "canonical_v2_fixed_policy_oof_exact_feature_set_reuse",
        "P4_to_P5_evidence_source": "new_outer_train_refit_with_primary_selected_candidate",
        "primary_P3_exact_benchmark_replay_required": True,
        "independent_policy_tuning": False,
        "feature_access_effect_is_causal_claim": False,
    }
    _require(dict(fixed) == expected_fixed, "Fixed-hyperparameter estimand drifted.")
    retuned = _mapping(contract.get("independently_retuned_estimand"), name="retuned estimand")
    expected_retuned = {
        "estimand_id": "independently_retuned_policy_performance",
        "question": "What held-out performance is achieved within the fixed candidate registry when each policy is tuned independently on its own outer-training data?",
        "every_policy_enters_inner_selection": True,
        "selection_scope": "independently_within_each_policy_and_outer_training_partition",
        "outer_test_used_for_selection": False,
        "primary_P3_exact_benchmark_replay_required": True,
        "best_achievable_claim_scope": "within_prespecified_model_family_candidate_grid_and_folds_only",
        "policy_or_model_retuning_effect_is_causal_claim": False,
    }
    _require(dict(retuned) == expected_retuned, "Retuned estimand drifted.")

    computational = _mapping(contract.get("computational_scope"), name="computational_scope")
    expected_computational = {
        "retuned_inner_candidate_fit_calls": 2400,
        "retuned_outer_refit_calls": 60,
        "new_fixed_schedule_outer_refit_calls": 20,
        "planned_new_estimator_fit_calls": 2480,
        "reused_fixed_oof_policy_count": 4,
        "new_fixed_oof_policy_count": 2,
        "retuned_policy_count": 6,
    }
    _require(dict(computational) == expected_computational, "Computational scope drifted.")
    evaluation = _mapping(contract.get("evaluation"), name="evaluation")
    _exact_list(evaluation.get("aggregate_metrics"), EXPECTED_AGGREGATE_METRICS, name="aggregate_metrics")
    _exact_list(evaluation.get("headline_comparison_metrics"), HEADLINE_METRICS, name="headline_comparison_metrics")
    _require(evaluation.get("raw_difference_definition") == "retuned_minus_fixed", "Raw difference definition drifted.")
    directions = _mapping(evaluation.get("direction_aligned_improvement"), name="metric directions")
    higher = tuple(directions.get("higher_is_better_metrics", ()))
    lower = tuple(directions.get("lower_is_better_metrics", ()))
    _require(not set(higher).intersection(lower), "Metric direction sets overlap.")
    _require(set(higher).union(lower) == set(EXPECTED_AGGREGATE_METRICS), "Metric direction sets are incomplete.")
    _require(evaluation.get("inferential_claim_from_point_difference_allowed") is False, "Point differences cannot support inference.")
    _require(evaluation.get("universal_best_policy_claim_allowed") is False, "Universal policy claims are prohibited.")

    publication = _mapping(contract.get("publication"), name="publication")
    _require(publication.get("local_output_root") == "reports/major_revision_v3_runs", "Local output root drifted.")
    for field in ("publish_employee_level_oof_rows", "publish_fold_assignments", "publish_candidate_search_rows", "publish_fitted_models"):
        _require(publication.get(field) is False, f"publication.{field} must be false.")
    _exact_list(
        publication.get("compact_outputs"),
        ("aggregate_metrics", "metric_comparison", "headline_policy_comparison", "selected_candidate_frequency", "provenance_receipt"),
        name="compact_outputs",
    )
    prohibited = publication.get("prohibited_claims")
    _require(
        isinstance(prohibited, list)
        and {
            "fixed_schedule_as_independently_optimized_policy_performance",
            "retuned_difference_as_pure_feature_access_effect",
            "retuned_difference_as_causal_model_retuning_effect",
            "best_achievable_outside_prespecified_candidate_grid",
            "universally_best_policy",
            "leakage_free",
            "prospectively_validated",
            "deployment_ready_hr_decision_system",
        }.issubset(prohibited),
        "Policy-retuning prohibited-claim set is incomplete.",
    )

    receipt = _load_json(Path(sources["canonical_v2_receipt"]["path"]))
    _require(receipt.get("status") == "passed_and_promoted", "Canonical-v2 receipt status drifted.")
    _require(receipt.get("generation", {}).get("run_id") == CANONICAL_V2_RUN_ID, "Canonical-v2 run identity drifted.")
    local_receipt = _validate_local_canonical_sources(sources, feature_sets, crosswalk)
    digest = _sha256_file(path)
    _require(SHA256_PATTERN.fullmatch(digest) is not None, "Policy-retuning contract digest is invalid.")
    return {
        "status": "passed",
        "contract_sha256": digest,
        "feature_contract_sha256": feature_receipt["contract_sha256"],
        "feature_contract_semantic_sha256": feature_receipt["contract_semantic_sha256"],
        "policy_count": 6,
        "policy_feature_counts": dict(zip(POLICY_IDS, POLICY_FEATURE_COUNTS)),
        "candidate_count": 8,
        "outer_splits": 10,
        "inner_splits": 5,
        "planned_new_estimator_fit_calls": 2480,
        "reused_fixed_policy_count": 4,
        "new_fixed_policy_count": 2,
        "retuned_policy_count": 6,
        "local_canonical_sources": local_receipt,
        "model_fit_count": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "contract_path",
        nargs="?",
        type=Path,
        default=DEFAULT_POLICY_RETUNING_CONTRACT_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(json.dumps(validate_policy_retuning_contract_v3(args.contract_path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_V2_ROOT",
    "DEFAULT_POLICY_RETUNING_CONTRACT_PATH",
    "HEADLINE_METRICS",
    "POLICY_FEATURE_COUNTS",
    "POLICY_IDS",
    "POLICY_NAMES",
    "PolicyRetuningContractError",
    "validate_policy_retuning_contract_v3",
]
