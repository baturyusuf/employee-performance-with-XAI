"""Fail-closed validation for the v3 subgroup and proxy-use diagnostic contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.canonical_loader import sha256_file
from src.utils.config_loader import PROJECT_ROOT


DEFAULT_SUBGROUP_PROXY_USE_CONTRACT = Path("configs/subgroup_proxy_use_v3.json")
LABELS = (2, 3, 4)
PRIMARY_SYSTEM = "no_salary_hike_no_attrition_no_department"
PROXY_REDUCED_SYSTEM = (
    "no_salary_hike_no_attrition_no_department_no_job_role"
)
SYSTEMS = (
    "no_salary_hike_no_attrition",
    PRIMARY_SYSTEM,
    PROXY_REDUCED_SYSTEM,
)
ATTRIBUTES = (
    "Age",
    "Gender",
    "MaritalStatus",
    "BusinessTravelFrequency",
    "EmpDepartment",
    "EducationBackground",
)
METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "quadratic_weighted_kappa",
    "ordinal_mae",
    "recall_class_2",
    "recall_class_3",
    "recall_class_4",
    "multiclass_brier",
    "log_loss",
)
EXPECTED_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "contract_id",
        "dataset_key",
        "target",
        "ordered_labels",
        "purpose",
        "source_contracts",
        "canonical_identity",
        "subgroup_audit",
        "simultaneous_bootstrap",
        "proxy_prediction_comparison",
        "job_role_permutation",
        "department_reconstructability",
        "computational_scope",
        "publication",
    }
)
EXPECTED_SOURCE_NAMES = frozenset(
    {
        "canonical_v2_receipt",
        "canonical_loader_config",
        "acquisition_manifest",
        "feature_availability",
        "subgroup_stage_contract",
        "subgroup_stage_metadata",
        "fairness_oof_predictions",
        "performance_subgroup_bootstrap_metadata",
        "proxy_metric_intervals",
        "proxy_policy_paired_differences",
        "proxy_bootstrap_metadata",
        "shared_fold_contract",
        "shared_outer_assignments",
        "shared_inner_assignments",
        "benchmark_stage_metadata",
        "baseline_xgboost_gate",
        "paired_model_differences",
        "candidate_search_results",
        "selected_hyperparameters",
        "benchmark_oof_predictions",
        "fitted_model_index",
        "transformed_feature_lineage",
    }
)
IDENTITY_FIELDS = (
    "run_id",
    "config_hash",
    "scientific_input_hash",
    "fold_contract_hash",
    "xgboost_model_set_sha256",
    "dataset_sha256",
)


class SubgroupProxyUseContractV3Error(RuntimeError):
    """Raised when the Phase 2C design or its exact source evidence drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SubgroupProxyUseContractV3Error(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SubgroupProxyUseContractV3Error(
            f"Could not read {path.as_posix()}: {exc}"
        ) from exc
    _require(isinstance(payload, dict), f"{path.name} must contain a JSON object.")
    return payload


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _exact(mapping: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, value in expected.items():
        _require(mapping.get(key) == value, f"{label} drifted for {key}.")


def _validate_source_hashes(contract: Mapping[str, Any]) -> None:
    sources = contract["source_contracts"]
    _require(set(sources) == EXPECTED_SOURCE_NAMES, "Phase 2C source inventory drifted.")
    for name, record in sources.items():
        _require(
            isinstance(record, Mapping) and set(record) == {"path", "sha256"},
            f"Source schema drifted for {name}.",
        )
        _require(_digest(record["sha256"]), f"Source digest is invalid for {name}.")
        path = PROJECT_ROOT / str(record["path"])
        _require(path.is_file(), f"Required Phase 2C source is absent: {name}.")
        _require(
            sha256_file(path) == record["sha256"],
            f"Phase 2C source hash drifted: {name}.",
        )


def _validate_identity_sources(contract: Mapping[str, Any]) -> None:
    identity = contract["canonical_identity"]
    sources = contract["source_contracts"]
    subgroup_metadata = _load_json(PROJECT_ROOT / sources["subgroup_stage_metadata"]["path"])
    benchmark_metadata = _load_json(PROJECT_ROOT / sources["benchmark_stage_metadata"]["path"])
    for field in IDENTITY_FIELDS:
        expected = identity[field]
        _require(
            subgroup_metadata.get(field) == expected,
            f"Canonical subgroup metadata identity drifted for {field}.",
        )
        if field in benchmark_metadata:
            _require(
                benchmark_metadata.get(field) == expected,
                f"Canonical benchmark metadata identity drifted for {field}.",
            )
    for field in ("run_id", "config_hash", "scientific_input_hash", "fold_contract_hash"):
        _require(field in benchmark_metadata, f"Canonical benchmark metadata lacks {field}.")
    _require(subgroup_metadata.get("status") == "complete", "Canonical subgroup stage is incomplete.")
    _require(benchmark_metadata.get("status") == "complete", "Canonical benchmark stage is incomplete.")
    _require(
        subgroup_metadata.get("performance_model_refit_in_stage") is False,
        "Canonical subgroup stage unexpectedly refit a performance model.",
    )


def _validate_oof_source(contract: Mapping[str, Any]) -> dict[str, Any]:
    source = PROJECT_ROOT / contract["source_contracts"]["fairness_oof_predictions"]["path"]
    frame = pd.read_csv(source)
    required = {
        *IDENTITY_FIELDS,
        "system_id",
        "policy",
        "sample_index",
        "outer_fold",
        "y_true",
        "y_pred",
        "prob_class_2",
        "prob_class_3",
        "prob_class_4",
    }
    _require(required.issubset(frame.columns), "Canonical fairness OOF schema drifted.")
    _require(len(frame) == 3600, "Canonical fairness OOF row count drifted.")
    _require(set(frame["system_id"].astype(str)) == set(SYSTEMS), "Canonical fairness system set drifted.")
    identity = contract["canonical_identity"]
    for field in IDENTITY_FIELDS:
        _require(
            set(frame[field].astype(str)) == {str(identity[field])},
            f"Canonical fairness OOF identity drifted for {field}.",
        )
    probability_columns = [f"prob_class_{label}" for label in LABELS]
    for system, rows in frame.groupby("system_id", sort=True):
        rows = rows.sort_values("sample_index")
        _require(len(rows) == 1200, f"OOF row count drifted for {system}.")
        _require(rows["sample_index"].nunique() == 1200, f"OOF samples repeat for {system}.")
        _require(set(rows["sample_index"].astype(int)) == set(range(1200)), f"OOF coverage drifted for {system}.")
        _require(set(rows["outer_fold"].astype(int)) == set(range(1, 11)), f"OOF folds drifted for {system}.")
        _require(set(rows["y_true"].astype(int)).issubset(LABELS), f"OOF target labels drifted for {system}.")
        probabilities = rows[probability_columns].to_numpy(float)
        _require(np.isfinite(probabilities).all(), f"OOF probabilities are non-finite for {system}.")
        _require(np.all((0.0 <= probabilities) & (probabilities <= 1.0)), f"OOF probabilities escaped [0,1] for {system}.")
        _require(np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-9), f"OOF simplex drifted for {system}.")
        expected_predictions = np.asarray(LABELS)[np.argmax(probabilities, axis=1)]
        _require(np.array_equal(expected_predictions, rows["y_pred"].to_numpy(int)), f"OOF argmax drifted for {system}.")
    aligned = [
        rows.sort_values("sample_index")
        for _, rows in frame.groupby("system_id", sort=True)
    ]
    for column in ("sample_index", "outer_fold", "y_true"):
        reference = aligned[0][column].to_numpy(int)
        _require(
            all(np.array_equal(reference, rows[column].to_numpy(int)) for rows in aligned[1:]),
            f"Cross-system OOF alignment drifted for {column}.",
        )
    return {"oof_rows": len(frame), "systems": len(SYSTEMS), "samples_per_system": 1200}


def _validate_proxy_sources(contract: Mapping[str, Any]) -> dict[str, int]:
    sources = contract["source_contracts"]
    intervals = pd.read_csv(PROJECT_ROOT / sources["proxy_metric_intervals"]["path"])
    paired = pd.read_csv(PROJECT_ROOT / sources["proxy_policy_paired_differences"]["path"])
    _require(len(intervals) == 6, "Department reconstructability metric grid drifted.")
    _require(
        set(intervals["system_id"].astype(str)) == {PRIMARY_SYSTEM, PROXY_REDUCED_SYSTEM},
        "Department reconstructability system set drifted.",
    )
    _require(
        set(intervals["metric"].astype(str)) == {"accuracy", "balanced_accuracy", "macro_f1"},
        "Department reconstructability metric set drifted.",
    )
    _require(len(paired) == 3, "Department reconstructability paired grid drifted.")
    _require(
        set(paired["metric"].astype(str)) == {"accuracy", "balanced_accuracy", "macro_f1"},
        "Department reconstructability paired metrics drifted.",
    )
    _require(set(paired["system_a"].astype(str)) == {PRIMARY_SYSTEM}, "Proxy paired system_a drifted.")
    _require(set(paired["system_b"].astype(str)) == {PROXY_REDUCED_SYSTEM}, "Proxy paired system_b drifted.")
    _require(np.isfinite(intervals[["point_estimate", "ci_low", "ci_high"]].to_numpy(float)).all(), "Proxy intervals are non-finite.")
    _require(np.isfinite(paired[["difference", "ci_low", "ci_high"]].to_numpy(float)).all(), "Proxy paired differences are non-finite.")
    return {"reconstructability_metric_rows": 6, "reconstructability_difference_rows": 3}


def validate_subgroup_proxy_use_contract_v3(
    contract_path: Path | str = DEFAULT_SUBGROUP_PROXY_USE_CONTRACT,
) -> dict[str, Any]:
    """Validate exact Phase 2C sources and declared analysis scope without fitting."""

    path = Path(contract_path)
    full_path = path if path.is_absolute() else PROJECT_ROOT / path
    contract = _load_json(full_path)
    _require(set(contract) == EXPECTED_TOP_LEVEL, "Phase 2C top-level inventory drifted.")
    _exact(
        contract,
        {
            "schema_version": 1,
            "contract_id": "subgroup_proxy_use_v3",
            "dataset_key": "inx_primary",
            "target": "PerformanceRating",
            "ordered_labels": [2, 3, 4],
            "purpose": "extend_exact_canonical_oof_subgroup_evidence_and_separate_department_reconstructability_from_performance_model_proxy_dependence",
        },
        "Phase 2C identity",
    )
    _validate_source_hashes(contract)
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
            "outer_folds": 10,
        },
        "Canonical identity",
    )
    _validate_identity_sources(contract)

    subgroup = contract["subgroup_audit"]
    _require(
        [row["system_id"] for row in subgroup.get("systems", [])] == list(SYSTEMS),
        "Subgroup system order drifted.",
    )
    _require(subgroup.get("primary_system_id") == PRIMARY_SYSTEM, "Primary subgroup system drifted.")
    _require(tuple(subgroup.get("attributes", ())) == ATTRIBUTES, "Subgroup attribute set drifted.")
    _require(subgroup.get("support_thresholds") == [20, 30, 50], "Support threshold grid drifted.")
    _require(subgroup.get("minimum_true_class_denominator") == 10, "Class support rule drifted.")
    _require(tuple(subgroup.get("metrics", ())) == METRICS, "Subgroup metric grid drifted.")
    _exact(
        subgroup.get("age_bins", {}),
        {
            "edges": [17, 29, 39, 49, 59, 200],
            "labels": ["18-29", "30-39", "40-49", "50-59", "60+"],
            "right_closed": True,
            "include_lowest": True,
        },
        "Age-bin contract",
    )
    for field in ("below_threshold_rows_retained",):
        _require(subgroup.get(field) is True, f"Subgroup {field} must be true.")
    for field in ("fairness_certification_allowed", "formal_discrimination_claim_allowed"):
        _require(subgroup.get(field) is False, f"Subgroup {field} must be false.")

    bootstrap = contract["simultaneous_bootstrap"]
    _exact(
        bootstrap,
        {
            "scope": "P3_primary_all_declared_attribute_threshold_metric_gap_cells",
            "n_resamples": 5000,
            "seed": 31041,
            "random_generator": "numpy_default_rng_pcg64",
            "stratify_by": ["outer_fold", "y_true"],
            "eligibility_scope": "fixed_from_complete_oof_before_resampling",
            "pointwise_interval": "percentile_95_two_sided_linear_quantile",
            "simultaneous_interval": "studentized_max_absolute_bootstrap_deviation_95_familywise",
            "model_training_variability_included": False,
        },
        "Simultaneous bootstrap",
    )
    comparison = contract["proxy_prediction_comparison"]
    _require(comparison.get("primary_system_id") == PRIMARY_SYSTEM, "Proxy comparison primary drifted.")
    _require(comparison.get("proxy_reduced_system_id") == PROXY_REDUCED_SYSTEM, "Proxy-reduced comparator drifted.")
    _require(comparison.get("causal_claim_allowed") is False, "Proxy comparison cannot permit causal claims.")
    permutation = contract["job_role_permutation"]
    _require(permutation.get("feature") == "EmpJobRole", "Permutation feature drifted.")
    _require(permutation.get("new_model_fit_calls") == 0, "Permutation cannot refit models.")
    _require(permutation.get("outcome_used_for_permutation") is False, "Permutation cannot use outcomes.")
    _require(permutation.get("seeds") == list(range(31042, 31062)), "Permutation seed grid drifted.")
    _require(
        permutation.get("schemes")
        == [
            "marginal_within_outer_test_fold",
            "department_conditional_within_outer_test_fold_and_department",
        ],
        "Permutation schemes drifted.",
    )
    scope = contract["computational_scope"]
    _require(all(scope.get(field) == 0 for field in scope), "Phase 2C must be offline and fit-free.")
    publication = contract["publication"]
    _require(publication.get("publish_employee_level_prediction_comparison") is False, "Employee comparison rows cannot be published.")
    _require(publication.get("publish_employee_level_permutation_rows") is False, "Employee permutation rows cannot be published.")
    _require(
        set(publication.get("prohibited_claims", ()))
        == {
            "fairness_certified",
            "no_discrimination",
            "department_used_because_reconstructable",
            "job_role_has_causal_effect",
            "P3_minus_JobRole_is_v3_P5",
            "deployment_ready_hr_decision",
        },
        "Phase 2C prohibited-claim set drifted.",
    )
    oof_receipt = _validate_oof_source(contract)
    proxy_receipt = _validate_proxy_sources(contract)
    return {
        "status": "validated",
        "contract_sha256": sha256_file(full_path),
        **oof_receipt,
        **proxy_receipt,
        "new_model_fit_calls": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


__all__ = [
    "ATTRIBUTES",
    "DEFAULT_SUBGROUP_PROXY_USE_CONTRACT",
    "LABELS",
    "METRICS",
    "PRIMARY_SYSTEM",
    "PROXY_REDUCED_SYSTEM",
    "SYSTEMS",
    "SubgroupProxyUseContractV3Error",
    "validate_subgroup_proxy_use_contract_v3",
]
