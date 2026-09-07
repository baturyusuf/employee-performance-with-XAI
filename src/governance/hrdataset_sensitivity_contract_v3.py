"""Fail-closed contract validation for HRDataset_v14 Phase 3A sensitivities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.external_adapters import build_feature_columns, load_external_dataset
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_HRDATASET_SENSITIVITY_CONTRACT = Path("configs/hrdataset_sensitivity_v3.json")
PRIMARY_FORMULATION = "primary_three_class"
ALTERNATIVE_FORMULATION = "raw_order_four_class"
FORMULATIONS = (PRIMARY_FORMULATION, ALTERNATIVE_FORMULATION)
BASELINES = ("majority_baseline", "stratified_baseline", "ordinal_median_baseline")
PRIMARY_LABELS = (2, 3, 4)
ALTERNATIVE_LABELS = (1, 2, 3, 4)
FEATURES = (
    "EmpJobRole",
    "EngagementSurvey",
    "EmpJobSatisfaction",
    "SpecialProjectsCount",
    "DaysLateLast30",
    "Absences",
    "ExperienceYearsAtThisCompany",
)
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "quadratic_weighted_kappa",
    "ordinal_mae",
    "two_level_reversal_rate",
    "nll_log_loss",
    "multiclass_brier",
    "ece_confidence",
    "ranked_probability_score",
)
PRIORITY_METRICS = (
    "macro_f1",
    "quadratic_weighted_kappa",
    "ordinal_mae",
    "nll_log_loss",
    "multiclass_brier",
    "ece_confidence",
    "ranked_probability_score",
)
EXPECTED_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "contract_id",
        "dataset_key",
        "purpose",
        "source_contracts",
        "feature_policy",
        "target_formulations",
        "design",
        "model_protocol",
        "calibration",
        "naive_baselines",
        "evaluation",
        "protocol_comparison_components",
        "computational_scope",
        "publication",
    }
)
EXPECTED_SOURCES = frozenset(
    {
        "raw_dataset",
        "schema_mapping",
        "acquisition_manifest",
        "model_grid",
        "canonical_config",
        "canonical_v2_receipt",
        "v2_external_metadata",
        "v2_primary_features",
        "v2_raw_intervals",
        "v2_calibrated_intervals",
        "v2_raw_oof",
        "v2_calibrated_oof",
    }
)


class HRDatasetSensitivityContractV3Error(RuntimeError):
    """Raised when an exact Phase 3A source or scientific decision drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HRDatasetSensitivityContractV3Error(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HRDatasetSensitivityContractV3Error(
            f"Could not read {path.as_posix()}: {exc}"
        ) from exc
    _require(isinstance(payload, dict), f"{path.name} must contain a JSON object.")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object.")
    return value


def _exact(mapping: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, value in expected.items():
        _require(mapping.get(key) == value, f"{label}.{key} drifted.")


def _validate_sources(contract: Mapping[str, Any]) -> dict[str, str]:
    sources = _mapping(contract.get("source_contracts"), "source_contracts")
    _require(set(sources) == EXPECTED_SOURCES, "Phase 3A source inventory drifted.")
    hashes: dict[str, str] = {}
    for name, value in sources.items():
        record = _mapping(value, f"source_contracts.{name}")
        _require(set(record) == {"path", "sha256"}, f"Source record drifted for {name}.")
        path = PROJECT_ROOT / str(record["path"])
        _require(path.is_file(), f"Bound Phase 3A source is absent: {name}.")
        observed = _sha256(path)
        _require(observed == record["sha256"], f"Bound source hash drifted: {name}.")
        hashes[name] = observed
    return hashes


def _validate_targets(contract: Mapping[str, Any]) -> dict[str, Any]:
    formulations = contract.get("target_formulations")
    _require(isinstance(formulations, list) and len(formulations) == 2, "Exactly two target formulations are required.")
    indexed = {str(item.get("formulation_id")): item for item in formulations if isinstance(item, Mapping)}
    _require(tuple(indexed) == FORMULATIONS, "Target formulation order or identity drifted.")
    expected = {
        PRIMARY_FORMULATION: {
            "role": "retained_primary_mapping",
            "ordered_labels": list(PRIMARY_LABELS),
            "mapping": {"PIP": 2, "Needs Improvement": 2, "Fully Meets": 3, "Exceeds": 4, "Exceptional": 4},
        },
        ALTERNATIVE_FORMULATION: {
            "role": "supplementary_mapping_sensitivity",
            "ordered_labels": list(ALTERNATIVE_LABELS),
            "mapping": {"PIP": 1, "Needs Improvement": 2, "Fully Meets": 3, "Exceeds": 4, "Exceptional": 4},
        },
    }
    for name, values in expected.items():
        _exact(indexed[name], values, f"target_formulations.{name}")
        _require(bool(str(indexed[name].get("rationale", "")).strip()), f"{name} rationale is absent.")
    raw = pd.read_csv(PROJECT_ROOT / contract["source_contracts"]["raw_dataset"]["path"])
    _require(len(raw) == 311 and "PerformanceScore" in raw, "HRDataset raw shape/target drifted.")
    observed_categories = set(raw["PerformanceScore"].astype(str))
    _require(observed_categories == {"PIP", "Needs Improvement", "Fully Meets", "Exceeds"}, "Observed raw target categories drifted.")
    supports: dict[str, dict[int, int]] = {}
    for name in FORMULATIONS:
        mapping = indexed[name]["mapping"]
        mapped = raw["PerformanceScore"].map(mapping)
        _require(not mapped.isna().any(), f"{name} leaves an observed target unmapped.")
        labels = tuple(indexed[name]["ordered_labels"])
        _require(set(mapped.astype(int)) == set(labels), f"{name} does not retain every declared label.")
        supports[name] = {int(label): int((mapped == label).sum()) for label in labels}
    _require(supports[PRIMARY_FORMULATION] == {2: 31, 3: 243, 4: 37}, "Primary mapping support drifted.")
    _require(supports[ALTERNATIVE_FORMULATION] == {1: 13, 2: 18, 3: 243, 4: 37}, "Alternative mapping support drifted.")
    return {"raw_category_support": raw["PerformanceScore"].value_counts().to_dict(), "mapped_support": supports}


def _validate_features_and_v2(contract: Mapping[str, Any]) -> dict[str, Any]:
    sources = contract["source_contracts"]
    dataset = load_external_dataset(
        "hrdataset_v14",
        raw_path=PROJECT_ROOT / sources["raw_dataset"]["path"],
        schema_mapping_path=PROJECT_ROOT / sources["schema_mapping"]["path"],
    )
    feature_policy = _mapping(contract.get("feature_policy"), "feature_policy")
    _exact(
        feature_policy,
        {
            "policy_id": "conservative_primary",
            "exact_features": list(FEATURES),
            "feature_count": 7,
            "same_exact_features_across_formulations": True,
            "training_partition_only_preprocessing": True,
        },
        "feature_policy",
    )
    observed = tuple(build_feature_columns(dataset, "conservative_primary"))
    _require(observed == FEATURES, "Adapted conservative-primary feature order drifted.")
    v2_features = pd.read_csv(PROJECT_ROOT / sources["v2_primary_features"]["path"])
    scoped = v2_features.loc[v2_features["policy"].astype(str) == "conservative_primary"].sort_values("raw_feature_order")
    _require(tuple(scoped["feature"].astype(str)) == FEATURES, "Canonical-v2 primary feature evidence drifted.")
    metadata = _load_json(PROJECT_ROOT / sources["v2_external_metadata"]["path"])
    _require(metadata.get("status") == "complete", "Canonical-v2 HRDataset stage is incomplete.")
    _require(metadata.get("dataset_sha256") == sources["raw_dataset"]["sha256"], "V2/raw dataset identity drifted.")
    _require(metadata.get("primary_policy") == "conservative_primary", "V2 primary policy drifted.")
    _require(metadata.get("protocol", {}).get("outer_splits") == 10, "V2 outer-fold reference drifted.")
    _require(metadata.get("protocol", {}).get("inner_splits") == 5, "V2 inner-fold reference drifted.")
    for name in ("v2_raw_oof", "v2_calibrated_oof"):
        frame = pd.read_csv(PROJECT_ROOT / sources[name]["path"])
        frame = frame.loc[frame["policy"].astype(str) == "conservative_primary"].copy()
        _require(len(frame) == 311 and frame["sample_index"].nunique() == 311, f"{name} exactly-once coverage drifted.")
        probabilities = frame[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float)
        _require(np.isfinite(probabilities).all(), f"{name} contains non-finite probabilities.")
        _require(np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-9), f"{name} simplex drifted.")
        _require(set(frame["outer_fold"].astype(int)) == set(range(1, 11)), f"{name} fold set drifted.")
    return {"feature_count": len(observed), "sample_count": len(dataset.canonical), "v2_reference_valid": True}


def validate_hrdataset_sensitivity_contract_v3(
    contract_path: Path | str = DEFAULT_HRDATASET_SENSITIVITY_CONTRACT,
) -> dict[str, Any]:
    """Validate the full Phase 3A scientific contract without fitting a model."""

    path = Path(contract_path)
    full = path if path.is_absolute() else PROJECT_ROOT / path
    contract = _load_json(full)
    _require(set(contract) == EXPECTED_TOP_LEVEL, "Phase 3A top-level inventory drifted.")
    _exact(
        contract,
        {
            "schema_version": 1,
            "contract_id": "hrdataset_sensitivity_v3",
            "dataset_key": "hrdataset_v14",
            "purpose": "independent_protocol_replication_target_mapping_and_repeated_nested_cv_sensitivity",
        },
        "contract",
    )
    source_hashes = _validate_sources(contract)
    target_receipt = _validate_targets(contract)
    data_receipt = _validate_features_and_v2(contract)
    design = _mapping(contract.get("design"), "design")
    _exact(
        design,
        {
            "repetitions": 5,
            "outer_strategy": "StratifiedKFold",
            "outer_splits": 5,
            "outer_shuffle": True,
            "inner_strategy": "StratifiedKFold",
            "inner_splits": 5,
            "inner_shuffle": True,
            "same_exact_outer_and_inner_folds_across_systems_within_formulation_and_repetition": True,
            "different_target_formulations_have_independently_stratified_folds": True,
            "every_sample_exactly_once_per_system_per_repetition": True,
            "outer_test_usage": "evaluation_only_never_tuning_preprocessing_calibration_mapping_or_seed_selection",
        },
        "design",
    )
    seeds = design.get("seed_schedule")
    _require(isinstance(seeds, list) and len(seeds) == 5, "Seed schedule must contain five repetitions.")
    _require([row.get("repetition") for row in seeds] == [1, 2, 3, 4, 5], "Repetition identities drifted.")
    for field in ("outer_seed", "inner_seed", "model_seed", "calibration_seed", "baseline_seed"):
        values = [row.get(field) for row in seeds]
        _require(all(isinstance(value, int) and not isinstance(value, bool) for value in values), f"{field} values must be integers.")
        _require(len(set(values)) == 5, f"{field} values must be unique by repetition.")
    model = _mapping(contract.get("model_protocol"), "model_protocol")
    _exact(
        model,
        {
            "model": "xgboost",
            "candidate_count": 8,
            "selection_metric": "macro_f1",
            "tie_break_metric": "quadratic_weighted_kappa",
            "primary_tie_tolerance": 0.001,
            "fixed_and_candidate_parameters_source": "source_contracts.model_grid",
            "outer_test_used_for_selection": False,
        },
        "model_protocol",
    )
    grid = load_config(PROJECT_ROOT / contract["source_contracts"]["model_grid"]["path"])
    xgb = grid.get("model_benchmark", {}).get("models", {}).get("xgboost", {})
    _require(len(xgb.get("candidates", ())) == 8, "Bound XGBoost candidate grid drifted.")
    calibration = _mapping(contract.get("calibration"), "calibration")
    _exact(
        calibration,
        {
            "methods": ["raw", "sigmoid"],
            "sigmoid_algorithm": "one_vs_rest_platt_logit_then_row_renormalize",
            "sigmoid_predeclared": True,
            "training_source": "selected_candidate_five_fold_inner_oof_probabilities_within_each_outer_training_partition",
            "outer_test_used_for_fit_or_method_selection": False,
            "n_bins": 10,
        },
        "calibration",
    )
    _require(tuple(contract.get("naive_baselines", ())) == BASELINES, "Naive baseline registry drifted.")
    evaluation = _mapping(contract.get("evaluation"), "evaluation")
    _require(tuple(evaluation.get("aggregate_metrics", ())) == METRICS, "Aggregate metric registry drifted.")
    _require(tuple(evaluation.get("priority_cv_metrics", ())) == PRIORITY_METRICS, "Priority metric registry drifted.")
    _require(evaluation.get("cross_formulation_metric_difference_allowed") is False, "Cross-formulation differences must remain prohibited.")
    _require(tuple(contract.get("protocol_comparison_components", ())) == ("folds", "tuning", "calibration", "SHAP", "subgroup", "proxy", "target", "features"), "Protocol comparison components drifted.")
    compute = _mapping(contract.get("computational_scope"), "computational_scope")
    _exact(compute, {"planned_xgboost_fit_calls": 2300, "planned_baseline_fit_calls": 150, "network_calls": 0, "paid_api_calls": 0}, "computational_scope")
    publication = _mapping(contract.get("publication"), "publication")
    _require(publication.get("publish_employee_level_oof") is False, "Employee-level OOF publication is prohibited.")
    _require(publication.get("publish_fitted_models_or_calibrators") is False, "Fitted-object publication is prohibited.")
    prohibited = set(publication.get("prohibited_claims", ()))
    _require({"external_validation", "locked_model_transport", "target_equivalence"}.issubset(prohibited), "Core prohibited claims drifted.")
    return {
        "status": "passed",
        "contract_path": full.relative_to(PROJECT_ROOT).as_posix(),
        "contract_sha256": _sha256(full),
        "source_hashes": source_hashes,
        **data_receipt,
        **target_receipt,
        "formulation_count": 2,
        "repetitions": 5,
        "outer_folds_per_repetition": 5,
        "inner_folds": 5,
        "candidate_count": 8,
        "planned_xgboost_fit_calls": 2300,
        "planned_baseline_fit_calls": 150,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


__all__ = [
    "ALTERNATIVE_FORMULATION",
    "ALTERNATIVE_LABELS",
    "BASELINES",
    "DEFAULT_HRDATASET_SENSITIVITY_CONTRACT",
    "FEATURES",
    "FORMULATIONS",
    "HRDatasetSensitivityContractV3Error",
    "METRICS",
    "PRIMARY_FORMULATION",
    "PRIMARY_LABELS",
    "PRIORITY_METRICS",
    "validate_hrdataset_sensitivity_contract_v3",
]
