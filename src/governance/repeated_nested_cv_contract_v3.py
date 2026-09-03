"""Fail-closed validator for the v3 repeated nested-CV sensitivity contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.governance.feature_availability_contract import (
    validate_feature_availability_contract,
)
from src.governance.ordinal_benchmark_contract_v3 import (
    EXPECTED_AGGREGATE_METRICS,
    validate_ordinal_benchmark_contract_v3,
)
from src.models.canonical_models import CANONICAL_MODEL_NAMES
from src.models.ordinal_models_v3 import (
    V3_NAIVE_BASELINE_NAMES,
    V3_ORDINAL_MODEL_NAMES,
)
from src.utils.config_loader import load_config


DEFAULT_REPEATED_CV_CONTRACT_PATH = Path("configs/repeated_nested_cv_v3.json")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
PRIORITY_METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "quadratic_weighted_kappa",
    "ordinal_mae",
)
SUMMARY_STATISTICS = ("mean", "sample_sd", "median", "minimum", "maximum")
ORDERING_OUTPUTS = (
    "rank_by_repetition",
    "winner_frequency",
    "mean_rank",
    "sample_sd_rank",
    "pairwise_rank_spearman",
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
        "information_policy",
        "design",
        "computational_scope",
        "model_registry",
        "preprocessing",
        "selection",
        "evaluation",
        "publication",
    }
)


class RepeatedNestedCVContractError(ValueError):
    """Raised when repeated-CV scope, seeds, sources, or claims drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RepeatedNestedCVContractError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RepeatedNestedCVContractError(
            f"Could not read repeated-CV contract {path.as_posix()}: {exc}"
        ) from exc
    _require(isinstance(payload, dict), "Repeated-CV contract must be a JSON object.")
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
        raise RepeatedNestedCVContractError(
            f"Could not hash bound source {path.as_posix()}: {exc}"
        ) from exc


def _validate_source_record(
    sources: Mapping[str, Any],
    name: str,
    expected_path: str,
) -> Mapping[str, Any]:
    record = _mapping(sources.get(name), name=f"source_contracts.{name}")
    _require(record.get("path") == expected_path, f"Source path drifted for {name}.")
    actual = _sha256_file(Path(expected_path))
    _require(record.get("sha256") == actual, f"Source hash drifted for {name}.")
    return record


def validate_repeated_nested_cv_contract_v3(
    contract_path: Path | str = DEFAULT_REPEATED_CV_CONTRACT_PATH,
) -> dict[str, Any]:
    """Validate the complete bounded 5×5×5 repeated-CV protocol."""

    path = Path(contract_path)
    contract = _load_json(path)
    _require(
        set(contract) == EXPECTED_TOP_LEVEL_KEYS,
        "Repeated-CV top-level inventory drifted: "
        f"missing={sorted(EXPECTED_TOP_LEVEL_KEYS - set(contract))}, "
        f"unexpected={sorted(set(contract) - EXPECTED_TOP_LEVEL_KEYS)}.",
    )
    exact_scalars = {
        "schema_version": 1,
        "contract_id": "repeated_nested_cv_v3",
        "dataset_key": "inx_primary",
        "task_type": "ordinal_multiclass_performance",
        "target": "PerformanceRating",
        "purpose": "training_and_fold_variability_sensitivity_not_sample_level_uncertainty",
    }
    for key, expected in exact_scalars.items():
        _require(contract.get(key) == expected, f"{key} must equal {expected!r}.")
    _exact_list(contract.get("ordered_labels"), (2, 3, 4), name="ordered_labels")

    sources = _mapping(contract.get("source_contracts"), name="source_contracts")
    expected_sources = {
        "canonical_loader_config": "configs/manuscript_final.yaml",
        "acquisition_manifest": "configs/data_acquisition.yaml",
        "feature_availability": "configs/feature_availability_v3.json",
        "nominal_model_grid": "configs/model_grid.yaml",
        "ordinal_benchmark": "configs/ordinal_benchmark_v3.json",
        "phase1b_aggregate_reference": (
            "reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/"
            "aggregate_metrics.csv"
        ),
    }
    _require(set(sources) == set(expected_sources), "Bound source inventory drifted.")
    source_hashes: dict[str, str] = {}
    for name, expected_path in expected_sources.items():
        record = _validate_source_record(sources, name, expected_path)
        source_hashes[name] = str(record["sha256"])

    feature_receipt = validate_feature_availability_contract(
        expected_sources["feature_availability"]
    )
    feature_source = _mapping(sources["feature_availability"], name="feature source")
    _require(
        feature_source.get("semantic_sha256")
        == feature_receipt["contract_semantic_sha256"],
        "Feature-contract semantic identity drifted.",
    )
    ordinal_receipt = validate_ordinal_benchmark_contract_v3(
        expected_sources["ordinal_benchmark"]
    )
    _require(
        ordinal_receipt["contract_sha256"] == source_hashes["ordinal_benchmark"],
        "Ordinal benchmark source receipt drifted.",
    )
    reference = _mapping(
        sources["phase1b_aggregate_reference"], name="phase1b aggregate reference"
    )
    _require(
        reference.get("use") == "descriptive_reference_only_never_selection_or_seed_choice",
        "Phase 1B results cannot select repetitions or seeds.",
    )

    information = _mapping(contract.get("information_policy"), name="information_policy")
    _require(information.get("policy_id") == "P3", "Repeated CV must use policy P3.")
    _require(
        information.get("retained_feature_count")
        == feature_receipt["policy_feature_counts"]["P3"]
        == 20,
        "Repeated-CV P3 feature count drifted.",
    )
    _require(
        information.get("timing_status") == "timestamp_unverified_cross_sectional",
        "Repeated-CV timing limitation must remain explicit.",
    )
    _require(
        information.get("prospective_validity_claim_allowed") is False,
        "Repeated CV cannot establish prospective validity.",
    )

    design = _mapping(contract.get("design"), name="design")
    required_design = {
        "repetitions": 5,
        "outer_strategy": "StratifiedKFold",
        "outer_splits": 5,
        "outer_shuffle": True,
        "inner_strategy": "StratifiedKFold",
        "inner_splits": 5,
        "inner_shuffle": True,
        "same_exact_folds_across_models_within_repetition": True,
        "different_fold_assignment_required_across_repetitions": True,
        "every_sample_exactly_once_per_model_per_repetition": True,
        "outer_test_usage": (
            "evaluation_only_never_tuning_selection_preprocessing_calibration_or_seed_choice"
        ),
    }
    for key, expected in required_design.items():
        _require(design.get(key) == expected, f"design.{key} drifted.")
    seed_schedule = design.get("seed_schedule")
    _require(isinstance(seed_schedule, list), "design.seed_schedule must be a list.")
    _require(len(seed_schedule) == 5, "Exactly five repetitions must be prespecified.")
    expected_seeds = tuple(
        (repetition, repetition * 1000 + 42, repetition * 1000 + 43, repetition * 1000 + 44)
        for repetition in range(1, 6)
    )
    observed_seeds: list[tuple[int, int, int, int]] = []
    for record in seed_schedule:
        mapped = _mapping(record, name="seed schedule record")
        _require(
            set(mapped) == {"repetition", "outer_seed", "inner_seed", "model_seed"},
            "Seed record field inventory drifted.",
        )
        values = tuple(mapped[key] for key in ("repetition", "outer_seed", "inner_seed", "model_seed"))
        _require(
            all(isinstance(value, int) and not isinstance(value, bool) for value in values),
            "Every repetition and seed must be an integer.",
        )
        observed_seeds.append(values)
    _require(tuple(observed_seeds) == expected_seeds, "Deterministic seed schedule drifted.")
    all_seed_values = [value for row in observed_seeds for value in row[1:]]
    _require(len(set(all_seed_values)) == 15, "All outer/inner/model seeds must be unique.")

    model_registry = _mapping(contract.get("model_registry"), name="model_registry")
    _exact_list(
        model_registry.get("nominal_models"),
        CANONICAL_MODEL_NAMES,
        name="nominal_models",
    )
    _exact_list(
        model_registry.get("ordinal_models"),
        V3_ORDINAL_MODEL_NAMES,
        name="ordinal_models",
    )
    _exact_list(
        model_registry.get("naive_baselines"),
        V3_NAIVE_BASELINE_NAMES,
        name="naive_baselines",
    )
    _require(
        model_registry.get("all_models_refitted_in_every_repetition") is True,
        "Every system must be refitted in every repetition.",
    )
    _require(
        model_registry.get("canonical_v2_oof_reuse_allowed") is False,
        "Canonical-v2 OOF rows cannot stand in for repeated fits.",
    )

    nominal_config = load_config(expected_sources["nominal_model_grid"])["model_benchmark"]
    nominal_candidates = sum(
        len(nominal_config["models"][name]["candidates"])
        for name in CANONICAL_MODEL_NAMES
    )
    ordinal_config = _load_json(Path(expected_sources["ordinal_benchmark"]))
    ordinal_candidates = sum(
        len(ordinal_config["ordinal_models"][name]["candidates"])
        for name in V3_ORDINAL_MODEL_NAMES
    )
    tuned_model_count = len(CANONICAL_MODEL_NAMES) + len(V3_ORDINAL_MODEL_NAMES)
    per_outer_fits = (
        (nominal_candidates + ordinal_candidates) * int(design["inner_splits"])
        + tuned_model_count
        + len(V3_NAIVE_BASELINE_NAMES)
    )
    outer_partitions = int(design["repetitions"]) * int(design["outer_splits"])
    planned_fits = per_outer_fits * outer_partitions
    computational = _mapping(
        contract.get("computational_scope"), name="computational_scope"
    )
    expected_computational = {
        "requested_preference": "10_repetitions_if_proportionate_otherwise_5",
        "selected_repetitions": 5,
        "decision_basis": "prespecified_bounded_cost_before_result_inspection",
        "tuned_model_count": tuned_model_count,
        "naive_baseline_count": len(V3_NAIVE_BASELINE_NAMES),
        "outer_partition_count": outer_partitions,
        "inner_validation_partition_count_per_candidate": 125,
        "estimator_fit_calls_per_outer_partition": per_outer_fits,
        "planned_estimator_fit_calls": planned_fits,
        "ten_repetition_estimator_fit_calls": planned_fits * 2,
        "escalation_rule": (
            "do_not_escalate_after_inspecting_results; a separate future 10-repetition run requires a new prespecified contract"
        ),
    }
    for key, expected in expected_computational.items():
        _require(computational.get(key) == expected, f"computational_scope.{key} drifted.")

    preprocessing = _mapping(contract.get("preprocessing"), name="preprocessing")
    _require(
        preprocessing.get("factory")
        == "src.models.canonical_models.build_common_preprocessor",
        "Common preprocessing factory drifted.",
    )
    _require(
        preprocessing.get("fit_scope") == "current_inner_or_outer_training_partition_only",
        "Preprocessing must remain training-partition-only.",
    )
    _require(
        preprocessing.get("outer_test_used_for_fit") is False,
        "Outer-test preprocessing is prohibited.",
    )

    selection = _mapping(contract.get("selection"), name="selection")
    expected_selection = {
        "scope": "independently_within_each_model_repetition_and_outer_training_partition",
        "primary_metric": "macro_f1",
        "tie_break_metric": "quadratic_weighted_kappa",
        "primary_tie_tolerance": 0.001,
        "candidate_definitions": "exact_source_contract_registries",
        "baselines_enter_hyperparameter_selection": False,
        "outer_test_used": False,
        "seed_or_repetition_selected_from_results": False,
    }
    for key, expected in expected_selection.items():
        _require(selection.get(key) == expected, f"selection.{key} drifted.")

    evaluation = _mapping(contract.get("evaluation"), name="evaluation")
    _exact_list(
        evaluation.get("all_per_repetition_metrics"),
        EXPECTED_AGGREGATE_METRICS,
        name="all_per_repetition_metrics",
    )
    _exact_list(
        evaluation.get("priority_variability_metrics"),
        PRIORITY_METRICS,
        name="priority_variability_metrics",
    )
    _exact_list(
        evaluation.get("variability_summary_statistics"),
        SUMMARY_STATISTICS,
        name="variability_summary_statistics",
    )
    _exact_list(
        evaluation.get("model_ordering_outputs"),
        ORDERING_OUTPUTS,
        name="model_ordering_outputs",
    )
    _require(
        evaluation.get("model_ordering_scope") == "six_tuned_models_only",
        "Ordering stability must cover the six tuned models only.",
    )
    directions = _mapping(evaluation.get("metric_directions"), name="metric_directions")
    _require(
        directions
        == {
            "macro_f1": "higher",
            "balanced_accuracy": "higher",
            "quadratic_weighted_kappa": "higher",
            "ordinal_mae": "lower",
        },
        "Priority metric directions drifted.",
    )
    _require(
        evaluation.get("interval_interpretation")
        == "empirical_repetition_range_not_confidence_interval",
        "Five-repetition range cannot be labelled a confidence interval.",
    )
    _require(
        evaluation.get("universal_best_model_claim_allowed") is False,
        "Repeated CV cannot authorize a universal winner claim.",
    )

    publication = _mapping(contract.get("publication"), name="publication")
    for field in (
        "publish_employee_level_oof_rows",
        "publish_fold_assignments",
        "publish_fitted_models",
    ):
        _require(publication.get(field) is False, f"publication.{field} must be false.")
    _require(
        publication.get("local_output_root") == "reports/major_revision_v3_runs",
        "Repeated-CV complete outputs must remain in the ignored local root.",
    )
    prohibited = publication.get("prohibited_claims")
    _require(
        isinstance(prohibited, list)
        and {
            "sample_level_confidence_interval_from_five_repetitions",
            "robust_to_training_variability_without_stable_ordering_evidence",
            "universally_best_model",
            "leakage_free",
            "prospectively_validated",
            "deployment_ready_hr_decision_system",
        }.issubset(prohibited),
        "Repeated-CV prohibited-claim set is incomplete.",
    )

    digest = _sha256_file(path)
    _require(SHA256_PATTERN.fullmatch(digest) is not None, "Contract digest is invalid.")
    return {
        "status": "passed",
        "contract_sha256": digest,
        "source_hashes": source_hashes,
        "repetitions": 5,
        "outer_splits": 5,
        "inner_splits": 5,
        "outer_partition_count": outer_partitions,
        "planned_estimator_fit_calls": planned_fits,
        "nominal_candidate_count": nominal_candidates,
        "ordinal_candidate_count": ordinal_candidates,
        "tuned_model_count": tuned_model_count,
        "total_system_count": tuned_model_count + len(V3_NAIVE_BASELINE_NAMES),
        "priority_metric_count": len(PRIORITY_METRICS),
        "model_fit_count": 0,
        "paid_api_calls": 0,
        "network_calls": 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "contract_path",
        nargs="?",
        type=Path,
        default=DEFAULT_REPEATED_CV_CONTRACT_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(
        json.dumps(
            validate_repeated_nested_cv_contract_v3(args.contract_path),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_REPEATED_CV_CONTRACT_PATH",
    "ORDERING_OUTPUTS",
    "PRIORITY_METRICS",
    "RepeatedNestedCVContractError",
    "SUMMARY_STATISTICS",
    "validate_repeated_nested_cv_contract_v3",
]
