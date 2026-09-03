"""Fail-closed validator for the additive v3 ordinal benchmark protocol."""

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
from src.models.canonical_models import CANONICAL_MODEL_NAMES
from src.models.ordinal_models_v3 import (
    V3_BASELINE_STRATEGIES,
    V3_NAIVE_BASELINE_NAMES,
    V3_ORDINAL_ESTIMATOR_PATHS,
    V3_ORDINAL_MODEL_NAMES,
    build_v3_ordinal_estimator,
)


DEFAULT_BENCHMARK_CONTRACT_PATH = Path("configs/ordinal_benchmark_v3.json")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
EXPECTED_AGGREGATE_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "weighted_precision",
    "macro_recall",
    "weighted_recall",
    "quadratic_weighted_kappa",
    "ordinal_mae",
    "adjacent_accuracy",
    "two_level_reversal_rate",
    "nll_log_loss",
    "multiclass_brier",
    "ece_confidence",
    "ranked_probability_score",
)


class OrdinalBenchmarkContractError(ValueError):
    """Raised when the v3 benchmark protocol is incomplete or unsafe."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OrdinalBenchmarkContractError(
            f"Could not read benchmark contract {path.as_posix()}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise OrdinalBenchmarkContractError("Benchmark contract must contain a JSON object.")
    return payload


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OrdinalBenchmarkContractError(f"{name} must be an object.")
    return value


def _exact_string_list(value: Any, *, expected: Sequence[str], name: str) -> None:
    if not isinstance(value, list) or tuple(value) != tuple(expected):
        raise OrdinalBenchmarkContractError(
            f"{name} must equal the exact ordered contract {list(expected)}."
        )


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise OrdinalBenchmarkContractError(
            f"Could not hash bound input {path.as_posix()}: {exc}"
        ) from exc


def validate_ordinal_benchmark_contract_v3(
    contract_path: Path | str = DEFAULT_BENCHMARK_CONTRACT_PATH,
) -> dict[str, Any]:
    """Validate source identities, estimators, metrics, folds, and publication limits."""

    path = Path(contract_path)
    contract = _load_json(path)
    exact_scalars = {
        "schema_version": 1,
        "contract_id": "ordinal_benchmark_v3",
        "dataset_key": "inx_primary",
        "task_type": "ordinal_multiclass_performance",
        "target": "PerformanceRating",
    }
    for key, expected in exact_scalars.items():
        if contract.get(key) != expected:
            raise OrdinalBenchmarkContractError(f"{key} must equal {expected!r}.")
    if contract.get("ordered_labels") != [2, 3, 4]:
        raise OrdinalBenchmarkContractError("ordered_labels must equal [2, 3, 4].")

    data_source = _mapping(contract.get("data_source"), name="data_source")
    bound_data_files = (
        (
            "canonical_loader_config_path",
            "canonical_loader_config_sha256",
            "configs/manuscript_final.yaml",
        ),
        (
            "acquisition_manifest_path",
            "acquisition_manifest_sha256",
            "configs/data_acquisition.yaml",
        ),
    )
    data_source_hashes: dict[str, str] = {}
    for path_key, hash_key, expected_path in bound_data_files:
        if data_source.get(path_key) != expected_path:
            raise OrdinalBenchmarkContractError(f"data_source.{path_key} drifted.")
        actual_hash = _file_sha256(Path(expected_path))
        if data_source.get(hash_key) != actual_hash:
            raise OrdinalBenchmarkContractError(f"data_source.{hash_key} drifted.")
        data_source_hashes[hash_key] = actual_hash
    if data_source.get("automatic_download_allowed") is not False:
        raise OrdinalBenchmarkContractError("V3 benchmark automatic download must remain disabled.")

    information = _mapping(contract.get("information_contract"), name="information_contract")
    if information.get("path") != "configs/feature_availability_v3.json":
        raise OrdinalBenchmarkContractError("Unexpected information-contract path.")
    feature_path = Path(str(information["path"]))
    feature_receipt = validate_feature_availability_contract(feature_path)
    if information.get("sha256") != feature_receipt["contract_sha256"]:
        raise OrdinalBenchmarkContractError("Information-contract byte hash drifted.")
    if information.get("semantic_sha256") != feature_receipt["contract_semantic_sha256"]:
        raise OrdinalBenchmarkContractError("Information-contract semantic hash drifted.")
    if information.get("primary_policy_id") != "P3" or information.get(
        "retained_feature_count"
    ) != feature_receipt["policy_feature_counts"]["P3"]:
        raise OrdinalBenchmarkContractError("Primary P3 policy identity/count drifted.")

    preprocessing = _mapping(contract.get("preprocessing"), name="preprocessing")
    if preprocessing.get("factory") != "src.models.canonical_models.build_common_preprocessor":
        raise OrdinalBenchmarkContractError("The common preprocessing factory is not exact.")
    if preprocessing.get("fit_scope") != "current_inner_or_outer_training_partition_only":
        raise OrdinalBenchmarkContractError("Preprocessing fit scope must remain training-only.")
    if preprocessing.get("outer_test_used_for_fit") is not False:
        raise OrdinalBenchmarkContractError("Outer-test preprocessing must be false.")

    folds = _mapping(contract.get("shared_nested_cv"), name="shared_nested_cv")
    required_fold_values = {
        "outer_strategy": "StratifiedKFold",
        "outer_splits": 10,
        "outer_shuffle": True,
        "inner_strategy": "StratifiedKFold",
        "inner_splits": 5,
        "inner_shuffle": True,
        "same_exact_outer_folds_across_all_models_and_baselines": True,
        "outer_test_usage": "evaluation_only_never_tuning_selection_preprocessing_or_calibration",
        "oof_coverage": "every_sample_exactly_once_per_model",
    }
    for key, expected in required_fold_values.items():
        if folds.get(key) != expected:
            raise OrdinalBenchmarkContractError(
                f"shared_nested_cv.{key} must equal {expected!r}."
            )
    for seed_key in ("outer_seed", "inner_seed", "model_seed"):
        seed = folds.get(seed_key)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise OrdinalBenchmarkContractError(
                f"shared_nested_cv.{seed_key} must be a nonnegative integer."
            )

    comparison = _mapping(
        contract.get("canonical_v2_comparison_source"),
        name="canonical_v2_comparison_source",
    )
    if comparison.get("run_id") != "canonical_v2_20260714T221501Z_483f96f":
        raise OrdinalBenchmarkContractError("Canonical v2 comparison run identity drifted.")
    if comparison.get("generation_commit") != (
        "483f96fdbaab16cb0f32d03d9dbe676a759af44a"
    ):
        raise OrdinalBenchmarkContractError("Canonical v2 generation commit drifted.")
    expected_v2_paths = {
        "fold_contract": "core/shared_folds/fold_contract.json",
        "outer_assignments": "core/shared_folds/fold_assignments.csv",
        "inner_assignments": "core/shared_folds/inner_fold_assignments.csv",
        "nominal_oof_predictions": "core/model_benchmarks/oof_predictions.csv",
    }
    for key, expected_path in expected_v2_paths.items():
        record = _mapping(comparison.get(key), name=f"canonical_v2_comparison_source.{key}")
        if record.get("path") != expected_path:
            raise OrdinalBenchmarkContractError(f"Canonical v2 {key} path drifted.")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise OrdinalBenchmarkContractError(
                f"Canonical v2 {key} hash must be a lowercase SHA-256 digest."
            )
    if comparison.get("reuse_boundary") != (
        "reuse_validated_v2_nominal_oof_evidence_without_refit_or_relabelling_and_fit_new_models_on_the_exact_same_persisted_folds"
    ):
        raise OrdinalBenchmarkContractError("Canonical v2 reuse boundary drifted.")

    selection = _mapping(contract.get("selection"), name="selection")
    if selection.get("primary_metric") != "macro_f1":
        raise OrdinalBenchmarkContractError("Selection primary metric must be macro_f1.")
    if selection.get("tie_break_metric") != "quadratic_weighted_kappa":
        raise OrdinalBenchmarkContractError("Selection tie-break must be QWK.")
    if selection.get("outer_test_used") is not False:
        raise OrdinalBenchmarkContractError("Outer test cannot be used for selection.")
    if selection.get("baselines_enter_hyperparameter_selection") is not False:
        raise OrdinalBenchmarkContractError("Naive baselines cannot enter tuning.")
    if "do_not_name_a_universal_best_model" not in str(selection.get("reporting_rule")):
        raise OrdinalBenchmarkContractError(
            "Reporting must prohibit a universal-best-model claim."
        )

    nominal = _mapping(contract.get("nominal_model_registry"), name="nominal_model_registry")
    if nominal.get("path") != "configs/model_grid.yaml":
        raise OrdinalBenchmarkContractError("Unexpected nominal model-registry path.")
    nominal_path = Path(str(nominal["path"]))
    if nominal.get("sha256") != _file_sha256(nominal_path):
        raise OrdinalBenchmarkContractError("Nominal model-registry hash drifted.")
    _exact_string_list(
        nominal.get("models"), expected=CANONICAL_MODEL_NAMES, name="nominal models"
    )

    ordinal_models = _mapping(contract.get("ordinal_models"), name="ordinal_models")
    if tuple(ordinal_models) != V3_ORDINAL_MODEL_NAMES:
        raise OrdinalBenchmarkContractError(
            "Ordinal model identity/order differs from the exact two-model registry."
        )
    candidate_counts: dict[str, int] = {}
    model_seed = int(folds["model_seed"])
    for model_name in V3_ORDINAL_MODEL_NAMES:
        specification = _mapping(ordinal_models[model_name], name=f"ordinal_models.{model_name}")
        if specification.get("estimator") != V3_ORDINAL_ESTIMATOR_PATHS[model_name]:
            raise OrdinalBenchmarkContractError(
                f"Estimator path drifted for {model_name}."
            )
        fixed = _mapping(specification.get("fixed_params"), name=f"{model_name}.fixed_params")
        candidates = specification.get("candidates")
        if not isinstance(candidates, list) or not candidates or not all(
            isinstance(candidate, Mapping) for candidate in candidates
        ):
            raise OrdinalBenchmarkContractError(
                f"{model_name}.candidates must be a non-empty object list."
            )
        semantic_candidates = {
            json.dumps(candidate, sort_keys=True, separators=(",", ":"))
            for candidate in candidates
        }
        if len(semantic_candidates) != len(candidates):
            raise OrdinalBenchmarkContractError(f"{model_name} has duplicate candidates.")
        for candidate in candidates:
            overlap = sorted(set(fixed).intersection(candidate))
            if overlap:
                raise OrdinalBenchmarkContractError(
                    f"{model_name} candidate overwrites fixed parameters: {overlap}."
                )
            try:
                estimator = build_v3_ordinal_estimator(
                    model_name,
                    {**dict(fixed), **dict(candidate)},
                    random_state=model_seed,
                )
                estimator._validate_hyperparameters()  # type: ignore[attr-defined]
            except (TypeError, ValueError) as exc:
                raise OrdinalBenchmarkContractError(
                    f"Invalid parameter contract for {model_name}: {exc}"
                ) from exc
        candidate_counts[model_name] = len(candidates)

    baselines = contract.get("naive_baselines")
    if not isinstance(baselines, list) or not all(isinstance(row, Mapping) for row in baselines):
        raise OrdinalBenchmarkContractError("naive_baselines must be an object list.")
    baseline_names = [str(row.get("name")) for row in baselines]
    if tuple(baseline_names) != V3_NAIVE_BASELINE_NAMES:
        raise OrdinalBenchmarkContractError("Naive baseline identity/order drifted.")
    for row in baselines:
        name = str(row["name"])
        if row.get("strategy") != V3_BASELINE_STRATEGIES[name]:
            raise OrdinalBenchmarkContractError(f"Baseline strategy drifted for {name}.")
        if row.get("fit_scope") != "current_outer_training_labels_only":
            raise OrdinalBenchmarkContractError(f"Baseline {name} is not training-only.")

    evaluation = _mapping(contract.get("evaluation"), name="evaluation")
    _exact_string_list(
        evaluation.get("aggregate_metrics"),
        expected=EXPECTED_AGGREGATE_METRICS,
        name="evaluation.aggregate_metrics",
    )
    if evaluation.get("legacy_severe_error_display_name_allowed") is not False:
        raise OrdinalBenchmarkContractError("Legacy severe-error display name must be disabled.")
    if evaluation.get("ranked_probability_score_normalized_domain") != [0.0, 1.0]:
        raise OrdinalBenchmarkContractError("Normalized RPS domain must equal [0, 1].")
    _exact_string_list(
        evaluation.get("per_class_fields"),
        expected=("class_label", "precision", "recall", "f1", "support"),
        name="evaluation.per_class_fields",
    )
    if evaluation.get("confusion_matrix_contract") != (
        "complete_ordered_true_by_predicted_grid_for_each_dataset_and_model"
    ):
        raise OrdinalBenchmarkContractError("Confusion-matrix contract drifted.")

    xai_reference = _mapping(contract.get("xai_reference"), name="xai_reference")
    if xai_reference.get("model_name") != "xgboost":
        raise OrdinalBenchmarkContractError("The prespecified XAI reference must be XGBoost.")
    if xai_reference.get("independent_of_predictive_ranking") is not True:
        raise OrdinalBenchmarkContractError(
            "XAI reference choice must remain independent of predictive ranking."
        )
    if "not_predictive_superiority" not in str(xai_reference.get("selection_basis")):
        raise OrdinalBenchmarkContractError(
            "XAI reference basis must explicitly reject predictive-superiority selection."
        )

    publication = _mapping(contract.get("publication"), name="publication")
    if publication.get("publish_employee_level_oof_rows") is not False:
        raise OrdinalBenchmarkContractError("Employee-level OOF rows cannot be published.")
    if publication.get("publish_fitted_models") is not False:
        raise OrdinalBenchmarkContractError("Fitted models cannot be published.")
    prohibited = publication.get("prohibited_claims")
    required_prohibited = {
        "universally_best_model",
        "leakage_free",
        "prospectively_validated",
        "causal_employee_performance_driver",
        "deployment_ready_hr_decision_system",
    }
    if not isinstance(prohibited, list) or set(prohibited) != required_prohibited:
        raise OrdinalBenchmarkContractError("Publication prohibited-claim set drifted.")

    return {
        "status": "passed",
        "contract_id": "ordinal_benchmark_v3",
        "contract_sha256": _file_sha256(path),
        "information_contract_sha256": feature_receipt["contract_sha256"],
        "information_contract_semantic_sha256": feature_receipt[
            "contract_semantic_sha256"
        ],
        "nominal_model_registry_sha256": _file_sha256(nominal_path),
        "canonical_loader_config_sha256": data_source_hashes[
            "canonical_loader_config_sha256"
        ],
        "acquisition_manifest_sha256": data_source_hashes[
            "acquisition_manifest_sha256"
        ],
        "ordered_labels": [2, 3, 4],
        "nominal_model_count": len(CANONICAL_MODEL_NAMES),
        "ordinal_model_count": len(V3_ORDINAL_MODEL_NAMES),
        "naive_baseline_count": len(V3_NAIVE_BASELINE_NAMES),
        "ordinal_candidate_counts": candidate_counts,
        "aggregate_metric_count": len(EXPECTED_AGGREGATE_METRICS),
        "model_fit_count": 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_BENCHMARK_CONTRACT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(json.dumps(validate_ordinal_benchmark_contract_v3(args.contract), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BENCHMARK_CONTRACT_PATH",
    "EXPECTED_AGGREGATE_METRICS",
    "OrdinalBenchmarkContractError",
    "validate_ordinal_benchmark_contract_v3",
]
