"""Independent closed-world validator for a completed v3 repeated nested-CV run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.canonical_loader import sha256_file
from src.experiments.manuscript_model_benchmark import (
    select_candidate_index,
    validate_benchmark_config,
)
from src.experiments.ordinal_benchmark_v3 import summarize_combined_oof_v3
from src.experiments.repeated_nested_cv_v3 import (
    ALL_MODEL_NAMES,
    TUNED_MODEL_NAMES,
    selected_candidate_frequency_v3,
    summarize_repeated_metrics_v3,
)
from src.experiments.shared_folds import (
    OUTER_COLUMNS,
    SharedFoldArtifacts,
    generate_inner_assignments,
    validate_shared_folds,
)
from src.governance.offline_runtime import validate_policy_receipt
from src.governance.repeated_nested_cv_contract_v3 import (
    DEFAULT_REPEATED_CV_CONTRACT_PATH,
    validate_repeated_nested_cv_contract_v3,
)
from src.models.canonical_models import CANONICAL_MODEL_NAMES
from src.models.ordinal_evaluation_v3 import ordinal_evaluation_bundle_v3
from src.models.ordinal_models_v3 import (
    V3_NAIVE_BASELINE_NAMES,
    V3_ORDINAL_MODEL_NAMES,
)
from src.utils.config_loader import PROJECT_ROOT, load_config


EXPECTED_FILES = frozenset(
    {
        "candidate_search_results.csv",
        "fold_contracts.json",
        "fold_metrics.csv",
        "model_rank_summary.csv",
        "oof_predictions.csv",
        "ordering_stability.csv",
        "rank_by_repetition.csv",
        "repetition_metrics.csv",
        "selected_candidate_frequency.csv",
        "selected_hyperparameters.csv",
        "stage_metadata.json",
        "variability_summary.csv",
    }
)
OUTPUT_HASH_FILES = EXPECTED_FILES - {"stage_metadata.json"}
EXPECTED_IMPLEMENTATIONS = frozenset(
    {
        "src/experiments/repeated_nested_cv_v3.py",
        "src/governance/repeated_nested_cv_contract_v3.py",
        "src/experiments/shared_folds.py",
        "src/models/canonical_models.py",
        "src/models/ordinal_models_v3.py",
        "src/models/ordinal_evaluation_v3.py",
    }
)
IDENTITY_COLUMNS = (
    "run_id",
    "repeated_contract_sha256",
    "scientific_input_sha256",
    "repetition",
    "outer_seed",
    "inner_seed",
    "model_seed",
    "fold_contract_hash",
)
BASELINE_STRATEGIES = {
    "majority_baseline": "majority",
    "stratified_baseline": "stratified",
    "ordinal_median_baseline": "ordinal_median",
}


class V3RepeatedNestedCVRunValidationError(RuntimeError):
    """Raised when persisted repeated nested-CV evidence is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3RepeatedNestedCVRunValidationError(message)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V3RepeatedNestedCVRunValidationError(
            f"Could not read {path.as_posix()}: {exc}"
        ) from exc


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise V3RepeatedNestedCVRunValidationError(
            f"Could not parse {path.name}: {exc}"
        ) from exc


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_blob(commit: str, relative_path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise V3RepeatedNestedCVRunValidationError(
            f"Could not resolve generation blob {commit}:{relative_path}: {exc}"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _assert_frame_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_columns: Sequence[str],
    context: str,
) -> None:
    _require(
        set(observed.columns) == set(expected.columns),
        f"{context} schema drifted: observed={sorted(observed.columns)}, "
        f"expected={sorted(expected.columns)}.",
    )
    columns = list(expected.columns)
    observed_sorted = observed.loc[:, columns].sort_values(list(sort_columns)).reset_index(drop=True)
    expected_sorted = expected.loc[:, columns].sort_values(list(sort_columns)).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            observed_sorted,
            expected_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-14,
        )
    except AssertionError as exc:
        raise V3RepeatedNestedCVRunValidationError(
            f"{context} does not match independent recomputation: {exc}"
        ) from exc


def _json_object(value: Any, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise V3RepeatedNestedCVRunValidationError(
            f"{context} is not valid JSON: {exc}"
        ) from exc
    _require(isinstance(payload, dict), f"{context} must contain a JSON object.")
    return payload


def _json_number_list(value: Any, *, context: str) -> list[float]:
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise V3RepeatedNestedCVRunValidationError(
            f"{context} is not valid JSON: {exc}"
        ) from exc
    _require(
        isinstance(payload, list) and len(payload) == 5,
        f"{context} must contain exactly five inner-fold scores.",
    )
    values = [float(item) for item in payload]
    _require(all(math.isfinite(item) for item in values), f"{context} contains non-finite values.")
    return values


def _validate_identity(
    frame: pd.DataFrame,
    *,
    metadata: Mapping[str, Any],
    fold_identities: Mapping[int, Mapping[str, Any]],
    context: str,
) -> None:
    _require(set(IDENTITY_COLUMNS).issubset(frame.columns), f"{context} identity schema drifted.")
    for repetition, rows in frame.groupby("repetition", sort=True):
        repetition = int(repetition)
        _require(repetition in fold_identities, f"{context} has an unknown repetition.")
        expected = fold_identities[repetition]
        values = {
            "run_id": metadata["run_id"],
            "repeated_contract_sha256": metadata["repeated_contract_sha256"],
            "scientific_input_sha256": metadata["scientific_input_sha256"],
            **expected,
        }
        for column, value in values.items():
            observed = set(rows[column].astype(str))
            _require(
                observed == {str(value)},
                f"{context} identity drifted for repetition={repetition}, column={column}.",
            )


def _model_definitions() -> dict[str, Mapping[str, Any]]:
    nominal_config = load_config("configs/model_grid.yaml")
    nominal = validate_benchmark_config(nominal_config)["models"]
    ordinal_config = _load_json(Path("configs/ordinal_benchmark_v3.json"))
    ordinal = ordinal_config.get("ordinal_models")
    _require(isinstance(ordinal, Mapping), "Ordinal-model source registry is absent.")
    definitions = {name: nominal[name] for name in CANONICAL_MODEL_NAMES}
    definitions.update({name: ordinal[name] for name in V3_ORDINAL_MODEL_NAMES})
    _require(set(definitions) == set(TUNED_MODEL_NAMES), "Tuned model registry drifted.")
    return definitions


def _rebuild_fold_artifacts(
    record: Mapping[str, Any],
    repetition_oof: pd.DataFrame,
) -> SharedFoldArtifacts:
    contract = record.get("contract")
    _require(isinstance(contract, Mapping), "A fold record lacks its contract.")
    reference_model = str(ALL_MODEL_NAMES[0])
    reference = repetition_oof[repetition_oof["model"] == reference_model].copy()
    _require(len(reference) == int(contract.get("n_rows", -1)), "Fold reference row count drifted.")
    reference = reference.set_index("sample_index")[["outer_fold", "y_true"]].astype(int)
    samples = contract.get("samples")
    _require(isinstance(samples, list), "Fold contract samples must be a list.")
    outer_rows: list[dict[str, Any]] = []
    for sample in samples:
        _require(isinstance(sample, Mapping), "Fold contract sample record drifted.")
        sample_index = int(sample["sample_index"])
        _require(sample_index in reference.index, "Fold contract sample is absent from OOF evidence.")
        _require(
            int(sample["y_true"]) == int(reference.loc[sample_index, "y_true"]),
            "Fold-contract target differs from OOF evidence.",
        )
        outer_rows.append(
            {
                "run_id": contract["run_id"],
                "config_hash": contract["config_hash"],
                "scientific_input_hash": contract["scientific_input_hash"],
                "dataset_key": contract["dataset_key"],
                "dataset_sha256": contract["dataset_sha256"],
                "fold_contract_hash": contract["fold_contract_hash"],
                "sample_index": sample_index,
                "sample_key_sha256": sample["sample_key_sha256"],
                "y_true": int(sample["y_true"]),
                "outer_fold": int(reference.loc[sample_index, "outer_fold"]),
            }
        )
    outer = pd.DataFrame(outer_rows, columns=OUTER_COLUMNS).sort_values("sample_index").reset_index(drop=True)
    inner = generate_inner_assignments(
        outer,
        inner_splits=int(contract["inner_splits"]),
        inner_seed=int(contract["inner_seed"]),
        fold_contract_hash=str(contract["fold_contract_hash"]),
    )
    artifacts = SharedFoldArtifacts(outer, inner, contract)
    try:
        validate_shared_folds(artifacts)
    except Exception as exc:
        raise V3RepeatedNestedCVRunValidationError(
            f"Fold contract could not be reconstructed: {exc}"
        ) from exc
    semantic_records = (
        outer[["sample_index", "outer_fold"]]
        .astype(int)
        .sort_values("sample_index")
        .to_dict(orient="records")
    )
    _require(
        record.get("outer_assignment_semantic_sha256")
        == _canonical_json_sha256(semantic_records),
        "Outer-assignment semantic hash drifted.",
    )
    return artifacts


def validate_repeated_nested_cv_run_v3(run_dir: Path | str) -> dict[str, Any]:
    """Rehash and independently recompute a complete five-repetition run."""

    root = Path(run_dir)
    _require(root.is_dir(), f"Run directory does not exist: {root.as_posix()}.")
    files = {path.name for path in root.iterdir() if path.is_file()}
    directories = [path.name for path in root.iterdir() if path.is_dir()]
    _require(not directories, f"Unexpected run subdirectories: {directories}.")
    _require(
        files == EXPECTED_FILES,
        "Run closed-world inventory drifted: "
        f"missing={sorted(EXPECTED_FILES - files)}, "
        f"unexpected={sorted(files - EXPECTED_FILES)}.",
    )
    metadata = _load_json(root / "stage_metadata.json")
    _require(isinstance(metadata, Mapping), "Stage metadata must be an object.")
    expected_metadata = {
        "schema_version": 1,
        "stage": "repeated_nested_cv_v3",
        "status": "complete",
        "evidence_status": "complete_five_repetition_exactly_once_oof",
        "feature_policy": "P3",
        "feature_count": 20,
        "sample_count": 1200,
        "ordered_labels": [2, 3, 4],
        "repetitions": 5,
        "outer_folds_per_repetition": 5,
        "inner_folds": 5,
        "model_count": 9,
        "tuned_model_count": 6,
        "planned_estimator_fit_calls": 5725,
        "candidate_search_row_count": 1100,
        "selected_hyperparameter_row_count": 225,
        "fold_metric_row_count": 225,
        "oof_prediction_row_count": 54000,
        "repetition_metric_row_count": 720,
        "outer_test_used_for_selection": False,
        "seed_or_repetition_selected_from_results": False,
        "employee_level_outputs_publication_authorized": False,
        "network_calls": 0,
        "paid_api_calls": 0,
    }
    for key, expected in expected_metadata.items():
        _require(metadata.get(key) == expected, f"Metadata {key} drifted.")
    try:
        validate_policy_receipt(metadata.get("runtime_policy", {}))
    except Exception as exc:
        raise V3RepeatedNestedCVRunValidationError(
            f"Offline runtime receipt drifted: {exc}"
        ) from exc

    output_hashes = metadata.get("output_hashes")
    _require(isinstance(output_hashes, Mapping), "Metadata output_hashes must be an object.")
    _require(set(output_hashes) == OUTPUT_HASH_FILES, "Output-hash inventory drifted.")
    for filename in sorted(OUTPUT_HASH_FILES):
        _require(
            output_hashes[filename] == sha256_file(root / filename),
            f"Output byte hash drifted for {filename}.",
        )

    scientific_inputs = metadata.get("scientific_inputs")
    _require(isinstance(scientific_inputs, Mapping), "scientific_inputs must be an object.")
    _require(
        metadata.get("scientific_input_sha256") == _canonical_json_sha256(scientific_inputs),
        "Scientific-input composite hash drifted.",
    )
    for key in ("repeated_contract_sha256", "source_tree_hash", "dataset_sha256", "git_identity"):
        _require(metadata.get(key) == scientific_inputs.get(key), f"Scientific identity {key} is inconsistent.")
    git_identity = metadata.get("git_identity")
    _require(isinstance(git_identity, Mapping), "Git identity must be an object.")
    commit = str(git_identity.get("commit", ""))
    _require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "Generation commit is invalid.")
    contract_blob = _git_blob(commit, DEFAULT_REPEATED_CV_CONTRACT_PATH.as_posix())
    _require(
        _sha256_bytes(contract_blob) == metadata["repeated_contract_sha256"],
        "Generation-commit repeated-CV contract hash drifted.",
    )
    generation_contract = json.loads(contract_blob.decode("utf-8"))
    contract_receipt = validate_repeated_nested_cv_contract_v3()
    _require(
        contract_receipt["contract_sha256"] == metadata["repeated_contract_sha256"],
        "Current bound repeated-CV contract differs from the generation contract.",
    )
    _require(
        contract_receipt["source_hashes"] == scientific_inputs.get("bound_source_hashes"),
        "Bound-source hash receipt drifted.",
    )
    for name, record in generation_contract["source_contracts"].items():
        path = str(record["path"])
        _require(
            _sha256_bytes(_git_blob(commit, path)) == scientific_inputs["bound_source_hashes"][name],
            f"Generation bound-source blob drifted for {name}.",
        )
    implementation_hashes = scientific_inputs.get("implementation_hashes")
    _require(isinstance(implementation_hashes, Mapping), "Implementation hashes are absent.")
    _require(set(implementation_hashes) == EXPECTED_IMPLEMENTATIONS, "Implementation inventory drifted.")
    for relative_path, expected_hash in implementation_hashes.items():
        _require(
            _sha256_bytes(_git_blob(commit, str(relative_path))) == expected_hash,
            f"Generation implementation hash drifted for {relative_path}.",
        )

    candidates = _read_csv(root / "candidate_search_results.csv")
    fold_metrics = _read_csv(root / "fold_metrics.csv")
    oof = _read_csv(root / "oof_predictions.csv")
    repetition_metrics = _read_csv(root / "repetition_metrics.csv")
    selected = _read_csv(root / "selected_hyperparameters.csv")
    variability = _read_csv(root / "variability_summary.csv")
    ranks = _read_csv(root / "rank_by_repetition.csv")
    rank_summary = _read_csv(root / "model_rank_summary.csv")
    stability = _read_csv(root / "ordering_stability.csv")
    frequency = _read_csv(root / "selected_candidate_frequency.csv")
    fold_records = _load_json(root / "fold_contracts.json")
    _require(isinstance(fold_records, list) and len(fold_records) == 5, "Fold-contract record count drifted.")

    expected_seed_rows = {
        int(row["repetition"]): {
            "repetition": int(row["repetition"]),
            "outer_seed": int(row["outer_seed"]),
            "inner_seed": int(row["inner_seed"]),
            "model_seed": int(row["model_seed"]),
        }
        for row in generation_contract["design"]["seed_schedule"]
    }
    fold_identities: dict[int, dict[str, Any]] = {}
    semantic_hashes: set[str] = set()
    artifacts_by_repetition: dict[int, SharedFoldArtifacts] = {}
    for record in fold_records:
        _require(isinstance(record, Mapping), "Fold-contract record must be an object.")
        repetition = int(record.get("repetition", -1))
        _require(repetition in expected_seed_rows, "Fold-contract repetition drifted.")
        for seed_name in ("outer_seed", "inner_seed", "model_seed"):
            _require(int(record.get(seed_name, -1)) == expected_seed_rows[repetition][seed_name], f"{seed_name} drifted.")
        scoped_oof = oof[oof["repetition"].astype(int) == repetition]
        artifacts = _rebuild_fold_artifacts(record, scoped_oof)
        artifacts_by_repetition[repetition] = artifacts
        semantic_hash = str(record["outer_assignment_semantic_sha256"])
        semantic_hashes.add(semantic_hash)
        fold_identities[repetition] = {
            **expected_seed_rows[repetition],
            "fold_contract_hash": str(artifacts.contract["fold_contract_hash"]),
        }
    _require(set(artifacts_by_repetition) == set(range(1, 6)), "Fold repetition coverage drifted.")
    _require(len(semantic_hashes) == 5, "Repeated outer-fold assignments are not distinct.")
    for name, frame in {
        "candidate search": candidates,
        "fold metrics": fold_metrics,
        "OOF": oof,
        "repetition metrics": repetition_metrics,
        "selected hyperparameters": selected,
    }.items():
        _validate_identity(frame, metadata=metadata, fold_identities=fold_identities, context=name)

    _require(len(oof) == 54_000, "Repeated OOF row count drifted.")
    _require(set(oof["evidence_source"].unique()) == {"v3_repeated_nested_outer_fold_fit"}, "OOF evidence-source label drifted.")
    probability_columns = ["prob_class_2", "prob_class_3", "prob_class_4"]
    labels = np.asarray([2, 3, 4])
    reference_targets: pd.Series | None = None
    for repetition, repetition_rows in oof.groupby("repetition", sort=True):
        _require(set(repetition_rows["model"].unique()) == set(ALL_MODEL_NAMES), "OOF model set drifted.")
        repetition_reference: pd.DataFrame | None = None
        for model_name, rows in repetition_rows.groupby("model", sort=False):
            _require(len(rows) == 1200 and rows["sample_index"].nunique() == 1200, f"OOF coverage drifted for repetition={repetition}, model={model_name}.")
            current = rows.set_index("sample_index")[["outer_fold", "y_true"]].astype(int).sort_index()
            if repetition_reference is None:
                repetition_reference = current
            else:
                _require(current.equals(repetition_reference), f"Models do not share folds in repetition {repetition}.")
            probability = rows[probability_columns].to_numpy(float)
            _require(np.all(np.isfinite(probability)), "OOF probabilities contain non-finite values.")
            _require(np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-9), "OOF probabilities do not sum to one.")
            _require(np.array_equal(labels[np.argmax(probability, axis=1)], rows["y_pred"].to_numpy(int)), "OOF predictions disagree with probability argmax.")
        _require(repetition_reference is not None, "A repetition has no OOF rows.")
        target = repetition_reference["y_true"]
        if reference_targets is None:
            reference_targets = target
        else:
            _require(target.equals(reference_targets), "OOF target identity drifted across repetitions.")

    definitions = _model_definitions()
    expected_candidate_rows = 0
    _require(not candidates["outer_test_used_for_selection"].astype(bool).any(), "Outer test entered candidate selection.")
    for (repetition, outer_fold, model_name), rows in candidates.groupby(["repetition", "outer_fold", "model"], sort=True):
        _require(model_name in definitions, "A non-tuned model entered candidate search.")
        model_candidates = [dict(item) for item in definitions[str(model_name)]["candidates"]]
        expected_candidate_rows += len(model_candidates)
        _require(sorted(rows["candidate_index"].astype(int)) == list(range(len(model_candidates))), "Candidate index grid drifted.")
        _require(rows["candidate_status"].eq("complete").all(), "Candidate status drifted.")
        _require(rows["n_inner_folds"].astype(int).eq(5).all(), "Inner-fold score count drifted.")
        ordered = rows.sort_values("candidate_index")
        macro_means: list[float] = []
        qwk_means: list[float] = []
        for expected_index, (_, row) in enumerate(ordered.iterrows()):
            _require(_json_object(row["parameters_json"], context="candidate parameters") == model_candidates[expected_index], "Candidate parameters drifted from source registry.")
            macro_scores = _json_number_list(row["inner_macro_f1_scores_json"], context="inner macro-F1 scores")
            qwk_scores = _json_number_list(row["inner_qwk_scores_json"], context="inner QWK scores")
            macro_mean = float(np.mean(macro_scores))
            qwk_mean = float(np.mean(qwk_scores))
            _require(math.isclose(float(row["inner_macro_f1_mean"]), macro_mean, rel_tol=0.0, abs_tol=1e-14), "Candidate macro-F1 mean drifted.")
            _require(math.isclose(float(row["inner_qwk_mean"]), qwk_mean, rel_tol=0.0, abs_tol=1e-14), "Candidate QWK mean drifted.")
            macro_means.append(macro_mean)
            qwk_means.append(qwk_mean)
        expected_selected = select_candidate_index(
            macro_means,
            qwk_means,
            practical_tie_tolerance=0.001,
            better_direction="higher",
        )
        selected_indices = ordered.loc[ordered["selected_by_protocol"].astype(bool), "candidate_index"].astype(int).tolist()
        _require(selected_indices == [expected_selected], "Candidate selection protocol drifted.")
    _require(expected_candidate_rows == len(candidates) == 1100, "Candidate-search row count drifted.")

    _require(len(selected) == 225, "Selected-hyperparameter row count drifted.")
    _require(not selected.duplicated(["repetition", "outer_fold", "model"]).any(), "Selection grid contains duplicates.")
    _require(not selected["outer_test_used_for_selection"].astype(bool).any(), "Outer test entered selection records.")
    expected_grid = {(r, f, m) for r in range(1, 6) for f in range(1, 6) for m in ALL_MODEL_NAMES}
    observed_grid = set(selected[["repetition", "outer_fold", "model"]].itertuples(index=False, name=None))
    _require(observed_grid == expected_grid, "Selection model/repetition/fold grid drifted.")
    for _, row in selected.iterrows():
        model_name = str(row["model"])
        key_mask = (
            (candidates["repetition"] == row["repetition"])
            & (candidates["outer_fold"] == row["outer_fold"])
            & (candidates["model"] == model_name)
        )
        if model_name in TUNED_MODEL_NAMES:
            _require(bool(row["selection_performed"]), "A tuned model lacks selection.")
            selected_candidates = candidates[key_mask & candidates["selected_by_protocol"].astype(bool)]
            _require(len(selected_candidates) == 1, "Selected candidate linkage drifted.")
            candidate = selected_candidates.iloc[0]
            _require(int(float(row["selected_candidate_index"])) == int(candidate["candidate_index"]), "Selected candidate index drifted.")
            _require(_json_object(row["selected_candidate_parameters_json"], context="selected parameters") == _json_object(candidate["parameters_json"], context="candidate parameters"), "Selected candidate parameters drifted.")
            _require(_json_object(row["fixed_parameters_json"], context="fixed parameters") == dict(definitions[model_name]["fixed_params"]), "Fixed model parameters drifted.")
            _require(math.isclose(float(row["selected_inner_macro_f1_mean"]), float(candidate["inner_macro_f1_mean"]), rel_tol=0.0, abs_tol=1e-14), "Selected macro-F1 score drifted.")
            _require(math.isclose(float(row["selected_inner_qwk_mean"]), float(candidate["inner_qwk_mean"]), rel_tol=0.0, abs_tol=1e-14), "Selected QWK score drifted.")
        else:
            _require(model_name in V3_NAIVE_BASELINE_NAMES and not bool(row["selection_performed"]), "Baseline selection flag drifted.")
            _require(pd.isna(row["selected_candidate_index"]), "A baseline has a candidate index.")
            _require(_json_object(row["fixed_parameters_json"], context="baseline parameters") == {"strategy": BASELINE_STRATEGIES[model_name]}, "Baseline strategy drifted.")

    selected_lookup = selected.set_index(["repetition", "outer_fold", "model"])["selected_candidate_index"]
    for (repetition, outer_fold, model_name), rows in oof.groupby(["repetition", "outer_fold", "model"], sort=False):
        expected_index = selected_lookup.loc[(repetition, outer_fold, model_name)]
        observed_indices = rows["selected_candidate_index"]
        if model_name in TUNED_MODEL_NAMES:
            _require(observed_indices.notna().all() and observed_indices.astype(int).eq(int(expected_index)).all(), "OOF selected-candidate lineage drifted.")
        else:
            _require(observed_indices.isna().all(), "Baseline OOF rows contain a candidate index.")

    expected_fold_rows: list[dict[str, Any]] = []
    for (repetition, outer_fold, model_name), rows in oof.groupby(["repetition", "outer_fold", "model"], sort=True):
        rows = rows.sort_values("sample_index")
        probability = rows[probability_columns].to_numpy(float)
        bundle = ordinal_evaluation_bundle_v3(
            rows["y_true"].astype(int),
            rows["y_pred"].astype(int),
            probability,
            labels=(2, 3, 4),
            dataset_key="inx_primary",
            model_name=str(model_name),
        )
        identity = {column: rows.iloc[0][column] for column in IDENTITY_COLUMNS}
        expected_fold_rows.append(
            {
                **identity,
                "outer_fold": int(outer_fold),
                "model": str(model_name),
                "n_train": 1200 - len(rows),
                "n_test": len(rows),
                **bundle["aggregate_metrics"],
            }
        )
    expected_fold_metrics = pd.DataFrame(expected_fold_rows)
    _assert_frame_equal(fold_metrics, expected_fold_metrics, sort_columns=("repetition", "outer_fold", "model"), context="Fold metrics")

    expected_repetition_frames: list[pd.DataFrame] = []
    for repetition, rows in oof.groupby("repetition", sort=True):
        aggregate, _, _ = summarize_combined_oof_v3(rows, labels=(2, 3, 4))
        identity_row = rows.iloc[0]
        for column in reversed(IDENTITY_COLUMNS):
            aggregate.insert(0, column, identity_row[column])
        expected_repetition_frames.append(aggregate)
    expected_repetition_metrics = pd.concat(expected_repetition_frames, ignore_index=True).sort_values(["repetition", "model_name", "metric"]).reset_index(drop=True)
    _assert_frame_equal(repetition_metrics, expected_repetition_metrics, sort_columns=("repetition", "model_name", "metric"), context="Repetition metrics")
    expected_variability, expected_ranks, expected_rank_summary, expected_stability = summarize_repeated_metrics_v3(repetition_metrics)
    _assert_frame_equal(variability, expected_variability, sort_columns=("model_name", "metric"), context="Variability summary")
    _assert_frame_equal(ranks, expected_ranks, sort_columns=("metric", "repetition", "rank", "model_name"), context="Ranks by repetition")
    _assert_frame_equal(rank_summary, expected_rank_summary, sort_columns=("metric", "model_name"), context="Model-rank summary")
    _assert_frame_equal(stability, expected_stability, sort_columns=("metric",), context="Ordering stability")
    expected_frequency = selected_candidate_frequency_v3(selected)
    _assert_frame_equal(frequency, expected_frequency, sort_columns=("model_name", "selected_candidate_index"), context="Selected-candidate frequency")

    priority = variability.pivot(index="model_name", columns="metric", values="mean")
    return {
        "status": "passed",
        "run_id": metadata["run_id"],
        "generation_commit": commit,
        "scientific_input_sha256": metadata["scientific_input_sha256"],
        "repeated_contract_sha256": metadata["repeated_contract_sha256"],
        "file_count": len(EXPECTED_FILES),
        "repetitions": 5,
        "model_count": 9,
        "distinct_outer_assignment_count": len(semantic_hashes),
        "oof_prediction_row_count": len(oof),
        "candidate_search_row_count": len(candidates),
        "fold_metric_row_count": len(fold_metrics),
        "repetition_metric_row_count": len(repetition_metrics),
        "best_mean_macro_f1_model": str(priority["macro_f1"].idxmax()),
        "best_mean_balanced_accuracy_model": str(priority["balanced_accuracy"].idxmax()),
        "best_mean_qwk_model": str(priority["quadratic_weighted_kappa"].idxmax()),
        "best_mean_ordinal_mae_model": str(priority["ordinal_mae"].idxmin()),
        "source_tree_receipt_internally_consistent": True,
        "generation_implementation_blobs_verified": len(EXPECTED_IMPLEMENTATIONS),
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(json.dumps(validate_repeated_nested_cv_run_v3(args.run_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_FILES",
    "V3RepeatedNestedCVRunValidationError",
    "validate_repeated_nested_cv_run_v3",
]
