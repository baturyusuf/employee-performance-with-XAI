"""Independent closed-world validator for a completed HRDataset v3 sensitivity run.

This module deliberately does not import the Phase 3A runner. It rebinds the
generation commit and source contracts, reconstructs every fold assignment,
replays the label-only baselines and persisted sigmoid calibrators, and
independently recalculates every aggregate result table.
"""

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
from src.data.external_adapters import load_external_dataset
from src.experiments.manuscript_calibration import (
    apply_sigmoid_calibrator,
    calibrator_from_parameter_rows,
    fit_sigmoid_calibrator,
)
from src.experiments.shared_folds import SharedFoldArtifacts, generate_shared_folds, validate_shared_folds
from src.governance.hrdataset_sensitivity_contract_v3 import (
    ALTERNATIVE_FORMULATION,
    BASELINES,
    DEFAULT_HRDATASET_SENSITIVITY_CONTRACT,
    FORMULATIONS,
    METRICS,
    PRIMARY_FORMULATION,
    PRIORITY_METRICS,
    validate_hrdataset_sensitivity_contract_v3,
)
from src.governance.offline_runtime import validate_policy_receipt
from src.models.canonical_models import aligned_predict_proba
from src.models.ordinal_evaluation_v3 import ordinal_evaluation_bundle_v3
from src.models.ordinal_models_v3 import build_v3_naive_baseline
from src.utils.config_loader import PROJECT_ROOT, load_config


SYSTEMS = ("xgboost_raw", "xgboost_sigmoid", *BASELINES)
DEFAULT_HRDATASET_SENSITIVITY_RUN = Path(
    "reports/major_revision_v3_runs/"
    "phase3a_v3_20260907T141213Z_f6e6a0a/hrdataset_sensitivity"
)
EXPECTED_FILES = frozenset(
    {
        "baseline_comparisons.csv",
        "calibration_training_oof.csv",
        "calibrator_parameters.csv",
        "candidate_search_results.csv",
        "canonical_v2_confusion_matrix.csv",
        "canonical_v2_per_class_metrics.csv",
        "confusion_matrices.csv",
        "cv_design_sensitivity.csv",
        "fold_contracts.json",
        "fold_metrics.csv",
        "oof_predictions.csv",
        "per_class_metrics.csv",
        "protocol_comparison.csv",
        "repetition_metrics.csv",
        "selected_hyperparameters.csv",
        "stage_metadata.json",
        "target_mapping_support.csv",
        "variability_summary.csv",
    }
)
OUTPUT_HASH_FILES = EXPECTED_FILES - {"stage_metadata.json"}
EXPECTED_IMPLEMENTATIONS = frozenset(
    {
        "src/experiments/hrdataset_sensitivity_v3.py",
        "src/governance/hrdataset_sensitivity_contract_v3.py",
        "src/experiments/shared_folds.py",
        "src/models/canonical_models.py",
        "src/models/ordinal_models_v3.py",
        "src/models/ordinal_evaluation_v3.py",
        "src/experiments/manuscript_calibration.py",
    }
)
BASE_IDENTITY_COLUMNS = (
    "run_id",
    "contract_sha256",
    "scientific_input_sha256",
    "dataset_sha256",
    "repetition",
    "outer_seed",
    "inner_seed",
    "model_seed",
    "calibration_seed",
    "baseline_seed",
    "fold_contract_hash",
    "formulation_id",
)
class HRDatasetSensitivityRunValidationV3Error(RuntimeError):
    """Raised when persisted Phase 3A evidence is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HRDatasetSensitivityRunValidationV3Error(message)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HRDatasetSensitivityRunValidationV3Error(
            f"Could not read {path.as_posix()}: {exc}"
        ) from exc


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, float_precision="round_trip")
    except Exception as exc:
        raise HRDatasetSensitivityRunValidationV3Error(
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_digest(value: Any, *, length: int = 64) -> bool:
    text = str(value)
    return len(text) == length and all(character in "0123456789abcdef" for character in text)


def _git_blob(commit: str, relative_path: str, *, required: bool = True) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        )
        return completed.stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        if not required:
            return None
        raise HRDatasetSensitivityRunValidationV3Error(
            f"Could not resolve generation blob {commit}:{relative_path}: {exc}"
        ) from exc


def _assert_frame_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_columns: Sequence[str],
    context: str,
    atol: float = 1e-14,
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
            atol=atol,
        )
    except AssertionError as exc:
        raise HRDatasetSensitivityRunValidationV3Error(
            f"{context} does not match independent recomputation: {exc}"
        ) from exc


def _json_object(value: Any, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HRDatasetSensitivityRunValidationV3Error(
            f"{context} is not valid JSON: {exc}"
        ) from exc
    _require(isinstance(payload, dict), f"{context} must contain an object.")
    return dict(payload)


def _json_number_list(value: Any, *, context: str) -> list[float]:
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HRDatasetSensitivityRunValidationV3Error(
            f"{context} is not valid JSON: {exc}"
        ) from exc
    _require(isinstance(payload, list) and len(payload) == 5, f"{context} must contain five scores.")
    values = [float(item) for item in payload]
    _require(all(math.isfinite(item) for item in values), f"{context} contains non-finite values.")
    return values


def _independent_selected_index(primary: Sequence[float], secondary: Sequence[float], tolerance: float) -> int:
    primary_array = np.asarray(primary, dtype=float)
    secondary_array = np.asarray(secondary, dtype=float)
    _require(
        primary_array.shape == secondary_array.shape
        and primary_array.ndim == 1
        and len(primary_array) > 0
        and np.isfinite(primary_array).all()
        and np.isfinite(secondary_array).all(),
        "Candidate selection vectors are invalid.",
    )
    optimum = float(primary_array.max())
    slack = np.finfo(float).eps * 8.0 * max(
        1.0, abs(optimum), abs(float(tolerance)), float(np.max(np.abs(primary_array)))
    )
    eligible = np.flatnonzero(optimum - primary_array <= float(tolerance) + slack)
    _require(len(eligible) > 0, "Candidate practical-tie pool is empty.")
    best_secondary = float(np.max(secondary_array[eligible]))
    winners = eligible[secondary_array[eligible] == best_secondary]
    return int(winners[0])


def _labels_by_formulation(contract: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        str(record["formulation_id"]): tuple(int(value) for value in record["ordered_labels"])
        for record in contract["target_formulations"]
    }


def _targets_by_formulation(dataset: Any, contract: Mapping[str, Any]) -> dict[str, pd.Series]:
    raw = dataset.raw["PerformanceScore"].astype(str).str.strip()
    result: dict[str, pd.Series] = {}
    for formulation in contract["target_formulations"]:
        formulation_id = str(formulation["formulation_id"])
        target = raw.map(dict(formulation["mapping"]))
        _require(not target.isna().any(), f"Target mapping is incomplete for {formulation_id}.")
        target = target.astype(int)
        target.index = dataset.canonical.index
        result[formulation_id] = target
    return result


def _identity_lookup(
    metadata: Mapping[str, Any],
    contract: Mapping[str, Any],
    fold_records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    seeds = {int(row["repetition"]): dict(row) for row in contract["design"]["seed_schedule"]}
    identities: dict[tuple[str, int], dict[str, Any]] = {}
    for record in fold_records:
        formulation_id = str(record.get("formulation_id", ""))
        repetition = int(record.get("repetition", -1))
        _require(formulation_id in FORMULATIONS and repetition in seeds, "Fold-record identity drifted.")
        seed = seeds[repetition]
        for field in ("outer_seed", "inner_seed", "model_seed", "calibration_seed", "baseline_seed"):
            _require(int(record.get(field, -1)) == int(seed[field]), f"Fold record {field} drifted.")
        fold_contract = record.get("contract")
        _require(isinstance(fold_contract, Mapping), "Fold record lacks its contract.")
        key = (formulation_id, repetition)
        _require(key not in identities, "Fold record grid contains a duplicate.")
        identities[key] = {
            "run_id": metadata["run_id"],
            "contract_sha256": metadata["contract_sha256"],
            "scientific_input_sha256": metadata["scientific_input_sha256"],
            "dataset_sha256": metadata["dataset_sha256"],
            "repetition": repetition,
            "outer_seed": int(seed["outer_seed"]),
            "inner_seed": int(seed["inner_seed"]),
            "model_seed": int(seed["model_seed"]),
            "calibration_seed": int(seed["calibration_seed"]),
            "baseline_seed": int(seed["baseline_seed"]),
            "fold_contract_hash": str(fold_contract.get("fold_contract_hash", "")),
            "formulation_id": formulation_id,
        }
    expected = {(formulation, repetition) for formulation in FORMULATIONS for repetition in range(1, 6)}
    _require(set(identities) == expected, "Fold record formulation/repetition grid drifted.")
    return identities


def _validate_identity(
    frame: pd.DataFrame,
    identities: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    context: str,
) -> None:
    _require(set(BASE_IDENTITY_COLUMNS).issubset(frame.columns), f"{context} identity schema drifted.")
    for (formulation_id, repetition), rows in frame.groupby(["formulation_id", "repetition"], sort=True):
        key = (str(formulation_id), int(repetition))
        _require(key in identities, f"{context} contains an unknown identity.")
        for column, expected in identities[key].items():
            observed = set(rows[column].astype(str))
            _require(observed == {str(expected)}, f"{context} identity drifted for {key}/{column}.")


def _rebuild_folds(
    dataset: Any,
    targets: Mapping[str, pd.Series],
    metadata: Mapping[str, Any],
    fold_records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], SharedFoldArtifacts]:
    artifacts: dict[tuple[str, int], SharedFoldArtifacts] = {}
    semantic_hashes: set[str] = set()
    for record in fold_records:
        formulation_id = str(record["formulation_id"])
        repetition = int(record["repetition"])
        observed_contract = record["contract"]
        source = dataset.canonical.copy()
        source["Phase3ATarget"] = targets[formulation_id].astype(int)
        rebuilt = generate_shared_folds(
            source,
            target_column="Phase3ATarget",
            id_column="ExternalSampleId",
            run_id=str(observed_contract["run_id"]),
            config_hash=str(metadata["contract_sha256"]),
            scientific_input_hash=str(metadata["scientific_input_sha256"]),
            dataset_key=f"hrdataset_v14_{formulation_id}",
            dataset_sha256=str(metadata["dataset_sha256"]),
            outer_splits=5,
            inner_splits=5,
            seed=int(record["outer_seed"]),
            inner_seed=int(record["inner_seed"]),
        )
        validate_shared_folds(rebuilt, source_frame=source)
        _require(
            _canonical_json_sha256(rebuilt.contract) == _canonical_json_sha256(observed_contract),
            f"Fold contract does not replay for {formulation_id}/repetition={repetition}.",
        )
        semantic_rows = (
            rebuilt.outer_assignments[["sample_index", "outer_fold", "y_true"]]
            .astype(int)
            .sort_values("sample_index")
            .to_dict(orient="records")
        )
        semantic_hash = _canonical_json_sha256(semantic_rows)
        _require(
            semantic_hash == record.get("outer_assignment_semantic_sha256"),
            "Outer-assignment semantic hash drifted.",
        )
        semantic_hashes.add(semantic_hash)
        artifacts[(formulation_id, repetition)] = rebuilt
    _require(len(semantic_hashes) == 10, "The ten target-specific outer assignments are not distinct.")
    return artifacts


def _probability_columns(labels: Sequence[int]) -> list[str]:
    return [f"prob_class_{int(label)}" for label in labels]


def _bundle(rows: pd.DataFrame, labels: Sequence[int], system: str) -> Mapping[str, Any]:
    probability = rows[_probability_columns(labels)].to_numpy(float)
    return ordinal_evaluation_bundle_v3(
        rows["y_true"].astype(int),
        rows["y_pred"].astype(int),
        probability,
        labels=labels,
        dataset_key="hrdataset_v14",
        model_name=system,
    )


def _validate_probability_rows(rows: pd.DataFrame, labels: Sequence[int], *, context: str) -> None:
    probability = rows[_probability_columns(labels)].to_numpy(float)
    _require(np.isfinite(probability).all(), f"{context} probabilities are non-finite.")
    _require(np.all((probability >= 0.0) & (probability <= 1.0)), f"{context} probabilities escaped [0,1].")
    _require(
        np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-9),
        f"{context} probability simplex drifted.",
    )
    if "y_pred" in rows:
        expected = np.asarray(labels, dtype=int)[np.argmax(probability, axis=1)]
        _require(np.array_equal(expected, rows["y_pred"].to_numpy(int)), f"{context} argmax prediction drifted.")


def _validate_candidate_selection(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    contract: Mapping[str, Any],
    identities: Mapping[tuple[str, int], Mapping[str, Any]],
) -> None:
    _validate_identity(candidates, identities, context="Candidate search")
    _validate_identity(selected, identities, context="Selected hyperparameters")
    _require(len(candidates) == 400, "Candidate-search row count drifted.")
    _require(len(selected) == 50, "Selected-hyperparameter row count drifted.")
    _require(not candidates["outer_test_used_for_selection"].astype(bool).any(), "Outer test entered candidate selection.")
    _require(not selected["outer_test_used_for_selection"].astype(bool).any(), "Outer test entered selected records.")
    grid = load_config(PROJECT_ROOT / contract["source_contracts"]["model_grid"]["path"])
    definition = grid["model_benchmark"]["models"]["xgboost"]
    candidate_definitions = [dict(value) for value in definition["candidates"]]
    tolerance = float(contract["model_protocol"]["primary_tie_tolerance"])
    expected_grid = {
        (formulation, repetition, outer_fold)
        for formulation in FORMULATIONS
        for repetition in range(1, 6)
        for outer_fold in range(1, 6)
    }
    observed_grid: set[tuple[str, int, int]] = set()
    for key, rows in candidates.groupby(["formulation_id", "repetition", "outer_fold"], sort=True):
        normalized_key = (str(key[0]), int(key[1]), int(key[2]))
        observed_grid.add(normalized_key)
        ordered = rows.sort_values("candidate_index")
        _require(ordered["candidate_index"].astype(int).tolist() == list(range(8)), "Candidate index grid drifted.")
        primary_means: list[float] = []
        secondary_means: list[float] = []
        for expected_index, (_, row) in enumerate(ordered.iterrows()):
            _require(
                _json_object(row["candidate_parameters_json"], context="candidate parameters")
                == candidate_definitions[expected_index],
                "Candidate parameters drifted from the bound grid.",
            )
            primary = _json_number_list(row["inner_macro_f1_scores_json"], context="inner macro-F1")
            secondary = _json_number_list(row["inner_qwk_scores_json"], context="inner QWK")
            primary_mean = float(np.mean(primary))
            secondary_mean = float(np.mean(secondary))
            _require(math.isclose(float(row["inner_macro_f1_mean"]), primary_mean, rel_tol=0.0, abs_tol=1e-14), "Candidate macro-F1 mean drifted.")
            _require(math.isclose(float(row["inner_qwk_mean"]), secondary_mean, rel_tol=0.0, abs_tol=1e-14), "Candidate QWK mean drifted.")
            primary_means.append(primary_mean)
            secondary_means.append(secondary_mean)
        expected_index = _independent_selected_index(primary_means, secondary_means, tolerance)
        selected_flags = ordered.loc[ordered["selected_by_protocol"].astype(bool), "candidate_index"].astype(int).tolist()
        _require(selected_flags == [expected_index], "Candidate selection rule drifted.")
        selected_row = selected.loc[
            (selected["formulation_id"].astype(str) == normalized_key[0])
            & (selected["repetition"].astype(int) == normalized_key[1])
            & (selected["outer_fold"].astype(int) == normalized_key[2])
        ]
        _require(len(selected_row) == 1, "Selected-candidate linkage is not one-to-one.")
        selected_record = selected_row.iloc[0]
        candidate_record = ordered.loc[ordered["candidate_index"].astype(int) == expected_index].iloc[0]
        _require(int(selected_record["selected_candidate_index"]) == expected_index, "Selected index drifted.")
        _require(
            _json_object(selected_record["selected_candidate_parameters_json"], context="selected parameters")
            == candidate_definitions[expected_index],
            "Selected parameters drifted.",
        )
        _require(
            _json_object(selected_record["fixed_parameters_json"], context="fixed parameters")
            == dict(definition["fixed_params"]),
            "Fixed XGBoost parameters drifted.",
        )
        _require(math.isclose(float(selected_record["selected_inner_macro_f1_mean"]), float(candidate_record["inner_macro_f1_mean"]), rel_tol=0.0, abs_tol=1e-14), "Selected macro-F1 lineage drifted.")
        _require(math.isclose(float(selected_record["selected_inner_qwk_mean"]), float(candidate_record["inner_qwk_mean"]), rel_tol=0.0, abs_tol=1e-14), "Selected QWK lineage drifted.")
    _require(observed_grid == expected_grid, "Candidate formulation/repetition/fold grid drifted.")


def _validate_oof_and_replay_baselines(
    oof: pd.DataFrame,
    targets: Mapping[str, pd.Series],
    labels_by_formulation: Mapping[str, Sequence[int]],
    identities: Mapping[tuple[str, int], Mapping[str, Any]],
    artifacts: Mapping[tuple[str, int], SharedFoldArtifacts],
    selected: pd.DataFrame,
) -> None:
    _validate_identity(oof, identities, context="OOF")
    _require(len(oof) == 15_550, "OOF row count drifted.")
    selected_lookup = selected.set_index(["formulation_id", "repetition", "outer_fold"])["selected_candidate_index"]
    expected_grid = {
        (formulation, repetition, system)
        for formulation in FORMULATIONS
        for repetition in range(1, 6)
        for system in SYSTEMS
    }
    observed_grid: set[tuple[str, int, str]] = set()
    for key, rows in oof.groupby(["formulation_id", "repetition", "system"], sort=True):
        formulation_id, repetition, system = str(key[0]), int(key[1]), str(key[2])
        observed_grid.add((formulation_id, repetition, system))
        labels = tuple(labels_by_formulation[formulation_id])
        _require(len(rows) == 311 and rows["sample_index"].nunique() == 311, "Exactly-once OOF coverage drifted.")
        _require(set(rows["sample_index"].astype(int)) == set(range(311)), "OOF sample set drifted.")
        current = rows.set_index("sample_index")[["outer_fold", "y_true"]].astype(int).sort_index()
        expected = artifacts[(formulation_id, repetition)].outer_assignments.set_index("sample_index")[["outer_fold", "y_true"]].astype(int).sort_index()
        _require(current.equals(expected), "OOF fold/target lineage drifted.")
        _require(np.array_equal(current["y_true"].to_numpy(int), targets[formulation_id].to_numpy(int)), "OOF target differs from mapped raw target.")
        _validate_probability_rows(rows, labels, context=f"OOF {formulation_id}/{repetition}/{system}")
        unused = {1, 2, 3, 4}.difference(labels)
        for label in unused:
            _require(rows[f"prob_class_{label}"].isna().all(), "An undeclared-class probability is populated.")
        for outer_fold, fold_rows in rows.groupby("outer_fold", sort=True):
            expected_index = int(selected_lookup.loc[(formulation_id, repetition, int(outer_fold))])
            if system in {"xgboost_raw", "xgboost_sigmoid"}:
                _require(fold_rows["selected_candidate_index"].notna().all(), "XGBoost OOF candidate lineage is absent.")
                _require(fold_rows["selected_candidate_index"].astype(int).eq(expected_index).all(), "XGBoost OOF candidate lineage drifted.")
            else:
                _require(fold_rows["selected_candidate_index"].isna().all(), "A baseline has a selected candidate.")
    _require(observed_grid == expected_grid, "OOF system grid drifted.")

    for formulation_id in FORMULATIONS:
        labels = tuple(labels_by_formulation[formulation_id])
        target = targets[formulation_id]
        for repetition in range(1, 6):
            outer = artifacts[(formulation_id, repetition)].outer_assignments
            baseline_seed = int(identities[(formulation_id, repetition)]["baseline_seed"])
            for outer_fold in range(1, 6):
                train_ids = outer.loc[outer["outer_fold"].astype(int) != outer_fold, "sample_index"].astype(int).tolist()
                test_ids = outer.loc[outer["outer_fold"].astype(int) == outer_fold, "sample_index"].astype(int).tolist()
                train = pd.DataFrame({"inert_baseline_design": np.zeros(len(train_ids), dtype=np.int8)}, index=train_ids)
                test = pd.DataFrame({"inert_baseline_design": np.zeros(len(test_ids), dtype=np.int8)}, index=test_ids)
                for baseline_position, baseline_name in enumerate(BASELINES):
                    estimator = build_v3_naive_baseline(
                        baseline_name,
                        random_state=baseline_seed + outer_fold * 10 + baseline_position,
                    )
                    estimator.fit(train, target.loc[train_ids])
                    expected_probability = aligned_predict_proba(estimator, test, labels=labels)
                    observed = oof.loc[
                        (oof["formulation_id"].astype(str) == formulation_id)
                        & (oof["repetition"].astype(int) == repetition)
                        & (oof["outer_fold"].astype(int) == outer_fold)
                        & (oof["system"].astype(str) == baseline_name)
                    ].set_index("sample_index").loc[test_ids]
                    np.testing.assert_allclose(
                        observed[_probability_columns(labels)].to_numpy(float),
                        expected_probability,
                        rtol=0.0,
                        atol=1e-15,
                        err_msg=f"Baseline replay drifted for {formulation_id}/{repetition}/{outer_fold}/{baseline_name}",
                    )


def _validate_calibration_replay(
    training: pd.DataFrame,
    parameters: pd.DataFrame,
    oof: pd.DataFrame,
    labels_by_formulation: Mapping[str, Sequence[int]],
    identities: Mapping[tuple[str, int], Mapping[str, Any]],
    artifacts: Mapping[tuple[str, int], SharedFoldArtifacts],
    selected: pd.DataFrame,
) -> float:
    _validate_identity(training, identities, context="Calibration training OOF")
    _require(len(training) == 12_440, "Calibration-training OOF row count drifted.")
    _require(not training["outer_test_used_for_fit_or_selection"].astype(bool).any(), "Outer test entered calibration training.")
    _require(len(parameters) == 175, "Calibrator parameter row count drifted.")
    _require(not parameters["outer_test_used_for_fit_or_selection"].astype(bool).any(), "Outer test entered calibrator parameters.")
    selected_lookup = selected.set_index(["formulation_id", "repetition", "outer_fold"])["selected_candidate_index"]
    max_error = 0.0
    expected_parameter_rows = 0
    for formulation_id in FORMULATIONS:
        labels = tuple(labels_by_formulation[formulation_id])
        for repetition in range(1, 6):
            identity = identities[(formulation_id, repetition)]
            outer = artifacts[(formulation_id, repetition)].outer_assignments
            for outer_fold in range(1, 6):
                train_ids = outer.loc[outer["outer_fold"].astype(int) != outer_fold, "sample_index"].astype(int).tolist()
                test_ids = outer.loc[outer["outer_fold"].astype(int) == outer_fold, "sample_index"].astype(int).tolist()
                scoped = training.loc[
                    (training["formulation_id"].astype(str) == formulation_id)
                    & (training["repetition"].astype(int) == repetition)
                    & (training["outer_fold"].astype(int) == outer_fold)
                ].sort_values("sample_index")
                _require(scoped["sample_index"].astype(int).tolist() == train_ids, "Calibration training membership drifted.")
                expected_target = outer.set_index("sample_index").loc[train_ids, "y_true"].to_numpy(int)
                _require(np.array_equal(scoped["y_true"].to_numpy(int), expected_target), "Calibration training target drifted.")
                selected_index = int(selected_lookup.loc[(formulation_id, repetition, outer_fold)])
                _require(scoped["selected_candidate_index"].astype(int).eq(selected_index).all(), "Calibration candidate lineage drifted.")
                _validate_probability_rows(scoped, labels, context="Calibration training")
                parameter_rows = parameters.loc[
                    (parameters["formulation_id"].astype(str) == formulation_id)
                    & (parameters["repetition"].astype(int) == repetition)
                    & (parameters["outer_fold"].astype(int) == outer_fold)
                ].copy()
                _require(len(parameter_rows) == len(labels), "Calibrator class-parameter grid drifted.")
                expected_parameter_rows += len(labels)
                for column, expected in identity.items():
                    if column == "calibration_seed":
                        continue
                    _require(set(parameter_rows[column].astype(str)) == {str(expected)}, f"Calibrator identity drifted for {column}.")
                _require(parameter_rows["calibration_seed"].astype(int).eq(int(identity["calibration_seed"]) + outer_fold).all(), "Calibrator effective seed drifted.")
                _require(parameter_rows["selected_candidate_index"].astype(int).eq(selected_index).all(), "Calibrator candidate lineage drifted.")
                _require(set(parameter_rows["class_label"].astype(int)) == set(labels), "Calibrator label set drifted.")
                reconstructed = calibrator_from_parameter_rows(parameter_rows)
                training_probability = scoped[_probability_columns(labels)].to_numpy(float)
                training_target = scoped["y_true"].to_numpy(int)
                refitted = fit_sigmoid_calibrator(
                    training_probability,
                    training_target,
                    labels,
                    seed=int(identity["calibration_seed"]) + outer_fold,
                )
                _require(refitted.parameter_sha256 == reconstructed.parameter_sha256, "Refitted sigmoid parameter hash drifted.")
                _require(refitted.training_probability_sha256 == reconstructed.training_probability_sha256, "Sigmoid training-probability hash drifted.")
                _require(refitted.training_labels_sha256 == reconstructed.training_labels_sha256, "Sigmoid training-label hash drifted.")
                raw = oof.loc[
                    (oof["formulation_id"].astype(str) == formulation_id)
                    & (oof["repetition"].astype(int) == repetition)
                    & (oof["outer_fold"].astype(int) == outer_fold)
                    & (oof["system"].astype(str) == "xgboost_raw")
                ].set_index("sample_index").loc[test_ids]
                sigmoid = oof.loc[
                    (oof["formulation_id"].astype(str) == formulation_id)
                    & (oof["repetition"].astype(int) == repetition)
                    & (oof["outer_fold"].astype(int) == outer_fold)
                    & (oof["system"].astype(str) == "xgboost_sigmoid")
                ].set_index("sample_index").loc[test_ids]
                replay = apply_sigmoid_calibrator(reconstructed, raw[_probability_columns(labels)].to_numpy(float))
                observed = sigmoid[_probability_columns(labels)].to_numpy(float)
                error = float(np.max(np.abs(replay - observed)))
                max_error = max(max_error, error)
                _require(error <= 1e-14, "Outer sigmoid probability replay drifted.")
    _require(expected_parameter_rows == len(parameters), "Unexpected calibrator parameter rows are present.")
    membership_counts = training.groupby(["formulation_id", "repetition", "sample_index"]).size()
    _require(membership_counts.eq(4).all(), "Each sample must enter exactly four outer-training partitions per repetition.")
    return max_error


def _recompute_metric_tables(
    oof: pd.DataFrame,
    labels_by_formulation: Mapping[str, Sequence[int]],
    identities: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for key, rows in oof.groupby(["formulation_id", "repetition", "system"], sort=True):
        formulation_id, repetition, system = str(key[0]), int(key[1]), str(key[2])
        labels = tuple(labels_by_formulation[formulation_id])
        for outer_fold, scoped in rows.groupby("outer_fold", sort=True):
            bundle = _bundle(scoped.sort_values("sample_index"), labels, system)
            fold_rows.append(
                {
                    **identities[(formulation_id, repetition)],
                    "system": system,
                    "outer_fold": int(outer_fold),
                    "n_train": 311 - len(scoped),
                    "n_test": len(scoped),
                    **bundle["aggregate_metrics"],
                }
            )
        bundle = _bundle(rows.sort_values("sample_index"), labels, system)
        summary_identity = {
            "run_id": identities[(formulation_id, repetition)]["run_id"],
            "contract_sha256": identities[(formulation_id, repetition)]["contract_sha256"],
            "scientific_input_sha256": identities[(formulation_id, repetition)]["scientific_input_sha256"],
            "formulation_id": formulation_id,
            "repetition": repetition,
            "system": system,
            "ordered_labels_json": json.dumps(list(labels), separators=(",", ":")),
            "n_samples": len(rows),
        }
        for metric, value in bundle["aggregate_metrics"].items():
            if metric in METRICS:
                metric_rows.append({**summary_identity, "metric": metric, "value": float(value)})
        class_rows.extend({**summary_identity, **record} for record in bundle["per_class_metrics"])
        confusion_rows.extend({**summary_identity, **record} for record in bundle["confusion_matrix"])
    repetition = pd.DataFrame(metric_rows)
    per_class = pd.DataFrame(class_rows).drop(columns=["dataset_key", "model_name"])
    confusion = pd.DataFrame(confusion_rows).drop(columns=["dataset_key", "model_name"])
    variability_rows: list[dict[str, Any]] = []
    for key, rows in repetition.groupby(["formulation_id", "system", "metric"], sort=True):
        values = rows.sort_values("repetition")["value"].to_numpy(float)
        _require(len(values) == 5 and np.isfinite(values).all(), "Variability input grid drifted.")
        variability_rows.append(
            {
                "formulation_id": str(key[0]),
                "system": str(key[1]),
                "metric": str(key[2]),
                "repetition_count": 5,
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "range_interpretation": "empirical_training_and_fold_variability_not_confidence_interval",
            }
        )
    return (
        pd.DataFrame(fold_rows),
        repetition,
        pd.DataFrame(variability_rows),
        per_class,
        confusion,
    )


def _baseline_comparisons(repetition: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    priority = repetition.loc[repetition["metric"].isin(PRIORITY_METRICS)]
    for key, scoped in priority.groupby(["formulation_id", "repetition", "metric"], sort=True):
        values = scoped.set_index("system")["value"]
        for baseline in BASELINES:
            metric = str(key[2])
            rows.append(
                {
                    "formulation_id": str(key[0]),
                    "repetition": int(key[1]),
                    "metric": metric,
                    "comparison": f"xgboost_raw_minus_{baseline}",
                    "xgboost_raw_value": float(values["xgboost_raw"]),
                    "baseline_value": float(values[baseline]),
                    "signed_arithmetic_difference": float(values["xgboost_raw"] - values[baseline]),
                    "better_direction": "higher" if metric in {"macro_f1", "quadratic_weighted_kappa"} else "lower",
                    "inference": "descriptive_matched_repetition_no_confidence_interval",
                }
            )
    return pd.DataFrame(rows)


def _canonical_v2_results(
    contract: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    class_outputs: list[pd.DataFrame] = []
    confusion_outputs: list[pd.DataFrame] = []
    metric_lookup: dict[str, dict[str, float]] = {}
    sources = contract["source_contracts"]
    for system, source_name in (("xgboost_raw", "v2_raw_oof"), ("xgboost_sigmoid", "v2_calibrated_oof")):
        frame = _read_csv(PROJECT_ROOT / sources[source_name]["path"])
        frame = frame.loc[frame["policy"].astype(str) == "conservative_primary"].sort_values("sample_index")
        probability = frame[["prob_class_2", "prob_class_3", "prob_class_4"]].to_numpy(float)
        bundle = ordinal_evaluation_bundle_v3(
            frame["y_true"].astype(int),
            frame["y_pred"].astype(int),
            probability,
            labels=(2, 3, 4),
            dataset_key="hrdataset_v14",
            model_name=system,
        )
        metric_lookup[system] = {
            name: float(value)
            for name, value in bundle["aggregate_metrics"].items()
            if name in METRICS
        }
        common = {
            "dataset_key": "hrdataset_v14",
            "design": "canonical_v2_single_10_outer_x_5_inner_nested_cv",
            "formulation_id": PRIMARY_FORMULATION,
            "system": system,
            "n_samples": 311,
            "source_run_id": str(frame["run_id"].iloc[0]),
        }
        class_outputs.append(
            pd.DataFrame([{**common, **row} for row in bundle["per_class_metrics"]]).drop(columns="model_name")
        )
        confusion_outputs.append(
            pd.DataFrame([{**common, **row} for row in bundle["confusion_matrix"]]).drop(columns="model_name")
        )
    return pd.concat(class_outputs, ignore_index=True), pd.concat(confusion_outputs, ignore_index=True), metric_lookup


def _cv_design(variability: pd.DataFrame, v2_metrics: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary = variability.loc[variability["formulation_id"].astype(str) == PRIMARY_FORMULATION]
    for system in ("xgboost_raw", "xgboost_sigmoid"):
        for metric in PRIORITY_METRICS:
            record = primary.loc[(primary["system"] == system) & (primary["metric"] == metric)]
            _require(len(record) == 1, "CV-design source cell is absent.")
            item = record.iloc[0]
            reference = float(v2_metrics[system][metric])
            minimum = float(item["minimum"])
            maximum = float(item["maximum"])
            rows.append(
                {
                    "formulation_id": PRIMARY_FORMULATION,
                    "system": system,
                    "metric": metric,
                    "canonical_v2_10x5_point_estimate": reference,
                    "repeated_5x5_mean": float(item["mean"]),
                    "repeated_5x5_sample_sd": float(item["sample_sd"]),
                    "repeated_5x5_minimum": minimum,
                    "repeated_5x5_maximum": maximum,
                    "ten_fold_inside_repeated_range": minimum <= reference <= maximum,
                    "signed_repeated_mean_minus_ten_fold": float(item["mean"]) - reference,
                    "interpretation": "descriptive_cv_design_sensitivity_not_equivalence_test",
                }
            )
    return pd.DataFrame(rows)


def _target_support(dataset: Any, contract: Mapping[str, Any]) -> pd.DataFrame:
    raw = dataset.raw["PerformanceScore"].astype(str).str.strip()
    support = raw.value_counts().to_dict()
    rows: list[dict[str, Any]] = []
    for formulation in contract["target_formulations"]:
        for raw_value, mapped_label in formulation["mapping"].items():
            rows.append(
                {
                    "formulation_id": str(formulation["formulation_id"]),
                    "formulation_role": str(formulation["role"]),
                    "raw_target_value": str(raw_value),
                    "mapped_label": int(mapped_label),
                    "observed_support": int(support.get(raw_value, 0)),
                    "observed_in_dataset": raw_value in support,
                    "mapping_rationale": str(formulation["rationale"]),
                    "target_equivalence_claim_allowed": False,
                }
            )
    return pd.DataFrame(rows)


def _protocol_comparison() -> pd.DataFrame:
    records = [
        ("folds", "canonical 10x5 plus v3 repeated 5x5", "canonical 10x5 plus v3 repeated 5x5", "same strategy; dataset-specific assignments", "partial", "not a common sample population"),
        ("tuning", "nested candidate selection for benchmark systems", "nested eight-candidate XGBoost selection", "same XGBoost selection implementation", "partial", "different dataset and fitted parameters"),
        ("calibration", "predeclared cross-fitted one-vs-rest sigmoid", "predeclared cross-fitted one-vs-rest sigmoid", "same algorithm", "partial", "probability quality is dataset conditional"),
        ("SHAP", "exact-fold OOF XGBoost raw-margin SHAP", "canonical-v2 exact-fold OOF XGBoost raw-margin SHAP", "same attribution implementation", "partial", "different feature families and no causal meaning"),
        ("subgroup", "support-aware descriptive diagnostics", "support-aware descriptive diagnostics", "same diagnostic framing", "False", "different attributes/support; no fairness certification"),
        ("proxy", "department reconstructability and proxy-use sensitivity", "department reconstructability in canonical v2", "partially shared diagnostic design", "False", "reconstructability is not performance-model use"),
        ("target", "native INX organisational rating 2/3/4", "dataset-specific mapped PerformanceScore; 3-class primary plus raw-order 4-class sensitivity", "different mapping implementation", "False", "semantics and prevalence are not equivalent"),
        ("features", "20-feature P3 primary policy", "seven-feature conservative-primary policy", "same training-only preprocessing", "False", "feature spaces and availability semantics differ"),
    ]
    return pd.DataFrame(
        records,
        columns=("component", "INX", "HRDataset_v14", "same_implementation", "same_semantics", "notes"),
    )


def _validate_sources_and_generation(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], int, int]:
    scientific_inputs = metadata.get("scientific_inputs")
    _require(isinstance(scientific_inputs, Mapping), "scientific_inputs must be an object.")
    _require(
        metadata.get("scientific_input_sha256") == _canonical_json_sha256(scientific_inputs),
        "Scientific-input composite hash drifted.",
    )
    for key in ("contract_sha256", "git_identity"):
        _require(metadata.get(key) == scientific_inputs.get(key), f"Scientific identity {key} is inconsistent.")
    git_identity = metadata.get("git_identity")
    _require(isinstance(git_identity, Mapping), "Git identity must be an object.")
    commit = str(git_identity.get("commit", ""))
    _require(_valid_digest(commit, length=40), "Generation commit is invalid.")
    contract_blob = _git_blob(commit, DEFAULT_HRDATASET_SENSITIVITY_CONTRACT.as_posix())
    _require(contract_blob is not None, "Generation contract blob is absent.")
    _require(_sha256_bytes(contract_blob) == metadata["contract_sha256"], "Generation contract hash drifted.")
    generation_contract = json.loads(contract_blob.decode("utf-8"))
    current_receipt = validate_hrdataset_sensitivity_contract_v3()
    _require(current_receipt["contract_sha256"] == metadata["contract_sha256"], "Current bound contract differs from generation.")
    _require(current_receipt["source_hashes"] == scientific_inputs.get("source_hashes"), "Bound source hashes drifted.")
    tracked_sources = 0
    local_sources = 0
    for name, record in generation_contract["source_contracts"].items():
        _require(scientific_inputs["source_hashes"].get(name) == record["sha256"], f"Generation source receipt drifted for {name}.")
        blob = _git_blob(commit, str(record["path"]), required=False)
        if blob is None:
            local_sources += 1
            _require(sha256_file(PROJECT_ROOT / str(record["path"])) == record["sha256"], f"Local bound source drifted for {name}.")
        else:
            tracked_sources += 1
            _require(_sha256_bytes(blob) == record["sha256"], f"Generation source blob drifted for {name}.")
    implementation_hashes = scientific_inputs.get("implementation_hashes")
    _require(isinstance(implementation_hashes, Mapping), "Implementation hashes are absent.")
    _require(set(implementation_hashes) == EXPECTED_IMPLEMENTATIONS, "Implementation inventory drifted.")
    for relative_path, expected_hash in implementation_hashes.items():
        blob = _git_blob(commit, str(relative_path))
        _require(blob is not None and _sha256_bytes(blob) == expected_hash, f"Generation implementation hash drifted for {relative_path}.")
    _require(_valid_digest(scientific_inputs.get("source_tree_hash")), "Source-tree receipt is invalid.")
    return generation_contract, tracked_sources, local_sources


def validate_hrdataset_sensitivity_run_v3(
    run_dir: Path | str = DEFAULT_HRDATASET_SENSITIVITY_RUN,
) -> dict[str, Any]:
    """Validate and independently recompute one complete Phase 3A run."""

    root = Path(run_dir)
    _require(root.is_dir(), f"Run directory does not exist: {root.as_posix()}.")
    files = {path.name for path in root.iterdir() if path.is_file()}
    directories = [path.name for path in root.iterdir() if path.is_dir()]
    _require(not directories, f"Unexpected run subdirectories: {directories}.")
    _require(
        files == EXPECTED_FILES,
        "Run closed-world inventory drifted: "
        f"missing={sorted(EXPECTED_FILES - files)}, unexpected={sorted(files - EXPECTED_FILES)}.",
    )
    metadata = _load_json(root / "stage_metadata.json")
    _require(isinstance(metadata, Mapping), "Stage metadata must be an object.")
    expected_metadata = {
        "schema_version": 1,
        "stage": "hrdataset_sensitivity_v3",
        "status": "complete",
        "evidence_status": "complete_five_repetition_two_mapping_exactly_once_oof",
        "sample_count": 311,
        "feature_policy": "conservative_primary",
        "feature_count": 7,
        "formulations": list(FORMULATIONS),
        "repetitions_per_formulation": 5,
        "outer_folds": 5,
        "inner_folds": 5,
        "systems": list(SYSTEMS),
        "xgboost_fit_calls": 2300,
        "baseline_fit_calls": 150,
        "candidate_search_row_count": 400,
        "oof_prediction_row_count": 15_550,
        "calibration_training_oof_row_count": 12_440,
        "repetition_metric_row_count": 500,
        "calibrator_parameter_row_count": 175,
        "outer_test_used_for_selection_or_calibration": False,
        "cross_formulation_metric_difference_computed": False,
        "locked_model_transport": False,
        "employee_level_output_publication_authorized": False,
        "network_calls": 0,
        "paid_api_calls": 0,
    }
    for key, expected in expected_metadata.items():
        _require(metadata.get(key) == expected, f"Metadata {key} drifted.")
    _require(_valid_digest(metadata.get("contract_sha256")), "Metadata contract hash is invalid.")
    _require(_valid_digest(metadata.get("scientific_input_sha256")), "Metadata scientific-input hash is invalid.")
    _require(metadata.get("dataset_sha256") == metadata["scientific_inputs"]["source_hashes"]["raw_dataset"], "Dataset identity drifted.")
    try:
        validate_policy_receipt(metadata.get("runtime_policy", {}))
    except Exception as exc:
        raise HRDatasetSensitivityRunValidationV3Error(f"Offline runtime receipt drifted: {exc}") from exc
    output_hashes = metadata.get("output_hashes")
    _require(isinstance(output_hashes, Mapping), "Metadata output hashes must be an object.")
    _require(set(output_hashes) == OUTPUT_HASH_FILES, "Output-hash inventory drifted.")
    for filename in sorted(OUTPUT_HASH_FILES):
        _require(output_hashes[filename] == sha256_file(root / filename), f"Output byte hash drifted for {filename}.")

    contract, tracked_sources, local_sources = _validate_sources_and_generation(metadata)
    sources = contract["source_contracts"]
    dataset = load_external_dataset(
        "hrdataset_v14",
        raw_path=PROJECT_ROOT / sources["raw_dataset"]["path"],
        schema_mapping_path=PROJECT_ROOT / sources["schema_mapping"]["path"],
    )
    labels_by_formulation = _labels_by_formulation(contract)
    targets = _targets_by_formulation(dataset, contract)
    fold_records = _load_json(root / "fold_contracts.json")
    _require(isinstance(fold_records, list) and len(fold_records) == 10, "Fold-contract record count drifted.")
    identities = _identity_lookup(metadata, contract, fold_records)
    artifacts = _rebuild_folds(dataset, targets, metadata, fold_records)

    candidates = _read_csv(root / "candidate_search_results.csv")
    selected = _read_csv(root / "selected_hyperparameters.csv")
    oof = _read_csv(root / "oof_predictions.csv")
    calibration_training = _read_csv(root / "calibration_training_oof.csv")
    calibrator_parameters = _read_csv(root / "calibrator_parameters.csv")
    _validate_candidate_selection(candidates, selected, contract, identities)
    _validate_oof_and_replay_baselines(oof, targets, labels_by_formulation, identities, artifacts, selected)
    max_calibration_error = _validate_calibration_replay(
        calibration_training,
        calibrator_parameters,
        oof,
        labels_by_formulation,
        identities,
        artifacts,
        selected,
    )

    expected_fold, expected_repetition, expected_variability, expected_class, expected_confusion = _recompute_metric_tables(
        oof, labels_by_formulation, identities
    )
    observed_fold = _read_csv(root / "fold_metrics.csv")
    observed_repetition = _read_csv(root / "repetition_metrics.csv")
    observed_variability = _read_csv(root / "variability_summary.csv")
    observed_class = _read_csv(root / "per_class_metrics.csv")
    observed_confusion = _read_csv(root / "confusion_matrices.csv")
    _validate_identity(observed_fold, identities, context="Fold metrics")
    _assert_frame_equal(observed_fold, expected_fold, sort_columns=("formulation_id", "repetition", "outer_fold", "system"), context="Fold metrics")
    _assert_frame_equal(observed_repetition, expected_repetition, sort_columns=("formulation_id", "repetition", "system", "metric"), context="Repetition metrics")
    _assert_frame_equal(observed_variability, expected_variability, sort_columns=("formulation_id", "system", "metric"), context="Variability summary")
    _assert_frame_equal(observed_class, expected_class, sort_columns=("formulation_id", "repetition", "system", "class_label"), context="Per-class metrics")
    _assert_frame_equal(observed_confusion, expected_confusion, sort_columns=("formulation_id", "repetition", "system", "true_label", "predicted_label"), context="Confusion matrices")

    expected_baseline = _baseline_comparisons(expected_repetition)
    _assert_frame_equal(_read_csv(root / "baseline_comparisons.csv"), expected_baseline, sort_columns=("formulation_id", "repetition", "metric", "comparison"), context="Baseline comparisons")
    v2_class, v2_confusion, v2_metrics = _canonical_v2_results(contract)
    _assert_frame_equal(_read_csv(root / "canonical_v2_per_class_metrics.csv"), v2_class, sort_columns=("system", "class_label"), context="Canonical-v2 per-class metrics")
    _assert_frame_equal(_read_csv(root / "canonical_v2_confusion_matrix.csv"), v2_confusion, sort_columns=("system", "true_label", "predicted_label"), context="Canonical-v2 confusion matrices")
    _assert_frame_equal(_read_csv(root / "cv_design_sensitivity.csv"), _cv_design(expected_variability, v2_metrics), sort_columns=("system", "metric"), context="CV-design sensitivity")
    _assert_frame_equal(_read_csv(root / "target_mapping_support.csv"), _target_support(dataset, contract), sort_columns=("formulation_id", "raw_target_value"), context="Target-mapping support")
    _assert_frame_equal(_read_csv(root / "protocol_comparison.csv"), _protocol_comparison(), sort_columns=("component",), context="Protocol comparison")

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for formulation_id in FORMULATIONS:
        summary[formulation_id] = {}
        for system in ("xgboost_raw", "xgboost_sigmoid"):
            scoped = expected_variability.loc[
                (expected_variability["formulation_id"] == formulation_id)
                & (expected_variability["system"] == system)
                & (expected_variability["metric"].isin(PRIORITY_METRICS))
            ]
            summary[formulation_id][system] = {
                str(row.metric): float(row.mean) for row in scoped.itertuples(index=False)
            }
    return {
        "status": "passed",
        "run_id": metadata["run_id"],
        "generation_commit": metadata["git_identity"]["commit"],
        "contract_sha256": metadata["contract_sha256"],
        "scientific_input_sha256": metadata["scientific_input_sha256"],
        "file_count": len(EXPECTED_FILES),
        "formulations": 2,
        "repetitions_per_formulation": 5,
        "distinct_outer_assignment_count": 10,
        "candidate_search_row_count": len(candidates),
        "oof_prediction_row_count": len(oof),
        "calibration_training_oof_row_count": len(calibration_training),
        "calibrator_parameter_row_count": len(calibrator_parameters),
        "fold_metric_row_count": len(observed_fold),
        "repetition_metric_row_count": len(observed_repetition),
        "per_class_metric_row_count": len(observed_class),
        "confusion_matrix_row_count": len(observed_confusion),
        "maximum_sigmoid_probability_replay_error": max_calibration_error,
        "generation_implementation_blobs_verified": len(EXPECTED_IMPLEMENTATIONS),
        "generation_tracked_source_blobs_verified": tracked_sources,
        "generation_local_hash_bound_sources_verified": local_sources,
        "source_tree_receipt_internally_consistent": True,
        "summary_priority_metric_means": summary,
        "cross_formulation_metric_difference_computed": False,
        "locked_model_transport": False,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(json.dumps(validate_hrdataset_sensitivity_run_v3(args.run_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_HRDATASET_SENSITIVITY_RUN",
    "EXPECTED_FILES",
    "HRDatasetSensitivityRunValidationV3Error",
    "validate_hrdataset_sensitivity_run_v3",
]
