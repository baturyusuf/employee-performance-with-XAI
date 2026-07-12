"""Deterministic, hash-bound fold assignments for nested manuscript evaluation.

This module owns split construction only.  It never fits a preprocessor or a
model.  All model, feature-policy, calibration, SHAP, and subgroup consumers
must read the same persisted assignment instead of constructing a splitter.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


SCHEMA_VERSION = 1
OUTER_ASSIGNMENT_FILENAME = "fold_assignments.csv"
INNER_ASSIGNMENT_FILENAME = "inner_fold_assignments.csv"
CONTRACT_FILENAME = "fold_contract.json"
SHA256_PATTERN_LENGTH = 64

OUTER_COLUMNS = (
    "run_id",
    "config_hash",
    "scientific_input_hash",
    "dataset_key",
    "dataset_sha256",
    "fold_contract_hash",
    "sample_index",
    "sample_key_sha256",
    "y_true",
    "outer_fold",
)
INNER_COLUMNS = (
    "run_id",
    "config_hash",
    "scientific_input_hash",
    "dataset_key",
    "dataset_sha256",
    "fold_contract_hash",
    "outer_fold",
    "sample_index",
    "sample_key_sha256",
    "y_true",
    "inner_fold",
    "inner_seed",
)


class SharedFoldContractError(RuntimeError):
    """Raised when split construction or persisted fold evidence is invalid."""


@dataclass(frozen=True)
class SharedFoldArtifacts:
    outer_assignments: pd.DataFrame
    inner_assignments: pd.DataFrame
    contract: Mapping[str, Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _frame_csv_bytes(frame: pd.DataFrame, columns: Sequence[str]) -> bytes:
    return frame.loc[:, list(columns)].to_csv(index=False, lineterminator="\n").encode("utf-8")


def _normalise_scalar(value: Any) -> Mapping[str, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        raise SharedFoldContractError("Identifier values must be non-null.")
    if isinstance(value, (str, int, float, bool)):
        return {"type": type(value).__name__, "value": value}
    return {"type": type(value).__name__, "value": str(value)}


def _sample_key(dataset_key: str, id_column: str, value: Any) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "dataset_key": dataset_key,
                "id_column": id_column,
                "identifier": _normalise_scalar(value),
            }
        )
    )


def _validate_identity(name: str, value: str, *, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SharedFoldContractError(f"{name} must be a non-empty string.")
    if sha256 and (len(value) != SHA256_PATTERN_LENGTH or any(c not in "0123456789abcdef" for c in value)):
        raise SharedFoldContractError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _validate_source_frame(
    frame: pd.DataFrame,
    *,
    target_column: str,
    id_column: str,
    outer_splits: int,
    inner_splits: int,
) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise SharedFoldContractError("Fold construction requires a non-empty dataframe.")
    missing = sorted({target_column, id_column}.difference(frame.columns))
    if missing:
        raise SharedFoldContractError(f"Fold source is missing columns: {missing}.")
    if not frame.index.is_unique:
        raise SharedFoldContractError("Sample positions/index values must be unique.")
    if not all(isinstance(value, (int, np.integer)) for value in frame.index):
        raise SharedFoldContractError("Sample positions/index values must be integers.")
    if frame[id_column].isna().any() or frame[id_column].duplicated().any():
        raise SharedFoldContractError("Identifier values must be non-null and unique.")
    if frame[target_column].isna().any():
        raise SharedFoldContractError("Target values must be non-null.")
    if outer_splits < 2 or inner_splits < 2:
        raise SharedFoldContractError("Outer and inner split counts must each be at least two.")
    class_counts = frame[target_column].value_counts()
    if len(class_counts) < 2:
        raise SharedFoldContractError("Stratified folds require at least two target classes.")
    if int(class_counts.min()) < outer_splits:
        raise SharedFoldContractError(
            "Every target class must contain at least outer_splits observations; "
            f"minimum={int(class_counts.min())}, outer_splits={outer_splits}."
        )


def _derive_inner_seed(seed: int, outer_fold: int) -> int:
    digest = hashlib.sha256(f"{int(seed)}:outer:{int(outer_fold)}".encode("ascii")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _fold_identity_payload(
    outer: pd.DataFrame,
    *,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    dataset_key: str,
    dataset_sha256: str,
    target_column: str,
    id_column: str,
    outer_splits: int,
    inner_splits: int,
    outer_seed: int,
    inner_seed: int,
) -> Mapping[str, Any]:
    samples = (
        outer[["sample_index", "sample_key_sha256", "y_true"]]
        .sort_values("sample_index")
        .to_dict(orient="records")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "config_hash": config_hash,
        "scientific_input_hash": scientific_input_hash,
        "dataset_key": dataset_key,
        "dataset_sha256": dataset_sha256,
        "target_column": target_column,
        "id_column": id_column,
        "outer_splits": int(outer_splits),
        "inner_splits": int(inner_splits),
        "shuffle": True,
        "outer_seed": int(outer_seed),
        "inner_seed": int(inner_seed),
        "samples": samples,
    }


def generate_inner_assignments(
    outer_assignments: pd.DataFrame,
    *,
    inner_splits: int,
    inner_seed: int,
    fold_contract_hash: str,
) -> pd.DataFrame:
    """Assign one inner validation fold to every outer-training observation."""

    required = {
        "run_id",
        "config_hash",
        "scientific_input_hash",
        "dataset_key",
        "dataset_sha256",
        "sample_index",
        "sample_key_sha256",
        "y_true",
        "outer_fold",
    }
    missing = sorted(required.difference(outer_assignments.columns))
    if missing:
        raise SharedFoldContractError(f"Outer assignments lack columns required for inner folds: {missing}.")
    outer_splits = sorted(pd.to_numeric(outer_assignments["outer_fold"], errors="raise").astype(int).unique())
    if outer_splits != list(range(1, len(outer_splits) + 1)):
        raise SharedFoldContractError("Outer fold labels must be contiguous one-based integers.")

    rows: list[dict[str, Any]] = []
    for outer_fold in outer_splits:
        training = outer_assignments[outer_assignments["outer_fold"] != outer_fold].copy()
        training = training.sort_values("sample_key_sha256").reset_index(drop=True)
        counts = training["y_true"].value_counts()
        if int(counts.min()) < inner_splits:
            raise SharedFoldContractError(
                f"Outer fold {outer_fold} has insufficient class support for {inner_splits} inner folds."
            )
        derived_inner_seed = _derive_inner_seed(inner_seed, outer_fold)
        splitter = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=derived_inner_seed,
        )
        assigned = np.zeros(len(training), dtype=int)
        for inner_fold, (_, validation_positions) in enumerate(
            splitter.split(np.zeros(len(training)), training["y_true"]),
            start=1,
        ):
            assigned[validation_positions] = inner_fold
        for position, record in training.iterrows():
            rows.append(
                {
                    "run_id": record["run_id"],
                    "config_hash": record["config_hash"],
                    "scientific_input_hash": record["scientific_input_hash"],
                    "dataset_key": record["dataset_key"],
                    "dataset_sha256": record["dataset_sha256"],
                    "fold_contract_hash": fold_contract_hash,
                    "outer_fold": int(outer_fold),
                    "sample_index": int(record["sample_index"]),
                    "sample_key_sha256": record["sample_key_sha256"],
                    "y_true": record["y_true"],
                    "inner_fold": int(assigned[position]),
                    "inner_seed": int(derived_inner_seed),
                }
            )
    return pd.DataFrame(rows, columns=INNER_COLUMNS).sort_values(
        ["outer_fold", "sample_index"]
    ).reset_index(drop=True)


def _class_distribution(series: pd.Series) -> Mapping[str, int]:
    counts = series.value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _fold_summaries(outer: pd.DataFrame, inner: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outer_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    for outer_fold in sorted(outer["outer_fold"].unique()):
        test = outer[outer["outer_fold"] == outer_fold]
        train = outer[outer["outer_fold"] != outer_fold]
        outer_rows.append(
            {
                "outer_fold": int(outer_fold),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "train_target_distribution": _class_distribution(train["y_true"]),
                "test_target_distribution": _class_distribution(test["y_true"]),
            }
        )
        scoped_inner = inner[inner["outer_fold"] == outer_fold]
        for inner_fold in sorted(scoped_inner["inner_fold"].unique()):
            validation = scoped_inner[scoped_inner["inner_fold"] == inner_fold]
            development = scoped_inner[scoped_inner["inner_fold"] != inner_fold]
            inner_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "n_inner_train": int(len(development)),
                    "n_inner_validation": int(len(validation)),
                    "validation_target_distribution": _class_distribution(validation["y_true"]),
                }
            )
    return outer_rows, inner_rows


def generate_shared_folds(
    frame: pd.DataFrame,
    *,
    target_column: str,
    id_column: str,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    dataset_key: str,
    dataset_sha256: str,
    outer_splits: int = 10,
    inner_splits: int = 3,
    seed: int = 42,
    inner_seed: int | None = None,
) -> SharedFoldArtifacts:
    """Create row-order-stable outer and nested-inner stratified assignments."""

    _validate_identity("run_id", run_id)
    _validate_identity("dataset_key", dataset_key)
    _validate_identity("config_hash", config_hash, sha256=True)
    _validate_identity("scientific_input_hash", scientific_input_hash, sha256=True)
    _validate_identity("dataset_sha256", dataset_sha256, sha256=True)
    _validate_source_frame(
        frame,
        target_column=target_column,
        id_column=id_column,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
    )

    resolved_inner_seed = int(seed if inner_seed is None else inner_seed)
    samples = pd.DataFrame(
        {
            "sample_index": [int(value) for value in frame.index],
            "sample_key_sha256": [
                _sample_key(dataset_key, id_column, value) for value in frame[id_column]
            ],
            "y_true": frame[target_column].to_numpy(),
        }
    ).sort_values("sample_key_sha256").reset_index(drop=True)
    if samples["sample_key_sha256"].duplicated().any():
        raise SharedFoldContractError("Hashed sample identities must be unique.")
    splitter = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=int(seed))
    assigned = np.zeros(len(samples), dtype=int)
    for outer_fold, (_, test_positions) in enumerate(
        splitter.split(np.zeros(len(samples)), samples["y_true"]),
        start=1,
    ):
        assigned[test_positions] = outer_fold
    samples["outer_fold"] = assigned
    outer = samples.assign(
        run_id=run_id,
        config_hash=config_hash,
        scientific_input_hash=scientific_input_hash,
        dataset_key=dataset_key,
        dataset_sha256=dataset_sha256,
    )
    identity_payload = _fold_identity_payload(
        outer,
        run_id=run_id,
        config_hash=config_hash,
        scientific_input_hash=scientific_input_hash,
        dataset_key=dataset_key,
        dataset_sha256=dataset_sha256,
        target_column=target_column,
        id_column=id_column,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        outer_seed=seed,
        inner_seed=resolved_inner_seed,
    )
    fold_contract_hash = _sha256_bytes(_canonical_json(identity_payload))
    outer["fold_contract_hash"] = fold_contract_hash
    outer = outer.loc[:, list(OUTER_COLUMNS)].sort_values("sample_index").reset_index(drop=True)
    inner = generate_inner_assignments(
        outer,
        inner_splits=inner_splits,
        inner_seed=resolved_inner_seed,
        fold_contract_hash=fold_contract_hash,
    )
    outer_summaries, inner_summaries = _fold_summaries(outer, inner)
    contract: dict[str, Any] = {
        **identity_payload,
        "fold_contract_hash": fold_contract_hash,
        "outer_assignment_file": OUTER_ASSIGNMENT_FILENAME,
        "outer_assignment_sha256": _sha256_bytes(_frame_csv_bytes(outer, OUTER_COLUMNS)),
        "inner_assignment_file": INNER_ASSIGNMENT_FILENAME,
        "inner_assignment_sha256": _sha256_bytes(_frame_csv_bytes(inner, INNER_COLUMNS)),
        "n_rows": int(len(outer)),
        "target_labels": [value.item() if isinstance(value, np.generic) else value for value in sorted(outer["y_true"].unique())],
        "target_distribution": _class_distribution(outer["y_true"]),
        "sample_identity": "sha256(dataset_key, id_column, typed identifier); raw identifiers are not persisted",
        "model_feature_exclusions": [id_column, target_column],
        "identifier_used_for_model": False,
        "target_used_for_model": False,
        "outer_fold_summaries": outer_summaries,
        "inner_fold_summaries": inner_summaries,
        "invariants": {
            "unique_sample_positions": True,
            "unique_identifiers": True,
            "outer_test_exactly_once": True,
            "outer_train_test_disjoint": True,
            "inner_is_strict_outer_train_subset": True,
            "outer_test_absent_from_inner": True,
            "stratified": True,
        },
    }
    artifacts = SharedFoldArtifacts(outer, inner, contract)
    validate_shared_folds(artifacts, source_frame=frame)
    return artifacts


def _require_columns(frame: pd.DataFrame, expected: Sequence[str], name: str) -> None:
    if list(frame.columns) != list(expected):
        raise SharedFoldContractError(
            f"{name} schema mismatch; expected={list(expected)}, observed={list(frame.columns)}."
        )


def _assert_stratified(frame: pd.DataFrame, fold_column: str, context: str) -> None:
    counts = frame.groupby([fold_column, "y_true"]).size().unstack(fill_value=0)
    if not counts.empty and bool(((counts.max(axis=0) - counts.min(axis=0)) > 1).any()):
        raise SharedFoldContractError(f"{context} is not class-stratified within one observation.")


def validate_shared_folds(
    artifacts: SharedFoldArtifacts,
    *,
    source_frame: pd.DataFrame | None = None,
) -> None:
    """Fail closed on any schema, membership, hash, or split-isolation defect."""

    outer = artifacts.outer_assignments
    inner = artifacts.inner_assignments
    contract = dict(artifacts.contract)
    _require_columns(outer, OUTER_COLUMNS, "Outer assignment")
    _require_columns(inner, INNER_COLUMNS, "Inner assignment")
    required_contract = {
        "schema_version",
        "run_id",
        "config_hash",
        "scientific_input_hash",
        "dataset_key",
        "dataset_sha256",
        "target_column",
        "id_column",
        "outer_splits",
        "inner_splits",
        "outer_seed",
        "inner_seed",
        "fold_contract_hash",
        "outer_assignment_sha256",
        "inner_assignment_sha256",
        "n_rows",
    }
    missing = sorted(required_contract.difference(contract))
    if missing:
        raise SharedFoldContractError(f"Fold contract is missing fields: {missing}.")
    if contract["schema_version"] != SCHEMA_VERSION:
        raise SharedFoldContractError("Unsupported fold contract schema version.")
    if outer.empty or len(outer) != int(contract["n_rows"]):
        raise SharedFoldContractError("Outer assignments do not match contract row count.")
    if outer["sample_index"].duplicated().any() or outer["sample_key_sha256"].duplicated().any():
        raise SharedFoldContractError("Every sample must occur exactly once in outer-test assignment evidence.")
    outer_splits = int(contract["outer_splits"])
    inner_splits = int(contract["inner_splits"])
    if sorted(outer["outer_fold"].astype(int).unique()) != list(range(1, outer_splits + 1)):
        raise SharedFoldContractError("Outer fold labels do not match the declared split count.")
    _assert_stratified(outer, "outer_fold", "Outer assignments")

    identity_fields = (
        "run_id",
        "config_hash",
        "scientific_input_hash",
        "dataset_key",
        "dataset_sha256",
        "fold_contract_hash",
    )
    for field in identity_fields:
        expected = contract[field]
        if set(outer[field].astype(str)) != {str(expected)} or set(inner[field].astype(str)) != {str(expected)}:
            raise SharedFoldContractError(f"Assignment identity mismatch for {field}.")

    if contract["outer_assignment_sha256"] != _sha256_bytes(_frame_csv_bytes(outer, OUTER_COLUMNS)):
        raise SharedFoldContractError("Outer assignment hash mismatch.")
    if contract["inner_assignment_sha256"] != _sha256_bytes(_frame_csv_bytes(inner, INNER_COLUMNS)):
        raise SharedFoldContractError("Inner assignment hash mismatch.")
    identity_payload = _fold_identity_payload(
        outer,
        run_id=str(contract["run_id"]),
        config_hash=str(contract["config_hash"]),
        scientific_input_hash=str(contract["scientific_input_hash"]),
        dataset_key=str(contract["dataset_key"]),
        dataset_sha256=str(contract["dataset_sha256"]),
        target_column=str(contract["target_column"]),
        id_column=str(contract["id_column"]),
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        outer_seed=int(contract["outer_seed"]),
        inner_seed=int(contract["inner_seed"]),
    )
    if _sha256_bytes(_canonical_json(identity_payload)) != contract["fold_contract_hash"]:
        raise SharedFoldContractError("Fold contract hash does not bind its scientific inputs and samples.")
    for field, expected_value in identity_payload.items():
        if contract.get(field) != expected_value:
            raise SharedFoldContractError(f"Fold contract field {field!r} does not match its assignment evidence.")

    replay_outer = outer[["sample_index", "sample_key_sha256", "y_true"]].sort_values(
        "sample_key_sha256"
    ).reset_index(drop=True)
    replay_fold = np.zeros(len(replay_outer), dtype=int)
    replay_splitter = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=int(contract["outer_seed"]),
    )
    for fold, (_, test_positions) in enumerate(
        replay_splitter.split(np.zeros(len(replay_outer)), replay_outer["y_true"]),
        start=1,
    ):
        replay_fold[test_positions] = fold
    replay_mapping = pd.Series(
        replay_fold,
        index=replay_outer["sample_index"].astype(int),
    ).sort_index()
    recorded_mapping = outer.set_index("sample_index")["outer_fold"].astype(int).sort_index()
    if not replay_mapping.equals(recorded_mapping):
        raise SharedFoldContractError("Outer assignments cannot be reproduced from the declared seed.")

    replay_inner = generate_inner_assignments(
        outer,
        inner_splits=inner_splits,
        inner_seed=int(contract["inner_seed"]),
        fold_contract_hash=str(contract["fold_contract_hash"]),
    )
    try:
        pd.testing.assert_frame_equal(inner.reset_index(drop=True), replay_inner)
    except AssertionError as exc:
        raise SharedFoldContractError(
            "Inner assignments cannot be reproduced from the declared inner seed."
        ) from exc

    expected_samples = set(outer["sample_index"].astype(int))
    for outer_fold in range(1, outer_splits + 1):
        outer_test = set(outer.loc[outer["outer_fold"] == outer_fold, "sample_index"].astype(int))
        expected_train = expected_samples - outer_test
        scoped = inner[inner["outer_fold"] == outer_fold]
        observed_train = set(scoped["sample_index"].astype(int))
        if observed_train != expected_train or bool(outer_test.intersection(observed_train)):
            raise SharedFoldContractError(
                f"Inner assignments for outer fold {outer_fold} are not the strict outer-training set."
            )
        if scoped["sample_index"].duplicated().any():
            raise SharedFoldContractError(
                f"Outer-training samples must occur exactly once in inner assignment evidence for outer fold {outer_fold}."
            )
        if sorted(scoped["inner_fold"].astype(int).unique()) != list(range(1, inner_splits + 1)):
            raise SharedFoldContractError(f"Inner fold labels are incomplete for outer fold {outer_fold}.")
        _assert_stratified(scoped, "inner_fold", f"Inner assignments for outer fold {outer_fold}")

    expected_outer_summaries, expected_inner_summaries = _fold_summaries(outer, inner)
    if contract.get("outer_fold_summaries") != expected_outer_summaries:
        raise SharedFoldContractError("Outer fold summaries do not match assignment evidence.")
    if contract.get("inner_fold_summaries") != expected_inner_summaries:
        raise SharedFoldContractError("Inner fold summaries do not match assignment evidence.")
    if contract.get("target_distribution") != _class_distribution(outer["y_true"]):
        raise SharedFoldContractError("Target distribution does not match assignment evidence.")
    if contract.get("outer_assignment_file") != OUTER_ASSIGNMENT_FILENAME:
        raise SharedFoldContractError("Unexpected outer assignment filename in fold contract.")
    if contract.get("inner_assignment_file") != INNER_ASSIGNMENT_FILENAME:
        raise SharedFoldContractError("Unexpected inner assignment filename in fold contract.")
    invariants = contract.get("invariants")
    if not isinstance(invariants, Mapping) or not invariants or not all(value is True for value in invariants.values()):
        raise SharedFoldContractError("Fold contract invariants must all be explicitly true.")

    if source_frame is not None:
        target_column = str(contract["target_column"])
        id_column = str(contract["id_column"])
        _validate_source_frame(
            source_frame,
            target_column=target_column,
            id_column=id_column,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
        )
        if set(int(value) for value in source_frame.index) != expected_samples:
            raise SharedFoldContractError("Source sample positions differ from the persisted assignment.")
        lookup = outer.set_index("sample_index")
        for sample_index, row in source_frame.iterrows():
            if lookup.loc[int(sample_index), "sample_key_sha256"] != _sample_key(
                str(contract["dataset_key"]), id_column, row[id_column]
            ):
                raise SharedFoldContractError("Source identifier does not match persisted sample identity.")
            if lookup.loc[int(sample_index), "y_true"] != row[target_column]:
                raise SharedFoldContractError("Source target does not match persisted fold evidence.")


def validate_consumer_fold_assignments(
    artifacts: SharedFoldArtifacts,
    consumer_rows: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> None:
    """Require every model/policy consumer to reproduce the exact outer mapping."""

    required = {"sample_index", "outer_fold", "fold_contract_hash", *group_columns}
    missing = sorted(required.difference(consumer_rows.columns))
    if missing:
        raise SharedFoldContractError(f"Consumer fold evidence is missing columns: {missing}.")
    expected = artifacts.outer_assignments.set_index("sample_index")["outer_fold"].astype(int)
    contract_hash = str(artifacts.contract["fold_contract_hash"])
    if consumer_rows.empty:
        raise SharedFoldContractError("Consumer fold evidence must not be empty.")
    grouped = consumer_rows.groupby(list(group_columns), dropna=False, sort=False)
    for group_key, group in grouped:
        label = group_key if isinstance(group_key, tuple) else (group_key,)
        if group["sample_index"].duplicated().any() or set(group["sample_index"].astype(int)) != set(expected.index):
            raise SharedFoldContractError(f"Consumer group {label} lacks exactly-once OOF coverage.")
        if set(group["fold_contract_hash"].astype(str)) != {contract_hash}:
            raise SharedFoldContractError(f"Consumer group {label} has a fold-contract identity mismatch.")
        observed = group.set_index("sample_index")["outer_fold"].astype(int).sort_index()
        if not observed.equals(expected.sort_index()):
            raise SharedFoldContractError(f"Consumer group {label} does not use the shared outer folds.")
        if "y_true" in group:
            expected_y = artifacts.outer_assignments.set_index("sample_index")["y_true"].sort_index()
            observed_y = group.set_index("sample_index")["y_true"].sort_index()
            if not observed_y.equals(expected_y):
                raise SharedFoldContractError(f"Consumer group {label} target identity mismatch.")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def write_shared_folds(artifacts: SharedFoldArtifacts, output_dir: str | Path) -> Mapping[str, Path]:
    validate_shared_folds(artifacts)
    output = Path(output_dir)
    paths = {
        "outer_assignments": output / OUTER_ASSIGNMENT_FILENAME,
        "inner_assignments": output / INNER_ASSIGNMENT_FILENAME,
        "contract": output / CONTRACT_FILENAME,
    }
    _atomic_write(paths["outer_assignments"], _frame_csv_bytes(artifacts.outer_assignments, OUTER_COLUMNS))
    _atomic_write(paths["inner_assignments"], _frame_csv_bytes(artifacts.inner_assignments, INNER_COLUMNS))
    _atomic_write(
        paths["contract"],
        json.dumps(artifacts.contract, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n",
    )
    return paths


def read_shared_folds(output_dir: str | Path) -> SharedFoldArtifacts:
    output = Path(output_dir)
    paths = {
        "outer": output / OUTER_ASSIGNMENT_FILENAME,
        "inner": output / INNER_ASSIGNMENT_FILENAME,
        "contract": output / CONTRACT_FILENAME,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SharedFoldContractError(f"Shared-fold artifacts are missing: {missing}.")
    try:
        outer = pd.read_csv(paths["outer"])
        inner = pd.read_csv(paths["inner"])
        contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SharedFoldContractError(f"Shared-fold artifacts cannot be read: {exc}") from exc
    artifacts = SharedFoldArtifacts(outer, inner, contract)
    validate_shared_folds(artifacts)
    return artifacts


__all__ = [
    "CONTRACT_FILENAME",
    "INNER_ASSIGNMENT_FILENAME",
    "INNER_COLUMNS",
    "OUTER_ASSIGNMENT_FILENAME",
    "OUTER_COLUMNS",
    "SharedFoldArtifacts",
    "SharedFoldContractError",
    "generate_inner_assignments",
    "generate_shared_folds",
    "read_shared_folds",
    "validate_consumer_fold_assignments",
    "validate_shared_folds",
    "write_shared_folds",
]
