"""Independent acceptance validator for the preserved Unit 2G HRDataset stage.

The validator never fits a model and never upgrades a stage-validation run to a
canonical package.  It verifies the completed atomic stage against its saved
generation commit/config/input identity, replays all persisted outer models and
sigmoid calibrators, recomputes exact-fold grouped SHAP for the conservative
primary model, and checks the publication/claim boundaries required for Unit 2G.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from src.data.canonical_loader import load_canonical_dataset
from src.data.external_adapters import load_external_dataset
from src.experiments import hrdataset_replication_core as replication_core
from src.experiments import manuscript_hrdataset_replication as replication_stage
from src.experiments.hrdataset_replication_diagnostics import (
    FoldModelReference,
    ReplicationIdentity,
    compute_exact_oof_grouped_shap,
    feature_policy_contract_sha256,
    model_set_sha256,
)
from src.experiments.manuscript_calibration import (
    SigmoidCalibrator,
    SigmoidClassParameters,
    apply_sigmoid_calibrator,
)
from src.governance.manuscript_contract import (
    canonical_config_hash,
    evidence_scope_contract_hash,
    manuscript_settings,
    scientific_input_hash,
)
from src.models.canonical_models import aligned_predict_proba
from src.models.oof_bootstrap import BootstrapProtocol, generate_stratified_resample_indices
from src.utils.config_loader import PROJECT_ROOT


class Unit2GValidationError(RuntimeError):
    """Raised when preserved Unit 2G evidence fails an acceptance check."""


LABELS = (2, 3, 4)
PRIMARY_POLICY = "conservative_primary"
EXPECTED_PRIMARY_FEATURES = (
    "EmpJobRole",
    "EngagementSurvey",
    "EmpJobSatisfaction",
    "SpecialProjectsCount",
    "DaysLateLast30",
    "Absences",
    "ExperienceYearsAtThisCompany",
)
EXPECTED_POLICY_ORDER = (
    "conservative_primary",
    "department_including_audit",
    "job_role_free_audit",
    "proxy_rich_audit",
    "temporality_restricted_audit",
)
REQUIRED_FIGURE_TABLE_SOURCES = (
    "target_support.csv",
    "raw_metric_intervals.csv",
    "calibration_metric_intervals.csv",
    "calibration_paired_differences.csv",
    "policy_pairwise_differences.csv",
    "external_replication_metadata.json",
)
PROBABILITY_COLUMNS = tuple(f"prob_class_{label}" for label in LABELS)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40,64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Unit2GValidationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Unit2GValidationError(f"Cannot read required JSON {path}: {exc}") from exc


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, float_precision="round_trip")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise Unit2GValidationError(f"Cannot read required CSV {path}: {exc}") from exc


def _git(project_root: Path, *arguments: str, text: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=text,
    )
    if completed.returncode != 0:
        stderr = completed.stderr if text else completed.stderr.decode("utf-8", errors="replace")
        raise Unit2GValidationError(
            f"Local Git command failed: git {' '.join(arguments)}: {stderr.strip()}"
        )
    return completed.stdout


def _git_file(project_root: Path, commit: str, relative_path: str) -> bytes:
    payload = _git(project_root, "show", f"{commit}:{relative_path}")
    assert isinstance(payload, bytes)
    return payload


def _git_worktree_file(project_root: Path, commit: str, relative_path: str) -> bytes:
    payload = _git(
        project_root,
        "cat-file",
        "--filters",
        f"--path={relative_path}",
        f"{commit}:{relative_path}",
    )
    assert isinstance(payload, bytes)
    return payload


def _source_tree_hash_at_commit(project_root: Path, commit: str) -> str:
    listing = _git(project_root, "ls-tree", "-r", "--name-only", commit, text=True)
    assert isinstance(listing, str)
    candidates = sorted(
        path
        for path in listing.splitlines()
        if (
            path.startswith("src/")
            or path.startswith("configs/")
            or path in {"requirements.txt", "requirements-dev.txt"}
        )
        and "/__pycache__/" not in path
        and not path.endswith(".pyc")
    )
    _require(bool(candidates), "Generation commit contains no source-tree candidates.")
    digest = hashlib.sha256()
    for relative in candidates:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(_git_file(project_root, commit, relative))
    return digest.hexdigest()


def canonical_csv_content_sha256(path: Path) -> tuple[str, int, int]:
    """Hash parsed CSV cells independently of byte-level quoting and newlines."""

    digest = hashlib.sha256()
    row_count = 0
    width: int | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if width is None:
                width = len(row)
            _require(len(row) == width, f"CSV row width drift in {path} at row {row_count + 1}.")
            digest.update(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            digest.update(b"\n")
            row_count += 1
    _require(width is not None and row_count > 1, f"CSV content is empty: {path}")
    return digest.hexdigest(), row_count - 1, int(width)


def _safe_stage_path(stage_root: Path, raw_path: Any) -> Path:
    value = str(raw_path)
    _require(value == value.strip() and value, f"Unsafe empty/whitespace artifact path: {value!r}")
    _require("\\" not in value, f"Artifact path is not POSIX portable: {value}")
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute() and ".." not in relative.parts and "." not in relative.parts,
        f"Artifact path escapes the stage: {value}",
    )
    candidate = (stage_root / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(stage_root.resolve())
    except ValueError as exc:
        raise Unit2GValidationError(f"Artifact path escapes the stage: {value}") from exc
    return candidate


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    _require(column in frame.columns, f"Required boolean column is absent: {column}")
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.casefold()
    _require(set(normalized).issubset({"true", "false"}), f"Invalid boolean values in {column}.")
    return normalized.eq("true")


def _assert_identity_columns(
    frame: pd.DataFrame,
    *,
    run_id: str,
    config_hash: str,
    scientific_input_hash_value: str,
) -> None:
    expected = {
        "run_id": run_id,
        "config_hash": config_hash,
        "scientific_input_hash": scientific_input_hash_value,
    }
    for column, value in expected.items():
        _require(column in frame.columns, f"Identity column {column} is absent.")
        _require(set(frame[column].astype(str)) == {value}, f"Identity drift in column {column}.")


def _collect_feature_values(value: Any) -> set[str]:
    observed: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key == "feature" and isinstance(nested, str):
                observed.add(nested)
            observed.update(_collect_feature_values(nested))
    elif isinstance(value, list):
        for nested in value:
            observed.update(_collect_feature_values(nested))
    return observed


def _validate_inventory(stage_root: Path, contract: Mapping[str, Any]) -> tuple[int, int]:
    _require(contract.get("status") == "complete", "Atomic stage contract is not complete.")
    _require(contract.get("inventory_mode") == "closed_world", "Stage inventory is not closed-world.")
    _require(contract.get("path_basis") == "stage_relative", "Stage paths are not stage-relative.")
    _require(math.isfinite(float(contract.get("elapsed_seconds", math.nan))), "Stage runtime is invalid.")
    _require(bool(contract.get("started_at")) and bool(contract.get("ended_at")), "Stage timing is incomplete.")

    output_rows = list(contract.get("outputs", ()))
    _require(len(output_rows) == 124, f"Expected 124 stage outputs, observed {len(output_rows)}.")
    output_paths: set[str] = set()
    for row in output_rows:
        relative = str(row.get("path", ""))
        _require(relative not in output_paths, f"Duplicate stage output path: {relative}")
        output_paths.add(relative)
        path = _safe_stage_path(stage_root, relative)
        _require(path.is_file() and not path.is_symlink(), f"Stage output is missing/link-like: {relative}")
        _require(path.stat().st_size == int(row["size_bytes"]) > 0, f"Stage size mismatch: {relative}")
        _require(_sha256_file(path) == row["sha256"], f"Stage SHA-256 mismatch: {relative}")

    actual = {
        path.relative_to(stage_root).as_posix()
        for path in stage_root.rglob("*")
        if path.is_file()
    }
    _require(
        actual == output_paths | {"stage_contract.json"},
        f"Closed-world stage inventory differs: missing={sorted(output_paths - actual)}, "
        f"extra={sorted(actual - output_paths - {'stage_contract.json'})}",
    )
    _require(not any(path.is_symlink() for path in stage_root.rglob("*")), "Stage contains a symlink.")
    return len(output_rows), len(actual)


def _validate_artifact_manifest(
    stage_root: Path,
    *,
    run_id: str,
    config_hash: str,
    scientific_input_hash_value: str,
) -> int:
    payload = _read_json(stage_root / "artifact_manifest.json")
    csv_frame = _read_csv(stage_root / "artifact_manifest.csv")
    rows = list(payload.get("artifacts", ()))
    _require(payload.get("status") == "complete", "Artifact manifest is not complete.")
    _require(int(payload.get("n_artifacts", -1)) == len(rows) == len(csv_frame) == 122, "Artifact count mismatch.")
    json_by_path = {str(row["path"]): row for row in rows}
    _require(len(json_by_path) == len(rows), "Artifact JSON contains duplicate paths.")
    csv_by_path = {str(row.path): row for row in csv_frame.itertuples(index=False)}
    _require(set(json_by_path) == set(csv_by_path), "Artifact JSON/CSV path sets differ.")
    _assert_identity_columns(
        csv_frame,
        run_id=run_id,
        config_hash=config_hash,
        scientific_input_hash_value=scientific_input_hash_value,
    )
    for relative, row in json_by_path.items():
        path = _safe_stage_path(stage_root, relative)
        csv_row = csv_by_path[relative]
        _require(path.is_file() and path.stat().st_size > 0, f"Artifact is missing/empty: {relative}")
        observed_hash = _sha256_file(path)
        _require(observed_hash == row["sha256"] == csv_row.sha256, f"Artifact hash drift: {relative}")
        _require(path.stat().st_size == int(row["size_bytes"]) == int(csv_row.size_bytes), f"Artifact size drift: {relative}")
    excluded = set(payload.get("self_excluded_files", ()))
    _require(
        excluded == {"artifact_manifest.csv", "artifact_manifest.json", "stage_contract.json"},
        "Artifact-manifest self-exclusion contract drifted.",
    )
    return len(rows)


def _generation_config(project_root: Path, commit: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(
            _git_file(project_root, commit, "configs/manuscript_final.yaml").decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Unit2GValidationError(f"Generation config cannot be parsed: {exc}") from exc
    _require(isinstance(parsed, Mapping), "Generation config is not a mapping.")
    return parsed


def _validate_generation_identity(
    project_root: Path,
    stage_root: Path,
    contract: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str, str, str]:
    commit = str(contract.get("git_commit", ""))
    config_hash = str(contract.get("config_hash", ""))
    saved_scientific_hash = str(contract.get("scientific_input_hash", ""))
    source_hash = str(contract.get("source_tree_hash", ""))
    _require(bool(GIT_OBJECT_RE.fullmatch(commit)), "Invalid generation git_commit.")
    for name, value in {
        "config_hash": config_hash,
        "scientific_input_hash": saved_scientific_hash,
        "source_tree_hash": source_hash,
    }.items():
        _require(bool(SHA256_RE.fullmatch(value)), f"Invalid generation {name}.")

    _git(project_root, "cat-file", "-e", f"{commit}^{{commit}}")
    config = _generation_config(project_root, commit)
    _require(canonical_config_hash(config) == config_hash, "Generation config hash does not replay.")
    scope_hash = evidence_scope_contract_hash(config, "core")
    _require(scope_hash == contract.get("scope_contract_hash"), "Core scope hash does not replay.")
    # ``source_tree_hash`` binds the exact historical checkout bytes. Git's
    # text=auto normalization does not retain each file's former worktree EOL
    # representation, so a later checkout cannot reconstruct that byte stream.
    # The normalized commit tree is nevertheless traversed in full here, and
    # the exact recorded digest is required to agree across independent stage,
    # input-manifest, metadata, and late prepublication receipts below.
    normalized_source_tree_hash = _source_tree_hash_at_commit(project_root, commit)
    _require(bool(SHA256_RE.fullmatch(normalized_source_tree_hash)), "Normalized commit source tree is invalid.")

    side_inputs = dict(contract.get("side_input_hashes", {}))
    _require(len(side_inputs) == 6, f"Expected six scientific side inputs, observed {len(side_inputs)}.")
    for key, receipt in side_inputs.items():
        relative = str(receipt.get("path", ""))
        current = (project_root / relative).resolve()
        _require(current.is_file(), f"Scientific side input is missing: {key}/{relative}")
        generation_bytes = _git_file(project_root, commit, relative)
        _require(len(generation_bytes) == int(receipt["size_bytes"]), f"Generation side-input size drift: {key}")
        _require(_sha256_bytes(generation_bytes) == receipt["sha256"], f"Generation side-input hash drift: {key}")
        _require(_sha256_file(current) == receipt["sha256"], f"Current side-input hash drift: {key}")

    dataset_hashes = dict(contract.get("dataset_hashes", {}))
    _require(set(dataset_hashes) == {"hrdataset_v14", "inx_primary"}, "Dataset receipt set drifted.")
    for key, receipt in dataset_hashes.items():
        relative = str(receipt.get("path", ""))
        current = (project_root / relative).resolve()
        _require(current.is_file(), f"Dataset input is missing: {key}/{relative}")
        expected_hash = str(receipt.get("sha256", ""))
        _require(_sha256_file(current) == expected_hash, f"Current dataset byte hash drift: {key}")
        _require(
            _sha256_bytes(_git_worktree_file(project_root, commit, relative)) == expected_hash,
            f"Generation dataset hash drift: {key}",
        )

    recomputed_scientific = scientific_input_hash(
        config_hash=config_hash,
        scope_contract_hash=scope_hash,
        dataset_hashes=dataset_hashes,
        side_input_hashes=side_inputs,
    )
    _require(recomputed_scientific == saved_scientific_hash, "Scientific-input hash does not replay.")

    prepublication = _read_json(stage_root / "prepublication_input_validation.json")
    _require(
        prepublication.get("status") == "passed_before_atomic_publication",
        "Prepublication input validation did not pass.",
    )
    _require(prepublication.get("git_commit") == commit, "Prepublication commit drifted.")
    _require(prepublication.get("source_tree_hash") == source_hash, "Prepublication source tree drifted.")
    _require(prepublication.get("tracked_worktree_clean_at_publication") is True, "Publication worktree was not clean.")
    _require(
        prepublication.get("untracked_files_restricted_to_current_run_root") is True,
        "Publication untracked-file boundary was not enforced.",
    )
    metadata = _read_json(stage_root / "external_replication_metadata.json")
    provisional = _read_json(stage_root.parent / "stage_validation_input_manifest.json")
    _require(metadata.get("source_tree_hash") == source_hash, "Stage metadata source tree drifted.")
    _require(provisional.get("source_tree_hash") == source_hash, "Input-manifest source tree drifted.")
    return config, commit, config_hash, saved_scientific_hash


def _validate_data_and_policy(
    project_root: Path,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[Any, Mapping[str, pd.DataFrame], Mapping[str, str], Mapping[str, Sequence[str]], str, str]:
    config_path = project_root / "configs" / "manuscript_final.yaml"
    schema_path = project_root / "data" / "external" / "hrdataset_v14" / "schema_mapping.json"
    raw_path = project_root / "data" / "external" / "hrdataset_v14" / "raw.csv"
    raw_hr = load_canonical_dataset(config_path, "hrdataset_v14")
    expected_raw_hash = contract["dataset_hashes"]["hrdataset_v14"]["sha256"]
    _require(raw_hr.receipt["actual_sha256"] == expected_raw_hash, "Canonical loader raw receipt drifted.")
    content_hash, data_rows, data_columns = canonical_csv_content_sha256(raw_path)
    _require((data_rows, data_columns) == (311, 36), "Canonical CSV content dimensions drifted.")
    schema_hash = _sha256_file(schema_path)
    _require(
        schema_hash == contract["side_input_hashes"]["external_hrdataset_v14_schema_mapping"]["sha256"],
        "Schema mapping hash drifted.",
    )
    dataset = load_external_dataset(
        "hrdataset_v14", raw_frame=raw_hr.frame, schema_mapping_path=schema_path
    )
    settings = manuscript_settings(config)
    external = {**dict(settings["external_replication"]), "resolved_seeds": dict(settings["seeds"])}
    frames, roles, forbidden, _ = replication_stage._feature_contract(dataset, external)
    _require(tuple(frames) == EXPECTED_POLICY_ORDER, "External policy order drifted.")
    _require(tuple(frames[PRIMARY_POLICY].columns) == EXPECTED_PRIMARY_FEATURES, "Primary feature policy drifted.")
    _require(
        int(frames[PRIMARY_POLICY]["ExperienceYearsAtThisCompany"].isna().sum()) == 2,
        "Derived-tenure missing-value quality contract drifted.",
    )
    _require(roles[PRIMARY_POLICY] == "canonical_external_primary", "Primary policy role drifted.")
    _require(
        dataset.canonical[dataset.target_column].astype(int).value_counts().sort_index().to_dict()
        == {2: 31, 3: 243, 4: 37},
        "Mapped target support drifted.",
    )
    policy_hash = feature_policy_contract_sha256(
        {policy: list(frame.columns) for policy, frame in frames.items()}
    )
    return dataset, frames, roles, forbidden, policy_hash, content_hash


def _validate_folds(stage_root: Path, *, run_id: str, config_hash: str, scientific_hash: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    outer = _read_csv(stage_root / "folds" / "fold_assignments.csv")
    inner = _read_csv(stage_root / "folds" / "inner_fold_assignments.csv")
    receipt = _read_json(stage_root / "folds" / "fold_contract.json")
    _assert_identity_columns(outer, run_id=run_id, config_hash=config_hash, scientific_input_hash_value=scientific_hash)
    _assert_identity_columns(inner, run_id=run_id, config_hash=config_hash, scientific_input_hash_value=scientific_hash)
    _require(len(outer) == 311 and outer["sample_index"].is_unique, "Outer OOF coverage is not exactly once.")
    _require(set(outer["sample_index"].astype(int)) == set(range(311)), "Outer sample population drifted.")
    _require(set(outer["outer_fold"].astype(int)) == set(range(1, 11)), "Outer fold identity drifted.")
    _require(outer["sample_key_sha256"].is_unique, "Outer sample-key identity is duplicated.")
    _require(outer["y_true"].astype(int).value_counts().sort_index().to_dict() == {2: 31, 3: 243, 4: 37}, "Outer target support drifted.")
    _require(len(inner) == 2799, f"Inner assignment count drifted: {len(inner)}")
    for outer_fold in range(1, 11):
        test_ids = set(outer.loc[outer["outer_fold"].astype(int) == outer_fold, "sample_index"].astype(int))
        train_ids = set(range(311)).difference(test_ids)
        rows = inner[inner["outer_fold"].astype(int) == outer_fold]
        _require(len(rows) == len(train_ids) and rows["sample_index"].is_unique, f"Inner coverage drifted for outer fold {outer_fold}.")
        _require(set(rows["sample_index"].astype(int)) == train_ids, f"Inner membership drifted for outer fold {outer_fold}.")
        _require(not set(rows["sample_index"].astype(int)).intersection(test_ids), f"Outer-test leakage in inner fold {outer_fold}.")
        _require(set(rows["inner_fold"].astype(int)) == set(range(1, 6)), f"Inner fold identity drifted for outer fold {outer_fold}.")
    fold_hash = str(receipt.get("fold_contract_hash", ""))
    _require(bool(SHA256_RE.fullmatch(fold_hash)), "Fold contract hash is invalid.")
    _require(set(outer["fold_contract_hash"].astype(str)) == {fold_hash}, "Outer fold hash drifted.")
    _require(set(inner["fold_contract_hash"].astype(str)) == {fold_hash}, "Inner fold hash drifted.")
    _require(int(receipt["outer_splits"]) == 10 and int(receipt["inner_splits"]) == 5, "10x5 fold receipt drifted.")
    return outer, inner, fold_hash


def _validate_selection_and_models(
    stage_root: Path,
    frames: Mapping[str, pd.DataFrame],
    outer: pd.DataFrame,
    *,
    run_id: str,
    config_hash: str,
    scientific_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[FoldModelReference], float]:
    selected = _read_csv(stage_root / "selected_hyperparameters.csv")
    searches = _read_csv(stage_root / "candidate_search_results.csv")
    fits = _read_csv(stage_root / "candidate_fit_receipts.csv")
    receipts = _read_csv(stage_root / "outer_model_receipts.csv")
    oof = _read_csv(stage_root / "raw_oof_predictions.csv")
    for frame in (selected, searches, fits, receipts, oof):
        _assert_identity_columns(frame, run_id=run_id, config_hash=config_hash, scientific_input_hash_value=scientific_hash)
    _require(len(selected) == 10 and selected["outer_fold"].is_unique, "Selected-hyperparameter rows drifted.")
    _require(not _bool_series(selected, "outer_test_used_for_selection").any(), "Outer test selected hyperparameters.")
    _require(len(searches) == 80, "Expected 80 candidate summaries.")
    _require(not _bool_series(searches, "outer_test_used_for_selection").any(), "Candidate summaries used outer test.")
    _require(len(fits) == 400, "Expected 400 candidate fit receipts.")
    _require(
        not fits.duplicated(["outer_fold", "inner_fold", "candidate_index"]).any(),
        "Candidate fit identity is duplicated.",
    )
    _require(not _bool_series(fits, "outer_test_used_for_selection").any(), "Candidate fit used outer test for selection.")
    _require(not _bool_series(fits, "outer_test_used_for_fit").any(), "Candidate fit trained on outer test.")
    _require(len(receipts) == 50, "Expected 50 persisted outer-model receipts.")
    _require(not receipts.duplicated(["policy", "outer_fold"]).any(), "Outer-model receipt identity is duplicated.")
    _require(not _bool_series(receipts, "outer_test_used_for_fit").any(), "Outer model trained on outer test.")
    _require(_bool_series(receipts, "selected_primary_parameters_reused").all(), "Audit policy did not reuse primary selected parameters.")
    _require(len(oof) == 1555, "Expected 1,555 raw OOF rows.")
    _require(tuple(dict.fromkeys(oof["policy"].astype(str))) == EXPECTED_POLICY_ORDER, "Raw OOF policy order drifted.")
    _require(set(oof["probability_method"].astype(str)) == {"raw"}, "Raw OOF method drifted.")

    selected_lookup = selected.set_index(selected["outer_fold"].astype(int))
    for outer_fold in range(1, 11):
        selected_row = selected_lookup.loc[outer_fold]
        search_rows = searches[searches["outer_fold"].astype(int) == outer_fold]
        chosen = search_rows[_bool_series(search_rows, "selected_by_protocol")]
        _require(len(chosen) == 1, f"Outer fold {outer_fold} has invalid selected candidate count.")
        _require(int(chosen.iloc[0]["candidate_index"]) == int(selected_row["selected_candidate_index"]), f"Outer fold {outer_fold} selected candidate drifted.")
        model_rows = receipts[receipts["outer_fold"].astype(int) == outer_fold]
        _require(set(model_rows["selected_candidate_index"].astype(int)) == {int(selected_row["selected_candidate_index"])}, f"Outer fold {outer_fold} model candidate drifted.")
        _require(set(model_rows["selected_candidate_parameters_json"].astype(str)) == {str(selected_row["selected_candidate_parameters_json"])}, f"Outer fold {outer_fold} model parameters drifted.")

    max_error = 0.0
    primary_refs: list[FoldModelReference] = []
    for row in receipts.itertuples(index=False):
        policy = str(row.policy)
        outer_fold = int(row.outer_fold)
        test_ids = sorted(
            outer.loc[outer["outer_fold"].astype(int) == outer_fold, "sample_index"].astype(int)
        )
        train_ids = sorted(set(range(311)).difference(test_ids))
        _require(int(row.n_test) == len(test_ids) and int(row.n_train) == len(train_ids), f"Model partition count drift: {policy}/{outer_fold}")
        _require(str(row.outer_test_sample_sha256) == replication_core._sample_set_sha256(test_ids), f"Outer-test sample hash drift: {policy}/{outer_fold}")
        _require(str(row.outer_train_sample_sha256) == replication_core._sample_set_sha256(train_ids), f"Outer-train sample hash drift: {policy}/{outer_fold}")
        model_path = _safe_stage_path(stage_root, row.model_artifact_path)
        _require(model_path.stat().st_size == int(row.model_size_bytes), f"Model size drift: {policy}/{outer_fold}")
        model_hash = _sha256_file(model_path)
        _require(model_hash == str(row.model_sha256), f"Model hash drift: {policy}/{outer_fold}")
        model = joblib.load(model_path)
        probabilities = aligned_predict_proba(model, frames[policy].loc[test_ids], labels=LABELS)
        expected = oof[
            (oof["policy"].astype(str) == policy)
            & (oof["outer_fold"].astype(int) == outer_fold)
        ].sort_values("sample_index")
        _require(expected["sample_index"].astype(int).tolist() == test_ids, f"OOF membership drift: {policy}/{outer_fold}")
        expected_probability = expected[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        error = float(np.max(np.abs(probabilities - expected_probability), initial=0.0))
        max_error = max(max_error, error)
        _require(error <= 1e-12, f"Model probability replay failed: {policy}/{outer_fold} error={error}")
        _require(str(row.outer_test_probability_sha256) == replication_core._array_sha256(probabilities, dtype="<f8"), f"Probability hash drift: {policy}/{outer_fold}")
        _require(set(expected["source_outer_model_sha256"].astype(str)) == {model_hash}, f"OOF model hash drift: {policy}/{outer_fold}")
        predicted = np.asarray(LABELS, dtype=int)[np.argmax(probabilities, axis=1)]
        _require(np.array_equal(predicted, expected["y_pred"].astype(int).to_numpy()), f"OOF label replay failed: {policy}/{outer_fold}")
        if policy == PRIMARY_POLICY:
            primary_refs.append(FoldModelReference(outer_fold=outer_fold, model_sha256=model_hash, model_path=model_path))

    for policy in EXPECTED_POLICY_ORDER:
        policy_rows = oof[oof["policy"].astype(str) == policy]
        _require(len(policy_rows) == 311 and policy_rows["sample_index"].is_unique, f"OOF exactly-once coverage failed: {policy}")
        _require(set(policy_rows["sample_index"].astype(int)) == set(range(311)), f"OOF population drift: {policy}")
        fold_lookup = outer.set_index("sample_index")["outer_fold"].astype(int)
        _require(
            policy_rows.set_index("sample_index")["outer_fold"].astype(int).sort_index().equals(fold_lookup.sort_index()),
            f"OOF outer-fold identity drift: {policy}",
        )
        probability = policy_rows[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        _require(np.isfinite(probability).all() and (probability >= 0).all() and (probability <= 1).all(), f"Invalid probabilities: {policy}")
        _require(np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-12), f"Probability normalization failed: {policy}")

    return selected, receipts, sorted(primary_refs, key=lambda item: item.outer_fold), max_error


def _validate_calibration(
    stage_root: Path,
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    raw_oof: pd.DataFrame,
    receipts: pd.DataFrame,
    *,
    run_id: str,
    config_hash: str,
    scientific_hash: str,
) -> float:
    training = _read_csv(stage_root / "calibration_training_oof.csv")
    fit_receipts = _read_csv(stage_root / "calibration_fit_receipts.csv")
    relationships = _read_csv(stage_root / "calibrator_model_relationships.csv")
    parameters = _read_csv(stage_root / "calibrator_parameters.csv")
    sigmoid = _read_csv(stage_root / "sigmoid_oof_predictions.csv")
    for frame in (training, fit_receipts, relationships, parameters, sigmoid):
        _assert_identity_columns(frame, run_id=run_id, config_hash=config_hash, scientific_input_hash_value=scientific_hash)
    _require(len(training) == 2799, "Calibration training OOF count drifted.")
    _require(len(fit_receipts) == 50 and not fit_receipts.duplicated(["outer_fold", "inner_fold"]).any(), "Calibration fit receipt identity drifted.")
    _require(not _bool_series(fit_receipts, "outer_test_used_for_fit").any(), "Calibration cross-fit used outer test.")
    _require(not _bool_series(fit_receipts, "outer_test_used_for_calibrator_fit").any(), "Calibrator fit used outer test.")
    _require(len(relationships) == 10 and relationships["outer_fold"].is_unique, "Calibrator relationship identity drifted.")
    for column in (
        "calibrator_applied_to_exact_source_outer_probabilities",
        "source_outer_model_preserved",
    ):
        _require(_bool_series(relationships, column).all(), f"Calibration relationship {column} is false.")
    for column in (
        "outer_test_used_for_model_selection",
        "outer_test_used_for_model_fit",
        "outer_test_used_for_calibrator_fit",
        "calibration_method_selected_from_outer_test",
    ):
        _require(not _bool_series(relationships, column).any(), f"Calibration leakage flag {column} is true.")
    _require(len(parameters) == 30, "Expected 30 sigmoid class-parameter rows.")
    _require(not _bool_series(parameters, "outer_test_used_for_fit").any(), "Sigmoid parameter fit used outer test.")
    _require(not _bool_series(parameters, "method_selected_from_outer_test").any(), "Sigmoid method used outer test.")
    _require(len(sigmoid) == 311 and sigmoid["sample_index"].is_unique, "Sigmoid OOF coverage is not exactly once.")
    _require(set(sigmoid["probability_method"].astype(str)) == {"predeclared_cross_fitted_sigmoid"}, "Sigmoid method label drifted.")

    primary_receipts = receipts[receipts["policy"].astype(str) == PRIMARY_POLICY].set_index(
        receipts[receipts["policy"].astype(str) == PRIMARY_POLICY]["outer_fold"].astype(int)
    )
    maximum_error = 0.0
    inner_lookup = inner.set_index([inner["outer_fold"].astype(int), inner["sample_index"].astype(int)])["inner_fold"].astype(int)
    for outer_fold in range(1, 11):
        outer_test = set(outer.loc[outer["outer_fold"].astype(int) == outer_fold, "sample_index"].astype(int))
        outer_train = set(range(311)).difference(outer_test)
        fold_training = training[training["outer_fold"].astype(int) == outer_fold].sort_values("sample_index")
        _require(len(fold_training) == len(outer_train) and fold_training["sample_index"].is_unique, f"Calibration exactly-once training failed: fold {outer_fold}")
        _require(set(fold_training["sample_index"].astype(int)) == outer_train, f"Calibration training membership drift: fold {outer_fold}")
        _require(not outer_test.intersection(fold_training["sample_index"].astype(int)), f"Calibration outer-test leakage: fold {outer_fold}")
        for row in fold_training.itertuples(index=False):
            _require(int(row.inner_fold) == int(inner_lookup.loc[(outer_fold, int(row.sample_index))]), f"Calibration inner-fold provenance drift: fold {outer_fold}/sample {row.sample_index}")

        relationship = relationships[relationships["outer_fold"].astype(int) == outer_fold].iloc[0]
        source_receipt = primary_receipts.loc[outer_fold]
        _require(str(relationship["source_outer_model_sha256"]) == str(source_receipt["model_sha256"]), f"Calibrator source-model drift: fold {outer_fold}")
        raw_test = raw_oof[
            (raw_oof["policy"].astype(str) == PRIMARY_POLICY)
            & (raw_oof["outer_fold"].astype(int) == outer_fold)
        ].sort_values("sample_index")
        raw_probability = raw_test[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        training_probability = fold_training[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        training_labels = fold_training["y_true"].astype(int).to_numpy()
        training_probability_hash = replication_core._array_sha256(training_probability, dtype="<f8")
        training_labels_hash = replication_core._array_sha256(training_labels, dtype="<i8")
        _require(str(relationship["source_outer_raw_probability_sha256"]) == replication_core._array_sha256(raw_probability, dtype="<f8"), f"Calibrator raw source hash drift: fold {outer_fold}")
        _require(str(relationship["calibration_training_probability_sha256"]) == training_probability_hash, f"Calibrator training probability hash drift: fold {outer_fold}")
        _require(str(relationship["calibration_training_labels_sha256"]) == training_labels_hash, f"Calibrator training label hash drift: fold {outer_fold}")
        parameter_rows = parameters[parameters["outer_fold"].astype(int) == outer_fold].sort_values("class_label")
        _require(parameter_rows["class_label"].astype(int).tolist() == list(LABELS), f"Calibrator class order drift: fold {outer_fold}")
        calibrator = SigmoidCalibrator(
            labels=LABELS,
            class_parameters=tuple(
                SigmoidClassParameters(
                    class_label=int(row.class_label),
                    coefficient=float(row.coefficient),
                    intercept=float(row.intercept),
                    n_positive=int(row.n_positive),
                    n_negative=int(row.n_negative),
                    n_iter=int(row.n_iter),
                )
                for row in parameter_rows.itertuples(index=False)
            ),
            seed=42,
            solver="lbfgs",
            regularization="l2_via_l1_ratio_zero",
            l1_ratio=0.0,
            c_value=1.0,
            fit_intercept=True,
            max_iter=1000,
            tolerance=1e-10,
            probability_clip=1e-6,
            threadpool_limit=1,
            training_probability_sha256=training_probability_hash,
            training_labels_sha256=training_labels_hash,
        )
        _require(calibrator.parameter_sha256 == str(relationship["calibrator_parameter_sha256"]), f"Calibrator parameter hash drift: fold {outer_fold}")
        _require(set(parameter_rows["calibrator_parameter_sha256"].astype(str)) == {calibrator.parameter_sha256}, f"Calibrator parameter-row hash drift: fold {outer_fold}")
        calibrated = apply_sigmoid_calibrator(calibrator, raw_probability)
        expected = sigmoid[sigmoid["outer_fold"].astype(int) == outer_fold].sort_values("sample_index")
        _require(expected["sample_index"].astype(int).tolist() == raw_test["sample_index"].astype(int).tolist(), f"Sigmoid sample identity drift: fold {outer_fold}")
        error = float(np.max(np.abs(calibrated - expected[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)), initial=0.0))
        maximum_error = max(maximum_error, error)
        _require(error <= 1e-12, f"Sigmoid replay failed: fold {outer_fold} error={error}")
        _require(str(relationship["calibrated_outer_probability_sha256"]) == replication_core._array_sha256(calibrated, dtype="<f8"), f"Calibrated probability hash drift: fold {outer_fold}")
    probability = sigmoid[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    _require(np.isfinite(probability).all() and np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-12), "Sigmoid probabilities are invalid.")
    return maximum_error


def _validate_bootstrap(stage_root: Path, primary_oof: pd.DataFrame) -> tuple[str, int]:
    receipt = _read_json(stage_root / "bootstrap_resample_plan.json")
    compressed_path = stage_root / "bootstrap_resample_indices.npy.zlib"
    compressed = compressed_path.read_bytes()
    _require(_sha256_bytes(compressed) == receipt["compressed_indices_sha256"], "Compressed bootstrap hash drifted.")
    uncompressed = zlib.decompress(compressed)
    _require(_sha256_bytes(uncompressed) == receipt["uncompressed_npy_sha256"], "Uncompressed bootstrap hash drifted.")
    indices = np.load(io.BytesIO(uncompressed), allow_pickle=False)
    _require(indices.dtype.str == "<i8" and tuple(indices.shape) == (5000, 311), "Bootstrap matrix shape/dtype drifted.")
    _require(int(indices.min()) >= 0 and int(indices.max()) < 311, "Bootstrap indices are out of bounds.")
    sample_order = _read_csv(stage_root / "bootstrap_sample_order.csv")
    _require(len(sample_order) == 311 and sample_order["sample_index"].is_unique, "Bootstrap sample order drifted.")
    order_bytes = sample_order[["sample_position", "sample_index", "outer_fold", "y_true"]].to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")
    _require(_sha256_bytes(order_bytes) == receipt["sample_order_sha256"], "Bootstrap sample-order hash drifted.")
    base = primary_oof[["sample_index", "outer_fold", "y_true"]].drop_duplicates()
    replay = generate_stratified_resample_indices(base, BootstrapProtocol())
    _require(replay.sorted_sample_ids == tuple(sample_order["sample_index"].astype(int)), "Bootstrap sorted sample IDs drifted.")
    _require(np.array_equal(replay.indices, indices), "Bootstrap index matrix does not replay.")
    _require(replay.resample_hash == receipt["resample_hash"], "Bootstrap resample hash does not replay.")
    return str(receipt["resample_hash"]), int(receipt["n_resamples"])


def _validate_shap(
    stage_root: Path,
    frames: Mapping[str, pd.DataFrame],
    forbidden: Mapping[str, Sequence[str]],
    outer: pd.DataFrame,
    primary_oof: pd.DataFrame,
    primary_refs: Sequence[FoldModelReference],
    *,
    identity: ReplicationIdentity,
    governance: Mapping[str, Mapping[str, Any]],
) -> tuple[int, float, float]:
    metadata = _read_json(stage_root / "shap" / "shap_metadata.json")
    _require(metadata.get("model_refit_in_diagnostic") is False, "SHAP diagnostic refit a model.")
    _require(int(metadata.get("n_samples", -1)) == 311 and int(metadata.get("n_outer_folds", -1)) == 10, "SHAP sample/fold count drifted.")
    _require(model_set_sha256(primary_refs) == identity.model_set_sha256, "SHAP model-set identity drifted.")
    replay_oof = primary_oof.copy()
    for field, value in identity.as_dict().items():
        if field not in replay_oof.columns:
            replay_oof[field] = value
    replay_features = frames[PRIMARY_POLICY].copy()
    replay_features.insert(0, "sample_index", replay_features.index.astype(int))
    recomputed = compute_exact_oof_grouped_shap(
        features=replay_features,
        fold_assignments=outer,
        oof_predictions=replay_oof,
        fold_models=primary_refs,
        policy_features={policy: list(frame.columns) for policy, frame in frames.items()},
        primary_policy=PRIMARY_POLICY,
        forbidden_features=forbidden[PRIMARY_POLICY],
        identity=identity,
        labels=LABELS,
        feature_governance=governance,
        top_k=5,
    )
    persisted = _read_csv(stage_root / "shap" / "local_grouped_shap_values.csv")
    key = ["sample_index", "outer_fold", "class_label", "feature"]
    left = persisted.sort_values(key).reset_index(drop=True)
    right = recomputed.local_values.sort_values(key).reset_index(drop=True)
    _require(len(left) == len(right) == 6531, "Local grouped SHAP row count drifted.")
    _require(left[key].astype(str).equals(right[key].astype(str)), "Local grouped SHAP identity drifted.")
    _require(left["model_sha256"].astype(str).equals(right["model_sha256"].astype(str)), "Local grouped SHAP model identity drifted.")
    difference = float(
        np.max(
            np.abs(
                left["grouped_shap_value"].to_numpy(dtype=float)
                - right["grouped_shap_value"].to_numpy(dtype=float)
            ),
            initial=0.0,
        )
    )
    _require(difference <= 1e-12, f"Grouped SHAP replay failed: error={difference}")
    features = set(left["feature"].astype(str))
    _require(features == set(EXPECTED_PRIMARY_FEATURES), "Grouped SHAP feature set drifted.")
    normalized_forbidden = {
        "".join(character for character in str(value).casefold() if character.isalnum())
        for value in forbidden[PRIMARY_POLICY]
    }
    _require(
        not {
            "".join(character for character in feature.casefold() if character.isalnum())
            for feature in features
        }.intersection(normalized_forbidden),
        "Forbidden feature appears in grouped SHAP.",
    )
    reason_features: set[str] = set()
    for path in sorted((stage_root / "shap" / "local_reason_codes").glob("*.json")):
        reason_features.update(_collect_feature_values(_read_json(path)))
    _require(reason_features.issubset(set(EXPECTED_PRIMARY_FEATURES)), "Forbidden/unexpected feature appears in reason-code JSON.")
    for path in sorted((stage_root / "shap" / "local_reason_codes").glob("*.csv")):
        frame = _read_csv(path)
        _require(set(frame["feature"].astype(str)).issubset(set(EXPECTED_PRIMARY_FEATURES)), f"Forbidden/unexpected feature appears in {path.name}.")
    stability = _read_csv(stage_root / "shap" / "shap_stability_pairwise.csv")
    _require(len(stability) == 45, "SHAP fold-pair stability count drifted.")
    _require(metadata.get("confidence_interval_for_fold_pairs") is False, "Dependent SHAP fold pairs claim a confidence interval.")
    max_additivity = float(left["shap_additivity_max_abs_error"].max())
    _require(max_additivity <= 1e-4, "SHAP additivity tolerance failed.")
    return len(left), difference, max_additivity


def _validate_support_claims_and_sources(stage_root: Path) -> tuple[int, int, int]:
    target = _read_csv(stage_root / "target_support.csv")
    overall = target[target["support_scale"].astype(str) == "mapped"]
    _require(set(overall["target_value"].astype(int)) == set(LABELS), "Overall target labels drifted.")
    _require(dict(zip(overall["target_value"].astype(int), overall["count"].astype(int), strict=True)) == {2: 31, 3: 243, 4: 37}, "Target-support table drifted.")
    _require(set(overall["n_total"].astype(int)) == {311}, "Target-support denominator drifted.")

    subgroup_metadata = _read_json(stage_root / "subgroup_diagnostics" / "subgroup_metadata.json")
    groups = _read_csv(stage_root / "subgroup_diagnostics" / "group_metrics.csv")
    disparities = _read_csv(stage_root / "subgroup_diagnostics" / "disparity_intervals.csv")
    _require(int(subgroup_metadata["minimum_group_support"]) == 30, "Subgroup support threshold drifted.")
    _require(int(subgroup_metadata["minimum_metric_denominator"]) == 10, "Subgroup metric-denominator threshold drifted.")
    _require(int(subgroup_metadata["n_resamples"]) == 5000, "Subgroup bootstrap count drifted.")
    _require(subgroup_metadata["inference_scope"] == "pointwise_descriptive", "Subgroup inference scope drifted.")
    _require(len(groups) == 391 and len(disparities) == 85, "Subgroup table row counts drifted.")
    _require(set(groups["minimum_group_support_threshold"].astype(int)) == {30}, "Group support denominator drifted.")
    _require(set(groups["minimum_metric_denominator_threshold"].astype(int)) == {10}, "Metric denominator drifted.")
    _require(set(disparities["n_resamples"].astype(int)) == {5000}, "Disparity bootstrap count drifted.")
    headline = int(_bool_series(disparities, "headline_eligible").sum())
    _require(headline == 30, f"Headline-eligible subgroup count drifted: {headline}")
    estimable = disparities[
        disparities["estimate_status"].astype(str)
        != "insufficient_subgroup_or_metric_support"
    ]
    _require(len(estimable) == 61, f"Estimable subgroup count drifted: {len(estimable)}")
    stable_intervals = disparities[
        disparities["estimate_status"].astype(str).str.startswith("support_sufficient")
    ]
    _require(len(stable_intervals) == 33, "Stable subgroup interval count drifted.")
    _require(
        (stable_intervals["n_valid_bootstrap"].astype(int) >= 4000).all(),
        "Stable subgroup bootstrap denominator is insufficient.",
    )

    proxy = _read_json(stage_root / "proxy_diagnostics" / "proxy_metadata.json")
    proxy_status = _read_csv(stage_root / "proxy_diagnostics" / "proxy_status.csv")
    expected_status = "not_estimated_insufficient_outer_training_class_support"
    _require(proxy.get("analysis_status") == expected_status, "Proxy support status drifted.")
    _require(int(proxy.get("models_fitted", -1)) == 0 and int(proxy.get("n_outer_training_missing_class_cells", -1)) == 1, "Proxy fail-closed denominator drifted.")
    _require(set(proxy_status["analysis_status"].astype(str)) == {expected_status}, "Proxy status table drifted.")

    transport = _read_json(stage_root / "cross_dataset_transport" / "transport_feasibility.json")
    _require(transport.get("locked_inx_model_transported") is False, "A locked INX model was transported.")
    _require(int(transport.get("n_common_safe_features", -1)) == 3, "Transport safe-feature count drifted.")
    _require(int(transport.get("minimum_common_feature_gate", -1)) == 5, "Transport gate drifted.")
    _require(transport.get("status") == "infeasible_too_few_common_safe_features", "Transport status drifted.")

    metadata = _read_json(stage_root / "external_replication_metadata.json")
    claim = str(metadata.get("claim_boundary", ""))
    _require("mapped-target replication" in claim and "not locked-model transport" in claim, "External claim boundary drifted.")
    _require("fairness proof" in claim and "causal evidence" in claim, "External prohibited-claim boundary is incomplete.")
    _require(int(metadata.get("network_calls", -1)) == 0 and int(metadata.get("paid_api_calls", -1)) == 0, "Network/API receipt drifted.")
    interpretation = (stage_root / "external_replication_interpretation.md").read_text(encoding="utf-8")
    _require("leakage-aware" in interpretation and "leakage-safe" not in interpretation.casefold(), "Leakage-aware terminology drifted.")
    _require("not a canonical" not in interpretation.casefold(), "Stage interpretation makes an unexpected canonical claim.")
    replication_stage._validate_portability_and_scope(stage_root)

    artifact_paths = set(_read_csv(stage_root / "artifact_manifest.csv")["path"].astype(str))
    for relative in REQUIRED_FIGURE_TABLE_SOURCES:
        _require((stage_root / relative).is_file(), f"Figure/table source is missing: {relative}")
        _require(relative in artifact_paths, f"Figure/table source is not manifest-bound: {relative}")
    return len(groups), len(disparities), headline


def _validate_noncanonical_package_boundary(stage_root: Path, run_id: str) -> Mapping[str, Any]:
    core_root = stage_root.parent
    run_root = core_root.parent
    provisional = _read_json(core_root / "stage_validation_input_manifest.json")
    _require(provisional.get("status") == "running", "Enclosing input manifest is not explicitly provisional.")
    _require(provisional.get("end_timestamp") is None and provisional.get("commands") == [], "Provisional manifest looks terminal.")
    _require(provisional.get("output_files") == [], "Provisional manifest registers package outputs.")
    forbidden_package_files = (
        core_root / "run_manifest.json",
        core_root / "final_evidence_manifest.json",
        core_root / "final_evidence_manifest.csv",
        core_root / "package_status.json",
        core_root / "claim_matrix.csv",
        run_root / "run_manifest.json",
        run_root / "package_status.json",
    )
    _require(not any(path.exists() for path in forbidden_package_files), "Stage-validation run has a package-level completion artifact.")
    pointer = run_root.parent / "latest" / "pointer.json"
    if pointer.is_file():
        _require(run_id not in pointer.read_text(encoding="utf-8"), "Stage-validation run was promoted to latest.")
    return {
        "atomic_stage_status": "complete",
        "enclosing_manifest_status": "running",
        "canonical_package": False,
        "promoted_to_latest": False,
    }


def validate_unit2g_stage(stage_root: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    stage = Path(stage_root)
    if not stage.is_absolute():
        stage = root / stage
    stage = stage.resolve()
    _require(stage.is_dir(), f"Unit 2G stage root is missing: {stage}")
    try:
        stage.relative_to(root)
    except ValueError as exc:
        raise Unit2GValidationError("Unit 2G stage root must be inside the repository.") from exc

    contract = _read_json(stage / "stage_contract.json")
    run_id = str(contract.get("run_id", ""))
    _require(run_id.startswith("stage_validation_hrdataset_"), "Unexpected Unit 2G run identity.")
    stage_output_count, stage_file_count = _validate_inventory(stage, contract)
    artifact_count = _validate_artifact_manifest(
        stage,
        run_id=run_id,
        config_hash=str(contract["config_hash"]),
        scientific_input_hash_value=str(contract["scientific_input_hash"]),
    )
    config, commit, config_hash, scientific_hash = _validate_generation_identity(root, stage, contract)
    dataset, frames, roles, forbidden, policy_hash, content_hash = _validate_data_and_policy(
        root, config, contract
    )
    metadata = _read_json(stage / "external_replication_metadata.json")
    _require(policy_hash == metadata["feature_policy_contract_sha256"], "Feature-policy hash drifted.")
    outer, inner, fold_hash = _validate_folds(
        stage, run_id=run_id, config_hash=config_hash, scientific_hash=scientific_hash
    )
    raw_oof = _read_csv(stage / "raw_oof_predictions.csv")
    selected, receipts, primary_refs, model_error = _validate_selection_and_models(
        stage,
        frames,
        outer,
        run_id=run_id,
        config_hash=config_hash,
        scientific_hash=scientific_hash,
    )
    calibration_error = _validate_calibration(
        stage,
        outer,
        inner,
        raw_oof,
        receipts,
        run_id=run_id,
        config_hash=config_hash,
        scientific_hash=scientific_hash,
    )
    primary_oof = raw_oof[raw_oof["policy"].astype(str) == PRIMARY_POLICY].copy()
    resample_hash, draw_count = _validate_bootstrap(stage, primary_oof)
    model_set_hash = model_set_sha256(primary_refs)
    identity = ReplicationIdentity(
        run_id=run_id,
        config_hash=config_hash,
        scientific_input_hash=scientific_hash,
        dataset_sha256=str(contract["dataset_hashes"]["hrdataset_v14"]["sha256"]),
        schema_mapping_sha256=str(
            contract["side_input_hashes"]["external_hrdataset_v14_schema_mapping"]["sha256"]
        ),
        fold_contract_hash=fold_hash,
        feature_policy_contract_sha256=policy_hash,
        model_set_sha256=model_set_hash,
    )
    settings = manuscript_settings(config)
    external = {**dict(settings["external_replication"]), "resolved_seeds": dict(settings["seeds"])}
    shap_rows, shap_error, max_additivity = _validate_shap(
        stage,
        frames,
        forbidden,
        outer,
        primary_oof,
        primary_refs,
        identity=identity,
        governance=replication_stage._governance_mapping(external),
    )
    group_rows, disparity_rows, headline_rows = _validate_support_claims_and_sources(stage)
    package_boundary = _validate_noncanonical_package_boundary(stage, run_id)
    current_head = str(_git(root, "rev-parse", "HEAD", text=True)).strip()

    return {
        "schema_version": 1,
        "status": "passed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_scope": "unit_2g_atomic_stage_acceptance_noncanonical",
        "reused_existing_output": True,
        "scientific_stage_rerun": False,
        "run_id": run_id,
        "stage_root": stage.relative_to(root).as_posix(),
        "validation_head": current_head,
        "generation_identity": {
            "git_commit": commit,
            "config_hash": config_hash,
            "scientific_input_hash": scientific_hash,
            "source_tree_hash": str(contract["source_tree_hash"]),
            "scope_contract_hash": str(contract["scope_contract_hash"]),
            "raw_dataset_sha256": str(contract["dataset_hashes"]["hrdataset_v14"]["sha256"]),
            "canonical_content_hash_algorithm": "sha256_utf8_json_rows_csv_utf8_sig_v1",
            "canonical_content_sha256": content_hash,
            "schema_mapping_sha256": identity.schema_mapping_sha256,
            "feature_policy_contract_sha256": policy_hash,
            "fold_contract_hash": fold_hash,
            "model_set_sha256": model_set_hash,
            "bootstrap_resample_hash": resample_hash,
        },
        "counts": {
            "stage_contract_outputs": stage_output_count,
            "stage_files_including_contract": stage_file_count,
            "artifact_manifest_rows": artifact_count,
            "samples": 311,
            "classes": {"2": 31, "3": 243, "4": 37},
            "outer_folds": 10,
            "inner_folds_per_outer": 5,
            "candidate_fit_receipts": 400,
            "persisted_outer_models": 50,
            "raw_oof_rows": 1555,
            "calibration_fit_receipts": 50,
            "calibration_training_oof_rows": 2799,
            "sigmoid_oof_rows": 311,
            "bootstrap_draws": draw_count,
            "local_grouped_shap_rows": shap_rows,
            "subgroup_metric_rows": group_rows,
            "subgroup_disparity_rows": disparity_rows,
            "subgroup_headline_eligible_rows": headline_rows,
        },
        "replay": {
            "maximum_outer_model_probability_error": model_error,
            "maximum_sigmoid_probability_error": calibration_error,
            "maximum_grouped_shap_value_error": shap_error,
            "maximum_shap_additivity_error": max_additivity,
        },
        "publication_boundary": {
            **package_boundary,
            "stage_role": "independent_mapped_target_replication_stage_validation",
            "source_tables": list(REQUIRED_FIGURE_TABLE_SOURCES),
            "proxy_status": "not_estimated_insufficient_outer_training_class_support",
            "locked_transport_status": "infeasible_too_few_common_safe_features",
            "ethics_status": "pending_manual_submission_blocker",
            "dataset_source_licence_status": "manual_review_required",
            "manuscript_claims_frozen": False,
        },
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_root", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    summary = validate_unit2g_stage(arguments.stage_root, project_root=arguments.project_root)
    if arguments.output is not None:
        output = arguments.output
        if not output.is_absolute():
            output = Path(arguments.project_root) / output
        _write_atomic_json(output.resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
