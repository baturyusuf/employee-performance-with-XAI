"""Run the verified INX model benchmark as a noncanonical, isolated trial.

This entry point exists so the expensive model-comparison decision gate can be
observed before the remainder of the core paper pipeline is technically frozen.
It deliberately does *not* call the canonical package builder, change a
``release_ready`` flag, update ``reports/manuscript_final/latest``, or execute
any downstream policy, calibration, SHAP, fairness, table, or figure stage.

Successful output is therefore auditable scientific trial evidence, but never
canonical release evidence.  A later clean canonical build must consume the
recorded user decision and regenerate every required downstream artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import pandas as pd

from src.data.canonical_loader import load_canonical_dataset
from src.experiments.manuscript_model_benchmark import (
    BENCHMARK_METRICS,
    EXPECTED_CANDIDATE_COUNTS,
    PRIMARY_PRACTICAL_TIE_TOLERANCE,
    run as run_model_benchmark,
)
from src.experiments.shared_folds import (
    CONTRACT_FILENAME,
    INNER_ASSIGNMENT_FILENAME,
    OUTER_ASSIGNMENT_FILENAME,
    generate_shared_folds,
    write_shared_folds,
)
from src.governance.manuscript_contract import (
    ACTUAL_INPUT_IDENTITY_FIELDS,
    DEFAULT_CONFIG_PATH,
    RunManifestError,
    create_run_manifest,
    finalize_run_manifest,
    load_manuscript_config,
    manuscript_settings,
    record_command,
    record_failure,
    register_artifact,
    sha256_file,
    utc_now_iso,
    validate_run_manifest,
    write_run_manifest,
)
from src.utils.config_loader import PROJECT_ROOT
from src.models.canonical_models import CANONICAL_MODEL_NAMES


RUN_KIND = "model_benchmark_trial"
EVIDENCE_SCOPE = "core"
TRIAL_ROOT = Path("reports/manuscript_final/trials")
REQUIRED_OUTER_SPLITS = 10
REQUIRED_INNER_SPLITS = 5
REQUIRED_GATE_METRIC = "macro_f1"
REQUIRED_GATE_TRIGGER_RULE = "point_estimate_gt_zero_and_paired_ci_low_gt_zero"
REQUIRED_BOOTSTRAP_RESAMPLES = 5000
REQUIRED_BOOTSTRAP_CONFIDENCE = 0.95
REQUIRED_BOOTSTRAP_METHOD = "paired_stratified_percentile"
REQUIRED_BOOTSTRAP_STRATA = ("outer_fold", "y_true")
REQUIRED_BOOTSTRAP_QUANTILE = "linear"
EXECUTED_STAGES = ("shared_folds", "model_benchmarks")
EXPECTED_SAMPLE_COUNT = 1200
EXPECTED_LABELS = (2, 3, 4)
EXPECTED_CANDIDATE_ROWS = REQUIRED_OUTER_SPLITS * sum(EXPECTED_CANDIDATE_COUNTS.values())
EXPECTED_MODEL_FOLD_ROWS = REQUIRED_OUTER_SPLITS * len(CANONICAL_MODEL_NAMES)
EXPECTED_OOF_ROWS = EXPECTED_SAMPLE_COUNT * len(CANONICAL_MODEL_NAMES)
EXPECTED_INNER_ASSIGNMENT_ROWS = REQUIRED_OUTER_SPLITS * (
    EXPECTED_SAMPLE_COUNT - EXPECTED_SAMPLE_COUNT // REQUIRED_OUTER_SPLITS
)
EXPECTED_MODEL_SUMMARY_ROWS = len(CANONICAL_MODEL_NAMES) * len(BENCHMARK_METRICS)
EXPECTED_PAIRED_ROWS = (len(CANONICAL_MODEL_NAMES) - 1) * len(BENCHMARK_METRICS)
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ModelBenchmarkTrialError(RuntimeError):
    """Raised when a noncanonical trial violates its fail-closed contract."""


@contextmanager
def _deny_network_connections() -> Iterable[None]:
    """Fail any TCP, UDP, or DNS network attempt made by the offline trial."""

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo

    def blocked(*args: Any, **kwargs: Any) -> Any:
        raise ModelBenchmarkTrialError(
            "Network access is prohibited during the offline model benchmark trial."
        )

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.socket.sendto = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.socket.sendto = original_sendto
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo


@dataclass(frozen=True)
class TrialResult:
    """Portable paths and decision state emitted by one completed trial."""

    run_id: str
    run_dir: Path
    run_manifest: Path
    gate_artifact: Path
    decision_required: bool


def _git_porcelain(project_root: Path) -> str:
    """Return the complete Git porcelain state or fail when Git is unavailable."""

    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ModelBenchmarkTrialError(
            "Cannot establish a clean Git worktree for the benchmark trial."
        ) from exc
    return completed.stdout.strip()


def _require_clean_worktree(project_root: Path) -> None:
    status = _git_porcelain(project_root)
    if status:
        raise ModelBenchmarkTrialError(
            "The model benchmark trial requires a clean worktree; commit or preserve "
            "all tracked and untracked changes before running it."
        )


def _repo_relative(path: str | Path, project_root: Path) -> str:
    candidate = Path(path)
    resolved = (project_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ModelBenchmarkTrialError(
            f"Trial path must remain inside the repository: {resolved}"
        ) from exc


def _safe_run_id(value: Any) -> str:
    run_id = str(value)
    if run_id in {".", ".."} or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ModelBenchmarkTrialError(f"Unsafe benchmark-trial run_id: {run_id!r}.")
    return run_id


def _entrypoint_command(config_path: Path, project_root: Path, requested_run_id: str | None) -> str:
    command = (
        "python -m src.experiments.run_model_benchmark_trial "
        f"--config {_repo_relative(config_path, project_root)}"
    )
    if requested_run_id is not None:
        command += f" --run-id {_safe_run_id(requested_run_id)}"
    return command


def _finish_command(
    record: MutableMapping[str, Any],
    *,
    status: str,
    return_code: int,
    elapsed_seconds: float,
) -> None:
    elapsed = float(elapsed_seconds)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ModelBenchmarkTrialError("Command elapsed_seconds must be finite and non-negative.")
    record.update(
        status=status,
        ended_at=utc_now_iso(),
        return_code=int(return_code),
        elapsed_seconds=elapsed,
    )


def _portable_failure_message(error: Exception, project_root: Path) -> str:
    """Remove the repository's machine-specific absolute prefix from failures."""

    message = str(error)
    for prefix in {str(project_root), project_root.as_posix()}:
        if prefix:
            message = message.replace(prefix, ".")
    return message


def _entrypoint_record(manifest: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    records = [
        record
        for record in manifest.get("commands", [])
        if isinstance(record, MutableMapping) and record.get("stage") == "entrypoint"
    ]
    if len(records) != 1:
        raise ModelBenchmarkTrialError("Trial manifest must contain exactly one entrypoint command.")
    return records[0]


def _require_trial_protocol(settings: Mapping[str, Any]) -> tuple[int, int]:
    try:
        outer = int(settings["evaluation"]["cv"]["n_splits"])
        inner = int(settings["model"]["nested_tuning"]["inner_splits"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelBenchmarkTrialError(
            "Canonical configuration does not declare the nested fold counts."
        ) from exc
    if (outer, inner) != (REQUIRED_OUTER_SPLITS, REQUIRED_INNER_SPLITS):
        raise ModelBenchmarkTrialError(
            "The approved model-benchmark trial requires exactly 10 outer and 5 inner folds; "
            f"observed outer={outer}, inner={inner}."
        )
    nested = settings["model"]["nested_tuning"]
    if nested.get("primary_practical_tie_tolerance") != PRIMARY_PRACTICAL_TIE_TOLERANCE:
        raise ModelBenchmarkTrialError(
            "The approved benchmark trial requires primary_practical_tie_tolerance=0.001."
        )
    try:
        bootstrap = settings["evaluation"]["bootstrap"]
        observed_bootstrap = {
            "n_resamples": int(bootstrap["n_resamples"]),
            "confidence_level": float(bootstrap["confidence_level"]),
            "method": str(bootstrap["method"]),
            "stratify_by": tuple(map(str, bootstrap["stratify_by"])),
            "quantile_method": str(bootstrap["quantile_method"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelBenchmarkTrialError(
            "Canonical configuration does not declare the complete paired OOF bootstrap contract."
        ) from exc
    required_bootstrap = {
        "n_resamples": REQUIRED_BOOTSTRAP_RESAMPLES,
        "confidence_level": REQUIRED_BOOTSTRAP_CONFIDENCE,
        "method": REQUIRED_BOOTSTRAP_METHOD,
        "stratify_by": REQUIRED_BOOTSTRAP_STRATA,
        "quantile_method": REQUIRED_BOOTSTRAP_QUANTILE,
    }
    if observed_bootstrap != required_bootstrap:
        raise ModelBenchmarkTrialError(
            "The approved model-benchmark trial requires the frozen 5,000-draw paired "
            f"OOF bootstrap contract; observed {observed_bootstrap}."
        )
    return outer, inner


def _assert_loader_receipt_matches_manifest(
    loaded_receipt: Mapping[str, Any],
    manifest_receipt: Mapping[str, Any],
) -> None:
    mismatches = {
        field: {"loaded": loaded_receipt.get(field), "manifest": manifest_receipt.get(field)}
        for field in ACTUAL_INPUT_IDENTITY_FIELDS
        if loaded_receipt.get(field) != manifest_receipt.get(field)
    }
    if mismatches:
        raise ModelBenchmarkTrialError(
            f"Canonical loader receipt differs from the scoped manifest: {mismatches}"
        )


def _manifest_model_grid(
    manifest: Mapping[str, Any],
    settings: Mapping[str, Any],
    project_root: Path,
) -> tuple[Path, str]:
    side_inputs = manifest.get("side_input_hashes")
    if not isinstance(side_inputs, Mapping):
        raise ModelBenchmarkTrialError("Scoped manifest has no scientific side-input records.")
    record = side_inputs.get("model_search_space")
    if not isinstance(record, Mapping):
        raise ModelBenchmarkTrialError("Scoped manifest does not bind model_search_space.")
    relative_path = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
        raise ModelBenchmarkTrialError("model_search_space side-input record is incomplete.")
    declared_path = settings.get("model", {}).get("search_space_config")
    if declared_path != relative_path:
        raise ModelBenchmarkTrialError(
            "model.search_space_config differs from the manifest-bound model_search_space path."
        )
    path = (project_root / relative_path).resolve()
    _repo_relative(path, project_root)
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ModelBenchmarkTrialError(
            "Manifest-bound model search space is missing or its SHA-256 changed."
        )
    return path, expected_hash


def _all_trial_artifacts(run_dir: Path, manifest_path: Path) -> list[Path]:
    if not run_dir.is_dir():
        return []
    return sorted(
        (
            path.resolve()
            for path in run_dir.rglob("*")
            if path.is_file() and path.resolve() != manifest_path.resolve()
        ),
        key=lambda path: path.as_posix(),
    )


def _validate_runner_result_paths(result: Mapping[str, Any], benchmark_dir: Path) -> None:
    """Reject a benchmark runner that reports a missing or out-of-stage file."""

    root = benchmark_dir.resolve()
    if not result:
        raise ModelBenchmarkTrialError("Benchmark runner returned no artifact paths.")
    for logical_name, raw_path in result.items():
        if not isinstance(raw_path, (str, Path)):
            raise ModelBenchmarkTrialError(
                f"Benchmark result {logical_name!r} is not a filesystem path."
            )
        path = Path(raw_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ModelBenchmarkTrialError(
                f"Benchmark result {logical_name!r} escaped its stage directory."
            ) from exc
        if not path.is_file():
            raise ModelBenchmarkTrialError(
                f"Benchmark result {logical_name!r} does not reference a generated file."
            )


def _artifact_stage(path: Path, run_dir: Path) -> str:
    relative = path.relative_to(run_dir)
    if not relative.parts or relative.parts[0] not in EXECUTED_STAGES:
        raise ModelBenchmarkTrialError(f"Out-of-scope trial artifact: {relative.as_posix()}")
    return relative.parts[0]


def _register_unregistered_artifacts(
    manifest: MutableMapping[str, Any],
    *,
    run_dir: Path,
    manifest_path: Path,
    project_root: Path,
) -> None:
    recorded = {
        str(record.get("path"))
        for record in manifest.get("output_files", [])
        if isinstance(record, Mapping)
    }
    for path in _all_trial_artifacts(run_dir, manifest_path):
        portable = _repo_relative(path, project_root)
        if portable in recorded:
            continue
        stage = _artifact_stage(path, run_dir)
        register_artifact(
            manifest,
            path,
            project_root=project_root,
            stage=stage,
            artifact_type=(
                "shared_fold_artifact" if stage == "shared_folds" else "model_benchmark_artifact"
            ),
        )
        recorded.add(portable)


def _read_and_validate_gate(
    gate_path: Path,
    *,
    manifest: Mapping[str, Any],
    fold_contract_hash: str,
    expected_resamples: int,
) -> tuple[dict[str, Any], bool]:
    try:
        payload = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBenchmarkTrialError(f"Cannot read benchmark decision gate: {gate_path}") from exc
    if not isinstance(payload, dict):
        raise ModelBenchmarkTrialError("Benchmark decision gate must be a JSON object.")
    expected_identity = {
        "run_id": manifest.get("run_id"),
        "config_hash": manifest.get("config_hash"),
        "scientific_input_hash": manifest.get("scientific_input_hash"),
        "fold_contract_hash": fold_contract_hash,
    }
    mismatches = {
        field: {"expected": expected, "observed": payload.get(field)}
        for field, expected in expected_identity.items()
        if payload.get(field) != expected
    }
    if mismatches:
        raise ModelBenchmarkTrialError(f"Benchmark gate identity mismatch: {mismatches}")
    if type(payload.get("gate_triggered")) is not bool:
        raise ModelBenchmarkTrialError("Benchmark gate_triggered must be a JSON boolean.")
    if payload.get("user_decision_required_if_triggered") is not True:
        raise ModelBenchmarkTrialError("Benchmark gate does not preserve the mandatory user decision.")
    if payload.get("gate_metric") != REQUIRED_GATE_METRIC:
        raise ModelBenchmarkTrialError(
            f"Benchmark gate metric must remain exactly {REQUIRED_GATE_METRIC!r}."
        )
    if payload.get("trigger_rule") != REQUIRED_GATE_TRIGGER_RULE:
        raise ModelBenchmarkTrialError(
            "Benchmark gate trigger_rule differs from the predeclared point-estimate + CI rule."
        )
    resample_hash = payload.get("resample_hash")
    if not isinstance(resample_hash, str) or _SHA256_PATTERN.fullmatch(resample_hash) is None:
        raise ModelBenchmarkTrialError("Benchmark gate resample_hash is not a lowercase SHA-256.")
    if payload.get("n_resamples") != expected_resamples:
        raise ModelBenchmarkTrialError(
            "Benchmark gate bootstrap count differs from the canonical trial protocol."
        )
    triggered = bool(payload["gate_triggered"])
    comparisons = payload.get("triggered_comparisons")
    if not isinstance(comparisons, list) or (triggered and not comparisons) or (not triggered and comparisons):
        raise ModelBenchmarkTrialError("Benchmark gate trigger/comparison evidence is inconsistent.")
    return payload, triggered


def _read_csv_contract(
    path: Path,
    *,
    name: str,
    required_columns: Iterable[str],
    expected_rows: int,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ModelBenchmarkTrialError(f"Cannot read benchmark {name} artifact.") from exc
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise ModelBenchmarkTrialError(f"Benchmark {name} lacks columns: {missing}.")
    if len(frame) != expected_rows:
        raise ModelBenchmarkTrialError(
            f"Benchmark {name} must contain exactly {expected_rows} rows; observed {len(frame)}."
        )
    return frame


def _boolean_series(series: pd.Series, *, context: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    converted = series.map(mapping)
    if converted.isna().any():
        raise ModelBenchmarkTrialError(f"Benchmark {context} contains a non-boolean value.")
    return converted.astype(bool)


def _validate_frame_identity(
    frame: pd.DataFrame,
    *,
    name: str,
    manifest: Mapping[str, Any],
    fold_contract_hash: str,
) -> None:
    expected = {
        "run_id": str(manifest.get("run_id")),
        "config_hash": str(manifest.get("config_hash")),
        "scientific_input_hash": str(manifest.get("scientific_input_hash")),
        "fold_contract_hash": fold_contract_hash,
    }
    for column, value in expected.items():
        if column not in frame.columns or set(frame[column].astype(str)) != {value}:
            raise ModelBenchmarkTrialError(
                f"Benchmark {name} does not carry one exact {column} identity."
            )


def _validate_model_outer_grid(frame: pd.DataFrame, *, name: str) -> None:
    observed_models = set(frame["model"].astype(str))
    if observed_models != set(CANONICAL_MODEL_NAMES):
        raise ModelBenchmarkTrialError(f"Benchmark {name} model set is incomplete.")
    outer = pd.to_numeric(frame["outer_fold"], errors="raise").astype(int)
    if set(outer) != set(range(1, REQUIRED_OUTER_SPLITS + 1)):
        raise ModelBenchmarkTrialError(f"Benchmark {name} outer-fold set is incomplete.")
    if frame.assign(_outer=outer).duplicated(["model", "_outer"]).any():
        raise ModelBenchmarkTrialError(f"Benchmark {name} repeats a model/outer-fold row.")


def _validate_benchmark_semantics(
    benchmark_dir: Path,
    *,
    manifest: Mapping[str, Any],
    fold_contract: Mapping[str, Any],
    expected_resamples: int,
) -> tuple[dict[str, Any], bool]:
    """Validate all fixed-size real-INX benchmark evidence before finalization."""

    output = benchmark_dir.resolve()
    fold_hash = str(fold_contract.get("fold_contract_hash"))
    if _SHA256_PATTERN.fullmatch(fold_hash) is None:
        raise ModelBenchmarkTrialError("Persisted fold_contract_hash is not a lowercase SHA-256.")
    if int(fold_contract.get("outer_splits", -1)) != REQUIRED_OUTER_SPLITS:
        raise ModelBenchmarkTrialError("Persisted fold contract is not 10-fold outer CV.")
    if int(fold_contract.get("inner_splits", -1)) != REQUIRED_INNER_SPLITS:
        raise ModelBenchmarkTrialError("Persisted fold contract is not five-fold inner CV.")
    if int(fold_contract.get("n_rows", -1)) != EXPECTED_SAMPLE_COUNT:
        raise ModelBenchmarkTrialError(
            f"Persisted INX fold contract must contain {EXPECTED_SAMPLE_COUNT} samples."
        )

    identity_columns = ("run_id", "config_hash", "scientific_input_hash", "fold_contract_hash")
    inner_assignments = _read_csv_contract(
        benchmark_dir.parent / "shared_folds" / INNER_ASSIGNMENT_FILENAME,
        name="inner_fold_assignments",
        required_columns=(*identity_columns, "outer_fold", "inner_fold", "sample_index"),
        expected_rows=EXPECTED_INNER_ASSIGNMENT_ROWS,
    )
    _validate_frame_identity(
        inner_assignments,
        name="inner_fold_assignments",
        manifest=manifest,
        fold_contract_hash=fold_hash,
    )
    if set(pd.to_numeric(inner_assignments["inner_fold"], errors="raise").astype(int)) != set(
        range(1, REQUIRED_INNER_SPLITS + 1)
    ):
        raise ModelBenchmarkTrialError("Persisted inner assignment labels are not exactly 1..5.")
    inner_counts = inner_assignments.groupby("outer_fold")["sample_index"].agg(["size", "nunique"])
    if (
        len(inner_counts) != REQUIRED_OUTER_SPLITS
        or not inner_counts["size"].eq(EXPECTED_SAMPLE_COUNT - EXPECTED_SAMPLE_COUNT // REQUIRED_OUTER_SPLITS).all()
        or not inner_counts["nunique"].eq(EXPECTED_SAMPLE_COUNT - EXPECTED_SAMPLE_COUNT // REQUIRED_OUTER_SPLITS).all()
    ):
        raise ModelBenchmarkTrialError("Persisted inner assignments do not cover each outer train exactly once.")

    candidate = _read_csv_contract(
        output / "candidate_search_results.csv",
        name="candidate_search_results",
        required_columns=(*identity_columns, "outer_fold", "model", "candidate_index", "n_inner_folds", "selected_by_protocol"),
        expected_rows=EXPECTED_CANDIDATE_ROWS,
    )
    _validate_frame_identity(candidate, name="candidate_search_results", manifest=manifest, fold_contract_hash=fold_hash)
    if set(pd.to_numeric(candidate["n_inner_folds"], errors="raise").astype(int)) != {REQUIRED_INNER_SPLITS}:
        raise ModelBenchmarkTrialError("Every candidate row must record n_inner_folds=5.")
    selected_flags = _boolean_series(candidate["selected_by_protocol"], context="selected_by_protocol")
    for model_name, candidate_count in EXPECTED_CANDIDATE_COUNTS.items():
        scoped = candidate[candidate["model"].astype(str) == model_name].copy()
        counts = scoped.groupby(pd.to_numeric(scoped["outer_fold"], errors="raise").astype(int)).size()
        if set(counts.index) != set(range(1, REQUIRED_OUTER_SPLITS + 1)) or not counts.eq(candidate_count).all():
            raise ModelBenchmarkTrialError(
                f"Candidate grid cardinality is invalid for model {model_name!r}."
            )
        expected_indices = set(range(candidate_count))
        for _, fold_rows in scoped.groupby("outer_fold"):
            if set(pd.to_numeric(fold_rows["candidate_index"], errors="raise").astype(int)) != expected_indices:
                raise ModelBenchmarkTrialError(
                    f"Candidate indices are incomplete for model {model_name!r}."
                )
    selected_by_group = candidate.assign(_selected=selected_flags).groupby(["model", "outer_fold"])["_selected"].sum()
    if len(selected_by_group) != EXPECTED_MODEL_FOLD_ROWS or not selected_by_group.eq(1).all():
        raise ModelBenchmarkTrialError("Every model/outer fold must select exactly one candidate.")

    selected = _read_csv_contract(
        output / "selected_hyperparameters.csv",
        name="selected_hyperparameters",
        required_columns=(*identity_columns, "outer_fold", "model", "selected_candidate_index"),
        expected_rows=EXPECTED_MODEL_FOLD_ROWS,
    )
    _validate_frame_identity(selected, name="selected_hyperparameters", manifest=manifest, fold_contract_hash=fold_hash)
    _validate_model_outer_grid(selected, name="selected_hyperparameters")
    selected_lookup = selected.set_index(["model", "outer_fold"])["selected_candidate_index"].astype(int)
    candidate_selected = candidate.assign(_selected=selected_flags)
    candidate_selected = candidate_selected[candidate_selected["_selected"]].set_index(
        ["model", "outer_fold"]
    )["candidate_index"].astype(int)
    if not selected_lookup.sort_index().equals(candidate_selected.sort_index()):
        raise ModelBenchmarkTrialError(
            "Selected-hyperparameter rows disagree with candidate selection evidence."
        )

    fold_metrics = _read_csv_contract(
        output / "fold_metrics.csv",
        name="fold_metrics",
        required_columns=(*identity_columns, "outer_fold", "model", "n_train", "n_test"),
        expected_rows=EXPECTED_MODEL_FOLD_ROWS,
    )
    _validate_frame_identity(fold_metrics, name="fold_metrics", manifest=manifest, fold_contract_hash=fold_hash)
    _validate_model_outer_grid(fold_metrics, name="fold_metrics")
    train_counts = pd.to_numeric(fold_metrics["n_train"], errors="raise").astype(int)
    test_counts = pd.to_numeric(fold_metrics["n_test"], errors="raise").astype(int)
    expected_test_count = EXPECTED_SAMPLE_COUNT // REQUIRED_OUTER_SPLITS
    expected_train_count = EXPECTED_SAMPLE_COUNT - expected_test_count
    if set(train_counts) != {expected_train_count} or set(test_counts) != {expected_test_count}:
        raise ModelBenchmarkTrialError(
            "Every benchmark fold must record n_train=1080 and n_test=120 for verified INX."
        )

    oof = _read_csv_contract(
        output / "oof_predictions.csv",
        name="oof_predictions",
        required_columns=(*identity_columns, "system_id", "model", "sample_index", "outer_fold", "y_true", "y_pred", "prob_class_2", "prob_class_3", "prob_class_4"),
        expected_rows=EXPECTED_OOF_ROWS,
    )
    _validate_frame_identity(oof, name="oof_predictions", manifest=manifest, fold_contract_hash=fold_hash)
    if set(oof["system_id"].astype(str)) != set(CANONICAL_MODEL_NAMES):
        raise ModelBenchmarkTrialError("OOF system set is incomplete.")
    if not oof["system_id"].astype(str).equals(oof["model"].astype(str)):
        raise ModelBenchmarkTrialError("OOF system_id and model columns disagree.")
    outer_path = benchmark_dir.parent / "shared_folds" / OUTER_ASSIGNMENT_FILENAME
    outer = _read_csv_contract(
        outer_path,
        name="outer_fold_assignments",
        required_columns=(*identity_columns, "sample_index", "outer_fold", "y_true"),
        expected_rows=EXPECTED_SAMPLE_COUNT,
    )
    _validate_frame_identity(
        outer,
        name="outer_fold_assignments",
        manifest=manifest,
        fold_contract_hash=fold_hash,
    )
    expected_samples = set(pd.to_numeric(outer["sample_index"], errors="raise").astype(int))
    outer_map = outer.set_index(pd.to_numeric(outer["sample_index"], errors="raise").astype(int))
    for system_id, rows in oof.groupby("system_id"):
        sample_ids = pd.to_numeric(rows["sample_index"], errors="raise").astype(int)
        if len(rows) != EXPECTED_SAMPLE_COUNT or sample_ids.duplicated().any() or set(sample_ids) != expected_samples:
            raise ModelBenchmarkTrialError(f"OOF coverage is not exactly once for {system_id!r}.")
        keyed = rows.assign(_sample=sample_ids).set_index("_sample").sort_index()
        expected_keyed = outer_map.sort_index()
        if not keyed["outer_fold"].astype(int).equals(expected_keyed["outer_fold"].astype(int)):
            raise ModelBenchmarkTrialError(f"OOF outer folds differ from shared assignments for {system_id!r}.")
        if not keyed["y_true"].astype(int).equals(expected_keyed["y_true"].astype(int)):
            raise ModelBenchmarkTrialError(f"OOF targets differ from shared assignments for {system_id!r}.")
    for label_column in ("y_true", "y_pred"):
        if not set(pd.to_numeric(oof[label_column], errors="raise").astype(int)).issubset(EXPECTED_LABELS):
            raise ModelBenchmarkTrialError(f"OOF {label_column} contains an undeclared class.")
    probabilities = oof[["prob_class_2", "prob_class_3", "prob_class_4"]].apply(
        pd.to_numeric, errors="raise"
    )
    if (
        not probabilities.map(math.isfinite).to_numpy().all()
        or ((probabilities < 0.0) | (probabilities > 1.0)).to_numpy().any()
        or not ((probabilities.sum(axis=1) - 1.0).abs() <= 1e-6).all()
    ):
        raise ModelBenchmarkTrialError("OOF class probabilities are not finite normalized probabilities.")

    summary = _read_csv_contract(
        output / "model_summary.csv",
        name="model_summary",
        required_columns=(*identity_columns, "system_id", "metric", "n_samples", "n_resamples", "resample_hash"),
        expected_rows=EXPECTED_MODEL_SUMMARY_ROWS,
    )
    _validate_frame_identity(summary, name="model_summary", manifest=manifest, fold_contract_hash=fold_hash)
    if set(summary["system_id"].astype(str)) != set(CANONICAL_MODEL_NAMES):
        raise ModelBenchmarkTrialError("Model summary system set is incomplete.")
    if set(summary["metric"].astype(str)) != set(BENCHMARK_METRICS):
        raise ModelBenchmarkTrialError("Model summary metric set is incomplete.")
    if summary.duplicated(["system_id", "metric"]).any():
        raise ModelBenchmarkTrialError("Model summary repeats a system/metric row.")

    paired = _read_csv_contract(
        output / "paired_model_differences.csv",
        name="paired_model_differences",
        required_columns=(*identity_columns, "comparison_id", "metric", "improvement_oriented_difference", "improvement_ci_low", "n_resamples", "n_valid", "resample_hash", "gate_eligible", "gate_triggered"),
        expected_rows=EXPECTED_PAIRED_ROWS,
    )
    _validate_frame_identity(paired, name="paired_model_differences", manifest=manifest, fold_contract_hash=fold_hash)
    expected_comparisons = {
        f"{model}_minus_xgboost" for model in CANONICAL_MODEL_NAMES if model != "xgboost"
    }
    if set(paired["comparison_id"].astype(str)) != expected_comparisons:
        raise ModelBenchmarkTrialError("Paired comparison set is incomplete.")
    if set(paired["metric"].astype(str)) != set(BENCHMARK_METRICS):
        raise ModelBenchmarkTrialError("Paired comparison metric set is incomplete.")
    if paired.duplicated(["comparison_id", "metric"]).any():
        raise ModelBenchmarkTrialError("Paired comparison repeats a comparison/metric row.")

    gate_payload, gate_triggered = _read_and_validate_gate(
        output / "baseline_xgboost_gate.json",
        manifest=manifest,
        fold_contract_hash=fold_hash,
        expected_resamples=expected_resamples,
    )
    resample_hash = str(gate_payload["resample_hash"])
    for name, frame in (("model_summary", summary), ("paired_model_differences", paired)):
        if set(frame["resample_hash"].astype(str)) != {resample_hash}:
            raise ModelBenchmarkTrialError(f"Benchmark {name} resample_hash differs from the gate.")
        if set(pd.to_numeric(frame["n_resamples"], errors="raise").astype(int)) != {expected_resamples}:
            raise ModelBenchmarkTrialError(f"Benchmark {name} resample count is inconsistent.")
    if set(pd.to_numeric(summary["n_samples"], errors="raise").astype(int)) != {EXPECTED_SAMPLE_COUNT}:
        raise ModelBenchmarkTrialError("Model summary does not report 1,200 OOF samples.")
    if set(pd.to_numeric(paired["n_valid"], errors="raise").astype(int)) != {expected_resamples}:
        raise ModelBenchmarkTrialError("Paired comparison contains incomplete bootstrap draws.")

    gate_eligible = _boolean_series(paired["gate_eligible"], context="gate_eligible")
    recorded_trigger = _boolean_series(paired["gate_triggered"], context="gate_triggered")
    eligible_rows = paired[gate_eligible].copy()
    if (
        len(eligible_rows) != len(expected_comparisons)
        or set(eligible_rows["metric"].astype(str)) != {REQUIRED_GATE_METRIC}
        or set(eligible_rows["comparison_id"].astype(str)) != expected_comparisons
    ):
        raise ModelBenchmarkTrialError("Primary gate rows are not the three macro-F1 comparisons.")
    point = pd.to_numeric(paired["improvement_oriented_difference"], errors="raise")
    ci_low = pd.to_numeric(paired["improvement_ci_low"], errors="raise")
    recomputed_trigger = gate_eligible & (point > 0.0) & (ci_low > 0.0)
    if not recorded_trigger.equals(recomputed_trigger):
        raise ModelBenchmarkTrialError("Paired gate_triggered values violate the point + CI rule.")
    triggered_comparisons = sorted(paired.loc[recomputed_trigger, "comparison_id"].astype(str))
    if triggered_comparisons != sorted(map(str, gate_payload["triggered_comparisons"])):
        raise ModelBenchmarkTrialError("Gate JSON triggered comparisons differ from paired evidence.")
    if gate_triggered is not bool(triggered_comparisons):
        raise ModelBenchmarkTrialError("Gate JSON decision differs from paired point + CI evidence.")

    model_index = _read_csv_contract(
        output / "fitted_model_index.csv",
        name="fitted_model_index",
        required_columns=(*identity_columns, "model", "outer_fold", "path", "sha256", "size_bytes"),
        expected_rows=EXPECTED_MODEL_FOLD_ROWS,
    )
    _validate_frame_identity(model_index, name="fitted_model_index", manifest=manifest, fold_contract_hash=fold_hash)
    _validate_model_outer_grid(model_index, name="fitted_model_index")
    for row in model_index.itertuples(index=False):
        relative = Path(str(row.path))
        if relative.is_absolute() or ".." in relative.parts:
            raise ModelBenchmarkTrialError("Model index contains a non-portable model path.")
        model_path = (output / relative).resolve()
        try:
            model_path.relative_to(output)
        except ValueError as exc:
            raise ModelBenchmarkTrialError("Model index path escapes the benchmark directory.") from exc
        if (
            not model_path.is_file()
            or sha256_file(model_path) != str(row.sha256)
            or model_path.stat().st_size != int(row.size_bytes)
        ):
            raise ModelBenchmarkTrialError(f"Fitted model file fails hash/size validation: {relative}")
    return gate_payload, gate_triggered


def _verify_registered_artifacts(
    manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    manifest_path: Path,
    project_root: Path,
) -> None:
    expected_paths = {
        _repo_relative(path, project_root)
        for path in _all_trial_artifacts(run_dir, manifest_path)
    }
    records = manifest.get("output_files")
    if not isinstance(records, list):
        raise ModelBenchmarkTrialError("Trial manifest output_files must be a list.")
    observed_paths = {
        str(record.get("path")) for record in records if isinstance(record, Mapping)
    }
    if observed_paths != expected_paths or len(observed_paths) != len(records):
        raise ModelBenchmarkTrialError(
            "Trial manifest must register every artifact exactly once; "
            f"missing={sorted(expected_paths - observed_paths)}, "
            f"unexpected={sorted(observed_paths - expected_paths)}."
        )
    run_dir_resolved = run_dir.resolve()
    for record in records:
        if not isinstance(record, Mapping):
            raise ModelBenchmarkTrialError("Trial artifact record must be an object.")
        path = (project_root / str(record["path"])).resolve()
        try:
            path.relative_to(run_dir_resolved)
        except ValueError as exc:
            raise ModelBenchmarkTrialError(f"Trial artifact escapes its immutable run root: {path}") from exc
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ModelBenchmarkTrialError(f"Trial artifact hash mismatch: {path}")
        if path.stat().st_size != record.get("size_bytes"):
            raise ModelBenchmarkTrialError(f"Trial artifact size mismatch: {path}")


def verify_trial_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Verify base provenance plus every trial-only noncanonical invariant."""

    root = Path(project_root).resolve()
    path = Path(manifest_path).resolve()
    manifest = validate_run_manifest(
        path,
        project_root=root,
        expected_evidence_scope=EVIDENCE_SCOPE,
        require_complete=True,
        verify_source_tree=True,
    )
    run_id = _safe_run_id(manifest.get("run_id"))
    expected_run_dir = (root / TRIAL_ROOT / run_id / EVIDENCE_SCOPE).resolve()
    if path != expected_run_dir / "run_manifest.json":
        raise ModelBenchmarkTrialError("Trial manifest is outside its required trials/<run_id>/core root.")
    if manifest.get("run_kind") != RUN_KIND:
        raise ModelBenchmarkTrialError("Manifest is not an explicit model_benchmark_trial.")
    if manifest.get("git_worktree_dirty") is not False:
        raise ModelBenchmarkTrialError("Trial manifest does not attest a clean starting worktree.")
    if manifest.get("canonical_release_eligible") is not False:
        raise ModelBenchmarkTrialError("A benchmark trial can never be canonical-release eligible.")
    if manifest.get("latest_pointer_updated") is not False:
        raise ModelBenchmarkTrialError("A benchmark trial must never update the latest pointer.")
    trial_contract = manifest.get("trial_contract")
    expected_trial_contract = {
        "required_outer_splits": REQUIRED_OUTER_SPLITS,
        "required_inner_splits": REQUIRED_INNER_SPLITS,
        "required_primary_practical_tie_tolerance": PRIMARY_PRACTICAL_TIE_TOLERANCE,
        "required_bootstrap": {
            "n_resamples": REQUIRED_BOOTSTRAP_RESAMPLES,
            "confidence_level": REQUIRED_BOOTSTRAP_CONFIDENCE,
            "method": REQUIRED_BOOTSTRAP_METHOD,
            "stratify_by": list(REQUIRED_BOOTSTRAP_STRATA),
            "quantile_method": REQUIRED_BOOTSTRAP_QUANTILE,
        },
        "allowed_stages": list(EXECUTED_STAGES),
        "allow_dataset_download": False,
        "allow_network_or_paid_api": False,
        "network_enforcement": (
            "socket_connect_connect_ex_sendto_create_connection_getaddrinfo_blocked"
        ),
        "canonical_builder_invoked": False,
    }
    if trial_contract != expected_trial_contract:
        raise ModelBenchmarkTrialError("Trial offline/stage contract is incomplete or changed.")
    if manifest.get("executed_stages") != list(EXECUTED_STAGES):
        raise ModelBenchmarkTrialError("Trial executed_stages differs from the two-stage allowlist.")
    if manifest.get("downstream_stages_executed") != []:
        raise ModelBenchmarkTrialError("A benchmark trial cannot execute downstream scientific stages.")
    entrypoint = _entrypoint_record(manifest)
    if (
        entrypoint.get("status") != "complete"
        or entrypoint.get("return_code") != 0
        or not entrypoint.get("ended_at")
    ):
        raise ModelBenchmarkTrialError("Entrypoint command is not finalized with return_code=0.")
    total_elapsed = manifest.get("elapsed_seconds")
    if (
        not isinstance(total_elapsed, (int, float))
        or not math.isfinite(float(total_elapsed))
        or float(total_elapsed) < 0.0
        or entrypoint.get("elapsed_seconds") != total_elapsed
    ):
        raise ModelBenchmarkTrialError("Trial total elapsed_seconds is missing or inconsistent.")
    stage_commands = {
        str(record.get("stage")): record
        for record in manifest.get("commands", [])
        if isinstance(record, Mapping) and record.get("stage") in EXECUTED_STAGES
    }
    if set(stage_commands) != set(EXECUTED_STAGES):
        raise ModelBenchmarkTrialError("Trial manifest lacks one of the two stage command records.")
    for stage, record in stage_commands.items():
        elapsed = record.get("elapsed_seconds")
        if (
            record.get("status") != "complete"
            or record.get("return_code") != 0
            or not record.get("ended_at")
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
        ):
            raise ModelBenchmarkTrialError(f"Stage {stage!r} has no valid runtime completion record.")
    _verify_registered_artifacts(
        manifest,
        run_dir=expected_run_dir,
        manifest_path=path,
        project_root=root,
    )
    gate_relative = manifest.get("gate_artifact")
    if not isinstance(gate_relative, str):
        raise ModelBenchmarkTrialError("Trial manifest has no gate_artifact path.")
    gate_path = (root / gate_relative).resolve()
    if gate_path.parent != expected_run_dir / "model_benchmarks":
        raise ModelBenchmarkTrialError("Gate artifact is outside the benchmark stage directory.")
    fold_contract_path = expected_run_dir / "shared_folds" / CONTRACT_FILENAME
    try:
        fold_contract = json.loads(fold_contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBenchmarkTrialError("Cannot verify the persisted shared-fold contract.") from exc
    settings = manuscript_settings(load_manuscript_config(root / str(manifest["config_path"])))
    expected_resamples = int(settings["evaluation"]["bootstrap"]["n_resamples"])
    _, triggered = _validate_benchmark_semantics(
        expected_run_dir / "model_benchmarks",
        manifest=manifest,
        fold_contract=fold_contract,
        expected_resamples=expected_resamples,
    )
    if manifest.get("decision_required") is not triggered:
        raise ModelBenchmarkTrialError("Manifest decision_required differs from the preserved gate JSON.")
    return manifest


def _run_trial_impl(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    run_id: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> TrialResult:
    """Execute a clean, two-stage real benchmark trial and stop at its gate."""

    trial_started_perf = time.perf_counter()
    root = Path(project_root).resolve()
    _require_clean_worktree(root)
    raw_config_path = Path(config_path)
    resolved_config = (
        (root / raw_config_path).resolve() if not raw_config_path.is_absolute() else raw_config_path.resolve()
    )
    _repo_relative(resolved_config, root)
    command_text = _entrypoint_command(resolved_config, root, run_id)
    manifest = create_run_manifest(
        resolved_config,
        evidence_scope=EVIDENCE_SCOPE,
        project_root=root,
        run_id=run_id,
        allow_dataset_download=False,
        initial_command=command_text,
    )
    if manifest.get("git_worktree_dirty") is not False:
        raise ModelBenchmarkTrialError("Scoped manifest did not confirm the clean worktree precondition.")
    resolved_run_id = _safe_run_id(manifest.get("run_id"))
    run_dir = (root / TRIAL_ROOT / resolved_run_id / EVIDENCE_SCOPE).resolve()
    expected_parent = (root / TRIAL_ROOT).resolve()
    try:
        run_dir.relative_to(expected_parent)
    except ValueError as exc:
        raise ModelBenchmarkTrialError("Resolved trial run directory escapes the trials root.") from exc
    if run_dir.exists():
        raise ModelBenchmarkTrialError(
            f"Benchmark trial run directory already exists and is immutable: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "run_manifest.json"
    manifest.update(
        {
            "run_kind": RUN_KIND,
            "canonical_release_eligible": False,
            "latest_pointer_updated": False,
            "executed_stages": [],
            "downstream_stages_executed": [],
            "decision_required": None,
            "gate_artifact": None,
            "elapsed_seconds": None,
            "trial_contract": {
                "required_outer_splits": REQUIRED_OUTER_SPLITS,
                "required_inner_splits": REQUIRED_INNER_SPLITS,
                "required_primary_practical_tie_tolerance": (
                    PRIMARY_PRACTICAL_TIE_TOLERANCE
                ),
                "required_bootstrap": {
                    "n_resamples": REQUIRED_BOOTSTRAP_RESAMPLES,
                    "confidence_level": REQUIRED_BOOTSTRAP_CONFIDENCE,
                    "method": REQUIRED_BOOTSTRAP_METHOD,
                    "stratify_by": list(REQUIRED_BOOTSTRAP_STRATA),
                    "quantile_method": REQUIRED_BOOTSTRAP_QUANTILE,
                },
                "allowed_stages": list(EXECUTED_STAGES),
                "allow_dataset_download": False,
                "allow_network_or_paid_api": False,
                "network_enforcement": (
                    "socket_connect_connect_ex_sendto_create_connection_getaddrinfo_blocked"
                ),
                "canonical_builder_invoked": False,
            },
        }
    )
    write_run_manifest(manifest, manifest_path, project_root=root, validate=True)

    active_stage = "preflight"
    active_command: MutableMapping[str, Any] | None = None
    active_started_perf: float | None = None
    gate_path: Path | None = None
    try:
        config = load_manuscript_config(resolved_config)
        settings = manuscript_settings(config)
        outer_splits, inner_splits = _require_trial_protocol(settings)
        model_grid_path, model_grid_hash = _manifest_model_grid(manifest, settings, root)

        active_stage = "shared_folds"
        active_started_perf = time.perf_counter()
        active_command = record_command(
            manifest,
            "internal-stage:shared_folds",
            stage=active_stage,
            status="started",
        )
        loaded = load_canonical_dataset(resolved_config, "inx_primary")
        manifest_receipt = manifest.get("actual_input_receipts", {}).get("inx_primary")
        if not isinstance(manifest_receipt, Mapping):
            raise ModelBenchmarkTrialError("Scoped manifest has no INX actual-input receipt.")
        _assert_loader_receipt_matches_manifest(loaded.receipt, manifest_receipt)
        target_column = str(settings["target"]["column"])
        identifiers = settings["governance_fields"]["identifier_fields"]
        if not isinstance(identifiers, list) or len(identifiers) != 1:
            raise ModelBenchmarkTrialError("Trial requires exactly one canonical identifier field.")
        cv = settings["evaluation"]["cv"]
        nested = settings["model"]["nested_tuning"]
        seeds = settings["seeds"]
        folds = generate_shared_folds(
            loaded.frame,
            target_column=target_column,
            id_column=str(identifiers[0]),
            run_id=resolved_run_id,
            config_hash=str(manifest["config_hash"]),
            scientific_input_hash=str(manifest["scientific_input_hash"]),
            dataset_key="inx_primary",
            dataset_sha256=str(manifest_receipt["actual_sha256"]),
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            seed=int(seeds[str(cv["seed"])]),
            inner_seed=int(seeds[str(nested["inner_seed"])]),
        )
        write_shared_folds(folds, run_dir / "shared_folds")
        _register_unregistered_artifacts(
            manifest,
            run_dir=run_dir,
            manifest_path=manifest_path,
            project_root=root,
        )
        manifest["executed_stages"].append(active_stage)
        _finish_command(
            active_command,
            status="complete",
            return_code=0,
            elapsed_seconds=time.perf_counter() - active_started_perf,
        )
        active_command = None
        active_started_perf = None
        write_run_manifest(manifest, manifest_path, project_root=root, validate=True)

        active_stage = "model_benchmarks"
        active_started_perf = time.perf_counter()
        active_command = record_command(
            manifest,
            "internal-stage:model_benchmarks",
            stage=active_stage,
            status="started",
        )
        result_paths = run_model_benchmark(
            resolved_config,
            model_grid_path=model_grid_path,
            shared_folds_dir=run_dir / "shared_folds",
            output_dir=run_dir / "model_benchmarks",
            run_id=resolved_run_id,
            config_hash=str(manifest["config_hash"]),
            scientific_input_hash=str(manifest["scientific_input_hash"]),
            model_grid_sha256=model_grid_hash,
        )
        _validate_runner_result_paths(result_paths, run_dir / "model_benchmarks")
        raw_gate_path = result_paths.get("baseline_gate")
        if not isinstance(raw_gate_path, (str, Path)):
            raise ModelBenchmarkTrialError("Benchmark runner returned no baseline_gate artifact.")
        gate_path = Path(raw_gate_path).resolve()
        if gate_path.parent != (run_dir / "model_benchmarks").resolve():
            raise ModelBenchmarkTrialError("Benchmark gate escaped its stage output directory.")
        expected_resamples = int(settings["evaluation"]["bootstrap"]["n_resamples"])
        _, decision_required = _validate_benchmark_semantics(
            run_dir / "model_benchmarks",
            manifest=manifest,
            fold_contract=folds.contract,
            expected_resamples=expected_resamples,
        )
        _register_unregistered_artifacts(
            manifest,
            run_dir=run_dir,
            manifest_path=manifest_path,
            project_root=root,
        )
        manifest["executed_stages"].append(active_stage)
        manifest["decision_required"] = decision_required
        manifest["gate_artifact"] = _repo_relative(gate_path, root)
        _finish_command(
            active_command,
            status="complete",
            return_code=0,
            elapsed_seconds=time.perf_counter() - active_started_perf,
        )
        active_command = None
        active_started_perf = None

        entrypoint = _entrypoint_record(manifest)
        total_elapsed = time.perf_counter() - trial_started_perf
        manifest["elapsed_seconds"] = total_elapsed
        _finish_command(
            entrypoint,
            status="complete",
            return_code=0,
            elapsed_seconds=total_elapsed,
        )
        finalize_run_manifest(manifest, status="complete")
        _verify_registered_artifacts(
            manifest,
            run_dir=run_dir,
            manifest_path=manifest_path,
            project_root=root,
        )
        write_run_manifest(
            manifest,
            manifest_path,
            project_root=root,
            validate=True,
            require_complete=True,
        )
        verify_trial_manifest(manifest_path, project_root=root)
        return TrialResult(
            run_id=resolved_run_id,
            run_dir=run_dir,
            run_manifest=manifest_path,
            gate_artifact=gate_path,
            decision_required=decision_required,
        )
    except Exception as exc:
        if active_command is not None:
            _finish_command(
                active_command,
                status="failed",
                return_code=1,
                elapsed_seconds=(
                    time.perf_counter() - active_started_perf
                    if active_started_perf is not None
                    else 0.0
                ),
            )
        try:
            _register_unregistered_artifacts(
                manifest,
                run_dir=run_dir,
                manifest_path=manifest_path,
                project_root=root,
            )
        except Exception as registration_exc:
            record_failure(
                manifest,
                stage="artifact_registration",
                error_type=type(registration_exc).__name__,
                message=_portable_failure_message(registration_exc, root),
            )
        record_failure(
            manifest,
            stage=active_stage,
            error_type=type(exc).__name__,
            message=_portable_failure_message(exc, root),
        )
        try:
            total_elapsed = time.perf_counter() - trial_started_perf
            manifest["elapsed_seconds"] = total_elapsed
            _finish_command(
                _entrypoint_record(manifest),
                status="failed",
                return_code=1,
                elapsed_seconds=total_elapsed,
            )
            finalize_run_manifest(manifest, status="failed")
            write_run_manifest(
                manifest,
                manifest_path,
                project_root=root,
                validate=False,
            )
        except Exception:
            # Preserve the original scientific failure.  The atomic writer was
            # already attempted; masking the cause would impede remediation.
            pass
        raise


def run_trial(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    run_id: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> TrialResult:
    """Run the trial with process-local network and DNS operations denied."""

    with _deny_network_connections():
        return _run_trial_impl(
            config_path,
            run_id=run_id,
            project_root=project_root,
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the clean, noncanonical 10x5 INX model-benchmark trial."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_trial(args.config, run_id=args.run_id)
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_manifest": _repo_relative(result.run_manifest, PROJECT_ROOT.resolve()),
                "gate_artifact": _repo_relative(result.gate_artifact, PROJECT_ROOT.resolve()),
                "decision_required": result.decision_required,
                "canonical_release_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ModelBenchmarkTrialError",
    "TrialResult",
    "main",
    "parse_args",
    "run_trial",
    "verify_trial_manifest",
]
