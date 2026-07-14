from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.experiments import build_manuscript_evidence as builder
from src.experiments import manuscript_hrdataset_replication as hr_stage


STAGE = "external_replication"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> builder.StageContext:
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    config_path = tmp_path / "configs" / "manuscript_final.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("manuscript_final: {}\n", encoding="utf-8")
    run_dir = tmp_path / "reports" / "manuscript_final" / "atomic-run" / "core"
    manifest: dict[str, Any] = {
        "git_commit": "a" * 40,
        "source_tree_hash": "b" * 64,
        "config_hash": "c" * 64,
        "evidence_scope": "core",
        "scope_contract_hash": "d" * 64,
        "dataset_hashes": {
            "hrdataset_v14": {"path": "data/hr.csv", "sha256": "e" * 64}
        },
        "actual_input_receipts": {
            "hrdataset_v14": {
                "dataset_key": "hrdataset_v14",
                "actual_path": "data/hr.csv",
                "actual_sha256": "e" * 64,
            }
        },
        "side_input_hashes": {
            "external_hrdataset_v14_schema_mapping": {
                "path": "data/schema.json",
                "sha256": "f" * 64,
                "size_bytes": 1,
            }
        },
        "scientific_input_hash": "1" * 64,
    }
    return builder.StageContext(
        config_path=config_path,
        config={},
        settings={},
        run_dir=run_dir,
        run_id="atomic-run",
        config_hash="c" * 64,
        manifest=manifest,
        evidence_scope="core",
        scope_contract={"dataset_keys": ["hrdataset_v14"]},
    )


def _write_contract(
    context: builder.StageContext,
    outputs: list[Path],
    *,
    inventory_mode: str | None = "closed_world",
) -> Path:
    stage_dir = context.run_dir / STAGE
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "stage": STAGE,
        "status": "complete",
        "path_basis": "stage_relative",
        "run_id": context.run_id,
        "config_hash": context.config_hash,
        "evidence_scope": context.evidence_scope,
        "scope_contract_hash": context.manifest["scope_contract_hash"],
        "git_commit": context.manifest["git_commit"],
        "source_tree_hash": context.manifest["source_tree_hash"],
        "dataset_hashes": context.manifest["dataset_hashes"],
        "actual_input_receipts": context.manifest["actual_input_receipts"],
        "side_input_hashes": context.manifest["side_input_hashes"],
        "scientific_input_hash": context.manifest["scientific_input_hash"],
        "started_at": "2026-07-13T00:00:00+00:00",
        "ended_at": "2026-07-13T00:00:01+00:00",
        "elapsed_seconds": 1.0,
        "outputs": [
            {
                "path": os.path.relpath(path.resolve(), stage_dir.resolve()).replace("\\", "/"),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in outputs
        ],
    }
    if inventory_mode is not None:
        payload["inventory_mode"] = inventory_mode
    contract = stage_dir / "stage_contract.json"
    contract.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def _runner_counter(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"count": 0}

    def runner(_context):
        calls["count"] += 1
        raise AssertionError("An atomically completed stage must never be recomputed during resume.")

    monkeypatch.setitem(builder.STAGE_RUNNERS, STAGE, runner)
    return calls


def test_hr_stage_writes_completion_contract_before_atomic_replace() -> None:
    source = inspect.getsource(hr_stage.run)
    assert "_write_atomic_stage_contract" in source
    assert source.index("_write_atomic_stage_contract") < source.index(
        "atomic_replace_directory"
    )


def test_resume_reuses_atomically_published_stage_without_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, monkeypatch)
    artifact = context.run_dir / STAGE / "artifact.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")
    contract = _write_contract(context, [artifact])
    calls = _runner_counter(monkeypatch)

    outputs = builder._execute_stage(context, STAGE, reuse_compatible=True)

    assert calls["count"] == 0
    assert outputs == [artifact, contract]


def test_resume_rejects_tampered_atomic_output_without_recomputing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, monkeypatch)
    artifact = context.run_dir / STAGE / "artifact.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")
    _write_contract(context, [artifact])
    artifact.write_text("metric,value\nmacro_f1,0.9\n", encoding="utf-8")
    calls = _runner_counter(monkeypatch)

    with pytest.raises(builder.ManuscriptBuildError, match="non-empty|compatible|contract"):
        builder._execute_stage(context, STAGE, reuse_compatible=True)
    assert calls["count"] == 0
    assert artifact.read_text(encoding="utf-8").endswith("0.9\n")


def test_resume_rejects_unlisted_extra_file_without_recomputing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, monkeypatch)
    artifact = context.run_dir / STAGE / "artifact.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")
    _write_contract(context, [artifact])
    extra = context.run_dir / STAGE / "stale_or_partial.csv"
    extra.write_text("must,not,be,admitted\n", encoding="utf-8")
    calls = _runner_counter(monkeypatch)

    with pytest.raises(builder.ManuscriptBuildError, match="non-empty|inventory|extra|contract"):
        builder._execute_stage(context, STAGE, reuse_compatible=True)
    assert calls["count"] == 0
    assert extra.is_file()


def test_resume_rejects_contract_output_outside_exact_stage_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, monkeypatch)
    outside = context.run_dir / "other_stage" / "foreign.csv"
    outside.parent.mkdir(parents=True)
    outside.write_text("foreign,evidence\n", encoding="utf-8")
    _write_contract(context, [outside])
    calls = _runner_counter(monkeypatch)

    with pytest.raises(builder.ManuscriptBuildError, match="stage root|outside|contract|non-empty"):
        builder._execute_stage(context, STAGE, reuse_compatible=True)
    assert calls["count"] == 0
    assert outside.is_file()


def test_resume_requires_explicit_closed_world_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, monkeypatch)
    artifact = context.run_dir / STAGE / "artifact.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")
    _write_contract(context, [artifact], inventory_mode=None)

    assert not builder._stage_cache_valid(context, STAGE)


def test_nonempty_stage_without_atomic_contract_is_preserved_and_not_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, monkeypatch)
    partial = context.run_dir / STAGE / "partial.csv"
    partial.parent.mkdir(parents=True)
    partial.write_text("partial\n", encoding="utf-8")
    calls = _runner_counter(monkeypatch)

    with pytest.raises(builder.ManuscriptBuildError, match="non-empty|contract"):
        builder._execute_stage(context, STAGE, reuse_compatible=True)
    assert calls["count"] == 0
    assert partial.read_text(encoding="utf-8") == "partial\n"


@pytest.mark.parametrize(
    ("mutation", "relative_extra"),
    [
        ("hidden_extra", ".partial.tmp"),
        ("nested_reserved_receipt", "nested/stage_contract.json"),
        ("pycache_extra", "__pycache__/cached.pyc"),
    ],
)
def test_stage_cache_rejects_every_unlisted_file_including_reserved_or_hidden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    relative_extra: str,
) -> None:
    context = _context(tmp_path, monkeypatch)
    artifact = context.run_dir / STAGE / "artifact.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")
    _write_contract(context, [artifact])
    extra = context.run_dir / STAGE / relative_extra
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text(mutation + "\n", encoding="utf-8")
    calls = _runner_counter(monkeypatch)

    assert not builder._stage_cache_valid(context, STAGE)
    with pytest.raises(builder.ManuscriptBuildError, match="non-empty|contract"):
        builder._execute_stage(context, STAGE, reuse_compatible=True)
    assert calls["count"] == 0
    assert extra.is_file()


def test_stage_cache_rejects_an_empty_hidden_partial_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, monkeypatch)
    artifact = context.run_dir / STAGE / "artifact.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")
    _write_contract(context, [artifact])
    partial = context.run_dir / STAGE / ".partial_cache"
    partial.mkdir()
    calls = _runner_counter(monkeypatch)

    assert not builder._stage_cache_valid(context, STAGE)
    with pytest.raises(builder.ManuscriptBuildError, match="non-empty|contract"):
        builder._execute_stage(context, STAGE, reuse_compatible=True)
    assert calls["count"] == 0
    assert partial.is_dir()


def test_stage_cache_rejects_wrong_stage_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, monkeypatch)
    artifact = context.run_dir / STAGE / "artifact.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")
    contract = _write_contract(context, [artifact])
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["stage"] = "different_stage"
    contract.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    assert not builder._stage_cache_valid(context, STAGE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("started_at", "not-a-timestamp"),
        ("ended_at", "2026-07-12T23:59:59+00:00"),
        ("elapsed_seconds", float("nan")),
    ],
)
def test_stage_cache_rejects_invalid_timing_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    context = _context(tmp_path, monkeypatch)
    artifact = context.run_dir / STAGE / "artifact.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")
    contract = _write_contract(context, [artifact])
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload[field] = value
    contract.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    assert not builder._stage_cache_valid(context, STAGE)


def test_stage_directory_symlink_redirect_is_rejected_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, monkeypatch)
    context.run_dir.mkdir(parents=True)
    external = tmp_path / "external-target"
    external.mkdir()
    stage_link = context.run_dir / STAGE
    try:
        stage_link.symlink_to(external, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable on this host: {exc}")
    calls = _runner_counter(monkeypatch)

    with pytest.raises(builder.ManuscriptBuildError, match="symlink|junction|reparse"):
        builder._execute_stage(context, STAGE, reuse_compatible=True)
    assert calls["count"] == 0
    assert not any(external.iterdir())


def test_production_atomic_writer_rename_is_immediately_cache_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, monkeypatch)
    monkeypatch.setattr(hr_stage, "PROJECT_ROOT", tmp_path)
    context.run_dir.mkdir(parents=True)
    final_stage = context.run_dir / STAGE
    staging = context.run_dir / "external_replication.__staging__"
    staging.mkdir()
    artifact = staging / "artifact.csv"
    artifact.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")

    hr_stage._write_atomic_stage_contract(
        staging,
        final_stage,
        run_id=context.run_id,
        config_hash=context.config_hash,
        scientific_input_hash=str(context.manifest["scientific_input_hash"]),
        scope_contract_hash=str(context.manifest["scope_contract_hash"]),
        git_commit=str(context.manifest["git_commit"]),
        source_tree_hash=str(context.manifest["source_tree_hash"]),
        dataset_hashes=context.manifest["dataset_hashes"],
        actual_input_receipts=context.manifest["actual_input_receipts"],
        side_input_hashes=context.manifest["side_input_hashes"],
        started_at="2026-07-13T00:00:00+00:00",
        elapsed_seconds=1.0,
    )
    os.replace(staging, final_stage)

    assert builder._stage_cache_valid(context, STAGE)


def test_external_stage_orphan_staging_is_preserved_and_blocks_duplicate_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, monkeypatch)
    context.run_dir.mkdir(parents=True)
    orphan = context.run_dir / ".hrdataset-replication-interrupted"
    orphan.mkdir()
    sentinel = orphan / "partial-model.bin"
    sentinel.write_bytes(b"partial but preserved")
    calls = _runner_counter(monkeypatch)

    with pytest.raises(builder.ManuscriptBuildError, match="interrupted staging"):
        builder._execute_stage(context, STAGE, reuse_compatible=True)
    assert calls["count"] == 0
    assert sentinel.read_bytes() == b"partial but preserved"


def test_linklike_artifact_is_rejected_even_when_bytes_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, monkeypatch)
    artifact = context.run_dir / STAGE / "artifact.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")
    _write_contract(context, [artifact])
    original = builder._is_linklike
    monkeypatch.setattr(
        builder,
        "_is_linklike",
        lambda path: path.resolve() == artifact.resolve() or original(path),
    )

    assert not builder._stage_cache_valid(context, STAGE)
