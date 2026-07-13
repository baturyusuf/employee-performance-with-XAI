from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from src.experiments import build_manuscript_evidence as builder
from src.governance import manuscript_contract
from src.utils.config_loader import PROJECT_ROOT


def _identity() -> dict:
    return {
        "manifest_schema_version": 3,
        "run_id": "resume-contract",
        "git_commit": "abc123",
        "git_status_sha256": hashlib.sha256(b"").hexdigest(),
        "source_tree_hash": "a" * 64,
        "config_path": "configs/manuscript_final.yaml",
        "config_hash": "b" * 64,
        "evidence_scope": "core",
        "scope_contract": {"dataset_keys": ["inx_primary", "hrdataset_v14"]},
        "scope_contract_hash": "c" * 64,
        "actual_input_receipts": {"inx_primary": {"actual_sha256": "d" * 64}},
        "dataset_hashes": {"inx_primary": {"sha256": "d" * 64}},
        "side_input_hashes": {"model_search_space": {"sha256": "e" * 64}},
        "scientific_input_hash": "f" * 64,
        "code_package_versions": {"python": "test"},
        "random_seeds": {"model": 42},
    }


def test_resume_worktree_allows_exact_run_files_and_rejects_unrelated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = PROJECT_ROOT / "reports/manuscript_final/resume-contract/core"

    def install(untracked: str) -> None:
        def fake_run(command, **kwargs):
            output = "" if command[1] == "status" else untracked
            return CompletedProcess(command, 0, stdout=output, stderr="")

        monkeypatch.setattr(builder.subprocess, "run", fake_run)

    install("reports/manuscript_final/resume-contract/core/run_manifest.json\0")
    builder._validate_resume_worktree(run_dir)

    install(
        "reports/manuscript_final/resume-contract/core/run_manifest.json\0"
        "unrelated.txt\0"
    )
    with pytest.raises(builder.ManuscriptBuildError, match="exact current run root"):
        builder._validate_resume_worktree(run_dir)


def test_dirty_explicit_resume_loads_only_matching_original_clean_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    provisional = {**_identity(), "git_worktree_dirty": True}
    existing = {**_identity(), "git_worktree_dirty": False, "status": "running"}
    monkeypatch.setattr(builder, "_load_existing_manifest", lambda *args, **kwargs: dict(existing))
    validated_roots: list[Path] = []
    monkeypatch.setattr(
        builder,
        "_validate_resume_worktree",
        lambda path: validated_roots.append(path),
    )

    selected, was_existing = builder._select_start_manifest(
        provisional,
        requested_run_id="resume-contract",
        reuse_compatible=True,
        manifest_path=manifest_path,
        run_dir=tmp_path,
        config_hash="b" * 64,
        evidence_scope="core",
    )
    assert was_existing is True
    assert selected == existing
    assert validated_roots == [tmp_path]

    incompatible = {**existing, "scientific_input_hash": "0" * 64}
    monkeypatch.setattr(
        builder,
        "_load_existing_manifest",
        lambda *args, **kwargs: dict(incompatible),
    )
    with pytest.raises(builder.ManuscriptBuildError, match="identity is incompatible"):
        builder._select_start_manifest(
            provisional,
            requested_run_id="resume-contract",
            reuse_compatible=True,
            manifest_path=manifest_path,
            run_dir=tmp_path,
            config_hash="b" * 64,
            evidence_scope="core",
        )


@pytest.mark.parametrize(
    ("requested_run_id", "reuse_compatible", "manifest_exists", "message"),
    [
        (None, True, True, "explicit"),
        ("resume-contract", False, True, "reuse"),
        ("resume-contract", True, False, "clean worktree"),
    ],
)
def test_dirty_state_cannot_be_relabelled_as_new_or_nonreusable_resume(
    tmp_path: Path,
    requested_run_id: str | None,
    reuse_compatible: bool,
    manifest_exists: bool,
    message: str,
) -> None:
    run_dir = tmp_path if manifest_exists else tmp_path / "new-scope"
    manifest_path = run_dir / "run_manifest.json"
    if manifest_exists:
        manifest_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(builder.ManuscriptBuildError, match=message):
        builder._select_start_manifest(
            {**_identity(), "git_worktree_dirty": True},
            requested_run_id=requested_run_id,
            reuse_compatible=reuse_compatible,
            manifest_path=manifest_path,
            run_dir=run_dir,
            config_hash="b" * 64,
            evidence_scope="core",
        )


def _started_command(stage: str) -> dict:
    return {
        "command": f"command:{stage}",
        "stage": stage,
        "status": "started",
        "started_at": "2026-07-13T00:00:00+00:00",
        "ended_at": None,
        "return_code": None,
    }


def test_resume_requires_exactly_one_active_entrypoint() -> None:
    manifest = {
        **_identity(),
        "status": "running",
        "commands": [],
    }
    with pytest.raises(builder.ManuscriptBuildError, match="exactly one active entrypoint"):
        builder._prepare_resumed_manifest(manifest, entrypoint_command="python -m resume")

    manifest["commands"] = [
        _started_command("entrypoint"),
        _started_command("entrypoint"),
    ]
    with pytest.raises(builder.ManuscriptBuildError, match="exactly one active entrypoint"):
        builder._prepare_resumed_manifest(manifest, entrypoint_command="python -m resume")


def test_resume_rejects_multiple_or_noncanonical_active_stages() -> None:
    manifest = {
        **_identity(),
        "status": "running",
        "commands": [
            _started_command("entrypoint"),
            _started_command("shared_folds"),
            _started_command("model_benchmarks"),
        ],
    }
    with pytest.raises(builder.ManuscriptBuildError, match="at most one active"):
        builder._prepare_resumed_manifest(manifest, entrypoint_command="python -m resume")

    manifest["commands"] = [
        _started_command("entrypoint"),
        _started_command("not_a_stage"),
    ]
    with pytest.raises(builder.ManuscriptBuildError, match="noncanonical stage"):
        builder._prepare_resumed_manifest(manifest, entrypoint_command="python -m resume")


def test_resume_closes_one_active_stage_and_records_new_entrypoint() -> None:
    manifest = {
        **_identity(),
        "status": "running",
        "commands": [
            _started_command("entrypoint"),
            _started_command("shared_folds"),
        ],
    }
    builder._prepare_resumed_manifest(manifest, entrypoint_command="python -m resume")

    assert [record["status"] for record in manifest["commands"][:2]] == [
        "interrupted",
        "interrupted",
    ]
    assert manifest["commands"][-1]["stage"] == "entrypoint"
    assert manifest["commands"][-1]["status"] == "started"


def test_scoped_clean_start_excludes_only_the_verified_sibling_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "reports" / "manuscript_final" / "same-run" / "core"
    allowed.mkdir(parents=True)
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("unrelated\n", encoding="utf-8")
    monkeypatch.setattr(manuscript_contract, "_run_git", lambda *_args: "")

    def fake_run(command, **_kwargs):
        assert command[1:4] == ["ls-files", "--others", "--exclude-standard"]
        return CompletedProcess(
            command,
            0,
            stdout=(
                "reports/manuscript_final/same-run/core/artifact.csv\0"
                "notes.txt\0"
            ),
            stderr="",
        )

    monkeypatch.setattr(manuscript_contract.subprocess, "run", fake_run)
    status, declared = manuscript_contract._scoped_git_status(tmp_path, [allowed])

    assert status == "?? notes.txt"
    assert declared == ["reports/manuscript_final/same-run/core"]


def test_completed_sibling_is_strictly_validated_before_clean_start_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "reports" / "manuscript_final"
    sibling = output_root / "same-run" / "core"
    sibling.mkdir(parents=True)
    manifest = {
        "status": "complete",
        "run_id": "same-run",
        "config_hash": "a" * 64,
        "evidence_scope": "core",
    }
    manifest_path = sibling / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    validated: list[tuple[Path, Path | None]] = []

    def strict(_manifest, *, run_dir, allowed_untracked_root=None, **_kwargs):
        validated.append((run_dir, allowed_untracked_root))
        return {"status": "passed"}

    monkeypatch.setattr(builder, "_validate_completed_run_package", strict)
    roots, package = builder._validated_sibling_scope_for_start(
        output_root=output_root,
        run_id="same-run",
        evidence_scope="supplementary",
        config_hash="a" * 64,
    )

    assert roots == [sibling]
    assert package == output_root / "same-run"
    assert validated == [(sibling, output_root / "same-run")]


def test_second_scope_final_validation_receives_verified_package_root_and_owned_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "reports" / "manuscript_final"
    package_root = output_root / "same-run"
    sibling = package_root / "core"
    sibling.mkdir(parents=True)
    config_path = tmp_path / "configs" / "manuscript_final.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("manuscript_final: {}\n", encoding="utf-8")
    stage_order = {"core": ("core_stage",), "supplementary": ("supp_stage",)}
    settings = {
        "evidence_scopes": {"core": {}, "supplementary": {}},
        "output": {
            "root": "reports/manuscript_final",
            "manifest_filename": "run_manifest.json",
        },
    }
    manifest = {
        "run_id": "same-run",
        "status": "running",
        "git_worktree_dirty": False,
        "scope_contract_hash": "a" * 64,
        "git_commit": "b" * 40,
        "source_tree_hash": "c" * 64,
        "scientific_input_hash": "d" * 64,
        "commands": [],
        "output_files": [],
    }
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(builder, "CANONICAL_STAGE_ORDERS", stage_order)
    monkeypatch.setattr(builder, "load_manuscript_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(builder, "manuscript_settings", lambda _config: settings)
    monkeypatch.setattr(builder, "canonical_config_hash", lambda _config: "e" * 64)
    monkeypatch.setattr(
        builder,
        "evidence_scope_contract",
        lambda _config, scope: {"stages": list(stage_order[scope])},
    )
    monkeypatch.setattr(builder, "validate_scope_release_ready", lambda *_args: {})
    monkeypatch.setattr(
        builder,
        "_validated_sibling_scope_for_start",
        lambda **_kwargs: ([sibling], package_root),
    )
    monkeypatch.setattr(builder, "create_run_manifest", lambda *_args, **_kwargs: dict(manifest))

    def write_manifest(payload, path, **_kwargs):
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    monkeypatch.setattr(builder, "write_run_manifest", write_manifest)

    def write_inputs(context):
        path = context.run_dir / "run_inputs" / "input_contract.json"
        path.parent.mkdir()
        path.write_text("{}\n", encoding="utf-8")
        return [path]

    monkeypatch.setattr(builder, "_write_input_snapshots", write_inputs)
    monkeypatch.setattr(builder, "_register_stage_files", lambda *_args, **_kwargs: None)

    def execute_stage(context, stage, **_kwargs):
        path = context.run_dir / stage / "result.txt"
        path.parent.mkdir()
        path.write_text("verified\n", encoding="utf-8")
        return [path]

    monkeypatch.setattr(builder, "_execute_stage", execute_stage)
    monkeypatch.setattr(builder, "validate_scope_artifact_paths", lambda *_args, **_kwargs: None)

    def write_root_file(context, name):
        path = context.run_dir / name
        path.write_text("verified\n", encoding="utf-8")
        return path

    monkeypatch.setattr(
        builder,
        "_write_claim_report",
        lambda context: write_root_file(context, "canonical_claim_boundaries.md"),
    )
    monkeypatch.setattr(
        builder,
        "_write_package_status",
        lambda context: write_root_file(context, "package_status.json"),
    )

    def final_manifest(run_dir, **_kwargs):
        csv_path = Path(run_dir) / "final_evidence_manifest.csv"
        json_path = Path(run_dir) / "final_evidence_manifest.json"
        csv_path.write_text("verified\n", encoding="utf-8")
        json_path.write_text("{}\n", encoding="utf-8")
        return {"csv": csv_path, "json": json_path}

    monkeypatch.setattr(builder, "build_final_evidence_manifest", final_manifest)
    monkeypatch.setattr(builder, "validate_final_evidence_manifest", lambda *_args, **_kwargs: {})
    strict_calls: list[dict] = []
    monkeypatch.setattr(
        builder,
        "_validate_completed_run_package",
        lambda *_args, **kwargs: strict_calls.append(kwargs) or {"status": "passed"},
    )

    result = builder.build(
        config_path,
        run_id="same-run",
        reuse_compatible=True,
        evidence_scope="supplementary",
    )

    assert result["run_dir"] == package_root / "supplementary"
    assert len(strict_calls) == 1
    assert strict_calls[0]["allowed_untracked_root"] == package_root
    assert isinstance(strict_calls[0]["allowed_lock_token"], str)
    assert not (package_root / "supplementary" / ".run.lock").exists()


def test_run_lock_prevents_duplicate_live_invocation_and_forbids_automatic_stale_takeover(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "core"
    run_dir.mkdir()
    lock, token, recovery = builder._acquire_run_lock(
        run_dir,
        run_id="lock-run",
        evidence_scope="core",
    )
    assert recovery is None
    with pytest.raises(builder.ManuscriptBuildError, match="locked|owner"):
        builder._acquire_run_lock(run_dir, run_id="lock-run", evidence_scope="core")
    with pytest.raises(builder.ManuscriptBuildError, match="while a run lock exists"):
        builder._validate_run_lock_for_package_check(
            run_dir,
            run_id="lock-run",
            evidence_scope="core",
            allowed_lock_token=None,
        )
    builder._validate_run_lock_for_package_check(
        run_dir,
        run_id="lock-run",
        evidence_scope="core",
        allowed_lock_token=token,
    )
    builder._release_run_lock(lock, token)

    lock.write_text(
        json.dumps(
            {
                "hostname": platform.node(),
                "pid": 999999,
                "token": "stale-token",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    before = lock.read_bytes()
    with pytest.raises(builder.ManuscriptBuildError, match="stale-lock takeover|manually"):
        builder._acquire_run_lock(run_dir, run_id="lock-run", evidence_scope="core")
    assert lock.read_bytes() == before
