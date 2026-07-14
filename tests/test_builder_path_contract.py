from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from src.experiments import build_manuscript_evidence as builder
from src.governance.manuscript_contract import (
    ManuscriptConfigError,
    RunManifestError,
    validate_manuscript_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = PROJECT_ROOT / "configs" / "manuscript_final.yaml"


def _canonical_config() -> dict[str, Any]:
    return json.loads(CANONICAL_CONFIG.read_text(encoding="utf-8"))


def _install_pre_manifest_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    output: dict[str, Any],
) -> tuple[dict[str, bool], Path]:
    """Stop if build reaches manifest creation; path validation must precede it."""

    settings = {
        "evidence_scopes": {
            "core": {
                "stages": list(builder.CORE_STAGE_ORDER),
                "release_ready": True,
            }
        },
        "output": dict(output),
    }
    config = {"manuscript_final": settings}
    called = {"create_run_manifest": False}

    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(builder, "load_manuscript_config", lambda _path: config)
    monkeypatch.setattr(builder, "manuscript_settings", lambda _config: settings)
    monkeypatch.setattr(builder, "canonical_config_hash", lambda _config: "a" * 64)
    monkeypatch.setattr(builder, "evidence_scope_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(builder, "validate_scope_release_ready", lambda *_args, **_kwargs: {})

    def manifest_creation_must_not_run(*_args, **_kwargs):
        called["create_run_manifest"] = True
        raise AssertionError("manifest creation ran before output-layout validation")

    monkeypatch.setattr(builder, "create_run_manifest", manifest_creation_must_not_run)
    return called, tmp_path / "reports" / "manuscript_final"


def _capture_build_failure(
    *,
    config_path: Path,
    run_id: str,
) -> BaseException | None:
    try:
        builder.build(config_path, run_id=run_id, evidence_scope="core")
    except BaseException as exc:  # assertions below inspect both type and side effects
        return exc
    return None


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        ".",
        "..",
        "../escaped",
        r"..\escaped",
        "nested/run",
        r"nested\run",
        r"C:\outside\run",
        r"\\server\share\run",
        "CON",
        "NUL.txt",
        "trailing.",
        " leading",
        "trailing ",
        "x" * 161,
    ],
)
def test_builder_rejects_nonportable_run_id_before_any_output_mutation(
    run_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called, output_root = _install_pre_manifest_builder(
        monkeypatch,
        tmp_path,
        output={"root": "reports/manuscript_final", "manifest_filename": "run_manifest.json"},
    )

    failure = _capture_build_failure(config_path=tmp_path / "config.json", run_id=run_id)

    assert isinstance(failure, (builder.ManuscriptBuildError, RunManifestError))
    assert "run_id" in str(failure)
    assert called["create_run_manifest"] is False
    assert not output_root.exists()


@pytest.mark.parametrize(
    "raw_root",
    [
        "../outside",
        "reports/../outside",
        ".",
        r"C:\absolute\evidence",
        r"\\server\share\evidence",
    ],
)
def test_config_rejects_absolute_or_noncanonical_output_root(raw_root: str) -> None:
    config = _canonical_config()
    config["manuscript_final"]["output"]["root"] = raw_root

    with pytest.raises(ManuscriptConfigError):
        validate_manuscript_config(config)


@pytest.mark.parametrize(
    "manifest_filename",
    [
        "../run_manifest.json",
        "nested/run_manifest.json",
        ".",
        r"C:\absolute\run_manifest.json",
        r"\\server\share\run_manifest.json",
    ],
)
def test_config_rejects_nonleaf_or_absolute_manifest_filename(
    manifest_filename: str,
) -> None:
    config = _canonical_config()
    config["manuscript_final"]["output"]["manifest_filename"] = manifest_filename

    with pytest.raises(ManuscriptConfigError):
        validate_manuscript_config(config)


@pytest.mark.parametrize(
    ("raw_root", "created_path"),
    [
        ("reports/../escaped-output", "escaped-output"),
        ("ABSOLUTE", "absolute-output"),
    ],
)
def test_builder_defensively_rejects_bad_output_root_without_mkdir(
    raw_root: str,
    created_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_root = (
        str(tmp_path / created_path) if raw_root == "ABSOLUTE" else raw_root
    )
    called, _ = _install_pre_manifest_builder(
        monkeypatch,
        tmp_path,
        output={"root": configured_root, "manifest_filename": "run_manifest.json"},
    )
    target = tmp_path / created_path

    failure = _capture_build_failure(
        config_path=tmp_path / "config.json",
        run_id="valid-run-1",
    )

    assert isinstance(failure, (builder.ManuscriptBuildError, ManuscriptConfigError))
    assert called["create_run_manifest"] is False
    assert not target.exists()


@pytest.mark.parametrize(
    "manifest_filename",
    [
        "../run_manifest.json",
        "nested/run_manifest.json",
        r"C:\absolute\run_manifest.json",
    ],
)
def test_builder_rejects_bad_manifest_filename_before_mkdir(
    manifest_filename: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called, output_root = _install_pre_manifest_builder(
        monkeypatch,
        tmp_path,
        output={
            "root": "reports/manuscript_final",
            "manifest_filename": manifest_filename,
        },
    )

    failure = _capture_build_failure(
        config_path=tmp_path / "config.json",
        run_id="valid-run-1",
    )

    assert isinstance(failure, (builder.ManuscriptBuildError, ManuscriptConfigError))
    assert called["create_run_manifest"] is False
    assert not output_root.exists()


@pytest.mark.parametrize(
    "raw_path",
    [
        "/core_figures/figure.svg",
        "../../core_figures/figure.svg",
        r"..\core_figures\figure.svg",
        "./core_figures/figure.svg",
    ],
)
def test_scope_artifact_paths_reject_raw_absolute_or_dot_segments(raw_path: str) -> None:
    with pytest.raises(builder.ManuscriptBuildError, match="invalid scope path"):
        builder.validate_scope_artifact_paths("core", [raw_path])


def test_scope_artifact_paths_accept_only_canonical_relative_stage_paths() -> None:
    builder.validate_scope_artifact_paths("core", ["core_figures/figure.svg"])


def test_no_reuse_rejects_an_existing_manifest_without_loading_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    provisional = {
        "run_id": "existing-run",
        "git_worktree_dirty": False,
        "status": "running",
    }
    loaded = {**provisional, "loaded_existing": True}
    calls = {"load": 0}

    def fake_load(*_args, **_kwargs):
        calls["load"] += 1
        return dict(loaded)

    monkeypatch.setattr(builder, "_load_existing_manifest", fake_load)

    with pytest.raises(builder.ManuscriptBuildError, match="reuse"):
        builder._select_start_manifest(
            provisional,
            requested_run_id="existing-run",
            reuse_compatible=False,
            manifest_path=manifest_path,
            run_dir=tmp_path,
            config_hash="a" * 64,
            evidence_scope="core",
        )

    assert calls["load"] == 0


def test_generated_run_id_collision_cannot_be_implicitly_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    provisional = {
        "run_id": "generated-collision",
        "git_worktree_dirty": False,
        "status": "running",
    }
    monkeypatch.setattr(
        builder,
        "_load_existing_manifest",
        lambda *_args, **_kwargs: dict(provisional),
    )

    with pytest.raises(builder.ManuscriptBuildError, match="explicit"):
        builder._select_start_manifest(
            provisional,
            requested_run_id=None,
            reuse_compatible=True,
            manifest_path=manifest_path,
            run_dir=tmp_path,
            config_hash="a" * 64,
            evidence_scope="core",
        )


def test_entrypoint_command_is_complete_and_machine_portable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    config = tmp_path / "configs" / "manuscript_final.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("{}\n", encoding="utf-8")

    command = builder._portable_entrypoint_command(
        config,
        evidence_scope="core",
        run_id="portable-run",
        reuse_compatible=False,
    )

    assert command == (
        "python -m src.experiments.build_manuscript_evidence "
        "--config configs/manuscript_final.yaml --scope core --run-id portable-run "
        "--no-reuse-compatible"
    )
    assert str(tmp_path) not in command
    assert ":\\" not in command


def test_promotion_cli_rejects_build_only_scope_option() -> None:
    with pytest.raises(SystemExit, match="scope"):
        builder.main(
            [
                "--config",
                "configs/manuscript_final.yaml",
                "--promote-run-id",
                "portable-run",
                "--scope",
                "supplementary",
            ]
        )


def test_stage_started_command_is_persisted_before_runner_execution() -> None:
    source = inspect.getsource(builder._build_impl)
    loop = source[source.index("command_record = record_command") :]
    assert loop.index("write_run_manifest") < loop.index("_execute_stage")


@pytest.mark.parametrize("with_sentinel", [False, True])
def test_new_run_rejects_existing_scope_root_without_manifest(
    tmp_path: Path,
    with_sentinel: bool,
) -> None:
    run_dir = tmp_path / "run" / "core"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "sentinel.txt"
    if with_sentinel:
        sentinel.write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(builder.ManuscriptBuildError, match="orphan|partial|manifest"):
        builder._select_start_manifest(
            {"run_id": "new-run", "git_worktree_dirty": False, "status": "running"},
            requested_run_id="new-run",
            reuse_compatible=True,
            manifest_path=run_dir / "run_manifest.json",
            run_dir=run_dir,
            config_hash="a" * 64,
            evidence_scope="core",
        )

    assert run_dir.is_dir()
    if with_sentinel:
        assert sentinel.read_text(encoding="utf-8") == "preserve me\n"


def test_other_scope_parent_does_not_block_absent_exact_scope(tmp_path: Path) -> None:
    package = tmp_path / "run"
    (package / "supplementary").mkdir(parents=True)
    run_dir = package / "core"
    provisional = {"run_id": "new-run", "git_worktree_dirty": False, "status": "running"}

    selected, existing = builder._select_start_manifest(
        provisional,
        requested_run_id="new-run",
        reuse_compatible=True,
        manifest_path=run_dir / "run_manifest.json",
        run_dir=run_dir,
        config_hash="a" * 64,
        evidence_scope="core",
    )

    assert selected is provisional
    assert existing is False
    assert not run_dir.exists()


def test_persisted_failure_message_removes_user_absolute_paths() -> None:
    raw = (
        r"failed at C:\Users\Researcher\private\artifact.csv "
        "and /home/researcher/private/output.json"
    )
    sanitized = builder._sanitized_failure_message(raw)

    assert r"C:\Users" not in sanitized
    assert "/home/researcher" not in sanitized
    assert "<user-path>" in sanitized
