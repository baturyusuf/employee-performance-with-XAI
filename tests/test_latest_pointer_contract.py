from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from src.experiments import build_manuscript_evidence as builder


RUN_ID = "pointer-contract-run"
CONFIG_HASH = "a" * 64
SCIENTIFIC_INPUT_HASH = "b" * 64
SCOPE_CONTRACT_HASH = "c" * 64
GIT_COMMIT = "d" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, _sha256(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _complete_scope(output_root: Path, scope: str, *, run_id: str = RUN_ID) -> Path:
    run_dir = output_root / run_id / scope
    stage = "core_tables" if scope == "core" else "supplementary_tables"
    evidence = run_dir / stage / "result.csv"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")

    builder.build_final_evidence_manifest(
        run_dir,
        run_id=run_id,
        config_hash=CONFIG_HASH,
        evidence_scope=scope,
        scope_contract_hash=SCOPE_CONTRACT_HASH,
        git_commit=GIT_COMMIT,
        source_tree_hash="e" * 64,
        scientific_input_hash=SCIENTIFIC_INPUT_HASH,
    )
    final_manifest = run_dir / "final_evidence_manifest.json"
    run_manifest = run_dir / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "run_id": run_id,
                "config_hash": CONFIG_HASH,
                "scientific_input_hash": SCIENTIFIC_INPUT_HASH,
                "scope_contract_hash": SCOPE_CONTRACT_HASH,
                "evidence_scope": scope,
                "git_commit": GIT_COMMIT,
                "source_tree_hash": "e" * 64,
                "config_path": "configs/manuscript_final.yaml",
                "manifest_schema_version": 3,
                "code_package_versions": {"python": "3.test"},
                "random_seeds": {"model": 42},
                "final_evidence_manifest": "final_evidence_manifest.json",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def _complete_package(output_root: Path, *, run_id: str = RUN_ID) -> Path:
    _complete_scope(output_root, "core", run_id=run_id)
    _complete_scope(output_root, "supplementary", run_id=run_id)
    return output_root / run_id


def _strict_complete_scope(output_root: Path, scope: str) -> Path:
    """Create a small package that exercises the production strict validator."""

    run_dir = output_root / RUN_ID / scope
    run_dir.mkdir(parents=True)
    scope_contract = {"stages": list(builder.CANONICAL_STAGE_ORDERS[scope])}
    identity = {
        "manifest_schema_version": 3,
        "run_id": RUN_ID,
        "config_path": "configs/manuscript_final.yaml",
        "config_hash": CONFIG_HASH,
        "evidence_scope": scope,
        "scope_contract": scope_contract,
        "scope_contract_hash": SCOPE_CONTRACT_HASH,
        "git_commit": GIT_COMMIT,
        "source_tree_hash": "e" * 64,
        "scientific_input_hash": SCIENTIFIC_INPUT_HASH,
        "dataset_hashes": {},
        "actual_input_receipts": {},
        "side_input_hashes": {},
        "code_package_versions": {"python": "3.test"},
        "random_seeds": {"model": 42},
    }
    context = builder.StageContext(
        config_path=builder.PROJECT_ROOT / identity["config_path"],
        config={},
        settings={},
        run_dir=run_dir,
        run_id=RUN_ID,
        config_hash=CONFIG_HASH,
        manifest=identity,
        evidence_scope=scope,
        scope_contract=scope_contract,
    )
    for stage in builder.CANONICAL_STAGE_ORDERS[scope]:
        artifact = run_dir / stage / "result.txt"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(f"verified {stage}\n", encoding="utf-8")
        builder._write_stage_metadata(
            context,
            stage,
            [artifact],
            started_at="2026-07-13T00:00:00+00:00",
            elapsed_seconds=1.0,
        )

    input_contract = run_dir / "run_inputs" / "input_contract.json"
    input_contract.parent.mkdir()
    input_contract.write_text('{"status":"complete"}\n', encoding="utf-8")
    builder._write_claim_report(context)
    builder._write_package_status(context)
    final_paths = builder.build_final_evidence_manifest(
        run_dir,
        run_id=RUN_ID,
        config_hash=CONFIG_HASH,
        evidence_scope=scope,
        scope_contract_hash=SCOPE_CONTRACT_HASH,
        git_commit=GIT_COMMIT,
        source_tree_hash="e" * 64,
        scientific_input_hash=SCIENTIFIC_INPUT_HASH,
    )
    final_payload = json.loads(final_paths["json"].read_text(encoding="utf-8"))

    def output_record(relative: str, stage: str) -> dict[str, object]:
        target = run_dir / relative
        return {
            "path": target.relative_to(builder.PROJECT_ROOT).as_posix(),
            "stage": stage,
            "size_bytes": target.stat().st_size,
            "sha256": _sha256(target),
        }

    manifest = {
        **identity,
        "status": "complete",
        "input_contract_snapshot": input_contract.relative_to(builder.PROJECT_ROOT).as_posix(),
        "commands": [
            {
                "command": (
                    "python -m src.experiments.build_manuscript_evidence "
                    f"--config configs/manuscript_final.yaml --scope {scope} --run-id {RUN_ID}"
                ),
                "stage": "entrypoint",
                "status": "complete",
                "return_code": 0,
            },
            *(
                {
                    "command": f"internal-stage:{stage}",
                    "stage": stage,
                    "status": "complete",
                    "return_code": 0,
                }
                for stage in builder.CANONICAL_STAGE_ORDERS[scope]
            ),
        ],
        "output_files": [
            *(output_record(row["path"], row["stage"]) for row in final_payload["files"]),
            output_record("final_evidence_manifest.csv", "final_manifest"),
            output_record("final_evidence_manifest.json", "final_manifest"),
        ],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _promotion_callable():
    promote = getattr(builder, "promote_latest_pointer", None)
    assert callable(promote), (
        "Pointer promotion must be a separate public operation named "
        "promote_latest_pointer; the scientific build must not mutate latest."
    )
    return promote


def _install_strict_package_validator(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def validate(manifest, *, run_dir, config_hash, evidence_scope, **_kwargs):
        if manifest.get("status") != "complete":
            raise builder.ManuscriptBuildError("Complete package run manifest is invalid.")
        builder.validate_final_evidence_manifest(
            run_dir / "final_evidence_manifest.json",
            run_dir=run_dir,
            expected_run_id=str(manifest["run_id"]),
            expected_config_hash=config_hash,
            expected_evidence_scope=evidence_scope,
            expected_scope_contract_hash=str(manifest["scope_contract_hash"]),
            expected_git_commit=GIT_COMMIT,
            expected_source_tree_hash="e" * 64,
            expected_scientific_input_hash=SCIENTIFIC_INPUT_HASH,
        )
        calls.append(evidence_scope)
        return {"status": "passed"}

    monkeypatch.setattr(builder, "_validate_completed_run_package", validate)
    return calls


def _called_function_names(function) -> set[str]:
    tree = ast.parse(inspect.getsource(function))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def test_scientific_build_does_not_publish_or_mutate_latest() -> None:
    calls = _called_function_names(builder.build)
    assert "_update_latest_pointer" not in calls
    assert "promote_latest_pointer" not in calls
    assert "_validate_completed_run_package" in _called_function_names(
        builder.promote_latest_pointer
    )


def test_separate_promotion_writes_only_portable_pointer_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "reports" / "manuscript_final"
    run_root = _complete_package(output_root)
    before = _inventory(run_root)
    calls = _install_strict_package_validator(monkeypatch)

    pointer_path = _promotion_callable()(run_root, output_root)

    assert pointer_path == output_root / "latest" / "pointer.json"
    assert pointer_path.is_file()
    assert _inventory(run_root) == before
    assert calls == ["core", "supplementary"]
    assert {
        path.relative_to(output_root / "latest").as_posix()
        for path in (output_root / "latest").rglob("*")
        if path.is_file()
    } == {"pointer.json"}

    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["pointer_kind"] == "manuscript_evidence_run"
    assert payload["status"] == "complete"
    assert payload["run_id"] == RUN_ID
    assert payload["config_hash"] == CONFIG_HASH
    assert payload["git_commit"] == GIT_COMMIT
    assert payload["config_path"] == "configs/manuscript_final.yaml"
    assert payload["random_seeds"] == {"model": 42}

    relative_target = Path(payload["relative_target"])
    assert not relative_target.is_absolute()
    assert (pointer_path.parent / relative_target).resolve() == run_root.resolve()
    assert set(payload["scopes"]) == {"core", "supplementary"}
    for scope in ("core", "supplementary"):
        scope_dir = run_root / scope
        scope_payload = payload["scopes"][scope]
        assert scope_payload["scientific_input_hash"] == SCIENTIFIC_INPUT_HASH
        assert scope_payload["scope_contract_hash"] == SCOPE_CONTRACT_HASH
        for key, expected in (
            ("run_manifest", scope_dir / "run_manifest.json"),
            ("final_evidence_manifest", scope_dir / "final_evidence_manifest.json"),
        ):
            reference = Path(scope_payload[key]["path"])
            assert not reference.is_absolute()
            resolved = (pointer_path.parent / reference).resolve()
            assert resolved == expected.resolve()
            assert scope_payload[key]["sha256"] == _sha256(expected)


def test_promotion_refuses_to_move_or_overwrite_historical_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "reports" / "manuscript_final"
    run_dir = _complete_package(output_root)
    historical = output_root / "latest" / "historical_result.csv"
    historical.parent.mkdir(parents=True)
    historical.write_text("historical,evidence\n", encoding="utf-8")
    before = _inventory(output_root / "latest")
    _install_strict_package_validator(monkeypatch)

    with pytest.raises(builder.ManuscriptBuildError, match="historical|migration|pointer-only"):
        _promotion_callable()(run_dir, output_root)

    assert _inventory(output_root / "latest") == before
    assert historical.read_text(encoding="utf-8") == "historical,evidence\n"


def test_promotion_rejects_a_rogue_empty_directory_under_pointer_only_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "reports" / "manuscript_final"
    run_dir = _complete_package(output_root)
    _install_strict_package_validator(monkeypatch)
    pointer = _promotion_callable()(run_dir, output_root)
    before = pointer.read_bytes()
    (output_root / "latest" / "rogue_empty").mkdir()

    with pytest.raises(builder.ManuscriptBuildError, match="historical|pointer-only"):
        _promotion_callable()(run_dir, output_root)

    assert pointer.read_bytes() == before
    assert (output_root / "latest" / "rogue_empty").is_dir()


def test_promotion_rejects_incomplete_or_identity_inconsistent_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "reports" / "manuscript_final"
    run_dir = _complete_package(output_root)
    manifest_path = run_dir / "core" / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    _install_strict_package_validator(monkeypatch)
    payload["status"] = "running"
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(builder.ManuscriptBuildError, match="complete"):
        _promotion_callable()(run_dir, output_root)
    assert not (output_root / "latest").exists()

    payload["status"] = "complete"
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    final_path = run_dir / "supplementary" / "final_evidence_manifest.json"
    final_payload = json.loads(final_path.read_text(encoding="utf-8"))
    final_payload["run_id"] = "wrong-run"
    final_path.write_text(json.dumps(final_payload) + "\n", encoding="utf-8")

    with pytest.raises(builder.ManuscriptBuildError, match="identity|run_id"):
        _promotion_callable()(run_dir, output_root)
    assert not (output_root / "latest").exists()


def test_promotion_rejects_rogue_package_root_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "reports" / "manuscript_final"
    run_root = _complete_package(output_root)
    rogue = run_root / "rogue.bin"
    rogue.write_bytes(b"not part of either scope")
    _install_strict_package_validator(monkeypatch)

    with pytest.raises(builder.ManuscriptBuildError, match="rogue|entry"):
        _promotion_callable()(run_root, output_root)
    assert rogue.is_file()
    assert not (output_root / "latest").exists()


def test_promotion_rejects_cross_scope_runtime_or_seed_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "reports" / "manuscript_final"
    run_root = _complete_package(output_root)
    supplementary = run_root / "supplementary" / "run_manifest.json"
    payload = json.loads(supplementary.read_text(encoding="utf-8"))
    payload["random_seeds"] = {"model": 99}
    supplementary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    _install_strict_package_validator(monkeypatch)

    with pytest.raises(builder.ManuscriptBuildError, match="identities differ"):
        _promotion_callable()(run_root, output_root)
    assert not (output_root / "latest").exists()


def test_promotion_runs_the_production_strict_validator_for_both_scopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Promotion must traverse strict stage/final/claim/status validation, not a stub."""

    output_root = tmp_path / "reports" / "manuscript_final"
    config_path = tmp_path / "configs" / "manuscript_final.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("manuscript_final: {}\n", encoding="utf-8")
    settings = {
        "evidence_scopes": {
            scope: {
                "stages": list(builder.CANONICAL_STAGE_ORDERS[scope]),
                "release_ready": True,
            }
            for scope in builder.CANONICAL_STAGE_ORDERS
        },
        "output": {
            "root": "reports/manuscript_final",
            "manifest_filename": "run_manifest.json",
        }
    }
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(builder, "load_manuscript_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(builder, "manuscript_settings", lambda _config: settings)
    monkeypatch.setattr(builder, "validate_scope_release_ready", lambda scopes, scope: scopes[scope])
    monkeypatch.setattr(builder, "validate_run_manifest", lambda *args, **kwargs: {})
    monkeypatch.setattr(builder, "_validate_resume_worktree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builder, "_validate_primary_artifacts", lambda _context: None)
    monkeypatch.setattr(
        builder,
        "_validate_input_snapshot_contract",
        lambda _context, root: [root / "input_contract.json"],
    )
    _strict_complete_scope(output_root, "core")
    _strict_complete_scope(output_root, "supplementary")

    production_validator = builder._validate_completed_run_package
    calls: list[str] = []

    def tracked_validator(*args, **kwargs):
        calls.append(str(kwargs["evidence_scope"]))
        return production_validator(*args, **kwargs)

    monkeypatch.setattr(builder, "_validate_completed_run_package", tracked_validator)
    core_manifest_path = output_root / RUN_ID / "core" / "run_manifest.json"
    core_manifest = json.loads(core_manifest_path.read_text(encoding="utf-8"))
    core_manifest["commands"][1], core_manifest["commands"][2] = (
        core_manifest["commands"][2],
        core_manifest["commands"][1],
    )
    core_manifest_path.write_text(json.dumps(core_manifest) + "\n", encoding="utf-8")
    with pytest.raises(builder.ManuscriptBuildError, match="canonical execution order"):
        builder.promote_latest_pointer(output_root / RUN_ID, output_root)
    core_manifest["commands"][1], core_manifest["commands"][2] = (
        core_manifest["commands"][2],
        core_manifest["commands"][1],
    )
    core_manifest_path.write_text(json.dumps(core_manifest) + "\n", encoding="utf-8")
    calls.clear()
    pointer = builder.promote_latest_pointer(output_root / RUN_ID, output_root)

    assert calls == ["core", "supplementary"]
    assert pointer.is_file()
    assert json.loads(pointer.read_text(encoding="utf-8"))["run_id"] == RUN_ID


def test_configured_promotion_and_cli_derive_the_canonical_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "configs" / "manuscript_final.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("manuscript_final: {}\n", encoding="utf-8")
    settings = {
        "output": {
            "root": "reports/manuscript_final",
            "manifest_filename": "run_manifest.json",
        }
    }
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(builder, "load_manuscript_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(builder, "manuscript_settings", lambda _config: settings)
    calls: list[tuple[Path, Path]] = []

    def promote(run_dir: Path, output_root: Path) -> Path:
        calls.append((run_dir, output_root))
        pointer = output_root / "latest" / "pointer.json"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text("{}\n", encoding="utf-8")
        return pointer

    monkeypatch.setattr(builder, "promote_latest_pointer", promote)
    pointer = builder.promote_configured_latest(config_path, run_id=RUN_ID)
    expected_root = tmp_path / "reports" / "manuscript_final"
    assert calls == [(expected_root / RUN_ID, expected_root)]
    assert pointer == expected_root / "latest" / "pointer.json"

    builder.main(
        [
            "--config",
            "configs/manuscript_final.yaml",
            "--promote-run-id",
            RUN_ID,
        ]
    )
    assert len(calls) == 2
    output = json.loads(capsys.readouterr().out)
    assert Path(output["latest_pointer"]) == pointer
