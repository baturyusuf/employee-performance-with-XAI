from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.experiments import build_manuscript_evidence as builder
from src.governance.manuscript_contract import RunManifestError, validate_run_manifest
from tests import test_artifact_run_manifest_consistency as _artifact_manifest_tests


TERMINAL_TIMESTAMP = "2026-07-13T00:00:01+00:00"
FINAL_GIT_COMMIT = "d" * 40
FINAL_SOURCE_TREE_HASH = "e" * 64
FINAL_SCIENTIFIC_INPUT_HASH = "f" * 64


def _complete_manifest(root: Path) -> dict:
    """Return an otherwise terminal complete manifest around real temporary bytes."""

    _, _, manifest = _artifact_manifest_tests.ArtifactRunManifestConsistencyTests()._fixture(root)
    manifest["git_worktree_dirty"] = False
    manifest["commands"] = [
        {
            "command": "python -m src.experiments.build_manuscript_evidence --scope core",
            "stage": "entrypoint",
            "status": "complete",
            "started_at": "2026-07-13T00:00:00+00:00",
            "ended_at": TERMINAL_TIMESTAMP,
            "return_code": 0,
            "elapsed_seconds": 1.0,
        },
        {
            "command": "internal-stage:fixture_core",
            "stage": "fixture_core",
            "status": "complete",
            "started_at": "2026-07-13T00:00:00+00:00",
            "ended_at": TERMINAL_TIMESTAMP,
            "return_code": 0,
            "elapsed_seconds": 1.0,
        },
    ]
    manifest["output_files"][0]["stage"] = "fixture_core"
    return manifest


def _failed_manifest(root: Path) -> dict:
    manifest = _complete_manifest(root)
    manifest["status"] = "failed"
    manifest["end_timestamp"] = TERMINAL_TIMESTAMP
    manifest["failure_information"] = [
        {
            "timestamp": TERMINAL_TIMESTAMP,
            "stage": "fixture_core",
            "error_type": "RuntimeError",
            "message": "intentional contract fixture failure",
        }
    ]
    entrypoint = manifest["commands"][0]
    entrypoint.update(status="failed", ended_at=TERMINAL_TIMESTAMP, return_code=1)
    return manifest


def _build_final_evidence(root: Path, *, empty_artifact: bool = False) -> dict[str, Path]:
    artifact = root / "policy_ablation" / "evidence.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("" if empty_artifact else "metric,value\nmacro_f1,0.5\n", encoding="utf-8")
    return builder.build_final_evidence_manifest(
        root,
        run_id="completion-contract",
        config_hash="a" * 64,
        evidence_scope="core",
        scope_contract_hash="b" * 64,
        git_commit=FINAL_GIT_COMMIT,
        source_tree_hash=FINAL_SOURCE_TREE_HASH,
        scientific_input_hash=FINAL_SCIENTIFIC_INPUT_HASH,
    )


def _validate_final_evidence(paths: dict[str, Path], root: Path) -> dict:
    return builder.validate_final_evidence_manifest(
        paths["json"],
        run_dir=root,
        expected_run_id="completion-contract",
        expected_config_hash="a" * 64,
        expected_evidence_scope="core",
        expected_scope_contract_hash="b" * 64,
        expected_git_commit=FINAL_GIT_COMMIT,
        expected_source_tree_hash=FINAL_SOURCE_TREE_HASH,
        expected_scientific_input_hash=FINAL_SCIENTIFIC_INPUT_HASH,
    )


def test_complete_manifest_requires_nonempty_registered_outputs(tmp_path: Path) -> None:
    manifest = _complete_manifest(tmp_path)
    manifest["output_files"] = []

    with pytest.raises(RunManifestError, match="output|artifact"):
        validate_run_manifest(manifest, project_root=tmp_path, require_complete=True)


@pytest.mark.parametrize(
    ("status", "ended_at", "return_code"),
    [
        ("started", None, None),
        ("complete", None, 0),
        ("complete", TERMINAL_TIMESTAMP, None),
        ("complete", TERMINAL_TIMESTAMP, 1),
    ],
)
def test_complete_manifest_requires_terminal_successful_entrypoint(
    tmp_path: Path,
    status: str,
    ended_at: str | None,
    return_code: int | None,
) -> None:
    manifest = _complete_manifest(tmp_path)
    manifest["commands"][0].update(
        status=status,
        ended_at=ended_at,
        return_code=return_code,
    )

    with pytest.raises(RunManifestError, match="entrypoint|command|return_code"):
        validate_run_manifest(manifest, project_root=tmp_path, require_complete=True)


def test_complete_manifest_rejects_unfinished_stage_command(tmp_path: Path) -> None:
    manifest = _complete_manifest(tmp_path)
    manifest["commands"][1].update(status="started", ended_at=None, return_code=None)

    with pytest.raises(RunManifestError, match="fixture_core|command|started"):
        validate_run_manifest(manifest, project_root=tmp_path, require_complete=True)


def test_complete_manifest_requires_clean_starting_worktree(tmp_path: Path) -> None:
    manifest = _complete_manifest(tmp_path)
    manifest["git_worktree_dirty"] = True

    with pytest.raises(RunManifestError, match="clean|git_worktree_dirty|dirty"):
        validate_run_manifest(manifest, project_root=tmp_path, require_complete=True)


def test_failed_manifest_requires_terminal_failed_entrypoint(tmp_path: Path) -> None:
    manifest = _failed_manifest(tmp_path)
    manifest["commands"][0].update(status="started", ended_at=None, return_code=None)

    with pytest.raises(RunManifestError, match="entrypoint|command|started"):
        validate_run_manifest(manifest, project_root=tmp_path)


def test_failed_manifest_requires_failure_information(tmp_path: Path) -> None:
    manifest = _failed_manifest(tmp_path)
    manifest["failure_information"] = []

    with pytest.raises(RunManifestError, match="failure_information|failure"):
        validate_run_manifest(manifest, project_root=tmp_path)


def test_failed_manifest_requires_nonzero_entrypoint_return_code(tmp_path: Path) -> None:
    manifest = _failed_manifest(tmp_path)
    manifest["commands"][0]["return_code"] = 0

    with pytest.raises(RunManifestError, match="entrypoint|return_code|failed"):
        validate_run_manifest(manifest, project_root=tmp_path)


@pytest.mark.parametrize(
    ("prior_status", "ended_at", "return_code"),
    [
        ("complete", TERMINAL_TIMESTAMP, 0),
        ("started", None, None),
    ],
)
def test_failed_manifest_rejects_noninterrupted_prior_entrypoint(
    tmp_path: Path,
    prior_status: str,
    ended_at: str | None,
    return_code: int | None,
) -> None:
    manifest = _failed_manifest(tmp_path)
    manifest["commands"].insert(
        0,
        {
            "command": "python -m src.experiments.build_manuscript_evidence --scope core",
            "stage": "entrypoint",
            "status": prior_status,
            "started_at": "2026-07-13T00:00:00+00:00",
            "ended_at": ended_at,
            "return_code": return_code,
        },
    )

    with pytest.raises(RunManifestError, match="entrypoint|interrupted|started"):
        validate_run_manifest(manifest, project_root=tmp_path)


def test_failed_manifest_allows_prior_interrupted_entrypoint(tmp_path: Path) -> None:
    manifest = _failed_manifest(tmp_path)
    manifest["commands"].insert(
        0,
        {
            "command": "python -m src.experiments.build_manuscript_evidence --scope core",
            "stage": "entrypoint",
            "status": "interrupted",
            "started_at": "2026-07-13T00:00:00+00:00",
            "ended_at": TERMINAL_TIMESTAMP,
            "return_code": None,
        },
    )

    validated = validate_run_manifest(manifest, project_root=tmp_path)
    assert validated["status"] == "failed"


def test_final_evidence_manifest_rejects_unmanifested_files(tmp_path: Path) -> None:
    paths = _build_final_evidence(tmp_path)
    (tmp_path / "policy_ablation" / "unmanifested.csv").write_text(
        "metric,value\nmacro_f1,0.9\n",
        encoding="utf-8",
    )

    with pytest.raises(builder.ManuscriptBuildError, match="unmanifested|file set|unlisted"):
        _validate_final_evidence(paths, tmp_path)


@pytest.mark.parametrize(
    "relative",
    ["policy_ablation/.partial.tmp", "policy_ablation/__pycache__/cached.pyc"],
)
def test_final_evidence_manifest_rejects_hidden_or_cache_files(
    tmp_path: Path,
    relative: str,
) -> None:
    paths = _build_final_evidence(tmp_path)
    extra = tmp_path / relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("unmanifested\n", encoding="utf-8")

    with pytest.raises(
        builder.ManuscriptBuildError,
        match="unmanifested|file set|scope path|hidden|cache",
    ):
        _validate_final_evidence(paths, tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hash_algorithm", "md5", "hash_algorithm|sha256"),
        ("n_files", 999, "n_files|file count"),
        ("git_commit", "0" * 40, "git_commit"),
        ("source_tree_hash", "0" * 64, "source_tree_hash"),
        ("scientific_input_hash", "0" * 64, "scientific_input_hash"),
    ],
)
def test_final_evidence_manifest_rejects_header_contract_tampering(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    paths = _build_final_evidence(tmp_path)
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    payload[field] = value
    paths["json"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(builder.ManuscriptBuildError, match=message):
        _validate_final_evidence(paths, tmp_path)


def test_final_evidence_manifest_rejects_zero_byte_artifact(tmp_path: Path) -> None:
    paths = _build_final_evidence(tmp_path, empty_artifact=True)

    with pytest.raises(builder.ManuscriptBuildError, match="empty|zero|size"):
        _validate_final_evidence(paths, tmp_path)


def test_final_evidence_manifest_requires_csv_json_parity(tmp_path: Path) -> None:
    paths = _build_final_evidence(tmp_path)
    paths["csv"].write_text(
        "run_id,config_hash,evidence_scope,scope_contract_hash,stage,path,file_type,size_bytes,sha256\n",
        encoding="utf-8",
    )

    with pytest.raises(builder.ManuscriptBuildError, match="CSV|csv|parity"):
        _validate_final_evidence(paths, tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("stage", "paid_llm", "stage/path"),
        ("file_type", "fabricated", "file-type"),
    ],
)
def test_final_evidence_manifest_rejects_row_semantic_tampering_even_with_csv_parity(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    paths = _build_final_evidence(tmp_path)
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    payload["files"][0][field] = value
    paths["json"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(payload["files"][0]))
        writer.writeheader()
        writer.writerows(payload["files"])

    with pytest.raises(builder.ManuscriptBuildError, match=message):
        _validate_final_evidence(paths, tmp_path)


def test_clean_existing_run_is_rejected_when_reuse_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        builder,
        "_load_existing_manifest",
        lambda *args, **kwargs: {"status": "complete", "run_id": "completion-contract"},
    )

    with pytest.raises(builder.ManuscriptBuildError, match="reuse|existing|new run ID"):
        builder._select_start_manifest(
            {"git_worktree_dirty": False, "run_id": "completion-contract"},
            requested_run_id="completion-contract",
            reuse_compatible=False,
            manifest_path=manifest_path,
            run_dir=tmp_path,
            config_hash="a" * 64,
            evidence_scope="core",
        )


def test_completed_run_reuse_validates_the_final_package_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "manuscript_final.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "reports" / "manuscript_final"
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(builder, "_validate_resume_worktree", lambda _run_dir: None)
    settings = {
        "evidence_scopes": {"core": {}},
        "output": {
            "root": "reports/manuscript_final",
            "manifest_filename": "run_manifest.json",
        },
    }
    monkeypatch.setattr(builder, "load_manuscript_config", lambda path: {})
    monkeypatch.setattr(builder, "manuscript_settings", lambda config: settings)
    monkeypatch.setattr(builder, "canonical_config_hash", lambda config: "a" * 64)
    monkeypatch.setattr(
        builder,
        "evidence_scope_contract",
        lambda config, scope: {"stages": list(builder.CORE_STAGE_ORDER)},
    )
    monkeypatch.setattr(builder, "validate_scope_release_ready", lambda scopes, scope: {})
    monkeypatch.setattr(
        builder,
        "create_run_manifest",
        lambda *args, **kwargs: {
            "run_id": "completion-contract",
            "status": "running",
            "git_worktree_dirty": False,
        },
    )
    monkeypatch.setattr(
        builder,
        "_select_start_manifest",
        lambda *args, **kwargs: (
            {
                "run_id": "completion-contract",
                "status": "complete",
                "git_worktree_dirty": False,
            },
            True,
        ),
    )

    with pytest.raises(builder.ManuscriptBuildError, match="final|package|manifest"):
        builder.build(
            config_path,
            run_id="completion-contract",
            reuse_compatible=True,
            evidence_scope="core",
        )


def test_strict_completed_package_revalidates_stage_receipts_and_final_registrations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(builder, "CANONICAL_STAGE_ORDERS", {"core": ("policy_ablation",)})
    monkeypatch.setattr(builder, "_validate_resume_worktree", lambda _root: None)
    monkeypatch.setattr(builder, "validate_run_manifest", lambda *args, **kwargs: {})
    monkeypatch.setattr(builder, "_validate_input_snapshot_contract", lambda *args: [])
    monkeypatch.setattr(builder, "_validate_package_status_contract", lambda *args: {})
    monkeypatch.setattr(builder, "_validate_claim_boundary_contract", lambda *args: None)
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "policy_ablation"
    stage_dir.mkdir(parents=True)
    artifact = stage_dir / "result.csv"
    artifact.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")
    identity = {
        "run_id": "strict-package",
        "config_hash": "a" * 64,
        "evidence_scope": "core",
        "scope_contract": {},
        "scope_contract_hash": "b" * 64,
        "git_commit": "c" * 40,
        "source_tree_hash": "d" * 64,
        "scientific_input_hash": "e" * 64,
        "dataset_hashes": {},
        "actual_input_receipts": {},
        "side_input_hashes": {},
    }
    config_path = tmp_path / "configs" / "manuscript_final.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(builder, "load_manuscript_config", lambda *args, **kwargs: {})
    strict_settings = {
        "evidence_scopes": {
            "core": {
                "stages": ["policy_ablation"],
                "release_ready": True,
            }
        }
    }
    monkeypatch.setattr(builder, "manuscript_settings", lambda _config: strict_settings)
    def validate_fixture_release(scopes, scope):
        definition = scopes[scope]
        if definition.get("release_ready") is not True:
            raise builder.ManuscriptBuildError(
                f"Evidence scope {scope!r} is not release-ready: "
                f"{definition.get('blocking_reason', 'pending')}"
            )
        return definition

    monkeypatch.setattr(builder, "validate_scope_release_ready", validate_fixture_release)
    receipt = {
        "stage": "policy_ablation",
        "status": "complete",
        "inventory_mode": "closed_world",
        "path_basis": "stage_relative",
        "started_at": "2026-07-13T00:00:00+00:00",
        "ended_at": "2026-07-13T00:00:01+00:00",
        "elapsed_seconds": 1.0,
        **identity,
        "outputs": [
            {
                "path": "result.csv",
                "sha256": builder.sha256_file(artifact),
                "size_bytes": artifact.stat().st_size,
            }
        ],
    }
    receipt_path = stage_dir / "stage_contract.json"
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    (run_dir / "canonical_claim_boundaries.md").write_text("claims\n", encoding="utf-8")
    (run_dir / "package_status.json").write_text("{}\n", encoding="utf-8")
    final = builder.build_final_evidence_manifest(
        run_dir,
        run_id=identity["run_id"],
        config_hash=identity["config_hash"],
        evidence_scope="core",
        scope_contract_hash=identity["scope_contract_hash"],
        git_commit=identity["git_commit"],
        source_tree_hash=identity["source_tree_hash"],
        scientific_input_hash=identity["scientific_input_hash"],
    )
    final_payload = json.loads(final["json"].read_text(encoding="utf-8"))

    def output_record(relative: str, stage: str) -> dict:
        target = run_dir / relative
        return {
            "path": target.relative_to(tmp_path).as_posix(),
            "sha256": builder.sha256_file(target),
            "size_bytes": target.stat().st_size,
            "stage": stage,
        }

    manifest = {
        **identity,
        "status": "complete",
        "config_path": "configs/manuscript_final.yaml",
        "input_contract_snapshot": "run/run_inputs/input_contract.json",
        "commands": [
            {
                "command": (
                    "python -m src.experiments.build_manuscript_evidence "
                    "--config configs/manuscript_final.yaml --scope core --run-id strict-package"
                ),
                "stage": "entrypoint",
                "status": "complete",
                "return_code": 0,
            },
            {
                "command": "internal-stage:policy_ablation",
                "stage": "policy_ablation",
                "status": "complete",
                "return_code": 0,
            }
        ],
        "output_files": [
            *(
                output_record(
                    row["path"],
                    "integration"
                    if row["path"] in {"canonical_claim_boundaries.md", "package_status.json"}
                    else row["stage"],
                )
                for row in final_payload["files"]
            ),
            output_record("final_evidence_manifest.csv", "final_manifest"),
            output_record("final_evidence_manifest.json", "final_manifest"),
        ],
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    strict_settings["evidence_scopes"]["core"]["release_ready"] = False
    strict_settings["evidence_scopes"]["core"]["blocking_reason"] = "technical freeze pending"
    with pytest.raises(builder.ManuscriptBuildError, match="not release-ready"):
        builder._validate_completed_run_package(
            manifest,
            manifest_path=manifest_path,
            run_dir=run_dir,
            config_hash=identity["config_hash"],
            evidence_scope="core",
            enforce_configured_layout=False,
        )
    strict_settings["evidence_scopes"]["core"]["release_ready"] = True

    result = builder._validate_completed_run_package(
        manifest,
        manifest_path=manifest_path,
        run_dir=run_dir,
        config_hash=identity["config_hash"],
        evidence_scope="core",
        enforce_configured_layout=False,
    )
    assert result["status"] == "passed"

    invalid_entrypoint = json.loads(json.dumps(manifest))
    invalid_entrypoint["commands"][0]["command"] = "not-a-reproduction-command"
    with pytest.raises(builder.ManuscriptBuildError, match="invalid or prohibited commands"):
        builder._validate_completed_run_package(
            invalid_entrypoint,
            manifest_path=manifest_path,
            run_dir=run_dir,
            config_hash=identity["config_hash"],
            evidence_scope="core",
            enforce_configured_layout=False,
        )

    extra_stage = json.loads(json.dumps(manifest))
    extra_stage["commands"].append(
        {
            "command": "internal-stage:paid_llm",
            "stage": "paid_llm",
            "status": "complete",
            "return_code": 0,
        }
    )
    with pytest.raises(builder.ManuscriptBuildError, match="noncanonical command stages"):
        builder._validate_completed_run_package(
            extra_stage,
            manifest_path=manifest_path,
            run_dir=run_dir,
            config_hash=identity["config_hash"],
            evidence_scope="core",
            enforce_configured_layout=False,
        )

    receipt["stage"] = "wrong_stage"
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    with pytest.raises(builder.ManuscriptBuildError, match="invalid stage contracts"):
        builder._validate_completed_run_package(
            manifest,
            manifest_path=manifest_path,
            run_dir=run_dir,
            config_hash=identity["config_hash"],
            evidence_scope="core",
            enforce_configured_layout=False,
        )

    receipt["stage"] = "policy_ablation"
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    manifest["output_files"] = [
        record
        for record in manifest["output_files"]
        if not str(record["path"]).endswith("final_evidence_manifest.json")
    ]
    with pytest.raises(builder.ManuscriptBuildError, match="Both final evidence manifests"):
        builder._validate_completed_run_package(
            manifest,
            manifest_path=manifest_path,
            run_dir=run_dir,
            config_hash=identity["config_hash"],
            evidence_scope="core",
            enforce_configured_layout=False,
        )


def test_package_status_and_claim_boundaries_are_semantically_validated(tmp_path: Path) -> None:
    run_dir = tmp_path / "core"
    run_dir.mkdir()
    manifest = {"scope_contract_hash": "b" * 64}
    context = builder.StageContext(
        config_path=tmp_path / "config.json",
        config={},
        settings={},
        run_dir=run_dir,
        run_id="semantic-package",
        config_hash="a" * 64,
        manifest=manifest,
        evidence_scope="core",
        scope_contract={"stages": list(builder.CORE_STAGE_ORDER)},
    )
    for stage in builder.CORE_STAGE_ORDER:
        artifact = run_dir / stage / "result.txt"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(f"{stage}\n", encoding="utf-8")
        builder._write_stage_metadata(
            context,
            stage,
            [artifact],
            started_at="2026-07-13T00:00:00+00:00",
            elapsed_seconds=1.0,
        )
    status_path = builder._write_package_status(context)
    builder._write_claim_report(context)

    builder._validate_package_status_contract(context)
    builder._validate_claim_boundary_contract(context)

    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["paid_api_calls"] = 1
    status_path.write_text(
        json.dumps(status) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(builder.ManuscriptBuildError, match="paid_api_calls|semantic"):
        builder._validate_package_status_contract(context)

    status = json.loads(builder._write_package_status(context).read_text(encoding="utf-8"))
    status["stages"][0]["elapsed_seconds"] = float("nan")
    status_path.write_text(json.dumps(status) + "\n", encoding="utf-8")
    with pytest.raises(builder.ManuscriptBuildError, match="semantic|runtime"):
        builder._validate_package_status_contract(context)

    status = json.loads(builder._write_package_status(context).read_text(encoding="utf-8"))
    status["stages"][0]["n_outputs"] = 999999
    status_path.write_text(json.dumps(status) + "\n", encoding="utf-8")
    with pytest.raises(builder.ManuscriptBuildError, match="semantic"):
        builder._validate_package_status_contract(context)

    builder._write_package_status(context)
    claim_path = run_dir / "canonical_claim_boundaries.md"
    claim_path.write_text(
        claim_path.read_text(encoding="utf-8")
        + "Autonomous HR decisions are allowed. SHAP proves causality.\n",
        encoding="utf-8",
    )
    with pytest.raises(builder.ManuscriptBuildError, match="exact configured scope"):
        builder._validate_claim_boundary_contract(context)
