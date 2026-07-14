from __future__ import annotations

import ast
import inspect
from subprocess import CompletedProcess
from pathlib import Path

import pandas as pd
import pytest

from src.data.canonical_loader import load_canonical_dataset
from src.data.external_adapters import load_external_dataset
from src.experiments import build_manuscript_evidence as builder
from src.experiments import hrdataset_replication_core as replication_core
from src.experiments import manuscript_hrdataset_replication as stage
from src.governance.manuscript_contract import (
    canonical_config_hash,
    create_run_manifest,
    evidence_scope_contract_hash,
    load_manuscript_config,
    manuscript_settings,
)
from src.utils.config_loader import PROJECT_ROOT


CONFIG = PROJECT_ROOT / "configs" / "manuscript_final.yaml"
SCHEMA = PROJECT_ROOT / "data" / "external" / "hrdataset_v14" / "schema_mapping.json"


def _preflight():
    config = load_manuscript_config(CONFIG)
    settings = manuscript_settings(config)
    external = {
        **dict(settings["external_replication"]),
        "resolved_seeds": dict(settings["seeds"]),
    }
    inx = load_canonical_dataset(CONFIG, "inx_primary")
    raw_hr = load_canonical_dataset(CONFIG, "hrdataset_v14")
    dataset = load_external_dataset(
        "hrdataset_v14",
        raw_frame=raw_hr.frame,
        schema_mapping_path=SCHEMA,
    )
    frames, roles, forbidden, rows = stage._feature_contract(dataset, external)
    return config, external, inx, raw_hr, dataset, frames, roles, forbidden, rows


def test_core_builder_routes_external_replication_only_to_new_stage() -> None:
    source = inspect.getsource(builder._run_external_replication)
    tree = ast.parse(source)
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    ]
    assert "run" in imports
    assert "manuscript_hrdataset_replication" in source
    assert "manuscript_external_evidence" not in source
    assert "expected_actual_input_receipts" in source
    assert "expected_side_input_hashes" in source
    assert "scientific_input_hash" in source
    assert "expected_git_worktree_dirty" in source


def test_real_preflight_has_exact_policy_features_and_three_feature_transport_gate() -> None:
    config, external, inx, raw_hr, dataset, frames, roles, forbidden, rows = _preflight()
    assert list(frames) == list(stage.POLICY_ORDER)
    assert list(frames["conservative_primary"].columns) == [
        "EmpJobRole",
        "EngagementSurvey",
        "EmpJobSatisfaction",
        "SpecialProjectsCount",
        "DaysLateLast30",
        "Absences",
        "ExperienceYearsAtThisCompany",
    ]
    assert int(frames["conservative_primary"]["ExperienceYearsAtThisCompany"].isna().sum()) == 2
    assert roles["conservative_primary"] == "canonical_external_primary"
    assert "Zip" in forbidden["proxy_rich_audit"]
    assert all(
        len(values) == len({value.casefold() for value in values})
        for values in forbidden.values()
    )
    validated_frames, validated_target, validated_ids, validated_forbidden = (
        replication_core._validate_inputs(
            frames,
            roles,
            forbidden,
            dataset.canonical[dataset.target_column].astype(int),
            dataset.canonical["EmpNumber"],
            primary_policy="conservative_primary",
        )
    )
    assert tuple(validated_frames) == tuple(stage.POLICY_ORDER)
    assert len(validated_target) == len(validated_ids) == len(dataset.canonical)
    assert validated_forbidden["conservative_primary"] == tuple(
        forbidden["conservative_primary"]
    )
    assert not rows.empty
    identity = {
        "run_id": "preflight-test",
        "config_hash": canonical_config_hash(config),
        "scientific_input_hash": "a" * 64,
        "dataset_sha256": raw_hr.receipt["actual_sha256"],
        "schema_mapping_sha256": "b" * 64,
        "fold_contract_hash": "c" * 64,
        "feature_policy_contract_sha256": "d" * 64,
        "model_set_sha256": "e" * 64,
    }
    overlap, assessment = stage._transport_evidence(
        inx,
        frames["conservative_primary"],
        config,
        identity=identity,
    )
    assert assessment["status"] == "infeasible_too_few_common_safe_features"
    assert assessment["locked_inx_model_transported"] is False
    assert assessment["common_safe_features"] == [
        "EmpJobRole",
        "EmpJobSatisfaction",
        "ExperienceYearsAtThisCompany",
    ]
    assert int(overlap["common_safe_feature"].sum()) == 3
    assert dataset.canonical[dataset.target_column].value_counts().sort_index().to_dict() == {
        2: 31,
        3: 243,
        4: 37,
    }


def test_governance_adapter_is_complete_and_never_defaults_primary_metadata() -> None:
    _, external, _, _, _, frames, _, _, _ = _preflight()
    governance = stage._governance_mapping(external)
    assert list(governance) == list(frames["conservative_primary"].columns)
    assert all(row["governance_category"] != "external_context_dependent" for row in governance.values())
    assert all("timing" in row["temporality_status"] for row in governance.values())
    assert all("model_scenario_only_warning" in row for row in governance.values())


def test_portability_and_scope_validator_rejects_user_paths_and_core_leakage(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "evidence.json").write_text('{"path":"reports/run/file.csv"}\n', encoding="utf-8")
    stage._validate_portability_and_scope(clean)

    (clean / "absolute.txt").write_text(
        "C:\\Users\\Example\\private\\artifact.csv\n", encoding="utf-8"
    )
    with pytest.raises(stage.HRDatasetStageError, match="absolute user path"):
        stage._validate_portability_and_scope(clean)

    (clean / "absolute.txt").unlink()
    (clean / "chatbot_results.csv").write_text("value\n1\n", encoding="utf-8")
    with pytest.raises(stage.HRDatasetStageError, match="forbidden core artifact path"):
        stage._validate_portability_and_scope(clean)


def test_derived_feature_quality_receipt_is_machine_readable() -> None:
    config, _, _, raw_hr, dataset, _, _, _, _ = _preflight()
    identity = {
        "run_id": "quality-test",
        "config_hash": canonical_config_hash(config),
        "scientific_input_hash": "a" * 64,
        "dataset_sha256": raw_hr.receipt["actual_sha256"],
    }
    quality = stage._derived_feature_quality(dataset, identity=identity)
    assert isinstance(quality, pd.DataFrame) and len(quality) == 1
    assert int(quality.iloc[0]["observed_missing_after_quality_rule"]) == 2
    assert int(quality.iloc[0]["observed_negative_after_quality_rule"]) == 0
    assert bool(quality.iloc[0]["raw_date_fields_used_as_model_inputs"]) is False


def test_canonical_stage_rejects_dirty_start_before_any_computation(tmp_path: Path) -> None:
    config = load_manuscript_config(CONFIG)
    with pytest.raises(stage.HRDatasetStageError, match="clean worktree"):
        stage.run(
            CONFIG,
            output_dir=tmp_path / "external",
            run_id="dirty-start-test",
            config_hash=canonical_config_hash(config),
            scientific_input_hash="a" * 64,
            expected_actual_input_receipts={},
            expected_side_input_hashes={},
            git_commit="deadbeef",
            source_tree_hash="b" * 64,
            scope_contract_hash="c" * 64,
            expected_git_worktree_dirty=True,
        )


def test_source_identity_start_gate_checks_actual_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):
        if command[1:3] == ["rev-parse", "HEAD"]:
            output = "abc123\n"
        elif command[1] == "status":
            output = " M tracked.py\n"
        else:
            output = ""
        return CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(stage.subprocess, "run", fake_run)
    with pytest.raises(stage.HRDatasetStageError, match="clean worktree"):
        stage._validate_source_identity_at_start(
            git_commit="abc123",
            expected_source_tree_hash="a" * 64,
            allowed_untracked_root=PROJECT_ROOT / "reports/manuscript_final/run/core",
        )


def test_source_identity_allows_only_builder_owned_current_run_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = PROJECT_ROOT / "reports/manuscript_final/test-run/core"

    def install_git_state(untracked: str) -> None:
        def fake_run(command, **kwargs):
            if command[1:3] == ["rev-parse", "HEAD"]:
                output = "abc123\n"
            elif command[1] == "status":
                output = ""
            else:
                output = untracked
            return CompletedProcess(command, 0, stdout=output, stderr="")

        monkeypatch.setattr(stage.subprocess, "run", fake_run)
        monkeypatch.setattr(stage, "compute_source_tree_hash", lambda root: "a" * 64)

    install_git_state("reports/manuscript_final/test-run/core/run_manifest.json\0")
    stage._validate_source_identity_at_start(
        git_commit="abc123",
        expected_source_tree_hash="a" * 64,
        allowed_untracked_root=run_root,
    )

    install_git_state(
        "reports/manuscript_final/test-run/core/run_manifest.json\0unrelated_notes.txt\0"
    )
    with pytest.raises(stage.HRDatasetStageError, match="outside the current run root"):
        stage._validate_source_identity_at_start(
            git_commit="abc123",
            expected_source_tree_hash="a" * 64,
            allowed_untracked_root=run_root,
        )


def test_output_contract_rejects_broad_or_noncanonical_run_roots() -> None:
    settings = manuscript_settings(load_manuscript_config(CONFIG))
    run_id = "unit2g-output-contract"
    expected = PROJECT_ROOT / "reports/manuscript_final" / run_id / "core"
    assert stage._validate_output_contract(
        settings,
        run_id=run_id,
        output=expected / "external_replication",
    ) == expected.resolve()

    with pytest.raises(stage.HRDatasetStageError, match="builder-owned run contract"):
        stage._validate_output_contract(
            settings,
            run_id=run_id,
            output=PROJECT_ROOT / "external_replication",
        )
    with pytest.raises(stage.HRDatasetStageError, match="portable|path component"):
        stage._validate_output_contract(
            settings,
            run_id="../broad-root",
            output=expected / "external_replication",
        )
    for dot_segment in (".", ".."):
        with pytest.raises(stage.HRDatasetStageError, match="portable|path component"):
            stage._validate_output_contract(
                settings,
                run_id=dot_segment,
                output=expected / "external_replication",
            )
    for reserved in ("CON", "NUL.txt"):
        with pytest.raises(stage.HRDatasetStageError, match="reserved Windows"):
            stage._validate_output_contract(
                settings,
                run_id=reserved,
                output=expected / "external_replication",
            )


def test_stage_recomputes_scope_side_input_and_composite_scientific_identity() -> None:
    manifest = create_run_manifest(CONFIG, evidence_scope="core")
    config = load_manuscript_config(CONFIG)
    settings = manuscript_settings(config)

    scope_contract, side_inputs = stage._validate_scope_and_side_inputs(
        config,
        supplied_scope_contract_hash=str(manifest["scope_contract_hash"]),
        records=manifest["side_input_hashes"],
    )
    dataset_hashes = stage._validate_scientific_identity(
        settings,
        config_hash=str(manifest["config_hash"]),
        scope_contract_hash=str(manifest["scope_contract_hash"]),
        scope_contract=scope_contract,
        receipts=manifest["actual_input_receipts"],
        side_inputs=side_inputs,
        supplied_scientific_input_hash=str(manifest["scientific_input_hash"]),
    )
    assert dataset_hashes == manifest["dataset_hashes"]
    assert manifest["scope_contract_hash"] == evidence_scope_contract_hash(config, "core")

    with pytest.raises(stage.HRDatasetStageError, match="scope_contract_hash"):
        stage._validate_scope_and_side_inputs(
            config,
            supplied_scope_contract_hash="f" * 64,
            records=manifest["side_input_hashes"],
        )

    incomplete_side_inputs = dict(manifest["side_input_hashes"])
    incomplete_side_inputs.pop("feature_taxonomy")
    with pytest.raises(stage.HRDatasetStageError, match="side-input set"):
        stage._validate_scope_and_side_inputs(
            config,
            supplied_scope_contract_hash=str(manifest["scope_contract_hash"]),
            records=incomplete_side_inputs,
        )

    with pytest.raises(stage.HRDatasetStageError, match="scientific_input_hash"):
        stage._validate_scientific_identity(
            settings,
            config_hash=str(manifest["config_hash"]),
            scope_contract_hash=str(manifest["scope_contract_hash"]),
            scope_contract=scope_contract,
            receipts=manifest["actual_input_receipts"],
            side_inputs=side_inputs,
            supplied_scientific_input_hash="e" * 64,
        )

    extra_receipts = dict(manifest["actual_input_receipts"])
    extra_receipts["unscoped_dataset"] = manifest["actual_input_receipts"]["inx_primary"]
    with pytest.raises(stage.HRDatasetStageError, match="dataset receipt set"):
        stage._validate_scientific_identity(
            settings,
            config_hash=str(manifest["config_hash"]),
            scope_contract_hash=str(manifest["scope_contract_hash"]),
            scope_contract=scope_contract,
            receipts=extra_receipts,
            side_inputs=side_inputs,
            supplied_scientific_input_hash=str(manifest["scientific_input_hash"]),
        )
