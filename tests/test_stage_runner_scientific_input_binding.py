from __future__ import annotations

import json

import pytest

from src.experiments import build_manuscript_evidence as manuscript_build
from src.governance.manuscript_contract import ACTUAL_INPUT_IDENTITY_FIELDS, sha256_file


def _context(tmp_path, *, gate_triggered: bool = False):
    run_dir = tmp_path / "run" / "core"
    (run_dir / "shared_folds").mkdir(parents=True)
    (run_dir / "model_benchmarks").mkdir(parents=True)
    fold_hash = "d" * 64
    (run_dir / "shared_folds" / "fold_contract.json").write_text(
        json.dumps({"fold_contract_hash": fold_hash}),
        encoding="utf-8",
    )
    comparisons = ["lightgbm_minus_xgboost"] if gate_triggered else []
    gate = {
        "run_id": "binding-test",
        "config_hash": "a" * 64,
        "scientific_input_hash": "b" * 64,
        "fold_contract_hash": fold_hash,
        "gate_metric": "macro_f1",
        "comparison_direction": "baseline_improvement_over_xgboost",
        "trigger_rule": "point_estimate_gt_zero_and_paired_ci_low_gt_zero",
        "gate_triggered": gate_triggered,
        "triggered_comparisons": comparisons,
        "user_decision_required_if_triggered": True,
        "n_resamples": 5000,
        "resample_hash": "c" * 64,
    }
    (run_dir / "model_benchmarks" / "baseline_xgboost_gate.json").write_text(
        json.dumps(gate),
        encoding="utf-8",
    )
    return manuscript_build.StageContext(
        config_path=manuscript_build.PROJECT_ROOT / "configs" / "manuscript_final.yaml",
        config={},
        settings={"evaluation": {"bootstrap": {"n_resamples": 5000}}},
        run_dir=run_dir,
        run_id="binding-test",
        config_hash="a" * 64,
        manifest={
            "evidence_scope": "core",
            "scope_contract_hash": "1" * 64,
            "git_commit": "2" * 40,
            "source_tree_hash": "3" * 64,
            "dataset_hashes": {"inx_primary": {"sha256": "4" * 64}},
            "actual_input_receipts": {
                "inx_primary": {
                    "actual_path": "data/inx.csv",
                    "actual_sha256": "4" * 64,
                    "row_count": 1,
                    "column_count": 2,
                }
            },
            "side_input_hashes": {
                "model_search_space": {
                    "path": "configs/model_grid.yaml",
                    "sha256": "5" * 64,
                    "size_bytes": 1,
                }
            },
            "scientific_input_hash": "b" * 64,
        },
    )


def _write_compatible_upstream_contracts(
    context: manuscript_build.StageContext,
) -> None:
    for stage, output_name in (
        ("shared_folds", "fold_contract.json"),
        ("model_benchmarks", "baseline_xgboost_gate.json"),
    ):
        manuscript_build._write_stage_metadata(
            context,
            stage,
            [context.run_dir / stage / output_name],
            started_at="2026-07-13T00:00:00+00:00",
            elapsed_seconds=0.0,
        )


def test_stage_loader_receipt_must_match_manifest_identity(tmp_path) -> None:
    receipt = {field: f"value-{field}" for field in ACTUAL_INPUT_IDENTITY_FIELDS}
    receipt["size_bytes"] = 123
    context = _context(tmp_path)
    context.manifest["actual_input_receipts"] = {"inx_primary": dict(receipt)}

    manuscript_build._validate_loaded_dataset_receipt(context, "inx_primary", receipt)
    changed = dict(receipt)
    changed["actual_sha256"] = "0" * 64
    with pytest.raises(manuscript_build.ManuscriptBuildError, match="does not match"):
        manuscript_build._validate_loaded_dataset_receipt(context, "inx_primary", changed)


def test_model_grid_path_and_hash_must_match_scoped_manifest(tmp_path) -> None:
    context = _context(tmp_path)
    path = manuscript_build.PROJECT_ROOT / "configs" / "model_grid.yaml"
    context.manifest["side_input_hashes"] = {
        "model_search_space": {
            "path": "configs/model_grid.yaml",
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    }
    observed, digest = manuscript_build._validated_side_input_path(
        context,
        "model_search_space",
        "configs/model_grid.yaml",
    )
    assert observed == path.resolve()
    assert digest == sha256_file(path)

    context.manifest["side_input_hashes"]["model_search_space"]["sha256"] = "0" * 64
    with pytest.raises(manuscript_build.ManuscriptBuildError, match="changed"):
        manuscript_build._validated_side_input_path(
            context,
            "model_search_space",
            "configs/model_grid.yaml",
        )


def test_nontriggered_baseline_gate_allows_downstream_execution(tmp_path) -> None:
    gate = manuscript_build.enforce_baseline_reference_gate(_context(tmp_path))
    assert gate["gate_triggered"] is False


def test_triggered_baseline_gate_requires_user_decision_before_downstream(tmp_path) -> None:
    with pytest.raises(
        manuscript_build.BaselineReferenceDecisionRequired,
        match="Downstream policy, calibration and SHAP stages were not started",
    ):
        manuscript_build.enforce_baseline_reference_gate(
            _context(tmp_path, gate_triggered=True)
        )


def test_baseline_gate_rejects_run_identity_drift(tmp_path) -> None:
    context = _context(tmp_path)
    path = context.run_dir / "model_benchmarks" / "baseline_xgboost_gate.json"
    gate = json.loads(path.read_text(encoding="utf-8"))
    gate["config_hash"] = "f" * 64
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(manuscript_build.ManuscriptBuildError, match="identity mismatch"):
        manuscript_build.enforce_baseline_reference_gate(context)


def test_shap_runner_receives_only_current_run_upstreams_and_full_identity(
    tmp_path,
    monkeypatch,
) -> None:
    from src.experiments import manuscript_shap_evidence

    context = _context(tmp_path)
    _write_compatible_upstream_contracts(context)
    captured = {}

    def fake_run(config_path, **kwargs):
        captured["config_path"] = config_path
        captured.update(kwargs)
        return {"status": "captured"}

    monkeypatch.setattr(manuscript_shap_evidence, "run", fake_run)

    result = manuscript_build._run_shap(context)

    assert result == {"status": "captured"}
    assert captured == {
        "config_path": context.config_path,
        "shared_folds_dir": context.run_dir / "shared_folds",
        "model_benchmarks_dir": context.run_dir / "model_benchmarks",
        "output_dir": context.run_dir / "oof_shap",
        "run_id": context.run_id,
        "config_hash": context.config_hash,
        "scientific_input_hash": context.manifest["scientific_input_hash"],
    }


@pytest.mark.parametrize("stage", ["shared_folds", "model_benchmarks"])
def test_shap_runner_fails_when_current_run_upstream_is_missing(
    tmp_path,
    stage: str,
) -> None:
    context = _context(tmp_path)
    _write_compatible_upstream_contracts(context)
    contract = context.run_dir / stage / "stage_contract.json"
    contract.unlink()

    with pytest.raises(manuscript_build.ManuscriptBuildError, match="contract is missing"):
        manuscript_build._run_shap(context)


def test_shap_runner_rejects_incompatible_upstream_identity(tmp_path) -> None:
    context = _context(tmp_path)
    _write_compatible_upstream_contracts(context)
    contract = context.run_dir / "model_benchmarks" / "stage_contract.json"
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["scientific_input_hash"] = "f" * 64
    contract.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(manuscript_build.ManuscriptBuildError, match="incompatible"):
        manuscript_build._run_shap(context)


def test_execute_stage_provides_an_empty_builder_owned_shap_directory(
    tmp_path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)

    def runner(stage_context):
        stage_dir = stage_context.run_dir / "oof_shap"
        assert stage_dir.is_dir()
        assert not any(stage_dir.iterdir())
        artifact = stage_dir / "artifact.csv"
        artifact.write_text("run_id,value\nbinding-test,1\n", encoding="utf-8")
        return {"artifact": artifact}

    monkeypatch.setitem(manuscript_build.STAGE_RUNNERS, "oof_shap", runner)
    outputs = manuscript_build._execute_stage(
        context,
        "oof_shap",
        reuse_compatible=False,
    )

    assert (context.run_dir / "oof_shap" / "artifact.csv") in outputs
    assert (context.run_dir / "oof_shap" / "stage_contract.json") in outputs
