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
        manifest={"scientific_input_hash": "b" * 64},
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
