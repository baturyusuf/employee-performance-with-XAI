from __future__ import annotations

import json
import socket
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.experiments import run_model_benchmark_trial as trial


def _settings(*, inner_splits: int = 5) -> dict:
    return {
        "target": {"column": "PerformanceRating", "labels": [2, 3, 4]},
        "governance_fields": {"identifier_fields": ["EmpNumber"]},
        "evaluation": {
            "cv": {"n_splits": 10, "seed": "cv"},
            "bootstrap": {
                "n_resamples": 5000,
                "confidence_level": 0.95,
                "method": "paired_stratified_percentile",
                "stratify_by": ["outer_fold", "y_true"],
                "quantile_method": "linear",
            },
        },
        "model": {
            "search_space_config": "configs/model_grid.yaml",
            "nested_tuning": {
                "inner_splits": inner_splits,
                "inner_seed": "inner_cv",
                "primary_practical_tie_tolerance": 0.001,
            },
        },
        "seeds": {"cv": 42, "inner_cv": 43},
    }


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "EmpNumber": [f"E{index:04d}" for index in range(1200)],
            "feature": [index % 11 for index in range(1200)],
            "PerformanceRating": [2 + index % 3 for index in range(1200)],
        },
        index=range(1200),
    )


def _atomic_test_manifest_writer(
    manifest,
    path,
    *,
    project_root,
    validate=True,
    require_complete=False,
):
    del project_root, validate, require_complete
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.unit-test.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def _install_trial_fixture(
    monkeypatch: pytest.MonkeyPatch,
    project_root: Path,
    *,
    gate_triggered: bool = True,
    gate_metric: str = "macro_f1",
    gate_trigger_rule: str = "point_estimate_gt_zero_and_paired_ci_low_gt_zero",
    inner_splits: int = 5,
    benchmark_failure: Exception | None = None,
):
    config_path = project_root / "configs" / "manuscript_final.yaml"
    grid_path = project_root / "configs" / "model_grid.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}\n", encoding="utf-8")
    grid_path.write_text('{"model_benchmark": {"fixture": true}}\n', encoding="utf-8")
    model_grid_hash = trial.sha256_file(grid_path)
    config = {"manuscript_final": _settings(inner_splits=inner_splits)}
    receipt = {
        "dataset_key": "inx_primary",
        "actual_path": "data/inx.csv",
        "actual_sha256": "c" * 64,
        "row_count": 1200,
        "column_count": 3,
        "schema_status": "passed",
        "target_column": "PerformanceRating",
        "target_distribution": {"2": 400, "3": 400, "4": 400},
    }

    def fake_create_manifest(
        observed_config_path,
        *,
        evidence_scope,
        project_root,
        run_id,
        allow_dataset_download,
        initial_command,
    ):
        assert Path(observed_config_path).resolve() == config_path.resolve()
        assert evidence_scope == "core"
        assert project_root == project_root_fixture
        assert allow_dataset_download is False
        assert initial_command.startswith(
            "python -m src.experiments.run_model_benchmark_trial --config configs/manuscript_final.yaml"
        )
        return {
            "run_id": run_id or "trial-fixture",
            "git_worktree_dirty": False,
            "config_path": "configs/manuscript_final.yaml",
            "config_hash": "a" * 64,
            "scientific_input_hash": "b" * 64,
            "evidence_scope": "core",
            "actual_input_receipts": {"inx_primary": dict(receipt)},
            "side_input_hashes": {
                "model_search_space": {
                    "path": "configs/model_grid.yaml",
                    "sha256": model_grid_hash,
                    "size_bytes": grid_path.stat().st_size,
                }
            },
            "commands": [
                {
                    "command": initial_command,
                    "stage": "entrypoint",
                    "status": "started",
                    "started_at": "2026-07-13T00:00:00+00:00",
                    "ended_at": None,
                    "return_code": None,
                }
            ],
            "output_files": [],
            "status": "running",
            "failure_information": [],
            "start_timestamp": "2026-07-13T00:00:00+00:00",
            "end_timestamp": None,
        }

    def fake_validate_manifest(
        manifest_path,
        *,
        project_root,
        expected_evidence_scope,
        require_complete,
        verify_source_tree,
        **kwargs,
    ):
        del kwargs
        assert project_root == project_root_fixture
        assert expected_evidence_scope == "core"
        assert require_complete is True
        assert verify_source_tree is True
        return json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    def fake_benchmark(
        observed_config_path,
        *,
        model_grid_path,
        shared_folds_dir,
        output_dir,
        run_id,
        config_hash,
        scientific_input_hash,
        model_grid_sha256,
    ):
        assert Path(observed_config_path).resolve() == config_path.resolve()
        assert Path(model_grid_path).resolve() == grid_path.resolve()
        assert model_grid_sha256 == model_grid_hash
        fold_contract = json.loads(
            (Path(shared_folds_dir) / "fold_contract.json").read_text(encoding="utf-8")
        )
        assert fold_contract["outer_splits"] == 10
        assert fold_contract["inner_splits"] == 5
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        (output / "partial_or_summary.csv").write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")
        if benchmark_failure is not None:
            raise benchmark_failure
        identity = {
            "run_id": run_id,
            "config_hash": config_hash,
            "scientific_input_hash": scientific_input_hash,
            "fold_contract_hash": fold_contract["fold_contract_hash"],
        }
        model_names = list(trial.CANONICAL_MODEL_NAMES)
        candidate_rows = []
        selected_rows = []
        fold_rows = []
        model_index_rows = []
        lineage_rows = []
        for outer_fold in range(1, 11):
            for model_name in model_names:
                candidate_count = trial.EXPECTED_CANDIDATE_COUNTS[model_name]
                for candidate_index in range(candidate_count):
                    candidate_rows.append(
                        {
                            **identity,
                            "outer_fold": outer_fold,
                            "model": model_name,
                            "candidate_index": candidate_index,
                            "n_inner_folds": 5,
                            "selected_by_protocol": candidate_index == 0,
                        }
                    )
                selected_rows.append(
                    {
                        **identity,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "selected_candidate_index": 0,
                    }
                )
                fold_rows.append(
                    {
                        **identity,
                        "outer_fold": outer_fold,
                        "model": model_name,
                        "n_train": 1080,
                        "n_test": 120,
                    }
                )
                model_relative = Path("models") / model_name / f"outer_fold_{outer_fold:02d}.joblib"
                model_path = output / model_relative
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_bytes(f"{model_name}:{outer_fold}".encode("ascii"))
                model_index_rows.append(
                    {
                        **identity,
                        "model": model_name,
                        "outer_fold": outer_fold,
                        "path": model_relative.as_posix(),
                        "sha256": trial.sha256_file(model_path),
                        "size_bytes": model_path.stat().st_size,
                    }
                )
                lineage_rows.append(
                    {
                        **identity,
                        "model": model_name,
                        "outer_fold": outer_fold,
                        "transformed_feature_index": 0,
                        "transformed_feature_name": "numeric__feature",
                    }
                )
        pd.DataFrame(candidate_rows).to_csv(output / "candidate_search_results.csv", index=False)
        pd.DataFrame(selected_rows).to_csv(output / "selected_hyperparameters.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
        pd.DataFrame(model_index_rows).to_csv(output / "fitted_model_index.csv", index=False)
        pd.DataFrame(lineage_rows).to_csv(output / "transformed_feature_lineage.csv", index=False)

        outer_assignments = pd.read_csv(Path(shared_folds_dir) / trial.OUTER_ASSIGNMENT_FILENAME)
        oof_rows = []
        for model_name in model_names:
            for row in outer_assignments.itertuples(index=False):
                probabilities = {2: 0.1, 3: 0.1, 4: 0.1}
                probabilities[int(row.y_true)] = 0.8
                oof_rows.append(
                    {
                        **identity,
                        "system_id": model_name,
                        "model": model_name,
                        "sample_index": int(row.sample_index),
                        "outer_fold": int(row.outer_fold),
                        "y_true": int(row.y_true),
                        "y_pred": int(row.y_true),
                        "prob_class_2": probabilities[2],
                        "prob_class_3": probabilities[3],
                        "prob_class_4": probabilities[4],
                    }
                )
        pd.DataFrame(oof_rows).to_csv(output / "oof_predictions.csv", index=False)

        resample_hash = "d" * 64
        summary_rows = [
            {
                **identity,
                "system_id": model_name,
                "metric": metric,
                "n_samples": 1200,
                "n_resamples": 5000,
                "resample_hash": resample_hash,
            }
            for model_name in model_names
            for metric in trial.BENCHMARK_METRICS
        ]
        pd.DataFrame(summary_rows).to_csv(output / "model_summary.csv", index=False)
        comparison_ids = [
            f"{model_name}_minus_xgboost"
            for model_name in model_names
            if model_name != "xgboost"
        ]
        triggered_id = comparison_ids[0] if gate_triggered else None
        paired_rows = []
        for comparison_id in comparison_ids:
            for metric in trial.BENCHMARK_METRICS:
                eligible = metric == "macro_f1"
                triggered = eligible and comparison_id == triggered_id
                paired_rows.append(
                    {
                        **identity,
                        "comparison_id": comparison_id,
                        "metric": metric,
                        "improvement_oriented_difference": 0.1 if triggered else 0.0,
                        "improvement_ci_low": 0.01 if triggered else 0.0,
                        "n_resamples": 5000,
                        "n_valid": 5000,
                        "resample_hash": resample_hash,
                        "gate_eligible": eligible,
                        "gate_triggered": triggered,
                    }
                )
        pd.DataFrame(paired_rows).to_csv(output / "paired_model_differences.csv", index=False)
        gate_path = output / "baseline_xgboost_gate.json"
        gate_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "scientific_input_hash": scientific_input_hash,
                    "fold_contract_hash": fold_contract["fold_contract_hash"],
                    "gate_metric": gate_metric,
                    "trigger_rule": gate_trigger_rule,
                    "gate_triggered": gate_triggered,
                    "triggered_comparisons": (
                        [triggered_id] if gate_triggered else []
                    ),
                    "user_decision_required_if_triggered": True,
                    "n_resamples": 5000,
                    "resample_hash": resample_hash,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        metadata = output / "stage_metadata.json"
        metadata.write_text(
            json.dumps({**identity, "stage": "model_benchmarks", "status": "complete"}) + "\n",
            encoding="utf-8",
        )
        return {
            "candidate_search_results": output / "candidate_search_results.csv",
            "selected_hyperparameters": output / "selected_hyperparameters.csv",
            "fold_metrics": output / "fold_metrics.csv",
            "oof_predictions": output / "oof_predictions.csv",
            "model_summary": output / "model_summary.csv",
            "paired_model_differences": output / "paired_model_differences.csv",
            "baseline_gate": gate_path,
            "metadata": metadata,
            "feature_lineage": output / "transformed_feature_lineage.csv",
            "model_index": output / "fitted_model_index.csv",
        }

    project_root_fixture = project_root.resolve()
    monkeypatch.setattr(trial, "_git_porcelain", lambda root: "")
    monkeypatch.setattr(trial, "create_run_manifest", fake_create_manifest)
    monkeypatch.setattr(trial, "load_manuscript_config", lambda path: config)
    monkeypatch.setattr(
        trial,
        "load_canonical_dataset",
        lambda config_path, dataset_key: SimpleNamespace(frame=_frame(), receipt=dict(receipt)),
    )
    monkeypatch.setattr(trial, "run_model_benchmark", fake_benchmark)
    monkeypatch.setattr(trial, "write_run_manifest", _atomic_test_manifest_writer)
    monkeypatch.setattr(trial, "validate_run_manifest", fake_validate_manifest)
    return config_path


@pytest.mark.parametrize("gate_triggered", [False, True])
def test_trial_is_noncanonical_hash_complete_and_stops_at_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_triggered: bool,
) -> None:
    config_path = _install_trial_fixture(
        monkeypatch,
        tmp_path,
        gate_triggered=gate_triggered,
    )
    latest = tmp_path / "reports" / "manuscript_final" / "latest"
    latest.mkdir(parents=True)
    sentinel = latest / "sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")

    result = trial.run_trial(
        config_path,
        run_id="trial-fixture",
        project_root=tmp_path,
    )

    assert result.decision_required is gate_triggered
    assert result.run_dir == (
        tmp_path / "reports" / "manuscript_final" / "trials" / "trial-fixture" / "core"
    ).resolve()
    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
    manifest = json.loads(result.run_manifest.read_text(encoding="utf-8"))
    assert manifest["run_kind"] == "model_benchmark_trial"
    assert manifest["canonical_release_eligible"] is False
    assert manifest["latest_pointer_updated"] is False
    assert manifest["executed_stages"] == ["shared_folds", "model_benchmarks"]
    assert manifest["downstream_stages_executed"] == []
    assert manifest["decision_required"] is gate_triggered
    assert manifest["status"] == "complete"
    assert manifest["end_timestamp"]
    assert isinstance(manifest["elapsed_seconds"], float)
    assert manifest["elapsed_seconds"] >= 0.0
    entrypoint = next(row for row in manifest["commands"] if row["stage"] == "entrypoint")
    assert entrypoint["status"] == "complete"
    assert entrypoint["return_code"] == 0
    assert entrypoint["ended_at"]
    assert entrypoint["elapsed_seconds"] == manifest["elapsed_seconds"]
    for stage in ("shared_folds", "model_benchmarks"):
        stage_command = next(row for row in manifest["commands"] if row["stage"] == stage)
        assert stage_command["status"] == "complete"
        assert stage_command["return_code"] == 0
        assert stage_command["ended_at"]
        assert stage_command["elapsed_seconds"] >= 0.0
    artifact_files = {
        path.resolve()
        for path in result.run_dir.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    }
    registered = {
        (tmp_path / row["path"]).resolve()
        for row in manifest["output_files"]
    }
    assert registered == artifact_files
    assert all(not Path(row["path"]).is_absolute() for row in manifest["output_files"])
    assert all("/latest/" not in f"/{row['path']}/" for row in manifest["output_files"])
    trial.verify_trial_manifest(result.run_manifest, project_root=tmp_path)
    assert not list(result.run_dir.rglob("*.tmp"))


def test_dirty_worktree_refuses_before_manifest_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trial, "_git_porcelain", lambda root: " M configs/model_grid.yaml")
    monkeypatch.setattr(
        trial,
        "create_run_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("manifest must not be created")),
    )

    with pytest.raises(trial.ModelBenchmarkTrialError, match="clean worktree"):
        trial.run_trial("configs/manuscript_final.yaml", project_root=tmp_path)

    assert not (tmp_path / "reports" / "manuscript_final" / "trials").exists()


def test_non_10x5_config_fails_and_persists_atomic_failed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _install_trial_fixture(monkeypatch, tmp_path, inner_splits=3)

    with pytest.raises(trial.ModelBenchmarkTrialError, match="10 outer and 5 inner"):
        trial.run_trial(config_path, run_id="wrong-folds", project_root=tmp_path)

    manifest_path = (
        tmp_path
        / "reports"
        / "manuscript_final"
        / "trials"
        / "wrong-folds"
        / "core"
        / "run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["end_timestamp"]
    assert manifest["elapsed_seconds"] >= 0.0
    assert manifest["canonical_release_eligible"] is False
    assert manifest["executed_stages"] == []
    assert manifest["failure_information"][-1]["stage"] == "preflight"
    entrypoint = next(row for row in manifest["commands"] if row["stage"] == "entrypoint")
    assert entrypoint["status"] == "failed"
    assert entrypoint["return_code"] == 1
    assert entrypoint["elapsed_seconds"] == manifest["elapsed_seconds"]
    assert not list(manifest_path.parent.rglob("*.tmp"))


def test_bootstrap_protocol_drift_fails_before_shared_folds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _install_trial_fixture(monkeypatch, tmp_path)
    settings = _settings()
    settings["evaluation"]["bootstrap"]["n_resamples"] = 4999
    monkeypatch.setattr(
        trial,
        "load_manuscript_config",
        lambda path: {"manuscript_final": settings},
    )
    monkeypatch.setattr(
        trial,
        "generate_shared_folds",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared folds must not run after bootstrap protocol drift")
        ),
    )

    with pytest.raises(trial.ModelBenchmarkTrialError, match="5,000-draw paired OOF bootstrap"):
        trial.run_trial(config_path, run_id="bootstrap-drift", project_root=tmp_path)

    manifest_path = (
        tmp_path
        / "reports"
        / "manuscript_final"
        / "trials"
        / "bootstrap-drift"
        / "core"
        / "run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["executed_stages"] == []
    assert manifest["failure_information"][-1]["stage"] == "preflight"


def test_practical_tie_tolerance_drift_fails_before_shared_folds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _install_trial_fixture(monkeypatch, tmp_path)
    settings = _settings()
    settings["model"]["nested_tuning"]["primary_practical_tie_tolerance"] = 0.002
    monkeypatch.setattr(
        trial,
        "load_manuscript_config",
        lambda path: {"manuscript_final": settings},
    )
    monkeypatch.setattr(
        trial,
        "generate_shared_folds",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("shared folds must not run after tie-tolerance drift")
        ),
    )

    with pytest.raises(trial.ModelBenchmarkTrialError, match="tolerance=0.001"):
        trial.run_trial(config_path, run_id="tolerance-drift", project_root=tmp_path)

    manifest_path = (
        tmp_path
        / "reports"
        / "manuscript_final"
        / "trials"
        / "tolerance-drift"
        / "core"
        / "run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["executed_stages"] == []
    assert manifest["failure_information"][-1]["stage"] == "preflight"


def test_benchmark_failure_registers_partial_artifacts_and_finalizes_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _install_trial_fixture(
        monkeypatch,
        tmp_path,
        benchmark_failure=RuntimeError("intentional benchmark failure"),
    )

    with pytest.raises(RuntimeError, match="intentional benchmark failure"):
        trial.run_trial(config_path, run_id="failed-benchmark", project_root=tmp_path)

    manifest_path = (
        tmp_path
        / "reports"
        / "manuscript_final"
        / "trials"
        / "failed-benchmark"
        / "core"
        / "run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["executed_stages"] == ["shared_folds"]
    assert manifest["downstream_stages_executed"] == []
    assert manifest["decision_required"] is None
    assert any(
        row["path"].endswith("model_benchmarks/partial_or_summary.csv")
        for row in manifest["output_files"]
    )
    benchmark_command = next(
        row for row in manifest["commands"] if row["stage"] == "model_benchmarks"
    )
    assert benchmark_command["status"] == "failed"
    assert benchmark_command["return_code"] == 1
    assert benchmark_command["elapsed_seconds"] >= 0.0
    assert manifest["failure_information"][-1]["stage"] == "model_benchmarks"


def test_trial_verifier_detects_post_registration_artifact_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _install_trial_fixture(monkeypatch, tmp_path, gate_triggered=False)
    result = trial.run_trial(config_path, run_id="tamper-check", project_root=tmp_path)
    summary = result.run_dir / "model_benchmarks" / "partial_or_summary.csv"
    summary.write_text("metric,value\nmacro_f1,0.9\n", encoding="utf-8")

    with pytest.raises(trial.ModelBenchmarkTrialError, match="hash mismatch"):
        trial.verify_trial_manifest(result.run_manifest, project_root=tmp_path)


def test_manifest_bound_model_grid_hash_blocks_runner_before_shared_folds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _install_trial_fixture(monkeypatch, tmp_path)
    (tmp_path / "configs" / "model_grid.yaml").write_text("changed\n", encoding="utf-8")
    monkeypatch.setattr(
        trial,
        "run_model_benchmark",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("benchmark must not run")),
    )

    with pytest.raises(trial.ModelBenchmarkTrialError, match="SHA-256 changed"):
        trial.run_trial(config_path, run_id="grid-mismatch", project_root=tmp_path)

    manifest_path = (
        tmp_path
        / "reports"
        / "manuscript_final"
        / "trials"
        / "grid-mismatch"
        / "core"
        / "run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["executed_stages"] == []
    assert manifest["failure_information"][-1]["stage"] == "preflight"


def test_reported_benchmark_paths_must_exist_inside_the_stage_directory(tmp_path: Path) -> None:
    stage = tmp_path / "trial" / "core" / "model_benchmarks"
    stage.mkdir(parents=True)
    inside = stage / "summary.csv"
    inside.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")
    trial._validate_runner_result_paths({"summary": inside}, stage)

    outside = tmp_path / "outside.csv"
    outside.write_text("not,allowed\n", encoding="utf-8")
    with pytest.raises(trial.ModelBenchmarkTrialError, match="escaped"):
        trial._validate_runner_result_paths({"summary": outside}, stage)
    with pytest.raises(trial.ModelBenchmarkTrialError, match="does not reference"):
        trial._validate_runner_result_paths({"summary": stage / "missing.csv"}, stage)


@pytest.mark.parametrize(
    ("gate_metric", "trigger_rule", "message"),
    [
        ("quadratic_weighted_kappa", trial.REQUIRED_GATE_TRIGGER_RULE, "metric"),
        ("macro_f1", "ci_only", "trigger_rule"),
    ],
)
def test_gate_metric_and_point_plus_ci_rule_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_metric: str,
    trigger_rule: str,
    message: str,
) -> None:
    config_path = _install_trial_fixture(
        monkeypatch,
        tmp_path,
        gate_metric=gate_metric,
        gate_trigger_rule=trigger_rule,
    )

    with pytest.raises(trial.ModelBenchmarkTrialError, match=message):
        trial.run_trial(config_path, run_id=f"invalid-gate-{message}", project_root=tmp_path)

    manifest_path = (
        tmp_path
        / "reports"
        / "manuscript_final"
        / "trials"
        / f"invalid-gate-{message}"
        / "core"
        / "run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    benchmark_command = next(
        row for row in manifest["commands"] if row["stage"] == "model_benchmarks"
    )
    assert benchmark_command["status"] == "failed"
    assert benchmark_command["return_code"] == 1
    assert benchmark_command["elapsed_seconds"] >= 0.0


def test_persisted_benchmark_semantics_fail_closed_on_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _install_trial_fixture(monkeypatch, tmp_path, gate_triggered=True)
    result = trial.run_trial(config_path, run_id="semantic-corruption", project_root=tmp_path)
    manifest = json.loads(result.run_manifest.read_text(encoding="utf-8"))
    contract_path = result.run_dir / "shared_folds" / trial.CONTRACT_FILENAME
    fold_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    benchmark_dir = result.run_dir / "model_benchmarks"

    def validate() -> None:
        trial._validate_benchmark_semantics(
            benchmark_dir,
            manifest=manifest,
            fold_contract=fold_contract,
            expected_resamples=5000,
        )

    candidate_path = benchmark_dir / "candidate_search_results.csv"
    original = candidate_path.read_bytes()
    candidate = pd.read_csv(candidate_path)
    candidate.loc[0, "n_inner_folds"] = 4
    candidate.to_csv(candidate_path, index=False)
    with pytest.raises(trial.ModelBenchmarkTrialError, match="n_inner_folds=5"):
        validate()
    candidate_path.write_bytes(original)

    fold_metrics_path = benchmark_dir / "fold_metrics.csv"
    original = fold_metrics_path.read_bytes()
    fold_metrics = pd.read_csv(fold_metrics_path)
    fold_metrics.loc[0, "n_train"] = 1079
    fold_metrics.to_csv(fold_metrics_path, index=False)
    with pytest.raises(trial.ModelBenchmarkTrialError, match="n_train=1080 and n_test=120"):
        validate()
    fold_metrics_path.write_bytes(original)

    oof_path = benchmark_dir / "oof_predictions.csv"
    original = oof_path.read_bytes()
    oof = pd.read_csv(oof_path)
    oof.loc[0, "prob_class_2"] = 0.9
    oof.to_csv(oof_path, index=False)
    with pytest.raises(trial.ModelBenchmarkTrialError, match="normalized probabilities"):
        validate()
    oof_path.write_bytes(original)

    paired_path = benchmark_dir / "paired_model_differences.csv"
    original = paired_path.read_bytes()
    paired = pd.read_csv(paired_path)
    triggered = paired["gate_triggered"].astype(str).str.casefold() == "true"
    assert triggered.sum() == 1
    paired.loc[triggered, "gate_triggered"] = False
    paired.to_csv(paired_path, index=False)
    with pytest.raises(trial.ModelBenchmarkTrialError, match=r"point \+ CI rule"):
        validate()
    paired_path.write_bytes(original)

    gate_path = benchmark_dir / "baseline_xgboost_gate.json"
    original = gate_path.read_bytes()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["resample_hash"] = "not-a-sha256"
    gate_path.write_text(json.dumps(gate) + "\n", encoding="utf-8")
    with pytest.raises(trial.ModelBenchmarkTrialError, match="resample_hash"):
        validate()
    gate_path.write_bytes(original)

    model_index = pd.read_csv(benchmark_dir / "fitted_model_index.csv")
    model_path = benchmark_dir / str(model_index.iloc[0]["path"])
    original = model_path.read_bytes()
    model_path.write_bytes(original + b"tampered")
    with pytest.raises(trial.ModelBenchmarkTrialError, match="hash/size"):
        validate()
    model_path.write_bytes(original)

    validate()


def test_public_trial_entrypoint_enforces_offline_network_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def attempt_network(*args, **kwargs):
        socket.getaddrinfo("example.com", 443)
        raise AssertionError("network guard did not run")

    monkeypatch.setattr(trial, "_run_trial_impl", attempt_network)
    with pytest.raises(trial.ModelBenchmarkTrialError, match="Network access is prohibited"):
        trial.run_trial()
