from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.experiments.build_manuscript_evidence import (
    ATOMIC_DIRECTORY_STAGE_RUNNERS,
    STAGE_ORPHAN_PREFIXES,
    STAGE_RUNNERS,
)
from src.experiments.manuscript_core_figures import CoreFigureGenerationError, run
from src.governance.core_figure_package import validate_core_figure_package
from src.governance.manuscript_contract import canonical_config_hash, sha256_file
from src.utils.config_loader import load_config


CONFIG = load_config("configs/manuscript_final.yaml")
IDENTITY = {
    "run_id": "core-figure-generator-test",
    "config_hash": canonical_config_hash(CONFIG),
    "scientific_input_hash": "b" * 64,
    "source_tree_hash": "c" * 64,
}


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _identity(**values: object) -> dict[str, object]:
    return {
        "run_id": IDENTITY["run_id"],
        "config_hash": IDENTITY["config_hash"],
        "scientific_input_hash": IDENTITY["scientific_input_hash"],
        **values,
    }


def _record(path: Path, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _stage_contract(run_root: Path, stage: str) -> None:
    root = run_root / stage
    outputs = [
        _record(path, path.relative_to(root).as_posix())
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "stage_contract.json"
    ]
    _json(
        root / "stage_contract.json",
        {
            "stage": stage,
            "status": "complete",
            "inventory_mode": "closed_world",
            "path_basis": "stage_relative",
            **IDENTITY,
            "outputs": outputs,
        },
    )


def _build_sources(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    run_root.mkdir()
    run_inputs = run_root / "run_inputs"
    snapshot = run_inputs / "canonical_config_snapshot.yaml"
    snapshot.parent.mkdir()
    snapshot.write_text("manuscript_final: {}\n", encoding="utf-8")
    _json(
        run_inputs / "input_contract.json",
        {
            "schema_version": 1,
            "contract_kind": "manuscript_run_inputs",
            "status": "complete",
            "inventory_mode": "closed_world",
            "path_basis": "run_inputs_relative",
            **IDENTITY,
            "n_snapshots": 1,
            "snapshots": [
                {
                    "logical_name": "canonical_config",
                    "input_kind": "canonical_config",
                    "source_path": "configs/manuscript_final.yaml",
                    "snapshot_path": "canonical_config_snapshot.yaml",
                    "source_sha256": sha256_file(snapshot),
                    "source_size_bytes": snapshot.stat().st_size,
                    "snapshot_sha256": sha256_file(snapshot),
                    "snapshot_size_bytes": snapshot.stat().st_size,
                }
            ],
        },
    )
    _json(
        run_root / "shared_folds" / "fold_contract.json",
        _identity(outer_splits=10, inner_splits=5, fold_contract_hash="d" * 64),
    )

    policy_rows = []
    for order, (policy, role, audit_only, macro, qwk, mae) in enumerate(
        (
            ("canonical_primary", "canonical_primary", False, 0.61, 0.55, 0.24),
            ("sensitive_free", "governed_sensitivity", False, 0.59, 0.52, 0.26),
            ("full_feature_upper_bound", "diagnostic_upper_bound", True, 0.70, 0.65, 0.18),
        ),
        start=1,
    ):
        policy_rows.append(
            _identity(
                policy=policy,
                policy_order=order,
                role=role,
                audit_only=audit_only,
                macro_f1_oof=macro,
                macro_f1_ci_low=macro - 0.03,
                macro_f1_ci_high=macro + 0.03,
                quadratic_weighted_kappa_oof=qwk,
                ordinal_mae_oof=mae,
                resample_hash="e" * 64,
            )
        )
    _csv(run_root / "policy_ablation" / "figure_leakage_policy_tradeoff_source.csv", policy_rows)

    summary_rows = []
    for model_index, model in enumerate(("xgboost", "logistic_regression", "random_forest", "lightgbm")):
        for metric_index, metric in enumerate(("macro_f1", "quadratic_weighted_kappa")):
            value = 0.55 + model_index * 0.02 + metric_index * 0.01
            summary_rows.append(
                _identity(system_id=model, metric=metric, point_estimate=value, ci_low=value - 0.03, ci_high=value + 0.03)
            )
    _csv(run_root / "model_benchmarks" / "model_summary.csv", summary_rows)
    _csv(
        run_root / "model_benchmarks" / "paired_model_differences.csv",
        [
            _identity(
                comparison_id="logistic_regression_minus_xgboost",
                metric="macro_f1",
                improvement_oriented_difference=-0.04,
                improvement_ci_low=-0.07,
                improvement_ci_high=-0.01,
            )
        ],
    )
    _json(
        run_root / "model_benchmarks" / "baseline_xgboost_gate.json",
        _identity(
            gate_metric="macro_f1",
            comparison_direction="baseline_improvement_over_xgboost",
            trigger_rule="point_estimate_gt_zero_and_paired_ci_low_gt_zero",
            gate_triggered=False,
            triggered_comparisons=[],
            n_resamples=5000,
            resample_hash="e" * 64,
        ),
    )

    bin_rows = []
    for method in ("raw", "sigmoid"):
        for label in (2, 3, 4):
            for number, predicted in enumerate((0.1, 0.4, 0.7), start=1):
                bin_rows.append(
                    _identity(
                        method=method,
                        class_label=label,
                        bin=number,
                        mean_predicted_probability=predicted,
                        observed_frequency=min(0.98, predicted + (0.03 if method == "raw" else 0.01)),
                        n_samples=20,
                    )
                )
    _csv(run_root / "sigmoid_calibration" / "calibration_bins.csv", bin_rows)
    _csv(
        run_root / "sigmoid_calibration" / "calibration_method_comparison.csv",
        [_identity(method="sigmoid", selected=True), _identity(method="raw", selected=False)],
    )
    interval_rows = []
    for method in ("raw", "sigmoid"):
        for metric, value in (("nll_log_loss", 0.44), ("multiclass_brier", 0.27), ("ece_confidence", 0.06)):
            estimate = value + (0.05 if method == "raw" else 0)
            interval_rows.append(
                _identity(system_id=method, metric=metric, point_estimate=estimate, ci_low=estimate - 0.02, ci_high=estimate + 0.02)
            )
    _csv(run_root / "sigmoid_calibration" / "calibration_metric_intervals.csv", interval_rows)
    _csv(
        run_root / "sigmoid_calibration" / "calibration_paired_differences.csv",
        [_identity(comparison_id="sigmoid_minus_raw", metric="nll_log_loss", improvement_oriented_difference=0.05)],
    )
    _json(
        run_root / "sigmoid_calibration" / "calibration_figure_source.json",
        _identity(selected_method="sigmoid", selection_rule="predeclared", n_bins=10, n_resamples=5000, resample_hash="e" * 64),
    )

    features = ("SignalA", "SignalB", "SignalC", "SignalD")
    _csv(
        run_root / "oof_shap" / "global_grouped_shap_importance.csv",
        [_identity(feature=feature, mean_abs_grouped_shap=1 / rank, rank=rank) for rank, feature in enumerate(features, start=1)],
    )
    ranking_rows = []
    for fold in range(1, 11):
        rotated = features[fold % len(features) :] + features[: fold % len(features)]
        ranking_rows.extend(
            _identity(outer_fold=fold, feature=feature, mean_abs_grouped_shap=1 / rank, rank=rank)
            for rank, feature in enumerate(rotated, start=1)
        )
    _csv(run_root / "oof_shap" / "fold_feature_rankings.csv", ranking_rows)
    _csv(
        run_root / "oof_shap" / "shap_stability_pairwise.csv",
        [
            _identity(outer_fold_a=fold_a, outer_fold_b=fold_b, top_k=3, top_k_jaccard=0.75, spearman_all_features=0.8)
            for fold_a in range(1, 11)
            for fold_b in range(fold_a + 1, 11)
        ],
    )
    _csv(
        run_root / "oof_shap" / "shap_stability_summary.csv",
        [
            _identity(
                top_k=3,
                n_outer_folds=10,
                n_fold_pairs=45,
                jaccard_mean=0.75,
                jaccard_std=0.1,
                jaccard_median=0.75,
                jaccard_min=0.5,
                jaccard_max=1.0,
                spearman_mean=0.8,
                spearman_std=0.1,
                spearman_median=0.8,
                spearman_min=0.6,
                spearman_max=1.0,
                confidence_interval_applicable=False,
            ),
        ],
    )
    _json(
        run_root / "oof_shap" / "shap_metadata.json",
        _identity(
            n_samples=120,
            n_outer_folds=10,
            n_raw_features=4,
            model_set_sha256="f" * 64,
            confidence_interval_for_fold_pairs=False,
            attribution_warning="Model attribution, not causality.",
            temporality_warning="Timing requires separate verification.",
        ),
    )

    support_rows = [
        _identity(support_scale="raw", target_column="PerformanceScore", target_value="Exceeds", count=12, proportion=0.1, n_total=120),
        _identity(support_scale="mapped", target_column="PerformanceRating", target_value=2, count=20, proportion=1 / 6, n_total=120),
        _identity(support_scale="mapped", target_column="PerformanceRating", target_value=3, count=80, proportion=2 / 3, n_total=120),
        _identity(support_scale="mapped", target_column="PerformanceRating", target_value=4, count=20, proportion=1 / 6, n_total=120),
    ]
    _csv(run_root / "external_replication" / "target_support.csv", support_rows)
    raw_rows = []
    for system in ("conservative_primary", "department_including_audit"):
        for metric, value in (("macro_f1", 0.66), ("quadratic_weighted_kappa", 0.54)):
            raw_rows.append(
                _identity(system_id=system, metric=metric, point_estimate=value, ci_low=value - 0.04, ci_high=value + 0.04, n_samples=120, n_resamples=5000)
            )
    _csv(run_root / "external_replication" / "raw_metric_intervals.csv", raw_rows)
    calibration_rows = []
    for system in ("raw", "sigmoid"):
        for metric, value in (("macro_f1", 0.64), ("quadratic_weighted_kappa", 0.58)):
            estimate = value + (0.02 if system == "sigmoid" else 0)
            calibration_rows.append(
                _identity(system_id=system, metric=metric, point_estimate=estimate, ci_low=estimate - 0.04, ci_high=estimate + 0.04, n_samples=120, n_resamples=5000)
            )
    _csv(run_root / "external_replication" / "calibration_metric_intervals.csv", calibration_rows)
    _csv(
        run_root / "external_replication" / "calibration_paired_differences.csv",
        [_identity(comparison_id="sigmoid_minus_raw", metric="macro_f1", improvement_oriented_difference=0.02)],
    )
    _csv(
        run_root / "external_replication" / "policy_pairwise_differences.csv",
        [
            _identity(
                comparison_id="department_including_audit_minus_conservative_primary",
                metric="macro_f1",
                improvement_oriented_difference=0.01,
                improvement_ci_low=-0.02,
                improvement_ci_high=0.04,
            )
        ],
    )
    _json(
        run_root / "external_replication" / "external_replication_metadata.json",
        _identity(
            scope="independent_external_replication",
            role="mapped_target_replication",
            task_type="ordinal_multiclass_performance",
            labels=[2, 3, 4],
            primary_policy="conservative_primary",
            claim_boundary="Independent mapped-target replication only.",
            network_calls=0,
            paid_api_calls=0,
        ),
    )

    for stage in ("shared_folds", "policy_ablation", "model_benchmarks", "sigmoid_calibration", "oof_shap", "external_replication"):
        _stage_contract(run_root, stage)
    return run_root


def test_generator_publishes_atomic_source_bound_package_that_passes_validator(tmp_path: Path) -> None:
    run_root = _build_sources(tmp_path)
    output = run_root / "core_figures"
    result = run(
        "configs/manuscript_final.yaml",
        run_root=run_root,
        output_dir=output,
        **IDENTITY,
    )
    assert result["output"] == output
    assert len(result["files"]) == 29
    assert len(list(output.glob("figure_*.png"))) == 7
    assert len(list(output.glob("figure_*.svg"))) == 7
    _stage_contract(run_root, "core_figures")
    validation = validate_core_figure_package(
        output,
        run_root=run_root,
        config=CONFIG,
        **IDENTITY,
    )
    assert validation["status"] == "passed"
    assert validation["artifact_count_excluding_stage_contract"] == 29
    assert not list(run_root.glob("core_figures.__staging__.*"))


def test_generator_rejects_mixed_source_identity_and_preserves_forensics(tmp_path: Path) -> None:
    run_root = _build_sources(tmp_path)
    source = run_root / "model_benchmarks" / "model_summary.csv"
    frame = pd.read_csv(source)
    frame.loc[0, "run_id"] = "wrong-run"
    frame.to_csv(source, index=False)
    with pytest.raises(CoreFigureGenerationError, match="mixed or wrong run_id"):
        run(
            "configs/manuscript_final.yaml",
            run_root=run_root,
            output_dir=run_root / "core_figures",
            **IDENTITY,
        )
    assert not (run_root / "core_figures").exists()
    assert len(list(run_root.glob("core_figures.__staging__.*"))) == 1


def test_builder_registers_core_figures_as_atomic_current_run_stage() -> None:
    assert "core_figures" in STAGE_RUNNERS
    assert "core_figures" in ATOMIC_DIRECTORY_STAGE_RUNNERS
    assert STAGE_ORPHAN_PREFIXES["core_figures"] == ("core_figures.__staging__.",)
