from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    mean_absolute_error,
)

from src.experiments import subgroup_proxy_use_v3 as diagnostics
from src.experiments.subgroup_proxy_use_v3 import (
    EXPECTED_LOCAL_FILES,
    _prepare_inputs,
    evaluate_subgroup_proxy_use_v3,
    metric_bundle_v3,
    preflight_subgroup_proxy_use_v3,
)
from src.governance.subgroup_proxy_use_contract_v3 import (
    ATTRIBUTES,
    DEFAULT_SUBGROUP_PROXY_USE_CONTRACT,
    LABELS,
    METRICS,
    PRIMARY_SYSTEM,
    SYSTEMS,
    SubgroupProxyUseContractV3Error,
    validate_subgroup_proxy_use_contract_v3,
)


@pytest.fixture(scope="module")
def real_inputs():
    return _prepare_inputs(DEFAULT_SUBGROUP_PROXY_USE_CONTRACT)


@pytest.fixture(scope="module")
def real_result(real_inputs):
    return evaluate_subgroup_proxy_use_v3(real_inputs)


def test_contract_validates_exact_canonical_sources() -> None:
    receipt = validate_subgroup_proxy_use_contract_v3()
    assert receipt["status"] == "validated"
    assert receipt["oof_rows"] == 3600
    assert receipt["samples_per_system"] == 1200
    assert receipt["reconstructability_metric_rows"] == 6
    assert receipt["new_model_fit_calls"] == 0
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0


def test_contract_rejects_relaxed_claim_boundary(tmp_path: Path) -> None:
    payload = json.loads(Path(DEFAULT_SUBGROUP_PROXY_USE_CONTRACT).read_text(encoding="utf-8"))
    payload["subgroup_audit"]["fairness_certification_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SubgroupProxyUseContractV3Error, match="must be false"):
        validate_subgroup_proxy_use_contract_v3(path)


def test_metric_bundle_matches_standard_metric_definitions() -> None:
    y_true = np.asarray([2, 2, 3, 3, 3, 4, 4, 4])
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.3, 0.6, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.2, 0.6, 0.2],
            [0.05, 0.15, 0.8],
        ]
    )
    y_pred = np.asarray(LABELS)[np.argmax(probabilities, axis=1)]
    observed, denominators = metric_bundle_v3(y_true, y_pred, probabilities)
    assert observed["macro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    )
    assert observed["balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(y_true, y_pred)
    )
    assert observed["quadratic_weighted_kappa"] == pytest.approx(
        cohen_kappa_score(y_true, y_pred, labels=LABELS, weights="quadratic")
    )
    assert observed["ordinal_mae"] == pytest.approx(mean_absolute_error(y_true, y_pred))
    assert observed["log_loss"] == pytest.approx(log_loss(y_true, probabilities, labels=LABELS))
    one_hot = np.eye(3)[np.searchsorted(np.asarray(LABELS), y_true)]
    assert observed["multiclass_brier"] == pytest.approx(
        np.mean(np.sum(np.square(probabilities - one_hot), axis=1))
    )
    assert denominators["recall_class_2"] == 2
    assert denominators["balanced_accuracy"] == 2


def test_real_subgroup_grid_is_complete_and_support_explicit(real_result) -> None:
    grid = real_result.subgroup_metric_grid
    assert len(grid) == 2025
    assert set(grid["system_id"]) == set(SYSTEMS)
    assert set(grid["support_threshold"]) == {20, 30, 50}
    assert set(grid["attribute"]) == set(ATTRIBUTES)
    assert set(grid["metric"]) == set(METRICS)
    assert grid.groupby(["system_id", "support_threshold", "attribute", "group"])[
        "metric"
    ].nunique().eq(9).all()
    assert set(grid["support_status"]) == {
        "eligible_descriptive_estimate",
        "insufficient_group_support",
        "insufficient_true_class_support",
    }
    unsupported = grid[~grid["eligible_for_gap"]]
    assert unsupported["point_estimate"].isna().all()


def test_age_bins_and_threshold_sensitivity_are_preserved(real_result) -> None:
    age = real_result.subgroup_metric_grid[
        (real_result.subgroup_metric_grid["system_id"] == PRIMARY_SYSTEM)
        & (real_result.subgroup_metric_grid["support_threshold"] == 20)
        & (real_result.subgroup_metric_grid["attribute"] == "Age")
        & (real_result.subgroup_metric_grid["metric"] == "macro_f1")
    ]
    assert age["group"].tolist() == ["18-29", "30-39", "40-49", "50-59", "60+"]
    assert age["group_n"].tolist() == [267, 506, 287, 137, 3]
    data_science = real_result.subgroup_metric_grid[
        (real_result.subgroup_metric_grid["system_id"] == PRIMARY_SYSTEM)
        & (real_result.subgroup_metric_grid["attribute"] == "EmpDepartment")
        & (real_result.subgroup_metric_grid["group"] == "Data Science")
        & (real_result.subgroup_metric_grid["metric"] == "macro_f1")
    ].sort_values("support_threshold")
    assert data_science["group_n"].tolist() == [20, 20, 20]
    assert data_science["eligible_for_gap"].tolist() == [True, False, False]


def test_all_gap_cells_are_retained_without_single_winner_selection(real_result) -> None:
    gaps = real_result.subgroup_gap_sensitivity
    assert len(gaps) == 486
    assert gaps.groupby(["system_id", "support_threshold", "attribute"])["metric"].nunique().eq(9).all()
    assert gaps["maximum_gap_selection_scope"].str.contains("all_cells_reported").all()
    assert set(gaps["status"]).issubset(
        {"estimable_descriptive_gap", "fewer_than_two_eligible_groups"}
    )


def test_primary_bootstrap_is_complete_multiplicity_aware_and_deterministic(real_result) -> None:
    intervals = real_result.primary_gap_bootstrap_intervals
    assert len(intervals) == 162
    assert intervals["n_resamples"].eq(5000).all()
    assert intervals["n_complete_familywise_draws"].eq(5000).all()
    assert intervals["resample_hash"].nunique() == 1
    assert intervals["resample_hash"].iloc[0] == (
        "d16aadb56f2dde124df62387447bb01ddf51236f7f9e9e01b6218e8f6265f646"
    )
    estimable = intervals[intervals["status"].str.startswith("exploratory")]
    assert (
        estimable["simultaneous_ci_low"] <= estimable["gap_max_minus_min"]
    ).all()
    assert (
        estimable["simultaneous_ci_high"] >= estimable["gap_max_minus_min"]
    ).all()
    assert estimable["interval_scope"].str.contains("familywise").all()
    assert not estimable["model_training_variability_included"].any()


def test_proxy_prediction_changes_are_paired_and_department_complete(real_result) -> None:
    sample = real_result.proxy_prediction_change_sample
    aggregate = real_result.proxy_prediction_change_by_department
    assert len(sample) == 1200
    assert sample["sample_index"].nunique() == 1200
    assert np.allclose(
        sample[["delta_prob_class_2", "delta_prob_class_3", "delta_prob_class_4"]].sum(axis=1),
        0.0,
        atol=1e-12,
    )
    assert len(aggregate) == 7
    assert aggregate["n_samples"].sum() == 2400
    overall = aggregate[aggregate["scope"] == "overall"].iloc[0]
    assert overall["n_samples"] == 1200
    assert overall["mean_total_variation"] == pytest.approx(0.0905677, abs=1e-6)
    assert overall["prediction_change_rate"] == pytest.approx(0.1075)
    assert overall["claim_boundary"].startswith("performance_model_dependence")


def test_jobrole_permutation_uses_both_prespecified_schemes(real_result) -> None:
    sample = real_result.jobrole_permutation_sample
    repetitions = real_result.jobrole_permutation_repetition
    summary = real_result.jobrole_permutation_summary
    assert len(sample) == 48000
    assert len(repetitions) == 40
    assert len(summary) == 2
    assert repetitions.groupby("scheme")["seed"].nunique().eq(20).all()
    assert repetitions["n_samples"].eq(1200).all()
    assert repetitions["job_role_value_changed_fraction"].between(0.0, 1.0).all()
    assert np.isfinite(
        repetitions[
            [
                "mean_total_variation",
                "prediction_change_rate",
                "mean_original_predicted_class_raw_margin_drop",
            ]
        ].to_numpy(float)
    ).all()
    means = summary.set_index("scheme")
    assert (
        means.loc["marginal_within_outer_test_fold", "mean_total_variation_mean"]
        > means.loc[
            "department_conditional_within_outer_test_fold_and_department",
            "mean_total_variation_mean",
        ]
    )


def test_reconstructability_is_kept_separate_from_performance_use(real_result) -> None:
    metrics = real_result.department_reconstructability_metrics
    differences = real_result.department_reconstructability_differences
    assert len(metrics) == 6 and len(differences) == 3
    assert not metrics["performance_dependence_claim_allowed"].any()
    assert not differences["performance_dependence_claim_allowed"].any()
    assert metrics["v3_interpretation"].str.contains("not_performance_model").all()


def test_preflight_is_offline_and_fit_free() -> None:
    receipt = preflight_subgroup_proxy_use_v3()
    assert receipt["status"] == "preflight_passed"
    assert receipt["oof_rows"] == 3600
    assert receipt["persisted_outer_model_count"] == 10
    assert receipt["planned_bootstrap_resamples"] == 5000
    assert receipt["planned_permutation_repetitions"] == 40
    assert receipt["planned_new_model_fit_calls"] == 0
    assert receipt["network_calls"] == receipt["paid_api_calls"] == 0


def test_runner_atomically_publishes_declared_local_inventory(
    real_inputs, real_result, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = {"commit": "a" * 40, "branch": "test"}
    monkeypatch.setattr(diagnostics, "_clean_git_identity", lambda: identity)
    monkeypatch.setattr(diagnostics, "source_tree_hash", lambda _root: "b" * 64)
    monkeypatch.setattr(diagnostics, "_prepare_inputs", lambda _path: real_inputs)
    monkeypatch.setattr(diagnostics, "evaluate_subgroup_proxy_use_v3", lambda _inputs: real_result)
    output = tmp_path / "subgroup_proxy_use"
    receipt = diagnostics.run_subgroup_proxy_use_v3(
        contract_path=DEFAULT_SUBGROUP_PROXY_USE_CONTRACT,
        output_dir=output,
        run_id="diagnostic_incomplete_never_canonical",
    )
    assert receipt["status"] == "complete"
    assert {path.name for path in output.iterdir()} == EXPECTED_LOCAL_FILES
    metadata = json.loads((output / "stage_metadata.json").read_text(encoding="utf-8"))
    assert metadata["subgroup_metric_rows"] == 2025
    assert metadata["jobrole_permutation_sample_rows"] == 48000
    assert metadata["new_performance_model_fit_calls"] == 0
    assert metadata["network_calls"] == metadata["paid_api_calls"] == 0
