from __future__ import annotations

import json
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.experiments import manuscript_fairness_proxy as fairness_proxy
from src.experiments.shared_folds import generate_shared_folds
from src.models.oof_bootstrap import BootstrapProtocol, generate_stratified_resample_indices
from src.utils.config_loader import load_config


LABELS = (2, 3, 4)


def _paired_subgroup_fixture(
    eligible_precision_groups: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """Build ten fold/label strata with three audit groups in every stratum."""

    data_rows: list[dict[str, str]] = []
    base_rows: list[dict[str, int | float | str]] = []
    sample_index = 0
    for outer_fold in range(1, 11):
        for group in ("A", "B", "C"):
            for y_true in LABELS:
                data_rows.append({"Gender": group})
                base_rows.append(
                    {
                        "sample_index": sample_index,
                        "outer_fold": outer_fold,
                        "y_true": y_true,
                        "group": group,
                    }
                )
                sample_index += 1

    prediction_rows: list[dict[str, int | float | str]] = []
    for policy in fairness_proxy.REQUIRED_POLICY_COMPARISONS:
        allowed = eligible_precision_groups[policy]
        for base in base_rows:
            y_true = int(base["y_true"])
            group = str(base["group"])
            y_pred = 4 if y_true == 4 and group in allowed else (3 if y_true == 4 else y_true)
            prediction_rows.append(
                {
                    "policy": policy,
                    "sample_index": int(base["sample_index"]),
                    "outer_fold": int(base["outer_fold"]),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    **{
                        f"prob_class_{label}": 0.8 if label == y_pred else 0.1
                        for label in LABELS
                    },
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    data = pd.DataFrame(data_rows)
    group_metrics = fairness_proxy.compute_group_metric_rows(
        predictions,
        data,
        labels=LABELS,
        attributes=["Gender"],
        transforms={},
        sensitive_attributes={"Gender"},
        minimum_group_support=20,
        minimum_class_denominator=5,
    )
    base = predictions[
        predictions["policy"].eq(fairness_proxy.ALIAS_POLICY)
    ][["sample_index", "outer_fold", "y_true"]]
    plan = generate_stratified_resample_indices(
        base,
        BootstrapProtocol(
            n_resamples=41,
            confidence_level=0.95,
            seed=117,
            strata_columns=("outer_fold", "y_true"),
            method="paired_stratified_percentile",
            quantile_method="linear",
        ),
    )
    return predictions, data, group_metrics, plan


def _subgroup_evidence(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    group_metrics: pd.DataFrame,
    plan: object,
):
    return fairness_proxy.compute_subgroup_bootstrap_evidence(
        group_metrics,
        predictions,
        data,
        labels=LABELS,
        attributes=["Gender"],
        transforms={},
        sensitive_attributes={"Gender"},
        plan=plan,
        confidence_level=0.95,
        minimum_valid_fraction=0.8,
        wide_interval_threshold=1.0,
    )


def test_paired_precision_difference_requires_two_common_eligible_groups() -> None:
    predictions, data, group_metrics, plan = _paired_subgroup_fixture(
        {
            fairness_proxy.ALIAS_POLICY: {"A", "B"},
            fairness_proxy.PRIMARY_POLICY: {"B", "C"},
            fairness_proxy.STRICT_POLICY: {"B", "C"},
        }
    )

    evidence = _subgroup_evidence(predictions, data, group_metrics, plan)
    comparison = evidence.paired_differences
    row = comparison[
        comparison["comparison_id"].eq(
            f"{fairness_proxy.ALIAS_POLICY}__minus__{fairness_proxy.PRIMARY_POLICY}"
        )
        & comparison["attribute"].eq("Gender")
        & comparison["metric"].eq("precision")
        & comparison["class_label"].eq(4)
    ].iloc[0]

    assert json.loads(row["common_eligible_groups_json"]) == ["B"]
    assert row["n_common_groups"] == 1
    assert row["paired_estimate_status"] == "insufficient_common_subgroup_or_metric_support"
    assert np.isnan(row["gap_difference"])
    assert np.isnan(row["ci_low"])
    assert np.isnan(row["ci_high"])
    assert row["valid_bootstrap_samples"] == 0


def test_identical_policy_systems_have_exact_zero_paired_estimates_independent_of_row_order() -> None:
    identical = {
        policy: {"A", "B", "C"} for policy in fairness_proxy.REQUIRED_POLICY_COMPARISONS
    }
    predictions, data, group_metrics, plan = _paired_subgroup_fixture(identical)

    first = _subgroup_evidence(predictions, data, group_metrics, plan).paired_differences
    reordered = _subgroup_evidence(
        predictions.sample(frac=1.0, random_state=7).reset_index(drop=True),
        data,
        group_metrics.sample(frac=1.0, random_state=11).reset_index(drop=True),
        plan,
    ).paired_differences

    columns = [
        "comparison_id",
        "attribute",
        "metric",
        "class_label",
        "common_eligible_groups_json",
        "n_common_groups",
        "gap_difference",
        "ci_low",
        "ci_high",
        "valid_bootstrap_samples",
        "paired_estimate_status",
    ]
    sort_by = ["comparison_id", "attribute", "metric", "class_label"]
    first = first.loc[:, columns].sort_values(sort_by, na_position="first").reset_index(drop=True)
    reordered = (
        reordered.loc[:, columns]
        .sort_values(sort_by, na_position="first")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(first, reordered, check_exact=True)

    supported = first[
        first["paired_estimate_status"].eq("support_sufficient_descriptive_estimate")
    ]
    assert not supported.empty
    np.testing.assert_array_equal(supported["gap_difference"].to_numpy(), 0.0)
    np.testing.assert_array_equal(supported["ci_low"].to_numpy(), 0.0)
    np.testing.assert_array_equal(supported["ci_high"].to_numpy(), 0.0)


def test_subgroup_gap_draw_batches_are_exactly_equivalent() -> None:
    identical = {
        policy: {"A", "B", "C"} for policy in fairness_proxy.REQUIRED_POLICY_COMPARISONS
    }
    predictions, data, group_metrics, plan = _paired_subgroup_fixture(identical)
    scoped_predictions = predictions[
        predictions["policy"].eq(fairness_proxy.PRIMARY_POLICY)
    ].sort_values("sample_index")
    scoped_metrics = group_metrics[
        group_metrics["policy"].eq(fairness_proxy.PRIMARY_POLICY)
        & group_metrics["attribute"].eq("Gender")
    ]
    group_values = data.loc[
        scoped_predictions["sample_index"].to_numpy(int), "Gender"
    ].to_numpy(str)

    one_batch = fairness_proxy._fixed_group_gap_draws(
        scoped_predictions,
        group_values,
        scoped_metrics,
        labels=LABELS,
        plan=plan,
        batch_size=plan.indices.shape[0],
    )
    many_batches = fairness_proxy._fixed_group_gap_draws(
        scoped_predictions,
        group_values,
        scoped_metrics,
        labels=LABELS,
        plan=plan,
        batch_size=7,
    )

    assert one_batch.keys() == many_batches.keys()
    for key in one_batch:
        np.testing.assert_allclose(
            one_batch[key], many_batches[key], rtol=0.0, atol=0.0, equal_nan=True
        )


def test_proxy_metric_draw_batches_are_exactly_equivalent() -> None:
    y_true = np.tile(np.arange(3), 8)
    y_pred = np.roll(y_true, 3)
    indices = np.random.default_rng(913).integers(0, len(y_true), size=(23, len(y_true)))

    one_batch = fairness_proxy._proxy_metric_draws(
        y_true,
        y_pred,
        range(3),
        indices,
        batch_size=len(indices),
    )
    many_batches = fairness_proxy._proxy_metric_draws(
        y_true,
        y_pred,
        range(3),
        indices,
        batch_size=4,
    )

    assert one_batch.keys() == many_batches.keys()
    for metric in one_batch:
        np.testing.assert_allclose(
            one_batch[metric], many_batches[metric], rtol=0.0, atol=0.0, equal_nan=True
        )


def _synthetic_proxy_frame(n_rows: int = 60) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(n_rows):
        rows.append(
            {
                "EmpNumber": f"EMP-{index:03d}",
                "PerformanceRating": LABELS[index % len(LABELS)],
                "Age": 21 + index % 38,
                "Gender": ("Female", "Male")[index % 2],
                "MaritalStatus": ("Single", "Married", "Divorced")[index % 3],
                "EmpDepartment": f"Department-{index % 6}",
                "EmpJobRole": f"Role-{index % 6}",
                "EmpLastSalaryHikePercent": 10 + index % 12,
                "Attrition": ("No", "Yes")[index % 5 == 0],
                "EmpJobSatisfaction": 1 + index % 4,
                "EducationBackground": f"Education-{index % 4}",
                "BusinessTravelFrequency": f"Travel-{index % 3}",
            }
        )
    return pd.DataFrame(rows)


def test_real_sklearn_proxy_uses_two_unique_shared_fold_systems_and_no_target_or_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _synthetic_proxy_frame()
    settings = load_config("configs/manuscript_final.yaml")["manuscript_final"]
    proxy_contract = copy.deepcopy(settings["proxy_analysis"])
    proxy_contract["watchlist"] = ["EmpJobSatisfaction"]
    config_hash = "a" * 64
    scientific_input_hash = "b" * 64
    dataset_sha256 = "d" * 64
    folds = generate_shared_folds(
        data,
        target_column="PerformanceRating",
        id_column="EmpNumber",
        run_id="synthetic-proxy-test",
        config_hash=config_hash,
        scientific_input_hash=scientific_input_hash,
        dataset_key="synthetic_inx",
        dataset_sha256=dataset_sha256,
        outer_splits=10,
        inner_splits=5,
        seed=42,
        inner_seed=117,
    )
    identity = {
        "run_id": "synthetic-proxy-test",
        "config_hash": config_hash,
        "scientific_input_hash": scientific_input_hash,
        "fold_contract_hash": str(folds.contract["fold_contract_hash"]),
        "xgboost_model_set_sha256": "c" * 64,
        "dataset_sha256": dataset_sha256,
    }

    fitted_pipelines = []
    real_pipeline_factory = fairness_proxy._proxy_pipeline

    def recording_pipeline(frame, proxy, seed):
        pipeline = real_pipeline_factory(frame, proxy, seed)
        fitted_pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(fairness_proxy, "_proxy_pipeline", recording_pipeline)
    monkeypatch.setattr(fairness_proxy, "REQUIRED_BOOTSTRAP_RESAMPLES", 31)
    monkeypatch.setattr(
        fairness_proxy,
        "feature_proxy_associations",
        lambda frame, target, random_state: pd.DataFrame(
            {
                "feature": frame.columns,
                "association_type": "unit_test_not_evaluated",
                "mutual_info": 0.0,
                "cramers_v": np.nan,
                "proxy_watchlist": False,
            }
        ),
    )

    evidence = fairness_proxy.generate_proxy_oof_evidence(
        data,
        settings,
        bundle=SimpleNamespace(folds=folds),
        identity=identity,
        proxy=proxy_contract,
    )

    assert set(evidence.feature_contracts["system_id"]) == set(
        fairness_proxy.UNIQUE_PROXY_POLICIES
    )
    assert len(evidence.fold_metrics) == 20
    assert not evidence.fold_metrics.duplicated(["system_id", "outer_fold"]).any()
    assert len(fitted_pipelines) == 20

    oof = evidence.oof_predictions
    assert set(oof["system_id"]) == set(fairness_proxy.UNIQUE_PROXY_POLICIES)
    assert len(oof) == 2 * len(data)
    assert oof.groupby(["system_id", "sample_index"]).size().eq(1).all()
    assert oof.groupby("system_id")["outer_fold"].nunique().eq(10).all()
    assert set(oof["task_type"]) == {"nominal_multiclass_proxy_diagnostic"}

    alias = evidence.equivalence[
        evidence.equivalence["reported_policy"].eq(fairness_proxy.ALIAS_POLICY)
    ].iloc[0]
    effective = evidence.equivalence[
        evidence.equivalence["reported_policy"].eq(fairness_proxy.PRIMARY_POLICY)
    ].iloc[0]
    assert alias["effective_system_id"] == fairness_proxy.PRIMARY_POLICY
    assert not bool(alias["fit_performed"])
    assert alias["predictor_contract_sha256"] == effective["predictor_contract_sha256"]

    forbidden = {"EmpDepartment", "PerformanceRating", "EmpNumber"}
    for row in evidence.feature_contracts.itertuples(index=False):
        raw_features = set(json.loads(row.feature_columns_json))
        assert forbidden.isdisjoint(raw_features)
        assert bool(row.proxy_target_absent_from_predictors)
    for pipeline in fitted_pipelines:
        assert forbidden.isdisjoint(set(pipeline.feature_names_in_))
        transformed = pipeline.named_steps["preprocessor"].get_feature_names_out()
        assert not any(any(name in value for name in forbidden) for value in transformed)

    flagged = evidence.associations[evidence.associations["proxy_watchlist"].astype(bool)]
    assert set(flagged["feature"]) == {"EmpJobSatisfaction"}
    assert set(evidence.associations["watchlist_source"]) == {"proxy_analysis.watchlist"}
    assert set(evidence.metric_intervals["task_type"]) == {
        "nominal_multiclass_proxy_diagnostic"
    }
    assert evidence.metric_intervals["minimum_proxy_target_class_support"].eq(10).all()
    assert evidence.metric_intervals["zero_support_outer_test_cells"].ge(0).all()
    assert evidence.metric_intervals["conditional_inference_note"].eq(
        fairness_proxy.CONDITIONAL_INFERENCE_NOTE
    ).all()
    assert evidence.label_mapping["task_type"] == "nominal_multiclass_proxy_diagnostic"
    assert evidence.label_mapping["minimum_class_support"] == 10
    reported = fairness_proxy._reported_proxy_comparison(
        evidence.metric_intervals, evidence.equivalence
    )
    manuscript_table = fairness_proxy.manuscript_fairness_proxy_table(
        pd.DataFrame(columns=["gap", "policy"]), reported
    )
    proxy_rows = manuscript_table[
        manuscript_table["analysis_type"].eq("department_reconstructability_proxy_risk")
    ]
    assert proxy_rows["minimum_subgroup_support"].eq(10).all()
    assert proxy_rows["zero_support_outer_test_cells"].notna().all()
    assert set(proxy_rows["task_type"]) == {"nominal_multiclass_proxy_diagnostic"}


def test_proxy_label_mapping_is_bound_to_complete_scientific_identity() -> None:
    identity = {
        "run_id": "run",
        "config_hash": "a" * 64,
        "scientific_input_hash": "b" * 64,
        "fold_contract_hash": "c" * 64,
        "xgboost_model_set_sha256": "d" * 64,
        "dataset_sha256": "e" * 64,
    }
    mapping = {
        "task_type": "nominal_multiclass_proxy_diagnostic",
        "proxy_target": "EmpDepartment",
    }

    payload = fairness_proxy._identity_bound_mapping(mapping, identity)

    assert all(payload[key] == value for key, value in identity.items())
    assert payload["proxy_target"] == "EmpDepartment"
