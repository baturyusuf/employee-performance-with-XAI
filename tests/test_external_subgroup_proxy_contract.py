from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.experiments.hrdataset_replication_diagnostics import (
    AuditAttributeSpec,
    HRDatasetDiagnosticsError,
    PROXY_LIMITATION,
    REQUIRED_BOOTSTRAP_RESAMPLES,
    SUBGROUP_LIMITATION,
    ReplicationIdentity,
    compute_proxy_reconstructability,
    compute_support_aware_subgroup_diagnostics,
)


LABELS = (2, 3, 4)


def _identity(run_id: str) -> ReplicationIdentity:
    return ReplicationIdentity(
        run_id=run_id,
        config_hash="a" * 64,
        scientific_input_hash="b" * 64,
        dataset_sha256="c" * 64,
        schema_mapping_sha256="d" * 64,
        fold_contract_hash="e" * 64,
        feature_policy_contract_sha256="f" * 64,
        model_set_sha256="1" * 64,
    )


def _attach_identity(frame: pd.DataFrame, identity: ReplicationIdentity) -> pd.DataFrame:
    output = frame.copy()
    for field_name, value in identity.as_dict().items():
        output[field_name] = value
    return output


def _subgroup_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ReplicationIdentity]:
    n_samples = 60
    sample_ids = np.arange(n_samples)
    y_true = np.tile(np.asarray(LABELS, dtype=int), n_samples // len(LABELS))
    y_pred = y_true.copy()
    error = np.arange(n_samples) % 7 == 0
    y_pred[error] = np.asarray(LABELS)[
        (np.searchsorted(np.asarray(LABELS), y_true[error]) + 1) % len(LABELS)
    ]
    probability = np.full((n_samples, len(LABELS)), 0.1, dtype=float)
    for row_index, predicted in enumerate(y_pred):
        probability[row_index, LABELS.index(int(predicted))] = 0.8
    predictions = pd.DataFrame(
        {
            "sample_index": sample_ids,
            "outer_fold": np.tile(np.arange(1, 6), n_samples // 5),
            "policy": "external_primary",
            "y_true": y_true,
            "y_pred": y_pred,
            **{
                f"prob_class_{label}": probability[:, label_index]
                for label_index, label in enumerate(LABELS)
            },
        }
    )
    folds = predictions[["sample_index", "outer_fold"]].copy()
    audit = pd.DataFrame(
        {
            "sample_index": sample_ids,
            "GenderAudit": np.where(sample_ids % 2 == 0, "G1", "G2"),
            "OperationalUnit": np.where(sample_ids % 2 == 0, "Unit A", "Unit B"),
            "RareAttribute": np.where(sample_ids == 0, "rare", "common"),
        }
    )
    identity = _identity("subgroup-test")
    return _attach_identity(predictions, identity), folds, audit, identity


def test_support_aware_subgroup_bootstrap_reports_denominators_valid_counts_and_categories() -> None:
    predictions, folds, audit, identity = _subgroup_fixture()
    evidence = compute_support_aware_subgroup_diagnostics(
        oof_predictions=predictions,
        fold_assignments=folds,
        audit_frame=audit,
        attributes=[
            AuditAttributeSpec("GenderAudit", "protected_sensitive"),
            AuditAttributeSpec("OperationalUnit", "exploratory_operational"),
        ],
        identity=identity,
        labels=LABELS,
        minimum_group_support=15,
        minimum_metric_denominator=5,
        n_resamples=REQUIRED_BOOTSTRAP_RESAMPLES,
        batch_size=250,
    )

    groups = evidence.group_metrics
    assert {
        "group_n",
        "metric_denominator",
        "metric_denominator_kind",
        "eligible_for_gap",
        "interpretation_category",
        "limitations",
    }.issubset(groups.columns)
    assert set(groups["interpretation_category"]) == {
        "protected_sensitive",
        "exploratory_operational",
    }
    tpr = groups[groups["metric"] == "true_positive_rate"]
    assert set(tpr["metric_denominator_kind"]) == {"actual_class_rows"}
    assert (tpr["metric_denominator"] >= 0).all()
    assert set(groups["limitations"]) == {SUBGROUP_LIMITATION}

    intervals = evidence.disparity_intervals
    assert not intervals.empty
    assert set(intervals["n_resamples"]) == {5000}
    assert intervals["resample_hash"].str.len().eq(64).all()
    assert intervals["n_valid_bootstrap"].between(0, 5000).all()
    assert intervals["valid_bootstrap_fraction"].between(0.0, 1.0).all()
    finite = intervals.dropna(subset=["point_estimate_gap", "ci_low", "ci_high"])
    assert finite["point_estimate_gap"].between(0.0, 1.0).all()
    assert finite["ci_low"].between(0.0, 1.0).all()
    assert finite["ci_high"].between(0.0, 1.0).all()
    assert (finite["ci_low"] <= finite["ci_high"]).all()
    assert set(intervals["inference_scope"]) == {"pointwise_descriptive"}
    assert set(intervals["multiplicity_adjustment"]) == {"none"}
    assert intervals["limitations"].str.contains("no .*fairness guarantee", case=False).all()
    assert evidence.metadata["n_resamples"] == 5000
    assert evidence.metadata["resample_hash"] == intervals["resample_hash"].iloc[0]


def test_subgroup_rows_with_only_one_supported_group_are_not_estimated_or_headlined() -> None:
    predictions, folds, audit, identity = _subgroup_fixture()
    evidence = compute_support_aware_subgroup_diagnostics(
        oof_predictions=predictions,
        fold_assignments=folds,
        audit_frame=audit,
        attributes=[AuditAttributeSpec("RareAttribute", "exploratory_operational")],
        identity=identity,
        labels=LABELS,
        minimum_group_support=10,
        minimum_metric_denominator=5,
        batch_size=500,
    )

    assert set(evidence.disparity_intervals["estimate_status"]) == {
        "insufficient_subgroup_or_metric_support"
    }
    assert evidence.disparity_intervals["point_estimate_gap"].isna().all()
    assert evidence.disparity_intervals["ci_low"].isna().all()
    assert (evidence.disparity_intervals["n_valid_bootstrap"] == 0).all()
    assert not evidence.disparity_intervals["headline_eligible"].any()


def _proxy_frames(
    departments: list[str], folds: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_ids = np.arange(len(departments))
    features = pd.DataFrame(
        {
            "sample_index": sample_ids,
            "safe_numeric": np.linspace(-1.0, 1.0, len(departments)),
            "safe_category": np.where(sample_ids % 2 == 0, "A", "B"),
            "DeptID": np.arange(len(departments)) % 6,
        }
    )
    fold_frame = pd.DataFrame({"sample_index": sample_ids, "outer_fold": folds})
    audit = pd.DataFrame({"sample_index": sample_ids, "EmpDepartment": departments})
    return features, fold_frame, audit


def test_realistic_singleton_department_support_emits_not_estimated_without_fitting() -> None:
    departments = (
        ["Production"] * 209
        + ["IT/IS"] * 50
        + ["Sales"] * 31
        + ["Software Engineering"] * 11
        + ["Administration"] * 9
        + ["Executive Office"]
    )
    folds = np.tile(np.arange(1, 11), 32)[: len(departments)]
    # Ensure the singleton belongs to an outer test fold, hence that fold's
    # training partition cannot contain Executive Office.
    folds[-1] = 1
    features, fold_frame, audit = _proxy_frames(departments, folds)
    identity = _identity("proxy-singleton-test")

    evidence = compute_proxy_reconstructability(
        features=features,
        fold_assignments=fold_frame,
        audit_frame=audit,
        predictor_sets={"department_free": ["safe_numeric", "safe_category"]},
        proxy_target="EmpDepartment",
        proxy_aliases=["DeptID"],
        identity=identity,
    )

    row = evidence.status.iloc[0]
    assert row["analysis_status"] == "not_estimated_insufficient_outer_training_class_support"
    assert row["models_fitted"] == 0
    assert row["n_resamples"] == 0
    assert row["classes_merged_or_dropped"] == False  # noqa: E712 - numpy/pandas bool
    assert row["headline_eligible"] == False  # noqa: E712
    assert json.loads(row["proxy_target_class_counts_json"]) == {
        "Administration": 9,
        "Executive Office": 1,
        "IT/IS": 50,
        "Production": 209,
        "Sales": 31,
        "Software Engineering": 11,
    }
    deficient = json.loads(row["outer_training_deficiencies_json"])
    assert any("Executive Office" in item["missing_training_classes"] for item in deficient)
    assert evidence.oof_predictions.empty
    assert evidence.metric_intervals.empty
    assert evidence.paired_differences.empty
    assert evidence.metadata["models_fitted"] == 0
    assert set(evidence.feature_contracts["proxy_target_and_aliases_absent"]) == {True}
    assert set(evidence.feature_contracts["limitations"]) == {PROXY_LIMITATION}


def test_proxy_target_or_alias_in_predictors_fails_before_any_fit() -> None:
    departments = ["A", "B", "C"] * 10
    features, folds, audit = _proxy_frames(departments, np.tile([1, 2, 3], 10))
    with pytest.raises(HRDatasetDiagnosticsError, match="target/aliases"):
        compute_proxy_reconstructability(
            features=features,
            fold_assignments=folds,
            audit_frame=audit,
            predictor_sets={"invalid": ["safe_numeric", "DeptID"]},
            proxy_target="EmpDepartment",
            proxy_aliases=["DeptID"],
            identity=_identity("proxy-alias-test"),
        )


def test_proxy_reconstructability_uses_exact_folds_and_paired_5000_draws_when_feasible() -> None:
    departments = ["A", "B", "C"] * 20
    folds_array = np.tile([1, 2, 3, 4, 5], 12)
    features, folds, audit = _proxy_frames(departments, folds_array)
    identity = _identity("proxy-feasible-test")
    evidence = compute_proxy_reconstructability(
        features=features,
        fold_assignments=folds,
        audit_frame=audit,
        predictor_sets={
            "numeric_only": ["safe_numeric"],
            "numeric_and_category": ["safe_numeric", "safe_category"],
        },
        proxy_target="EmpDepartment",
        proxy_aliases=["DeptID"],
        identity=identity,
        n_resamples=REQUIRED_BOOTSTRAP_RESAMPLES,
        batch_size=500,
    )

    status = evidence.status.iloc[0]
    assert status["analysis_status"] == "estimated_descriptive_proxy_risk"
    assert status["models_fitted"] == 10
    assert status["n_resamples"] == 5000
    assert len(evidence.oof_predictions) == 2 * len(features)
    assert not evidence.oof_predictions.duplicated(["system_id", "sample_index"]).any()
    assert set(evidence.metric_intervals["metric"]) == {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    }
    assert set(evidence.metric_intervals["n_resamples"]) == {5000}
    assert set(evidence.metric_intervals["n_valid"]) == {5000}
    assert evidence.metric_intervals["point_estimate"].between(0.0, 1.0).all()
    assert evidence.metric_intervals["ci_low"].between(0.0, 1.0).all()
    assert evidence.metric_intervals["ci_high"].between(0.0, 1.0).all()
    assert len(evidence.paired_differences) == 3
    assert evidence.paired_differences["difference"].between(-1.0, 1.0).all()
    assert set(evidence.metric_intervals["resample_hash"]) == {
        evidence.metadata["resample_hash"]
    }
    assert not evidence.metric_intervals["headline_eligible"].any()
    assert set(evidence.metric_intervals["limitations"]) == {PROXY_LIMITATION}
