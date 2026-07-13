from __future__ import annotations

import pandas as pd
import pytest

from src.experiments.hrdataset_replication_core import (
    HRDatasetReplicationError,
    HRDatasetReplicationProtocol,
    PRIMARY_PRACTICAL_TIE_TOLERANCE,
    PRIMARY_SELECTION_METRIC,
    PRODUCTION_INNER_SPLITS,
    PRODUCTION_OUTER_SPLITS,
    SELECTION_TIE_BREAK_METRIC,
    evaluate_hrdataset_replication,
)
from src.utils.config_loader import load_config


def test_dataset_specific_nested_selection_and_model_receipts(
    hrdataset_replication_evidence,
) -> None:
    evidence = hrdataset_replication_evidence
    result = evidence["result"]

    assert result.canonical_eligible is False
    assert result.protocol_metadata["test_only_reduction"] == {
        "candidate_indices": [0],
        "bootstrap_resamples": 20,
    }
    assert result.folds.contract["dataset_key"] == "hrdataset_v14"
    assert result.folds.contract["outer_splits"] == PRODUCTION_OUTER_SPLITS
    assert result.folds.contract["inner_splits"] == PRODUCTION_INNER_SPLITS
    assert result.folds.contract["outer_seed"] == 42
    assert result.folds.contract["inner_seed"] == 43
    assert len(result.folds.outer_assignments) == len(evidence["target"])
    assert len(result.folds.inner_assignments) == len(evidence["target"]) * 9

    candidate_receipts = result.candidate_fit_receipts
    assert len(candidate_receipts) == PRODUCTION_OUTER_SPLITS * PRODUCTION_INNER_SPLITS
    assert set(candidate_receipts["candidate_index"]) == {0}
    assert set(candidate_receipts["outer_test_used_for_selection"]) == {False}
    assert set(candidate_receipts["outer_test_used_for_fit"]) == {False}
    assert set(candidate_receipts["threadpool_limit"]) == {1}
    assert set(candidate_receipts["warning_count"]) == {0}
    assert candidate_receipts["candidate_model_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()

    search = result.candidate_search_results
    assert len(search) == PRODUCTION_OUTER_SPLITS
    assert set(search["selection_metric"]) == {PRIMARY_SELECTION_METRIC}
    assert set(search["selection_tie_break_metric"]) == {SELECTION_TIE_BREAK_METRIC}
    assert set(search["primary_practical_tie_tolerance"]) == {
        PRIMARY_PRACTICAL_TIE_TOLERANCE
    }
    assert search["selected_by_protocol"].all()
    assert not search["outer_test_used_for_selection"].any()

    selected = result.selected_hyperparameters
    assert len(selected) == PRODUCTION_OUTER_SPLITS
    assert set(selected["selected_candidate_index"]) == {0}
    assert not selected["outer_test_used_for_selection"].any()

    receipts = result.outer_model_receipts
    assert len(receipts) == PRODUCTION_OUTER_SPLITS * len(evidence["policy_frames"])
    assert receipts["selected_primary_parameters_reused"].all()
    assert not receipts["outer_test_used_for_fit"].any()
    assert receipts["model_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    for _, group in receipts.groupby("outer_fold"):
        assert group["selected_candidate_index"].nunique() == 1
        assert group["selected_candidate_parameters_json"].nunique() == 1
        assert group["fixed_parameters_json"].nunique() == 1
        assert set(group["policy"]) == set(evidence["policy_frames"])

    lineage = result.transformed_feature_lineage
    assert set(lineage["model_sha256"]) == set(receipts["model_sha256"])
    assert lineage.groupby(["policy", "outer_fold"])["transformed_feature_index"].nunique().min() > 0
    assert not lineage.duplicated(
        ["policy", "outer_fold", "transformed_feature_index"]
    ).any()


def test_production_protocol_drift_and_forbidden_features_fail_before_fit(
    hrdataset_replication_evidence,
) -> None:
    with pytest.raises(HRDatasetReplicationError, match="production protocol drifted"):
        HRDatasetReplicationProtocol(outer_splits=5)

    evidence = hrdataset_replication_evidence
    contaminated = {
        name: frame.copy() for name, frame in evidence["policy_frames"].items()
    }
    contaminated["department_free"]["DeptID"] = 1
    with pytest.raises(HRDatasetReplicationError, match="contains forbidden model features"):
        evaluate_hrdataset_replication(
            contaminated,
            evidence["policy_roles"],
            evidence["forbidden"],
            evidence["target"],
            evidence["identifiers"],
            load_config("configs/model_grid.yaml"),
            primary_policy="department_free",
            run_id="fail-before-fit",
            config_hash="a" * 64,
            scientific_input_hash="b" * 64,
            dataset_sha256="c" * 64,
            test_only_overrides=evidence["test_overrides"],
        )

    with pytest.raises(HRDatasetReplicationError, match="keys must be strings"):
        evaluate_hrdataset_replication(
            {1: evidence["policy_frames"]["department_free"]},
            {1: "primary"},
            {1: evidence["forbidden"]["department_free"]},
            evidence["target"],
            evidence["identifiers"],
            load_config("configs/model_grid.yaml"),
            primary_policy="1",
            run_id="fail-before-fit",
            config_hash="a" * 64,
            scientific_input_hash="b" * 64,
            dataset_sha256="c" * 64,
            test_only_overrides=evidence["test_overrides"],
        )


def test_raw_oof_is_exactly_once_and_fold_bound(
    hrdataset_replication_evidence,
) -> None:
    evidence = hrdataset_replication_evidence
    result = evidence["result"]
    raw = result.raw_oof_predictions
    outer = result.folds.outer_assignments[["sample_index", "outer_fold", "y_true"]]

    assert len(raw) == len(evidence["target"]) * len(evidence["policy_frames"])
    for policy, group in raw.groupby("policy"):
        assert len(group) == len(evidence["target"])
        assert not group["sample_index"].duplicated().any()
        observed = group[["sample_index", "outer_fold", "y_true"]].sort_values(
            "sample_index"
        ).reset_index(drop=True)
        expected = outer.sort_values("sample_index").reset_index(drop=True)
        pd.testing.assert_frame_equal(observed, expected)
        assert set(group["probability_method"]) == {"raw"}
        assert group.filter(regex=r"^prob_class_").sum(axis=1).sub(1.0).abs().max() < 1e-12
