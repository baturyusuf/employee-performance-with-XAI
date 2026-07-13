from __future__ import annotations

import numpy as np

from src.experiments.hrdataset_replication_core import EXPECTED_LABELS


def test_sigmoid_training_is_inner_oof_and_outer_test_isolated(
    hrdataset_replication_evidence,
) -> None:
    evidence = hrdataset_replication_evidence
    result = evidence["result"]
    outer = result.folds.outer_assignments
    training = result.calibration_training_oof
    receipts = result.calibration_fit_receipts

    assert len(receipts) == 10 * 5
    assert not receipts["outer_test_used_for_fit"].any()
    assert not receipts["outer_test_used_for_calibrator_fit"].any()
    assert set(receipts["preprocessing_fit_scope"]) == {
        "inner_development_partition_only"
    }
    assert set(receipts["threadpool_limit"]) == {1}
    assert set(receipts["warning_count"]) == {0}
    for outer_fold in range(1, 11):
        outer_test = set(
            outer.loc[outer["outer_fold"].astype(int) == outer_fold, "sample_index"].astype(int)
        )
        outer_train = set(
            outer.loc[outer["outer_fold"].astype(int) != outer_fold, "sample_index"].astype(int)
        )
        scoped = training[training["outer_fold"].astype(int) == outer_fold]
        assert len(scoped) == len(outer_train)
        assert not scoped["sample_index"].duplicated().any()
        assert set(scoped["sample_index"].astype(int)) == outer_train
        assert not set(scoped["sample_index"].astype(int)).intersection(outer_test)
        assert set(scoped["inner_fold"].astype(int)) == {1, 2, 3, 4, 5}


def test_sigmoid_is_applied_only_to_exact_untouched_outer_model_probabilities(
    hrdataset_replication_evidence,
) -> None:
    result = hrdataset_replication_evidence["result"]
    raw = result.raw_oof_predictions[
        result.raw_oof_predictions["policy"] == "department_free"
    ]
    calibrated = result.calibrated_oof_predictions
    probability_columns = [f"prob_class_{label}" for label in EXPECTED_LABELS]

    assert len(calibrated) == len(raw)
    assert not calibrated["sample_index"].duplicated().any()
    assert set(calibrated["probability_method"]) == {
        "predeclared_cross_fitted_sigmoid"
    }
    for outer_fold in range(1, 11):
        raw_fold = raw[raw["outer_fold"].astype(int) == outer_fold].sort_values(
            "sample_index"
        )
        calibrated_fold = calibrated[
            calibrated["outer_fold"].astype(int) == outer_fold
        ].sort_values("sample_index")
        assert raw_fold["sample_index"].tolist() == calibrated_fold["sample_index"].tolist()
        assert raw_fold["source_outer_model_sha256"].nunique() == 1
        source_hash = raw_fold["source_outer_model_sha256"].iloc[0]
        assert set(calibrated_fold["source_outer_model_sha256"]) == {source_hash}
        replay = result.calibrators[outer_fold].transform(
            raw_fold[probability_columns].to_numpy(dtype=float)
        )
        np.testing.assert_allclose(
            replay,
            calibrated_fold[probability_columns].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        )


def test_calibrator_receipts_bind_selected_candidate_training_and_source_model(
    hrdataset_replication_evidence,
) -> None:
    result = hrdataset_replication_evidence["result"]
    parameters = result.calibrator_parameters
    relationships = result.calibrator_model_relationships
    selected = result.selected_hyperparameters.set_index("outer_fold")
    models = result.outer_model_receipts[
        result.outer_model_receipts["policy"] == "department_free"
    ].set_index("outer_fold")

    assert len(parameters) == 10 * len(EXPECTED_LABELS)
    assert len(relationships) == 10
    assert relationships["calibrator_applied_to_exact_source_outer_probabilities"].all()
    assert relationships["source_outer_model_preserved"].all()
    assert not relationships["outer_test_used_for_model_selection"].any()
    assert not relationships["outer_test_used_for_model_fit"].any()
    assert not relationships["outer_test_used_for_calibrator_fit"].any()
    assert not relationships["calibration_method_selected_from_outer_test"].any()
    assert set(parameters["calibration_method"]) == {"sigmoid"}
    assert not parameters["outer_test_used_for_fit"].any()
    assert not parameters["method_selected_from_outer_test"].any()
    assert parameters["calibrator_parameter_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    for outer_fold, group in parameters.groupby("outer_fold"):
        assert set(group["class_label"].astype(int)) == set(EXPECTED_LABELS)
        assert group["calibrator_parameter_sha256"].nunique() == 1
        assert group["training_probability_sha256"].nunique() == 1
        assert group["training_labels_sha256"].nunique() == 1
        assert set(group["source_outer_model_sha256"]) == {
            models.loc[outer_fold, "model_sha256"]
        }
        assert set(group["selected_candidate_index"].astype(int)) == {
            int(selected.loc[outer_fold, "selected_candidate_index"])
        }
        relationship = relationships.set_index("outer_fold").loc[outer_fold]
        assert relationship["source_outer_model_sha256"] == models.loc[
            outer_fold, "model_sha256"
        ]
        assert relationship["source_outer_raw_probability_sha256"] == models.loc[
            outer_fold, "outer_test_probability_sha256"
        ]
        assert relationship["calibrator_parameter_sha256"] == group[
            "calibrator_parameter_sha256"
        ].iloc[0]
        assert relationship["calibration_training_probability_sha256"] == group[
            "training_probability_sha256"
        ].iloc[0]
        assert relationship["calibration_training_labels_sha256"] == group[
            "training_labels_sha256"
        ].iloc[0]

    calibration_differences = result.calibration_differences
    assert set(calibration_differences["comparison_id"]) == {"sigmoid_minus_raw"}
    assert not calibration_differences["primary_gate_comparison"].any()
    assert not calibration_differences["gate_eligible"].any()
