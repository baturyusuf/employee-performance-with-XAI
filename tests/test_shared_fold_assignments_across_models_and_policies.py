from __future__ import annotations

import pandas as pd
import pytest

from src.experiments.shared_folds import (
    SharedFoldContractError,
    generate_shared_folds,
    validate_consumer_fold_assignments,
)


def _artifacts():
    frame = pd.DataFrame(
        {
            "EmpNumber": [f"E{index:03d}" for index in range(60)],
            "feature": [index % 7 for index in range(60)],
            "PerformanceRating": [2 + index % 3 for index in range(60)],
        },
        index=range(60),
    )
    return generate_shared_folds(
        frame,
        target_column="PerformanceRating",
        id_column="EmpNumber",
        run_id="consumer-test",
        config_hash="a" * 64,
        scientific_input_hash="b" * 64,
        dataset_key="inx_primary",
        dataset_sha256="c" * 64,
        outer_splits=10,
        inner_splits=5,
        seed=42,
        inner_seed=43,
    )


def _consumer_rows(artifacts, groups: list[str], group_column: str) -> pd.DataFrame:
    rows = []
    for group in groups:
        for record in artifacts.outer_assignments.itertuples(index=False):
            rows.append(
                {
                    group_column: group,
                    "sample_index": record.sample_index,
                    "outer_fold": record.outer_fold,
                    "fold_contract_hash": record.fold_contract_hash,
                    "y_true": record.y_true,
                }
            )
    return pd.DataFrame(rows)


def test_all_models_and_policies_bind_the_identical_outer_assignment() -> None:
    artifacts = _artifacts()
    models = _consumer_rows(
        artifacts,
        ["multinomial_logistic_regression", "random_forest", "lightgbm", "xgboost"],
        "model",
    )
    policies = _consumer_rows(
        artifacts,
        [
            "full_feature_upper_bound",
            "no_salary_hike",
            "no_salary_hike_no_attrition",
            "no_salary_hike_no_attrition_no_department",
            "no_salary_hike_no_attrition_no_department_no_job_role",
        ],
        "policy",
    )

    validate_consumer_fold_assignments(artifacts, models, group_columns=["model"])
    validate_consumer_fold_assignments(artifacts, policies, group_columns=["policy"])
    model_map = models.groupby("sample_index")["outer_fold"].nunique()
    policy_map = policies.groupby("sample_index")["outer_fold"].nunique()
    assert model_map.eq(1).all()
    assert policy_map.eq(1).all()


@pytest.mark.parametrize("defect", ["fold_drift", "missing_sample", "duplicate_sample", "contract_drift"])
def test_consumer_validation_fails_closed_on_unpaired_oof_evidence(defect: str) -> None:
    artifacts = _artifacts()
    rows = _consumer_rows(artifacts, ["xgboost", "random_forest"], "model")
    xgb_rows = rows.index[rows["model"] == "xgboost"].tolist()
    if defect == "fold_drift":
        index = xgb_rows[0]
        observed = int(rows.loc[index, "outer_fold"])
        rows.loc[index, "outer_fold"] = 1 if observed != 1 else 2
    elif defect == "missing_sample":
        rows = rows.drop(index=xgb_rows[0])
    elif defect == "duplicate_sample":
        rows = pd.concat([rows, rows.loc[[xgb_rows[0]]]], ignore_index=True)
    else:
        rows.loc[xgb_rows[0], "fold_contract_hash"] = "0" * 64

    with pytest.raises(SharedFoldContractError):
        validate_consumer_fold_assignments(artifacts, rows, group_columns=["model"])


def test_target_identity_must_match_the_fold_contract() -> None:
    artifacts = _artifacts()
    rows = _consumer_rows(artifacts, ["xgboost"], "model")
    rows.loc[0, "y_true"] = 4 if int(rows.loc[0, "y_true"]) != 4 else 3

    with pytest.raises(SharedFoldContractError, match="target identity"):
        validate_consumer_fold_assignments(artifacts, rows, group_columns=["model"])
