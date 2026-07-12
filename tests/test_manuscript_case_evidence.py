from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from src.governance.manuscript_contract import canonical_config_hash
from src.llm.manuscript_case_evidence import (
    ManuscriptCaseEvidenceError,
    _top_local_shap,
    run,
)


CONFIG = Path("configs/manuscript_final.yaml")
RUN_ID = "case-evidence-unit"
PRIMARY = "no_salary_hike_no_attrition_no_department"


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _bound(**values: object) -> dict:
    return {"run_id": RUN_ID, "config_hash": canonical_config_hash(CONFIG), **values}


def _stage_fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "run"
    shap = root / "shap"
    calibration = root / "calibration"
    counterfactual = root / "counterfactual"
    fairness = root / "fairness"
    policy = root / "policy"
    external = root / "external" / "hrdataset_v14"
    output = root / "llm"

    _write_csv(
        calibration / "calibration_predictions.csv",
        [
            _bound(
                policy=PRIMARY,
                method="sigmoid",
                fold=1,
                sample_index=10,
                y_true=3,
                y_pred=3,
                prob_class_2=0.10,
                prob_class_3=0.75,
                prob_class_4=0.15,
            ),
            _bound(
                policy=PRIMARY,
                method="sigmoid",
                fold=2,
                sample_index=11,
                y_true=2,
                y_pred=3,
                prob_class_2=0.30,
                prob_class_3=0.55,
                prob_class_4=0.15,
            ),
        ],
    )
    _write_csv(
        calibration / "calibration_method_comparison.csv",
        [
            _bound(
                method="sigmoid",
                selected=True,
                selection_rank_sum=1.0,
                nll_log_loss_mean=0.51,
                multiclass_brier_mean=0.22,
                ece_confidence_mean=0.04,
            )
        ],
    )
    local_rows: list[dict] = []
    for sample_index, predicted, fold in [(10, 3, 1), (11, 3, 2)]:
        local_rows.extend(
            [
                _bound(
                    policy=PRIMARY,
                    sample_index=sample_index,
                    fold=fold,
                    class_label=predicted,
                    feature="EmpEnvironmentSatisfaction",
                    grouped_shap_value=0.22,
                    abs_grouped_shap_value=0.22,
                ),
                _bound(
                    policy=PRIMARY,
                    sample_index=sample_index,
                    fold=fold,
                    class_label=predicted,
                    feature="YearsSinceLastPromotion",
                    grouped_shap_value=-0.11,
                    abs_grouped_shap_value=0.11,
                ),
            ]
        )
    _write_csv(shap / "local_grouped_shap_values.csv", local_rows)
    _write_csv(
        shap / "shap_stability_summary.csv",
        [
            _bound(
                policy=PRIMARY,
                top_k=10,
                n_fold_pairs=45,
                jaccard_mean=0.71,
                spearman_mean=0.82,
            )
        ],
    )
    _write_csv(
        shap / "representative_cases.csv",
        [
            _bound(sample_index=10, case_type="correct_high_confidence"),
            _bound(sample_index=11, case_type="misclassification_low_confidence"),
        ],
    )
    _write_csv(
        counterfactual / "actionability_by_case.csv",
        [
            _bound(
                policy=PRIMARY,
                sample_index=10,
                intervention_mode="employee_only",
                eligible_for_upward_shift=True,
                valid=False,
                desired_class=4,
                changed_features="",
                probability_gain=None,
                cost=None,
                failure_reason="no_candidate_reached_desired_class",
            ),
            _bound(
                policy=PRIMARY,
                sample_index=11,
                intervention_mode="employee_only",
                eligible_for_upward_shift=True,
                valid=True,
                desired_class=4,
                changed_features="TrainingTimesLastYear",
                probability_gain=0.18,
                cost=0.7,
                failure_reason="",
            ),
        ],
    )
    _write_csv(
        counterfactual / "actionability_summary.csv",
        [
            _bound(
                policy=PRIMARY,
                intervention_mode="employee_only",
                n_total_oof_cases=2,
                n_eligible_oof_cases=2,
                n_valid_counterfactuals=1,
                validity_rate=0.5,
            )
        ],
    )
    _write_csv(
        policy / "policy_summary.csv",
        [
            _bound(policy="full_feature_upper_bound", macro_f1_mean=0.95),
            _bound(policy=PRIMARY, macro_f1_mean=0.78),
        ],
    )
    _write_csv(
        policy / "leakage_sensitivity_index.csv",
        [
            _bound(
                reference_policy="full_feature_upper_bound",
                policy=PRIMARY,
                metric="macro_f1",
                index_mean=0.18,
            )
        ],
    )
    _write_csv(
        fairness / "manuscript_fairness_proxy_table.csv",
        [
            _bound(
                policy=PRIMARY,
                attribute="Gender",
                metric="macro_f1",
                class_label=None,
                gap=0.08,
                ci_low=0.01,
                ci_high=0.16,
                minimum_subgroup_support=45,
                valid_bootstrap_samples=5000,
                interpretation_category="supported_descriptive_audit",
            )
        ],
    )

    _write_csv(
        external / "oof_predictions.csv",
        [
            _bound(
                dataset_key="hrdataset_v14",
                policy="department_free",
                fold=1,
                sample_index=20,
                y_true=3,
                y_pred=3,
                prob_class_2=0.12,
                prob_class_3=0.70,
                prob_class_4=0.18,
            ),
            _bound(
                dataset_key="hrdataset_v14",
                policy="department_free",
                fold=2,
                sample_index=21,
                y_true=4,
                y_pred=3,
                prob_class_2=0.10,
                prob_class_3=0.62,
                prob_class_4=0.28,
            ),
        ],
    )
    _write_csv(
        external / "policy_summary.csv",
        [
            _bound(
                dataset_key="hrdataset_v14",
                policy="department_free",
                macro_f1=0.69,
                nll_log_loss=0.62,
                multiclass_brier=0.29,
                ece_confidence=0.07,
            )
        ],
    )
    _write_csv(
        external / "feature_policy_audit.csv",
        [
            _bound(
                dataset_key="hrdataset_v14",
                policy="department_free",
                excluded_leakage_columns="PerformanceRating;PerfScoreID",
                excluded_sensitive_columns="Gender;MaritalStatus",
            )
        ],
    )
    _write_csv(
        external / "target_support.csv",
        [
            _bound(dataset_key="hrdataset_v14", label=2, n_rows=30),
            _bound(dataset_key="hrdataset_v14", label=3, n_rows=200),
            _bound(dataset_key="hrdataset_v14", label=4, n_rows=81),
        ],
    )
    return {
        "root": root,
        "shap": shap,
        "calibration": calibration,
        "counterfactual": counterfactual,
        "fairness": fairness,
        "policy": policy,
        "external": external.parent,
        "output": output,
    }


def _fake_external_shap(**kwargs) -> pd.DataFrame:
    predictions: pd.DataFrame = kwargs["predictions"]
    selections = kwargs["selections"]
    output_path: Path = kwargs["output_path"]
    rows = []
    lookup = predictions.set_index("sample_index")
    for selection in selections:
        prediction = lookup.loc[selection.sample_index]
        for feature, value in [("EngagementSurvey", 0.25), ("Absences", -0.13)]:
            rows.append(
                _bound(
                    dataset_name="hrdataset_v14",
                    policy="department_free",
                    sample_index=selection.sample_index,
                    fold=int(prediction["fold"]),
                    class_label=int(prediction["y_pred"]),
                    predicted_class=int(prediction["y_pred"]),
                    feature=feature,
                    grouped_shap_value=value,
                    abs_grouped_shap_value=abs(value),
                    evaluation_scope="out_of_fold_selected_case",
                )
            )
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def test_offline_builder_emits_complete_bound_preflight(tmp_path: Path) -> None:
    paths = _stage_fixture(tmp_path)
    with patch(
        "src.llm.manuscript_case_evidence._generate_external_oof_local_shap",
        side_effect=_fake_external_shap,
    ), patch(
        "src.llm.manuscript_case_evidence.load_external_dataset",
        return_value=SimpleNamespace(),
    ), patch(
        "src.llm.manuscript_case_evidence.audit_attribute_columns",
        return_value=["Gender", "RaceEthnicity", "EmpDepartment"],
    ):
        outputs = run(
            CONFIG,
            output_dir=paths["output"],
            run_id=RUN_ID,
            config_hash=canonical_config_hash(CONFIG),
            shap_dir=paths["shap"],
            calibration_dir=paths["calibration"],
            counterfactual_dir=paths["counterfactual"],
            fairness_dir=paths["fairness"],
            policy_dir=paths["policy"],
            external_dir=paths["external"],
            inx_sample_size=2,
            hr_sample_size=2,
        )

    report = json.loads(outputs["preflight_report"].read_text(encoding="utf-8"))
    assert report["cases_requested"] == 4
    assert report["cases_complete"] == 4
    assert report["cases_incomplete"] == 0
    assert report["n_cases_complete"] == 4
    assert report["real_api_execution_allowed"] is False
    assert report["api_call_attempted"] is False
    records = [json.loads(line) for line in outputs["complete_case_evidence"].read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert {record["run_id"] for record in records} == {RUN_ID}
    assert {record["config_hash"] for record in records} == {canonical_config_hash(CONFIG)}
    assert {record["dataset_name"] for record in records} == {"inx_primary", "hrdataset_v14"}
    assert all(record["evidence"]["shap"]["grouped_shap_values"] for record in records)
    assert (paths["output"] / "hrdataset_v14_oof_local_grouped_shap_values.csv").is_file()


def test_builder_has_no_real_api_execution_path(tmp_path: Path) -> None:
    with pytest.raises(ManuscriptCaseEvidenceError, match="cannot execute a real LLM/API batch"):
        run(
            CONFIG,
            output_dir=tmp_path,
            run_id=RUN_ID,
            real_api_execution=True,
        )


def test_external_local_shap_requires_oof_fold_identity() -> None:
    legacy = pd.DataFrame(
        [
            {
                "sample_index": 1,
                "predicted_class": 3,
                "feature": "EngagementSurvey",
                "grouped_shap_value": 0.2,
                "abs_grouped_shap_value": 0.2,
            }
        ]
    )
    with pytest.raises(ManuscriptCaseEvidenceError, match="Legacy full-fit SHAP"):
        _top_local_shap(
            legacy,
            sample_index=1,
            predicted_class=3,
            top_k=5,
            external=True,
        )
