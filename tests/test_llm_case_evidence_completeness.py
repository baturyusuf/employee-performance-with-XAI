from __future__ import annotations

from dataclasses import replace

from src.llm.evidence_preflight import (
    build_evidence_preflight_report,
    validate_complete_case_evidence,
    wilson_interval,
)
from src.llm.evidence_schema import (
    CalibrationEvidence,
    CompleteCaseEvidence,
    CounterfactualEvidence,
    FairnessEvidence,
    GovernanceEvidence,
    LeakageEvidence,
    PredictionEvidence,
    ShapEvidence,
)


PRIMARY_POLICY = "no_salary_hike_no_attrition_no_department"


def complete_evidence(case_id: str = "inx_primary_case_1") -> CompleteCaseEvidence:
    return CompleteCaseEvidence(
        prediction=PredictionEvidence(
            case_id=case_id,
            predicted_class=3,
            true_class=3,
            class_probabilities={"2": 0.10, "3": 0.70, "4": 0.20},
            confidence=0.70,
            uncertainty_flag=False,
            model_name="xgboost",
            feature_policy=PRIMARY_POLICY,
            leakage_safe_status="canonical_primary_oof",
            dataset_name="inx_primary",
        ),
        shap=ShapEvidence(
            top_positive_features=[{"feature": "EmpEnvironmentSatisfaction", "grouped_shap_value": 0.2}],
            top_negative_features=[{"feature": "YearsSinceLastPromotion", "grouped_shap_value": -0.1}],
            grouped_shap_values={"EmpEnvironmentSatisfaction": 0.2, "YearsSinceLastPromotion": -0.1},
            class_specific_shap_values={"EmpEnvironmentSatisfaction": 0.2, "YearsSinceLastPromotion": -0.1},
            shap_stability_summary={"mean_jaccard": 0.8},
            explanation_stability_warning="SHAP is attribution, not causality.",
        ),
        fairness=FairnessEvidence(
            audited_groups=["Gender", "MaritalStatus"],
            subgroup_metrics={},
            disparity_gaps={},
            bootstrap_ci={},
            low_support_warnings=[],
            proxy_risk_warnings=["Removing group variables does not prove fairness."],
        ),
        calibration=CalibrationEvidence(
            log_loss=0.5,
            brier_score=0.2,
            expected_calibration_error=0.04,
            calibration_warning="Probabilities are uncertain research outputs.",
        ),
        counterfactual=CounterfactualEvidence(
            counterfactual_mode="employee_only",
            validity=None,
            changed_features=[],
            probability_gain=None,
            proximity_cost=None,
            actionability_label="no_valid_scenario",
            failed_reason="No valid scenario was identified.",
            warning="Model scenario, not an employee prescription.",
        ),
        leakage=LeakageEvidence(
            feature_policy=PRIMARY_POLICY,
            excluded_leakage_features=[
                "Age",
                "Gender",
                "MaritalStatus",
                "EmpDepartment",
                "EmpLastSalaryHikePercent",
                "Attrition",
                "EmpNumber",
                "PerformanceRating",
            ],
            full_feature_score=None,
            leakage_safe_score=0.8,
            leakage_sensitivity_index=None,
            leakage_warning="Upper-bound models are diagnostic only.",
        ),
        governance=GovernanceEvidence(
            intended_use="Research-grade decision support only.",
            prohibited_use="No autonomous HR decisions.",
            model_card_summary="Canonical model-card evidence.",
            deployment_status="research_only",
            required_warnings=["Human review is required."],
            human_review_required=True,
        ),
        evidence_sources=["unit-test-fixture"],
    )


def test_complete_case_evidence_requires_local_shap_section() -> None:
    complete = validate_complete_case_evidence(complete_evidence())
    assert complete.complete is True
    assert complete.missing_fields == ()

    incomplete_evidence = replace(complete_evidence(), shap=None)
    incomplete = validate_complete_case_evidence(incomplete_evidence)
    assert incomplete.complete is False
    assert "shap" in incomplete.missing_fields
    assert any(field.startswith("shap.") for field in incomplete.missing_fields)


def test_canonical_primary_forbidden_feature_is_an_evidence_blocker() -> None:
    evidence = complete_evidence()
    evidence.shap = ShapEvidence(
        top_positive_features=[{"feature": "Gender", "grouped_shap_value": 0.3}],
        top_negative_features=[],
        grouped_shap_values={"Gender": 0.3},
        class_specific_shap_values={"Gender": 0.3},
        shap_stability_summary={"mean_jaccard": 0.8},
        explanation_stability_warning="SHAP is attribution, not causality.",
    )
    readiness = validate_complete_case_evidence(evidence)
    assert readiness.complete is False
    assert readiness.forbidden_features == {"Gender": ("Gender",)}


def test_preflight_reports_completeness_separately_from_text_compliance() -> None:
    records = [
        {"evidence": complete_evidence("complete")},
        {
            "evidence": replace(complete_evidence("incomplete"), shap=None),
            "notes": "Local reason-code/SHAP evidence unavailable for this sampled case.",
        },
    ]
    report = build_evidence_preflight_report(records, run_id="unit", run_mode="dry_run")
    assert report["cases_requested"] == 2
    assert report["cases_complete"] == 1
    assert report["cases_incomplete"] == 1
    assert report["complete_case_rate"] == 0.5
    assert report["missing_fields_by_category"]["shap"] >= 1
    assert report["real_api_execution_allowed"] is False
    assert report["cases"][1]["diagnostic_category"] == "local_evidence_generation_coverage_gap"


def test_wilson_interval_does_not_encode_all_pass_as_certain_population_success() -> None:
    low, high = wilson_interval(80, 80)
    assert low is not None and 0.0 < low < 1.0
    assert high == 1.0
