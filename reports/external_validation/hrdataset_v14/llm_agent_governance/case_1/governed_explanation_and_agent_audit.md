# External Governed Explanation and Agent Audit: hrdataset_v14_department_free_1

## Prediction Evidence

- Predicted class: 3
- True class: 3
- Confidence: 0.9914753287713834
- Feature policy: `department_free`

## Governed Explanation

The external research model predicted class 3 with high confidence (0.9915) and the true class is also 3. The main SHAP attributions were positive for ExperienceYearsAtThisCompany, SpecialProjectsCount, DaysLateLast30, EngagementSurvey, and EmpStatusID, while RecruitmentSource and Absences were negative. SHAP is attribution, not causality, and the prediction requires human review.

## Agent Supervisor Status

research_only_with_warnings

## Warnings

- This model is decision support only and is not for autonomous HR decisions.
- SHAP is attribution, not causality.
- Probability estimates should be interpreted as approximate confidence, not objective certainty.
- Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- Removing EmpDepartment does not prove fairness.
- EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- Small subgroup findings may be unstable and should be treated as diagnostic audit evidence.
- Prediction requires human review.
- Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- External validation is required before deployment.
