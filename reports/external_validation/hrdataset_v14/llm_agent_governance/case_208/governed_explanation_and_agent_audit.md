# External Governed Explanation and Agent Audit: hrdataset_v14_department_free_208

## Prediction Evidence

- Predicted class: 3
- True class: 4
- Confidence: 0.9979336908401608
- Feature policy: `department_free`

## Governed Explanation

This is a research-only external validation case. The model predicted class 3 with high confidence (0.9979), while the true class is 4. SHAP highlights feature attributions such as Salary, Absences, DateOfHire, EngagementSurvey, and Zip, but SHAP is attribution, not causality. Predictions require human review.

## Agent Supervisor Status

research_only; not deployment-ready for autonomous HR use

## Warnings

- This model is decision support only and is not for autonomous HR decisions.
- SHAP is attribution, not causality.
- Probability estimates should be interpreted as approximate confidence, not objective certainty.
- Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- Removing EmpDepartment does not prove fairness.
- EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- Some subgroup metrics are unavailable; bootstrap_ci and subgroup_metrics are empty in the provided evidence.
- Prediction requires human review.
- Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- External validation is required before deployment.
