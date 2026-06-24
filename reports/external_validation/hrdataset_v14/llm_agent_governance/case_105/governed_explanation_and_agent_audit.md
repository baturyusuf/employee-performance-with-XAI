# External Governed Explanation and Agent Audit: hrdataset_v14_department_free_105

## Prediction Evidence

- Predicted class: 2
- True class: 2
- Confidence: 0.94413728577052
- Feature policy: `department_free`

## Governed Explanation

For this research-only external validation case, the model predicted class 2 with confidence 0.9441 and the true class is also 2. The main SHAP attribution is DaysLateLast30 with a positive value of 5.2809, while EngagementSurvey is the largest negative attribution at -0.3420. SHAP explains model attribution, not causality, and the prediction requires human review.

## Agent Supervisor Status

research_only_with_warnings

## Warnings

- SHAP is attribution, not causality.
- Probability estimates should be interpreted as approximate confidence, not objective certainty.
- Removing EmpDepartment does not prove fairness.
- EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- This model is decision support only and is not for autonomous HR decisions.
- Prediction requires human review.
- Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- External validation is required before deployment.
