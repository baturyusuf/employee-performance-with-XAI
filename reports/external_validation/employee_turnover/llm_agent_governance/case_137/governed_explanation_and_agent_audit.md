# External Governed Explanation and Agent Audit: employee_turnover_without_last_evaluation_137

## Prediction Evidence

- Predicted class: 1
- True class: 1
- Confidence: 0.9933976554848654
- Feature policy: `without_last_evaluation`

## Governed Explanation

This is a research-only external validation case. The model predicted class 1 with confidence 0.9934, and the provided SHAP attributions indicate the largest positive contributions came from EmpJobSatisfaction, ProjectCount, and AverageMonthlyHours. SHAP is attribution, not causality, and the prediction requires human review.

## Agent Supervisor Status

research_only

## Warnings

- This model is decision support only and is not for autonomous HR decisions.
- SHAP is attribution, not causality.
- Probability estimates should be interpreted as approximate confidence, not objective certainty.
- Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- Removing EmpDepartment does not prove fairness.
- Prediction requires human review.
- Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- External validation is required before deployment.
