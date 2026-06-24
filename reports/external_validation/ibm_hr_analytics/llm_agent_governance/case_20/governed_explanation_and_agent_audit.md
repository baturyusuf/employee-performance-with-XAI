# External Governed Explanation and Agent Audit: ibm_hr_analytics_department_free_20

## Prediction Evidence

- Predicted class: 3
- True class: 3
- Confidence: 0.9346573531035772
- Feature policy: `department_free`

## Governed Explanation

This is a research-only external validation case. The model predicted class 3 with confidence 0.9347, but this is a diagnostic estimate that requires human review. SHAP attributions indicate the largest positive contributions came from EmpHourlyRate, DailyRate, MonthlyIncome, DistanceFromHome, and YearsWithCurrManager, while the largest negative contributions came from EmpEnvironmentSatisfaction and StockOptionLevel. These are attributions, not causal effects.

## Agent Supervisor Status

research_only

## Warnings

- This model is decision support only and is not for autonomous HR decisions.
- SHAP is attribution, not causality.
- Probability estimates should be interpreted as approximate confidence, not objective certainty.
- Removing EmpDepartment does not prove fairness.
- EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- Some subgroup evidence is limited or unavailable; bootstrap_ci and subgroup_metrics are empty, and several Age groups have n_samples < 30.
- Prediction requires human review.
- Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- External validation is required before deployment.
