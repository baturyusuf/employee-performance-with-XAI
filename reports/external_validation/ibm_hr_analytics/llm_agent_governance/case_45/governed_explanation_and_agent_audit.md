# External Governed Explanation and Agent Audit: ibm_hr_analytics_department_free_45

## Prediction Evidence

- Predicted class: 3
- True class: 3
- Confidence: 0.9966751427553432
- Feature policy: `department_free`

## Governed Explanation

The model predicted class 3 with high confidence (0.9967) for this research-only external validation case. SHAP attributions indicate the largest positive contributions came from MonthlyIncome, MonthlyRate, EmpHourlyRate, ExperienceYearsAtThisCompany, and DistanceFromHome, while YearsSinceLastPromotion and ExperienceYearsInCurrentRole were among the largest negative contributions. This is attribution, not causality, and the prediction requires human review.

## Agent Supervisor Status

research_only

## Warnings

- Probability estimates should be interpreted as approximate confidence, not objective certainty.
- SHAP is attribution, not causality.
- This model is decision support only and is not for autonomous HR decisions.
- Removing EmpDepartment does not prove fairness.
- EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- Prediction requires human review.
- Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- External validation is required before deployment.
