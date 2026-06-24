# External Governed Explanation and Agent Audit: employee_turnover_without_last_evaluation_1158

## Prediction Evidence

- Predicted class: 0
- True class: 1
- Confidence: 0.9971588726971716
- Feature policy: `without_last_evaluation`

## Governed Explanation

This is a research-only external validation case. The model predicted class 0 with probability 0.9972, while the true class is 1. SHAP shows the largest positive attributions for EmpJobSatisfaction, ExperienceYearsAtThisCompany, and WorkAccident, and the largest negative attributions for EmpDepartment and AverageMonthlyHours. SHAP is attribution, not causality, and the prediction requires human review.

## Agent Supervisor Status

research_only

## Warnings

- This model is decision support only and is not for autonomous HR decisions.
- SHAP is attribution, not causality.
- Probability estimates should be interpreted as approximate confidence, not objective certainty.
- Removing EmpDepartment does not prove fairness.
- EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- Small subgroup findings may be unstable and should be treated as diagnostic audit evidence.
- Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- Prediction requires human review.
- Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- External validation is required before deployment.
