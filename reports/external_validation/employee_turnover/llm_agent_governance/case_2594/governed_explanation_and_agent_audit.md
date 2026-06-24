# External Governed Explanation and Agent Audit: employee_turnover_without_last_evaluation_2594

## Prediction Evidence

- Predicted class: 0
- True class: 0
- Confidence: 0.9999867445015652
- Feature policy: `without_last_evaluation`

## Governed Explanation

This is a research-only external validation case. The model predicted class 0 with very high confidence (0.9999867445015652), and the true class is also 0. SHAP shows the largest positive attributions for EmpJobSatisfaction, AverageMonthlyHours, ExperienceYearsAtThisCompany, ProjectCount, and WorkAccident; the largest negative attributions are SalaryBand and PromotionLast5Years. SHAP is attribution, not causality, and the prediction requires human review.

## Agent Supervisor Status

research_only_with_warnings

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
