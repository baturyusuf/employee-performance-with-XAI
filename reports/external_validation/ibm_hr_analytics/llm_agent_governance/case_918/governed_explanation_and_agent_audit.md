# External Governed Explanation and Agent Audit: ibm_hr_analytics_department_free_918

## Prediction Evidence

- Predicted class: 3
- True class: 4
- Confidence: 0.9962332659171286
- Feature policy: `department_free`

## Governed Explanation

This is a research-only external validation case. The model predicted class 3 with confidence 0.9962, while the true class is 4. SHAP highlights MonthlyIncome, YearsSinceLastPromotion, BusinessTravelFrequency, YearsWithCurrManager, and ExperienceYearsInCurrentRole as positive attributions, and ExperienceYearsAtThisCompany and EmpWorkLifeBalance as negative attributions. SHAP is attribution, not causality, and the prediction requires human review.

## Agent Supervisor Status

research_only

## Warnings

- Probability estimates should be interpreted as approximate confidence, not objective certainty.
- SHAP is attribution, not causality.
- Removing EmpDepartment does not prove fairness.
- EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- Small subgroup findings may be unstable and should be treated as diagnostic audit evidence.
- Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- This model is decision support only and is not for autonomous HR decisions.
- Prediction requires human review.
- Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- External validation is required before deployment.
