# External Governed Explanation and Agent Audit: ibm_hr_analytics_department_free_174

## Prediction Evidence

- Predicted class: 3
- True class: 3
- Confidence: 0.5043424963951111
- Feature policy: `department_free`

## Governed Explanation

This is a research-only external validation case. The model predicted class 3 with low confidence (0.5043 vs 0.4957 for class 4), so human review is required. SHAP attributions indicate the largest positive contributions came from EmpHourlyRate, YearsWithCurrManager, DistanceFromHome, EmpJobRole, and YearsSinceLastPromotion, while the largest negative contributions came from MonthlyRate, EmpEducationLevel, EmpJobSatisfaction, EmpRelationshipSatisfaction, and ExperienceYearsAtThisCompany. These are attributions, not causal effects.

## Agent Supervisor Status

research_only_with_warnings

## Warnings

- This model is decision support only and is not for autonomous HR decisions.
- Probability estimates should be interpreted as approximate confidence, not objective certainty.
- SHAP is attribution, not causality.
- Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- Removing EmpDepartment does not prove fairness.
- Small subgroup findings may be unstable and should be treated as diagnostic audit evidence.
- Prediction requires human review.
- Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- External validation is required before deployment.
