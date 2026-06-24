# External Governed Explanation and Agent Audit: employee_turnover_without_last_evaluation_14587

## Prediction Evidence

- Predicted class: 1
- True class: 1
- Confidence: 0.9335513005767696
- Feature policy: `without_last_evaluation`

## Governed Explanation

The model predicts class 1 with confidence 0.9336 for this external research case. The largest positive SHAP attribution is AverageMonthlyHours (3.1403), while the largest negative attributions are EmpJobSatisfaction (-0.5184) and ExperienceYearsAtThisCompany (-0.4227). SHAP is attribution, not causality, and the prediction requires human review.

## Agent Supervisor Status

research_only

## Warnings

- This model is decision support only and is not for autonomous HR decisions.
- SHAP is attribution, not causality.
- Probability estimates should be interpreted as approximate confidence, not objective certainty.
- Removing EmpDepartment does not prove fairness.
- EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- Prediction requires human review.
- Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- External validation is required before deployment.
