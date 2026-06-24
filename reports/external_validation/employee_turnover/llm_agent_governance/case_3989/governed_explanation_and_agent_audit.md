# External Governed Explanation and Agent Audit: employee_turnover_without_last_evaluation_3989

## Prediction Evidence

- Predicted class: 0
- True class: 0
- Confidence: 0.5003685802111453
- Feature policy: `without_last_evaluation`

## Governed Explanation

This is a research-only external validation case with an uncertain prediction: class 0 was predicted with probability 0.5004 versus 0.4996 for class 1. SHAP attributions indicate the largest positive contributions were ProjectCount and EmpJobSatisfaction, while ExperienceYearsAtThisCompany and SalaryBand were the largest negative contributions. SHAP is attribution, not causality, and the prediction requires human review.

## Agent Supervisor Status

research_only_external_validation

## Warnings

- Probability estimates should be interpreted as approximate confidence, not objective certainty.
- SHAP is attribution, not causality.
- This model is decision support only and is not for autonomous HR decisions.
- Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- Removing EmpDepartment does not prove fairness.
- Leakage-safe status is external_research_candidate with excluded leakage-risk fields under the selected policy.
- Prediction requires human review.
- Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- External validation is required before deployment.
