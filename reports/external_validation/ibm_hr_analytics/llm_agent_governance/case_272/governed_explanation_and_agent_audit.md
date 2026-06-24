# External Governed Explanation and Agent Audit: ibm_hr_analytics_department_free_272

## Prediction Evidence

- Predicted class: 3
- True class: 4
- Confidence: 0.8071770788308543
- Feature policy: `department_free`

## Governed Explanation

This is a research-only external validation case. The model predicted class 3 with confidence 0.8072, while the true class is 4. SHAP highlights are attribution only, not causality, and the prediction requires human review.

## Agent Supervisor Status

research_only

## Warnings

- This model is decision support only and is not for autonomous HR decisions.
- SHAP is attribution, not causality.
- Probability estimates should be interpreted as approximate confidence, not objective certainty.
- Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- Removing EmpDepartment does not prove fairness.
- EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- Prediction requires human review.
- Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- External validation is required before deployment.
