# External Governed Explanation and Agent Audit: hrdataset_v14_department_free_209

## Prediction Evidence

- Predicted class: 2
- True class: 2
- Confidence: 0.4969988261648003
- Feature policy: `department_free`

## Governed Explanation

This is a research-only external validation case with an uncertain prediction. The model predicted class 2 with probability 0.4969988261648003, and the true class is also 2. SHAP shows attribution, not causality, with DaysLateLast30 as the largest positive attribution and EngagementSurvey as the largest negative attribution.

## Agent Supervisor Status

research_only_with_warnings

## Warnings

- This model is decision support only and is not for autonomous HR decisions.
- Probability estimates should be interpreted as approximate confidence, not objective certainty.
- SHAP is attribution, not causality.
- Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- Removing EmpDepartment does not prove fairness.
- EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- Some subgroup metrics are unavailable; bootstrap_ci and subgroup_metrics are empty in the provided evidence.
- Prediction requires human review.
- Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- External validation is required before deployment.
