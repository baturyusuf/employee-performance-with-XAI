# External Governed Explanation and Agent Audit: hrdataset_v14_department_free_49

## Prediction Evidence

- Predicted class: 3
- True class: 3
- Confidence: 0.9993913929476668
- Feature policy: `department_free`

## Governed Explanation

This is a research-only external validation case. The model predicted class 3 with high confidence (0.9993913929476668), and the true class is also 3. SHAP highlights feature attributions such as ExperienceYearsAtThisCompany, RecruitmentSource, Zip, Salary, and DaysLateLast30, but SHAP is attribution, not causality. Prediction requires human review.

## Agent Supervisor Status

research_only

## Warnings

- This model is decision support only and is not for autonomous HR decisions.
- SHAP is attribution, not causality.
- Probability estimates should be interpreted as approximate confidence, not objective certainty.
- Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions.
- Removing EmpDepartment does not prove fairness.
- EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk.
- Small subgroup findings may be unstable and should be treated as diagnostic audit evidence.
- Prediction requires human review.
- Full-feature models are leakage-warning upper-bound baselines only, not deployable models.
- External validation is required before deployment.
