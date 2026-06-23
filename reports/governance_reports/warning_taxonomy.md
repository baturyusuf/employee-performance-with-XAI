# Canonical Governance Warning Taxonomy

This taxonomy normalizes LLM and agent warning text into stable IDs so evaluations compare warning concepts rather than surface wording.

## Mandatory Warning IDs

- `deployment.not_autonomous`
- `deployment.human_review_required`
- `causality.shap_not_causal`
- `leakage.full_feature_upper_bound_only`
- `fairness.department_removal_not_fairness_proof`
- `proxy.jobrole_department_proxy`
- `actionability.counterfactual_not_prescription`
- `validation.external_validation_required`

## Warning Table

| Warning ID | Category | Severity | Mandatory | Canonical Message |
|---|---|---|---:|---|
| `deployment.not_autonomous` | deployment | high | True | This model is decision support only and is not for autonomous HR decisions. |
| `deployment.human_review_required` | deployment | high | True | Prediction requires human review. |
| `causality.shap_not_causal` | causality | high | True | SHAP is attribution, not causality. |
| `leakage.full_feature_upper_bound_only` | leakage | high | True | Full-feature models are leakage-warning upper-bound baselines only, not deployable models. |
| `leakage.salary_attrition_excluded` | leakage | high | False | EmpLastSalaryHikePercent and Attrition are excluded from final candidates because they are leakage-risk or outcome-proximal variables. |
| `fairness.department_removal_not_fairness_proof` | fairness | high | True | Removing EmpDepartment does not prove fairness. |
| `proxy.jobrole_department_proxy` | proxy | high | True | EmpJobRole may proxy EmpDepartment, so department removal does not eliminate proxy risk. |
| `calibration.probabilities_approximate` | calibration | medium | False | Probability estimates should be interpreted as approximate confidence, not objective certainty. |
| `actionability.counterfactual_not_prescription` | actionability | high | True | Counterfactuals may not be employee-actionable and must not be treated as employee prescriptions. |
| `fairness.small_subgroup_instability` | fairness | medium | False | Small subgroup findings may be unstable and should be treated as diagnostic audit evidence. |
| `validation.external_validation_required` | validation | high | True | External validation is required before deployment. |

## Interpretation
- This is not an ethics score.
- The taxonomy supports reproducible evaluation of warning coverage and consistency.
- Case-specific agents may emit only the relevant subset of warnings; mandatory warnings are enforced at the governed explanation layer.
