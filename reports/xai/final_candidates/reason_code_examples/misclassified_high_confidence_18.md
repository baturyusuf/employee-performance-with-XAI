# Reason Code Example: misclassified_high_confidence

Sample index: 18
True class: 4
Predicted class: 3
Predicted probability: 0.8736 (high confidence band)

## Top Supporting Features
- EmpEnvironmentSatisfaction = 4 | SHAP 0.2659 | manager_controllable | not_sensitive
- EmpJobRole = Developer | SHAP 0.2175 | organisation_controllable | possible_proxy
- YearsWithCurrManager = 3 | SHAP 0.1778 | organisation_controllable | possible_proxy
- EmpHourlyRate = 86 | SHAP 0.1301 | organisation_controllable | possible_proxy
- OverTime = Yes | SHAP 0.0980 | manager_controllable | possible_proxy

## Top Opposing Features
- YearsSinceLastPromotion = 2 | SHAP -0.1412 | organisation_controllable | possible_proxy
- EducationBackground = Medical | SHAP -0.1357 | immutable | possible_proxy
- EmpJobInvolvement = 2 | SHAP -0.0619 | employee_controllable | not_sensitive
- TotalWorkExperienceInYears = 5 | SHAP -0.0472 | immutable | possible_proxy
- EmpJobSatisfaction = 4 | SHAP -0.0444 | manager_controllable | not_sensitive

## Governance Warnings
- SHAP is attribution, not causality.
- This model is decision support, not autonomous employee evaluation.
- Job role may proxy department.
- Department removal does not prove fairness.
- Counterfactuals may require manager or organisation intervention.
- Probability estimates may be imperfectly calibrated.
- Small subgroup results may be unstable.
- External validation is required before deployment.
