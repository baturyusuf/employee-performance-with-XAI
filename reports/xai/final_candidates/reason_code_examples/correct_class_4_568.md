# Reason Code Example: correct_class_4

Sample index: 568
True class: 4
Predicted class: 4
Predicted probability: 0.9054 (high confidence band)

## Top Supporting Features
- DistanceFromHome = 24 | SHAP 0.6562 | immutable | possible_proxy
- EmpEnvironmentSatisfaction = 4 | SHAP 0.5128 | manager_controllable | not_sensitive
- EmpJobSatisfaction = 4 | SHAP 0.3224 | manager_controllable | not_sensitive
- ExperienceYearsInCurrentRole = 4 | SHAP 0.2271 | organisation_controllable | possible_proxy
- BusinessTravelFrequency = Travel_Frequently | SHAP 0.1446 | organisation_controllable | possible_proxy

## Top Opposing Features
- EmpJobRole = Manufacturing Director | SHAP -0.3753 | organisation_controllable | possible_proxy
- EmpEducationLevel = 1 | SHAP -0.0434 | employee_controllable | possible_proxy
- EmpWorkLifeBalance = 2 | SHAP -0.0291 | manager_controllable | possible_proxy
- EmpHourlyRate = 30 | SHAP -0.0229 | organisation_controllable | possible_proxy
- YearsSinceLastPromotion = 1 | SHAP -0.0147 | organisation_controllable | possible_proxy

## Governance Warnings
- SHAP is attribution, not causality.
- This model is decision support, not autonomous employee evaluation.
- Job role may proxy department.
- Department removal does not prove fairness.
- Counterfactuals may require manager or organisation intervention.
- Probability estimates may be imperfectly calibrated.
- Small subgroup results may be unstable.
- External validation is required before deployment.
