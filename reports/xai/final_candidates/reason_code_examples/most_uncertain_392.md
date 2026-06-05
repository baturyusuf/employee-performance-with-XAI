# Reason Code Example: most_uncertain

Sample index: 392
True class: 3
Predicted class: 3
Predicted probability: 0.4270 (low confidence band)

## Top Supporting Features
- EmpHourlyRate = 93 | SHAP 0.1802 | organisation_controllable | possible_proxy
- YearsWithCurrManager = 0 | SHAP 0.1623 | organisation_controllable | possible_proxy
- EmpJobSatisfaction = 2 | SHAP 0.1036 | manager_controllable | not_sensitive
- DistanceFromHome = 8 | SHAP 0.0893 | immutable | possible_proxy
- BusinessTravelFrequency = Travel_Rarely | SHAP 0.0615 | organisation_controllable | possible_proxy

## Top Opposing Features
- EmpEnvironmentSatisfaction = 1 | SHAP -0.6403 | manager_controllable | not_sensitive
- YearsSinceLastPromotion = 1 | SHAP -0.3315 | organisation_controllable | possible_proxy
- EmpJobRole = Research Director | SHAP -0.1277 | organisation_controllable | possible_proxy
- EmpWorkLifeBalance = 4 | SHAP -0.0725 | manager_controllable | possible_proxy
- ExperienceYearsAtThisCompany = 3 | SHAP -0.0538 | immutable | possible_proxy

## Governance Warnings
- SHAP is attribution, not causality.
- This model is decision support, not autonomous employee evaluation.
- Job role may proxy department.
- Department removal does not prove fairness.
- Counterfactuals may require manager or organisation intervention.
- Probability estimates may be imperfectly calibrated.
- Small subgroup results may be unstable.
- External validation is required before deployment.
