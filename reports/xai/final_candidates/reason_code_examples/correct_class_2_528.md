# Reason Code Example: correct_class_2

Sample index: 528
True class: 2
Predicted class: 2
Predicted probability: 0.9836 (high confidence band)

## Top Supporting Features
- EmpEnvironmentSatisfaction = 2 | SHAP 1.6565 | manager_controllable | not_sensitive
- EmpJobRole = Finance Manager | SHAP 0.7740 | organisation_controllable | possible_proxy
- YearsSinceLastPromotion = 4 | SHAP 0.7484 | organisation_controllable | possible_proxy
- EmpHourlyRate = 91 | SHAP 0.2580 | organisation_controllable | possible_proxy
- EmpJobLevel = 4 | SHAP 0.2341 | organisation_controllable | possible_proxy

## Top Opposing Features
- EmpWorkLifeBalance = 3 | SHAP -0.1076 | manager_controllable | possible_proxy
- NumCompaniesWorked = 3 | SHAP -0.0450 | immutable | possible_proxy
- EmpEducationLevel = 5 | SHAP -0.0285 | employee_controllable | possible_proxy
- TrainingTimesLastYear = 6 | SHAP -0.0013 | employee_controllable | not_sensitive
- BusinessTravelFrequency = Travel_Rarely | SHAP -0.0006 | organisation_controllable | possible_proxy

## Governance Warnings
- SHAP is attribution, not causality.
- This model is decision support, not autonomous employee evaluation.
- Job role may proxy department.
- Department removal does not prove fairness.
- Counterfactuals may require manager or organisation intervention.
- Probability estimates may be imperfectly calibrated.
- Small subgroup results may be unstable.
- External validation is required before deployment.
