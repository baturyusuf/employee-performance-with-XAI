# Reason Code Example: correct_class_3

Sample index: 376
True class: 3
Predicted class: 3
Predicted probability: 0.9949 (high confidence band)

## Top Supporting Features
- DistanceFromHome = 28 | SHAP 0.6831 | immutable | possible_proxy
- TotalWorkExperienceInYears = 24 | SHAP 0.4654 | immutable | possible_proxy
- EmpEnvironmentSatisfaction = 4 | SHAP 0.3063 | manager_controllable | not_sensitive
- YearsSinceLastPromotion = 0 | SHAP 0.2346 | organisation_controllable | possible_proxy
- OverTime = Yes | SHAP 0.1791 | manager_controllable | possible_proxy

## Top Opposing Features
- TrainingTimesLastYear = 3 | SHAP -0.1147 | employee_controllable | not_sensitive
- BusinessTravelFrequency = Travel_Frequently | SHAP -0.0862 | organisation_controllable | possible_proxy
- EmpJobLevel = 4 | SHAP -0.0761 | organisation_controllable | possible_proxy
- EducationBackground = Medical | SHAP -0.0258 | immutable | possible_proxy
- EmpWorkLifeBalance = 2 | SHAP -0.0014 | manager_controllable | possible_proxy

## Governance Warnings
- SHAP is attribution, not causality.
- This model is decision support, not autonomous employee evaluation.
- Job role may proxy department.
- Department removal does not prove fairness.
- Counterfactuals may require manager or organisation intervention.
- Probability estimates may be imperfectly calibrated.
- Small subgroup results may be unstable.
- External validation is required before deployment.
