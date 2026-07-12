# Local Reason Code: correct_class_3

Run ID: `diagnostic_shap_order_fix`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`  
Sample index: `376`; OOF fold: `2`  
True class: `3`; predicted class: `3`; confidence: `0.9968`

## Top Supporting Attributions

- `DistanceFromHome` = `28`; grouped SHAP `0.86967`.
- `YearsSinceLastPromotion` = `0`; grouped SHAP `0.34715`.
- `EmpEnvironmentSatisfaction` = `4`; grouped SHAP `0.29779`.
- `TotalWorkExperienceInYears` = `24`; grouped SHAP `0.29694`.
- `EmpJobSatisfaction` = `3`; grouped SHAP `0.15921`.
- `EmpJobInvolvement` = `4`; grouped SHAP `0.15287`.
- `EmpJobRole` = `Manufacturing Director`; grouped SHAP `0.13452`.
- `OverTime` = `Yes`; grouped SHAP `0.12286`.
- `YearsWithCurrManager` = `2`; grouped SHAP `0.10284`.
- `EmpRelationshipSatisfaction` = `2`; grouped SHAP `0.09937`.

## Top Opposing Attributions

- `EducationBackground` = `Medical`; grouped SHAP `-0.06727`.
- `TrainingTimesLastYear` = `3`; grouped SHAP `-0.05334`.
- `BusinessTravelFrequency` = `Travel_Frequently`; grouped SHAP `-0.02494`.

## Warnings

- SHAP values are model attributions, not causal effects.
- Counterfactual model scenarios are not employee prescriptions.
- Removing sensitive or organisational fields does not prove fairness; proxy risk may remain.
- This is research-grade decision support, not an autonomous HR decision system.
