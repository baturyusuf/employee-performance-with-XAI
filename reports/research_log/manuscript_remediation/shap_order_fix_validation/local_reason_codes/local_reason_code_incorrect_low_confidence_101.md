# Local Reason Code: incorrect_low_confidence

Run ID: `diagnostic_shap_order_fix`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`  
Sample index: `101`; OOF fold: `6`  
True class: `3`; predicted class: `4`; confidence: `0.5001`

## Top Supporting Attributions

- `EmpEnvironmentSatisfaction` = `4`; grouped SHAP `0.36219`.
- `EmpJobSatisfaction` = `4`; grouped SHAP `0.28795`.
- `ExperienceYearsInCurrentRole` = `4`; grouped SHAP `0.22019`.
- `YearsSinceLastPromotion` = `0`; grouped SHAP `0.18127`.
- `ExperienceYearsAtThisCompany` = `5`; grouped SHAP `0.17447`.
- `DistanceFromHome` = `2`; grouped SHAP `0.07847`.
- `TotalWorkExperienceInYears` = `13`; grouped SHAP `0.04924`.
- `EmpWorkLifeBalance` = `3`; grouped SHAP `0.03714`.
- `YearsWithCurrManager` = `4`; grouped SHAP `0.03031`.
- `EmpEducationLevel` = `3`; grouped SHAP `0.02950`.

## Top Opposing Attributions

- `NumCompaniesWorked` = `8`; grouped SHAP `-0.13398`.
- `EmpJobInvolvement` = `2`; grouped SHAP `-0.08702`.
- `BusinessTravelFrequency` = `Travel_Rarely`; grouped SHAP `-0.07000`.
- `EmpRelationshipSatisfaction` = `3`; grouped SHAP `-0.03485`.
- `EmpJobRole` = `Human Resources`; grouped SHAP `-0.02262`.

## Warnings

- SHAP values are model attributions, not causal effects.
- Counterfactual model scenarios are not employee prescriptions.
- Removing sensitive or organisational fields does not prove fairness; proxy risk may remain.
- This is research-grade decision support, not an autonomous HR decision system.
