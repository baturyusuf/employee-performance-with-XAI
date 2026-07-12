# Local Reason Code: most_uncertain

Run ID: `manuscript_final_20260712T175251Z_c664ef152ff3`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`  
Sample index: `1164`; OOF fold: `4`  
True class: `3`; predicted class: `3`; confidence: `0.5001`

## Top Supporting Attributions

- `ExperienceYearsInCurrentRole` = `1`; grouped SHAP `0.54774`.
- `YearsWithCurrManager` = `2`; grouped SHAP `0.12577`.
- `EmpJobLevel` = `2`; grouped SHAP `0.03890`.
- `EmpJobInvolvement` = `3`; grouped SHAP `0.02330`.
- `BusinessTravelFrequency` = `Travel_Rarely`; grouped SHAP `0.02127`.
- `TrainingTimesLastYear` = `2`; grouped SHAP `0.01354`.
- `DistanceFromHome` = `13`; grouped SHAP `0.01042`.

## Top Opposing Attributions

- `EmpEnvironmentSatisfaction` = `2`; grouped SHAP `-0.46090`.
- `YearsSinceLastPromotion` = `1`; grouped SHAP `-0.26426`.
- `EmpJobRole` = `Sales Executive`; grouped SHAP `-0.24259`.
- `ExperienceYearsAtThisCompany` = `3`; grouped SHAP `-0.15503`.
- `EmpHourlyRate` = `46`; grouped SHAP `-0.12491`.
- `EducationBackground` = `Medical`; grouped SHAP `-0.06025`.
- `EmpRelationshipSatisfaction` = `3`; grouped SHAP `-0.04606`.
- `OverTime` = `No`; grouped SHAP `-0.03376`.
- `EmpJobSatisfaction` = `1`; grouped SHAP `-0.02424`.
- `NumCompaniesWorked` = `2`; grouped SHAP `-0.01613`.

## Warnings

- SHAP values are model attributions, not causal effects.
- Counterfactual model scenarios are not employee prescriptions.
- Removing sensitive or organisational fields does not prove fairness; proxy risk may remain.
- This is research-grade decision support, not an autonomous HR decision system.
