# Local Reason Code: correct_class_2

Run ID: `manuscript_final_20260712T181754Z_c664ef152ff3`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`  
Sample index: `251`; OOF fold: `3`  
True class: `2`; predicted class: `2`; confidence: `0.9812`

## Top Supporting Attributions

- `EmpEnvironmentSatisfaction` = `2`; grouped SHAP `1.62103`.
- `YearsSinceLastPromotion` = `2`; grouped SHAP `0.71505`.
- `EmpJobRole` = `Research Director`; grouped SHAP `0.44547`.
- `EmpHourlyRate` = `88`; grouped SHAP `0.35659`.
- `NumCompaniesWorked` = `9`; grouped SHAP `0.26134`.
- `EmpJobLevel` = `3`; grouped SHAP `0.17874`.
- `EmpRelationshipSatisfaction` = `3`; grouped SHAP `0.15992`.
- `EducationBackground` = `Medical`; grouped SHAP `0.14374`.
- `ExperienceYearsInCurrentRole` = `7`; grouped SHAP `0.14009`.
- `YearsWithCurrManager` = `13`; grouped SHAP `0.11912`.

## Top Opposing Attributions

- `TotalWorkExperienceInYears` = `20`; grouped SHAP `-0.09618`.
- `ExperienceYearsAtThisCompany` = `18`; grouped SHAP `-0.07959`.
- `EmpEducationLevel` = `4`; grouped SHAP `-0.06836`.
- `DistanceFromHome` = `9`; grouped SHAP `-0.05121`.
- `EmpWorkLifeBalance` = `2`; grouped SHAP `-0.02347`.
- `BusinessTravelFrequency` = `Travel_Frequently`; grouped SHAP `-0.00620`.
- `OverTime` = `No`; grouped SHAP `-0.00026`.

## Warnings

- SHAP values are model attributions, not causal effects.
- Counterfactual model scenarios are not employee prescriptions.
- Removing sensitive or organisational fields does not prove fairness; proxy risk may remain.
- This is research-grade decision support, not an autonomous HR decision system.
