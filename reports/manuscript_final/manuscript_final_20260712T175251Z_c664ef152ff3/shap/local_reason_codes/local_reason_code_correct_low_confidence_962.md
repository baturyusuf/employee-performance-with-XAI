# Local Reason Code: correct_low_confidence

Run ID: `manuscript_final_20260712T175251Z_c664ef152ff3`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`  
Sample index: `962`; OOF fold: `7`  
True class: `2`; predicted class: `2`; confidence: `0.4480`

## Top Supporting Attributions

- `EmpEnvironmentSatisfaction` = `1`; grouped SHAP `1.67528`.
- `YearsSinceLastPromotion` = `15`; grouped SHAP `0.29174`.
- `EmpJobRole` = `Manager`; grouped SHAP `0.27385`.
- `EmpJobLevel` = `5`; grouped SHAP `0.24635`.
- `ExperienceYearsInCurrentRole` = `7`; grouped SHAP `0.12081`.
- `YearsWithCurrManager` = `12`; grouped SHAP `0.07277`.
- `OverTime` = `Yes`; grouped SHAP `0.05490`.
- `NumCompaniesWorked` = `1`; grouped SHAP `0.03863`.

## Top Opposing Attributions

- `EmpWorkLifeBalance` = `4`; grouped SHAP `-0.52062`.
- `ExperienceYearsAtThisCompany` = `33`; grouped SHAP `-0.26748`.
- `EmpHourlyRate` = `79`; grouped SHAP `-0.24670`.
- `EducationBackground` = `Life Sciences`; grouped SHAP `-0.08550`.
- `DistanceFromHome` = `2`; grouped SHAP `-0.07118`.
- `EmpRelationshipSatisfaction` = `4`; grouped SHAP `-0.05771`.
- `EmpJobInvolvement` = `2`; grouped SHAP `-0.04273`.
- `TrainingTimesLastYear` = `2`; grouped SHAP `-0.04028`.
- `EmpEducationLevel` = `4`; grouped SHAP `-0.02977`.
- `BusinessTravelFrequency` = `Non-Travel`; grouped SHAP `-0.02220`.

## Warnings

- SHAP values are model attributions, not causal effects.
- Counterfactual model scenarios are not employee prescriptions.
- Removing sensitive or organisational fields does not prove fairness; proxy risk may remain.
- This is research-grade decision support, not an autonomous HR decision system.
