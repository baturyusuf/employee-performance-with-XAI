# Local Reason Code: minority_class_4_correct

Run ID: `manuscript_final_20260712T175251Z_c664ef152ff3`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`  
Sample index: `25`; OOF fold: `3`  
True class: `4`; predicted class: `4`; confidence: `0.5051`

## Top Supporting Attributions

- `EmpWorkLifeBalance` = `4`; grouped SHAP `0.51086`.
- `EmpEnvironmentSatisfaction` = `3`; grouped SHAP `0.35515`.
- `TrainingTimesLastYear` = `3`; grouped SHAP `0.16145`.
- `EmpJobRole` = `Manager`; grouped SHAP `0.15798`.
- `DistanceFromHome` = `3`; grouped SHAP `0.12747`.
- `EmpRelationshipSatisfaction` = `1`; grouped SHAP `0.10278`.
- `ExperienceYearsInCurrentRole` = `6`; grouped SHAP `0.10270`.
- `EmpJobSatisfaction` = `1`; grouped SHAP `0.07441`.
- `EducationBackground` = `Marketing`; grouped SHAP `0.02695`.
- `EmpEducationLevel` = `4`; grouped SHAP `0.02479`.

## Top Opposing Attributions

- `TotalWorkExperienceInYears` = `34`; grouped SHAP `-0.22258`.
- `EmpHourlyRate` = `31`; grouped SHAP `-0.15155`.
- `BusinessTravelFrequency` = `Travel_Rarely`; grouped SHAP `-0.09677`.
- `NumCompaniesWorked` = `1`; grouped SHAP `-0.05465`.
- `EmpJobInvolvement` = `2`; grouped SHAP `-0.02477`.
- `ExperienceYearsAtThisCompany` = `34`; grouped SHAP `-0.00815`.

## Warnings

- SHAP values are model attributions, not causal effects.
- Counterfactual model scenarios are not employee prescriptions.
- Removing sensitive or organisational fields does not prove fairness; proxy risk may remain.
- This is research-grade decision support, not an autonomous HR decision system.
