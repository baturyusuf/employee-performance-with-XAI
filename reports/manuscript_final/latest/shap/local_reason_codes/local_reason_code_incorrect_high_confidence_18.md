# Local Reason Code: incorrect_high_confidence

Run ID: `manuscript_final_20260712T181754Z_c664ef152ff3`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`  
Sample index: `18`; OOF fold: `7`  
True class: `4`; predicted class: `3`; confidence: `0.9975`

## Top Supporting Attributions

- `NumCompaniesWorked` = `0`; grouped SHAP `0.37895`.
- `EmpEnvironmentSatisfaction` = `4`; grouped SHAP `0.29498`.
- `EmpHourlyRate` = `86`; grouped SHAP `0.25584`.
- `EmpJobRole` = `Developer`; grouped SHAP `0.24196`.
- `YearsWithCurrManager` = `3`; grouped SHAP `0.20337`.
- `OverTime` = `Yes`; grouped SHAP `0.19680`.
- `DistanceFromHome` = `2`; grouped SHAP `0.16320`.
- `ExperienceYearsInCurrentRole` = `2`; grouped SHAP `0.12097`.
- `EmpRelationshipSatisfaction` = `4`; grouped SHAP `0.10466`.
- `TrainingTimesLastYear` = `2`; grouped SHAP `0.06732`.

## Top Opposing Attributions

- `EmpJobInvolvement` = `2`; grouped SHAP `-0.04699`.
- `EducationBackground` = `Medical`; grouped SHAP `-0.04491`.
- `YearsSinceLastPromotion` = `2`; grouped SHAP `-0.04052`.

## Warnings

- SHAP values are model attributions, not causal effects.
- Counterfactual model scenarios are not employee prescriptions.
- Removing sensitive or organisational fields does not prove fairness; proxy risk may remain.
- This is research-grade decision support, not an autonomous HR decision system.
