# SHAP Stability Interpretation for Final Candidates

SHAP values are model attributions, not causal effects. Stability measures whether global grouped feature-importance rankings are consistent across CV folds.

## Stability Summary

### `no_salary_hike_no_attrition`
- Mean top-10 Jaccard: 0.8212.
- Mean Spearman rank correlation: 0.8716.
- Frequently recurring top features: EmpEnvironmentSatisfaction, YearsSinceLastPromotion, ExperienceYearsInCurrentRole, EmpWorkLifeBalance, EmpDepartment.

### `no_salary_hike_no_attrition_no_department`
- Mean top-10 Jaccard: 0.7606.
- Mean Spearman rank correlation: 0.8717.
- Frequently recurring top features: EmpEnvironmentSatisfaction, YearsSinceLastPromotion, EmpJobRole, ExperienceYearsInCurrentRole, EmpWorkLifeBalance.

### `no_salary_hike_no_attrition_no_department_no_job_role`
- Mean top-10 Jaccard: 0.7350.
- Mean Spearman rank correlation: 0.8340.
- Frequently recurring top features: EmpEnvironmentSatisfaction, YearsSinceLastPromotion, ExperienceYearsInCurrentRole, EmpWorkLifeBalance, EmpHourlyRate.

## Required Answers

### Are the top explanatory features stable across folds?
Use top-k Jaccard and Spearman jointly. High top-k Jaccard supports stable global discussion; low Spearman means lower-ranked features should not be overinterpreted.

### Does removing EmpDepartment change the explanation structure?
Removing EmpDepartment should be interpreted as an explanation-structure shift. If EmpJobRole or other organisational variables become prominent, describe this as proxy reliance rather than fairness mitigation.

### Does removing EmpJobRole make explanations more or less stable?
The strict job-role-free model must be judged jointly with utility. Higher stability alone is not sufficient if performance, calibration, or actionability are worse.

### Which explanations are safe to discuss in the paper?
Discuss only recurring grouped features from stable top-k rankings. Avoid local causal language and prescriptive HR claims.

### Which explanations require proxy/fairness warnings?
EmpJobRole, BusinessTravelFrequency, EducationBackground, DistanceFromHome, tenure, job level, and organisational-history features require proxy or governance warnings when important.
