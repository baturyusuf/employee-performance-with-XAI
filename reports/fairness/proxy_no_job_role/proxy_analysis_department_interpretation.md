# Department Proxy Analysis

Feature set tested: `no_salary_hike_no_attrition_no_department_no_job_role`
Number of remaining features: 19

## Proxy classifier result

A balanced logistic-regression proxy classifier predicted `EmpDepartment` from the remaining model features with macro-F1=0.2412 and balanced accuracy=0.2949.

This does not prove discriminatory behavior. It estimates how much direct department information can still be reconstructed after removing `EmpDepartment`.

## Top proxy-associated features

- EducationBackground: mutual_info=0.2119, cramers_v=0.3664
- EmpJobLevel: mutual_info=0.0679
- EmpWorkLifeBalance: mutual_info=0.0345
- DistanceFromHome: mutual_info=0.0341
- EmpHourlyRate: mutual_info=0.0262
- EmpRelationshipSatisfaction: mutual_info=0.0250
- TrainingTimesLastYear: mutual_info=0.0218
- ExperienceYearsInCurrentRole: mutual_info=0.0215
- YearsWithCurrManager: mutual_info=0.0200
- EmpJobInvolvement: mutual_info=0.0198

## Interpretation

If department is highly reconstructable from remaining features, department-free modeling reduces direct department use but does not eliminate organisational proxy risk. `EmpJobRole` is especially important to inspect because it may encode department structure.

## Required follow-up

- Compare fairness gaps for department-including and department-free candidate models.
- Add proxy warnings to local reason codes when top features are organisational proxies.
- Do not claim department removal proves fairness.
