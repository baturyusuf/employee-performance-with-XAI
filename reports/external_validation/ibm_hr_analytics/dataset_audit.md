# Dataset Audit: IBM HR Analytics Employee Attrition and Performance

Dataset name: `ibm_hr_analytics`
Recommended role: schema-compatible robustness
Task type: `restricted_ordinal_performance`
Source URL: `https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/master/WA_Fn-UseC_-HR-Employee-Attrition.csv`
Source note: Public mirror of the IBM HR Analytics Employee Attrition and Performance Kaggle dataset.

## Shape

- Rows: 1470
- Raw columns: 35
- Canonical columns: 36

## Target

- Raw target column: `PerformanceRating`
- Canonical target column: `PerformanceRating`
- Final target mapping: `{"identity_numeric_mapping": true}`

## Target Distribution

- 3: 1244 (0.8463)
- 4: 226 (0.1537)

## Feature Mapping Highlights

- `Age` -> `Age` (sensitive_audit_only)
- `Attrition` -> `Attrition` (leakage_risk)
- `DailyRate` -> `DailyRate` (proxy_risk)
- `Department` -> `EmpDepartment` (proxy_risk)
- `EducationField` -> `EducationBackground` (proxy_risk)
- `EmployeeCount` -> `EmployeeCount` (leakage_risk)
- `EmployeeNumber` -> `EmpNumber` (id)
- `Gender` -> `Gender` (sensitive_audit_only)
- `HourlyRate` -> `EmpHourlyRate` (proxy_risk)
- `JobLevel` -> `EmpJobLevel` (proxy_risk)
- `JobRole` -> `EmpJobRole` (proxy_risk)
- `MaritalStatus` -> `MaritalStatus` (sensitive_audit_only)
- `MonthlyIncome` -> `MonthlyIncome` (proxy_risk)
- `MonthlyRate` -> `MonthlyRate` (proxy_risk)
- `Over18` -> `Over18` (leakage_risk)
- `PercentSalaryHike` -> `EmpLastSalaryHikePercent` (leakage_risk)
- `PerformanceRating` -> `PerformanceRating` (target;leakage_risk)
- `StandardHours` -> `StandardHours` (leakage_risk)
- `StockOptionLevel` -> `StockOptionLevel` (proxy_risk)
- `YearsAtCompany` -> `ExperienceYearsAtThisCompany` (proxy_risk)
- `YearsSinceLastPromotion` -> `YearsSinceLastPromotion` (proxy_risk)
- `YearsWithCurrManager` -> `YearsWithCurrManager` (proxy_risk)

## Leakage-Risk Columns

`Attrition`, `EmpLastSalaryHikePercent`, `EmployeeCount`, `Over18`, `PerformanceRating`, `StandardHours`

## Sensitive / Audit-Only Columns

`Age`, `Gender`, `MaritalStatus`

## Proxy-Risk Columns

`DailyRate`, `EducationBackground`, `EmpDepartment`, `EmpHourlyRate`, `EmpJobLevel`, `EmpJobRole`, `ExperienceYearsAtThisCompany`, `MonthlyIncome`, `MonthlyRate`, `StockOptionLevel`, `YearsSinceLastPromotion`, `YearsWithCurrManager`

## Available Audit Attributes

`Gender`, `MaritalStatus`, `EmpDepartment`, `EmpJobRole`, `Age`, `DailyRate`, `EducationBackground`, `EmpHourlyRate`, `EmpJobLevel`, `ExperienceYearsAtThisCompany`, `MonthlyIncome`, `MonthlyRate`, `StockOptionLevel`, `YearsSinceLastPromotion`, `YearsWithCurrManager`

## Claim Boundaries

- This audit is schema and target evidence, not proof of fairness or causal validity.
- SHAP explanations in downstream reports must be attribution-only.
- Counterfactuals in downstream reports must be framed as model scenarios, not employee prescriptions.
- This project remains research-grade decision support only and must not make autonomous HR decisions.
