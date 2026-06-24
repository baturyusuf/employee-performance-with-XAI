# Dataset Audit: HRDataset_v14

Dataset name: `hrdataset_v14`
Recommended role: independent replication and direct external performance validation
Task type: `ordinal_multiclass_performance`
Source URL: `https://raw.githubusercontent.com/pouyasattari/HR-Dataset-Analysis/main/HRDataset_v14.csv`
Source note: Public mirror of the HRDataset_v14 / Human Resources Data Set commonly distributed through Kaggle.

## Shape

- Rows: 311
- Raw columns: 36
- Canonical columns: 39

## Target

- Raw target column: `PerformanceScore`
- Canonical target column: `PerformanceRating`
- Final target mapping: `{"Exceeds": 4, "Exceptional": 4, "Fully Meets": 3, "Needs Improvement": 2, "PIP": 2}`

## Target Distribution

- 2: 31 (0.0997)
- 3: 243 (0.7814)
- 4: 37 (0.1190)

## Feature Mapping Highlights

- `Absences` -> `Absences` (proxy_risk)
- `CitizenDesc` -> `CitizenDesc` (sensitive_audit_only)
- `DOB` -> `DOB` (sensitive_audit_only)
- `DateofTermination` -> `DateofTermination` (leakage_risk)
- `DaysLateLast30` -> `DaysLateLast30` (proxy_risk)
- `Department` -> `EmpDepartment` (proxy_risk)
- `EmpID` -> `EmpNumber` (id)
- `Employee_Name` -> `Employee_Name` (id)
- `EmploymentStatus` -> `EmploymentStatus` (leakage_risk)
- `GenderID` -> `GenderID` (sensitive_audit_only)
- `HispanicLatino` -> `HispanicLatino` (sensitive_audit_only)
- `LastPerformanceReview_Date` -> `LastPerformanceReview_Date` (leakage_risk)
- `ManagerID` -> `ManagerID` (leakage_risk)
- `ManagerName` -> `ManagerName` (leakage_risk)
- `MaritalDesc` -> `MaritalStatus` (sensitive_audit_only)
- `MaritalStatusID` -> `MaritalStatusID` (sensitive_audit_only)
- `PerfScoreID` -> `PerfScoreID` (leakage_risk)
- `PerformanceScore` -> `PerformanceScore` (target;leakage_risk)
- `Position` -> `EmpJobRole` (proxy_risk)
- `RaceDesc` -> `RaceEthnicity` (sensitive_audit_only)
- `RecruitmentSource` -> `RecruitmentSource` (proxy_risk)
- `Salary` -> `Salary` (proxy_risk)
- `Sex` -> `Gender` (sensitive_audit_only)
- `SpecialProjectsCount` -> `SpecialProjectsCount` (proxy_risk)
- `State` -> `State` (proxy_risk)
- `TermReason` -> `TermReason` (leakage_risk)
- `Termd` -> `Termd` (leakage_risk)
- `Zip` -> `Zip` (proxy_risk)

## Leakage-Risk Columns

`DateofTermination`, `EmploymentStatus`, `LastPerformanceReview_Date`, `ManagerID`, `ManagerName`, `PerfScoreID`, `PerformanceRating`, `PerformanceScore`, `TermReason`, `Termd`

## Sensitive / Audit-Only Columns

`CitizenDesc`, `DOB`, `Gender`, `GenderID`, `HispanicLatino`, `MaritalStatus`, `MaritalStatusID`, `RaceEthnicity`

## Proxy-Risk Columns

`Absences`, `DaysLateLast30`, `EmpDepartment`, `EmpJobRole`, `ExperienceYearsAtThisCompany`, `RecruitmentSource`, `Salary`, `SpecialProjectsCount`, `State`, `Zip`

## Available Audit Attributes

`Gender`, `RaceEthnicity`, `HispanicLatino`, `MaritalStatus`, `EmpDepartment`, `EmpJobRole`, `CitizenDesc`, `DOB`, `GenderID`, `MaritalStatusID`, `Absences`, `DaysLateLast30`, `ExperienceYearsAtThisCompany`, `RecruitmentSource`, `Salary`, `SpecialProjectsCount`, `State`, `Zip`

## Claim Boundaries

- This audit is schema and target evidence, not proof of fairness or causal validity.
- SHAP explanations in downstream reports must be attribution-only.
- Counterfactuals in downstream reports must be framed as model scenarios, not employee prescriptions.
- This project remains research-grade decision support only and must not make autonomous HR decisions.
