from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


@dataclass(frozen=True)
class Settings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    raw_data_path: Path = field(init=False)
    interim_data_path: Path = field(init=False)
    processed_data_path: Path = field(init=False)
    reports_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)

    target_col: str = "PerformanceRating"
    id_col: str = "EmpNumber"

    allowed_target_labels: Set[int] = field(default_factory=lambda: {2, 3, 4})

    expected_columns: List[str] = field(default_factory=lambda: [
        "EmpNumber",
        "Age",
        "Gender",
        "EducationBackground",
        "MaritalStatus",
        "EmpDepartment",
        "EmpJobRole",
        "BusinessTravelFrequency",
        "DistanceFromHome",
        "EmpEducationLevel",
        "EmpEnvironmentSatisfaction",
        "EmpHourlyRate",
        "EmpJobInvolvement",
        "EmpJobLevel",
        "EmpJobSatisfaction",
        "NumCompaniesWorked",
        "OverTime",
        "EmpLastSalaryHikePercent",
        "EmpRelationshipSatisfaction",
        "TotalWorkExperienceInYears",
        "TrainingTimesLastYear",
        "EmpWorkLifeBalance",
        "ExperienceYearsAtThisCompany",
        "ExperienceYearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
        "Attrition",
        "PerformanceRating",
    ])

    column_aliases: Dict[str, str] = field(default_factory=lambda: {
        "EmployeeNumber": "EmpNumber",
        "BusinessTravel": "BusinessTravelFrequency",
        "Department": "EmpDepartment",
        "JobRole": "EmpJobRole",
        "Education": "EmpEducationLevel",
        "EnvironmentSatisfaction": "EmpEnvironmentSatisfaction",
        "HourlyRate": "EmpHourlyRate",
        "JobInvolvement": "EmpJobInvolvement",
        "JobLevel": "EmpJobLevel",
        "JobSatisfaction": "EmpJobSatisfaction",
        "PercentSalaryHike": "EmpLastSalaryHikePercent",
        "RelationshipSatisfaction": "EmpRelationshipSatisfaction",
        "WorkLifeBalance": "EmpWorkLifeBalance",
        "YearsAtCompany": "ExperienceYearsAtThisCompany",
        "YearsInCurrentRole": "ExperienceYearsInCurrentRole",
    })

    ordinal_columns: List[str] = field(default_factory=lambda: [
        "EmpEducationLevel",
        "EmpEnvironmentSatisfaction",
        "EmpJobInvolvement",
        "EmpJobLevel",
        "EmpJobSatisfaction",
        "EmpRelationshipSatisfaction",
        "EmpWorkLifeBalance",
    ])

    categorical_columns: List[str] = field(default_factory=lambda: [
        "Gender",
        "EducationBackground",
        "MaritalStatus",
        "EmpDepartment",
        "EmpJobRole",
        "BusinessTravelFrequency",
        "OverTime",
        "Attrition",
    ])

    numeric_columns: List[str] = field(default_factory=lambda: [
        "Age",
        "DistanceFromHome",
        "EmpHourlyRate",
        "NumCompaniesWorked",
        "EmpLastSalaryHikePercent",
        "TotalWorkExperienceInYears",
        "TrainingTimesLastYear",
        "ExperienceYearsAtThisCompany",
        "ExperienceYearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
    ])

    fairness_sensitive_columns: List[str] = field(default_factory=lambda: [
        "Gender",
        "MaritalStatus",
    ])

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_data_path", self.project_root / "data" / "raw" / "inx_employee_performance.csv")
        object.__setattr__(self, "interim_data_path", self.project_root / "data" / "interim" / "inx_employee_performance_validated.csv")
        object.__setattr__(self, "processed_data_path", self.project_root / "data" / "processed" / "inx_employee_performance_processed.csv")
        object.__setattr__(self, "reports_dir", self.project_root / "reports")
        object.__setattr__(self, "artifacts_dir", self.project_root / "models_artifacts")


SETTINGS = Settings()