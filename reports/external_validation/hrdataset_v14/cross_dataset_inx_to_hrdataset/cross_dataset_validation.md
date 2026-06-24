# INX-to-HRDataset Cross-Dataset Validation

Status: reported as infeasible/too limited.

Common department-free safe features found: 3
Common features: EmpJobRole, EmpJobSatisfaction, ExperienceYearsAtThisCompany

The overlap is too weak for a scientifically defensible train-on-INX/test-on-HRDataset performance claim. Forcing this experiment would primarily measure schema mismatch rather than model transportability.

Decision: do not claim cross-dataset external validation from this feature overlap. Use HRDataset_v14 as independent external replication instead.
