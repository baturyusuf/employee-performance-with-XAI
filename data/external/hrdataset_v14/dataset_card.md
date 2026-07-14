# HRDataset_v14 Dataset Card

## Role

Role: independent external performance-target replication because it includes a mappable `PerformanceScore` target. The dataset-specific model is trained independently; this is not transport and testing of a locked INX model.

## Source

Raw CSV path: `data/external/hrdataset_v14/raw.csv`

Retrieval URL: `https://raw.githubusercontent.com/pouyasattari/HR-Dataset-Analysis/main/HRDataset_v14.csv`

This file is treated as a public mirror of HRDataset_v14 / Human Resources Data Set. Reports must state the mirror provenance and avoid stronger source-authenticity claims than the repository evidence supports.

## Target

`PerformanceScore` is mapped to canonical `PerformanceRating`:

- `PIP` -> 2
- `Needs Improvement` -> 2
- `Fully Meets` -> 3
- `Exceeds` -> 4
- `Exceptional` -> 4 if present

This predeclared mapped-target design merges `PIP` and `Needs Improvement` as
label 2, retains `Fully Meets` as label 3, and maps `Exceeds`/`Exceptional` to
label 4. It permits an ordered 2/3/4 replication analysis, but it does not
establish semantic, measurement, or prevalence equivalence with the INX
`PerformanceRating` target. Raw and mapped class support must be reported.

## Modeling Role

Use for independent mapped-target replication of the leakage-aware HR XAI
protocol. The canonical external primary policy is `conservative_primary`.
`department_including_audit`, `job_role_free_audit`, `proxy_rich_audit`, and
`temporality_restricted_audit` are audit-only sensitivity variants and must not
replace or be pooled with the primary result.

No policy may use identifiers, target or target-adjacent fields,
termination/employment-status fields, sensitive fields, `DeptID`, `PositionID`,
`MarriedID`, `EmpStatusID`, `FromDiversityJobFairID`, or raw `DateOfHire`.
The primary policy additionally excludes department text, salary, state, ZIP,
and recruitment source while retaining `EmpJobRole`. The proxy-rich audit may
restore salary, state, and recruitment source, but never ZIP.

`ExperienceYearsAtThisCompany` is deterministically derived from hire date to
the recorded last-review date. The configured source and reference columns are
mandatory; missing or unparseable date counts must exactly match the hashed
schema contract (currently zero for both), and no current-date or dataset-wide
maximum fallback is permitted. The two negative source durations are set to
missing and imputed only within each training partition. Raw dates never enter
a model; this timing-unverified derived context is neither causal nor actionable.

## Claim Boundaries

The dataset supports independent external performance-target replication, not
locked-model transport, universal external validation, causal validation,
fairness proof, legal compliance, deployment readiness, or autonomous HR
decision support. SHAP remains attribution only. Counterfactual evidence is
outside the core replication package.

Source authenticity, retrieval date, citation, licence, and redistribution
rights remain `manual_review_required`; the recorded mirror URL is not an
approved automatic-download source.
