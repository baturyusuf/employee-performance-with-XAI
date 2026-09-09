# Submission-Ready Package Index

This directory is the controlled submission-finalization workspace for the MDPI *Information* manuscript. It is separate from internal development/reviewer-simulation artifacts.

## Manuscript

- `main.md` — finalized submission-facing scientific manuscript.
- Authoritative bibliography — `manuscript/mdpi_information/references.bib` (validated source blob recorded in `supplement/VALIDATED_SOURCE_IDENTITIES.md`).

## Journal-facing supplement

New finalization materials stored here:

- `supplement/TABLE_S4_PER_CLASS_METRICS.md`
- `supplement/TABLE_S5_PROTOCOL_AUDIT_GATES.md`
- `supplement/TABLE_S6_SUBGROUP_UNCERTAINTY.md`

Existing validated scientific source artifacts incorporated by reference without retyping or scientific modification:

- S1 — `reports/research_log/major_revision_round2/METHOD_REPRODUCIBILITY_TABLES/TABLE_S1_FEATURE_AVAILABILITY.md`
- S2 — `reports/research_log/major_revision_round2/METHOD_REPRODUCIBILITY_TABLES/TABLE_S2_PREPROCESSING_HYPERPARAMETER_SEARCH.md`
- S3 — `reports/research_log/major_revision_round2/METHOD_REPRODUCIBILITY_TABLES/TABLE_S3_CV_SEED_CONTRACT.md`
- Confusion matrices — `reports/research_log/major_revision_round2/selection_objective_sensitivity/confusion_matrix.csv`
- Supplementary Evidence Ledger — `reports/research_log/major_revision_round2/SUPPLEMENTARY_EVIDENCE_LEDGER_ROUND2.csv`

`Supplement/source-copy` manifests record the exact Git blob identities so these validated sources can be copied byte-identically when the official MDPI production ZIP is assembled.

## Submission documents

- `COVER_LETTER.md`
- `AUTHOR_DECLARATIONS_DRAFT.md`
- `AUTHOR_ACTION_REQUIRED_FINAL.md`
- `DATA_AVAILABILITY_FINAL_DRAFT.md`
- `AI_DISCLOSURE_DRAFT.md`
- `SUBMISSION_READINESS_CHECKLIST.md`

## Audits

- `FINAL_MANUSCRIPT_AUDIT.md`
- `DATASET_PROVENANCE_RIGHTS_ASSESSMENT.md`
- `GIT_HISTORY_RISK_ASSESSMENT.md`
- `SOFTWARE_LICENSE_DECISION.md`
- `MDPI_POLICY_VERIFICATION.md`
- `REFERENCE_QC_STATUS.md`
- `FIGURE_TABLE_QC_STATUS.md`
- `TEMPLATE_COMPILE_BLOCKER.md`

## Current status

Scientific/manuscript finalization is complete within the validated evidence boundary. No new model experiment was introduced. The package remains **NOT SUBMISSION READY** until author/institution declarations and the exact current official MDPI-template compile/visual-QC gate are completed.

No tag, GitHub Release, DOI, raw-data publication, software-licence selection, or destructive Git-history rewrite is authorized by this package.