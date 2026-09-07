# Core Dataset Data Quality Report

Source run: `phase3b_v3_20260907T154418Z_0d9643b`

## Scope and method

This aggregate-only audit covers the two datasets in the canonical core evidence scope: INX and HRDataset_v14. The IBM and turnover datasets are supplementary and are not represented here as core datasets. Exact pinned CSV bytes were loaded offline; no automatic download, model fitting, paid API, source-row repair, or row-level publication occurred.

Missingness treats whitespace-only strings as missing for audit purposes. Duplicate checks report exact rows, predictor-equal rows after excluding the raw target, and conflicting-target predictor groups. A nonmissing mode share of at least 99% defines near-constant. Identifier candidates require a declaration, an exact row-order sequence, or both a name signal and at least 98% uniqueness. Numeric, temporal, and text/code rules were fixed in `configs/data_quality_v3.json` before the exact run.

Raw and cleaned schema hashes are SHA-256 digests of canonical JSON records containing ordered position, column name, and pandas dtype. The cleaned schema is the canonical pre-model table: INX remains the verified 28-column frame; HRDataset_v14 has 39 columns after governed renaming/string trimming, a generated excluded row key, review-minus-hire tenure derivation, and mapped 2/3/4 target creation. Per-fold preprocessing remains outside this audit and is training-only in the modeling protocols.

## Manuscript-ready summary

| Dataset | Rows | Raw columns | Cleaned columns | Missing cells | Exact duplicate extra rows | Near-constant columns | Identifier candidates in primary features | Rules with findings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| INX (primary) | 1,200 | 28 | 28 | 0 (0.00%) | 0 | 0 | 0 | 0 |
| HRDataset_v14 (independent replication) | 311 | 36 | 39 | 215 (1.92%) | 0 | 0 | 0 | 3 |

The machine-readable version of this table is `manuscript_ready_data_quality.csv`.

## Findings

### INX

The 1,200 × 28 raw table has no effective missing cells, blank strings, exact duplicate rows, predictor-equal duplicate rows, constant columns, or near-constant columns. The declared numeric domains and four tenure relations have no violations. PerformanceRating support is 194/874/132 for labels 2/3/4. EmpNumber is the single detected identifier candidate and is absent from the P3 primary feature set.

These are results under the declared rules, not proof that the data are error-free, representative, authentic, prospectively valid, or licensed for every use.

### HRDataset_v14

The 311 × 36 raw table has no exact or predictor-equal duplicate rows, no constant/near-constant columns, and 215 effective missing cells: 207 DateofTermination values and 8 ManagerID values. The termination-date missingness is structurally aligned with the 207 records whose Termd flag is zero; all 104 terminated records have a termination date. Missing ManagerID remains missing and is not repaired.

Three declared rules have findings, each affecting two rows (0.64% of the checked rows): the last performance-review date precedes the hire date; PerformanceScore text disagrees with PerfScoreID; and two rows are minority Department texts within their DeptID group. The governed tenure derivation preserves the first finding by setting the two negative durations to missing. Performance modeling uses the PerformanceScore text mapping, not PerfScoreID, and excludes both target aliases. The conservative primary policy also excludes Department/DeptID and every declared or generated identifier candidate.

Raw target support is PIP 13, Needs Improvement 18, Fully Meets 243, and Exceeds 37. The retained mapped target support is 31/243/37 for labels 2/3/4. This dataset-specific mapping does not establish construct or prevalence equivalence with INX.

## Identifier and leakage interpretation

The audit detects/examines INX EmpNumber and HRDataset_v14 Employee_Name, EmpID, ManagerName, ManagerID, plus the generated ExternalSampleId. All are excluded from their declared primary model feature sets. High cardinality alone is not treated as identity, so Salary and DOB are not mislabeled merely because their observed values are mostly unique. Identifier exclusion reduces a direct memorization channel; it does not prove that indirect proxy leakage is absent.

## Construct and generalization boundaries

PerformanceRating/PerformanceScore are recorded organizational performance ratings. The pinned source documentation does not establish either target as true employee capability, objective productivity, or future potential. The data-quality audit does not create external generalization evidence, validate a locked INX model on HRDataset_v14, establish causal validity or fairness, certify deployment readiness, or resolve source authenticity and licence review.

No source values were corrected. Any future correction would require a separately justified, versioned sensitivity analysis rather than silent cleaning.

## Package contents

- `dataset_summary.csv`: dataset-level dimensions, schema hashes, missingness, duplicates, identifiers, and rule findings.
- `column_profiles.csv` and `categorical_cardinality.csv`: complete aggregate raw-column profiles without example values.
- `duplicate_audit.csv` and `identifier_audit.csv`: duplicate/target-conflict counts and primary-feature exclusion checks.
- `rule_anomaly_audit.csv`: all 52 declared domain, temporal, and consistency rules, including zero-retaining rows.
- `target_distribution.csv`: complete raw and governed mapped target support.
- `raw_schema.csv` and `cleaned_schema.csv`: ordered schemas, transformations, missing counts, and schema hashes.
- `manuscript_ready_data_quality.csv`: compact two-row table for later claim-matrix/manuscript work.
- `provenance_receipt.json` and `manifest.json`: exact source-run validation, publication controls, byte sizes, and hashes.
