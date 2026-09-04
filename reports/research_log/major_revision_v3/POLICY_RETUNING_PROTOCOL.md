# Fixed-Schedule and Independently Retuned Policy Protocol — v3

Date: 2026-09-04

Status: complete clean-commit execution independently validated; compact aggregate evidence published

## Purpose and two estimands

Phase 1D separates two questions that cannot be answered by one feature-policy experiment.

1. **Fixed primary schedule / feature-access sensitivity:** how held-out predictions change when the permitted feature set changes while the primary P3 fold-specific XGBoost candidate schedule is held fixed. This is a controlled model-and-schedule comparison, not a causal feature-access effect.
2. **Independently retuned policy performance:** what held-out performance each policy achieves when candidate selection is repeated independently inside that policy's outer-training data. This is conditional on the prespecified model family, candidate grid, and folds; it is not a pure causal retuning effect or an unconstrained best-achievable claim.

The raw difference is always `retuned - fixed`. A second direction-aligned column changes the sign for lower-is-better metrics. A point difference is descriptive and does not authorize a significance, causal, or universally best-policy claim.

## Information policies and historical boundary

The exact Phase 1A P0–P5 contract is reused, with 26, 24, 21, 20, 13, and 6 retained features respectively. The scenario remains timestamp-unverified and cross-sectional. P4 `STRICT_PROSPECTIVE` is a prospective-plausibility sensitivity, not prospective validation; none of the policies is called leakage-free.

The fixed-schedule estimand reuses immutable canonical-v2 OOF evidence only where the feature set is exactly identical:

| v3 policy | Fixed-schedule source | Evidence action |
| --- | --- | --- |
| P0 | `full_feature_upper_bound` | Exact OOF reuse |
| P1 | `no_salary_hike_no_attrition_sensitive_retaining_audit` | Exact OOF reuse |
| P2 | `no_salary_hike_no_attrition` | Exact OOF reuse |
| P3 | `no_salary_hike_no_attrition_no_department` | Exact OOF reuse and primary-benchmark replay |
| P4 | none | New outer-training fit using the primary P3 fold candidate |
| P5 | none | New outer-training fit using the primary P3 fold candidate |

The historical v2 policies `no_salary_hike` and `no_salary_hike_no_attrition_no_department_no_job_role` remain unmodified. They are excluded from the P0–P5 comparison because they are not exact members of the v3 information-policy contract; they are neither deleted nor suppressed.

## Folds, tuning, and computation

Both estimands use the exact persisted canonical-v2 10-fold outer and 5-fold inner assignments. Every policy/estimand must contain one and only one held-out prediction for each of the 1,200 employees. Outer-test rows are evaluation-only and cannot enter preprocessing, tuning, candidate selection, or policy choice.

Only XGBoost is evaluated so that the policy question is not mixed with model-family selection. The exact eight-candidate XGBoost registry from `configs/model_grid.yaml` is used. For the retuned estimand, each policy and outer fold fits all eight candidates across five inner folds, chooses by inner macro-F1, and uses QWK to break ties within the fixed 0.001 macro-F1 pool. Preprocessing is fitted only on the active inner- or outer-training partition. Estimator threads are fixed at one and early stopping is disabled.

The complete execution requires 2,480 new estimator fits: 2,400 inner candidate fits, 60 retuned outer refits, and 20 new P4/P5 fixed-schedule outer fits. P0–P3 fixed OOF rows are reused, not refitted. Retuned P3 must exactly reproduce the canonical primary candidate schedule, labels, and probabilities within an absolute tolerance of `1e-12`.

## Metrics, outputs, and publication controls

All 16 Phase 1B aggregate metrics are reported for both estimands. The headline table includes macro-F1, QWK, balanced accuracy, and ordinal MAE. Complete local evidence includes candidate search, selected candidates, fold metrics, fixed and retuned OOF predictions, policy features, comparisons, and provenance under the ignored `reports/major_revision_v3_runs/` root.

Only aggregate metrics, full metric comparisons, headline comparisons, selected-candidate frequencies, and provenance are eligible for a later governed compact export. Employee-level OOF rows, fold assignments, candidate-search rows, fitted models, raw data, and secrets are prohibited from Git publication.

## Verified implementation preflight

The fit-free real-data preflight passed on all 1,200 INX rows with target support 194/874/132. It verified the six policy feature counts, the exact 10×5 fold system, eight candidates, 4,800 reusable fixed OOF rows, and 2,480 planned new fits. It made zero model fits, network calls, or paid-API calls. The policy-retuning contract SHA-256 is `d10c6f6c5e3a61e3895220f4d43a8d682e4d98c83b165f6694b20570ae950d22`.

An in-memory P3/outer-fold-1 diagnostic then completed eight candidate records, one selected-hyperparameter record, 120 reused fixed OOF rows, and 120 independently retuned OOF rows. The P3 candidate schedule, labels, and probabilities replayed exactly. The diagnostic persisted nothing, was explicitly marked `diagnostic_incomplete_never_canonical`, and recorded zero attempted network operations and zero paid-API calls. These values establish execution feasibility only and are not scientific results.

## Complete execution and independent validation

Complete run `phase1d_v3_20260904T063324Z_823c848` executed from clean commit `823c84866b461266c75f3224527f679a86ab670e`. It produced 480 candidate-search records, 60 selected-hyperparameter records, 7,200 fixed and 7,200 retuned OOF rows, 120 fold-metric rows, 192 aggregate metrics, 96 full comparisons, and six headline policy rows. The enforced runtime recorded zero attempted network operations and zero paid-API calls.

The independent validator checks the exact 12-file closed world and output hashes; binds the contract, source artifacts, and five implementation blobs to the generation commit; reconstructs the policy feature sets and canonical fold/target lineage; proves exact fixed P0–P3 OOF reuse; replays all 60 policy/fold candidate selections from the 480 training-only candidate records; validates probability simplexes and exactly-once coverage; confirms fixed-schedule lineage for P4/P5; and recomputes every fold, aggregate, comparison, and headline result from OOF evidence. P3 label, candidate, and probability replay error is exactly zero.

## Results

Independent retuning raises macro-F1 point estimates for P0, P1, P2, P4, and P5 by 0.0029, 0.0117, 0.0185, 0.0157, and 0.0087 respectively. P3 is unchanged by construction and exact replay. P2 also improves QWK by 0.0324, balanced accuracy by 0.0174, and ordinal MAE by 0.0267 in the favorable direction.

The direction is not uniform. P0 balanced accuracy falls by 0.0115. P5 macro-F1 rises by 0.0087 and ordinal MAE improves by 0.0442, while QWK falls by 0.0344 and balanced accuracy by 0.0071. P4 improves macro-F1, QWK, and balanced accuracy but has unchanged ordinal MAE. These are descriptive point differences under one fixed fold system, not intervals or significance tests.

P0's very high scores are retained adverse governance evidence: the policy includes outcome-proximal/timing-risk fields and is an information-rich diagnostic upper bound, not a deployable system. P4 remains only a prospective-plausibility sensitivity, and P5 does not establish absence of residual proxies or fairness. The tracked `phase1d_policy_retuning/` package contains only the four aggregate tables, bounded README, provenance, and manifest. Employee-level OOF rows, folds, fold metrics, candidate-search rows, selected fold records, raw data, and fitted models remain local and ignored.
