# Repeated Nested-CV Protocol — v3

Date: 2026-09-04

Status: complete clean-commit execution independently validated; compact aggregate evidence published

## Estimand

This Phase 1C sensitivity measures variation caused by refitting models under different deterministic stratified outer/inner fold assignments. It complements, but does not replace, the canonical-v2 paired sample-level OOF bootstrap. Five repetition values are descriptive training/fold-variability observations; their minimum–maximum range is not a sample-level confidence interval.

The analysis remains a timestamp-unverified cross-sectional P3 experiment with 20 features. It cannot establish prospective validity, leakage elimination, causal effects, fairness, or deployment readiness.

## Prespecified computational scope

The reviewer brief prefers 10 repetitions × 5 outer folds × 5 inner folds when proportionate and permits 5×5×5 when the larger design is too costly. The frozen contract selects 5×5×5 before result inspection because the complete nine-system design contains six independently tuned model families and requires 5,725 estimator-fit calls. Ten repetitions would require 11,450 fit calls. Results cannot be used to add, remove, or select a seed; any future 10-repetition execution would require a new prespecified contract and separate identity.

The five exact seed triples are:

| Repetition | Outer seed | Inner seed | Model seed |
| ---: | ---: | ---: | ---: |
| 1 | 1042 | 1043 | 1044 |
| 2 | 2042 | 2043 | 2044 |
| 3 | 3042 | 3043 | 3044 |
| 4 | 4042 | 4043 | 4044 |
| 5 | 5042 | 5043 | 5044 |

All systems share the exact same folds within each repetition. Outer assignments must differ across repetitions, and every sample must receive exactly one outer-test prediction per system and repetition.

## Models and selection

Every repetition refits all nine Phase 1B systems:

- nominal multinomial logistic regression, Random Forest, LightGBM, and XGBoost;
- proportional-odds ordinal logistic regression and cumulative-threshold XGBoost;
- training-only majority, seeded-stratified, and lower-ordinal-median baselines.

The four nominal OOF sets from canonical v2 are not reused in this sensitivity. The six fitted model families use their exact source-registry candidate definitions. Candidate selection is independent within each repetition/model/outer-training partition: macro-F1 is primary, QWK breaks ties inside the fixed 0.001 macro-F1 pool, and the outer test is excluded from preprocessing, tuning, selection, calibration, and seed choice. Naive baselines never enter hyperparameter selection.

## Outputs and interpretation

All 16 Phase 1B metrics are computed from each system's complete OOF predictions within each repetition. The priority training-variability report covers macro-F1, balanced accuracy, QWK, and ordinal MAE and reports mean, sample SD, median, minimum, and maximum across the five repetitions.

Ordering stability is restricted to the six tuned models and includes rank by repetition, winner frequency, mean/sample-SD rank, and all ten pairwise Spearman correlations between repetition-specific rank vectors. Rankings remain metric-specific; no universal winner is allowed.

The complete local package will contain candidate/selection/fold records, fold contracts, row-level OOF predictions, repetition metrics, variability summaries, rank evidence, and provenance under the ignored `reports/major_revision_v3_runs/` root. A later governed export may publish only aggregate repetition, variability, rank, candidate-frequency, and provenance evidence. Employee-level OOF rows, fold assignments, raw data, and fitted models are not authorized for Git publication.

## Verified preflight

The fit-free real-data preflight passed on 1,200 INX rows with target support 194/874/132 and the exact 20-feature P3 frame. It generated five valid 5×5 nested fold contracts and five distinct semantic outer-assignment hashes. The contract SHA-256 is `5681e521cfbaff5963494212fcc047116056fd7d66393346df0463aef7553af9`. Preflight model fits, network calls, and paid-API calls were all zero.

An in-memory repetition-1/outer-fold-1 real-data diagnostic then completed 229 estimator-fit calls in approximately 52 seconds. It produced 44 candidate records, nine model/fold metric records, 2,160 held-out prediction rows, and 144 repetition-metric rows, but persisted nothing and carried the explicit `diagnostic_incomplete_never_canonical` status. The enforced runtime recorded zero attempted network operations and zero paid-API calls. These diagnostic values test execution feasibility only and are not scientific results.

## Complete execution and validation

Complete run `phase1c_v3_20260903T215015Z_78649c4` executed from clean commit `78649c426e69fb5270f9d027b11ba6ba87d71a41`. It produced five distinct outer-fold assignments, 1,100 candidate-search records, 225 selected-hyperparameter records, 225 fold-metric records, 54,000 OOF prediction rows, and 720 repetition-metric rows. Every one of the nine systems has exactly 1,200 held-out predictions in each repetition. The enforced runtime receipt records zero attempted network operations and zero paid-API calls.

The independent run validator checks the 12-file closed world and every byte hash, binds the repeated-CV contract, six source implementations, and all source contracts to the exact generation commit, reconstructs and validates every outer and nested-inner fold system, verifies training-only selection and OOF probability lineage, and recomputes all fold/repetition metrics and all higher-order summaries. Validation passed without relying on the runner's acceptance result.

## Results

Across the five repetitions, mean macro-F1 is 0.6288 (sample SD 0.0099) for XGBoost and 0.6249 (0.0137) for LightGBM. LightGBM wins three repetitions and XGBoost wins two, with equal mean rank 1.6. Cumulative-threshold XGBoost has the highest mean balanced accuracy at 0.6524 (0.0074) and wins four repetitions; LightGBM wins one.

Random Forest wins all five repetitions for QWK and ordinal MAE. Its mean QWK is 0.6311 (sample SD 0.0022), and mean ordinal MAE is 0.1597 (0.0015). The mean pairwise rank Spearman correlations across the six tuned models are 0.926 for macro-F1, 0.897 for balanced accuracy, 0.943 for QWK, and 0.926 for ordinal MAE.

These findings support only metric-specific claims. The classification winner changes across repetitions and criteria, while the ordinal-error winner is stable within these five prespecified repetitions. Five repetition values are not independent employee samples; their ranges are descriptive and are not confidence intervals. The proportional-odds model remains weaker than the nominal comparators.

The tracked `phase1c_repeated_nested_cv/` package contains nine files and only repetition-level or higher summaries, selection frequencies, and provenance. Employee-level OOF rows, fold assignments, fold-level metrics, raw data, candidate-search records, and fitted models remain local and ignored.
