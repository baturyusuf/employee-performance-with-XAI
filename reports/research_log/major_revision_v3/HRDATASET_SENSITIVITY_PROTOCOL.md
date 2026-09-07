# HRDataset_v14 Target-Mapping and Repeated-CV Sensitivity Protocol

Date frozen: 2026-09-07
Phase: 3A
Status: implementation contract

## Estimand and language boundary

This stage is an **independent protocol replication** (also described as a cross-dataset protocol replication). It retrains models in HRDataset_v14, uses a different seven-feature schema, and maps a dataset-specific organisational performance rating. It is not locked-model transport or evidence of target equivalence, universal external validity, deployment readiness, fairness, or causality.

## Target formulations

The retained primary mapping is `PIP + Needs Improvement -> 2`, `Fully Meets -> 3`, and `Exceeds + Exceptional -> 4`. Its results remain the primary cross-dataset replication sensitivity, with the canonical v2 10-outer-fold result retained as a descriptive reference.

The supplementary formulation preserves the raw ordered categories: `PIP -> 1`, `Needs Improvement -> 2`, `Fully Meets -> 3`, and `Exceeds + Exceptional -> 4`. This isolates the consequence of merging the two low-rating categories without inventing a new construct. `Exceptional` is absent from the observed 311 rows; its mapping only makes the contract total over the documented category vocabulary.

Metrics from the three- and four-class targets are shown side by side but are not subtracted or treated as estimates of the same estimand.

## Repeated nested cross-validation

Each formulation is evaluated with five prespecified repetitions of stratified 5-fold outer and 5-fold inner CV. Within an outer training partition, eight frozen XGBoost candidates are selected by mean inner macro-F1, with QWK as the prespecified practical-tie breaker. Preprocessing, class weights, candidate selection, and calibration are fit using outer-training data only. Outer-test observations are evaluation-only.

Raw probabilities and the predeclared one-vs-rest sigmoid calibration are reported. The sigmoid is fit to five-fold selected-candidate OOF probabilities generated entirely inside each outer-training partition and is then applied once to the corresponding untouched outer-test probabilities. No calibration method is selected from outer-test results.

Three training-only baselines are evaluated on the same outer folds: majority class, stratified random draws from outer-training prevalence, and the lower ordinal median class.

## Outputs and interpretation

For every repetition/system, the stage reports accuracy, macro-F1, balanced accuracy, QWK, ordinal MAE, two-level reversal rate, log loss, multiclass Brier, ten-bin top-label ECE, and RPS. It also retains class-specific precision/recall/F1/support and complete confusion matrices. Variability summaries are descriptive across the five training/fold repetitions; their ranges are not confidence intervals.

The canonical v2 10-fold primary-mapping OOF predictions are used only to produce a fit-free per-class/confusion reference and a descriptive CV-design comparison. The v2 package is never mutated.

All row-level OOF predictions, fold assignments, fitted objects, and calibration parameters stay outside the tracked compact package. Scientific execution is offline with zero paid API calls.
