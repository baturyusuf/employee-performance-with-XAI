# Ordinal Benchmark Protocol — v3

Date: 2026-09-03

Status: implementation validated; full clean-commit execution pending

## Evidence composition

The benchmark will compare nine systems on the exact persisted INX outer folds:

- four immutable canonical-v2 nominal OOF comparators: multinomial logistic regression, random forest, LightGBM, and XGBoost;
- two newly fitted ordinal models: proportional-odds ordinal logistic regression and nonlinear cumulative-threshold XGBoost;
- three newly fitted training-only naive comparators: majority class, seeded stratified random, and lower ordinal median.

The four v2 nominal predictions are reused without refitting or relabelling. Their exact OOF and fold artifacts are hash-bound in `configs/ordinal_benchmark_v3.json`. The new models consume the same 10 outer by 5 inner persisted split assignments, the same P3 feature set, and training-partition-only preprocessing.

## Ordinal model definitions

The proportional-odds model uses a cumulative-logit likelihood with one shared linear coefficient vector and strictly increasing learned thresholds. It is fitted by deterministic L-BFGS-B with an analytic gradient and coefficient-only L2 regularization. Candidate selection varies `C` and training-derived class balancing.

The nonlinear ordinal model fits one binary XGBoost model for each event `P(y > threshold)` using training labels only. Independently estimated cumulative probabilities are projected rowwise onto a non-increasing sequence by the pool-adjacent-violators algorithm before differencing them into class probabilities. This projection is a coherence correction, not calibration.

## Selection and evaluation

- Candidate selection is performed independently within each model and outer-training partition.
- Primary inner metric: macro-F1; deterministic tie-break: QWK within a 0.001 macro-F1 tolerance.
- Outer-test observations are excluded from preprocessing, fitting, weighting, candidate selection, and tie-breaking.
- Baselines have no tuned hyperparameters and learn only their required class statistic from the current outer-training labels.
- Aggregate evaluation contains 16 metrics, including normalized ranked probability score (RPS) and the manuscript-facing `two_level_reversal_rate` name.
- Each model also receives a complete three-class precision/recall/F1/support table and a 3×3 ordered confusion grid.
- Metric-specific rankings will be reported; no universally best model will be declared.
- XGBoost remains the prespecified XAI reference for exact-fold TreeSHAP continuity, independently of predictive ranking.

## Source and publication boundaries

The contract binds the canonical loader and acquisition manifest, the full feature-availability contract, the nominal model-grid file, and four canonical-v2 fold/OOF artifacts. Full execution requires a clean Git worktree and records the exact commit plus a source-tree hash before fitting; identity and source bytes are rechecked before atomic publication.

Employee-level OOF rows remain under the ignored local `reports/major_revision_v3_runs/` root. Fitted models and row-level data are not authorized for Git publication. A later governed export may admit only compact aggregate, per-class, confusion, selection, and provenance evidence.

## Verified preflight and diagnostic

The real-data fit-free preflight passed with dataset SHA-256 `b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a`, 1,200 samples, target support 194/874/132, 20 P3 features, fold-contract hash `c1300316fe5baec24e789c06aec35dd4f283fa4843b71c7aab1edbf4818f8e91`, and 4,800 aligned v2 nominal OOF rows.

A one-outer-fold real-data diagnostic completed in 9.743 seconds with 14 candidate rows, five system fold rows, and 600 held-out prediction rows. Its explicit status is `diagnostic_incomplete_never_canonical`; it was not persisted and its candidate choices are not scientific results. The complete run remains pending until this implementation is committed and the repository is clean.
