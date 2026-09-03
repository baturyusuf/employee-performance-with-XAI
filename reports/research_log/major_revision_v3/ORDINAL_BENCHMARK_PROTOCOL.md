# Ordinal Benchmark Protocol — v3

Date: 2026-09-03

Status: complete exact-commit execution independently validated; compact aggregate evidence published

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

A one-outer-fold real-data diagnostic completed in 9.743 seconds with 14 candidate rows, five system fold rows, and 600 held-out prediction rows. Its explicit status is `diagnostic_incomplete_never_canonical`; it was not persisted and its candidate choices are not scientific results.

## Completed Phase 1B result

The complete run `phase1b_v3_20260903T130912Z_dc5cb8b` was generated from clean commit `dc5cb8b96b096bb2efc6c242403b7e51f870a01b`. Its scientific-input SHA-256 is `09e39f0920369ecfbc5be28d731833c7695e2618de47825ffa4529362dfc0b2a`; the benchmark-contract SHA-256 is `39bcc62580515888783120a00ed807c0ede0f4c46f587f50897aced4c7999b02`. The local closed-world run contains nine files, 10,800 combined exactly-once OOF rows, 6,000 newly fitted extension rows, 140 candidate-selection rows, 144 aggregate metric rows, 27 per-class rows, and 81 confusion cells. Scientific execution made zero network and zero paid-API calls.

Independent post-run validation rehashed every output, resolved the configuration and three principal implementation modules from the generation commit, revalidated the four immutable v2 source artifacts, checked fold/target/probability lineage, and recomputed aggregate, per-class, and confusion outputs from OOF rows. The tracked [`compact evidence package`](phase1b_ordinal_benchmark/README.md) contains only aggregate/fold-level results and provenance; employee-level OOF rows and fitted models remain excluded.

The result is metric-dependent. Cumulative-threshold XGBoost has the highest macro-F1 (0.6255) and balanced accuracy (0.6745), but improves macro-F1 over nominal XGBoost by only 0.0044 and has much worse log loss (1.3053). Random Forest has the highest QWK (0.6317) and lowest ordinal MAE (0.1583), LightGBM has the lowest RPS (0.0804) and Brier score (0.3173), and nominal XGBoost has the lowest log loss (0.5515). Proportional-odds logistic is weaker than the nominal models. No interval/significance conclusion or universal-best-model claim is attached to these contrasts.
