# Phase 1B Ordinal Benchmark — Compact Evidence

Source run: `phase1b_v3_20260903T130912Z_dc5cb8b`

This package contains aggregate and fold-level evidence only. Employee-level OOF rows, raw data, and fitted models are deliberately excluded.

## Nine-system OOF comparison

| Model | Macro-F1 | Balanced accuracy | QWK | Ordinal MAE | Two-level reversal | RPS | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cumulative-threshold XGBoost | 0.6255 | 0.6745 | 0.5550 | 0.3142 | 0.0017 | 0.1029 | 1.3053 | 0.4156 | 0.0363 |
| XGBoost | 0.6210 | 0.6360 | 0.5676 | 0.2433 | 0.0025 | 0.0860 | 0.5515 | 0.3426 | 0.0375 |
| LightGBM | 0.6055 | 0.6218 | 0.5883 | 0.1983 | 0.0025 | 0.0804 | 0.5883 | 0.3173 | 0.0864 |
| Random Forest | 0.5923 | 0.6253 | 0.6317 | 0.1583 | 0.0017 | 0.0822 | 0.5982 | 0.3413 | 0.2086 |
| Multinomial logistic regression | 0.5062 | 0.5240 | 0.3710 | 0.3550 | 0.0125 | 0.1134 | 0.7142 | 0.4415 | 0.0815 |
| Proportional-odds logistic | 0.4844 | 0.5531 | 0.3927 | 0.4683 | 0.0092 | 0.1405 | 0.8982 | 0.5791 | 0.2066 |
| Stratified baseline | 0.3304 | 0.3310 | 0.0243 | 0.4467 | 0.0267 | 0.2233 | 15.1383 | 0.8400 | 0.4200 |
| Ordinal-median baseline | 0.2809 | 0.3333 | 0.0000 | 0.2717 | 0.0000 | 0.1358 | 9.7919 | 0.5433 | 0.2717 |
| Majority baseline | 0.2809 | 0.3333 | 0.0000 | 0.2717 | 0.0000 | 0.1358 | 9.7919 | 0.5433 | 0.2717 |

## Bounded interpretation

- Cumulative-threshold XGBoost has the highest macro-F1 (0.6255) and balanced accuracy (0.6745), but its advantage over nominal XGBoost in macro-F1 is only 0.0044 and no interval or significance conclusion is yet attached to that contrast.
- Random Forest remains strongest on QWK (0.6317) and ordinal MAE (0.1583). LightGBM has the lowest RPS (0.0804) and multiclass Brier score (0.3173); nominal XGBoost has the lowest log loss (0.5515).
- The proportional-odds model does not improve the benchmark: macro-F1 is 0.4844, QWK is 0.3927, and ordinal MAE is 0.4683.
- Cumulative-threshold XGBoost's raw log loss (1.3053) is materially worse than its classification ranking suggests. Calibration and probability-quality claims must therefore remain metric-specific.
- Majority and ordinal-median baselines coincide because class 3 is both the training majority and ordinal median in every outer fold. Their zero two-level-reversal rate is achieved by always predicting the middle class and must not be treated as overall superiority.
- These are cross-sectional P3 results under timestamp-unverified availability assumptions. They do not establish prospective validity, causality, fairness, deployment readiness, or a universally best model.
- Four nominal OOF prediction sets are immutable canonical-v2 evidence reused without refitting or relabelling. The two ordinal models and three naive baselines are newly fitted on the exact same persisted folds.

## Files

- `aggregate_metrics.csv`: full-OOF values for 16 metrics and all nine systems.
- `per_class_metrics.csv`: precision, recall, F1, and support for each class/system.
- `confusion_matrix.csv`: complete ordered 3×3 confusion grid for every system.
- `extension_fold_metrics.csv`: five newly fitted systems across ten outer folds.
- `selected_hyperparameters_by_fold.csv`: selection records without employee rows.
- `candidate_search_summary.csv`: candidate-level cross-fold selection summary.
- `provenance_receipt.json` and `manifest.json`: immutable identities and file hashes.
