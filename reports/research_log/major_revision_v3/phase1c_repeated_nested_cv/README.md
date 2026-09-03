# Phase 1C Repeated Nested-CV — Compact Evidence

Source run: `phase1c_v3_20260903T215015Z_78649c4`

This package contains repetition-level and higher-order summaries only. Employee-level OOF predictions, fold assignments, fold metrics, raw data, candidate-search rows, and fitted models are deliberately excluded.

## Five-repetition priority metrics

Values are mean ± sample SD across five prespecified 5×5 nested-CV repetitions.

| Model | Macro-F1 | Balanced accuracy | QWK | Ordinal MAE |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.6288 ± 0.0099 | 0.6446 ± 0.0094 | 0.5833 ± 0.0156 | 0.2335 ± 0.0150 |
| LightGBM | 0.6249 ± 0.0137 | 0.6364 ± 0.0127 | 0.6070 ± 0.0136 | 0.1902 ± 0.0051 |
| Cumulative-threshold XGBoost | 0.6155 ± 0.0059 | 0.6524 ± 0.0074 | 0.5451 ± 0.0083 | 0.3062 ± 0.0092 |
| Random Forest | 0.5955 ± 0.0027 | 0.6283 ± 0.0012 | 0.6311 ± 0.0022 | 0.1597 ± 0.0015 |
| Multinomial logistic regression | 0.5037 ± 0.0095 | 0.5195 ± 0.0113 | 0.3817 ± 0.0126 | 0.3212 ± 0.0644 |
| Proportional-odds logistic | 0.4772 ± 0.0097 | 0.5492 ± 0.0230 | 0.3789 ± 0.0121 | 0.4730 ± 0.0375 |
| Stratified baseline | 0.3345 ± 0.0122 | 0.3356 ± 0.0116 | 0.0076 ± 0.0290 | 0.4617 ± 0.0338 |
| Ordinal-median baseline | 0.2809 ± 0.0000 | 0.3333 ± 0.0000 | 0.0000 ± 0.0000 | 0.2717 ± 0.0000 |
| Majority baseline | 0.2809 ± 0.0000 | 0.3333 ± 0.0000 | 0.0000 ± 0.0000 | 0.2717 ± 0.0000 |

## Bounded interpretation

- Macro-F1 does not have a single repetition winner: LightGBM wins 3/5 and XGBoost 2/5. Their mean ranks are both 1.6, while mean macro-F1 is 0.6288 for XGBoost and 0.6249 for LightGBM.
- Cumulative-threshold XGBoost has the highest mean balanced accuracy (0.6524) and wins 4/5 repetitions; LightGBM wins the remaining repetition.
- Random Forest wins all 5/5 repetitions on both QWK (mean 0.6311, SD 0.0022) and ordinal MAE (mean 0.1597, SD 0.0015).
- Mean pairwise rank Spearman correlations are 0.926 for macro-F1, 0.897 for balanced accuracy, 0.943 for QWK, and 0.926 for ordinal MAE.
- Model ordering is therefore metric-specific: ordinal-error ordering is stable at the winner, whereas the classification winner varies across repetitions. No universally best model is identified.
- The reported minimum–maximum ranges are empirical repetition ranges, not confidence intervals and not sample-level uncertainty estimates.
- Five repetitions were fixed before result inspection as the bounded-cost option (5,725 planned estimator fits); no seed or repetition was selected after seeing results.
- Every system was refitted in every repetition. Canonical-v2 OOF predictions were not reused for this estimand.
- These remain cross-sectional P3 results under timestamp-unverified feature availability. They do not establish prospective validity, causality, fairness, or deployment readiness.

## Files

- `repetition_metrics.csv`: all 16 metrics for nine systems in each repetition.
- `variability_summary.csv`: mean, sample SD, median, minimum, and maximum for four priority metrics.
- `rank_by_repetition.csv`: six tuned-model ranks and winners for each priority metric.
- `model_rank_summary.csv`: descriptive rank and winner-frequency summaries.
- `ordering_stability.csv`: pairwise rank-correlation and winner-stability summaries.
- `selected_candidate_frequency.csv`: fold-level tuning-choice frequencies without outer-test results or employee rows.
- `provenance_receipt.json` and `manifest.json`: source validation, immutable identities, and byte hashes.
