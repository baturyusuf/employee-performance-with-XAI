# Selection-Objective Sensitivity and Empirical-Prior Baseline

Validated local source run: `round2_selection_v4_20260908T135109Z_8b67f9e`.

Macro-F1 selection reuses the exact Phase 1B OOF predictions. QWK selection swaps the primary and tie-break metrics inside the same inclusive 0.001 pool and refits every model in every outer fold. The empirical prior uses only the current outer-training labels.

## Candidate changes

| Model | Changed folds | Total folds |
| --- | ---: | ---: |
| cumulative_threshold_xgboost | 10 | 10 |
| lightgbm | 10 | 10 |
| logistic_regression | 1 | 10 |
| proportional_odds_logistic | 2 | 10 |
| random_forest | 4 | 10 |
| xgboost | 10 | 10 |

## Ranking diagnostics

`leader_changed` and `full_ordering_changed` are reported separately. A small lower-order permutation is not labelled material dependence.

| Metric | Leader (macro-F1 selection) | Leader (QWK selection) | Leader changed | Full ordering changed |
| --- | --- | --- | --- | --- |
| macro_f1 | cumulative_threshold_xgboost | xgboost | true | true |
| balanced_accuracy | cumulative_threshold_xgboost | xgboost | true | true |
| quadratic_weighted_kappa | random_forest | xgboost | true | true |
| ordinal_mae | random_forest | xgboost | true | true |
| ranked_probability_score | lightgbm | cumulative_threshold_xgboost | true | true |
| nll_log_loss | xgboost | xgboost | false | true |
| multiclass_brier | lightgbm | cumulative_threshold_xgboost | true | true |
| ece_confidence | cumulative_threshold_xgboost | cumulative_threshold_xgboost | false | true |
| two_level_reversal_rate | cumulative_threshold_xgboost | cumulative_threshold_xgboost | false | true |

## Maximum observed model-level score change by metric

These are descriptive absolute magnitudes, not a composite materiality decision.

| Metric | Maximum absolute change |
| --- | ---: |
| balanced_accuracy | 0.053662 |
| ece_confidence | 0.065093 |
| macro_f1 | 0.034246 |
| multiclass_brier | 0.152897 |
| nll_log_loss | 0.816727 |
| ordinal_mae | 0.160833 |
| quadratic_weighted_kappa | 0.076913 |
| ranked_probability_score | 0.036340 |
| two_level_reversal_rate | 0.003333 |

## Outer-training empirical-prior probability baseline

| Metric | Value |
| --- | ---: |
| ece_confidence | 0.000000 |
| multiclass_brier | 0.431305 |
| nll_log_loss | 0.768298 |
| ranked_probability_score | 0.116719 |

## Interpretation boundary

The tables describe sensitivity to a prespecified hyperparameter-selection objective under one dataset, policy, fold schedule, candidate registry, and seed contract. They do not establish a universal best model, statistical significance, deployment validity, or causal HR evidence. Row-level OOF predictions and fitted internals are intentionally excluded from this compact package.
