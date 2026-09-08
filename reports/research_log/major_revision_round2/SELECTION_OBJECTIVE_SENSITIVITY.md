# Selection-Objective Sensitivity

Status: **independently validated aggregate evidence**

Source run: `round2_selection_v4_20260908T135109Z_8b67f9e`

Generation commit: `8b67f9e23ec8aaba52e4b5422d63cd33eb3590b9`

Scientific-input SHA-256: `872cc2195755c7ff943146fdb617c3a36b19fd101d3616060a6f7100bfaab796`

Compact payload-set SHA-256: `0860fbbc182a4de1eaca555b8650fdd312573928b802ea1468a18763b40bf94a`

## Design

Both regimes use the exact P3 feature contract, candidate registries, 10 outer × 5 inner fold identities, preprocessing, and seeds. Regime A selects by inner macro-F1 with QWK as the tie-break inside the inclusive 0.001 pool and reuses the validated Phase 1B OOF predictions. Regime B selects by inner QWK with macro-F1 as the tie-break inside the same pool and refits all six models in all ten outer folds. No new inner-CV model fit was performed because both scores had been retained for every exhaustively evaluated candidate.

The complete run contained 440 candidate-evidence rows, 120 regime/model/fold selections, 60 QWK outer refits, and 14,400 exactly-once model-regime OOF rows. The independent validator reconstructed the selections, verified exact historical OOF replay, and recomputed every metric, per-class row, and confusion cell.

## Candidate-selection changes

| Model | Changed folds | Total folds |
| --- | ---: | ---: |
| Cumulative-threshold XGBoost | 10 | 10 |
| LightGBM | 10 | 10 |
| Logistic regression | 1 | 10 |
| Proportional-odds logistic regression | 2 | 10 |
| Random Forest | 4 | 10 |
| Nominal XGBoost | 10 | 10 |
| **All models** | **37** | **60** |

Exact fold-level candidate indices and their inner-score evidence are in [`selection_schedule.csv`](selection_objective_sensitivity/selection_schedule.csv) and [`selected_candidate_changes.csv`](selection_objective_sensitivity/selected_candidate_changes.csv).

## Principal OOF results

| Model | Selection | Macro-F1 | QWK | Ordinal MAE | RPS | Rating-4 recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cumulative-threshold XGBoost | Macro-F1 | 0.625459 | 0.554974 | 0.314167 | 0.102926 | 0.416667 |
| Cumulative-threshold XGBoost | QWK | 0.591213 | 0.631887 | 0.153333 | 0.066586 | 0.000000 |
| LightGBM | Macro-F1 | 0.605488 | 0.588329 | 0.198333 | 0.080406 | 0.060606 |
| LightGBM | QWK | 0.598731 | 0.609746 | 0.170000 | 0.075613 | 0.030303 |
| Logistic regression | Macro-F1 | 0.506221 | 0.371011 | 0.355000 | 0.113414 | 0.174242 |
| Logistic regression | QWK | 0.501317 | 0.365640 | 0.360833 | 0.113113 | 0.174242 |
| Proportional-odds logistic regression | Macro-F1 | 0.484377 | 0.392708 | 0.468333 | 0.140480 | 0.409091 |
| Proportional-odds logistic regression | QWK | 0.463053 | 0.398315 | 0.529167 | 0.152184 | 0.515152 |
| Random Forest | Macro-F1 | 0.592340 | 0.631678 | 0.158333 | 0.082178 | 0.000000 |
| Random Forest | QWK | 0.590600 | 0.633509 | 0.156667 | 0.083204 | 0.000000 |
| Nominal XGBoost | Macro-F1 | 0.621021 | 0.567602 | 0.243333 | 0.085974 | 0.174242 |
| Nominal XGBoost | QWK | 0.601378 | 0.641778 | 0.152500 | 0.067467 | 0.007576 |

Values are rounded to six decimals; [`aggregate_metrics.csv`](selection_objective_sensitivity/aggregate_metrics.csv), [`per_class_metrics.csv`](selection_objective_sensitivity/per_class_metrics.csv), and [`confusion_matrix.csv`](selection_objective_sensitivity/confusion_matrix.csv) are authoritative.

## Separate ranking diagnostics

- The leader changed for six of the nine prespecified evaluation metrics: macro-F1, balanced accuracy, QWK, ordinal MAE, RPS, and multiclass Brier.
- The leader did not change for log loss, confidence ECE, or two-level reversal rate.
- The exact six-model ordering changed for all nine metrics.
- Under macro-F1 selection, cumulative-threshold XGBoost led macro-F1, Random Forest led QWK and ordinal MAE, LightGBM led RPS and Brier, and nominal XGBoost led log loss.
- Under QWK selection, nominal XGBoost led macro-F1, balanced accuracy, QWK, ordinal MAE, and log loss; cumulative-threshold XGBoost led RPS, Brier, ECE, and two-level reversal rate.

These are separate descriptive facts, not one binary material-dependence classification. The exact ordering, leader margins, signed model-level changes, and rank-position changes are retained in [`ranking_changes.csv`](selection_objective_sensitivity/ranking_changes.csv) and [`metric_effect_magnitudes.csv`](selection_objective_sensitivity/metric_effect_magnitudes.csv).

## Interpretation

Changing the selection objective altered many selected candidates and several metric leaders in this frozen benchmark, but the effects were model- and metric-specific. For cumulative-threshold XGBoost, QWK selection increased QWK by 0.076913 and reduced ordinal MAE by 0.160833 while reducing macro-F1 by 0.034246; its rating-4 recall simultaneously changed from 0.416667 to 0.000000. Nominal XGBoost became the QWK and ordinal-MAE leader under QWK selection, yet its rating-4 recall was only 0.007576. Random Forest retained zero rating-4 recall in both regimes.

The study is descriptive and conditional on one dataset, feature policy, candidate registry, fold schedule, and seed contract. It is not a hypothesis test, universal model ranking, or deployment validation. In particular, strong QWK or ordinal MAE can coexist with near-complete or complete failure on rating 4.
