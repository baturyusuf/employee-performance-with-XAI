# Repeated selection-objective sensitivity

## Design

The analysis reuses the five exact Phase 1C repeated 5×5 nested-CV split identities, its 1,100 hash-bound inner-candidate records, and the six trained-system registries without adding a candidate or seed. Macro-F1 selection uses QWK as the tie-break; QWK selection swaps those roles. Both use the same inclusive 0.001 primary practical-tie pool and lowest-index final tie rule. The outer test partition is evaluation-only.

The 36,000 tuned-model macro-F1-selected OOF rows are reused from the independently validated Phase 1C run. Exactly 150 new QWK-selected outer fits were performed: five repetitions × five outer folds × six models. No inner fit or baseline fit was added. The scientific lineage is 5,725 prior Phase 1C fits plus 150 new fits. Employee-level OOF rows remain local and are not published.

## Result

The selection-objective trade-off is **strongly replicated for nominal and cumulative-threshold XGBoost across all five split identities, with model-specific heterogeneity**. This classification is based on direction consistency, not a confidence interval: nominal XGBoost gained QWK and improved ordinal MAE in 5/5 repetitions while rating-4 recall fell in 5/5; cumulative-threshold XGBoost gained QWK in 5/5 while QWK-selected rating-4 recall was zero in 5/5. Random Forest changed much less and already had near-zero rating-4 recall under both regimes. The effect is therefore persistent for the boosting systems that drive the canonical contrast, but it is not universal across models and its magnitude varies by split.

| System | Selection | Mean macro-F1 | Mean QWK | Mean ordinal MAE | Mean rating-4 recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Nominal XGBoost | macro_f1 | 0.6288 | 0.5833 | 0.2335 | 0.1894 |
| Nominal XGBoost | qwk | 0.6021 | 0.6377 | 0.1550 | 0.0167 |
| Cumulative-threshold XGBoost | macro_f1 | 0.6155 | 0.5451 | 0.3062 | 0.3288 |
| Cumulative-threshold XGBoost | qwk | 0.5933 | 0.6374 | 0.1522 | 0.0000 |
| Random Forest | macro_f1 | 0.5955 | 0.6311 | 0.1597 | 0.0106 |
| Random Forest | qwk | 0.5952 | 0.6351 | 0.1565 | 0.0061 |

## Repetition-level direction checks

| Quantity | Positive | Negative | Zero | Mean delta | Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal XGB ΔQWK | 5/5 | 0/5 | 0/5 | 0.0544 | 0.0349 to 0.0820 |
| Nominal XGB ΔMAE | 0/5 | 5/5 | 0/5 | -0.0785 | -0.1017 to -0.0600 |
| Nominal XGB Δrating-4 recall | 0/5 | 5/5 | 0/5 | -0.1727 | -0.1818 to -0.1515 |
| Cumulative XGB ΔQWK | 5/5 | 0/5 | 0/5 | 0.0923 | 0.0817 to 0.0990 |
| Cumulative XGB Δrating-4 recall | 0/5 | 5/5 | 0/5 | -0.3288 | -0.3485 to -0.2879 |
| RF ΔQWK | 4/5 | 0/5 | 1/5 | 0.0040 | 0.0000 to 0.0096 |
| RF Δrating-4 recall | 0/5 | 2/5 | 3/5 | -0.0045 | -0.0152 to 0.0000 |

Deltas are QWK-selected minus macro-F1-selected. Negative MAE is an improvement; negative rating-4 recall is deterioration.

Candidate selections changed in 15, 20, 17, 20, and 20 of 30 model-fold opportunities across repetitions (92/150 overall). Leaders changed for 7, 5, 7, 4, and 7 of seven reported metrics, while the full six-model ordering changed for all seven metrics in every repetition. These are separate descriptive facts, not a composite materiality flag.

## Claim boundary

The canonical 10×5 result remains a valid result for its persisted split identity. The repeated analysis strengthens the directional robustness of the XGBoost selection trade-off while showing model and magnitude heterogeneity. It does not show that QWK selection always harms the highest class, that the effect is universal, or that five repetitions are independent inferential samples.

Exact aggregate results are in `REPEATED_SELECTION_RESULTS.csv`; fold-level selected candidate IDs are in `repeated_selection/selection_schedule.csv`. The independent receipt reports 72,000 locally validated OOF rows, 58 unchanged-candidate replay cells, zero prediction-label differences in those cells, and a maximum floating-point serialization difference of 3.39×10⁻²¹.
