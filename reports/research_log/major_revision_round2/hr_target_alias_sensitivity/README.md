# HR Target-Alias Matched-Sample Sensitivity

Validated local source run: `round2_hr_alias_v4_20260908T141019Z_ac50c6e`.

The historical 311-row Phase 3A result is preserved. Its existing predictions are also restricted fit-free to the 309 rows without target-alias disagreements. A separate model is selected and refitted after excluding those two rows, then evaluated on exactly the same 309-row population.

## Arm metrics

| Arm | N | Macro-F1 | Balanced accuracy | QWK | Ordinal MAE | RPS | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_canonical_311 | 311 | 0.657026 | 0.661772 | 0.549509 | 0.180064 | 0.072499 | 0.535011 | 0.290338 | 0.075406 |
| restricted_canonical_309 | 309 | 0.662383 | 0.662666 | 0.546473 | 0.177994 | 0.071863 | 0.531707 | 0.287557 | 0.073421 |
| exclusion_refit_309 | 309 | 0.666593 | 0.660564 | 0.546134 | 0.177994 | 0.071886 | 0.542510 | 0.288226 | 0.062224 |

## Comparison boundaries

- `sample_removal_effect`: `historical_canonical_311` → `restricted_canonical_309`; `fit_free_population_change_only`.
- `matched_refit_effect`: `restricted_canonical_309` → `exclusion_refit_309`; `primary_training_and_data_rule_sensitivity_on_identical_population`.

The first contrast changes only the evaluated sample set and is fit-free. The second is the primary training/data-rule sensitivity because both arms contain identical sample, fold, and target identities.

## Candidate schedule

| Outer fold | Historical candidate | Exclusion/refit candidate | Changed |
| ---: | ---: | ---: | --- |
| 1 | 2 | 6 | true |
| 2 | 7 | 7 | false |
| 3 | 4 | 4 | false |
| 4 | 4 | 4 | false |
| 5 | 1 | 7 | true |

## Interpretation boundary

This limited repetition-1/raw-XGBoost analysis is a descriptive data-quality sensitivity. It is not an equivalence test, target-validity proof, robustness certification, calibration analysis, or cross-dataset performance claim. Row identities, OOF predictions, and candidate-level inner-fold scores are intentionally excluded.
