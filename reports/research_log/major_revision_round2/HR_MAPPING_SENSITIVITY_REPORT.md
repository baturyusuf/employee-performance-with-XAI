# HR Target-Construction and Alias Sensitivity

Status: **independently validated aggregate evidence**

This report keeps three distinct questions separate: target-formulation sensitivity, cross-validation-design sensitivity, and the limited target-alias exclusion/refit sensitivity. None is an equivalence test, and none validates the HRDataset_v14 target as construct-equivalent to the INX outcome.

## Target-formulation sensitivity

The retained three-class mapping combines `PIP` and `Needs Improvement`, with support 31/243/37 across labels 2/3/4. The supplementary raw-order four-class formulation keeps those categories separate, with support 13/18/243/37 across labels 1/2/3/4.

| Target formulation | System | Macro-F1 | QWK | Ordinal MAE | Log loss | Brier | ECE | RPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained three-class | XGBoost raw | 0.6531 | 0.5339 | 0.1916 | 0.5593 | 0.3033 | 0.0721 | 0.0759 |
| Retained three-class | XGBoost sigmoid | 0.6274 | 0.6044 | 0.1280 | 0.4216 | 0.2324 | 0.0522 | 0.0589 |
| Raw-order four-class | XGBoost raw | 0.5847 | 0.6288 | 0.2360 | 0.6646 | 0.3618 | 0.0995 | 0.0604 |
| Raw-order four-class | XGBoost sigmoid | 0.5481 | 0.6621 | 0.1640 | 0.5201 | 0.2771 | 0.0386 | 0.0488 |

These values describe different target estimands. They are therefore shown side by side but are not subtracted or labelled as improvements. Within each formulation, sigmoid calibration had metric-specific effects: it lowered macro-F1 while improving QWK, ordinal MAE, log loss, Brier, confidence ECE, and RPS.

Source: the validated Phase 3A [`variability_summary.csv`](../major_revision_v3/phase3a_hrdataset_sensitivity/variability_summary.csv) and [`target_mapping_support.csv`](../major_revision_v3/phase3a_hrdataset_sensitivity/target_mapping_support.csv).

## Cross-validation-design sensitivity

For the retained three-class formulation, 11 of 14 canonical-v2 10×5 point estimates fell inside the ranges observed across five prespecified 5×5 nested-CV repetitions. Three estimates fell outside: raw XGBoost macro-F1 and sigmoid XGBoost Brier and RPS.

This is a descriptive comparison of point estimates with observed repetition ranges, not an equivalence test or confidence-interval analysis. The 11/14 result does not authorize an unqualified claim of robustness to CV design.

Source: the validated Phase 3A [`cv_design_sensitivity.csv`](../major_revision_v3/phase3a_hrdataset_sensitivity/cv_design_sensitivity.csv).

## Target-alias disagreement sensitivity

The Phase 3B audit identified two disagreements between the text target and its numeric alias. The canonical text-target rule and historical 311-row result remain unchanged. The limited Round 2 analysis uses only Phase 3A repetition 1, raw XGBoost, the same seven-feature policy, and the original fold identities.

Source run: `round2_hr_alias_v4_20260908T141019Z_ac50c6e`

Generation commit: `ac50c6e81722053bfc32b1186954e203334732d8`

Scientific-input SHA-256: `f347e0bd66cedcef70f578949d1b25abf923c2a60b8585c0154c023acea500e5`

Compact payload-set SHA-256: `44e7e4cea084103ba9f67f74492c1b49bfc742177a86525f3215242dfc6e7a47`

| Arm | N | Macro-F1 | Balanced accuracy | QWK | Ordinal MAE | RPS | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical canonical | 311 | 0.657026 | 0.661772 | 0.549509 | 0.180064 | 0.072499 | 0.535011 | 0.290338 | 0.075406 |
| Canonical predictions restricted fit-free | 309 | 0.662383 | 0.662666 | 0.546473 | 0.177994 | 0.071863 | 0.531707 | 0.287557 | 0.073421 |
| Exclusion and refit | 309 | 0.666593 | 0.660564 | 0.546134 | 0.177994 | 0.071886 | 0.542510 | 0.288226 | 0.062224 |

### Primary matched-population comparison

The primary training/data-rule comparison is **canonical predictions restricted to 309 rows versus exclusion/refit predictions on those identical 309 rows**. Exclusion/refit changed macro-F1 by +0.004210, balanced accuracy by -0.002102, QWK by -0.000339, ordinal MAE by +0.000000, RPS by +0.000024, log loss by +0.010803, Brier by +0.000668, and confidence ECE by -0.011197. Candidate selection changed in 2 of 5 outer folds.

The mixed directions are small and metric-specific. They do not show a material deterioration or improvement across the metric set, and they do not validate either target representation. The refit asks only how excluding the audited disagreements from training changes repetition-1 predictions on the retained population.

### Fit-free sample-removal comparison

The 311→309 contrast is reported separately and measures only removal of two evaluation rows from the existing canonical OOF predictions; it performs no refit. Macro-F1 changed by +0.005357, QWK by -0.003035, ordinal MAE by -0.002071, and RPS by -0.000636. These changes must not be attributed to retraining.

The population receipt confirms that the two primary 309-row arms have identical sample, fold, and target identities; the validator reproduced the 311-row OOF source exactly and replayed the 309-row restriction fit-free. Employee identities, row-level OOF predictions, and candidate-level scores are not published.

Authoritative Round 2 evidence: [`aggregate_metrics.csv`](hr_target_alias_sensitivity/aggregate_metrics.csv), [`comparison_deltas.csv`](hr_target_alias_sensitivity/comparison_deltas.csv), [`per_class_metrics.csv`](hr_target_alias_sensitivity/per_class_metrics.csv), and [`population_receipt.json`](hr_target_alias_sensitivity/population_receipt.json).

## Claim boundaries

- The four-class and retained three-class results concern different estimands; their scores are not direct treatment-style effects or improvements.
- The CV-design comparison is descriptive and does not establish statistical equivalence.
- The target-alias analysis is one prespecified repetition and one uncalibrated model system; it does not replace the five-repetition Phase 3A results.
- No locked-model transport, target equivalence, causal inference, fairness certification, autonomous HR-decision, or deployment-readiness claim is supported.
