# Phase 3A HRDataset_v14 Sensitivity — Compact Evidence

Source run: `phase3a_v3_20260907T141213Z_f6e6a0a`

This package reports an independent, cross-dataset protocol replication on HRDataset_v14. It is not external validation of a locked INX model: the feature space, target semantics, fitted parameters, and sample population differ, and no INX model was transported.

## Prespecified design

Both target formulations use the same seven-feature conservative-primary policy, five fixed repetitions of 5-fold outer by 5-fold inner nested cross-validation, eight-candidate XGBoost selection by inner macro-F1 with QWK tie-breaking, outer-training-only cross-fitted sigmoid calibration, and three label-only naive baselines. Each of the 311 records appears exactly once in outer-test predictions for every system and repetition.

The retained three-class mapping combines PIP and Needs Improvement (31/243/37 across labels 2/3/4). The supplementary raw-order four-class mapping keeps PIP and Needs Improvement separate (13/18/243/37 across labels 1/2/3/4). These are different estimands; metric differences between them are deliberately not computed and target equivalence is not claimed.

## Five-repetition descriptive means

| Target formulation | System | Macro-F1 | QWK | Ordinal MAE | Log loss | Brier | Confidence ECE | RPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained three-class | XGBoost raw | 0.6531 | 0.5339 | 0.1916 | 0.5593 | 0.3033 | 0.0721 | 0.0759 |
| Retained three-class | XGBoost sigmoid | 0.6274 | 0.6044 | 0.1280 | 0.4216 | 0.2324 | 0.0522 | 0.0589 |
| Raw-order four-class | XGBoost raw | 0.5847 | 0.6288 | 0.2360 | 0.6646 | 0.3618 | 0.0995 | 0.0604 |
| Raw-order four-class | XGBoost sigmoid | 0.5481 | 0.6621 | 0.1640 | 0.5201 | 0.2771 | 0.0386 | 0.0488 |

Calibration effects are metric-specific. In both formulations sigmoid improves QWK, ordinal MAE, log loss, Brier, confidence ECE, and RPS but lowers macro-F1. The method was predeclared, not selected from outer-test results.

## CV-design sensitivity and baselines

For the retained mapping, 11 of 14 canonical-v2 10×5 point estimates lie inside the observed five-repetition 5×5 ranges. Raw macro-F1 and sigmoid Brier/RPS fall outside those ranges. These are descriptive design comparisons, not equivalence tests or confidence intervals; the evidence does not authorize an unqualified claim of robustness to CV design.

The baseline table retains every repetition-level comparison against majority, stratified, and ordinal-median predictors. Per-class precision/recall/F1/support and full zero-retaining confusion grids are included because aggregate scores can hide the rare PIP and Needs Improvement classes.

## Interpretation boundaries

- Repetition ranges and sample standard deviations describe training/split variability under five prespecified seeds; they are not confidence intervals.
- The dataset is small and class-imbalanced, especially under the four-class mapping. Results are conditional on this public table and its unresolved source-authenticity/licence review.
- No target equivalence, locked-model transport, external-validation, causal, fairness-certification, autonomous HR-decision, or deployment-readiness claim is allowed.
- The row-level OOF predictions, calibration-training rows, fold assignments, candidate scores, selected-fold records, calibrator parameters, raw data, and fitted objects remain local and are excluded from Git.

## Files

- `target_mapping_support.csv`: observed support and rationale for both mappings.
- `repetition_metrics.csv` and `variability_summary.csv`: complete repetition-level metrics and five-repetition summaries.
- `per_class_metrics.csv` and `confusion_matrices.csv`: class-specific results and complete confusion grids.
- `baseline_comparisons.csv`: matched-repetition raw-XGBoost versus naive-baseline comparisons.
- `cv_design_sensitivity.csv`: retained-mapping 10×5 versus repeated-5×5 descriptive comparison.
- `canonical_v2_per_class_metrics.csv` and `canonical_v2_confusion_matrix.csv`: independently recomputed canonical-v2 class evidence.
- `selected_candidate_frequency.csv`: aggregate selection counts, including zero-count candidates.
- `protocol_comparison.csv`: explicit INX/HRDataset implementation and semantic boundaries.
- `provenance_receipt.json` and `manifest.json`: independent validation, exclusions, lineage, and byte hashes.
