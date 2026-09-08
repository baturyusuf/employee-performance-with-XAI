# Training-Only Empirical-Prior Probability Baseline

Status: **independently validated aggregate evidence**

The baseline estimates the three-class prevalence separately from each outer-training partition and assigns that fixed probability vector to every untouched row in the corresponding outer-test fold. It performs no model fit, does not enter hyperparameter selection, and never uses an outer-test label to construct its probabilities.

## Probability metrics

| Metric | Empirical-prior value |
| --- | ---: |
| Log loss | 0.768298 |
| Multiclass Brier | 0.431305 |
| Normalized RPS | 0.116719 |
| Top-label confidence ECE | 0.000000 |

The exact metrics are in [`empirical_prior_metrics.csv`](selection_objective_sensitivity/empirical_prior_metrics.csv); the ten training-only prevalence vectors and class counts are in [`empirical_prior_fold_parameters.csv`](selection_objective_sensitivity/empirical_prior_fold_parameters.csv). The independent validator reconstructed every vector from the canonical target and persisted outer folds and matched all 1,200 local OOF baseline rows before those row-level records were excluded from publication.

## Comparator boundary

The empirical-prior values provide a meaningful non-informative probability comparator. The existing majority, stratified, and ordinal-median systems remain hard-label comparators for classification and ordinal-error metrics; their one-hot probability scores should not be used to establish probability-quality superiority.

The zero top-label ECE does not mean that the prior is a perfectly calibrated individual predictor. Because the baseline emits nearly constant fold-level probabilities and always selects the majority class, a confidence-only ECE can equal zero when aggregate confidence matches aggregate accuracy. Log loss, Brier, and RPS remain necessary complementary views. The baseline is not a trained model, individualized forecast, or deployment benchmark.
