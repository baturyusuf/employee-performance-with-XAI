# Reproducibility Notes for the Ordinal Models

## Common preprocessing

Preprocessing is fitted anew inside the current inner-development or outer-training partition. Numeric variables receive median imputation followed by standard scaling. Categorical variables receive most-frequent imputation and dense one-hot encoding with unknown categories ignored. Outer-test rows are used only for evaluation and never to fit preprocessing, select a candidate, select a calibration method, or choose a seed or information policy.

## Proportional-odds model

The proportional-odds logistic model is a cumulative-link model with shared feature coefficients across ordered cut points and fitted thresholds constrained to increase. Its shared-slope proportional-odds assumption is a model restriction, not an empirical statement that effects are identical in the data. Candidate selection varies regularization strength and optional class balancing under the same nested-CV rule as the other trained models.

## Cumulative-threshold XGBoost

For ordered labels 2/3/4, cumulative-threshold XGBoost independently fits binary XGBoost tasks for each observed training threshold, estimating `P(Y > 2)` and `P(Y > 3)`. Independently fitted tasks can cross. For each row, a nonincreasing pool-adjacent-violators (PAVA) projection corrects the cumulative probabilities before class probabilities are reconstructed by differencing: `P(Y=2)=1-P(Y>2)`, `P(Y=3)=P(Y>2)-P(Y>3)`, and `P(Y=4)=P(Y>3)`. The resulting vector is clipped only for numerical safety and normalized to the probability simplex.

This construction supplies ordered probabilities but does not guarantee superior extreme-class recall. Candidate selection, class imbalance, cumulative decomposition, PAVA projection, and ordinal-distance metrics are plausible methodological explanations for observed trade-offs unless directly isolated by a prespecified experiment.

## Selection and probability evaluation

The canonical regime selects the highest inner macro-F1 candidate, admits candidates within an inclusive 0.001 primary-score tolerance, then uses QWK and finally the lowest candidate index as deterministic tie-breaks. Round 2 reverses macro-F1 and QWK only in the designated sensitivity regime; it does not add an MAE- or RPS-selected regime. Probability reporting includes log loss, multiclass Brier score, confidence ECE, and ranked probability score. The RPS is the mean squared cumulative-probability error across the `K-1` ordered thresholds.

All results remain conditional on the stated data, feature policy, folds, seeds, candidate registries, and calibration contract. No table establishes prospective validity, leakage freedom, causal feature effects, fairness, or deployment readiness.
