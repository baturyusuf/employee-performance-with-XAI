# Per-Class Extreme-Class Report

Status: **independently validated aggregate evidence**

The table below reports the historical macro-F1-selection regime on the exact 1,200-case P3 OOF population. It makes the rating-4 behavior visible alongside aggregate ordinal metrics.

| Model | Rating | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cumulative-threshold XGBoost | 2 | 0.797357 | 0.932990 | 0.859857 | 194 |
| Cumulative-threshold XGBoost | 3 | 0.870015 | 0.673913 | 0.759510 | 874 |
| Cumulative-threshold XGBoost | 4 | 0.185811 | 0.416667 | 0.257009 | 132 |
| LightGBM | 2 | 0.823256 | 0.912371 | 0.865526 | 194 |
| LightGBM | 3 | 0.849673 | 0.892449 | 0.870536 | 874 |
| LightGBM | 4 | 0.119403 | 0.060606 | 0.080402 | 132 |
| Random Forest | 2 | 0.836449 | 0.922680 | 0.877451 | 194 |
| Random Forest | 3 | 0.851738 | 0.953089 | 0.899568 | 874 |
| Random Forest | 4 | 0.000000 | 0.000000 | 0.000000 | 132 |
| Nominal XGBoost | 2 | 0.821101 | 0.922680 | 0.868932 | 194 |
| Nominal XGBoost | 3 | 0.854217 | 0.811213 | 0.832160 | 874 |
| Nominal XGBoost | 4 | 0.151316 | 0.174242 | 0.161972 | 132 |

Exact rows and the complete ordered confusion matrices are in [`per_class_metrics.csv`](selection_objective_sensitivity/per_class_metrics.csv) and [`confusion_matrix.csv`](selection_objective_sensitivity/confusion_matrix.csv).

Random Forest had the historical benchmark's highest QWK (0.631678) and lowest ordinal MAE (0.158333) while assigning no OOF case to rating 4, producing rating-4 recall and F1 of zero. This is a metric-behavior result: distance-weighted aggregate performance can conceal complete failure on an extreme class. It does not establish why the failure occurred, and no causal explanation is claimed.

The QWK-selection sensitivity reinforces the reporting need rather than resolving it: cumulative-threshold XGBoost's rating-4 recall fell to zero, nominal XGBoost's was 0.007576, and Random Forest's remained zero even as QWK/MAE rankings improved. Full QWK-regime rows remain in the same source tables.
