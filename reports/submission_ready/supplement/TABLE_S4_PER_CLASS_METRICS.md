# Supplementary Table S4. Per-class metrics under the canonical macro-F1-selected P3 benchmark

These results use the declared unmodified hard-decision rules. No class-specific threshold optimization was performed. Alternative decision thresholds were not evaluated.

| System | Rating | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cumulative-threshold XGBoost | 2 | 0.797357 | 0.932990 | 0.859857 | 194 |
| Cumulative-threshold XGBoost | 3 | 0.870015 | 0.673913 | 0.759510 | 874 |
| Cumulative-threshold XGBoost | 4 | 0.185811 | 0.416667 | 0.257009 | 132 |
| Nominal XGBoost | 2 | 0.821101 | 0.922680 | 0.868932 | 194 |
| Nominal XGBoost | 3 | 0.854217 | 0.811213 | 0.832160 | 874 |
| Nominal XGBoost | 4 | 0.151316 | 0.174242 | 0.161972 | 132 |
| LightGBM | 2 | 0.823256 | 0.912371 | 0.865526 | 194 |
| LightGBM | 3 | 0.849673 | 0.892449 | 0.870536 | 874 |
| LightGBM | 4 | 0.119403 | 0.060606 | 0.080402 | 132 |
| Random Forest | 2 | 0.836449 | 0.922680 | 0.877451 | 194 |
| Random Forest | 3 | 0.851738 | 0.953089 | 0.899568 | 874 |
| Random Forest | 4 | 0.000000 | 0.000000 | 0.000000 | 132 |

**Source boundary.** Values are copied from the independently validated `PER_CLASS_EXTREME_CLASS_REPORT.md` / `selection_objective_sensitivity/per_class_metrics.csv`. They are descriptive OOF results and do not establish that alternative threshold policies would produce identical class-4 behavior.