# Ten-by-five nested OOF four-model benchmark

Paired sample-level OOF bootstrap, 5,000 draws. The superiority gate is macro-F1-specific and did not replace the predeclared XGBoost XAI reference.

| model | analytical_role | macro_f1_display | macro_f1_interval_display | qwk_display | qwk_interval_display | balanced_accuracy_display | ordinal_mae_display | severe_error_rate_display | baseline_minus_xgboost_macro_f1_display | baseline_minus_xgboost_macro_f1_interval_display |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | XGBoost reference | 0.6210 | [0.5973, 0.6447] | 0.5676 | [0.5352, 0.5997] | 0.6360 | 0.2433 | 0.0025 | N/A | N/A |
| logistic_regression | predeclared baseline | 0.5062 | [0.4803, 0.5318] | 0.3710 | [0.3260, 0.4149] | 0.5240 | 0.3550 | 0.0125 | -0.1148 | [-0.1476, -0.0832] |
| random_forest | predeclared baseline | 0.5923 | [0.5796, 0.6048] | 0.6317 | [0.6037, 0.6583] | 0.6253 | 0.1583 | 0.0017 | -0.0287 | [-0.0499, -0.0080] |
| lightgbm | predeclared baseline | 0.6055 | [0.5833, 0.6292] | 0.5883 | [0.5559, 0.6199] | 0.6218 | 0.1983 | 0.0025 | -0.0155 | [-0.0381, 0.0064] |
