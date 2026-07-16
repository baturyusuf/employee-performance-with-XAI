# Raw and cross-fitted sigmoid probability metrics

Calibration concerns probability reliability. Sigmoid-minus-raw paired differences do not imply automatic classification improvement or deployment readiness.

| method | log_loss_display | log_loss_interval_display | log_loss_sigmoid_minus_raw_display | brier_score_display | brier_score_interval_display | brier_score_sigmoid_minus_raw_display | ece_display | ece_interval_display | ece_sigmoid_minus_raw_display | fitting_scope | outer_test_used_for_fitting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.5515 | [0.5180, 0.5862] | N/A | 0.3426 | [0.3215, 0.3643] | N/A | 0.0375 | [0.0295, 0.0659] | N/A | not_applicable_raw_probabilities | False |
| sigmoid | 0.4556 | [0.4359, 0.4764] | -0.0958 | 0.2634 | [0.2503, 0.2769] | -0.0792 | 0.0419 | [0.0294, 0.0584] | 0.0044 | outer-training five-fold cross-fitted probabilities only | False |
