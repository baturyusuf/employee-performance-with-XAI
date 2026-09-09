# Supplementary Table S6. P3 subgroup gaps, eligibility, and exploratory simultaneous intervals

Support threshold: n >= 30. Intervals are exploratory simultaneous bootstrap intervals from 5,000 prespecified resamples and condition on the observed sample, fitted models, folds, and eligibility rules. They are not confirmatory fairness tests, discrimination evidence, or population-level confidence guarantees.

| Attribute | Metric | Gap | Eligible / declared groups | Exploratory simultaneous interval |
| --- | --- | ---: | ---: | --- |
| Age | Macro-F1 | 0.0301 | 4 / 5 | [0.0000, 0.1657] |
| Age | QWK | 0.0653 | 4 / 5 | [0.0000, 0.3335] |
| Age | MAE | 0.0770 | 4 / 5 | [0.0000, 0.2411] |
| Gender | Macro-F1 | 0.0363 | 2 / 2 | [0.0000, 0.1515] |
| Gender | QWK | 0.0527 | 2 / 2 | [0.0000, 0.2420] |
| Gender | MAE | 0.0508 | 2 / 2 | [0.0000, 0.1719] |
| Marital Status | Macro-F1 | 0.0101 | 3 / 3 | [0.0000, 0.1135] |
| Marital Status | QWK | 0.0381 | 3 / 3 | [0.0000, 0.2431] |
| Marital Status | MAE | 0.0291 | 3 / 3 | [0.0000, 0.1485] |
| Business Travel | Macro-F1 | 0.0528 | 3 / 3 | [0.0000, 0.1987] |
| Business Travel | QWK | 0.0711 | 3 / 3 | [0.0000, 0.3472] |
| Business Travel | MAE | 0.0613 | 3 / 3 | [0.0000, 0.2195] |
| Department | Macro-F1 | 0.2179 | 5 / 6 | [0.0000, 0.5281] |
| Department | QWK | 0.4388 | 3 / 6 | [0.1275, 0.7501] |
| Department | MAE | 0.1193 | 5 / 6 | [0.0000, 0.3687] |
| Education | Macro-F1 | 0.0413 | 5 / 6 | [0.0000, 0.3766] |
| Education | QWK | 0.1190 | 4 / 6 | [0.0000, 0.5064] |
| Education | MAE | 0.0891 | 5 / 6 | [0.0000, 0.3211] |

The nonzero lower endpoint for the Department QWK exploratory interval is not treated as confirmatory statistical proof. Eligibility is support-dependent, only 3 of 6 Department groups enter that QWK gap, and the interval family does not incorporate full model-training uncertainty.