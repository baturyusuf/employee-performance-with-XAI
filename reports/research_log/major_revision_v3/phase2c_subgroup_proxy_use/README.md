# Phase 2C Subgroup and Proxy-Use Diagnostics — Compact Evidence

Source run: `phase2c_v3_20260907T070905Z_e314bb5`

This package contains support-aware aggregate OOF diagnostics only. The 1,200 employee-level paired prediction rows, 48,000 employee-level permutation rows, folds, raw data, and fitted models are deliberately excluded.

## Complete subgroup audit

The package retains all 2,025 declared group-metric rows for three exact canonical systems, six attributes, nine metrics, and support thresholds 20/30/50. Unsupported group or class-denominator cells remain explicit with missing estimates and a status; they are excluded from maximum-minus-minimum gaps rather than hidden.

All 486 declared gap cells are present. P3 has 162 pointwise and exploratory familywise simultaneous intervals based on 5,000 employee-level bootstrap repetitions stratified by outer fold and true class. Eligibility is fixed before resampling. The simultaneous family covers every estimable P3 attribute/threshold/metric cell, so this package does not select only the largest observed gap. These intervals condition on the fitted models/folds and are not confirmatory fairness inference.

## Exact P3 versus P3-minus-JobRole prediction changes

Overall mean total variation is 0.0906; the argmax prediction changes for 10.75% of cases; and the macro-F1 difference (proxy-reduced minus P3) is -0.0330. Department-specific rows retain probability, confidence-margin, ordinal-margin, prediction-change, and ordinal-shift summaries.

The comparator is exactly the canonical P3 model refitted without JobRole. It is not v3 P5, which has a broader timing/proxy exclusion contract. The paired differences therefore combine direct feature removal with the resulting refit and are not isolated causal effects.

## JobRole perturbation sensitivity

| Scheme | Mean total variation | Prediction-change rate | Mean macro-F1 change | Mean raw-margin drop |
| --- | ---: | ---: | ---: | ---: |
| Marginal within outer fold | 0.1024 | 0.1305 | -0.0432 | 0.1537 |
| Department-conditional within outer fold | 0.0529 | 0.0739 | -0.0034 | 0.0509 |

Each scheme uses 20 prespecified outcome-blind shuffles and the exact persisted P3 outer-fold models, with no refitting. Repetition variation is descriptive perturbation variability, not a confidence interval. Marginal shuffling can create out-of-distribution combinations; conditioning only on department is not a fully conditional permutation test.

## Department reconstructability is a different question

With JobRole in the reconstruction feature space, department reconstruction accuracy/balanced accuracy/macro-F1 are 0.9792/0.9806/0.9685. Without JobRole they are 0.2908/0.3028/0.2474.

Reconstructability shows that department information exists in a feature space. It does not prove that the performance model used department. The paired/refit and permutation tables address model-output dependence separately, and neither establishes causality or discrimination.

## Interpretation boundaries

- This is a same-dataset, exactly-once-OOF exploratory audit conditional on the observed sample, support rules, folds, models, policies, and perturbations.
- Small groups and rare true-class denominators are not silently pooled or suppressed. Age 60+ and other ineligible cells remain visible as unsupported.
- Maximum gaps are selected over groups and metrics; all cells and multiplicity-aware exploratory intervals must be considered together.
- No table certifies fairness, proves absence or presence of discrimination, identifies causal JobRole or department effects, validates legal compliance, or supports autonomous HR decisions or deployment readiness.

## Files

- `subgroup_metric_grid.csv`: complete supported/unsupported group metric grid.
- `subgroup_gap_sensitivity.csv`: all maximum-minus-minimum gaps at n=20/30/50.
- `primary_gap_bootstrap_intervals.csv`: P3 pointwise and simultaneous exploratory intervals.
- `proxy_prediction_change_by_department.csv`: paired aggregate output changes overall/by department.
- `jobrole_permutation_repetition.csv` and `jobrole_permutation_summary.csv`: 40 repetition-level and two scheme-level perturbation summaries.
- `department_reconstructability_metrics.csv` and `department_reconstructability_differences.csv`: preserved reconstruction evidence, explicitly separated from performance-model use.
- `diagnostic_receipt.json`, `provenance_receipt.json`, and `manifest.json`: method scope, independent validation, exclusions, lineage, and byte hashes.
