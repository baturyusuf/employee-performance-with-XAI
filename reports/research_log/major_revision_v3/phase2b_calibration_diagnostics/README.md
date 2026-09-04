# Phase 2B Extended Calibration Diagnostics — Compact Evidence

Source run: `phase2b_v3_20260904T120838Z_21d1aec`

This package contains aggregate OOF calibration diagnostics and reliability figures only. Employee-level OOF probabilities, fold assignments, fitted performance models, probability-calibrator objects/parameters, and raw data are deliberately excluded.

## Preserved method

The primary sigmoid method was predeclared in canonical v2. Within each outer fold, three one-vs-rest Platt calibrators were trained only on five-fold cross-fitted outer-training probabilities; their positive outputs were divided by the row sum to restore the multiclass probability simplex. The untouched outer-test fold was evaluation-only. Phase 2B fits no performance model or probability calibrator and does not use these results to select a method.

## Aggregate results

All displayed metrics are lower-is-better. A positive direction-aligned value favors sigmoid.

| Metric | Raw | Sigmoid | Sigmoid − raw | Direction-aligned improvement |
| --- | ---: | ---: | ---: | ---: |
| Log loss | 0.5515 | 0.4556 | -0.0958 | +0.0958 |
| Multiclass Brier | 0.3426 | 0.2634 | -0.0792 | +0.0792 |
| Top-label ECE | 0.0375 | 0.0419 | +0.0044 | -0.0044 |
| Macro classwise ECE | 0.1070 | 0.0249 | -0.0820 | +0.0820 |
| Mean cumulative ECE | 0.0779 | 0.0184 | -0.0595 | +0.0595 |
| Normalized RPS | 0.0860 | 0.0669 | -0.0190 | +0.0190 |

Sigmoid improves the log-loss, multiclass-Brier, macro classwise-ECE, mean cumulative-ECE, and normalized-RPS point estimates, but its top-label ECE is worse. The legacy paired interval for the top-label-ECE difference spans zero. No interval was estimated for the new diagnostic differences, so this package does not support an ‘all metrics improved’ claim.

## Reliability and ordinal definitions

- Top-label ECE bins maximum predicted probability against argmax correctness; classwise ECE treats each class as one-vs-rest; cumulative ECE uses the ordered events Y≤2 and Y≤3.
- Every ECE uses ten fixed equal-width bins on [0,1]. The first bin is closed at both ends; later bins are left-open/right-closed. Empty bins remain explicit in `extended_reliability_bins.csv` and are omitted only from plotted lines.
- Normalized RPS is the mean squared cumulative-distribution error over the two nontrivial thresholds. It equals the mean of their binary Brier scores.
- Calibration intercept/slope values are unpenalized pooled exactly-once-OOF descriptive diagnostics fitted on the same prediction set. They are not confidence-bounded future-calibration validation.

## Interpretation boundaries

- ECE depends on the selected bins, and rare-class/high-probability bins can be empty or sparse. The diagrams must be read together with bin support.
- Raw versus sigmoid is a predeclared evaluation comparison, not test-set selection among calibration methods.
- The probabilities are model outputs for this cross-sectional research construct, not objective employee-outcome probabilities or validated decision thresholds.
- The evidence does not establish prospective calibration, fairness, causal effects, human usefulness, legal compliance, or deployment readiness for HR decisions.

## Files

- `calibration_metric_summary.csv` and `method_comparison.csv`: aggregate raw/sigmoid metrics and bounded contrasts.
- `classwise_calibration_metrics.csv` and `cumulative_calibration_metrics.csv`: per-event ECE, Brier, log loss, intercept, and slope.
- `extended_reliability_bins.csv`: all 120 top-label/classwise/cumulative bins, including empty bins.
- `classwise_reliability.*` and `cumulative_reliability.*`: 300-DPI PNG and editable SVG reliability diagrams.
- `diagnostic_receipt.json`, `provenance_receipt.json`, and `manifest.json`: method, lineage, exclusions, independent validation, and byte hashes.
