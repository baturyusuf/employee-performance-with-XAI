# Extended Multiclass and Ordinal Calibration Diagnostics — v3

Date: 2026-09-04

Status: design, implementation, clean exact-commit execution, and independent full-run validation complete; compact publication pending

## Preserved source calibration

Phase 2B does not select or fit a new calibration method. It consumes the exact 1,200-row OOF raw and predeclared sigmoid probability sets from canonical v2. The raw comparator is the unchanged XGBoost OOF probability matrix. For each outer fold, the existing sigmoid method fitted three one-vs-rest Platt models on five-fold cross-fitted probabilities from that outer-training partition only, transformed each class probability on its logit scale, and divided the three positive sigmoid outputs by their row sum. The untouched outer-test fold was evaluation-only. Sigmoid was fixed before outer-test inspection; raw-versus-sigmoid results cannot be used to retroactively choose a method.

The extension performs zero new performance-model fits and zero new probability-calibrator fits. It reuses the exact persisted OOF rows and binds their bytes, metadata, validation receipt, protocol, existing bins, method summaries, paired differences, and canonical receipt by SHA-256.

## Reliability definitions

All ECE variants use ten fixed equal-width bins over [0,1]. The first bin includes both endpoints; later bins are left-open and right-closed. Empty bins remain explicit with zero support and missing mean probability/frequency. This convention matches the existing canonical classwise bin evidence.

- Top-label ECE weights the absolute difference between mean maximum probability and argmax accuracy in each nonempty bin by its fraction of all 1,200 samples.
- Classwise ECE treats each ordered class as a one-vs-rest binary event and applies the same support-weighted absolute-gap definition to that class probability.
- Cumulative ECE uses the two nontrivial ordered events `Y<=2` and `Y<=3`, with probabilities `p(2)` and `p(2)+p(3)`.

Classwise and cumulative reliability diagrams display only nonempty-bin points while the source table retains every declared bin. ECE is binning-dependent; sparse rare-class bins are not hidden.

## Ordinal probability diagnostics

The normalized ranked probability score is the mean squared cumulative-distribution error over the two nontrivial ordered thresholds and all samples. It lies in [0,1] and lower is better. Threshold-specific binary Brier scores are also reported; their unweighted mean equals the normalized RPS by construction and is checked as an invariant.

## Calibration intercept and slope

For each one-vs-rest class event and each cumulative event, a descriptive pooled-OOF logistic calibration regression jointly estimates an intercept and slope using the clipped logit of the supplied probability. Deterministic Newton–Raphson starts at intercept 0 and slope 1, uses no statistical penalty, and stops when the maximum absolute step is below `1e-10`. A `1e-12` diagonal term is used only to solve the Hessian system numerically, not as an estimation penalty.

Ideal calibration corresponds descriptively to intercept 0 and slope 1. These coefficients are fitted and evaluated on the same exactly-once OOF prediction set, so they are diagnostics rather than future-calibration validation; no confidence interval or prospective guarantee is attached.

## Comparison and claim boundaries

Raw and sigmoid are compared for log loss, multiclass Brier, top-label ECE, macro classwise ECE, mean cumulative ECE, and RPS. Differences are reported as sigmoid minus raw together with a lower-is-better direction-aligned change. Existing paired bootstrap evidence may be retained for the three legacy metrics, but new diagnostic differences are descriptive and no method is selected from them.

The observed canonical evidence already shows why the boundary matters: sigmoid improves log loss and multiclass Brier but worsens the point estimate of top-label ECE. The extension must report all favorable, adverse, and sparse-bin findings. It cannot support “calibration improved everything,” objective employee-outcome probabilities, future calibration, validated HR thresholds, or deployment readiness.

## Implementation validation

The fail-closed contract validator binds all nine canonical calibration sources and their identities before calculation. The runner then recomputes the three legacy probability metrics, requires exact replay of the existing 60-row classwise reliability grid, produces 120 explicit top-label/classwise/cumulative bin rows, and checks that normalized RPS equals the mean of the two cumulative binary Brier scores. Atomic publication requires a clean, unchanged Git identity and an offline runtime receipt. Focused tests and the complete 898-test repository regression passed; no model or probability calibrator was refitted.

Clean-commit run `phase2b_v3_20260904T120838Z_21d1aec` produced the exact 11-file local package from pushed commit `21d1aecb6e61511e95aee498ab81c54fe6e5a6ab`. An independent validator rebound the generation Git blobs and all canonical sources, verified output hashes and offline runtime, independently recomputed every metric/regression/bin/contrast row, and checked both PNG/SVG figure contracts. It passed with 120 reliability bins, including 24 explicit empty bins. Compact aggregate publication remains pending; the complete run stays local and ignored.

The validated raw/sigmoid values are 0.5515/0.4556 for log loss, 0.3426/0.2634 for multiclass Brier, 0.0375/0.0419 for top-label ECE, 0.1070/0.0249 for macro classwise ECE, 0.0779/0.0184 for mean cumulative ECE, and 0.0860/0.0669 for normalized RPS. Thus the predeclared sigmoid method improves five point estimates but worsens top-label ECE; only the three legacy contrasts retain their existing paired bootstrap intervals.
