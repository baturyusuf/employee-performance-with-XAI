# Canonical Calibration Method Rationale

Run ID: `manuscript_final_20260712T175019Z_c664ef152ff3`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`

Calibration uses outer stratified 10-fold evaluation. Within each outer training fold, the model is fitted on an inner training subset and raw, sigmoid, and isotonic probability outputs are evaluated only on the untouched outer test fold. Sigmoid/isotonic calibrators see only the inner calibration subset.

Selected method: `sigmoid` by the predeclared lowest aggregate rank across log loss, multiclass Brier score, and ECE (rank sum 4.000).

## Probability-Use Warning

Probabilities are approximate model confidence estimates, not objective employee-performance probabilities. Use probability bands with calibration warnings and human review; do not use exact probabilities as autonomous HR decision thresholds.

Isotonic calibration is flexible and may be unstable for the minority class because each inner calibration split is small. Method selection is descriptive for this declared protocol and dataset, not a guarantee of future calibration.
