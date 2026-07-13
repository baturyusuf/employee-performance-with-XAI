# Next Actions

1. Obtain the calibration-training decision: recommended five-inner-fold OOF cross-fitted sigmoid refits, equivalent cross-fit predictions persisted by the benchmark stage, or a single 20% calibration holdout with a different reduced-training base model.
2. After approval, refactor calibration to shared folds, exact benchmark raw OOF, predeclared sigmoid, outer-test isolation, paired 5,000-resample OOF uncertainty, atomic outputs and no core isotonic selection.
3. Update subgroup/proxy and HRDataset external consumers to the same run/fold/task contracts; keep IBM/turnover supplementary.
4. Regenerate benchmark, policy, calibration and SHAP together in the complete core build under one clean commit/config/run identity.
5. Build the revised core-only figures/tables and claim matrix from that same run; finish lock/CI/EOL portability and D5 packaging before release claims.

No model-reference user decision is required because the predeclared macro-F1 gate did not trigger. Do not edit the manuscript, call APIs, publish, push or merge.
