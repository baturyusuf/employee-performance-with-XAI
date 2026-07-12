# Next Actions

1. Receive and record the user's metric choice: A (`macro_f1`, recommended), B (`quadratic_weighted_kappa`), or C (macro-F1 inner selection plus multiplicity-controlled macro-F1/QWK gate).
2. Replace the fail-closed null fields in `configs/model_grid.yaml` and the pending metric status in `configs/manuscript_final.yaml`; rerun focused/full contract gates.
3. Create a tested Unit 2B checkpoint commit on `finalization/leakage-aware-v2` if not already checkpointed.
4. Run the real INX four-model restrained nested benchmark with the immutable 10×3 folds and 5,000 paired OOF bootstrap resamples.
5. If the baseline gate triggers, stop before policy/calibration/SHAP and request the XAI-reference decision. Otherwise persist the benchmark result as a noncanonical trial pending the complete scoped rebuild.
6. Refactor policy ablation, predeclared sigmoid calibration, OOF SHAP and subgroup/proxy consumers to use the shared-fold contract and exact selected XGBoost outer-fold models; remove the conflicting legacy fixed-parameter source.

Do not cite or reuse the in-memory predecision fold hash. Do not run/reuse the old `reports/manuscript_final/latest` package. Do not edit the manuscript, call APIs, publish, push or merge.
