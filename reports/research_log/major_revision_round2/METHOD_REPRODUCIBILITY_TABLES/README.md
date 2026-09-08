# Round 2 Method Reproducibility Tables

These additive tables are generated from hash-pinned, previously approved scientific contracts. They do not modify historical Phase 1–5 evidence or the manuscript.

- `TABLE_S1_FEATURE_AVAILABILITY.csv/.md`: all 28 declared fields and P0–P5 inclusion decisions.
- `TABLE_S2_PREPROCESSING_HYPERPARAMETER_SEARCH.csv/.md`: common preprocessing, six complete trained-model registries, and both Round 2 selection regimes.
- `TABLE_S3_CV_SEED_CONTRACT.csv/.md`: exact fold, seed, OOF, calibration, score-source, and reuse/refit boundaries.
- `METHODS_NOTES.md`: self-contained ordinal-model and probability-evaluation prose for later manuscript integration after claim approval.
- `SOURCE_HASHES.json`: exact source identity.

## Manuscript-facing policy labels

- P0: Information-Rich Diagnostic
- P1: Leakage-Risk-Controlled
- P2: Direct-Sensitive-Attribute-Excluded
- P3: Primary Leakage-Aware
- P4: Prospective-Plausibility
- P5: Strict Proxy-Reduced Prospective-Plausibility

P4 and P5 are timestamp-unverified sensitivity policies, not prospective validation. No policy is labelled leakage-free.

## Source hashes

- `configs/feature_availability_v3.json` — `6dd5fdde534e379cceacfaa01e865d1551310fb632b691f5b937ef39394e93cf`
- `configs/model_grid.yaml` — `d8fb0584d7106f8941ca8e1b0e0a9c58e13f39519c6c75b383ff333d92d41617`
- `configs/ordinal_benchmark_v3.json` — `39bcc62580515888783120a00ed807c0ede0f4c46f587f50897aced4c7999b02`
- `configs/policy_retuning_v3.json` — `d10c6f6c5e3a61e3895220f4d43a8d682e4d98c83b165f6694b20570ae950d22`
- `configs/repeated_nested_cv_v3.json` — `5681e521cfbaff5963494212fcc047116056fd7d66393346df0463aef7553af9`
- `configs/hrdataset_sensitivity_v3.json` — `9a94052e8fc96ff0894c89cfe99eca898e1e0da5ed697c7fabac99e4fb22799b`
- `configs/calibration_diagnostics_v3.json` — `258a81b0a3a4038218ca9f185252ed9fcc12c8f7056314e9760f5285ffec831b`
- `configs/selection_objective_sensitivity_v4.json` — `ed46b263644c1d1fc49969c3d7a2936e2452eab02240e66a4b53016d9eb3f719`
- `configs/hr_target_alias_sensitivity_v4.json` — `87f447ac5ad9e6b33254c8f63d34cef0c0b4ebbd183610b6db009372ac5d6fb7`
