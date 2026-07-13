# Persistent Decision Log

The fixed decisions are recorded in `../finalization_v2/00_scope_and_fixed_decisions.md` and `../finalization_v2/04_decision_log.md`.

Accepted on 2026-07-13:

- D1: Logistic Regression + Random Forest + LightGBM.
- D2: restrained nested tuning with 10 outer × 5 inner folds and outer-test isolation.
- D3: user-provided pinned files for all datasets; sanitized publication export; only approved-manifest URL acquisition when local data is absent.
- D4: application pending; institution/unit/reference/date remain unknown because supplied strings were placeholders.
- D5: small artifacts in Git; full package prepared for Release/Zenodo; pointer-only latest; no publication yet.
- Dedicated branch/commit permission: `finalization/leakage-aware-v2`, tested checkpoints only.
- Model selection: macro-F1 is the sole primary inner-CV metric; QWK is secondary and is used only inside the inclusive absolute `0.001` macro-F1 tie pool, followed by lowest candidate index.
- Stop gate: trigger only when baseline-minus-XGBoost macro-F1 is positive and the paired OOF bootstrap 95% CI lower bound is greater than zero; then request user direction before choosing the XAI reference model.
- Bootstrap: 5,000 paired sample-level OOF draws, stratified by outer fold and true class, percentile 95% interval with linear quantiles.

Observed on 2026-07-13: verified trial `benchmark-10x5-20260713-6a80074` did not trigger the baseline superiority gate. XGBoost therefore remains the predeclared XAI reference without a new user decision. Secondary QWK differences remain reportable but do not override the accepted macro-F1 gate.

Still prohibited without explicit approval: force-push, merge, release publication, Git history alteration, unapproved mirror use, or new scientific-protocol choices.
