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

Engineering decision on 2026-07-13: canonical SHAP must load the exact prediction-producing XGBoost pipeline for each outer fold, replay its OOF evidence at `1e-12` probability tolerance, and contain no model/preprocessor fit or splitter path. The 45 fold pairs are dependent descriptive units, so no pairwise t confidence interval is reported. The historical `6a80074` trial remains immutable and reader-verifiable at its documented `1e-6` tolerance, but its pre-warning-cleanup config/probabilities and missing nested one-hot feature-name lineage make it intentionally ineligible as a canonical SHAP upstream. No user scientific decision was required.

Engineering/scientific consequence of accepted D2 on 2026-07-13: leakage-policy variants are a matched feature-access sensitivity analysis, not independently optimized leaderboard entries. The exact benchmark OOF rows are reused for the primary policy; every other declared policy receives the primary policy's selected XGBoost candidate from the same outer fold. All policy intervals/differences use the same 5,000 paired sample-level resamples as the benchmark. The legacy full-feature name is retained, but its role is an information-rich diagnostic comparator whose superiority is not guaranteed. No new user decision was required because D2 already says not to tune each leakage policy independently.

Pending material decision after Unit 2D: choose the calibration-training data protocol. The recommended option is five-inner-fold OOF cross-fitted sigmoid calibration refitted within the calibration stage, which preserves the exact full-outer-train benchmark model for outer-test evaluation and costs about 2–3 minutes. An equivalent option persists selected-candidate inner-OOF predictions from the benchmark stage, avoiding later refits but broadening the benchmark artifact contract. A single 20% holdout costs about 1.5–2 minutes but changes the base model to one trained on only 80% of outer training data. Outer-test-based selection and fitting a calibrator on in-sample predictions from the full outer-training model are prohibited.

Still prohibited without explicit approval: force-push, merge, release publication, Git history alteration, unapproved mirror use, or new scientific-protocol choices.
