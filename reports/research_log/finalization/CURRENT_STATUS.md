# Current Status

- Current phase: completed/verified real four-model benchmark trial; recording the non-triggered gate and hardening probability/warning behavior before downstream policy, calibration and OOF SHAP integration.
- Last completed task: clean-commit offline trial `benchmark-10x5-20260713-6a80074` completed from commit `6a80074`; all 53 manifest outputs and 40 model hashes reverified; baseline superiority gate did not trigger.
- Work currently in progress: persist trial results/hashes/runtime; resolve XGBoost row-normalization and LightGBM feature-name warning hygiene before any canonical probability evidence is built.
- Files modified but not finalized: persistent finalization logs. The immutable untracked trial directory contains 54 files/91,820,515 bytes under `reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core`; it is a D5 external-package candidate, not a canonical Git result.
- Latest tests: pre-run focused benchmark/trial tests 41 passed; full pytest 345 passed plus 4 subtests with 2 historical skips; unittest 162 passed with 2 skips; compileall/diff/manuscript/secret gates passed. Post-run `verify_trial_manifest` also passed.
- Real benchmark result: XGBoost macro-F1 `0.621021` (95% CI `0.597319–0.644690`). Baseline-minus-XGBoost macro-F1 differences were Logistic Regression `-0.114800` (`-0.147597–-0.083224`), Random Forest `-0.028681` (`-0.049949–-0.008049`) and LightGBM `-0.015533` (`-0.038121–0.006382`). No baseline met the positive-point plus positive-CI-lower gate.
- Secondary result: QWK was XGBoost `0.567602`, LightGBM `0.588329`, Random Forest `0.631678`, Logistic Regression `0.371011`. QWK is reported but is not gate-eligible under the user-approved protocol.
- Known failures/open risks: repeated XGBoost probability-sum and LightGBM feature-name warnings occurred. Persisted XGBoost max row-sum deviation was `8.38e-08`; exact renormalization changed no argmax and changed aggregate log loss by only `1.81e-10`, so the macro-F1 gate is unaffected, but canonical probability evidence must be warning-clean. Core/supplementary remain not release-ready; downstream shared-model adoption and the existing calibration/figures/tables/CI/lock/licence/history/EOL blockers remain.
- Exact next action: complete independent warning audit, add exact float64 probability renormalization and regression tests, then refactor the next downstream consumer to use the persisted shared folds/exact selected XGBoost fold models.
- Decisions awaiting the user: none from the model gate; verified ethics metadata later; acquisition mismatch only if triggered.
- Paid API/network calls in this phase: zero; process-local TCP/UDP/DNS denial remained active.
- Manuscript edits: none.

The trial is real decision evidence but explicitly `canonical_release_eligible=false`; final manuscript numbers still require the clean complete core rebuild.
