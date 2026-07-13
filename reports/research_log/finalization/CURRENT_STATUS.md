# Current Status

- Current phase: Unit 2C-A exact selected-XGBoost OOF SHAP binding is implemented and fully tested; preparing its checkpoint.
- Last completed task: same-run, hash-verified XGBoost fold-model reader; exact OOF replay; fitted-axis grouping; no-refit SHAP; descriptive dependent-fold stability; atomic output publication; and builder wiring all passed focused/full gates.
- Work currently in progress: persist exact commands/results and checkpoint Unit 2C-A without staging the historical 91.8 MB trial package.
- Files modified but not finalized: canonical SHAP config, benchmark artifact reader, canonical SHAP axis helper, SHAP/build integration, five focused test files, and persistent logs. The immutable untracked trial directory remains read-only and unstaged.
- Latest tests: Unit 2C-A focused 59 passed plus 4 subtests; full pytest 389 passed plus 4 subtests with 2 historical skips; unittest 164 passed with 2 skips; compileall/diff/manuscript/secret/path/100 MB gates passed.
- Real benchmark result: XGBoost macro-F1 `0.621021` (95% CI `0.597319–0.644690`). Baseline-minus-XGBoost macro-F1 differences were Logistic Regression `-0.114800` (`-0.147597–-0.083224`), Random Forest `-0.028681` (`-0.049949–-0.008049`) and LightGBM `-0.015533` (`-0.038121–0.006382`). No baseline met the positive-point plus positive-CI-lower gate.
- Secondary result: QWK was XGBoost `0.567602`, LightGBM `0.588329`, Random Forest `0.631678`, Logistic Regression `0.371011`. QWK is reported but is not gate-eligible under the user-approved protocol.
- Known failures/open risks: the historical `6a80074` reader/OOF replay passes, but its old ndarray preprocessing lacks nested OneHotEncoder feature-name lineage and is intentionally rejected by the new SHAP axis; it cannot feed canonical SHAP. A current-code fold-1 refit passed axis/grouped-SHAP checks, but final SHAP artifacts require a new same-commit benchmark+SHAP core run. Policy/calibration/fairness/external/figures/tables/lock/licence/history/EOL blockers remain.
- Exact next action: checkpoint Unit 2C-A, then record/refactor leakage-policy ablation to consume shared folds and paired sample-level bootstrap evidence without independently regenerating splits.
- Decisions awaiting the user: none from the model gate; verified ethics metadata later; acquisition mismatch only if triggered.
- Paid API/network calls in this phase: zero; both real-data diagnostics installed explicit process-local TCP/UDP/DNS denial guards.
- Manuscript edits: none.

The trial is real decision evidence but explicitly `canonical_release_eligible=false`; final manuscript numbers still require the clean complete core rebuild.
