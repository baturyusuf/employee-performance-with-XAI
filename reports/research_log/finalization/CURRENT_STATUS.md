# Current Status

- Current phase: Unit 2D shared-fold leakage-policy ablation checkpoint is implementation-complete and fully tested; work is paused at the calibration-training decision gate.
- Last completed task: the policy stage now reuses exact primary XGBoost OOF evidence, applies the same fold-selected parameter schedule to five non-primary policies on the exact shared folds, persists exactly-once OOF/fit/bootstrap evidence, and removes fold-t/Wilcoxon inference from the canonical path.
- Work currently in progress: none beyond the user decision gate. Calibration implementation must not start until the within-outer-training design is selected.
- Files modified but not finalized: none in the Unit 2D checkpoint. The immutable 91.8 MB historical trial directory remains local, read-only, untracked and unstaged under D5.
- Latest tests: focused Unit 2D/identity/bootstrap suite 92 passed; full pytest 403 passed plus 4 subtests with 2 historical skips; unittest 174 passed with 2 skips; compileall/diff/manuscript/secret/path/100 MB/README/terminology gates passed.
- Real benchmark result: XGBoost macro-F1 `0.621021` (95% CI `0.597319–0.644690`). Baseline-minus-XGBoost macro-F1 differences were Logistic Regression `-0.114800` (`-0.147597–-0.083224`), Random Forest `-0.028681` (`-0.049949–-0.008049`) and LightGBM `-0.015533` (`-0.038121–0.006382`). No baseline met the positive-point plus positive-CI-lower gate.
- Secondary result: QWK was XGBoost `0.567602`, LightGBM `0.588329`, Random Forest `0.631678`, Logistic Regression `0.371011`. QWK is reported but is not gate-eligible under the user-approved protocol.
- Known failures/open risks: no canonical policy artifact exists yet because exact `1e-12` primary OOF replay requires a new same-commit benchmark; the historical trial is intentionally incompatible. Calibration still regenerates wrong folds, uses legacy parameters, selects methods from outer-test results and uses fold-t intervals. Fairness/external/figures/tables/lock/licence/history/EOL blockers remain.
- Exact next action: obtain the calibration-training decision, then implement the selected predeclared sigmoid protocol against the shared-fold/benchmark contract.
- Decisions awaiting the user: calibration training protocol (cross-fitted refit, benchmark-persisted cross-fit, or single holdout); verified ethics metadata later; acquisition mismatch only if triggered.
- Paid API/network calls in this phase: zero; both real-data diagnostics installed explicit process-local TCP/UDP/DNS denial guards.
- Manuscript edits: none.

The Unit 2D real-INX fold-1 diagnostic passed five non-primary current-code fits with 1,080 train/120 test rows, exact selected-parameter/lineage checks, zero warnings, machine-precision probabilities, zero writes and an unchanged historical manifest. It is a bounded implementation diagnostic, not manuscript evidence. Final numbers still require the clean complete core rebuild.
