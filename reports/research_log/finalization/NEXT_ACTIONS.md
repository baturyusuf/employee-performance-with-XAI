# Next Actions

1. Stage the tested 10×5 benchmark/trial implementation, dependency provenance and persistent logs; run staged whitespace/diff checks.
2. Create an authorized checkpoint commit on `finalization/leakage-aware-v2` and confirm the worktree is clean.
3. Run the verified-real-INX offline noncanonical trial with 10 outer × 5 inner folds and 5,000 paired OOF bootstrap draws.
4. Verify runtime, input/side-input/config/source hashes, exact-once OOF coverage, model hashes, selection records, intervals and the macro-F1 baseline gate.
5. Persist the trial command, run ID, result hashes, runtime and gate outcome in both finalization log sets.
6. If any baseline has positive macro-F1 point advantage and paired 95% CI lower bound above zero, stop before downstream XAI and request the reference-model decision. Otherwise continue automatically.
7. Refactor policy ablation, predeclared sigmoid calibration, OOF SHAP and subgroup/proxy consumers to use the shared-fold contract and exact selected XGBoost outer-fold models; remove the conflicting legacy fixed-parameter source.
8. Before final release claims, make source-tree verification cross-checkout EOL-stable and finish dependency lock/CI gates.

Do not cite dirty preflight hashes, run/reuse old `reports/manuscript_final/latest`, edit the manuscript, call APIs, publish, push or merge.
