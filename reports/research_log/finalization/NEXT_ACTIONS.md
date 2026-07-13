# Next Actions

1. Commit and push the checkpoint-hash/log synchronization; verify the tracked worktree is clean and no Unit 2G process is active.
2. Keep the local 91.8 MB diagnostic trial, raw datasets, caches, environments, partial outputs and historical packages outside version control. Keep the trial exclusion local-only.
3. Execute the real HRDataset_v14 production stage offline from the clean checkpoint with all eight candidates and 5,000 resamples. Use a visibly noncanonical stage-validation run ID and do not create/promote a complete package manifest.
4. Validate every persisted input, side input, fold, model, OOF, calibration, SHAP, support, path and artifact hash. Retain the output only as stage-validation evidence until the complete core package exists.
5. If that stage fails, preserve the failure and fix only demonstrated implementation defects; do not relax the approved feature policy, folds, primary metric, calibration or claim boundary to improve scores.
6. Continue the supplementary heuristic-search/task-bounded robustness work, task-aware metric schema, dataset acquisition/publication transition, replacement table/figure set, dependency lock, global core no-network guard/CI assertion, and manuscript-support source maps.
7. Enable a scope's `release_ready` flag only when every declared runner and contract is complete. Then run the final clean two-scope rebuild, explicitly migrate the historical physical `latest`, and promote only the verified package-level pointer.
8. Freeze the claim matrix exclusively from that complete clean package. Manuscript editing remains outside the current authorization.

Do not call paid APIs, invent ethics/licence facts, delete local raw data, admit historical/diagnostic output as v2 evidence, or make causal/fairness/deployment/actionability claims.
