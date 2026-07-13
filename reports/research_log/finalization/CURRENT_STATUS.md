# Current Status

- Current phase: Unit 2B approved 10 outer × 5 inner benchmark contract is tested and ready for a clean checkpoint, followed by the real noncanonical four-model trial.
- Last completed task: froze macro-F1 selection/gate, QWK-only `0.001` tie-breaking, the exact 5,000-draw paired OOF bootstrap, 1,080/120 fold denominators, and direct runtime dependency provenance; reran real-INX in-memory preflight and all regression gates.
- Work currently in progress: update/stage persistent logs, create the tested checkpoint commit, verify a clean worktree, then execute `src.experiments.run_model_benchmark_trial`.
- Files modified but not finalized: `configs/manuscript_final.yaml`, `configs/model_grid.yaml`, `requirements.txt`; shared-fold/benchmark/bootstrap/build sources; their focused tests; new trial entrypoint/test; persistent finalization logs. Exact paths are the current `git status --short` output.
- Latest tests: 41 focused benchmark/trial tests passed; full pytest 345 passed plus 4 subtests with 2 historical skips; unittest 162 passed with 2 skips; compileall, diff, manuscript-no-change and high-entropy secret scan passed.
- Latest real-input preflight: verified INX SHA-256 `b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a`; 1,200 rows; class support 194/874/132; 10 outer folds; 10,800 inner assignments; five inner folds; validation size 216. It was an in-memory dirty-worktree check and wrote no artifact/model.
- Known failures/open risks: no real benchmark/gate outcome exists yet; core/supplementary scopes remain not release-ready; downstream policy/calibration/SHAP/fairness consumers do not yet consume shared folds/exact selected models; conflicting legacy fixed-XGBoost consumer remains; calibration/figures/tables/CI/lock/licence/history blockers remain. Raw-byte source-tree hashing is EOL-sensitive across checkouts and must be fixed before portable release claims.
- Exact next action: stage the intended correction, run `git diff --cached --check`, checkpoint commit on `finalization/leakage-aware-v2`, confirm clean worktree, then run the real offline noncanonical benchmark trial.
- Decisions awaiting the user: baseline-reference model only if the macro-F1 superiority gate triggers; verified ethics metadata later; acquisition mismatch only if triggered.
- Paid API/network calls in this phase: zero.
- Manuscript edits: none.

The earlier 10×3 preflight and every dirty preflight hash are superseded engineering diagnostics and cannot be cited or reused as scientific evidence.
