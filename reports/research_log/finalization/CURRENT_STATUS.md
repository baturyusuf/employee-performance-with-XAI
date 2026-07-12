# Current Status

- Current phase: Unit 2B shared-fold/nested-benchmark implementation checkpoint; real benchmark blocked on the primary metric decision
- Last completed task: shared folds, exact four-model factories, restrained nested search, paired OOF bootstrap, manifest-bound inputs and enforced baseline stop gate implemented and fully regression-tested
- Work currently in progress: persistent log/issue update and tested checkpoint commit preparation; no real model fitting
- Files modified but not finalized: canonical/model-grid configs; shared-fold, model-factory, benchmark, bootstrap and build-orchestration modules; Unit 2B tests and finalization logs (see `git status`)
- Latest tests: Unit 2B focused 83 passed; full pytest 314 passed plus 4 subtests with 2 historical skips; unittest 162 passed with 2 skips; compileall/diff/manuscript checks passed
- Real-input preflight: 1,200 samples; 10 exact outer folds; 10,800 inner assignments; three inner folds per outer fold; no artifact written and no model fitted
- Known failures: `selection_metric`/`baseline_gate_metric` intentionally null; core/supp scopes not release-ready; downstream policy/calibration/SHAP/fairness consumers do not yet use shared folds/selected fold models; conflicting legacy fixed-XGBoost consumer remains; no clean full v2 build; calibration/figures/tables/CI/lock/licence/history blockers remain
- Exact next action: obtain A/B/C metric decision, freeze/hash the metric contract, checkpoint Unit 2B infrastructure, then run the real four-model nested OOF benchmark and apply the baseline-over-XGBoost stop gate
- Decisions awaiting user: A=macro-F1, B=QWK, or C=macro-F1 selection plus corrected co-primary macro-F1/QWK gate; verified ethics metadata later; baseline-reference decision only if the result gate triggers; acquisition mismatch only if triggered
- Paid API/network calls in this phase: zero
- Manuscript edits: none
