# Current Status

- Current phase: Unit 2A implementation/regression complete; checkpoint preparation
- Last completed task: immutable scoped manifests/stage graphs, core import isolation, external split and fail-closed entrypoints
- Work currently in progress: final Unit 2A diff/log review and authorized checkpoint commit
- Files modified but not finalized: Unit 2A config/orchestration/manifest/external/dataset-card/legacy-figure files, contract tests and finalization logs
- Latest tests: pytest 250 passed + 4 subtests, 2 historical skips; unittest 162 passed, 2 skipped; focused Unit 2A 50 passed, 2 skipped; compileall/diff/manuscript checks passed
- Known failures: scopes intentionally not release-ready; no clean full v2 build; entrypoint completion contract open; test-selected calibration; invalid primary uncertainty; missing shared folds/baselines/nested tuning; new core figures/tables absent; no CI/lock; raw licence/history blockers
- Exact next action: create the authorized tested Unit 2A checkpoint, then record Unit 2B shared-fold/nested-benchmark acceptance criteria before edits
- Decisions awaiting user: verified ethics institution/unit/reference/date later; baseline-over-XGBoost gate only if triggered; acquisition mismatch only if triggered
- Paid API calls in this phase: zero
- Manuscript edits: none
