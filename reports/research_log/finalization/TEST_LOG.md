# Persistent Test Log

Date: 2026-07-13

- Pytest: 188 passed plus 4 subtests, exit 0, 14.75 seconds
- Unittest: 161 passed, exit 0, 4.764 seconds
- Compileall: exit 0
- API keys were removed from the subprocess environment
- Worktree remained clean after tests

Important: baseline green tests do not establish scientific readiness. They do not cover every confirmed v2 issue in `../finalization_v2/02_issue_register.csv`. The original baseline had only real-input preflights; the later noncanonical 10x5 real-model trial is recorded below, while a full canonical scientific integration run has still not completed.

## Unit 1A — 2026-07-13

- New actual-input/interim/acquisition contract tests: 11 passed.
- Full pytest: 199 passed plus 4 subtests.
- Unittest: 161 passed.
- Compileall: passed.
- Pinned real-data preflight: five logical tasks passed, zero downloads.
- Diff check: passed.
- Manuscript diff: none.

This closes the Unit 1A implementation/test checkpoint only. V2-001 is not resolved until loader receipts and side inputs are bound into the release manifest/cache and a full real-data v2 build passes.

## Unit 1B — 2026-07-13

- Manifest/cache/snapshot contract: 21 passed.
- External explicit-input/claim-boundary contract: 23 passed.
- Real-data manifest validation: schema 2; five logical receipts; seven side inputs; passed.
- Full pytest: 218 passed plus 4 subtests.
- Full unittest: 161 passed.
- Compileall, diff check and manuscript no-change check: passed.
- Paid/API/network calls: zero.

V2-001 and V2-002 have passed implementation and real-input preflight but remain open pending a clean, cache-disabled full v2 build and artifact verification.

## Unit 2A — 2026-07-13

- Scoped manifest/external/orchestration/dataset-card/final-manifest/legacy-figure contracts: 50 passed, 2 historical skips.
- Core static import graph: no LLM/chatbot/agent/OpenAI reachability.
- Core/supplementary real-input manifests: passed with exact scoped receipts and side inputs.
- Core entrypoint incomplete-release gate: expected fail-closed before artifact creation.
- Full pytest: 250 passed, 2 skipped, plus 4 subtests.
- Full unittest: 162 passed, 2 skipped.
- Compileall, diff and manuscript no-change checks: passed.
- Scientific experiments/artifacts and network/API calls: zero.

V2-011 and V2-013 pass their implementation contracts but remain open until real scope packages are release-ready and validated. V2-012 and V2-021 remain in progress.

## Unit 2B — 2026-07-13

- Shared-fold, model-factory, nested-isolation, OOF-bootstrap, gate and runner-input-binding contracts: 83 passed.
- Full pytest: 314 passed, 2 skipped, plus 4 subtests.
- Full unittest: 162 passed, 2 skipped.
- Compileall, diff check and manuscript no-change check: passed.
- Real canonical-loader fold preflight: 1,200 samples, 10 outer folds, 10,800 inner assignments, three inner folds per outer fold; passed in memory with no written artifact.
- Paid/API/network calls and real model fits: zero.

At that historical checkpoint, V2-007 through V2-010 had tested reusable infrastructure but the metric decision was still pending. That condition is superseded by the accepted 10×5 correction below; the issues remain open until the real benchmark and downstream adoption succeed.

## Unit 2B 10×5 Correction — 2026-07-13

- Corrected trial/benchmark/fold/bootstrap/input-binding focused suite: 104 passed.
- Full pytest: 343 passed, 2 skipped, plus 4 subtests.
- Full unittest: 162 passed, 2 skipped.
- Compileall, diff and manuscript no-change checks: passed.
- Verified real-INX 10×5 preflight: passed in memory; 1,200 rows, five inner folds, validation partitions of 216; no model fit/artifact.
- Offline network denial regression: passed.
- Real four-model benchmark: not yet run; requires clean correction checkpoint first.

## Unit 2B 10×5 Final Hardening — 2026-07-13

- Focused benchmark/trial tests after exact tie-tolerance, bootstrap, denominator and dependency-provenance hardening: 41 passed.
- Full pytest: 345 passed, 2 skipped, plus 4 subtests.
- Full unittest: 162 passed, 2 skipped.
- Compileall, diff, manuscript-no-change and high-entropy secret-pattern scan: passed.
- Verified real-INX in-memory preflight: 1,200 rows, support 194/874/132, 10 outer folds, five inner folds, 10,800 inner assignments and validation size 216.
- Manifest preflight recorded `joblib 1.5.3` and `threadpoolctl 3.6.0`.
- Real models/trial artifacts/API/network calls: zero. The standalone trial command has not yet executed.

## Real Four-Model Trial — 2026-07-13

- Entry point exit: 0; shell runtime 725.2 seconds; manifest runtime 722.522 seconds.
- `verify_trial_manifest`: passed after completion.
- Verified structure: 1,200 shared outer rows; 10×5 folds; 300 candidate rows; 40 selected/fold-metric/model-index rows; 4,800 exactly-once OOF rows; 36 model-summary rows; 27 paired rows; 5,000 valid bootstrap draws; 40 model hashes.
- Gate: not triggered; all three baseline-minus-XGBoost macro-F1 rows failed the positive-point plus positive-CI-lower condition.
- Output identity: commit `6a80074`, config `7e70bf66…`, scientific input `8be7c5d7…`, fold contract `9fd24f0c…`, resamples `3528e437…`.
- Warning diagnostic: exact XGBoost renormalization changed zero argmax labels and aggregate log loss by `1.81e-10`; warning cleanup remains required before canonical probability evidence.
- API/network/manuscript edits: zero.

## Unit 2C-0 Probability and Feature Warning Hygiene — 2026-07-13

- Focused canonical-model/benchmark/nested/trial suite: 63 passed.
- Full pytest: 350 passed, 2 skipped, plus 4 subtests.
- Full unittest: 162 passed, 2 skipped.
- Compileall, diff, manuscript-no-change and high-entropy secret scan: passed.
- Real fold-1 diagnostic: LightGBM and XGBoost each had 0/120 label mismatches versus the immutable trial; transformed output was a named 46-column DataFrame; warnings were zero.
- Maximum probability difference: LightGBM `3.33e-16`; XGBoost `7.02e-08` from deliberate exact row normalization. New row-sum deviations were at most `2.22e-16`.
- Scientific artifacts: none. The existing trial was not modified or rerun.

## Unit 2C-A Exact-Model OOF SHAP — 2026-07-13

- Reader/axis/exact-OOF/policy/wiring/shared-fold/forbidden-feature focused suite: 59 passed plus 4 subtests.
- Full pytest: 389 passed, 2 skipped, plus 4 subtests.
- Full unittest: 164 passed, 2 skipped.
- Compileall, diff, manuscript-no-change, high-entropy secret, scientific-diff absolute-path and 100 MB candidate scans: passed.
- Historical reader replay: 10 exact XGBoost models, 1,200 OOF rows, gate false, model-set SHA-256 `492aa445efc0df9348b9c85714ec09a539d6195fd46c2151c102b6ba02a1c607`; immutable manifest unchanged.
- Historical axis check: expected fail-closed because old nested OneHotEncoder has no `feature_names_in_`; this proves the historical trial cannot feed canonical SHAP.
- Current-code real fold-1 SHAP diagnostic: 120 samples, 46 transformed features, 20 raw groups, grouped shape `(120,3,20)`, maximum sum error `0`, zero warnings and zero label mismatches.
- Scientific artifacts: none; canonical benchmark+SHAP regeneration remains pending.

## Unit 2D Initial Policy Binding Checks — 2026-07-13

- Builder current-run upstream binding plus repository cross-config policy projection suite: 22 passed in 1.64 seconds.
- The shared policy names in `configs/feature_sets.yaml` now match the canonical exclusions, and the file is explicitly labelled a legacy compatibility projection whose shared names fail closed on drift.
- Main policy execution/bootstrap refactor and its broader tests are still in progress; no scientific artifact was generated.

## Unit 2D Final Shared-Fold Policy Gates — 2026-07-13

- Initial comprehensive policy/shared-fold/bootstrap/reader suite: 75 passed in 22.15 seconds.
- Post-independent-review policy/config/side-input suite: 35 passed in 7.82 seconds.
- Final expanded focused suite: 92 passed in 26.85 seconds.
- Full pytest: 403 passed, 2 skipped, plus 4 subtests in 83.35 seconds.
- Full unittest: 174 passed, 2 skipped in 6.889 seconds.
- Compileall, diff check, manuscript no-change, high-entropy secret, absolute user-path, 100 MB candidate-file, README local-link and active leakage-aware terminology gates: passed.
- Real INX fold-1 diagnostic: five non-primary fits each used 1,080 train/120 test rows; feature/selected-parameter lineage passed; maximum probability simplex error `2.22e-16`; warnings/files/network calls zero; historical trial manifest unchanged.
- One attempted focused command failed before collection because it named nonexistent `tests/test_oof_bootstrap_uncertainty.py`; the corrected exact suite used the three repository bootstrap test modules and passed. No scientific execution occurred in the failed attempt.
- Canonical Unit 2D artifact package: not generated; requires the new same-commit benchmark upstream.

## Unit 2D Checkpoint Revalidation — 2026-07-13

- Full pytest after README/traceability repairs: 403 passed, 2 skipped, plus 4 subtests in 83.35 seconds.
- Full unittest: 174 passed, 2 skipped in 6.904 seconds.
- Compileall, diff, manuscript no-change, secret, scientific-diff absolute-path, 100 MB candidate and active leakage-aware terminology checks: passed.
- README clean-checkout audit: all 17 local links exist and are represented in the Git index; the local untracked trial path is plain text, not a broken GitHub link.
- Issue register CSV: 26 rows parsed with the declared schema.
- One parallel wrapper failed because it imported nonexistent `config_hash`; corrected `canonical_config_hash` loading passed and reproduced the recorded config/feature-policy hashes. This was not a scientific/test failure.

## Unit 2E Option-A Calibration Implementation - 2026-07-13

- Calibration/config/upstream/bootstrap focused suite: 85 passed plus 7 subtests in 19.77 seconds after all independent-review fixes.
- Independent read-only review suite: 48 passed plus 7 subtests; no material implementation blocker.
- Compileall and diff checks: passed.
- Warning-free sigmoid smoke: passed on scikit-learn 1.8.0 with one-thread L2 fitting; zero warnings and exact parameter replay.
- Real INX fold-1 diagnostic: five inner fits; 1,080 cross-fitted training rows; 120 untouched outer-test rows; zero warnings; 1.082 fit seconds; maximum calibrated simplex error `2.22e-16`.
- Canonical real-run status: not run. Current config hash `d755ecc3...` is incompatible with immutable historical benchmark hash `7e70bf66...`; this fail-closed state is expected. A same-config run is deferred until all remaining core inputs are frozen.
- First full-pytest attempt failed during collection because the supplementary counterfactual module imported deleted private calibration helper `_fit_pipeline`. It ran no test/scientific stage. The module was decoupled to a local canonical one-thread supplementary fit helper; the two direct counterfactual modules then passed 5 tests.
- Full pytest after the fix: 425 passed, 2 skipped and 11 subtests in 83.64 seconds.
- Full unittest: 173 passed with 2 skips in 7.089 seconds.
- Added a static regression preventing future supplementary imports of calibration private helpers; combined calibration/counterfactual focus passed 25 tests.
- Final hygiene: manuscript unchanged; no secret-pattern match; no absolute user path in scientific diff; no candidate file over 100 MB; no active leakage-safe terminology; `pip check` clean; 17 README local links tracked; issue register parsed 26 rows.
- Final post-regression rerun after adding the private-import guard: pytest 426 passed, 2 skipped and 11 subtests in 88.59 seconds; unittest 174 passed with 2 skips in 7.128 seconds; compileall and diff check passed.
- Final all-ten-fold real-INX diagnostic: 50/50 selected-candidate inner fits, ten calibrators, 10,800 training OOF rows, 1,200 raw plus 1,200 sigmoid test rows, ten source-model hashes and zero warnings in 12.303 seconds. Noncanonical in-memory diagnostic only; no artifact.
- Tested implementation checkpoint: `0f820b3` (`feat(calibration): cross-fit sigmoid on benchmark folds`). Tracked files were clean immediately after commit; only the deliberately unstaged local historical trial remained visible.
