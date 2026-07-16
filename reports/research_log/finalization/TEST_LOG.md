# Persistent Test Log

## GitHub publication-assets validation - 2026-07-16

- Aggregate publication subset: 50 files, 6,337,343 bytes, maximum file 2,195,799 bytes.
- Reproducible repository-relative canonical-JSON inventory SHA-256: `645f5295c36f2d1e0a3b7809e67d805f1bb16b906121ca91e032bc1d36e2d228`.
- Sensitive/employee-identifier table and figure-source header findings: zero.
- Raw dataset, secret-pattern, machine-absolute-path, lock/partial/staging, and forbidden full-package candidate findings: zero.
- Canonical asset bytes were not modified; directory inventories match the frozen figure/table manifests and stage contracts.
- Production validation of the real canonical figure/core-table/supplementary-table directories: exit 0 in 2.770 seconds; seven figures, eleven core tables, three supplementary tables, complete stage contracts, and closed-world inventories passed.
- Focused publication/manifest/CI-contract pytest: exit 0; 22 passed, 1 skipped in 8.37 seconds (14.335 seconds wall).
- README audit: 64 local links, zero missing targets, and zero linked files outside the Git index.
- Staged allowlist: 61 files total, including exactly 50 admitted canonical assets; zero unstaged files, unexpected canonical paths, forbidden full-package paths, raw data, employee-identifier headers, secrets, machine-absolute paths, or manuscript changes.
- Canonical SVG and generated claim-boundary bytes retain generator-owned trailing spaces so their frozen manifest hashes remain unchanged; the editable/noncanonical staged subset passes `git diff --check` when those immutable generated files are excluded.
- The clean-worktree repository gate and hosted CI are post-commit checks; their outcomes belong to the checkpoint handoff because they cannot precede the commit they validate.

## Final canonical validation - 2026-07-14

- Focused final implementation tests: 103 passed, exit 0.
- Complete pytest: 752 passed, 2 skipped, 11 subtests passed, exit 0, 139.36 seconds.
- Unittest discovery: 179 tests passed, 1 skipped, exit 0, 9.487 seconds.
- Compileall: exit 0, 0.131 seconds.
- Core build: exit 0, 1,641.749 seconds.
- Supplementary build: exit 0, 1,926.755 seconds.
- Atomic promotion: exit 0, 5.968 seconds.
- Strict post-promotion validation: exit 0, 5.677 seconds.
- Independent canonical Unit 2G replay: exit 0, 8.321 seconds; zero model, sigmoid, and grouped-SHAP value drift.
- Closed-world manifest, forbidden-feature, raw/secret, path portability, README/manuscript, large-file/staging, and offline runtime gates: passed.
- Final package scan findings: zero machine paths, active `leakage-safe` terms, secrets, raw datasets, and lock/partial/staging directories.
- Scientific network and paid API calls: zero.

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

## Unit 2F Baseline Audit - 2026-07-13

- Test/model execution: none; this was the required pre-modification read-only audit.
- Tracked worktree: clean at audit start; only the immutable historical trial was untracked.
- Historical/shared fold comparison: 1,200 rows each, 1,091 mismatches, 109 matches (`9.0833%`).
- Historical v1 subgroup/proxy package: rejected for v2 because config/run/scientific identity is stale/incomplete and metadata contains absolute Windows paths.
- Proxy inference: rejected because published CIs are Student-t intervals over ten dependent CV folds.
- Positive controls to preserve: metric-specific denominators, group-support fields, sensitive/operational categories, target removal and limitations.
- Three independent audits completed without edits and agreed on the Unit 2F acceptance contract.

## Unit 2F Focused Implementation Validation - 2026-07-13

- Initial exact-upstream/config/support/builder suite: 79 passed in 21.38 seconds.
- Expanded suite including dynamic common-estimand, deterministic batching and real-sklearn synthetic proxy behavior: 84 passed in 20.02 seconds.
- Post-hardening direct subgroup/proxy subset: 31 passed in 4.29 seconds.
- `py_compile`, canonical config validation and `git diff --check`: passed. Intermediate pre-review config hash: `be2b3f9f7e052df42ad9dc413d10e29bfc2ad6dd63a38513d21957aa9908523f`; it is superseded below.
- Dynamic tests prove: canonical `outer_fold` only; two common groups required for paired gaps; identical systems yield exact-zero paired point/CI independent of row order; batch sizes produce bit-identical subgroup/proxy draws; two unique proxy systems create 20 shared-fold fits and exactly-once OOF rows; the alias performs no fit; proxy target, performance target and identifier are absent from raw and transformed predictors.
- Intermediate pre-review real INX in-memory diagnostic: 1,200 rows, 20 warning-free fits, 2,400 exactly-once OOF rows, 5,000 target-specific bootstrap draws in batches of 200, 5.233 measured seconds, exact historical outer-map match, zero files/network/API calls. Diagnostic-only macro-F1: job-role-retained `0.968543` (pointwise 95% CI `0.956635-0.980215`); job-role-removed `0.247368` (`0.226694-0.268709`). No manuscript claim is admitted; the final diagnostic is recorded below.
- Post-review direct focus: 44 passed. Expanded integration focus: 102 passed in 20.30 seconds.
- Full repository pytest: 467 passed, 2 skipped and 11 subtests in 85.74 seconds.
- Full unittest: 178 tests passed with 2 skips in 7.045 seconds. Compileall passed.
- Independent follow-up review: 102 focused tests passed; no remaining P0/P1 defect.
- Final config validation: `3c9588c1327ac563a85586835b19b30768860165dc26b61fcf7aafbce3bb1421`.
- Final real-INX no-write/no-network diagnostic: 20 fits, 2,400 exactly-once nominal proxy OOF rows, 5,000 batched draws, exact outer-map match, minimum overall class support 20, minimum nonzero fold support 1, two zero-support cells, 3.776 seconds. Diagnostic values are not manuscript evidence.
- Final hygiene: manuscript unchanged; zero secret, absolute-path, >100 MB candidate, active leakage-safe or legacy proxy-import matches; 17 README links valid; 27 issue rows parsed; `pip check` clean.
- Tested implementation checkpoint: `a490d1e` (`feat(fairness): bind subgroup and proxy diagnostics to OOF evidence`). Immediate tracked status was clean; only the excluded historical trial remained untracked.

## Unit 2G External Replication Baseline Audit - 2026-07-13

- Scientific model execution: none; required pre-modification audit only.
- Verified HRDataset: SHA-256 `cb19996755c93c0a8d6527f59da4701c80aef65eff854906546dce286249813c`, 311 rows, 36 columns, mapping hash `4988bde12fbd0198102f22f4078fd31ba20ea4285160363d9cf101610e9f19d0`.
- Raw target counts: Exceeds 37, Fully Meets 243, Needs Improvement 18, PIP 13. Mapped counts: class 2 = 31, class 3 = 243, class 4 = 37; no unmapped value; 10 × 5 nested stratification feasible.
- Baseline external tests: 19 passed in 6.46 seconds.
- Confirmed uncovered defects: static rather than nested model selection; fold-mean CI; direct department/job-role aliases; ambiguous governance fields; no cross-fitted sigmoid; incomplete SHAP stability; legacy unsupported subgroup helpers; incomplete scientific identity; non-atomic writes.
- Source/licence authenticity: manual review required. Network/download/model/artifact/manuscript operations: zero.

## Unit 2G Implementation Gate - 2026-07-13

- Broad external implementation/config/model/calibration/bootstrap/SHAP/subgroup/stage/manifest regression: 120 passed in 33.61 seconds.
- Post-review selected suite: 68 passed in 20.94 seconds.
- Earlier manifest/stage identity focus: 12 passed in 2.10 seconds.
- Compileall: passed.
- `git diff --check`: passed.
- Manuscript diff: unchanged.
- Independent read-only review: no remaining P0/P1 defect.
- Real-data execution: pending; implementation status must not be upgraded to scientific-result completion until its artifact manifest, identities, schemas and claims are validated.
- Expected department proxy status: `not_estimated_insufficient_outer_training_class_support`; this is a support limitation, not a test failure and not a numerical result.

## Unit 2G Interrupted Full-Gate and Recovery Record - 2026-07-13

- Full pytest attempt: 519 passed, 2 skipped, 11 subtests passed, 11 failed in 103.20 seconds. Four failures were validation-order message contracts; seven were temporary-project side-input roots/fixtures. Production validation order/root inference and fixtures were corrected; the exact combined focus then passed 64.
- Independent review then found that stage-start cleanliness rejected builder-created current-run files. Exact current-run allowance plus unrelated-untracked rejection was added; integration focus passed 97.
- Two subsequent full-suite processes were deliberately terminated, not failed or abandoned: the first when review found the builder-output collision, the second when review found the still-open builder-resume collision. Process recovery confirmed neither remained running and neither wrote a scientific artifact.
- Current external/config/stage/input focus after exact output-root and date-fallback hardening: 96 passed in 10.54 seconds; compileall and diff check passed.
- Full pytest/unittest/hygiene gate is pending until the builder-resume regression is fixed.

## Unit 2G Final Publication-Contract Gate - 2026-07-13

- Resume/trust focus after interruption: 125 passed, 2 platform symlink skips.
- External scientific focus before final independent audit: 80 passed.
- Scientific P0 regressions added for case-insensitive forbidden aliases and canonical probability replay; post-fix independent external focus: 96 passed, 1 platform skip.
- Integrated trust/science focus after all fixes: 161 passed, 1 platform skip. One preceding assertion-only failure was caused by an outdated expected error regex and was corrected without changing production behavior.
- Independent post-fix trust focus: 132 passed, 2 platform skips; no P0 remained. Its release-ready promotion P1 was then fixed and tested.
- Final trust/release focus: 141 passed, 2 platform skips. A preceding two-failure run exposed only incomplete synthetic timing/runner fixtures; production validation was not weakened.
- Final full pytest: 648 passed, 3 skipped, 11 subtests passed in 109.16 seconds.
- Final unittest: 178 passed, 2 skipped in 7.573 seconds.
- Compileall: exit 0.
- `pip check`: `No broken requirements found`, exit 0.
- Canonical config validation: `5af0262e83a3720f8dca0b4d6795bdffc6bb2cefedc901ae0a47f9262d07f305`; scope hashes unchanged.
- Git diff check and manuscript no-change check: passed before documentation synchronization; repeated final hygiene follows before staging.
- Independent Unit 2G real-input diagnostic: exact 10x5 fold structure and production evaluator path, reduced candidate/bootstrap test budget, exact SHAP replay error 0, no file/network/API output. Diagnostic-only; it is not canonical evidence.
- Production 8-candidate/5,000-draw Unit 2G execution: not run. V2-028 remains open pending clean-commit execution and artifact validation.

## Unit 2G checkpoint documentation/hygiene gate - 2026-07-13

- `git diff --check`: passed.
- Manuscript no-change gate: passed.
- Index/raw/100 MB/reparse/secret/active-terminology gates: passed with zero findings.
- README audit: 17 local-link occurrences, 15 unique targets; every target exists and is represented in the Git index.
- Issue register: 28 rows with all required v2 columns.
- Historical trial preservation: 54 files, 91,820,515 bytes, untracked and locally excluded; no trial file entered the candidate set.
- Candidate portability review: four path-like literals, all required sanitization regexes or intentional negative-test fixtures; zero artifact/metadata machine-path leakage.
- This is a repository checkpoint gate, not a scientific execution. No production Unit 2G artifact was generated.
- One staged-review wrapper failed before evaluation because `$home` conflicts with PowerShell's read-only `$HOME`; rerunning it with `$pathHits` passed. No file or staged content changed during the failed wrapper.

## Unit 2G atomic-output recovery validation - 2026-07-14

- Orphan/process/port/lock audit: passed. Zero repository/run processes or listeners, zero duplicate experiments, zero stage locks/partials/staging residue; no process terminated.
- Closed-world stage inventory: passed. 124/124 contract outputs and 122/122 JSON/CSV artifact records match path, size and SHA-256; 125 stage files are non-empty and portable.
- Input and side-input binding: passed against saved generation identity. Actual HRDataset/INX receipts, six side inputs, config `5af0262e...`, core scope `af80b8a7...`, scientific input `71f1fc46...` and source tree `706690fc...` recompute/match.
- Outer/inner folds: passed. 311 samples exactly once across 10 outer folds; every outer-training row exactly once across five inner folds; no outer-test intersection.
- Nested tuning/models: passed. 400 candidate fit receipts, 80 candidate summaries, ten selected schedules and 50 model artifacts; all model hashes/sizes/partitions validate and independent replay gives maximum raw-OOF probability error zero.
- Cross-fitted sigmoid: passed. 50 inner fits, 2,799 outer-training OOF rows, ten source-model relationships and 30 class parameters; all outer-test selection/fit flags false; reconstructed calibrators replay with maximum error zero.
- Bootstrap: passed. Compressed/uncompressed hashes, shape `(5000, 311)`, sample order and index bounds validate.
- OOF SHAP: passed. `311 x 3 x 7 = 6,531` local rows from the exact ten prediction models; forbidden features absent; replay error zero; maximum additivity error `4.47e-06`; no refit; 45 descriptive dependent fold pairs and six deterministic cases.
- Subgroup/proxy: passed. Support thresholds 30/10 and 5,000 draws are explicit; 61 estimable/24 unsupported disparity intervals, 30 stable headline-eligible rows. Department proxy correctly publishes no fabricated estimate (`models_fitted=0`) because fold 4 lacks the singleton class in training.
- Transport/claim boundary: passed. Three safe common features are below the five-feature gate, locked transport is false, and the only admitted role is independent mapped-target replication.
- Package boundary: passed as a negative gate. No canonical run/final manifest, package status, claim matrix or `latest` promotion exists. The provisional input manifest correctly fails current-HEAD validation; the complete atomic stage remains valid under its saved identity.
- Focused pytest: `72 passed in 29.58s` (exit 0).
- D5 preservation check: passed. All 126 files and 65,412,766 bytes remain local after index-only removal; zero run-root paths remain tracked at the pending tip.

Scientific reporting checks found no atomic-stage defect but established two mandatory limitations for the canonical package: external subgroup rows need explicit raw-OOF/source identity, and grouped SHAP summaries need explicit raw-margin units. Sigmoid probability-quality gains must be separated from its macro-F1 reduction and zero class-4 argmax count. Unit 2G is not rerun.

Checkpoint documentation/hygiene gate: compileall, `pip check`, `git diff --check`, manuscript no-change, README link existence, issue CSV schema/uniqueness/completeness, raw candidate, 100 MB, secret, added absolute-path, active terminology and D5 local-preservation checks all passed. Counts: 20 README links (16 unique), 30 issues, 141 candidate paths (126 intentional deletions plus 15 small tracked files), zero prohibited findings, zero tracked stage files and 126 preserved local stage files. One first wrapper attempt used reserved `$HOME` as `$home` and exited before final scan evaluation; the corrected `$pathHits` wrapper passed and no file/index state changed during the failed attempt.

Final staged review: 141 paths consist exactly of 126 deletion-only entries under the ignored stage-validation root and 15 approved small documentation/log/ignore files. Unstaged and untracked counts are zero. Added-line secret and absolute-home scans are zero; manuscript diff is zero. A post-index-removal rehash again passed all 122 artifact-manifest rows and 124 stage-contract outputs, proving the local evidence bytes were not altered.

Push/recovery gate: the normal push timed out after 184 seconds with no remote response. The exact repository push/remote PID tree had zero listeners/endpoints and was terminated only after a graceful attempt failed; Git Credential Manager was not terminated. A public read-only GitHub ref check confirmed remote SHA `e25f403`, while local cleanup SHA is `b7b2ad3`. Result: push failed/recorded, local checkpoint coherent, no second retry authorized for this checkpoint.

## V2-029 reporting-contract gate - 2026-07-14

- Compileall over the three production modules and three focused test modules: passed.
- Config/SHAP/subgroup contract suite: 56 passed in 5.84 seconds.
- Expanded external config/nested/calibration/bootstrap/SHAP/subgroup/stage/manifest suite: 80 passed in 27.18 seconds.
- Regressions cover missing/sigmoid/mixed subgroup probability sources; config-owned raw method; invalid/mutated source receipts; semantic-hash row-order invariance; published hash scope/algorithm/columns; SHAP provider/config unit and additivity-space drift; explicit unit fields in local/global/class/fold evidence, metadata and fold receipts; and JSON/Markdown reason-code serialization.
- Canonical config hash: `ac32f7d80695e95adbad458ef31d9f1790b16e1eec306aaba57c5233f304e2f8`; core/supplementary scope hashes remain `af80b8a7...`/`18bbb5cb...`.
- `git diff --check`: passed. No Unit 2G rerun, scientific artifact mutation, manuscript edit, network or paid API call occurred.
- Synchronized checkpoint/hygiene rerun: 80 tests passed in 27.11 seconds; 30 issue rows and 21 README links validated; changed-file raw/100-MB/secret/absolute-path/manuscript scans returned zero findings. Independent final review passed 66 relevant tests in 6.01 seconds, diff check passed, and no P0/P1 defect remained.
- Full checkpoint pytest: 656 passed, 3 skipped and 11 subtests passed in 143.88 seconds. Full unittest: 178 passed, 2 skipped in 10.895 seconds. Both exited 0. Ruff was unavailable in the locked environment; its optional probe exited before linting and did not change dependencies.
- Final pre-stage gate: `pip check`, compileall, diff/manuscript, issue/README and raw/100-MB/secret/absolute-path/process checks all passed; zero prohibited findings and zero active repository execution were observed.
- Staged checkpoint review passed for exactly 20 files with zero unstaged/untracked, raw-data, manuscript, 100-MB, secret or absolute-path findings. Tested implementation commit: `9c603534268e7ba953cc1a05b23225b4fde488f5`.
- Documentation-only log-sync review passed for 9 files and produced `1639e182f877839995799319b434e1c356d131c1`. The subsequent push was not a test failure: it exited 1 because noninteractive HTTPS credentials were unavailable. Public remote verification remained `e25f403`; no orphan process or ref movement occurred.

## Post-Unit-2G read-only recovery gate - 2026-07-14

- Branch/worktree gate: passed at `8ce39e7939f00ee3269ca7ceaf829740cfc8130b`; branch exact, index/worktree clean, five commits ahead of remote.
- Process/endpoint gate: passed. Zero Python/pytest/model process and zero repository/run TCP listener or UDP endpoint; no process terminated. Ambiguous generic IDE Git metadata processes were not treated as repository experiments.
- Residue gate: passed. Zero stage/repository `.lock`, `.partial`, `.tmp`, `.incomplete` or staging path.
- Unit 2G stage contract: 124/124 output hashes and sizes passed.
- Unit 2G artifact manifest: 122/122 JSON/CSV rows agreed and passed; zero missing/unlisted/duplicate/unsafe path.
- Structural evidence gate: 50 models, 400 candidate fits, 1,555 raw OOF rows, 311 sigmoid rows, 50 calibration fits, 2,799 calibration-training rows, ten model/calibrator links, 6,531 local SHAP rows, 45 stability pairs, six cases, 391 subgroup metric rows and 85 disparity rows present.
- Package boundary: passed as a negative gate. The enclosing manifest remains `status=running` with no final package manifest/promotion, so the stage is valid atomic evidence but not canonical.
- Production rerun, model refit, API/network call, artifact write and manuscript edit: zero.

## V2-021a core figure-plan contract gate - 2026-07-14

- Initial compile and focused figure/core/legacy-isolation gate: 36 passed, 1 skipped.
- Expanded config/scope/manifest/figure gate after the first fixture repair: 114 passed, 2 skipped.
- Independent review: 25 passed, 1 skipped; no P0 in the V2-021a diff and all source filenames matched production writers.
- First full pytest: 662 passed, 3 skipped and 11 subtests, 9 failures. Every failure came from one temporary side-input config that removed the newly required canonical `core_figures` graph; no scientific code or artifact failed.
- Post-repair fixture/manifest/figure focus: 58 passed.
- Final full pytest: 671 passed, 3 skipped and 11 subtests in 111.10 seconds, exit 0.
- Final unittest: 178 passed, 2 skipped in 8.012 seconds, exit 0.
- Compileall and `pip check`: exit 0; no broken requirements.
- Open P1s: retire old numbered SHAP Figure 6/7 previews and replace the legacy v1 stem validator before any canonical core-figure build.
- Figure/artifact generation, scientific stage execution, API/network acquisition, Unit 2G rerun and manuscript edit: zero.
- Final post-documentation focus: 66 passed in 9.84 seconds; diff and manuscript no-change checks passed.
- Tested implementation checkpoint: `6da8273b458fd249d47d9bb5c75ebe9ff364617f` (`feat(figures): freeze leakage-aware core plan`).

## Unit 2G reusable acceptance gate - 2026-07-14

- Recovery gate: passed on required branch `finalization/leakage-aware-v2` at synchronized local/upstream HEAD `eab2b32150245fe7d406afcfb64827be67797752`; worktree/index were clean before changes.
- Process/endpoint gate: passed. No task-owned Python, pytest, Git, Node, shell, model or validation execution and no repository/run listener; no PID terminated.
- Residue/preservation gate: passed. No lock, partial, temporary, incomplete, staging or interrupted atomic output. The ignored 126-file/65,412,766-byte stage is preserved locally with zero tracked paths.
- Independent stage validator: exit 0 in 12.6 seconds. All 124 contract outputs, 122 manifest artifacts and 125 stage files passed the closed-world path/size/SHA-256 inventory. Raw bytes, canonical parsed content, schema, policy, side inputs, source/config/scientific identities and generation receipt agree.
- Scientific replay: passed. Exactly 311 outer rows, 10 outer x 5 inner folds, 400 candidates, 50 selected persisted models, 1,555 raw OOF predictions, 50 calibration fits, 2,799 calibration-training rows, ten sigmoid calibrators, 311 calibrated predictions, 5,000 paired bootstrap draws and 6,531 grouped-SHAP rows were verified. Maximum model, calibrator and SHAP replay error was zero; maximum recorded additivity error was `4.468303814064711e-06`.
- Policy/claim gate: passed. Forbidden primary features are absent from model, SHAP and reason-code evidence; mapped target support is 31/243/37; subgroup/proxy/transport denominators agree; proxy estimation fails closed; transport is infeasible; source tables are manifest-bound. The enclosing package remains provisional/noncanonical with no promotion, frozen claim matrix or release-ready flag.
- Focused Unit 2G/manifest gate: 126 passed, 1 skipped in 22.09 seconds, exit 0 (25.7 seconds wall time).
- Complete pytest gate: 676 passed, 3 skipped and 11 subtests in 100.44 seconds, exit 0 (104.4 seconds wall time).
- Complete unittest gate: 178 tests, 2 skipped in 6.608 seconds, exit 0 (10.6 seconds wall time).
- Compile gate: `compileall -q src tests` exited 0 in 0.3 seconds.
- Scientific rerun, model refit, manuscript edit, network call and paid API call: zero.
- Consolidated manifest/receipt, forbidden-feature, raw/secret, portability, README-link, manuscript-diff, large-file and no-network/no-paid-API hygiene gate: exit 0 in 1.192 seconds; 24 README links and 14 exact checkpoint candidates checked, with zero prohibited findings.
- Exact staged-diff gate: exit 0 in 0.535 seconds; 14 allowlisted files, zero unstaged/untracked/raw/model/environment/large/secret/absolute/manuscript/full-package findings, and `git diff --cached --check` passed. Scientific checkpoint `0e3f50c91693b0a0f22502c2f006d516178b5d88` was created from that index.
- Push is externally blocked, not a scientific/test failure. Attempt one timed out after 124 seconds at Credential Manager and its exact Git-only transport chain was stopped; attempt two exited 128 in 1.3 seconds because no noninteractive HTTPS username was available. Remote stayed `eab2b32`; no process or ref ambiguity remains.

## V2-032 figure-collision and package-contract gate - 2026-07-14

- Recovery/base gate: passed from clean synchronized required branch HEAD `fa6f4b1`; no interrupted process, listener, residue or Unit 2G mutation.
- Final focused gate: `24 passed` in 2.22 seconds, exit 0. It covers exact frozen-plan derivation, current-run identity/source hashes, closed-world stage receipts, PNG/SVG pairs and dimensions, caption/source identity, stale-run rejection, missing/manual/obsolete package rejection, primary-feature exclusion, legacy-generator isolation and atomic OOF-SHAP failure behavior.
- Development failures retained for audit: 21 passed/3 failed, then 23 passed/1 failed. Failures were limited to test regex expectations and correcting the external-primary forbidden-feature test scope; production checks were not weakened.
- Complete pytest: `682 passed, 2 skipped, 11 subtests passed` in 99.13 seconds, exit 0.
- Complete unittest: 176 tests, 1 skipped in 6.587 seconds, exit 0.
- Compileall: exit 0 in 0.122 seconds.
- Scientific experiment, figure generation, Unit 2G rerun, API/network use and manuscript modification: zero.
- Exact 18-file staged hygiene/diff gate: passed; checkpoint `5cd144a757a1a88271e01dd46a738c59a22aef43` created.

## V2-012 supplementary heuristic-search gate - 2026-07-14

- Scope/terminology contract: passed. The stage is supplementary-only and output terminology is limited to heuristic counterfactual-search success with explicit causal/employee-advice/intervention/feasibility prohibitions.
- OOF/training isolation: passed. Models, prototypes, domains and scales use the outer-training partition; case predictions are exact OOF; full execution requires ten fold receipts and complete eligible coverage.
- Budget/scope contract: passed. Four taxonomy-labelled scopes are distinct; restricted 50x2, primary 100x3 and expanded 250x3 budgets filter one shared 750-candidate pool per case/scope; within-scope inclusion is validated and cross-scope monotonicity is prohibited.
- Output/statistical contract: passed. Denominators, failures, candidates, gain, cost and sparsity are explicit; search-success uses Wilson 95% intervals; successful-case numeric summaries use 5,000-draw percentile bootstrap with the conditional estimand named directly.
- Development failures retained: 35 passed/1 failed because a test over-required restricted-budget success; then a bounded real-INX diagnostic failed before output because diagnostic receipt cardinality incorrectly required all ten folds. The test and diagnostic-only cardinality were corrected without weakening full production checks.
- Bounded real-INX diagnostic: exit 0 in 3.818 seconds; two eligible cases, one evaluated fold, 48 case rows, four primary summaries, 12 sensitivity rows, 11 verified inventory entries and the complete 12-file schema. Temporary output was safely removed and is not publication evidence.
- Final focused gate after the last scientific wording change: 11 passed in 0.32 seconds, exit 0.
- Complete pytest: 687 passed, 2 skipped and 11 subtests in 100.87 seconds, exit 0 (104.753 seconds wall time).
- Complete unittest: 176 tests, 1 skipped in 7.188 seconds, exit 0 (11.015 seconds wall time).
- Compileall: exit 0 in 0.110 seconds. Canonical config validation/hash: exit 0, `ff4afa35c0f48ecf052be78af2074a2498bfd5af3697e0f8d863de0cb8952b59`.
- Pre-stage hygiene: exit 0 in 0.818 seconds; 23 candidates, 32 complete unique issues and 24 README links/18 unique tracked targets; zero staged, raw/model/environment, 10-MB, reparse, secret, absolute-path, manuscript, active terminology, network/API-import or legacy normative-term findings.
- Production supplementary artifact, manuscript modification, Unit 2G mutation, network call and paid API call: zero.
- Staged-review provenance finding: corrected. Each OOF/search row now resolves to the actual serialized fold-model SHA-256; summaries/protocol/inventory resolve to the ordered model-set SHA-256. This was a production identity fix, not a relaxed test.
- Post-fix focus: 37 passed in 1.90 seconds, exit 0 (5.395 seconds wall time).
- Post-fix bounded real-INX diagnostic: exit 0 in 3.946 seconds; one actual fold-model hash, one model-set hash, 48 case rows, four OOF rows and 11 inventory entries all mapped/hashed correctly; temporary output removed.
- Final post-fix pytest: 687 passed, 2 skipped and 11 subtests in 118.36 seconds, exit 0 (122.461 seconds wall time).
- Final post-fix unittest: 176 tests, 1 skipped in 7.389 seconds, exit 0 (11.319 seconds wall time). Compileall: exit 0 in 0.114 seconds.
- Exact staged-diff gate: exit 0 in 0.617 seconds; 23 allowlisted files, 1,896 insertions/707 deletions, maximum 118,791 bytes, zero unstaged/untracked/raw/model/environment/10-MB/secret/absolute/manuscript/network/legacy findings, 32 valid issues, 24 README links and 26 model-identity references. Checkpoint `7226effd30835fc678b0bb21644f45ac0464dff6` was created.
- V2-012 receipt/push gate: five-file documentation review passed; receipt `08bc14e` created; normal authenticated push succeeded in one attempt and origin now contains the checkpoint/receipt chain.

## V2-013 supplementary external gate - 2026-07-14

- Real-data preflight: passed for all three retained tasks with exact local receipt hashes, complete target mappings, unique mapped identifiers and minimum class support above ten.
- Protocol/claim gate: passed. Three separate non-comparable strata; no direct primary validation, locked transport or transportability claim; ordinal metrics use literal `N/A`; restricted class 4 is the explicit positive class for binary ranking/probability metrics.
- Nested/OOF/model gate: passed. Exact 10x5 folds, primary-policy-only macro-F1 selection, balanced-accuracy tie-break, same-fold candidate reuse, complete OOF coverage, actual serialized model hashes/paths and zero-error replay.
- Uncertainty gate: passed. Fold summaries are descriptive only and production intervals use 5,000 paired sample-level OOF bootstrap draws separately within each task.
- Atomic/inventory gate: passed. Runner-owned files publish through contained sibling staging; task/stage manifests are closed-world and portable; builder adds its separately declared `stage_contract.json` envelope.
- Final focused suite: 53 passed in 10.42 seconds. Expanded integration: 196 passed, 1 skipped in 32.93 seconds.
- Bounded real-IBM diagnostic: exit 0 in 19.837 seconds; 1,470 samples, 4,410 OOF rows, 30 models, zero replay error, 53 inventory records; explicitly noncanonical/reduced and automatically removed.
- Complete pytest: 690 passed, 2 skipped and 11 subtests in 117.72 seconds, exit 0 (121.8 seconds wall time).
- Unittest: 176 tests, 1 skipped in 7.370 seconds, exit 0 (11.312 seconds wall time). Compileall: exit 0 in 0.112 seconds. Pip dependency check: exit 0 in 0.631 seconds with no broken requirements.
- Staged-diff/hygiene gate: 22 exact allowlisted files; 2,196 insertions/29 deletions; maximum 119,380 bytes; 32 unique issues; 24 README links/18 unique targets; zero staged raw/model/environment/large files, secret/absolute/manuscript/active-legacy/network findings. Existing tracked raw-history blockers remain documented and were not added by V2-013.
- Checkpoint/push gate: commit `906f6360a971833d4cec39fd0d19873b7c567169` created successfully; one normal authenticated push exited 0 and synchronized local/origin.
- No production artifact, network/API call, manuscript edit or Unit 2G mutation occurred.

## V2-015 workbook/CSV provenance gate - 2026-07-14

- Byte/schema contract: passed. Tracked BIFF8 signature and SHA-256 `d7d224e7...`, canonical CSV SHA-256 `b8deac0a...`, portable paths, exact two-sheet order, 1200 x 28 data contract and config/provenance bindings are explicit.
- Live normalized comparison: passed with Excel COM 16.0 in 1.012 seconds. Workbook and CSV shapes are both 1201 x 28 including the header; both normalized content hashes are `b5caa2eec9a46ad184cc452e1d1df01abc80658db7fdd61e2cb8939943e23fbb`; mismatch count is zero.
- Partial codebook gate: passed. Seven exact blocks are hash-bound through a complete explicit mapping, including `RelationshipSatisfaction -> EmpRelationshipSatisfaction`; every mapped target exists. `complete_data_dictionary=false` because coverage is 7/28, and semantic/source/licence authority remains manual.
- Privacy/atomicity gate: passed in tests. Receipts contain no employee values; mismatch evidence contains coordinates/column names only; writes are sibling-temp, flushed/fsynced and atomically replaced; dirty/test-reader output is noncanonical.
- Development failure retained: initial focus reported 20 passed/3 failed because Windows `Path` defaults were not normalized to portable separators. The production fix changed the boundary, not the acceptance criteria.
- Final focused gate before staged review: 23 passed in 2.01 seconds, exit 0 (5.176 seconds wall).
- Provenance/manifest integration: 77 passed in 4.73 seconds, exit 0 (7.814 seconds wall).
- Initial complete pytest: 696 passed, 2 skipped and 11 subtests in 119.04 seconds, exit 0 (123.193 seconds wall).
- Staged-review finding: fixed. Atomic-write cleanup can no longer swallow a secondary unlink failure; the primary exception carries an explicit cleanup note. The new negative test passes.
- Final post-review focus: 24 passed in 1.87 seconds, exit 0 (4.903 seconds wall).
- Final post-review pytest: 697 passed, 2 skipped and 11 subtests in 117.32 seconds, exit 0 (121.237 seconds wall).
- Final post-review unittest: 176 tests, 1 skipped in 7.196 seconds, exit 0 (11.059 seconds wall). Compileall: exit 0 in 0.122 seconds.
- Clean production receipt: intentionally pending the pushed implementation commit; no diagnostic receipt is admitted.
- First production CLI attempt: failed safely before Excel/output because argparse emitted a Windows-backslash default path; no partial receipt exists. The new CLI-default regression test passes.
- Final post-CLI-fix focus: 25 passed in 2.56 seconds, exit 0 (5.541 seconds wall).
- Final post-CLI-fix pytest: 698 passed, 2 skipped and 11 subtests in 117.97 seconds, exit 0 (121.926 seconds wall).
- Final post-CLI-fix unittest: 176 tests, 1 skipped in 7.287 seconds, exit 0 (11.093 seconds wall). Compileall: exit 0 in 0.114 seconds.
- Clean production receipt: passed in 1.408 seconds from Git commit `8f02e5569f1073b1dd3e0861e29d5f9189d79173` using Excel COM 16.0. `canonical_eligible=true`, `git_worktree_dirty=false`, exact sheet order, zero matrix mismatches, no employee values, zero network/API calls and zero residual temp sibling.
- Independent receipt validation: passed in 0.216 seconds. Workbook SHA-256 `d7d224e7...`, CSV SHA-256 `b8deac0a...`, normalized content SHA-256 `b5caa2ee...`, source-tree SHA-256 `d2407e99...`, and receipt SHA-256 `90e75733...` are bound. The codebook remains explicitly incomplete at 7/28 columns.

## V2-014 current-tip/export gate - 2026-07-14

- Local preservation: passed. Fourteen exact files, 2,335,429 total bytes and every configured SHA-256 remained present after index-only removal; all 14 are ignored.
- Current-tip contract: staged state removes exactly those 14 paths while retaining cards, schema mappings, data README and placeholders. No local deletion/history rewrite is performed.
- Export contract: tests pass for exact-commit allowlisting, closed ZIP membership/hashes/sizes, required tracked documentation, forbidden data-path rejection, history/`.git` absence, portable paths/content, secret scan, no symlinks and ephemeral archive cleanup.
- Development failure retained: 19 passed/1 failed because an absolute internal receipt `Path` was joined twice on Windows. Explicit containment plus drive-letter rejection fixed production behavior without weakening the contract.
- Final focused gate: 20 passed in 2.73 seconds, exit 0 (5.950 seconds wall).
- Provenance/path/data integration: 84 passed in 6.04 seconds, exit 0 (9.222 seconds wall).
- Complete pytest: 703 passed, 2 skipped and 11 subtests in 120.78 seconds, exit 0 (124.708 seconds wall).
- Unittest: 176 tests, 1 skipped in 7.336 seconds, exit 0 (11.177 seconds wall). Compileall: exit 0 in 0.114 seconds.
- Implementation checkpoint/push: commit `9342b0c9a02788ff9e9867b13f2f824662fd1cf3`, normal push exit 0; local/origin synchronized.
- Clean history-free export receipt: production exit 0 in 2.24 seconds from clean commit `9342b0c`; `canonical_eligible=true`, 305 members, 1,077,955 archive bytes, 4,005,919 uncompressed bytes, archive SHA-256 `1917059f...`, member-manifest SHA-256 `df89a3a6...`, no retained archive and no raw employee value in the receipt.
- Independent archive rebuild: exit 0 in 1.122 seconds; exact archive hash reproduced. All 14 local files/2,335,429 bytes remain present and ignored. Forbidden tracked/member paths, Git metadata, portable-content findings, secrets, symlinks, network/API calls and residual temporary siblings are zero. Receipt SHA-256 is `3ddad1b7...`.

## V2-018 dependency isolation/lock gate - 2026-07-14

- Static contract: passed; 4 exact groups, 22 unique direct packages, 96 sorted exact pins, one CPython 3.14 baseline, safe relative includes, and zero forbidden legacy/OpenAI packages in core/supplementary.
- Initial focused gate: 79 passed with one Python 3.14 invalid-escape warning in a new negative-test regex. Test literal corrected to a raw string; no production behavior changed.
- Corrected focused gate: 79 passed in 4.69 seconds, exit 0 (7.758 seconds wall), no warnings.
- Resolver checks: core dry run exit 0 in 5.728 seconds; full development dry run exit 0 in 20.953 seconds. No install occurred in either dry run.
- Complete pytest: 711 passed, 2 skipped and 11 subtests in 122.21 seconds, exit 0 (126.185 seconds wall). The preceding wrapper timeout produced no pytest verdict and left no process; it is not counted as a suite run.
- Unittest: 176 tests, 1 skipped in 7.361 seconds, exit 0 (11.192 seconds wall). Compileall: exit 0 in 0.114 seconds.
- Fresh isolated core install: passed. CPython 3.14.0, 31 locked non-bootstrap distributions, all 13 direct imports, `pip check` passed, production environment validator passed, inventory SHA-256 `a7ac622b...`.
- Final review finding: fixed. Development validation now evaluates the exact Windows-only `pywin32` marker instead of requiring that package on Linux; arbitrary direct markers are rejected.
- Post-marker focused gate: 80 passed in 5.07 seconds, exit 0 (8.245 seconds wall).
- Final post-fix pytest: 712 passed, 2 skipped and 11 subtests in 122.06 seconds, exit 0 (126.148 seconds wall).
- Final post-fix unittest: 176 tests, 1 skipped in 7.303 seconds, exit 0 (11.100 seconds wall). Compileall: exit 0 in 0.122 seconds.
- Implementation checkpoint/push: commit `498e8ad59166f275d120f78ce133cce122961f13`, normal push exit 0; local/origin synchronized.
- Clean exact-commit dependency receipt: production exit 0 in 0.685 seconds; canonical-eligible core profile, source tree `a2b361b8...`, config `65519abf...`, lock `482cbf32...`, 31 distributions, inventory `a7ac622b...`, 13 direct versions and zero missing/unlocked/mismatched/core-forbidden packages.
- Independent receipt validation: exit 0 in 0.218 seconds; receipt SHA-256 `8b77e727...`, size 2,476 bytes, zero atomic temp siblings.
- Temporary environment cleanup: passed after the first safety scan self-matched and refused deletion. Exact contained run path had no external process owner and was removed; zero matching temp directories remain.

## V2-020 complete-build no-network/API gate - 2026-07-14

- Runtime contract: process-wide socket/DNS/connect/send/listener denial; caught-attempt poisoning; six API credential variables cleared/restored; shell/non-Git/remote-Git child rejection; exact local read-only Git allowlist; zero-attempt package-status contract.
- Initial focus: 92 passed in 5.41 seconds, exit 0 (8.595 seconds wall).
- Post-Git-allowlist focus: 93 passed in 4.65 seconds, exit 0 (7.677 seconds wall).
- First complete pytest: 724 passed, 1 failed, 2 skipped and 11 subtests in 123.57 seconds. Failure was an existing source-introspection test looking for stage code in the new public wrapper.
- Compatibility fix: test inspects `_build_impl`, where command persistence still occurs; production code unchanged.
- Post-fix focus: 107 passed in 3.35 seconds, exit 0 (6.329 seconds wall).
- Final pytest: 725 passed, 2 skipped and 11 subtests in 122.38 seconds, exit 0 (126.377 seconds wall).
- Unittest: 176 tests, 1 skipped in 7.563 seconds, exit 0 (11.357 seconds wall). Compileall: exit 0 in 0.115 seconds.
- Real release-blocked core entrypoint: expected `ManuscriptBuildError`, exit 0 diagnostic in 0.139 seconds; credential environment restored and no run created.

## V2-019 CI/release-candidate gate - 2026-07-14

- Workflow contract: both YAML files parse; read-only `contents` permission, credential-free checkout, CPython 3.14, exact development install, dependency/`pip check`, six cleared API credential variables, offline-runtime focus, pytest, unittest, compileall, manuscript diff and production repository gate are present.
- Manual release contract: exact run ID/40-character commit/`VALIDATE_ONLY` inputs; labelled trusted Windows self-hosted runner; ignored local scientific inputs preserved; read-only two-scope validation; no artifact upload, promotion, push, release, Zenodo or build command.
- Dependency environment: disposable CPython 3.14 install exit 0 in 86.8 seconds; 22 declared direct packages, 96 exact pins; `pip check` passed.
- Final focused gate: 35 passed in 1.42 seconds, exit 0 (4.570 seconds wall).
- Complete-cycle failure 1: broad config-loading cascade because PyYAML resolved JSON exponent scalars as strings. A lexical config change proved the diagnosis but did not cover JSON-dumped fixtures.
- Root fix: `load_config` parses JSON-compatible content with `json.loads` before YAML fallback; a dedicated numeric-exponent regression was added and original canonical config bytes were restored.
- Failure-focused verification: 122 passed in 12.01 seconds, exit 0 (15.217 seconds wall).
- Final pytest: 730 passed, 2 skipped and 11 subtests in 122.16 seconds, exit 0 (126.145 seconds wall).
- Final unittest: 177 tests, 1 skipped in 8.178 seconds, exit 0 (11.861 seconds wall).
- Compileall: exit 0 in 0.104 seconds.
- Hosted status at the initial local checkpoint: pending; superseded by the passed execution receipt below.

### V2-019 hosted execution receipt

- Initial hosted run `29350928894`: failed only at full pytest after every earlier step passed; authenticated job logs showed 22 exact missing-local-input failures for intentionally ignored datasets/workbook.
- Isolation contract: those real-data integrations skip only when their exact ignored input is absent. The trusted release workflow separately requires all five inputs before any test, preventing a candidate validation from silently skipping them.
- Affected local real-data focus: 94 passed in 15.31 seconds, exit 0 (18.662 seconds wall).
- Post-fix local pytest: 730 passed, 2 skipped and 11 subtests in 123.44 seconds, exit 0 (127.411 seconds wall).
- Post-fix local unittest: 177 tests, 1 skipped in 7.945 seconds, exit 0 (11.733 seconds wall). Compileall: exit 0 in 0.103 seconds.
- Green hosted run `29351557672` at commit `700adb235e66927df0ea4bf6b78ec811c9ffc1ee`: all steps passed. Focus 35 in 0.67 seconds; pytest 709 passed, 23 explicit skips and 11 subtests in 104.65 seconds; unittest 177 in 6.516 seconds with 10 skips; compileall and the production repository gate passed.

## V2-021 production core-figure gate - 2026-07-14

- Hosted predecessor receipt: Actions run `29352015443` completed successfully at synchronized commit `ea34ed658c141ad9158cf1543c409718753d2464`.
- Locked disposable environment: corrected exact-lock install exit 0 in 83.4 seconds; `pip check` and pandas/Matplotlib/pytest imports passed. The first command used the wrong lock path and installed nothing.
- Initial combined focus: 38 passed/one failed in 6.70 seconds (11.774 wall); failure was the existing SVG declaration prohibition. Production fix removes `DOCTYPE` and rejects any remaining declaration/entity.
- Corrected combined focus: 39 passed in 5.51 seconds (8.683 wall).
- Real-schema generator focus after SHAP/calibration alignment: 3 passed in 4.44 seconds (7.527 wall).
- Complete pytest one: 732 passed, two skipped, 11 subtests and one generic promotion-fixture failure in 131.27 seconds (135.211 wall). Production validation was not weakened.
- Exact fixture regression: one passed in 0.35 seconds (3.338 wall).
- Final complete pytest: 733 passed, two skipped and 11 subtests in 128.34 seconds (132.292 wall), exit 0.
- Final unittest: 177 tests, one skipped in 7.334 seconds (11.190 wall), exit 0.
- Compileall: exit 0 in 0.121 seconds.
- Config/source contract: exit 0 in 0.177 seconds; config `a866bd6f...`, source tree `fcae16c9...`.
- Retained scientific/figure artifact, real-data refit, Unit 2G rerun, network/API scientific call and manuscript modification: zero.
- Consolidated manifest/config, forbidden-feature, raw/secret, portability, README-link, manuscript-diff, large-file/staging and no-network-import gate: exit 0 in 1.143 seconds; 16 candidates, 32 issues, 29 local links/23 unique targets, maximum 127,236 bytes and zero findings.
## V2-022 authoritative metric and table contracts - 2026-07-14

- Targeted compileall before testing: pass.
- Initial focused pytest: exit 1; 43 passed, 2 failed, 3 errors; 12.74 seconds pytest / 18.635 seconds wall. Root cause: one removed `MappingProxyType` import still required by existing aligned-OOF output.
- Corrected focused pytest: exit 0; 48 passed; 17.75 seconds pytest / 20.906 seconds wall.
- Initial complete pytest before staged review: exit 0; 738 passed, 2 skipped, 11 subtests passed; 134.66 seconds pytest / 138.527 seconds wall.
- Post-review table-validator focus: exit 0; 30 passed; 17.26 seconds pytest / 20.446 seconds wall.
- Final post-review complete pytest: exit 0; 739 passed, 2 skipped, 11 subtests passed; 132.92 seconds pytest / 136.855 seconds wall.
- Final unittest discovery: exit 0; 177 tests, 1 skipped; 7.964 seconds unittest / 11.710 seconds wall.
- Final compileall over `src` and `tests`: exit 0; 0.117 seconds.
- Locked environment: CPython 3.14 exact development constraints; `pip check` clean.
- Config hash: `1bcd1386736a6eb0838ebfc0976682343ac80de684cd7065b5487dfa673060ef`.
- Source-tree hash: `fba949d2088c166ee2afb3df906236d3ef3a31be13a3f85bc60636655063bcf2`.
- Metric-schema hash: `98ae57b622a56983192975c5bff94374ed51322f13c76af633c3c96ceb198cdd`.
- Pre-commit `ci_repository_gate`: expected exit 1 at the clean-worktree precondition; repeat required after the checkpoint commit.
- No manuscript edit, real-data fit, production table/figure artifact, API call, promotion or release occurred.
## V2-023 execution-readiness scope freeze - 2026-07-14

- Focused scope/figure/manifest/builder validation: exit 0; 114 passed; 5.50 seconds pytest / 8.635 seconds wall.
- Initial complete pytest: exit 1; 732 passed, 2 skipped, 11 subtests passed, 9 fixture failures; 131.34 seconds pytest / 135.344 seconds wall.
- Corrected affected focus: exit 0; 31 passed; 2.62 seconds pytest / 5.605 seconds wall.
- Final complete pytest: exit 0; 741 passed, 2 skipped, 11 subtests passed; 132.28 seconds pytest / 136.238 seconds wall.
- Final unittest discovery: exit 0; 177 tests, 1 skipped; 7.937 seconds unittest / 11.797 seconds wall.
- Final compileall: exit 0; 0.117 seconds.
- The nine initial failures were a stale test fixture with `release_ready=false`; no production contract was weakened.
- No scientific build or artifact execution occurred.
## Offline-runtime clean-build defect - 2026-07-14

- Clean core attempt `canonical_v2_20260714T175804Z_4d08ca2`: exit 1 in 1.239 seconds at shared-fold import; zero model fits; failed receipt preserved.
- First fix focus: exit 1; 50 passed/1 test-design failure; the test deliberately poisoned the boundary and incorrectly expected clean exit.
- Corrected offline/scope/manifest focus: exit 0; 51 passed; 4.15 seconds pytest / 7.208 seconds wall.
- Final complete pytest: exit 0; 743 passed, 2 skipped, 11 subtests passed; 134.15 seconds pytest / 137.961 seconds wall.
- Final unittest: exit 0; 177 tests, 1 skipped; 8.087 seconds unittest / 11.967 seconds wall.
- Final compileall: exit 0; 0.114 seconds.
- Fresh-process sklearn import under the active offline boundary: passed.
- Source-tree hash before checkpoint: `868a69aac143fe852a5738dd5a720be5f217243dfcd410d69c4954ca617ca8f7`.

## Registry/report-subset clean-build repair - 2026-07-14

- Failed production attempt: `canonical_v2_20260714T180533Z_770c8d0`, exit 1 after 635.282 seconds; shared folds and benchmark completed, policy fitting never started, package `status=failed`, 71 files/92,337,983 bytes preserved and inadmissible.
- Environment invocation diagnostics: Windows Store `python` failed before interpreter startup; bare CPython 3.14 reported `No module named pytest`; one repository-interpreter invocation named nonexistent tests and collected zero. None is a code-test verdict.
- Corrected focused pytest: exit 0; 43 passed; 4.99 seconds pytest / 8.272 seconds wall.
- Complete pytest: exit 0; 747 passed, 2 skipped, 11 subtests passed; 140.55 seconds pytest / 144.800 seconds wall.
- Unittest discovery: exit 0; 179 tests, 1 skipped; 7.552 seconds unittest / 11.531 seconds wall.
- Compileall over `src` and `tests`: exit 0; 0.125 seconds.
- Config hash: `51415c2ce68c89d9ce2b042b0a7a811fe3e98180b72460006f9d0465f6bf49b7`.
- Pre-checkpoint source-tree hash: `0285772ef5941d55837750edeaa8e688754ac2dbb5fec5a0954fa35917468096`.
- Metric-registry hash: `98ae57b622a56983192975c5bff94374ed51322f13c76af633c3c96ceb198cdd`.

## Pre-migration complete candidate and alias-migration validation - 2026-07-14

- Clean repository gate at `35b121b`: exit 0; 1,947 tracked files; 32 issue rows; 30 README links; zero raw/environment/large/secret/machine-path findings.
- Core build `canonical_v2_20260714T182539Z_35b121b`: exit 0; 1,614.358 seconds; 354 files/224,172,392 bytes at immediate scope completion.
- Supplementary build under the same run: exit 0; 1,862.575 seconds.
- Strict two-scope read-only release validation: exit 0; 5.673 seconds; `status=valid`.
- Identity: commit `35b121bc9e541cf2736cfe0fc2912e327d86dbd2`; config `51415c2c...`; source `0285772e...`; core/supplementary scientific inputs `34a5e692...`/`e2b41f9f...`.
- Historical alias comparison: 215 common files exact; zero missing named-run files; zero content drift; one alias-only 139-byte pointer; zero reparse points.
- Preserved named v1 run: 215 tracked files, 166,264,643 bytes, inventory SHA-256 `ea0237497fd9bf38057e6303c0ed3c60b5a4fc12be512847e86aed19dc427db9` after alias removal.
- Promotion status: intentionally pending. The valid candidate predates the required migration commit and strict Git identity prohibits relabelling it.

## Atomic-directory publication repair - 2026-07-14

- Failed run `canonical_v2_20260714T192934Z_0c2868b`: exit 1 in 1,066.336 seconds; calibration staging rename WinError 5 followed by cleanup WinError 32; manifest failed; 93 files/105,400,103 bytes plus staging preserved.
- Initial atomic/stage focus: exit 1; 145 passed, 1 skipped, 2 stale literal-`os.replace` assertion failures.
- Corrected atomic/stage focus: exit 0; 147 passed, 1 skipped; 19.22 seconds pytest / 22.625 seconds wall.
- Complete pytest one: exit 1; 750 passed, 2 skipped, 11 subtests passed; one stale minimum tracked-file-count assertion after verified alias migration.
- Corrected inventory assertion: direct test exit 0; requires no physical `latest` and exactly 215 named v1 files.
- Final complete pytest: exit 0; 751 passed, 2 skipped, 11 subtests passed; 135.02 seconds pytest / 139.048 seconds wall.
- Final unittest: exit 0; 179 tests, 1 skipped; 7.683 seconds unittest / 11.742 seconds wall.
- Final compileall: exit 0; 0.124 seconds.
- Config hash: `51415c2ce68c89d9ce2b042b0a7a811fe3e98180b72460006f9d0465f6bf49b7`.
- Pre-checkpoint source-tree hash: `e5527c99d36bf230c79fb49b89a1a2619961aab3ea10718d042ee19361877ad7`.
- Post-review cleanup-preservation focus: exit 0; 148 passed, 1 skipped; 19.37 seconds pytest / 22.846 seconds wall.
- Final post-review pytest: exit 0; 752 passed, 2 skipped, 11 subtests passed; 134.51 seconds pytest / 138.459 seconds wall.
- Final post-review unittest: exit 0; 179 tests, 1 skipped; 7.670 seconds unittest / 11.659 seconds wall.
- Final post-review compileall: exit 0; 0.124 seconds.

## Compact manuscript-support asset export - 2026-07-16

- Direct canonical closed-world/source identity validation: exit 0; 1.643 seconds; 539 manifest records, 545 physical files, 446,595,486 physical bytes, exact pointer/receipt/config/source/scientific identities.
- Existing current-HEAD strict CLI: expected fail-closed generation-commit mismatch after documentation-only HEAD advancement; no scientific rerun or canonical mutation.
- Final export: exit 0; 7.343 seconds; atomic publication under `manuscript/mdpi_information/assets/`.
- Publication validator: exit 0; 109 files, 10,338,351 bytes, seven main and three supplementary PNG/SVG pairs, eight main tables, manifest SHA-256 `fbe7355b956df01ad9817f27b42dc13c0f3e0e33e7f0e5c42a2477beb9d001e1`.
- Export/source parity: exit 0; 308 bindings across 108 manifest-declared files; 1.845 seconds.
- Initial focus: exit 1; 15 passed, 1 skipped, 2 test-schema failures; 3.72 seconds pytest/6.9 seconds wall. Production package remained valid.
- Corrected focus: exit 0; 17 passed, 1 skipped; 4.15 seconds pytest/7.357 seconds wall.
- Complete pytest: exit 0; 758 passed, 2 skipped, 11 subtests passed; 129.91 seconds pytest/133.988 seconds wall.
- Unittest discovery: exit 0; 179 tests, 1 skipped; 8.214 seconds unittest/12.246 seconds wall.
- Compileall over `src`, `tests`, and `tools`: exit 0; 0.312 seconds.
- Candidate hygiene: exit 0; 109 assets/10,338,351 bytes; maximum file 2,195,799 bytes; no figure at or above 10 MiB; zero forbidden binary/row-level extensions, staging/temp files, machine paths, secrets, active `leakage-safe` terms, forbidden primary SHAP features, manuscript changes, or committed/worktree scientific-source changes.
- README audit: exit 0; 43 local links resolved. `git diff --check`: exit 0 with informational Windows LF-to-CRLF warnings only.
- Tracking regression: compact assets unignored; complete canonical root, raw data, model, OOF, and local-SHAP representatives ignored.
- Manuscript, DOCX, PDF, LaTeX, scientific configuration, scientific source, canonical artifacts, fits, OOF, bootstrap, calibration, and SHAP recomputation: zero changes/executions.
