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
