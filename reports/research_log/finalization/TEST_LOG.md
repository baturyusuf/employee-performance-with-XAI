# Persistent Test Log

Date: 2026-07-13

- Pytest: 188 passed plus 4 subtests, exit 0, 14.75 seconds
- Unittest: 161 passed, exit 0, 4.764 seconds
- Compileall: exit 0
- API keys were removed from the subprocess environment
- Worktree remained clean after tests

Important: baseline green tests do not establish scientific readiness. They do not cover every confirmed v2 issue in `../finalization_v2/02_issue_register.csv`. Real-input preflights have run, but no v2 real-model benchmark or full scientific integration run had completed at this baseline.

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
