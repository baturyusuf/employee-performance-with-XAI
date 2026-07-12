# Persistent Test Log

Date: 2026-07-13

- Pytest: 188 passed plus 4 subtests, exit 0, 14.75 seconds
- Unittest: 161 passed, exit 0, 4.764 seconds
- Compileall: exit 0
- API keys were removed from the subprocess environment
- Worktree remained clean after tests

Important: baseline green tests do not establish scientific readiness. They do not cover the confirmed v2 issues in `../finalization_v2/02_issue_register.csv`. No v2 real-data integration test has run.

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
