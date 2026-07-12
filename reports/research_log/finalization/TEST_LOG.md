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
