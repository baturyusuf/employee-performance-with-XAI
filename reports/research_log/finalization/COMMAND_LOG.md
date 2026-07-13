# Persistent Command Log

Authoritative detailed baseline commands and results: `../finalization_v2/06_commands_and_tests.md`.

Last exact scientific validation commands:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
```

No canonical v2 build has been run.

Unit 1A exact commands and results are recorded in `../finalization_v2/06_commands_and_tests.md`. The real-data preflight used local files only and passed for all five logical tasks. Full pytest now passes 199 tests plus 4 subtests.

Unit 1B exact commands and results are recorded in `../finalization_v2/06_commands_and_tests.md`. A schema-v2 in-memory manifest was created and validated against all five real logical datasets and seven side inputs; no scientific stage was run. Full pytest now passes 218 tests plus 4 subtests. No network/API call occurred.

Unit 2A exact commands and results are recorded in `../finalization_v2/06_commands_and_tests.md`. Scoped schema-v3 manifests validated real inputs for core and supplementary independently. Accepted entrypoints are fail-closed while release readiness is false. Full pytest now passes 250 tests plus 4 subtests with 2 historical skips; no scientific stage or network/API call ran.

Historical Unit 2B commands and results are recorded in `../finalization_v2/06_commands_and_tests.md`. Shared-fold/model/bootstrap contracts passed 83 focused tests; full pytest passed 314 tests plus 4 subtests with 2 skips; unittest passed 162 with 2 skips. A real-input in-memory 10×3 fold preflight passed for 1,200 rows without writing artifacts or fitting models. At that checkpoint the primary metric was pending; both the 10×3 protocol and pending-metric condition are superseded by the accepted correction below. No network/API call occurred.

The corrected Unit 2B 10×5 protocol/trial commands are recorded in `../finalization_v2/06_commands_and_tests.md`. Focused tests passed 104; full pytest passed 343 plus 4 subtests with 2 skips; unittest passed 162 with 2 skips. The approved 10×5 real-input preflight passed without output/model fitting. The earlier 10×3 record is superseded and cannot be reused.

The final pre-run hardening commands and exact in-memory preflight are recorded in `../finalization_v2/06_commands_and_tests.md`. Focused benchmark/trial tests passed 41; full pytest passed 345 plus 4 subtests with 2 skips; unittest passed 162 with 2 skips; compileall/diff/manuscript/secret checks passed. The real trial entrypoint has not yet been executed and no gate outcome exists.

The real offline trial command, manifest verifier, diagnostic commands, runtime, hashes and model results are recorded in `../finalization_v2/06_commands_and_tests.md`. Trial exit was 0 after 725.2 shell seconds; manifest elapsed time was 722.522 seconds. The macro-F1 baseline gate did not trigger. No API/network/manuscript operation occurred.

Unit 2C-0 exact warning-hygiene commands are recorded in `../finalization_v2/06_commands_and_tests.md`. Focused tests passed 63; full pytest passed 350 plus 4 subtests with 2 skips; unittest passed 162 with 2 skips. A two-model real fold-1 replay produced zero label mismatches and zero warnings. No artifact/API/network/manuscript operation occurred.
