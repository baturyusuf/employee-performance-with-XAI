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
