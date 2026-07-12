# Commands and Tests

Date: 2026-07-13

## Git and Repository Baseline

```powershell
git branch --show-current
git rev-parse HEAD
git status --short
git remote -v
git rev-list --left-right --count HEAD...origin/main
git ls-files
git ls-tree -r -l HEAD
git count-objects -vH
```

Result: `main`; HEAD `1c7c343bda401629a3619f92267384916f0708d0`; origin synchronized; initial worktree clean; 1,850 tracked files; about 601 MiB tracked content.

## Baseline Pytest

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q
```

Exit status: 0. Result: 188 passed and 4 subtests passed in 14.75 seconds.

## Baseline Unittest

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
```

Exit status: 0. Result: 161 tests passed in 4.764 seconds.

## Compile

```powershell
.\myenv\Scripts\python.exe -m compileall -q src tests
```

Exit status: 0.

## Actual Input Audit

The raw and interim INX files were loaded with the repository CSV parser, hashed with SHA-256, and compared. Raw: `b8deac0...`; interim: `afc97e8...`; direct frame equality false because the identifier name differs by BOM normalization. A normalized cell-level comparison found identical scientific cells. `load_validated_or_raw_data()` still selects interim because it exists, so the manifest/cache binding remains false.

A read-only Excel COM audit compared the first INX workbook sheet against the normalized CSV and found 1,200 x 28 identical cells and zero duplicate identifiers. This was not a repository script and is not accepted as the final reproducible equivalence test.

## Path and Artifact Audit

```powershell
rg -n 'C:(\\\\|\\|/)+Users|/home/' reports/manuscript_final/latest
```

Result: multiple absolute-user-path hits in run manifest, stage metadata, LLM evidence and chatbot suites.

## CI Audit

- `.github/workflows`: absent.
- GitHub connector query for workflow runs at HEAD: zero runs.
- Local `gh`: unavailable.

## Worktree Safety

The worktree remained clean after baseline tests. Only finalization documentation was added after the read-only audit completed. No manuscript or scientific code file was changed.

## Unit 1A — Pinned Real-Data Preflight

```powershell
.\myenv\Scripts\python.exe -m src.data.canonical_loader --config configs/manuscript_final.yaml --datasets all
```

Exit status: 0. Verified five logical tasks against four local physical files. Actual SHA-256, ordered schema, row/column counts and target distributions all matched. `acquisition_method=existing_local_file` for every task; automatic download remained false.

## Unit 1A — Focused Tests

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_actual_input_hash_binding.py tests/test_no_implicit_interim_fallback.py tests/test_data_acquisition_preflight.py
```

Exit status: 0. Result: 11 passed in 1.26 seconds.

Coverage includes configured-path precedence, actual receipt hash, config/acquisition path consistency, no implicit interim fallback, undeclared dataset failure, unapproved missing-data failure, approved temporary download validation, mismatch quarantine/reporting and ordered-schema drift.

## Unit 1A — Full Regression Gate

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git diff --quiet -- manuscript/mdpi_information/main.md
```

Results: pytest 199 passed plus 4 subtests in 12.47 seconds; unittest 161 passed in 4.444 seconds; compileall exit 0; diff check exit 0; manuscript unchanged. No paid service or real dataset download was attempted.

## Unit 1A Checkpoint

```powershell
git commit -m "fix(data): bind canonical inputs to pinned acquisition contract"
```

Commit: `f4e2dd7`. The staged diff check printed five Markdown trailing-whitespace findings, but the command chain did not fail closed before commit. The defect was recorded and corrected without amend/history rewriting. Future commit gates must use explicit conditional execution rather than semicolon-separated commands.

## Unit 1B — Manifest, Side-Input and Cache Binding

Real dataset/side-input manifest preflight:

```powershell
.\myenv\Scripts\python.exe -c "from src.governance.manuscript_contract import create_run_manifest,validate_run_manifest; m=create_run_manifest('configs/manuscript_final.yaml',run_id='unit1b_real_preflight'); validate_run_manifest(m); print({'schema':m['manifest_schema_version'],'datasets':len(m['actual_input_receipts']),'side_inputs':len(m['side_input_hashes']),'scientific_input_hash':m['scientific_input_hash'],'dirty':m['git_worktree_dirty']})"
```

Exit status: 0. Result: manifest schema 2, five logical dataset receipts, seven side inputs, aggregate scientific-input hash `38dfd51794a837c09cfb67a16eac283d5dc568c94bfa7043a7f5cd14ad6f3b67`. `dirty=true` is expected for this implementation preflight and is not accepted as a final release run.

Focused manifest/cache/snapshot contract:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_artifact_run_manifest_consistency.py tests/test_side_input_hash_binding.py tests/test_cache_invalidates_on_scientific_input_change.py
```

Exit status: 0. Result: 21 passed in 3.09 seconds.

Focused external explicit-input contract:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_external_explicit_input_binding.py tests/test_manuscript_external_evidence.py tests/test_external_validation.py tests/test_external_claim_boundaries.py
```

Exit status: 0. Result: 23 passed in 7.40 seconds.

Full regression gates:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --porcelain -- manuscript/mdpi_information/main.md
```

Results: pytest 218 passed plus 4 subtests in 14.58 seconds; unittest 161 passed in 4.789 seconds; compileall exit 0; diff check exit 0; manuscript status empty.

One hygiene wrapper incorrectly used the PowerShell expression `if (git diff --quiet ...)`, which interpreted the command's empty stdout as false and reported a nonexistent manuscript change. Direct `git status --porcelain -- manuscript/mdpi_information/main.md` confirmed no change; the corrected fail-closed wrapper then passed. This was a command-wrapper defect, not a repository or manuscript failure.

Network/API result: all paid-service environment variables were cleared; no dataset was missing, no approved acquisition URL was invoked, and no API/network call occurred.

The first Unit 1B commit attempt was correctly blocked before commit by `git diff --cached --check` because a new test file had one blank line at EOF. The whitespace-only defect was removed with no history rewrite; the scientific code and test outcomes were unaffected.
