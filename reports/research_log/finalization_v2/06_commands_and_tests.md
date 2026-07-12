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

## Unit 2A — Core/Supplementary Scope Isolation

Scoped real-input manifest preflight:

```powershell
.\myenv\Scripts\python.exe -c "from src.governance.manuscript_contract import create_run_manifest,validate_run_manifest; [(lambda m,s: (validate_run_manifest(m,expected_evidence_scope=s), print(s, sorted(m['actual_input_receipts']), sorted(m['side_input_hashes']), m['scope_contract_hash'], m['scientific_input_hash'])))(create_run_manifest('configs/manuscript_final.yaml',evidence_scope=s,run_id='unit2a_'+s),s) for s in ('core','supplementary')]"
```

Exit status: 0. Core receipts: `hrdataset_v14`, `inx_primary`; five scoped side inputs. Supplementary receipts: `employee_turnover`, `ibm_hr_analytics`, `ibm_hr_analytics_attrition`, `inx_primary`; six scoped side inputs.

Fail-closed core entrypoint check:

```powershell
.\myenv\Scripts\python.exe -m src.experiments.build_core_paper_evidence --config configs/manuscript_final.yaml
```

Expected exit status: 1. The command failed before output creation with `Evidence scope 'core' is not release-ready`; it did not execute or simulate a scientific stage.

Focused contracts:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_core_scope_contract.py tests/test_core_build_contains_no_llm_or_chatbot_stage.py tests/test_no_cross_scope_cache_reuse.py tests/test_scoped_run_manifest_inputs.py tests/test_external_scope_contract.py tests/test_external_explicit_input_binding.py tests/test_manuscript_external_evidence.py tests/test_dataset_card_required_fields.py tests/test_final_evidence_manifest_hashes.py tests/test_manuscript_figures_generated.py
```

Exit status: 0. Result: 50 passed, 2 skipped in 7.44 seconds. Skips refer only to the deliberately rejected historical unscoped `latest` package.

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

Final results: pytest 250 passed, 2 skipped and 4 subtests passed in 13.99 seconds; unittest 162 passed with 2 skipped in 3.259 seconds; compileall/diff/manuscript gates passed.

An earlier full pytest attempt produced one failure because the legacy Figures 1-4 test still supplied the core config after LLM settings were removed. The legacy generator was made fail-closed before creating files and its test was corrected to assert exclusion from core. The next full run passed.

Network/API result: credentials were cleared; no network function, approved dataset URL or paid service was invoked. Scientific artifacts generated: none.

The first Unit 2A commit attempt was correctly blocked before commit because the staged diff contained one blank line at EOF in each of two new contract tests. Both whitespace-only defects were removed; no history was created or rewritten and scientific/test behavior was unchanged.

## Unit 2B — Shared Folds, Nested Benchmark and Paired OOF Bootstrap

Focused contract regressions after integration hardening:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_shared_fold_artifact_contract.py tests/test_shared_fold_assignments_across_models_and_policies.py tests/test_nested_search_outer_test_isolation.py tests/test_bootstrap_is_stratified_paired_and_deterministic.py tests/test_oof_bootstrap_intervals_are_domain_valid.py tests/test_paired_model_difference_bootstrap.py tests/test_paired_model_gate_contract.py tests/test_canonical_model_factory.py tests/test_manuscript_model_benchmark.py tests/test_stage_runner_scientific_input_binding.py tests/test_core_scope_contract.py tests/test_core_build_contains_no_llm_or_chatbot_stage.py
```

Exit status: 0. Result: 83 passed in 16.65 seconds.

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

Results: pytest 314 passed, 2 skipped and 4 subtests passed in 31.23 seconds; unittest 162 passed with 2 skipped in 3.345 seconds; compileall/diff/manuscript gates passed.

Real-input fold-contract preflight used the canonical loader and an in-memory scoped manifest, then called `generate_shared_folds(... outer_splits=10, inner_splits=3, seed=42, inner_seed=43)` and `validate_shared_folds` without writing outputs. Exit status: 0. Result: 1,200/1,200 exact outer assignments, 10 outer folds, 10,800 inner assignments, exactly three inner folds for each outer fold, pinned INX SHA-256 `b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a`. The emitted predecision/dirty-tree fold hash is diagnostic only and must not be reused.

One initial focused-test command invoked the broken Windows Store `python.exe` shim and failed before Python started; `py -3.14` then proved that the system interpreter has no pytest. All authoritative tests use `.\myenv\Scripts\python.exe` and passed. During development, an unbound `inner_splits` variable, bootstrap sorting/direction defects and inactive LightGBM row subsampling were detected by focused checks/review and corrected before the full suite. No real model benchmark was executed because the selection/gate metric decision is intentionally pending.

Network/API result: credentials were cleared; no dataset acquisition, network/API call or paid service occurred. Scientific artifacts generated: none. Manuscript edits: none.

The first Unit 2B staged checkpoint gate was blocked by `git diff --cached --check` because two new bootstrap test files had one blank line at EOF. The whitespace-only defects were removed with `apply_patch`; no commit or history mutation occurred before correction.

Checkpoint command `git commit -m "feat(protocol): add shared nested OOF benchmark contract"` exited 0 and created `8e9b5b9b9f66815abf7f9a599535a36737ea1706`. No push, merge, release or history rewrite occurred.
