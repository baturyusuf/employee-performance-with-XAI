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

## Unit 2B 10×5 Correction and Trial Preflight

Accepted contract: macro-F1 primary selection/gate; QWK secondary tie-break only within an inclusive absolute `0.001` macro-F1 pool; 10 outer × 5 inner folds; gate requires positive point estimate and paired OOF 95% CI lower bound above zero.

Focused corrected protocol/trial/input suite:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
.\myenv\Scripts\python.exe -m pytest -q tests/test_model_benchmark_trial_entrypoint.py tests/test_manuscript_model_benchmark.py tests/test_canonical_model_factory.py tests/test_shared_fold_artifact_contract.py tests/test_shared_fold_assignments_across_models_and_policies.py tests/test_nested_search_outer_test_isolation.py tests/test_bootstrap_is_stratified_paired_and_deterministic.py tests/test_oof_bootstrap_intervals_are_domain_valid.py tests/test_paired_model_difference_bootstrap.py tests/test_paired_model_gate_contract.py tests/test_stage_runner_scientific_input_binding.py tests/test_actual_input_hash_binding.py tests/test_side_input_hash_binding.py
```

Exit status: 0. Result: 104 passed in 47.36 seconds.

Full gates:

```powershell
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --porcelain -- manuscript/mdpi_information/main.md
```

Results: pytest 343 passed, 2 skipped and 4 subtests in 64.84 seconds; unittest 162 passed with 2 skips in 3.334 seconds; compileall/diff/manuscript gates passed.

The verified-INX 10×5 in-memory preflight invoked the canonical loader, scoped manifest creation, `generate_shared_folds` using config counts/seeds and `validate_shared_folds` without writing. Exit 0: 1,200 rows; target 2/3/4 support 194/874/132; 10 outer folds; 10,800 nested rows; five inner folds per outer fold; inner validation size 216. INX SHA-256 `b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a`; config hash `87198ca980d867a4889ee402fd5979fdae0d6a5563c8390b863295fdd66161e5`; diagnostic dirty scientific-input hash `3434668a5a7778ce0e70095fe8cbf3c39718d82c44d4604f16f5c6deced8fc58`. No artifact/model was created.

During review, the semantic verifier initially referenced a nonexistent `outer_fold_assignments.csv`; production writes `fold_assignments.csv`. This was caught before the expensive run, replaced with `OUTER_ASSIGNMENT_FILENAME`, and covered by the real filename fixture. A process-local socket/DNS denial was then added and tested. No network/API call or manuscript edit occurred.

## Unit 2B Final Pre-Run Hardening

Focused command:

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_model_benchmark_trial_entrypoint.py tests/test_manuscript_model_benchmark.py
```

Exit 0: 41 passed in 35.99 seconds.

Full gates:

```powershell
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --short -- manuscript
rg --pcre2 -n '(?<![A-Za-z])sk-(?:proj-)?[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}' --glob '!myenv/**' --glob '!reports/manuscript_final/**' --glob '!*.ipynb' .
```

Results: pytest 345 passed, 2 skipped and 4 subtests in 64.70 seconds; unittest 162 passed with 2 skips in 3.244 seconds; compileall/diff/manuscript gates passed; the secret expression returned no match. The `rg` no-match exit is the expected passing condition.

Exact fresh real-input in-memory preflight command:

```powershell
$env:OPENAI_API_KEY=$null
$env:OPENAI_AGENTS_API_KEY=$null
$env:AZURE_OPENAI_API_KEY=$null
@'
import json
from src.data.canonical_loader import load_canonical_dataset
from src.experiments.shared_folds import generate_shared_folds, validate_shared_folds
from src.governance.manuscript_contract import create_run_manifest, load_manuscript_config, manuscript_settings
config_path = "configs/manuscript_final.yaml"
config = load_manuscript_config(config_path)
settings = manuscript_settings(config)
manifest = create_run_manifest(config_path, evidence_scope="core", run_id="dirty-preflight-10x5", allow_dataset_download=False)
loaded = load_canonical_dataset(config_path, "inx_primary")
receipt = manifest["actual_input_receipts"]["inx_primary"]
folds = generate_shared_folds(
    loaded.frame,
    target_column=settings["target"]["column"],
    id_column=settings["governance_fields"]["identifier_fields"][0],
    run_id=manifest["run_id"],
    config_hash=manifest["config_hash"],
    scientific_input_hash=manifest["scientific_input_hash"],
    dataset_key="inx_primary",
    dataset_sha256=receipt["actual_sha256"],
    outer_splits=settings["evaluation"]["cv"]["n_splits"],
    inner_splits=settings["model"]["nested_tuning"]["inner_splits"],
    seed=settings["seeds"][settings["evaluation"]["cv"]["seed"]],
    inner_seed=settings["seeds"][settings["model"]["nested_tuning"]["inner_seed"]],
)
validate_shared_folds(folds)
print(json.dumps({
    "artifact_written": False,
    "git_worktree_dirty": manifest["git_worktree_dirty"],
    "config_hash": manifest["config_hash"],
    "scientific_input_hash": manifest["scientific_input_hash"],
    "inx_sha256": receipt["actual_sha256"],
    "rows": len(loaded.frame),
    "target_support": receipt["target_distribution"],
    "outer_rows": len(folds.outer_assignments),
    "outer_folds": folds.contract["outer_splits"],
    "inner_rows": len(folds.inner_assignments),
    "inner_folds": folds.contract["inner_splits"],
    "inner_validation_sizes": sorted({row["n_inner_validation"] for row in folds.contract["inner_fold_summaries"]}),
    "fold_contract_hash": folds.contract["fold_contract_hash"],
    "joblib_version": manifest["code_package_versions"]["joblib"],
    "threadpoolctl_version": manifest["code_package_versions"]["threadpoolctl"],
}, sort_keys=True))
'@ | .\myenv\Scripts\python.exe -
```

Exit 0 in 3.5 seconds. Result: pinned INX SHA-256 `b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a`; 1,200 rows; support 194/874/132; 10 outer folds; 10,800 inner assignments; five inner folds; validation size 216; `joblib 1.5.3`; `threadpoolctl 3.6.0`. Dirty diagnostic config hash `7e70bf6646a542ad32e10ab3718654aa8232a46e44e2083ed10e2cfe526da595`, scientific-input hash `8be7c5d79f2b39af3e04f1c8a14a0ae70d2180c48c425595f69f23ac2e76b34a`, and fold hash `f133710d7c45283951f0b5f36e9f87e273bdeb1ed22a317c65fc34cc32a48373` are engineering diagnostics only and must not be cited/reused.

The command `python -m src.experiments.run_model_benchmark_trial` was **not** executed in this checkpoint. `reports/manuscript_final/trials/` did not exist; no real model, trial artifact, gate result, API call or network call occurred. Manuscript status remained empty.
