# Commands and Runs

Commands shown here are exact in intent; all were executed from the repository root with PowerShell on 2026-07-12. Read-only inspection commands using `Get-Content`, `Get-ChildItem`, and `rg` are summarized by group to keep the log reviewable.

## Git Baseline

```powershell
git branch --show-current
git rev-parse HEAD
git status --short --untracked-files=all
git log -1 --format=fuller
git log --oneline --decorate --max-count=12
```

Result: branch `main`, commit `18347488bdb4eed60f115ceeff70c420071ceef0`, clean and synchronized with `origin/main`.

## Repository and Required-File Inspection

```powershell
rg --files configs src tests
Get-Content configs/feature_sets.yaml
Get-Content configs/evaluation.yaml
Get-Content configs/fairness.yaml
Get-Content configs/counterfactuals.yaml
Get-Content configs/llm_agent_eval_openai_final_80.yaml
Get-Content configs/chatbot_guardrail_eval.yaml
```

The required experiment, explainability, governance, LLM, chatbot, report, model-card, decision-log, registry, external-validation, and manifest paths were read. No missing required source path was found.

## Environment

```powershell
.\myenv\Scripts\python.exe --version
.\myenv\Scripts\python.exe -m pip --version
.\myenv\Scripts\python.exe -m pip freeze
.\myenv\Scripts\python.exe -m pip check
.\myenv\Scripts\python.exe -c "import sys,platform; print(sys.version); print(sys.executable); print(platform.platform())"
```

Result: repository Python 3.14.0, pip 25.2, 131 installed distributions, no broken requirements. Bare `python` failed because of a broken Windows Store alias; the repository interpreter succeeded.

## Dataset Inventory

A read-only Python script recursively hashed data files with SHA-256 and loaded CSVs using delimiter detection to record shape and target support. It also compared raw and validated INX data for logical equality.

Result: hashes/shapes/support are recorded in `00_repository_baseline.md`. The historical `.xls` hash was captured, but shape extraction failed because `xlrd` is absent.

## Artifact Inventory

A read-only Python script calculated each file SHA-256 and UTC modification time under the ten required report roots, then calculated a deterministic root snapshot digest from sorted `path|size|mtime|hash` records.

Result: root digests are in `00_repository_baseline.md`; key file details are in `07_artifact_manifest.csv`.

## Existing Manifest Verification

```powershell
.\myenv\Scripts\python.exe -
```

The inline read-only script loaded `reports/manuscript_assets/final_evidence_manifest/final_evidence_manifest.json`, recomputed every referenced SHA-256, and checked file existence.

Result: 29 entries checked; 29 matched; 0 missing; 0 mismatched.

## Provenance Comparison

```powershell
git diff --name-status 6f9df0a85d26bf1519634001357c8f91340a740c..HEAD
git show -s --format='%H %cI %s' 6f9df0a85d26bf1519634001357c8f91340a740c
git show -s --format='%H %cI %s' c703039f0b501c8b61926a8844c9b2992e3e098d
```

Result: June 5 registry provenance points to the pre-code commit while the relevant files were committed afterward, confirming a dirty-tree run.

## Tests

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\myenv\Scripts\python.exe -m pytest --collect-only -q -p no:cacheprovider
.\myenv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\myenv\Scripts\python.exe -m unittest discover -s tests -v
```

Result: 100 collected; pytest 100 passed; unittest 100 passed. See `06_test_log.md`.

## API/External Calls

- Paid OpenAI calls: none.
- Other network calls: none.
- Dependencies installed: none.

## Phase 2 Focused Test

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_manuscript_policy_ablation.py
```

Result: 4 passed in 1.45 seconds. No scientific experiment was run.

## Phase 4 Focused Tests

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_manuscript_calibration.py tests/test_manuscript_policy_ablation.py
```

Result: 7 passed in 1.39 seconds. No scientific experiment was run.

## Unapproved API Side-Effect Audit

The usage ledger was filtered read-only for rows dated 2026-07-12. Result: 24 OpenAI Chat Completions requests, all case 528, from 14:15:19Z to 14:18:12Z; 156,571 total tokens; estimated USD 0.1096371. The exact triggering path was inferred from the Phase 8 runner test: safe prompt evaluation constructed an environment-driven auto explainer.

The following regression command was then run with a before/after SHA-256 guard around the usage ledger:

```powershell
$before=(Get-FileHash reports/llm_explanations/llm_usage_log.csv -Algorithm SHA256).Hash
.\myenv\Scripts\python.exe -m pytest -q tests/test_llm_runtime_config.py tests/test_chatbot_guardrails.py tests/test_guardrail_suite_minimums_and_categories.py
$after=(Get-FileHash reports/llm_explanations/llm_usage_log.csv -Algorithm SHA256).Hash
```

Result: 30 passed in 0.72 seconds. Before and after hashes both `ECE79E016DC8745D8D1D2862D4C05C81F2C4B09913540CC55285FE7D3B904A7E`; no further API usage was logged.

One Phase 1 optional inline validator demonstration exited with a quoting-related `SyntaxError`. It wrote nothing; the same behavior is covered by a passing unit test.

## Phase 1/2/4/5 Combined Focused Tests

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_shap_outputs_match_primary_policy.py tests/test_manuscript_calibration.py tests/test_manuscript_policy_ablation.py tests/test_canonical_feature_policy_consistency.py tests/test_artifact_run_manifest_consistency.py tests/test_forbidden_features_absent_from_primary_artifacts.py
```

Result: 27 passed and 4 subtests passed in 2.26 seconds.

## Phase 3 Validation

Focused task/claim/metrics/external suites passed 34 tests. A shared full-suite run reported 147 passed and 4 subtests in 53.49 seconds, but it also exposed the auto-provider API cost-safety defect; the defect was fixed and its calls are not scientific evidence.

## Phase 6 Tests and Cost Benchmark

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_counterfactual_protocol_is_oof.py tests/test_counterfactual_denominators_reported.py
```

Result: 5 passed in 1.45 seconds.

```powershell
.\myenv\Scripts\python.exe -m src.experiments.manuscript_counterfactual_actionability --config configs/manuscript_final.yaml --output-dir reports/research_log/manuscript_remediation/counterfactual_cost_benchmark --run-id counterfactual_cost_benchmark_20260712 --max-cases 10
```

Result: diagnostic-only benchmark completed. Stage metadata reports 8,317 candidate evaluations in 8.9159 seconds for 10 eligible cases; final population choice remains pending.

## 2026-07-12 — Integration Safety Regression

```powershell
$before=(Get-FileHash reports\llm_explanations\llm_usage_log.csv -Algorithm SHA256).Hash
.\myenv\Scripts\python.exe -m pytest -q tests\test_final_evidence_manifest_hashes.py tests\test_guardrail_suite_minimums_and_categories.py tests\test_chatbot_guardrails.py tests\test_llm_runtime_config.py
$after=(Get-FileHash reports\llm_explanations\llm_usage_log.csv -Algorithm SHA256).Hash
```

Result: 32 passed in 0.84 seconds; usage-ledger SHA-256 remained `ECE79E016DC8745D8D1D2862D4C05C81F2C4B09913540CC55285FE7D3B904A7E`.

```powershell
$before=(Get-FileHash reports\llm_explanations\llm_usage_log.csv -Algorithm SHA256).Hash
.\myenv\Scripts\python.exe -m pytest -q tests\test_deterministic_agent_audit_offline.py tests\test_llm_case_evidence_completeness.py tests\test_real_llm_preflight_blocks_incomplete_evidence.py tests\test_forbidden_features_absent_from_primary_artifacts.py
$after=(Get-FileHash reports\llm_explanations\llm_usage_log.csv -Algorithm SHA256).Hash
```

Result: 12 passed plus 4 subtests in 2.49 seconds; usage-ledger hash unchanged.

```powershell
.\myenv\Scripts\python.exe -m pytest -q <20 canonical acceptance test files>
```

Result: 76 passed, 2 skipped, plus 4 subtests in 10.04 seconds. The two skips are intentional pre-run checks for `reports/manuscript_final/latest`; they must execute after the canonical build. Usage-ledger SHA-256 before/after remained `ECE79E016DC8745D8D1D2862D4C05C81F2C4B09913540CC55285FE7D3B904A7E`.

```powershell
.\myenv\Scripts\python.exe -c "from src.governance.manuscript_contract import canonical_config_hash; print(canonical_config_hash('configs/manuscript_final.yaml'))"
git diff --check
```

Result: config hash `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`; diff check passed.

## 2026-07-12 — Canonical Attempt 1

```powershell
.\myenv\Scripts\python.exe -m src.experiments.build_manuscript_evidence --config configs\manuscript_final.yaml
```

Result: failed safely during SHAP with `ShapEvidenceError: Raw feature-family order changed across SHAP folds`. Run manifest: `reports/manuscript_final/manuscript_final_20260712T175019Z_c664ef152ff3/run_manifest.json`. Usage-ledger SHA-256 was unchanged.

## 2026-07-12 — SHAP Order-Fix Verification

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests\test_shap_outputs_match_primary_policy.py
.\myenv\Scripts\python.exe -m src.experiments.manuscript_shap_evidence --config configs\manuscript_final.yaml --output-dir reports\research_log\manuscript_remediation\shap_order_fix_validation --run-id diagnostic_shap_order_fix --config-hash c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3
```

Result: 5 tests passed; complete ten-fold SHAP diagnostic succeeded in 11 seconds.

## 2026-07-12 — Canonical Attempt 2

```powershell
.\myenv\Scripts\python.exe -m src.experiments.build_manuscript_evidence --config configs\manuscript_final.yaml
```

Result: failed closed at CompleteCaseEvidence identity validation because `representative_cases.csv` lacked run/config columns. All stages through provenance completed; usage-ledger hash remained unchanged. Failed manifest: `reports/manuscript_final/manuscript_final_20260712T175251Z_c664ef152ff3/run_manifest.json`.

## 2026-07-12 — CompleteCaseEvidence Integration Fix Verification

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests\test_shap_outputs_match_primary_policy.py tests\test_manuscript_case_evidence.py
.\myenv\Scripts\python.exe -m src.experiments.manuscript_shap_evidence <failed-run identity, diagnostic output>
.\myenv\Scripts\python.exe -m src.llm.manuscript_case_evidence <diagnostic SHAP plus completed run-local upstream stages>
.\myenv\Scripts\python.exe -m src.agents.manuscript_deterministic_audit <80 complete diagnostic evidence records>
```

Result: selector/LLM tests passed; preflight 80 requested/selected/complete, 0 incomplete, real API disallowed; offline agent audit completed; usage-ledger hash unchanged.

## 2026-07-12 — Successful Canonical Run

```powershell
.\myenv\Scripts\python.exe -m src.experiments.build_manuscript_evidence --config configs\manuscript_final.yaml
```

Result: success. Run ID `manuscript_final_20260712T181754Z_c664ef152ff3`; stable `reports/manuscript_final/latest/`; usage-ledger SHA-256 unchanged.

```powershell
.\myenv\Scripts\python.exe -c "<validate complete run manifest, final evidence manifest, latest mirror, and all seven figures>"
```

Result: run status complete; 214 registered outputs; final manifest 212/212 files verified; latest 212/212 verified; seven PNG/SVG figure pairs verified.

## 2026-07-12 — Final Test and Safety Gate

```powershell
.\myenv\Scripts\python.exe -m compileall -q src tests
.\myenv\Scripts\python.exe -m unittest discover -s tests -q
.\myenv\Scripts\python.exe -m pytest -q
rg -l --hidden <secret patterns> -g '!myenv/**' -g '!.git/**' -g '!reports/**' -g '!data/**' -g '!manuscript/**' .
rg --files --hidden -g '.env' -g '!myenv/**' -g '!.git/**' .
```

Result: compileall exit 0; unittest 161/161; pytest 188 plus 4 subtests; zero secret-pattern file hits; zero workspace `.env` files in scan scope; usage-ledger hash unchanged.

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests\test_manuscript_figures_generated.py tests\test_final_evidence_manifest_hashes.py
.\myenv\Scripts\python.exe -c "<parse and validate the 32-row issue inventory>"
git diff --quiet -- manuscript\mdpi_information\main.md
git diff --check
```

Result: 7/7 latest-package tests passed; issue inventory has 32 unique IDs; corrected native-command exit check confirms no manuscript diff; diff check passed. An earlier PowerShell conditional incorrectly treated an empty native-command output as false and printed a manuscript alert; direct status/diff inspection confirmed it was a command-check error only.

## 2026-07-12 — README and Git-Ignore Publication Audit

```powershell
git status --porcelain=v1 -uall
git check-ignore --no-index -v --stdin
Get-Content -Raw .gitignore
```

Result: after the README update, 28 modified and 781 untracked publication paths were checked; zero matched ignore rules. `.gitignore` was not changed. Ignored content is limited to local virtual-environment files, caches, Python bytecode, IDE state, and environment-secret patterns.

```powershell
# Parse every local Markdown target in README.md and verify it with Test-Path.
git diff --check
git diff --quiet -- .gitignore
git diff --quiet -- manuscript/mdpi_information/main.md
```

Result: 40 README links checked, zero missing; diff check exit 0; `.gitignore` unchanged; manuscript unchanged. The untracked set is approximately 492 MiB, with no individual file at or above GitHub's 100 MiB hard file limit.
