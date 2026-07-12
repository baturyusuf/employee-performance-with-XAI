# Test Log

## Baseline — 2026-07-12

Environment: `myenv/Scripts/python.exe`, Python 3.14.0, pytest 9.1.1, commit `18347488bdb4eed60f115ceeff70c420071ceef0`, clean worktree.

### Collection

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\myenv\Scripts\python.exe -m pytest --collect-only -q -p no:cacheprovider
```

- Result: 100 tests collected in 14.56 seconds.
- Exit status: 0.

### Pytest Baseline

```powershell
.\myenv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

- Result: 100 passed in 4.67 seconds.
- Exit status: 0.

Independent repeat:

```powershell
.\myenv\Scripts\python.exe -m pytest -q
```

- Result: 100 passed in 4.13 seconds.
- Exit status: 0.

### Unittest Baseline

```powershell
.\myenv\Scripts\python.exe -m unittest discover -s tests -v
```

- Result: Ran 100 tests in 1.367 seconds; OK.
- Exit status: 0.

### Post-Test State

- Git worktree remained clean.
- Baseline failures: none.
- Acceptance-test gap: the manuscript-remediation tests listed in `01_issue_inventory.csv` do not yet exist.

## Phase 2 Implementation — 2026-07-12

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_manuscript_policy_ablation.py
```

- Result: 4 passed in 1.45 seconds.
- Exit status: 0.
- Scope: exact policy application, explicit ID/target exclusions, Holm adjustment, common run/config propagation, paired policy tests, and leakage sensitivity direction.

## Phase 4 Implementation — 2026-07-12

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_manuscript_calibration.py tests/test_manuscript_policy_ablation.py
```

- Result: 7 passed in 1.39 seconds.
- Exit status: 0.
- Scope: calibration-bin coverage including 0/1 boundaries, predeclared method selection, fold uncertainty fields, and Phase 2 regressions.

## Cost-Safety Regression — 2026-07-12

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_llm_runtime_config.py tests/test_chatbot_guardrails.py tests/test_guardrail_suite_minimums_and_categories.py
```

- Result: 30 passed in 0.72 seconds.
- Exit status: 0.
- Usage-ledger SHA-256 before: `ECE79E016DC8745D8D1D2862D4C05C81F2C4B09913540CC55285FE7D3B904A7E`.
- Usage-ledger SHA-256 after: `ECE79E016DC8745D8D1D2862D4C05C81F2C4B09913540CC55285FE7D3B904A7E`.
- Scope: explicit offline defaults even with an API key, mixed-intent routing, versioned suite categories, and Wilson interval outputs.

## Phase 1/2/4/5 Combined — 2026-07-12

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_shap_outputs_match_primary_policy.py tests/test_manuscript_calibration.py tests/test_manuscript_policy_ablation.py tests/test_canonical_feature_policy_consistency.py tests/test_artifact_run_manifest_consistency.py tests/test_forbidden_features_absent_from_primary_artifacts.py
```

- Result: 27 passed and 4 subtests passed in 2.26 seconds.
- Exit status: 0.
- Scope: canonical config/policy, run manifest, forbidden artifacts, common-fold ablation, nested calibration, OOF SHAP contract and representative strata.

## Phase 3 Task/Claim Tests — 2026-07-12

- Focused/relevant task-metric, external-claim, metrics, external-validation, data-audit, config, and package tests: 34 passed.
- Shared full suite before cost-safety remediation: 147 passed and 4 subtests in 53.49 seconds.
- Caveat: that full run helped reveal the auto-provider cost defect and is not the final acceptance run. A post-remediation complete suite remains required.

## Phase 6 Counterfactual Protocol — 2026-07-12

```powershell
.\myenv\Scripts\python.exe -m pytest -q tests/test_counterfactual_protocol_is_oof.py tests/test_counterfactual_denominators_reported.py
```

- Result: 5 passed in 1.45 seconds.
- Exit status: 0.
- Scope: training-prototype/domain isolation, relational constraints, actionability-mode feature controls, explicit total/eligible/valid denominators, failure counts, and non-degenerate Wilson intervals.

## 2026-07-12 — Integration/Guardrail Cost-Safety Focused Suite

- Command: `.\myenv\Scripts\python.exe -m pytest -q tests\test_final_evidence_manifest_hashes.py tests\test_guardrail_suite_minimums_and_categories.py tests\test_chatbot_guardrails.py tests\test_llm_runtime_config.py`
- Result: **32 passed** in 0.84 seconds.
- API safety assertion: `llm_usage_log.csv` SHA-256 before and after was identical (`ECE79E016DC8745D8D1D2862D4C05C81F2C4B09913540CC55285FE7D3B904A7E`).

## 2026-07-12 — Canonical Evidence/Agent Safety Suite

- Command: `.\myenv\Scripts\python.exe -m pytest -q tests\test_deterministic_agent_audit_offline.py tests\test_llm_case_evidence_completeness.py tests\test_real_llm_preflight_blocks_incomplete_evidence.py tests\test_forbidden_features_absent_from_primary_artifacts.py`
- Result: **12 passed plus 4 subtests** in 2.49 seconds.
- API safety assertion: usage-ledger SHA-256 remained unchanged.

## 2026-07-12 — Combined Pre-Run Acceptance Suite

- Scope: all 20 requested/new canonical contract, policy, calibration, SHAP, counterfactual, external, fairness, provenance, CompleteCaseEvidence, deterministic agent, guardrail, figure, and manifest test files.
- Result: **76 passed, 2 skipped, plus 4 subtests** in 10.04 seconds.
- Skip reason: the two tests intentionally wait for the final `latest` package.
- API safety assertion: usage-ledger hash unchanged before/after.
- `git diff --check`: passed.

## 2026-07-12 — README/Git Publication Checks

- README local-link audit: **40/40 targets exist**; zero missing.
- Complete modified/untracked ignore audit: **0/809 paths ignored** (28 modified plus 781 untracked).
- Canonical representative paths checked with `git check-ignore --no-index`: zero matches.
- `.gitignore` diff: none.
- `git diff --check`: exit 0.
- Manuscript diff: none (`git diff --quiet -- manuscript/mdpi_information/main.md` exit 0).
- Scientific tests were not rerun because this handoff changed documentation/log files only; the final canonical scientific results and manifests were not modified.

## 2026-07-12 — SHAP Canonical Data Regression

- Focused test: **5 passed** in 1.46 seconds.
- Real-data stage: complete ten-fold canonical-policy SHAP run passed and generated all required global/class/local/stability and Figure 6/7 outputs.
- Failed-attempt run remained failed and was not promoted to `latest`.

## 2026-07-12 — CompleteCaseEvidence Real-Data Integration Regression

- Selector plus Phase 7 focused tests: **8 passed**, followed by **10 passed** after the explicit external schema-alias fix.
- Real-data deterministic preflight: **80 requested, 80 selected, 80 complete, 0 incomplete**.
- API safety: `real_api_execution_allowed=false`, `api_call_attempted=false`, and usage-ledger hash unchanged.
- Seven-role offline deterministic agent audit: completed for all 80 evidence records.

## 2026-07-12 — Final Canonical Acceptance Gate

- Successful canonical entry point: passed.
- Complete run manifest with live source-tree verification: passed; 214 outputs registered.
- Final evidence manifest: 212/212 files passed in versioned root and `latest`.
- Figure validation: Figures 1–7 present and non-empty in PNG/SVG.
- CompleteCaseEvidence: 80/80 complete; zero incomplete; real API execution false.
- `python -m compileall -q src tests`: exit 0.
- `python -m unittest discover -s tests -q`: **161 passed** in 4.374 seconds.
- `python -m pytest -q`: **188 passed plus 4 subtests** in 12.12 seconds.
- Secret-pattern filename scan: zero hits; workspace `.env` scan: zero files.
- Usage-ledger hash before/after canonical command and final test suite: identical (`ECE79E016DC8745D8D1D2862D4C05C81F2C4B09913540CC55285FE7D3B904A7E`).

## 2026-07-12 — Post-Handoff Pointer Checks

- Figure/final-manifest tests against populated `latest`: **7 passed** in 1.93 seconds; no skips.
- Issue inventory: 32 rows, all IDs unique.
- Manuscript diff: none (`git diff --quiet` exit 0).
- `git diff --check`: passed.
