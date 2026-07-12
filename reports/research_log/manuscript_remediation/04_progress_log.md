# Progress Log

## 2026-07-12 — Phase 0 Baseline Audit

- Confirmed clean `main` worktree at commit `18347488bdb4eed60f115ceeff70c420071ceef0`.
- Inspected all required configuration files and required scientific/governance/LLM/chatbot modules.
- Inspected required report roots, model-selection outputs, model card, decision log, experiment registry, and existing evidence manifest.
- Captured interpreter, operating system, installed package versions, dependency gaps, dataset hashes/shapes/support, artifact root snapshots, key artifact hashes, and baseline tests.
- Verified 100 existing pytest tests and 100 unittest tests pass.
- Verified the existing 29-file evidence manifest has no missing files or hash mismatches.
- Confirmed feature-policy split definition, historical Age-containing SHAP, dirty-tree provenance, task-inapplicable binary metrics, in-sample counterfactual evaluation, 30 incomplete LLM cases, limited guardrail strata, support-poor fairness summaries, and incomplete dataset cards.
- Confirmed the final-80 configuration's live dependency on the legacy Age-containing SHAP tree, a mixed-intent chatbot routing bypass, and an incomplete hard-coded LLM forbidden-feature list.
- Created the mandatory remediation traceability files.
- Scientific source code modified: none.
- Manuscript modified: no.
- Paid API calls: none.

## Next Phase

Create the canonical manuscript configuration and run-contract implementation, then add fail-fast policy/provenance validation before rerunning scientific experiments.

## 2026-07-12 — Phase 2 Implementation Started

- Added `src/experiments/manuscript_policy_ablation.py`.
- The stage resolves exact exclusions from the canonical config, creates one fold assignment shared by every policy, runs one XGBoost contract, calculates fold metrics/uncertainty, paired Wilcoxon tests with Holm correction, leakage sensitivity indices, interpretation, manuscript table, and PNG/SVG trade-off figure source.
- Added an explicit audit-only sensitive-retaining contrast so leakage-variable removal can be separated from compound demographic-governance exclusions.
- Added `tests/test_manuscript_policy_ablation.py`; 4 tests pass.
- Scientific results generated: none yet. The stage will run only after the canonical config/run contract is finalized.

## 2026-07-12 — Phase 4 Implementation

- Added `src/experiments/manuscript_calibration.py` for the canonical primary policy.
- Enforced outer stratified 10-fold evaluation, an inner calibration split, raw/sigmoid/isotonic methods, outer-test-only scoring, and ten calibration bins.
- Added fold metrics, OOF probabilities, class-wise bins, uncertainty, predeclared aggregate-rank method selection, explicit probability warning, class reliability PNG/SVG outputs, and coherent six-panel Figure 5 PNG/SVG generation.
- Added `tests/test_manuscript_calibration.py`; combined Phase 2/4 focused suite passes 7 tests.
- Scientific results generated: none yet; run awaits canonical config/run integration.

## 2026-07-12 — API Cost-Safety Incident and Remediation

- Detected 24 newly appended real OpenAI Chat Completions rows during deterministic guardrail test work, all for case 528.
- Preserved the rows in `reports/llm_explanations/llm_usage_log.csv`; they are not accepted as scientific evidence.
- Ledger totals: 136,392 input tokens; 123,648 cached input tokens; 20,179 output tokens; 156,571 total tokens; estimated USD 0.1096371.
- Root cause: environment-driven `auto` provider treated machine-key presence as permission, and safe guardrail prompts invoked the default governed explainer.
- Patched `GovernedExplainer` and `build_llm_client` so default/auto execution is offline unless real execution is explicitly required.
- Ran 30 runtime/chatbot/guardrail tests while hashing the usage log before and after; all passed and the hash remained unchanged.
- No further paid execution is authorized.

## 2026-07-12 — Phase 1 Canonical Contract

- Added `configs/manuscript_final.yaml` with one exact primary policy and the required dataset roles, task schemas, CV/calibration/SHAP/counterfactual/LLM/chatbot/figure/provenance contracts.
- Canonical semantic config hash: `738ec00ae6d2b64494c8e00f74fa3a6e32afd4d4afbff54c10900020287358fc`.
- Added `src/governance/manuscript_contract.py` with semantic config hashing, dataset/package/source provenance, run ID/manifest creation, atomic writes, command/failure recording, artifact registration, and compatibility validation.
- Added the three Phase 1 acceptance-test files. Combined Phase 1/2/4/5 focused suite: 27 passed plus 4 subtests.
- No scientific experiment was run during Phase 1.

## 2026-07-12 — Phase 3 Task and Claim Boundaries

- Added `src/models/task_schema.py` and made `classification_metrics` task-aware.
- Binary and restricted-target tasks now return N/A for inapplicable ordinal metrics; binary Brier, ROC-AUC, and average precision are explicit.
- Added centralized external claim boundaries and task-separated report generation; HRDataset uses independent external performance-target replication.
- Added `test_task_metric_applicability.py` and `test_external_claim_boundaries.py`.
- Canonical external scientific reports have not yet been regenerated.

## 2026-07-12 — Phase 5 OOF SHAP Implementation

- Added `src/experiments/manuscript_shap_evidence.py`.
- It produces OOF grouped SHAP for every INX case, global/class tables, fold rankings, pairwise Jaccard/Spearman stability, representative strata, per-case JSON/CSV/Markdown reason codes, governance metadata, artifact-policy validation, and Figures 6/7 PNG+SVG.
- Added `test_shap_outputs_match_primary_policy.py`; combined Phase 1/2/4/5 suite passes 27 tests plus 4 subtests.
- Scientific SHAP results are pending the canonical run.

## 2026-07-12 — Phase 8 Deterministic Guardrails

- Added versioned prompt suite v2.0.0 with 80 unsafe and 34 safe prompts while retaining the original 50/25 cases.
- Added paraphrase, indirect ranking, Turkish, obfuscation, hierarchy attack, uncertainty suppression, direct-advice, retrieval failure, and conflicting-evidence categories.
- Fixed mixed-intent routing and added category-level Wilson intervals with bounded safety wording.
- The API cost-safety incident associated with test execution is documented separately above; the runtime default has been corrected.

## 2026-07-12 — Phase 6 OOF Counterfactual Protocol and Cost Benchmark

- Added `src/experiments/manuscript_counterfactual_actionability.py` and both requested OOF/denominator test files.
- Each case is excluded from model fitting, prototype selection, scale estimation, and domain construction; desired-class prototypes and IQR scales use only the outer training fold.
- Added training-domain prototype changes, relational tenure constraints, comparable numeric/categorical costs, five intervention modes, explicit failure reasons, Wilson validity intervals, bootstrap mean intervals, denominators, and qualitative-only examples.
- Focused tests: 5 passed.
- Diagnostic benchmark: 10 eligible cases, 8,317 candidate evaluations, 8.916 stage seconds (21.2 seconds including process startup/import and output work).
- Projected final compute: all approximately 1,196 eligible cases 18–22 minutes; stratified 400-case sample 6–8 minutes. User decision requested; no final validity result has been generated.

## 2026-07-12 — Phase 13 Canonical Integration Started

- The user continued after the counterfactual cost checkpoint; Decision D-007 now records the recommended all-eligible OOF population as accepted.
- Added `src/experiments/build_manuscript_evidence.py` as the single canonical entry point.
- The orchestrator creates a versioned run, executes only run-local stages, permits reuse only when run/config/source/dataset hashes and every cached output hash match, registers every output in the run manifest, validates forbidden primary features, validates all 80 CompleteCaseEvidence records, generates all seven figures, builds independent final evidence manifests, and publishes `reports/manuscript_final/latest` only after success.
- Added `tests/test_final_evidence_manifest_hashes.py`.
- Extended the deterministic guardrail runner with canonical output-directory, run-ID, config-hash, and run-registration overrides so it cannot silently write a separate historical run during integration.
- No real LLM/API execution is part of the canonical entry point.

## 2026-07-12 — Deterministic Agent-Audit Integration

- Added `src/agents/manuscript_deterministic_audit.py` to execute all five evidence-audit roles, explanation compliance, and supervisor aggregation for canonical CompleteCaseEvidence records.
- The renderer is explicitly forced to `provider=offline`; outputs are labelled deterministic pipeline validation only, with `real_llm_used=false`, `api_call_attempted=false`, and no real-LLM quality claim.
- Replaced the hard-coded `CompleteCaseEvidence` forbidden-feature list with the exact canonical primary-policy exclusions, including recursive grouped/class-specific SHAP keys.
- Added `tests/test_deterministic_agent_audit_offline.py`.
- Declared the scientific runtime and raw-workbook inspection dependencies in `requirements.txt` and populated `environment.yml`; actual run versions remain captured in the run manifest.
- Added an automated historical artifact indexer that hashes and labels pre-canonical files as not admitted while preserving them in place.

## 2026-07-12 — Phases 3, 7, 9, and 10 Integration Complete

- Added a run-local external-evidence stage with task-separated metrics and claims, verified mappings/support, exact OOF HRDataset local SHAP, and locked-transport infeasibility based on three safe common features: `EmpJobRole`, `EmpJobSatisfaction`, and `ExperienceYearsAtThisCompany`.
- Added the canonical 40 INX + 40 HRDataset CompleteCaseEvidence builder. It rejects full-fit/OOF evidence mixing and hard-blocks API execution.
- Added support-aware common-fold fairness/proxy evidence with 5,000 stratified bootstrap replicates and explicit support/stability categories.
- Added validated dataset cards whose ambiguous retrieval, authenticity, licence, and citation fields remain `manual_review_required`.
- Final pre-run semantic config hash after all canonical sections and package-version fields: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`.
- Combined acceptance suite: 76 passed, 2 pre-run canonical-package checks skipped, plus 4 subtests; no API-usage ledger change.

## 2026-07-12 — Canonical Attempt 1 Failed Safely and SHAP Order Fix

- Preserved failed run `reports/manuscript_final/manuscript_final_20260712T175019Z_c664ef152ff3/`; it is not the stable latest package.
- Failure occurred at SHAP grouped-family ordering after policy and calibration stages completed.
- Added explicit grouped-axis alignment while retaining strict missing/extra-family failure behavior and a regression test.
- Full ten-fold real-data SHAP diagnostic completed successfully after the fix.
- API usage ledger was unchanged throughout the failed attempt.

## 2026-07-12 — Canonical Attempt 2 Failed Closed at Evidence Identity Gate

- Preserved failed run `reports/manuscript_final/manuscript_final_20260712T175251Z_c664ef152ff3/`; it is not `latest` and will not be reused.
- The accepted all-eligible OOF counterfactual stage completed: 1,200 unique OOF cases, 1,196 eligible cases, 1,087,050 candidate evaluations, and 1,075.94 stage seconds. These diagnostic counts come from the executable failed-run protocol and are not promoted as final manuscript estimates.
- The LLM preflight correctly blocked a representative-case table that omitted run/config identity.
- Added mandatory selection-table identity propagation and explicit external filename-schema resolution.
- Real-data integration diagnostic subsequently produced 80/80 complete cases, zero incomplete cases, and a successful seven-role offline deterministic audit, with no API call and no usage-ledger change.

## 2026-07-12 — Canonical Manuscript Package Complete

- Successful authoritative run: `manuscript_final_20260712T181754Z_c664ef152ff3`.
- All ten run-local stages completed under one config/source/data/run contract; `reports/manuscript_final/latest/` exposes the successful package only.
- Run manifest validation passed with current source-tree verification and 214 registered outputs.
- Independent final evidence manifest passed for 212/212 referenced package files, both in the versioned root and through `latest`.
- All seven figures passed PNG/SVG existence/non-empty validation.
- Canonical CompleteCaseEvidence preflight passed 80 requested/selected/complete, zero incomplete; API execution remained disallowed.
- Successful canonical run and final tests made zero API calls; the usage-ledger SHA-256 remained unchanged.
- Final compileall, 161-test unittest, 188-test-plus-4-subtest pytest, and filename-only secret scan all passed.
- Updated the issue inventory, unresolved-items handoff, artifact manifest, and final completion report. Manuscript edits: none.

## 2026-07-12 — GitHub/AI-Reviewer Handoff Documentation

- Replaced the legacy README with a canonical evidence-package landing page.
- Corrected the unsupported implication that the canonical run executed a real paid OpenAI batch: it now records 80/80 complete evidence records, deterministic offline auditing, and zero canonical paid calls.
- Added direct links to the canonical configuration, run/evidence manifests, all evidence-stage directories, remediation logs, claim boundaries, and unresolved items.
- Audited 28 modified and 781 untracked paths against Git ignore rules; none is ignored and `.gitignore` was intentionally left unchanged.
- Confirmed that only local virtual-environment, cache, bytecode, IDE-state, and environment-secret patterns remain ignored.
- Recorded that the untracked publication set is approximately 492 MiB and no individual file exceeds 100 MiB.
- Validated all 40 README local links, `git diff --check`, and manuscript non-modification.
