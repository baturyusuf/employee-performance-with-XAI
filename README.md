# Employee Performance with XAI — Canonical Manuscript Evidence Package

This repository contains a reproducible, manuscript-support evidence package for employee `PerformanceRating` prediction (classes 2/3/4) with XGBoost. It covers leakage-policy ablation, nested calibration, grouped SHAP attribution and stability, out-of-fold counterfactual scenarios, task-aware external evidence, fairness/proxy diagnostics, governed case evidence, deterministic multi-agent auditing, chatbot guardrails, dataset provenance, and manuscript-ready figures.

> **Research decision support only.** This project must not be used to make or automate hiring, firing, promotion, compensation, discipline, ranking, screening, or other individual employment decisions. It is not deployment-ready and does not establish legal compliance.

## Canonical Status

The authoritative evidence package is:

| Item | Value |
| --- | --- |
| Run ID | `manuscript_final_20260712T181754Z_c664ef152ff3` |
| Config | [`configs/manuscript_final.yaml`](configs/manuscript_final.yaml) |
| Config SHA-256 | `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3` |
| Versioned output | [`reports/manuscript_final/manuscript_final_20260712T181754Z_c664ef152ff3/`](reports/manuscript_final/manuscript_final_20260712T181754Z_c664ef152ff3/) |
| Stable handoff | [`reports/manuscript_final/latest/`](reports/manuscript_final/latest/) |
| Run manifest | [`run_manifest.json`](reports/manuscript_final/latest/run_manifest.json) |
| Evidence manifest | [`final_evidence_manifest.json`](reports/manuscript_final/latest/final_evidence_manifest.json) |
| Completion report | [`09_final_completion_report.md`](reports/research_log/manuscript_remediation/09_final_completion_report.md) |
| Open items | [`08_unresolved_items.md`](reports/research_log/manuscript_remediation/08_unresolved_items.md) |

The successful run completed all ten evidence stages, registered 214 outputs, and independently verified all 212 files admitted to the final evidence manifest. Historical reports outside `reports/manuscript_final/latest/` are retained for traceability but are **not** canonical unless explicitly indexed as compatible.

The current package records a dirty worktree and Git base commit `18347488bdb4eed60f115ceeff70c420071ceef0`. The source-tree hash validates the generating `src/` and `configs/` contents, but a fresh run after committing the remediation is recommended to bind the final package to a clean commit.

The current untracked publication set is approximately 492 MiB because it includes versioned evidence, the physical `latest` mirror, and preserved diagnostic outputs. No individual file exceeds GitHub's 100 MiB hard file limit, but publishing every artifact will create a large commit.

## Canonical Model Contract

The primary policy is defined once as:

```text
no_salary_hike_no_attrition_no_department
```

The canonical primary model excludes these fields exactly:

```text
Age
Gender
MaritalStatus
EmpDepartment
EmpLastSalaryHikePercent
Attrition
EmpNumber
PerformanceRating  # target, never an input
```

The primary estimator is XGBoost under common stratified 10-fold out-of-fold evaluation. The full-feature model is a diagnostic upper bound only and is never deployable evidence. Sensitive-retaining variants are audit-only.

## Generated Evidence Snapshot

These values are summaries of executable run-local artifacts; the linked CSV/JSON files are authoritative.

| Evidence | Canonical result | Source |
| --- | --- | --- |
| Primary internal OOF model | Accuracy 0.8492; macro-F1 0.5987; QWK 0.6380; ordinal MAE 0.1525 | [`policy_summary.csv`](reports/manuscript_final/latest/policy/policy_summary.csv) |
| Diagnostic full-feature upper bound | Macro-F1 0.9051; primary-policy delta 0.3063; Holm-adjusted p=0.0293; leakage-sensitivity index 0.3383 | [`policy_pairwise_tests.csv`](reports/manuscript_final/latest/policy/policy_pairwise_tests.csv), [`leakage_sensitivity_index.csv`](reports/manuscript_final/latest/policy/leakage_sensitivity_index.csv) |
| Selected calibration | Sigmoid; log loss 0.4551; multiclass Brier 0.2608; ECE 0.06385 | [`calibration_method_comparison.csv`](reports/manuscript_final/latest/calibration/calibration_method_comparison.csv) |
| Grouped SHAP stability | Top-10 pairwise Jaccard 0.8465; Spearman 0.9134 | [`shap_stability_summary.csv`](reports/manuscript_final/latest/shap/shap_stability_summary.csv) |
| OOF counterfactual evaluation | All 1,196 eligible cases evaluated; employee-only 28/1,196 (2.34%); employee+manager 352/1,196 (29.43%); organization/no-salary 359/1,196 (30.02%) | [`actionability_summary.csv`](reports/manuscript_final/latest/counterfactual/actionability_summary.csv) |
| External performance replication | HRDataset_v14 department-free accuracy 0.8617; macro-F1 0.6437; QWK 0.5942; ordinal MAE 0.1383 | [`performance_target_replication.csv`](reports/manuscript_final/latest/external/performance_target_replication.csv) |
| LLM case-evidence readiness | 80 requested, 80 complete, 0 incomplete; real API execution disabled | [`preflight_report.json`](reports/manuscript_final/latest/llm/preflight_report.json) |
| Deterministic chatbot regression suite | 80/80 unsafe prompts refused safely; 34/34 safe prompts answered; Wilson intervals reported | [`guardrail_evaluation_summary.md`](reports/manuscript_final/latest/chatbot/guardrail_evaluation_summary.md) |

The canonical run made **0 paid API calls**. It validates complete evidence records and deterministic offline governance behavior; it does **not** support real-LLM faithfulness or compliance-rate claims under the new canonical contract. A paid OpenAI batch requires separate cost estimation and explicit user approval.

## Evidence Map

| Area | Canonical artifacts |
| --- | --- |
| Policy/leakage | [`policy/`](reports/manuscript_final/latest/policy/): folds, summaries, pairwise tests, uncertainty, leakage sensitivity, trade-off figure |
| Calibration | [`calibration/`](reports/manuscript_final/latest/calibration/): raw/sigmoid/isotonic comparison, class-wise bins, reliability plots, Figure 5 |
| SHAP/reason codes | [`shap/`](reports/manuscript_final/latest/shap/): grouped global/class/local attribution, fold rankings, stability, validation, Figures 6–7 |
| Counterfactuals | [`counterfactual/`](reports/manuscript_final/latest/counterfactual/): OOF protocol, all-case results, denominators, failure reasons, uncertainty |
| External evidence | [`external/`](reports/manuscript_final/latest/external/): task-aware replication/robustness strata and transport feasibility gate |
| Fairness/proxy | [`fairness/`](reports/manuscript_final/latest/fairness/): subgroup support, bootstrap intervals, proxy comparisons, limitations |
| LLM/agents | [`llm/`](reports/manuscript_final/latest/llm/): CompleteCaseEvidence, preflight, deterministic seven-role audits |
| Chatbot | [`chatbot/`](reports/manuscript_final/latest/chatbot/): versioned safe/unsafe suites, category results, Wilson intervals |
| Provenance | [`provenance/`](reports/manuscript_final/latest/provenance/): machine-readable dataset cards and validation report |
| Figures | [`figures/`](reports/manuscript_final/latest/figures/): Figures 1–7 in PNG/SVG plus source data and metadata |
| Claim boundaries | [`canonical_claim_boundaries.md`](reports/manuscript_final/latest/canonical_claim_boundaries.md) |
| Remediation history | [`reports/research_log/manuscript_remediation/`](reports/research_log/manuscript_remediation/) |

The remediation log contains the repository baseline, issue inventory, root-cause analysis, decisions, progress, commands, tests, artifact inventory, unresolved items, and final completion report (`00` through `09`). This is the recommended starting point for human or AI review.

## Dataset Roles and Allowed Claims

| Dataset/task | Canonical role | Claim boundary |
| --- | --- | --- |
| INX Future Inc Employee Performance | Primary development and internal OOF evaluation | Internal three-class evidence only |
| HRDataset_v14 | Independent external performance-target replication | Independently trained replication; not locked INX-model transport |
| IBM performance | Restricted-target performance robustness | Classes 3/4 only; not equivalent to the primary three-class task |
| IBM attrition | Related binary-task transfer | Not employee-performance validation |
| Employee Turnover | Related binary-task transfer | Not employee-performance validation |

Only three department-free safe common fields were verified for INX-to-HRDataset transport (`EmpJobRole`, `EmpJobSatisfaction`, and `ExperienceYearsAtThisCompany`), so locked cross-dataset transport is marked infeasible rather than forced. Binary-task ordinal and severe-error metrics are N/A by code, not reported as zero.

## Reproduce the Canonical Package

Create a virtual environment and install dependencies:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the single canonical entry point from the repository root:

```powershell
.\myenv\Scripts\python.exe -m src.experiments.build_manuscript_evidence --config configs/manuscript_final.yaml
```

The command validates inputs, creates a versioned run, executes or reuses only contract-compatible stages, generates tables and Figures 1–7, verifies hashes, and refreshes `reports/manuscript_final/latest/`. It does not perform paid API calls under the canonical configuration.

Run verification:

```powershell
.\myenv\Scripts\python.exe -m pytest -q
.\myenv\Scripts\python.exe -m unittest discover -s tests
.\myenv\Scripts\python.exe -m compileall src tests
```

Recorded final checks passed: 188 pytest tests plus 4 subtests, 161 unittest tests, compileall, final-manifest integrity, figure validation, and the declared secret scan. Exact recorded commands and outcomes are in [`06_test_log.md`](reports/research_log/manuscript_remediation/06_test_log.md) and [`05_commands_and_runs.md`](reports/research_log/manuscript_remediation/05_commands_and_runs.md).

## Scientific and Governance Boundaries

- SHAP values are model attributions, not causal effects.
- Counterfactuals are constrained model scenarios, not recommendations to employees or managers.
- Removing sensitive or group variables does not prove fairness or eliminate proxy risk.
- Department reconstructability is proxy-risk evidence, not proof of causal or discriminatory model use.
- Fixed-suite guardrail pass rates do not establish comprehensive safety or zero failure probability.
- Calibration supports cautious probability interpretation; it does not make individual HR decisions reliable or permissible.
- Attrition, turnover, and restricted-target experiments are not direct external validation of three-class performance prediction.
- No output authorizes autonomous or individual-level HR action.

## Remaining Manual or Author Decisions

- The author must choose whether manuscript terminology says *leakage-safe*, *leakage-aware*, or *leakage-reduced*; this repository reports the evidence without changing the manuscript title.
- Dataset source authenticity, licence status, retrieval history, and citations remain `manual_review_required` where code cannot verify them.
- A canonical real-LLM batch remains optional, paid, and approval-gated.
- The historical INX workbook-to-CSV provenance comparison still requires the declared `xlrd` dependency and manual source review.
- The provider billing dashboard should be checked for the earlier preserved pre-safeguard API incident; it is not part of canonical scientific evidence.

See [`08_unresolved_items.md`](reports/research_log/manuscript_remediation/08_unresolved_items.md) for the complete bounded list.

## Repository Layout

```text
configs/                         Canonical and supporting experiment configuration
data/                            Primary and external datasets plus mappings/cards
reports/manuscript_final/        Versioned canonical evidence packages and stable latest handoff
reports/research_log/            Baseline, issue, decision, command, test, and completion records
src/experiments/                 Canonical orchestration and scientific stages
src/explainability/              Legacy/supporting XAI implementations
src/governance/                  Contracts, provenance, claims, figures, and manifests
src/llm/                         Evidence schemas, preflight, and governed explanation support
src/chatbot/                     Guardrails and deterministic evaluation
src/agents/                      Deterministic governance audits
tests/                           Unit, contract, artifact, and integration tests
manuscript/                      Author-owned manuscript; not modified by remediation
```

## Suggested Review Order

For another researcher or AI reviewer:

1. Read this README and [`configs/manuscript_final.yaml`](configs/manuscript_final.yaml).
2. Read the [`final completion report`](reports/research_log/manuscript_remediation/09_final_completion_report.md) and [`unresolved items`](reports/research_log/manuscript_remediation/08_unresolved_items.md).
3. Inspect [`canonical_claim_boundaries.md`](reports/manuscript_final/latest/canonical_claim_boundaries.md).
4. Verify [`run_manifest.json`](reports/manuscript_final/latest/run_manifest.json) and [`final_evidence_manifest.json`](reports/manuscript_final/latest/final_evidence_manifest.json).
5. Use the run-local CSV/JSON files—not historical summaries—as the source for manuscript numbers.
6. Consult [`01_issue_inventory.csv`](reports/research_log/manuscript_remediation/01_issue_inventory.csv), [`03_decision_log.md`](reports/research_log/manuscript_remediation/03_decision_log.md), and [`06_test_log.md`](reports/research_log/manuscript_remediation/06_test_log.md) for engineering traceability.

The manuscript itself was intentionally not edited during this remediation.
