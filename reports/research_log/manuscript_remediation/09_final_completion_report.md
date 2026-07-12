# Final Completion Report

Status: **canonical manuscript evidence package complete; author/manual-review items remain explicitly bounded**

## Authoritative Run

- Run ID: `manuscript_final_20260712T181754Z_c664ef152ff3`
- Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`
- Git base commit: `18347488bdb4eed60f115ceeff70c420071ceef0`
- Versioned root: `reports/manuscript_final/manuscript_final_20260712T181754Z_c664ef152ff3/`
- Stable handoff: `reports/manuscript_final/latest/`
- Run manifest: `reports/manuscript_final/latest/run_manifest.json`
- Final evidence manifest: `reports/manuscript_final/latest/final_evidence_manifest.json`
- Claim matrix: `reports/manuscript_final/latest/canonical_claim_boundaries.md`

The manifest records the dirty-worktree state, source-tree hash, frozen input/config/environment snapshots, dataset hashes, package versions, seeds, commands, timestamps, output paths, failures, sizes, and SHA-256 hashes. The completed run registers 214 outputs. Its independent final evidence manifest verifies 212 package files. `latest` was independently revalidated against the same hashes.

## Issues Resolved

- One exact primary policy now excludes `Age`, `Gender`, `MaritalStatus`, `EmpDepartment`, `EmpLastSalaryHikePercent`, `Attrition`, `EmpNumber`, and `PerformanceRating` everywhere in the canonical primary model, SHAP, reason-code, and LLM-evidence path.
- Full-feature and reduced-feature policies were rerun with common folds, preprocessing, XGBoost settings, uncertainty, pairwise tests, and leakage-sensitivity indices. The full-feature result is labelled diagnostic only.
- Task-aware metric applicability now encodes ordinal, restricted-target, attrition, and turnover boundaries. Binary severe-error and other ordinal metrics are N/A rather than zero.
- HRDataset_v14 is independent external performance-target replication; IBM restricted-target and IBM/turnover binary tasks remain non-comparable robustness/related-task strata. Locked transport is explicitly false and infeasible under the verified three-feature safe overlap.
- Calibration was regenerated with nested outer-ten-fold evaluation, inner calibration splits, raw/sigmoid/isotonic comparison, ten bins, uncertainty, warnings, and Figure 5.
- Canonical OOF grouped SHAP now covers global/class/local evidence, ten-fold rankings and stability, representative strata, governance metadata, forbidden-feature validation, and Figures 6–7.
- Counterfactual evaluation is OOF, uses fold-training-only prototypes/scales/domains, covers every eligible case, reports denominators/uncertainty/failures, and separates intervention modes from qualitative examples.
- All 80 selected INX/HRDataset LLM-evaluation cases have complete fold-consistent evidence. Real execution remains blocked without explicit paid approval.
- The deterministic guardrail suite is versioned and expanded to 80 unsafe plus 34 safe cases across paraphrase, ranking, Turkish, obfuscation, injection, warning suppression, direct advice, retrieval failure, conflict, and mixed intent, with category Wilson intervals.
- Fairness/proxy evidence is support-aware, common-fold, bootstrap-bounded, and labels sensitive audits separately from exploratory operational diagnostics.
- Dataset cards are machine-readable, validated, and explicit about manual source/licence review.
- Figures 1–7 are generated reproducibly in PNG/SVG with source data and run/config metadata; Figure 4 is a component status matrix, not a composite score.
- Historical artifacts remain in place and are hashed/indexed as not admitted to the canonical package.
- Runtime defaults no longer infer paid execution from API-key presence.

## Experiments Regenerated

- Common-fold feature-policy/leakage ablation.
- Nested calibration comparison and reliability data.
- OOF grouped SHAP global, class, local, stability, and reason-code evidence.
- All-eligible OOF counterfactual actionability evaluation.
- External replication/restricted-target/binary-related-task experiments and safe-overlap transport gate.
- Common-fold fairness and department-proxy analyses with configured bootstrap uncertainty.
- Complete 40 INX + 40 HRDataset case-evidence preflight and seven-role offline deterministic governance audit.
- Versioned deterministic chatbot guardrail suite.
- Dataset provenance/cards, G-XAIR component readiness data, and all seven manuscript figures.

All scientific numeric results are generated in and must be cited from the run-local CSV/JSON/figure-source files; this completion report does not substitute hand-entered values for those authoritative artifacts.

## Verification

- Clean canonical entry point: passed.
- Run-manifest integrity with current source-tree verification: passed.
- Final-evidence-manifest integrity: 212/212 referenced package files passed.
- Figure package: seven figures, PNG and SVG, passed.
- CompleteCaseEvidence preflight: 80 requested, 80 selected, 80 complete, 0 incomplete; real API execution disallowed.
- `compileall`: passed.
- `unittest`: 161 tests passed.
- `pytest`: 188 tests plus 4 subtests passed.
- Secret scan: zero matching files and zero workspace `.env` files within the declared scan scope.
- API-safety hash guard: `reports/llm_explanations/llm_usage_log.csv` was unchanged across the canonical run and final tests.
- Manuscript file modified: no.

Two failed end-to-end attempts are preserved with failure information. The first exposed and fixed a grouped-SHAP axis-order assumption; the second exposed and fixed missing run/config identity on the representative-case table and an external filename-schema mismatch. Neither failed run was reused or promoted.

## Paid API Accounting

- Successful canonical run: **0 paid API calls**.
- Canonical tests/preflight/agent audit/chatbot: **0 paid API calls**.
- Earlier remediation safety incident: 24 unintended pre-safeguard calls, preserved as non-scientific history; repository ledger estimate USD 0.1096371. Provider billing remains authoritative.

## Claims Supported by the Canonical Package

- Common-fold leakage-policy sensitivity for the declared XGBoost contract, with full-feature evidence only as a diagnostic upper bound.
- Internal OOF three-class performance, nested probability calibration, grouped SHAP attribution/stability, and OOF counterfactual model-scenario evidence for the canonical primary policy.
- Independent HRDataset_v14 performance-target replication with explicit mapping, support, provenance, and non-transport limitations.
- IBM restricted-target performance robustness and IBM/Employee Turnover related binary-task evidence in separate, non-comparable strata.
- Support-aware descriptive subgroup disparities and department reconstructability as proxy-risk evidence.
- Complete canonical evidence readiness for the selected 80-case LLM evaluation and deterministic offline governance-pipeline validation.
- Fixed-suite deterministic chatbot guardrail results with finite-sample Wilson intervals and non-comprehensive-safety wording.

## Claims Not Supported or Prohibited

- Real-LLM faithfulness/compliance rates under the new canonical evidence contract; no paid canonical batch was run.
- Autonomous hiring, firing, promotion, ranking, compensation, discipline, performance evaluation, or other HR decisions.
- Deployment readiness, legal compliance, or comprehensive safety.
- Causal SHAP explanations or causal/prescriptive counterfactuals.
- A fairness guarantee from sensitive-feature removal, subgroup gaps, or proxy analysis.
- Direct employee-performance external validation from attrition, turnover, or restricted-target tasks.
- Locked INX-model transport to HRDataset_v14.
- Verified dataset licence/source authenticity where cards state `manual_review_required`.
- Zero LLM, agent, guardrail, calibration, subgroup, or model failure probability.

The package is research-grade decision support only. It must not be presented or implemented as an autonomous HR decision system.

## Repository Publication Handoff

- `README.md` now points reviewers to the canonical run/config/manifests and generated result sources, rather than foregrounding incompatible historical reports.
- The README explicitly distinguishes 80/80 case-evidence readiness and offline deterministic governance checks from a real paid LLM evaluation; canonical paid calls remain zero.
- A complete ignore audit found that none of the 28 modified or 781 untracked publication paths is excluded by `.gitignore`; all research code, tests, configuration, logs, and artifacts are available to `git add -A`.
- `.gitignore` remains unchanged because its active exclusions cover only local environments, caches, bytecode, IDE state, and environment-secret files.
- The untracked publication set is approximately 492 MiB; no individual file exceeds GitHub's 100 MiB hard file limit, but publishing all preserved and mirrored artifacts will produce a large commit.
- README link validation, diff hygiene, and the manuscript non-modification check passed.
