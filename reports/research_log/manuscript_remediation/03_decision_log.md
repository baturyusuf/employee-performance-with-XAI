# Decision Log

## D-000 — Freeze Baseline Before Scientific Modification

- Date: 2026-07-12
- Status: accepted
- Decision: perform read-only diagnosis and create only the mandated remediation logs before scientific-code changes.
- Basis: explicit user operating rule and Phase 0 requirement.

## D-001 — Treat Existing Scientific Outputs as Historical Until Proven Compatible

- Date: 2026-07-12
- Status: accepted
- Decision: preserve existing artifacts, but do not admit any into the new manuscript package unless a machine-readable compatibility check proves identical run/config/data/code inputs.
- Basis: existing stages have fragmented run identities and incomplete provenance.

## D-002 — Canonical Primary Policy Follows the Explicit Submission Contract

- Date: 2026-07-12
- Status: accepted
- Decision: the future canonical primary policy will explicitly exclude `Age`, `Gender`, `MaritalStatus`, `EmpDepartment`, `EmpLastSalaryHikePercent`, `Attrition`, ID, and target fields in one definition. Audit frames may retain sensitive/audit fields, but the primary model and its SHAP/reason-code artifacts may not.
- Basis: explicit user requirement and existing generated model-card claim. This resolves rather than renames the current split definition.

## D-003 — No Paid API Execution Is Authorized

- Date: 2026-07-12
- Status: accepted
- Decision: perform deterministic and dry-run validation only. Before any new real OpenAI batch, provide case/call/token/cost scope and request explicit approval.
- Basis: explicit cost-control rule.

## D-004 — Do Not Edit the Manuscript

- Date: 2026-07-12
- Status: accepted
- Decision: inspect `manuscript/mdpi_information/main.md` only to understand evidence claims. Deliver an evidence/claim matrix for author-led revision, with no manuscript edits.
- Basis: explicit scope boundary.

## D-005 — Preserve Current Leakage Terminology Pending Author Decision

- Date: 2026-07-12
- Status: pending_author_decision_if_needed
- Decision: code and reports will expose measured leakage sensitivity and exact exclusions. The remediation will not change the manuscript title or final terminology from leakage-safe to leakage-aware/reduced.
- Basis: title/claim terminology is explicitly reserved to the user.

## D-006 — Do Not Authenticate Ambiguous Dataset Licences or Sources by Assumption

- Date: 2026-07-12
- Status: accepted
- Decision: record mirror URLs and hashes, require all fields, and mark licence/source authenticity `manual_review_required` until independently confirmed by the author.
- Basis: code cannot establish legal/source authenticity, and manual judgement requires user input.

## D-007 — Use All Eligible OOF Cases for Counterfactual Evaluation

- Date: 2026-07-12
- Status: accepted
- Decision: evaluate every eligible out-of-fold case. The ten-eligible-case benchmark evaluated 8,317 candidates in 8.916 stage seconds. Using the historically observed approximately 1,196 eligible OOF cases, the full evaluation projects to roughly 994,700 candidate evaluations and 18–22 minutes; this is moderate, incurs no API cost, avoids sampling variance, and yields the strongest denominator accounting.
- User response: `continue` after the recommendation and cost comparison were presented; interpreted as acceptance of the recommended all-eligible option.
- Basis: explicit material-cost checkpoint followed by user authorization to continue.

## D-008 — API-Key Presence Is Never Execution Authorization

- Date: 2026-07-12
- Status: accepted_after_incident
- Decision: all default/deterministic LLM construction is offline. Real execution requires an explicit real provider and `require_real_llm=True` after the user has approved the estimated batch cost.
- Incident: Phase 8 tests unintentionally made 24 OpenAI calls because `auto` mode used the available machine key. Estimated ledger cost is USD 0.1096371; billing dashboard remains authoritative.
- Evidence preservation: the appended usage-ledger rows are retained and labelled non-scientific/unapproved rather than deleted.

## D-009 — Authoritative Canonical Run Selection

- Date: 2026-07-12
- Status: accepted_by_contract
- Decision: `manuscript_final_20260712T181754Z_c664ef152ff3` is the sole authoritative manuscript-evidence run and the target exposed by `reports/manuscript_final/latest/`.
- Exclusion: failed attempts `manuscript_final_20260712T175019Z_c664ef152ff3` and `manuscript_final_20260712T175251Z_c664ef152ff3` remain immutable diagnostic history and are not reused or admitted.
- Basis: only the selected run completed every stage and passed run-manifest, final-manifest, figure, CompleteCaseEvidence, full-test, compile, and secret-scan gates.

## D-010 — Canonical LLM Scope Is Evidence Readiness and Offline Governance Validation

- Date: 2026-07-12
- Status: accepted_with_cost_boundary
- Decision: the package contains 80/80 complete canonical evidence records and an offline deterministic seven-role audit, but makes no real-LLM faithfulness/compliance claim.
- Paid execution: zero calls in the canonical run. A new real batch remains a separate explicitly costed/approved decision.
- Basis: no paid batch was authorised; readiness completion satisfies the non-costly remediation scope without misrepresenting offline output as real LLM evidence.

## D-011 — Preserve Local/Secret Ignore Rules; Publish All Research Outputs

- Date: 2026-07-12
- Status: accepted_by_audit
- Decision: leave `.gitignore` unchanged because none of the 28 modified or 781 untracked remediation paths is ignored. Preserve exclusions for virtual environments, caches, bytecode, IDE state, and environment-secret files.
- Basis: a complete `git check-ignore --no-index` audit returned zero matches for the current modified/untracked publication set. Removing the remaining rules would publish local/generated or potentially sensitive files without adding any research artifact.

## D-012 — Make README the Canonical Review Entry Point

- Date: 2026-07-12
- Status: accepted
- Decision: replace legacy-first and real-LLM-overclaiming README content with the canonical run identity, exact primary policy, generated-result source links, single entry point, claim boundaries, verification status, unresolved decisions, and suggested reviewer order.
- Basis: the user requested a GitHub-ready handoff that other AI reviewers can understand without relying on chat history.
