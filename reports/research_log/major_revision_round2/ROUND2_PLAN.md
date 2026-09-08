# Round 2 Major Revision Plan

Status: **PRESPECIFIED PLAN — APPROVED FOR SCIENTIFIC EXECUTION AFTER CORRECTED CHECKPOINT PUSH**

Prepared: 2026-09-08

Baseline branch: `finalization/leakage-aware-v2`

Frozen baseline commit: `0805b793a3b2a2adb077dfdf7f271e309bd5774f`

Working branch: `revision/round2-major-v4`

Historical approved Phase 5A claim-set SHA-256: `1664b188df14135d3b3de8642de3d080c25a3fbadc440615c2bd04b2f14ddabe`

## 1. Purpose and governing priority

This additive round addresses the unresolved concerns from the second independent review. The priority order is scientific validity, fair experimental design, reproducibility, complete reporting, and only then the magnitude of observed scores. Adverse, null, unstable, and metric-conflicting results will be retained.

This plan does not authorize a release, tag, DOI, raw-data publication, history rewrite, licence conclusion, ethics determination, or invented author declaration. No paid API call is permitted.

## 2. Frozen history and write boundaries

The following are historical evidence and will not be overwritten, relabelled, or silently incorporated into a new canonical identity:

- canonical v2 run `canonical_v2_20260714T221501Z_483f96f`;
- every existing package under `reports/research_log/major_revision_v3/phase1b_*` through `phase5b_*`;
- the approved Phase 5A claim matrix and its digest;
- the Phase 5B package and reviewer-response draft; and
- baseline commit `0805b793a3b2a2adb077dfdf7f271e309bd5774f`.

All new artifacts will be additive under `reports/research_log/major_revision_round2/` and ignored complete-run roots under `reports/major_revision_round2_runs/`. Employee-level OOF rows, fold memberships, candidate rows, calibrator parameters, fitted models, and raw data remain local and excluded from the compact tracked package.

`manuscript/mdpi_information/main.md`, generated `main.tex`, and `references.bib` will remain unchanged through experiment execution, independent validation, aggregate export, and the new claim-matrix freeze. Because Round 2 necessarily introduces claims outside the old digest, manuscript rewriting requires explicit approval of the new Round 2 claim digest. The historical Phase 5B package itself will remain unchanged after that approval as well.

## 3. Prespecified execution order

```text
Plan approval
→ implementation and focused tests
→ clean implementation checkpoints
→ new scientific experiments
→ independent validators and aggregate-only exports
→ Round 2 claim matrix and numerical consistency audit
→ explicit digest-specific user approval
→ manuscript/bibliography rewrite
→ Round 2 reviewer response
→ final review simulation and submission-readiness audit
```

No new result will be inserted piecemeal into the manuscript.

## 4. Experiment R2-E1 — selection-objective sensitivity

### 4.1 Scientific question

> Does the apparent ordinal ranking depend on the hyperparameter-selection objective?

The answer will be descriptive and conditional on the exact INX P3 feature policy, candidate registries, folds, preprocessing, seeds, and OOF estimand. No significance or universal-winner claim is planned.

### 4.2 Common contract

- Dataset/target: INX `PerformanceRating`, ordered labels 2/3/4, support 194/874/132.
- Feature policy: exact P3 20-feature contract from `configs/feature_availability_v3.json`.
- Outer/inner folds: exact persisted canonical-v2 10 outer × 5 inner identities.
- Seeds: outer 42, inner 43, model 44, matching the frozen Phase 1B contract.
- Preprocessing: numeric median imputation and standard scaling; categorical most-frequent imputation and dense one-hot encoding with unknown categories ignored; all fitted only within the current inner-development or outer-training partition.
- Candidate grids: byte-bound existing grids only; no candidate may be added or removed after results are inspected.
- Models: multinomial logistic regression, Random Forest, LightGBM, nominal XGBoost, proportional-odds logistic regression, and cumulative-threshold XGBoost.
- Naive baselines do not enter selection.
- Outer-test rows are evaluation-only and cannot affect preprocessing, candidate scores, selection, calibration, seed choice, or policy choice.

### 4.3 Regime A — historical macro-F1 selection

- Primary metric: mean inner-fold macro-F1, higher is better.
- Inclusive practical tie pool: absolute primary gap no greater than 0.001.
- Tie-break: highest mean inner-fold QWK, then lowest candidate index.
- OOF evidence: reuse the independently validated Phase 1B exactly-once OOF predictions without refitting or relabelling.

### 4.4 Regime B — new QWK selection

- Primary metric: mean inner-fold QWK, higher is better.
- Inclusive practical tie pool: absolute primary gap no greater than 0.001.
- Tie-break: highest mean inner-fold macro-F1, then lowest candidate index.
- Candidate-score source: reuse the complete exhaustive inner-fold candidate evaluations already generated on the exact folds. This is valid because every candidate was evaluated independently of the historical selection choice and both macro-F1 and QWK were persisted. The validator will bind the canonical-v2 nominal candidate table and the complete local Phase 1B ordinal candidate table by path, size, SHA-256, fold identity, candidate registry, and `outer_test_used_for_selection=false`.
- Outer refits: refit all six selected systems in every outer fold under the QWK schedule, for exactly 60 outer-model fits. Refitting all six/fold combinations avoids selective reuse based on whether a candidate happens to match Regime A.
- OOF evidence: every sample must occur exactly once per model under Regime B.

### 4.5 Excluded optional Regime C

An ordinal-MAE- or RPS-selected third regime is not included in this round. Existing exhaustive candidate receipts do not retain these candidate-level inner scores, so adding Regime C would require a materially larger inner-CV rerun. Excluding it now prevents a result-contingent expansion. A future Regime C would require a separate prespecified contract before inspecting new results.

### 4.6 Required outputs and decision rule

For both regimes and every trained model, report:

- macro-F1, balanced accuracy, QWK, ordinal MAE, two-level reversal rate;
- normalized RPS, log loss, multiclass Brier, top-label ECE;
- class 2/3/4 precision, recall, F1, and support; and
- complete ordered confusion matrices.

The main sensitivity table will contain at least:

| Model | Selection objective | Macro-F1 | QWK | Ordinal MAE | RPS | Class-4 recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |

The report will not collapse sensitivity into one binary “ranking depends” conclusion. For every evaluation metric it will report separately:

- `leader_changed`: whether the best-ranked model changes between Regimes A and B;
- `full_ordering_changed`: whether the exact six-model ordering changes, even when the leader does not;
- selected-candidate changes: model/fold indicators and counts, including unchanged selections; and
- metric-effect magnitudes: signed and absolute Regime B minus Regime A score changes for every model, together with rank-position changes and the score separation around any leader change.

A lower-rank permutation with negligible score changes will be described only as a small lower-order change, not as material dependence. Leader changes, full-ordering changes, candidate changes, and score magnitudes will remain distinct descriptive findings; no composite binary materiality label or hypothesis test is prespecified.

## 5. Experiment R2-E2 — training-only empirical-prior probability baseline

For each of the exact ten outer folds, compute the class-prevalence vector using only outer-training labels:

`P(Y=k) = n_train,k / n_train`.

Assign that fixed training-derived vector to every row in the untouched outer-test fold. The baseline has no hyperparameter selection and performs no model fit. It will be reported for log loss, multiclass Brier, normalized RPS, and top-label ECE. Probability sums, declared label order, training-only derivation, and fold/sample coverage must validate exactly.

The majority, stratified, and ordinal-median hard-label baselines remain valid comparators for macro-F1, balanced accuracy, QWK, ordinal MAE, and reversal rate. Their one-hot probability scores will be removed from probability-quality comparison tables or explicitly labelled non-meaningful as probabilistic comparators; they will not be used to claim probability-quality superiority.

## 6. Experiment R2-E3 — HR target-alias disagreement sensitivity

The Phase 3B audit found two `PerformanceScore` versus `PerfScoreID` disagreements. The canonical text-target rule and the historical 311-row result remain unchanged. A limited matched-sample sensitivity will be run as follows:

- target: retained three-class text mapping only;
- model: raw-probability XGBoost only;
- features/grid/preprocessing: exact Phase 3A conservative seven-feature and eight-candidate contract;
- design: Phase 3A repetition-1 5 outer × 5 inner fold identity, restricted consistently after excluding the two disagreement rows;
- historical canonical arm (311 rows): preserve and report the independently validated Phase 3A repetition-1 raw-XGBoost result without refitting;
- restricted canonical arm (309 rows): remove the two disagreement rows from the existing canonical OOF predictions and recompute all metrics fit-free on the remaining rows;
- exclusion/refit arm (309 rows): refit candidate selection and outer models after excluding exactly the two audited disagreement rows, then evaluate on exactly the same 309 row identities as the restricted canonical arm;
- calibration and naive baselines: excluded from this limited sensitivity;
- outputs: row-identity match receipt, support, selected-candidate schedule, macro-F1, balanced accuracy, QWK, ordinal MAE, RPS, log loss, Brier, ECE, per-class metrics, confusion matrix, and signed deltas; and
- interpretation: descriptive data-quality sensitivity only, with no equivalence test or robustness claim.

The primary training/data-rule sensitivity comparison is the matched-population contrast `restricted canonical predictions (309 rows)` versus `exclusion/refit predictions (309 rows)`. The separate `historical canonical predictions (311 rows)` versus `restricted canonical predictions (309 rows)` contrast is labelled only as the fit-free sample-removal effect. It cannot be attributed to refitting or the exclusion training rule. The 311-row and 309-row results will never be used as the primary refitting comparison.

The row identities and employee-level predictions remain local and untracked. The compact report may state only that two prespecified audited rows were removed and publish aggregate results.

## 7. Existing evidence promoted into Round 2 reporting

The following require no new scientific fit but do require deterministic extraction, independent source-row validation, and publication-ready aggregate tables.

### 7.1 Extreme-class behavior

Create a per-class table for cumulative-threshold XGBoost, nominal XGBoost, LightGBM, and Random Forest, including precision/recall/F1 for ratings 2/3/4. Preserve Random Forest class-4 recall and F1 even if zero. Link the full confusion matrices. The discussion boundary is metric behavior: aggregate ordinal metrics can conceal complete failure on an extreme class. No untested causal explanation will be stated.

### 7.2 Timing/information sensitivity

Move the P3→P4 and P4→P5 changes into the central Results/Discussion/Conclusion story and the abstract after claim approval. Use both fixed and retuned estimands where relevant, with the retuned macro-F1 sequence P0/P3/P4/P5 clearly labelled. P4 is never timestamp-verified prospective evidence; P3 is not declared confirmed leakage; no policy is called leakage-free.

### 7.3 Manuscript-facing policy names

Internal IDs and historical config names remain unchanged. New manuscript-facing labels are frozen as:

| ID | Round 2 manuscript-facing label |
| --- | --- |
| P0 | Information-Rich Diagnostic |
| P1 | Leakage-Risk-Controlled |
| P2 | Direct-Sensitive-Attribute-Excluded |
| P3 | Primary Leakage-Aware |
| P4 | Prospective-Plausibility |
| P5 | Strict Proxy-Reduced Prospective-Plausibility |

Every policy table/figure caption will independently state that timestamps are unavailable and P4/P5 are sensitivities, not prospective validation.

### 7.4 Subgroup reporting

Extract the P3, support-threshold n≥30 maximum-minus-minimum gap for macro-F1, QWK, and ordinal MAE for Age, Gender, Marital Status, Business Travel, Department, and Education Background. Preserve group endpoints, eligible-group counts, statuses, and exploratory interval labels where available. Publish all six attributes, including the large department gap, without selecting only favorable cells. The table is a descriptive diagnostic, not evidence of fairness, discrimination, protected-class effects, or legal compliance.

### 7.5 HR target-mapping sensitivity

Report retained three-class support 31/243/37 and raw-order four-class support 13/18/243/37. Report raw and sigmoid XGBoost metrics separately for each estimand, including the existing 3-class and 4-class macro-F1/QWK results. Do not compute or describe a cross-formulation “improvement” because the target estimands differ. The required interpretation is that replication conclusions are sensitive to target construction and metric choice.

### 7.6 HR CV-design sensitivity

Report every one of the 14 canonical-v2 10×5 point-estimate comparisons and identify the three values outside the repeated-5×5 ranges. Use the accurate statement “11 of 14 were inside; 3 of 14 were outside.” This is descriptive CV-design sensitivity, not equivalence or unqualified robustness evidence.

## 8. Self-contained Methods and supplementary contracts

Generate publication-ready tables after the evidence freeze:

- **Table S1 — Complete Feature Availability and Governance Contract:** every feature, semantic family, timing status, risk type, governance type, P0–P5 inclusion, justification, and explicit timestamp caveat.
- **Table S2 — Preprocessing and Hyperparameter Search Contract:** numeric/categorical processing, unknown-category handling, all six trained-model fixed parameters and candidate grids, selection objectives, 0.001 tie tolerance, tie-breaks, failure policy, and outer-test isolation.
- **Table S3 — Exact CV and Seed Contract:** fold counts, shuffle flags, seeds, persisted fold identities, OOF coverage, calibration isolation, selection-score sources, and all reuse/refit boundaries.

Methods prose will explain cumulative-threshold XGBoost as independently fitted binary tasks for `P(Y > threshold)`, rowwise nonincreasing PAVA projection to correct crossings, and reconstruction of ordered class probabilities by differencing. It will also describe proportional-odds/cumulative-link assumptions and the complete preprocessing contract.

## 9. Literature and bibliography v4

Create a new additive Round 2 literature/provenance package; do not alter Phase 4A. Verify bibliographic metadata against publisher pages, official proceedings, or primary papers. At minimum cover:

- proportional-odds/cumulative-link modelling;
- ordinal classification/evaluation;
- ranked probability score or proper ordinal probability evaluation;
- Random Forest;
- XGBoost; and
- LightGBM.

The 25-work cap is not retained when it conflicts with method citation completeness. The new novelty statement is fixed as: “The contribution is not the novelty of any single component, but their operational integration into a traceable evaluation contract.” No world-first or exhaustive-review claim is permitted.

## 10. Abstract estimand rule

Use the explicit-design alternative: the abstract will say “in the frozen canonical INX benchmark” for the single canonical INX OOF values and “across five HRDataset repetitions” for HR means. It will not imply that these are the same estimand. The abstract will prioritize information/timing sensitivity, selection-objective sensitivity, extreme-class failure, metric-specific calibration, bounded subgroup/proxy diagnostics, and target-mapping sensitivity rather than accumulating many numbers.

## 11. Round 2 claim matrix and rewrite gate

Create a new `ROUND2_CLAIM_MATRIX.md` plus machine-readable source register. Every item will be classified as `retained`, `modified`, `superseded`, `new`, or `prohibited`. Every numerical sentence must map to one exact aggregate source row/value/hash and every narrative sentence to a hash-bound source anchor. Generate a new claim-set digest without changing the historical Phase 5A files.

Before manuscript rewrite, run a numerical-consistency audit across Phase 5B, Round 2 sources, and the new matrix. If any result changes a principal conclusion—and in all cases before introducing new Round 2 claims into `main.md`—request explicit digest-specific user approval. No approval is inferred from a generic instruction to continue.

After approval, remove visible `(Cxxx)` markers and the full digest from ordinary academic prose. The manuscript will retain one concise traceability sentence; complete IDs and hashes move to the Supplementary Evidence Ledger.

## 12. Manuscript discussion obligations

The eventual rewrite must answer directly:

1. Why “best model” has no single answer.
2. Whether selection objective changes ordinal ranking.
3. How Random Forest can achieve strong QWK/MAE while missing rating 4.
4. What the P3→P4 timing-uncertain feature removal shows and does not show.
5. Why SHAP stability and deletion faithfulness are distinct.
6. Why proxy reconstructability is not discrimination evidence.
7. Why HR target mapping changes interpretation.
8. Why the work is not prospective HR deployment evidence.

Possible explanations—selection objective, imbalance, cumulative decomposition, projection, extreme-class trade-offs, and ordinal-distance metric behavior—will be labelled plausible methodological explanations unless directly tested.

## 13. Reviewer response

Create `REVIEWER_RESPONSE_ROUND2.md` with separate responses for:

1. ordinal task but macro-F1 selection;
2. aggregate ordinal metrics concealing class-4 failure;
3. timing sensitivity not central enough;
4. subgroup analyses underreported;
5. target-mapping sensitivity underreported;
6. CV-design sensitivity underreported;
7. Methods not self-contained;
8. policy labels too strong;
9. probability baselines inappropriate;
10. ordinal/core-model literature incomplete;
11. target-alias disagreements;
12. claim IDs/hashes reducing readability; and
13. abstract estimand inconsistency.

Each response will state the previous weakness, change, experiment/source, manuscript location, and resulting interpretation. Final page/line numbers wait for a real compiled MDPI PDF.

## 14. Test and validation gates

Every new scientific package will have focused contract, execution, independent-recomputation, compact-export, closed-world, tamper-rejection, and publication-hygiene tests as applicable. Mandatory scientific checks include:

- exactly-once OOF coverage for every system/regime;
- exact shared outer/inner fold identities;
- no outer-test tuning, preprocessing, calibration, seed choice, or policy choice;
- correct primary/tie-break direction and inclusive 0.001 tie pool;
- QWK higher-is-better and ordinal MAE lower-is-better orientation;
- probability finiteness/simplex/order and RPS definition;
- empirical-prior derivation from outer-training labels only;
- cumulative-probability monotonicity and PAVA behavior;
- complete confusion/per-class grids and support sums;
- exact source-row/hash identity for publication tables and claims;
- manuscript number ↔ Round 2 claim matrix identity;
- unchanged historical Phase 5A and Phase 5B package bytes; and
- zero employee-level/raw/model/calibrator evidence in tracked compact packages.

Final gates after authoring: focused tests, full pytest, unittest discovery, compileall/import checks, `git diff --check`, README/link/path audit, secret/large/raw-data scan, stale-number audit, forbidden-wording audit, and Markdown/LaTeX/table/figure parity. Tests will not be loosened merely to make a defect pass.

## 15. Git discipline and checkpoints

No force push, destructive history operation, merge, raw employee-data commit, row-level OOF/SHAP commit, fitted-model commit, tag, release, or DOI creation is allowed. Use ordinary pushes after tested atomic commits. Planned commit sequence:

1. `research: plan round2 major revision`
2. `feat(eval): add selection-objective sensitivity`
3. `feat(eval): add probability prior baseline`
4. `feat(hr): add target-alias sensitivity`
5. `feat(reporting): add per-class and subgroup publication tables`
6. `docs(methods): expand reproducibility contract`
7. `docs(literature): add ordinal and core model references`
8. `research: freeze round2 claim matrix`
9. `docs(manuscript): revise second-round major revision`
10. `docs(research): close round2 handoff`

The exact grouping may be tightened when one validator and its package must remain atomic, but scientific and manuscript gates will not be collapsed.

## 16. Required deliverables

The Round 2 directory will contain at minimum:

- `ROUND2_PLAN.md`
- `ROUND2_REVISION_REPORT.md`
- `SELECTION_OBJECTIVE_SENSITIVITY.md`
- `PER_CLASS_EXTREME_CLASS_REPORT.md`
- `SUBGROUP_MANUSCRIPT_SUMMARY.csv`
- `HR_MAPPING_SENSITIVITY_REPORT.md`
- `PROBABILITY_BASELINE_REPORT.md`
- `METHOD_REPRODUCIBILITY_TABLES/` with S1/S2/S3
- `ROUND2_CLAIM_MATRIX.md`
- `RESULTS_COMPARISON_ROUND2.csv`
- `REVIEWER_RESPONSE_ROUND2.md`
- `FINAL_REVIEW_SIMULATION_ROUND2.md`
- `AUTHOR_ACTION_REQUIRED.md`

Each scientific subpackage will also contain provenance, manifest, and independent-validation receipts.

## 17. Author-only and release blockers

`AUTHOR_ACTION_REQUIRED.md` will retain affiliations, ORCID, author-approved CRediT roles, funding, conflicts, ethics/IRB, consent, dataset redistribution rights, software-licence choice, and Git-history legal/remediation decisions. Each item will state the missing information, reason, manuscript location, and safe options without inventing a value.

After scientific completion and claim approval, the current official MDPI *Information* template will be verified from an authoritative source. If a TeX engine and complete class bundle are available, the manuscript will be compiled and visually checked. Otherwise PDF/template compliance remains an explicit blocker. Public tag/release/DOI creation still requires separate user authorization after rights, licence, declaration, and history decisions are resolved.

## 18. First checkpoint acceptance criteria

Before any Round 2 scientific result is computed:

- this plan is committed on `revision/round2-major-v4`;
- baseline and historical claim identities above are correct;
- the working tree is clean;
- the plan is linked from the root README;
- `git diff --check` and link validation pass; and
- the branch is pushed for user review.

Scientific execution begins only after the corrected plan checkpoint is pushed. The user explicitly authorized implementation and scientific execution to continue without a second plan-approval wait once the two requested corrections are incorporated and pushed.

## 19. Approval record and remaining gate

On 2026-09-08, the user generally approved plan commit `0573b22f970d86ad862ef7fee5853286a155446c`, required the matched-sample HR comparison and non-binary selection-sensitivity reporting now incorporated above, approved omission of the optional MAE/RPS-selected third regime, prohibited result-contingent scope expansion, and authorized implementation and scientific experiments after this corrected checkpoint is pushed.

This approval does not approve a future Round 2 claim digest, manuscript rewrite, release, tag, or DOI. Digest-specific approval of the new Round 2 claim matrix remains mandatory before any manuscript or bibliography rewrite.
