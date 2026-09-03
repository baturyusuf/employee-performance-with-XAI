# Reviewer-Brief Coverage Audit Before v3

Date: 2026-09-03
Compared against: canonical v2 run `canonical_v2_20260714T221501Z_483f96f`, tracked compact manuscript assets, current source/tests, and current Markdown/LaTeX manuscript.

## Executive finding

The v2 package is internally validated for its frozen protocol, but it does **not** close the complete current reviewer brief. Of the 43 numbered sections (0–42), this audit classifies 13 as addressed, 13 as partial, 11 as missing, and 6 as approval/final-authoring pending. Consequently, v2 must remain immutable while missing work is added under a new v3 evidence identity. The current manuscript must not be revised from v1 until the v3 claim matrix is frozen and approved.

## Post-audit implementation update — 2026-09-03

The classifications below are the frozen pre-v3 baseline and therefore are not silently rewritten as implementation proceeds. Requirement 1 and the primary-dataset portion of Requirement 2 now have an implemented INX contract covering all 28 acquisition-schema fields, eight distinct risk types, and six exactly validated nested policy maps. The generated report is `FEATURE_AVAILABILITY_GOVERNANCE_CONTRACT.md`; its source contract SHA-256 is `6dd5fdde534e379cceacfaa01e865d1551310fb632b691f5b937ef39394e93cf`. HR-specific mapping remains assigned to Phase 3A, so the broader cross-dataset work is not declared complete.

Status meanings:

- `addressed`: executable or canonical evidence and claim boundaries materially satisfy the requirement.
- `partial`: useful evidence exists, but at least one explicitly requested experiment, metric, contract, or boundary is absent.
- `missing`: no admissible canonical implementation/evidence was found for the requested item.
- `pending`: intentionally deferred until experiments or user approval/manual metadata are complete.

## Requirement-by-requirement assessment

| No. | Requirement | Status | Evidence found | Gap / v3 action |
| ---: | --- | --- | --- | --- |
| 0 | Repository/manuscript audit, baseline reproduction, plan | addressed | v2 audit/finalization logs; clean canonical build and independent replay; current plan | Preserve v2; this audit reconciles the broader brief. |
| 1 | Prediction time and feature availability/governance contract | partial | feature taxonomy and policy tables; temporality warnings for external features | No single canonical prediction-time scenario and no complete contract with all requested availability/evidence fields. |
| 2 | Separate leakage, sensitive, and proxy concepts | partial | distinct policy exclusions and claim boundaries | Taxonomy exists but is not yet a complete manuscript-ready three-part feature contract for every raw feature. |
| 3 | At least two genuinely ordinal-aware models and RPS where possible | missing | no proportional-odds/cumulative-link/CORAL/CORN/cumulative-threshold implementation found | Add proportional-odds logistic and nonlinear cumulative-threshold boosting under shared nested CV; add RPS. |
| 4 | Separate predictive ranking from XAI reference choice | addressed | Table 3 reports four models; Random Forest leads QWK/MAE while XGBoost is explicitly the XAI reference | Preserve metric-specific ranking language; never call XGBoost universally best. |
| 5 | Repeated nested CV training/fold uncertainty | missing | single deterministic 10×5 nested OOF plus case-conditional bootstrap | Add repeated 5×5 outer/inner sensitivity and ordering stability. |
| 6 | Fixed-HP and independently retuned policy experiments | partial | fixed selected-schedule matched sensitivity is canonical | Independent policy retuning is explicitly disabled; add it as a separate estimand. |
| 7 | Verified literature review and positioning table | missing | stale/narrow v1 bibliography; no 15–25-study positioning table | Perform source-verified review; rebuild mechanistic novelty argument. |
| 8 | SHAP fold, seed, and resampling stability | partial | exact-fold SHAP and 45 dependent fold-pair comparisons | Add seed and stratified resampling stability; retain descriptive dependence boundary. |
| 9 | Fully documented multiclass grouped SHAP implementation/additivity | addressed | raw-margin unit, exact prediction-producing model, class/family aggregation metadata, additivity/axis tests | Carry the formal algorithm and library settings into v3 methods. |
| 10 | SHAP model-faithfulness deletion/perturbation test | missing | only obsolete LLM-text “faithfulness” code was found; no model-output deletion test | Add top-1/3/5 family deletion against repeated random baselines with OOD limitation. |
| 11 | No unsupported human-evaluation claims | addressed | canonical claim boundaries prohibit human usefulness/trust/usability claims | Preserve explicit “no human-subject or HR-expert evaluation” limitation. |
| 12 | Extended calibration diagnostics | partial | leakage-aware cross-fitted sigmoid, 10-bin reliability, log loss/Brier/top-label ECE | Add class-wise ECE/reliability, cumulative/RPS calibration and suitable intercept/slope diagnostics. |
| 13 | Expanded subgroup analysis and threshold sensitivity | partial | support-aware Age, Gender, MaritalStatus, BusinessTravelFrequency, Department diagnostics with pointwise bootstrap | Requested QWK/MAE/class recalls/log loss/Brier and 20/30/50 threshold sensitivity are incomplete; multiplicity remains descriptive. |
| 14 | Proxy reconstructability versus proxy-use sensitivity | partial | Department reconstruction and job-role removal policy performance are available | Add performance-prediction change/probability change by department and grouped permutation/proxy-use evidence. |
| 15 | HRDataset_v14 framed as protocol replication | addressed | canonical language says independently trained mapped-target replication and rejects locked transport | Preserve side-by-side protocol/semantics table. |
| 16 | HR target-mapping sensitivity | missing | only primary PIP/Needs Improvement → 2, Fully Meets → 3, Exceeds/Exceptional → 4 mapping | Add one defensible alternative formulation without replacing the primary mapping. |
| 17 | HR repeated five-fold CV sensitivity | missing | one 10×5 nested protocol only | Add repeated 5×5 sensitivity and small-sample instability reporting. |
| 18 | Majority, stratified/random, ordinal-median baselines | missing | no canonical dummy baseline implementation found | Add all three on identical folds/metrics. |
| 19 | OOF confusion matrices and per-class precision/recall/F1 | missing | canonical compact tables contain aggregate metrics; legacy confusion code is inadmissible | Generate INX and HR primary OOF confusion/per-class tables. |
| 20 | Rename/define severe error as two-level reversal | partial | metric is computed correctly but still displayed as `severe_error_rate` | Rename manuscript-facing metric to two-level ordinal error/extreme-class reversal and retain exact definition. |
| 21 | Canonical data-quality audit | missing | legacy data-audit utilities/tests exist, but no v2 manuscript-ready canonical report/table | Add complete INX and HR data-quality report with schema hashes and anomaly handling. |
| 22 | Bounded construct-validity language | addressed | targets are described as organisational performance ratings/scores and equivalence is prohibited | Retain construct-validity limitation; do not infer true capability/productivity/potential. |
| 23 | Bounded external generalization | addressed | locked transport is explicitly absent/infeasible; HR is independently retrained | Preserve limitation that genuine organizational transport was not tested. |
| 24 | Bounded fairness language | addressed | subgroup/proxy outputs are descriptive and explicitly not fairness/discrimination proof | Preserve sociotechnical/legal-context limitations. |
| 25 | Publication-level reproducibility | addressed | clean offline canonical build, exact locks, manifests, hashes, one builder, CI, compact asset package | Extend rather than weaken these contracts for v3. |
| 26 | Immutable release preparation | partial | pointer/manifest architecture and publication receipts exist | No v3 tag/release notes/final commit or DOI; prepare but do not publish without explicit instruction. |
| 27 | Dataset provenance/licence verification | partial | exact byte hashes and partial workbook equivalence exist | Authenticity, citation, licence, and redistribution rights remain manual/unverified. |
| 28 | Ethics/declarations | partial | claim boundaries prohibit invented approval; pending status recorded | Institution/unit/reference/date and final declarations remain unresolved manual inputs. |
| 29 | Keep test counts out of main scientific results | pending | canonical handoff separates engineering receipts from result tables | Enforce during manuscript rewrite. |
| 30 | Rewrite all manuscript sections from final evidence | pending | current `main.md`/`main.tex` are stale v1 and intentionally untouched | Rewrite only after v3 experiments and claim approval. |
| 31 | Reassess title | pending | current title is stale and contains LLM/multi-agent/leakage-safe framing | Select bounded v3 title after strict-prospective/ordinal findings. |
| 32 | Evidence-specific abstract | pending | current abstract is stale v1 | Draft after final results and claim freeze. |
| 33 | Discussion organized around information contract/ordinal/calibration/XAI/proxy/replication/reproducibility | pending | current discussion is stale v1 | Rewrite after final evidence. |
| 34 | Keep unresolvable limitations open | addressed | timestamps, construct validity, transport, fairness, human usefulness, ethics/licence are explicitly bounded | Carry all open limitations into the final register/manuscript. |
| 35 | Scientific quality-gate tests | partial | v2 has extensive fold/calibration/policy/SHAP/hash/source tests | Add tests for every new v3 ordinal/repeated/retuned/faithfulness/mapping/data-quality component. |
| 36 | Manual scientific sanity checks | addressed | v2 validators replay models/calibrators/SHAP, check simplex, folds, resamples, directions, hashes | Repeat and extend for v3 tables. |
| 37 | Complete requested deliverable set | missing | compact tables/figures and reproducibility logs exist | Revision report, experiment summary, result comparison, limitations register, reviewer response, revised manuscript/diff, and final review simulation remain. |
| 38 | Non-defensive reviewer response | pending | no v3 response letter exists | Draft after final evidence with exact experiment/result/location references. |
| 39 | Accuracy need not improve; report adverse findings | addressed | v2 exposes the full-feature/primary gap, Random Forest ordinal superiority, sigmoid ECE degradation, and HR temporality collapse | Preserve complete prespecified reporting in v3. |
| 40 | Required phase priority | partial | v2 covered many Phase 1–4 components but under a narrower protocol | Execute the missing work in the requested scientific → XAI → replication → publication order. |
| 41 | Autonomous work with explicit blockers | addressed | logs preserve failures, reruns, provenance/licence/ethics/transport/human blockers | Continue without unnecessary questions; stop only at genuine authority/data gates. |
| 42 | Final strict-review simulation | missing | no `FINAL_REVIEW_SIMULATION.md` found | Produce after manuscript and deliverables are complete. |

## Immediate decision

The correct next step is not manuscript authoring. It is Phase 1 v3 contract/model implementation, beginning with an immutable prediction-time/feature-availability contract and ordinal/naive benchmark extensions. The v2 scientific package remains valid for its narrower estimands and may be used as a reproducibility reference, but not as evidence that the broader reviewer brief is complete.
