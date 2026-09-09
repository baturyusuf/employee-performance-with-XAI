# Independent review B — XAI and governance

Reviewed 2026-09-09. Scope: manuscript/main.md as read at SHA-256 `5f7ede45efc9d32e5cee78ba0c9a130e66fbf9a7766f93f0f56ac4ae0b1e64e1`, with spot checks against the frozen SHAP and subgroup contracts and current journal policies. This is an internal pre-submission simulation, not journal peer review or an acceptance prediction. The reviewer did not edit the manuscript or run experiments.

**Decision: bounded methodological presentation is defensible after two specific reporting clarifications; submission remains blocked by author/institution and final production decisions.** The work cannot be presented as an ethics clearance, fairness certification, causal explanation, prospective validation, or demonstrated human benefit. The draft generally respects these boundaries.

## Findings and dispositions

| ID | Priority | Criticism | Assessment | Evidence and bounded resolution |
|---|---|---|---|---|
| B01 | P1 reporting | Section 3.8 does not fully identify the TreeSHAP reference semantics or deletion estimand. A reader could interpret the +0.2676 drop as a true-class/global-ranking or calibrated-probability result. | VALID | Frozen configs/shap_stability_faithfulness_v3.json specifies SHAP 0.51.0, tree_path_dependent, no external background, local absolute grouped SHAP ranked for each observation's original predicted class, training-fold numeric median/categorical mode masks, and drop in that same original predicted class's raw-model probability. Add a short exact description in Methods and identify the probability event in §4.4/Table 8 or its caption. Keep the OOD/noncausal caveat. No new number or analysis is needed. |
| B02 | P2 reporting | Section 3.10/Table 9 labels intervals “simultaneous” without stating their nominal level, construction and actual multiplicity family. The displayed subset might be mistaken for the full family. | VALID | Frozen configs/subgroup_proxy_use_v3.json specifies 95% studentized maximum absolute bootstrap-deviation intervals across all estimable declared P3 attribute/threshold/metric gap cells, including support-threshold sensitivity; fixed eligibility and no model-training variability. Add the level and procedure/family to Methods or Table 9 notes. Keep the distinction from confirmatory fairness inference. |
| B03 | P2 construct terminology | The field “Gender” is listed without explaining what the source recorded or whether its semantics were verified. | PARTIALLY VALID | This is a source-schema label, and the draft does not claim biological or gender-identity validity. Still, the current journal guide requests explicit sex/gender use. Add a brief statement that the audit uses source-record categories; collection semantics and construct validity are not established. Do not relabel the values or infer protected status beyond available evidence. |
| B04 | P0 submission | Ethics/consent and full AI/human-review declarations remain placeholders; final code access/licence and author metadata are unresolved. | VALID; already tracked | Main endmatter and declarations/AUTHOR_DECISIONS.md make these open. Preserve the blocked status, obtain documentary decisions, and synchronize exact approved wording. Automated checks cannot supply institutional determinations or human attestations. |
| B05 | P1 final package | An anonymous, independently inspectable evidence route must exist for reviewers, while code access cannot be overstated without a licence and safe history strategy. | VALID; package-dependent | Main §3.12 and data statement promise a ledger/aggregate supplement, while durable software access is explicitly pending. Check the finished supplement's actual source bindings, identifying links/properties and files before approval. Do not call the existing repository open source or make a DOI/release claim. |
| B06 | — | The SHAP stability result proves that the explanations are causal, employee-specific or useful to decision makers. | NOT SUPPORTED | §§3.8, 4.4, 5.5 and Limitations explicitly distinguish numerical validity, ranking agreement, model-level masking and human/causal interpretation. Preserve that separation. |
| B07 | — | The Department gap demonstrates discrimination or removing JobRole establishes fairness. | NOT SUPPORTED | The draft reports all six attributes, restricted eligibility, simultaneous exploratory intervals and distinct refit/permutation/reconstruction questions, with explicit non-fairness/noncausal boundaries. |
| B08 | — | P3→P4 identifies a causal temporal-leakage effect, or HRDataset transports the trained INX model. | NOT SUPPORTED | The draft states timestamp absence, simultaneous information/feature-count changes, nominal-XGBoost-specific policy scope, separately fitted HR models and distinct target estimands. |

## Suggested wording for valid reporting findings

For §3.8, adapt within the frozen claim boundary:

> TreeSHAP used tree-path-dependent raw-margin explanations without a separate background dataset. For deletion, features were ranked per observation by absolute grouped attribution for the original predicted class. Masking used only the corresponding outer-training partition's numeric medians and categorical modes; the reported drop concerns that original class's uncalibrated predicted probability.

For §3.10/Table 9, adapt within the frozen claim boundary:

> The 95% simultaneous exploratory intervals use the studentized maximum absolute bootstrap deviation over all estimable prespecified P3 attribute, support-threshold and metric gap cells. They condition on fixed fitted models, folds and support eligibility and exclude model-training variability.

For source-category terminology, if confirmed against the source descriptions:

> “Gender” denotes the source-record field; its collection process and correspondence to sex or gender identity are not established. Subgroup comparisons describe these recorded categories.

No additional model, fairness metric, conditional-permutation experiment or human study is requested for this bounded submission. The current limitations must remain visible. The claims may be narrower than a deployment contribution; this is consistent with the manuscript's evaluation-protocol contribution.

## Final recheck required

After integration, record the new manuscript hash and disposition of B01–B03. B04–B05 can only close with the required real author/institution evidence and the finished package inspection. This report must not be treated as a final submission pass merely because the other methodological allegations were unsupported.
