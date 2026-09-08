# Final Review Simulation

## Editorial recommendation

**Major revision before submission readiness.** The scientific manuscript is substantially stronger and internally bounded, but non-scientific submission blockers and compiled-template validation remain open.

## Reviewer A — Machine-learning methods

**Strengths:** Prespecified information policies; nine-system common-fold benchmark; exact OOF accounting; repeated nested selection; distinction between fixed and retuned estimands; metric-specific conclusions.

**Remaining concerns:** Five repetitions give bounded descriptive variability rather than broad uncertainty. No timestamped prospective cohort exists. P0 must remain visually and verbally marked diagnostic-only in every future derivative.

**Assessment:** Scientific methods are coherent for a cross-sectional audit study. Claims must not drift toward prospective prediction.

## Reviewer B — Explainable AI and calibration

**Strengths:** Exact fold-model/explanation identity; raw-feature grouping contract; separate stability and deletion diagnostics; multiple calibration views; adverse top-label-ECE result retained.

**Remaining concerns:** Stability pairs are dependent; deletion masking may be out of distribution; no human evaluation; ECE depends on bins and sparse support.

**Assessment:** Acceptable as model-diagnostic XAI, not as causal explanation or human-usefulness evidence.

## Reviewer C — HR governance and ethics

**Strengths:** Organizational-rating construct is named; subgroup and proxy questions are separated; fairness/discrimination/legal claims are withheld; intended use excludes autonomous employment decisions.

**Remaining concerns:** Source authenticity and rights remain incomplete; institutional review and consent wording are unresolved; no real organizational workflow or stakeholder evaluation exists.

**Assessment:** Governance framing is appropriately cautious, but declarations must be institutionally resolved before submission.

## Reviewer D — Reproducibility and reporting

**Strengths:** Digest-specific claim approval; source selectors and hashes; aggregate compact packages; raw/row-level exclusions; Markdown/LaTeX generation; verified citation set.

**Remaining concerns:** No software licence; raw paths remain in Git history; no final immutable release/archive/DOI; no local MDPI-template PDF build.

**Assessment:** Computational traceability is strong, while public-release and typesetting readiness remain incomplete.

## Claim-boundary stress test

| Tempting statement | Decision | Safe replacement |
| --- | --- | --- |
| P3 eliminates leakage | Reject | P3 is leakage-aware under timestamp-unverified assumptions |
| Random Forest is best | Reject | Random Forest leads QWK and ordinal MAE under P3 |
| SHAP identifies performance drivers | Reject | TreeSHAP attributes fitted-model raw margins |
| Calibration improved | Qualify | Several point estimates improved; top-label ECE worsened |
| JobRole causes department bias | Reject | JobRole is associated with output sensitivity and department reconstructability under declared diagnostics |
| External validation succeeded | Reject | HRDataset_v14 independently replicated the mapped-target protocol |
| Data are public, so redistribution is permitted | Reject | Public presence does not establish rights; raw redistribution remains unapproved |

## Release decision

The revised scientific narrative and claim traceability pass Phase 5B structural review. Submission/public-release readiness does not pass until declarations, data rights, software licensing, history handling, final release identity, archival citation, and template compilation are resolved.

