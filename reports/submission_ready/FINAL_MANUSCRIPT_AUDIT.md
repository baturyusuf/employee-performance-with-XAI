# Final Manuscript Audit

## Scope

This audit covers the submission-facing manuscript at `reports/submission_ready/main.md`. It does not authorize a public release, DOI, raw-data redistribution, software licence, or unresolved author/institution declaration.

## Scientific-scope corrections

| Final-review item | Status | Finalization action |
| --- | --- | --- |
| 4.1 P0-P5 model scope | PASS | Methods, Results, Table 6 caption, Discussion, Conclusion, and Abstract explicitly identify the analysis as a nominal-XGBoost policy-sensitivity experiment. |
| 4.2 Prespecified vs preregistered | PASS | Methods distinguishes revision-stage freezing from preregistration; Limitations states that the broader program evolved after prior dataset inspection and nested CV does not remove higher-level researcher adaptation. |
| 4.3 Reproducible title | PASS WITH AUTHOR SIGN-OFF | Submission-facing title changed to `Traceable`, consistent with unresolved upstream provenance/release identity. |
| 4.4 HRDataset terminology | PASS | Reworded as `partial cross-dataset protocol replication`; locked-model transport/full replication language removed. |
| 4.5 HRDataset seven features | PASS | Exact seven-feature policy is listed and timestamp verification is explicitly denied. |
| 4.6 Hard-label probability baseline | PASS | RPS/log-loss cells for hard-label baselines are `—`; training-only empirical-prior predictor is the probability reference. |
| 4.7 Hard decision rule | PASS | Unmodified fitted-estimator prediction rule stated; custom ordinal argmax documented; no class-specific threshold optimization and no alternative threshold evaluation stated. |
| 4.8 Per-class Supplement S4 | PASS | Full precision/recall/F1/support for cumulative-threshold XGBoost, nominal XGBoost, LightGBM, and Random Forest added; manuscript points to S4/confusion matrices. |
| 4.9 Subgroup uncertainty | PASS | `See ledger` removed; eligibility counts are visible in Table 8 and detailed exploratory simultaneous intervals are in S6. |
| 4.10 Feature-count confounding | PASS | P3/P4/P5 = 20/13/6 feature dimensionality confounding is explicit in Discussion and Limitations. |
| 4.11 P0 upper-bound wording | PASS | Replaced with `information-rich diagnostic comparator`. |
| 4.12 Internal workflow wording | PASS | `user-approved`, `Round 2 approval`, and `paid API call` wording removed from submission-facing scientific prose; evidence control is expressed academically. |
| 4.13 Journal supplement cleanup | PASS | Reviewer simulations, approval records, revision diffs, and development logs are excluded from the journal-facing supplement definition. |
| 4.14 Operational novelty | PASS | Reusable Protocol/Audit Gates Table S5 added with stage/input/procedure/output/failure/claim-boundary fields. |
| 4.15 Abstract simplification | PASS | Abstract substantially shortened and refocused on selection-objective sensitivity, extreme-class failure, and nominal-XGBoost P3-to-P4 information sensitivity. |

## Evidence checks used

- Policy-retuning contract fixes the model family to XGBoost, candidate count to 8, and retained feature counts to 26/24/21/20/13/6.
- HRDataset sensitivity contract fixes the seven exact features used in the partial cross-dataset analysis.
- Per-class values were copied from the validated per-class/extreme-class report.
- Subgroup point estimates, eligibility counts, and intervals were copied from the publication-facing subgroup summary.
- Hard-decision wording was checked against the experiment/model implementation: outer hard labels use the fitted estimator's prediction rule; custom ordinal estimators use probability argmax; no class-specific threshold-optimization path is present in the declared benchmark.

No new model family, dataset, seed search, hyperparameter optimization, fairness metric, XAI method, calibration method, or score-improvement experiment was introduced.

## Journal-facing scientific package

- Finalized manuscript: PASS
- Supplement S4/S5/S6: PASS
- Existing S1/S2/S3 scientific contracts: PASS as source artifacts; journal-package copies must remain byte-identical to the validated originals
- Confusion-matrix evidence: PASS as existing source artifact
- Evidence ledger: PASS as existing versioned source artifact
- Cover letter draft: PASS, pending author assertions
- Data Availability draft: PASS, pending rights sign-off
- AI disclosure draft: PASS, pending author sign-off

## Items not honestly closable without external sign-off

- Complete postal affiliations, author order confirmation, ORCID, final corresponding-author fields
- Final CRediT roles
- Funding and conflicts declarations
- Acknowledgments
- Institutional ethics/IRB determination and informed-consent wording
- Dataset redistribution/legal clearance
- Software licence selection
- Destructive historical Git remediation decision, if required
- Final public release/tag/archive/DOI approval

## Template/PDF status

The current official MDPI LaTeX page identifies template bundles updated 23 June 2026. The exact official bundle is not stored in the repository and could not be safely substituted with an older or third-party copy during this pass. Therefore:

- official-template application: **BLOCKED**
- official-template PDF compile: **BLOCKED**
- visual journal-layout verification: **BLOCKED**

No `PDF verified` claim is made.

## Current editorial decision

**NOT SUBMISSION READY**

The scientific manuscript finalization is complete, but submission remains blocked by author/institution declarations and the exact current official MDPI template/PDF gate. No additional scientific experiment is indicated by this audit.