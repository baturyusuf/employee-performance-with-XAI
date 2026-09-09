# Round 2 Response to Reviewers

This point-by-point response records the substantive Round 2 revisions. The headings summarize the scientific issues addressed; they are not presented as verbatim reviewer quotations. All numerical statements are confined to the approved Round 2 claim boundary. Release, DOI, dataset rights, software licensing, ethics/IRB, and author declarations remain outside this approval.

## 1. The ordinal task was tuned with a nominal objective

**Response.** We added a prespecified selection-objective sensitivity that reverses macro-F1 and QWK while keeping the P3 features, candidate registries, outer/inner fold identities, seeds, tolerance, and evaluation metrics fixed. Candidate selections changed in 37/60 model-fold decisions, metric leaders changed for 6/9 metrics, and the full six-model ordering changed for 9/9 metrics. These are now reported as separate facts rather than one binary “ranking depends” conclusion.

**Evidence and manuscript location.** Selection-objective evidence is in `SELECTION_OBJECTIVE_SENSITIVITY.md` and its validated compact package. Methods Section 3.5 specifies the two regimes; Results Section 4.2 and Table 4 report system-level effects; Discussion Section 5.1 limits their interpretation.

## 2. Aggregate metrics concealed extreme-class failure

**Response.** We added rating-4 precision/recall/F1 evidence and made the limitation visible in the Abstract, Results, Discussion, and Conclusions. Random Forest led canonical QWK (0.6317) and ordinal MAE (0.1583) while rating-4 recall was 0.0000. Under QWK selection, nominal XGBoost reached QWK 0.6418 and MAE 0.1525 while rating-4 recall was 0.0076. We therefore state explicitly that aggregate QWK or MAE gains do not guarantee class-4 success.

**Evidence and manuscript location.** The source is `PER_CLASS_EXTREME_CLASS_REPORT.md` and the selection-objective package. The revised account appears in the Abstract, Results Sections 4.1–4.2, Table 4, Discussion Section 5.1, Limitations, and Conclusions.

## 3. Timing and information availability were peripheral

**Response.** The P3→P4 transition is now a main result. Macro-F1/QWK changed by −0.2094/−0.3581 under the fixed schedule and −0.1937/−0.3308 under independent retuning; ordinal MAE increased by 0.2400 in both. The P4→P5 step is also reported. P4 and P5 are consistently labelled timestamp-unverified sensitivities, not prospective validation.

**Evidence and manuscript location.** The comparison is source-bound in `headline_policy_comparison.csv`. Policy definitions appear in Methods Sections 3.3 and 3.7; Results Section 4.4 and Table 6 contain the estimates; Discussion Section 5.2 states the boundary.

## 4. Subgroup reporting was incomplete

**Response.** We now report every prespecified P3 subgroup attribute at the n≥30 rule, not only the largest gap. Age, Gender, Marital Status, Business Travel, Department, and Education each have visible macro-F1, QWK, and MAE gaps. Department had the largest reported gaps (0.2179, 0.4388, and 0.1193), but only 3/6 Department groups were eligible for QWK. The text explicitly rejects fairness, discrimination, and legal-compliance inference.

**Evidence and manuscript location.** Publication-ready endpoints, interval fields, supports, and eligibility counts are in `SUBGROUP_MANUSCRIPT_SUMMARY.csv`. The design is in Methods Section 3.10; all six attributes appear in Results Section 4.6 and Table 8; interpretation is in Discussion Section 5.3.

## 5. The HR target mapping was treated as fixed

**Response.** We added a target-formulation sensitivity that retains the three-class support 31/243/37 and separately evaluates a four-class support 13/18/243/37. Raw and sigmoid results are shown side by side. Because the formulations define different estimands, we do not subtract them or label one an improvement over the other.

**Evidence and manuscript location.** The source is `HR_MAPPING_SENSITIVITY_REPORT.md` and the validated target-formulation package. The estimands are defined in Methods Sections 3.2 and 3.11; Results Section 4.8 and Table 10 report both; Discussion Section 5.4 explains noncomparability.

## 6. Cross-validation design sensitivity was not visible

**Response.** We compare the canonical 10×5 estimates with ranges from the repeated 5×5 design. Eleven of 14 estimates fell inside those ranges; raw macro-F1 and sigmoid Brier and RPS were outside. We describe range inclusion as a stability diagnostic, not as a confidence interval or equivalence test.

**Evidence and manuscript location.** The analysis is in `HR_MAPPING_SENSITIVITY_REPORT.md`. Methods Section 3.11 defines the comparison; Results Section 4.8 reports the 11/14 finding and three exceptions; Discussion Section 5.4 and Limitations retain the boundary.

## 7. The modelling methods were not sufficiently self-contained

**Response.** We added training-partition-only preprocessing, all six trained-model candidate counts and search dimensions, outer-test exclusions, the proportional-odds shared-slope restriction, and cumulative-threshold XGBoost probability construction. The latter now states the independent `P(Y>2)`/`P(Y>3)` fits, rowwise nonincreasing PAVA correction, class-probability differencing, and normalization.

**Evidence and manuscript location.** Methods Sections 3.4–3.6 provide the narrative. Supplementary Tables S1–S3 give all governed features, exact candidate registries and fixed settings, folds, seeds, OOF coverage, calibration isolation, and reuse/refit boundaries.

## 8. Policy terminology was stronger than the evidence

**Response.** We froze manuscript-facing labels as Information-Rich Diagnostic, Leakage-Risk-Controlled, Direct-Sensitive-Attribute-Excluded, Primary Leakage-Aware, Prospective-Plausibility, and Strict Proxy-Reduced Prospective-Plausibility. No policy is called leakage-free; P4/P5 remain timestamp-unverified and P3 is not described as confirmed leakage.

**Evidence and manuscript location.** The labels and boundaries appear in Methods Section 3.3, Table 2, Results Section 4.4, Discussion Section 5.2, and Limitations.

## 9. Probability baselines and ECE could be misread

**Response.** A training-only empirical-prior comparator was added. It produced log loss 0.7683, Brier 0.4313, RPS 0.1167, and top-label ECE 0.0000. The manuscript explains that zero top-label ECE for a constant prior is not perfect calibration and retains complementary proper scores. It also reports that sigmoid calibration improved several metrics while the top-label ECE point estimate worsened.

**Evidence and manuscript location.** Evidence is in `PROBABILITY_BASELINE_REPORT.md` and the canonical calibration package. Methods Sections 3.5 and 3.9 define the metrics; Results Sections 4.2 and 4.5 and Table 7 report them; Discussion Section 5.5 states the limitation.

## 10. Core ordinal and model references were incomplete

**Response.** Nine verified primary-source references were added for proportional odds, ordinal threshold reduction, ordinal evaluation, weighted kappa, ranked probability scoring/proper scoring rules, Random Forest, XGBoost, and LightGBM. The contribution statement is bounded to operational integration into a traceable evaluation contract; no world-first or exhaustive-review claim is made.

**Evidence and manuscript location.** Verification records are in `LITERATURE_V4`. Related Work Section 2.2 and Methods Sections 3.4–3.5 cite the additions, and Section 2.6 contains the bounded positioning statement.

## 11. HR target-alias disagreements could mix sample and refit effects

**Response.** The historical 311-row result is preserved. We first remove the two disagreement rows from the existing canonical predictions, giving fit-free 309-row metrics. We then compare those predictions with independently selected/refitted predictions on the identical 309-row population. The primary matched-population changes were +0.0042 macro-F1, −0.0003 QWK, 0.0000 MAE, +0.0108 log loss, and −0.0112 ECE. The separate 311→309 changes are labelled only as sample-removal effects.

**Evidence and manuscript location.** The validated evidence is in `hr_target_alias_sensitivity` and `HR_MAPPING_SENSITIVITY_REPORT.md`. The design is explicit in Methods Section 3.11; Results Section 4.9 and Table 11 separate the three arms; Discussion Section 5.4 preserves the interpretation.

## 12. Claim identifiers and hashes impaired readability

**Response.** Visible claim IDs and the full claim digest were removed from ordinary manuscript prose. A short evidence-control statement remains in Methods Section 3.12. Complete identifiers, source selectors, paths, values, and hashes are retained in the Supplementary Evidence Ledger and validation package.

## 13. The abstract did not reflect the governing sensitivities

**Response.** The Abstract now reports selection-objective changes, extreme-class failure, the P3→P4 fixed and retuned contrasts, all six subgroup coverage with the largest Department gaps, both HR target formulations, and the 11/14 CV-design result. It closes with the conditional interpretation and unresolved timestamp, construct, rights, and prospective-validity limits.

## Remaining non-scientific blockers

This revision does not resolve or imply approval for release/tag creation, DOI assignment, dataset redistribution rights, a software licence, ethics/IRB wording, consent wording, author contributions, funding, conflicts of interest, affiliations, or other author declarations. Those items remain in `AUTHOR_ACTION_REQUIRED.md` and must be completed by the appropriate authors or institutions before submission or release.
