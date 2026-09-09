# Round 2 Final Review Simulation

## Overall recommendation

**Scientific revision status: pass with author-action conditions.** The rewritten manuscript is internally aligned with the approved Round 2 claim boundary and now exposes the principal sensitivities a methods, HR-analytics, or applied-ML reviewer is likely to examine. No scientific overclaim requiring another experiment was found. Submission and release remain blocked by the author/institution items listed below.

## Simulated methods review

The revised Methods are materially more reproducible. They specify training-only preprocessing, the six trained systems, candidate counts and search dimensions, the shared fold design, the 0.001 inclusive tie pool, and the exact difference between macro-F1 and QWK selection. The proportional-odds restriction and cumulative-threshold XGBoost construction are explicit. Supplementary Tables S1–S3 carry the full feature, candidate, seed, and reuse/refit contracts.

Potential concern: the paper still uses public cross-sectional tables without feature timestamps. This is handled appropriately by treating P4/P5 as timestamp-unverified sensitivity analyses and by avoiding prospective-validity language.

## Simulated statistical review

The manuscript distinguishes exactly-once OOF point estimates, five-repetition summaries, bootstrap subgroup intervals, and descriptive CV-range inclusion. Repetition ranges are not called confidence intervals. Target formulations are not subtracted. The 309-row HR comparison correctly holds the evaluation population fixed before attributing changes to exclusion/refitting.

Potential concern: selection-objective sensitivity changes every complete ordering, which could be overinterpreted. The text avoids this by separately reporting 37/60 selected-candidate changes, 6/9 leader changes, 9/9 full-ordering changes, and model-level metric effects. Lower-order swaps are not automatically labelled material.

## Simulated ordinal-modelling review

The main scientific message is appropriately class-aware. Random Forest's QWK 0.6317 and MAE 0.1583 coexist with rating-4 recall 0.0000. QWK-selected nominal XGBoost reaches QWK 0.6418 and MAE 0.1525 with rating-4 recall 0.0076. The manuscript repeatedly states that aggregate QWK/MAE improvement does not guarantee extreme-class success. This resolves the main risk of presenting ordinal scores as sufficient.

Potential concern: the manuscript does not add an MAE- or RPS-selected third regime. This is appropriately identified as a prespecified scope decision, not a missing post-hoc analysis.

## Simulated HR/governance review

All six prespecified subgroup attributes are visible. Department's macro-F1/QWK/MAE gaps (0.2179/0.4388/0.1193) are presented with the QWK eligibility limitation, and no fairness or discrimination conclusion is asserted. JobRole refitting, permutation, and Department reconstruction remain distinct diagnostics. Recorded performance ratings are not equated with objective capability or future potential.

Potential concern: the source and rights chain for both public datasets is unresolved. The Data Availability Statement excludes raw employee rows and does not infer permission from public availability.

## Simulated reproducibility review

The manuscript no longer embeds visible claim codes or a long digest. Traceability is preserved through the Supplementary Evidence Ledger, source register, claim matrix, and deterministic validation outputs. The scientific package remains separate from release identity and legal/author declarations.

## Numerical spot checks

- Canonical leaders: cumulative-threshold XGBoost macro-F1 0.6255; Random Forest QWK 0.6317 and MAE 0.1583; LightGBM RPS 0.0804; nominal XGBoost log loss 0.5515.
- Selection sensitivity: 37/60 selected candidates changed; 6/9 leaders changed; 9/9 full orderings changed.
- Extreme class: Random Forest rating-4 recall 0.0000 in both regimes; QWK-selected nominal XGBoost 0.0076.
- P3→P4: fixed macro-F1/QWK −0.2094/−0.3581; retuned −0.1937/−0.3308; MAE +0.2400 in both.
- Subgroups: all six attributes present; Department gaps 0.2179/0.4388/0.1193; QWK eligible groups 3/6.
- HR mapping: three-class support 31/243/37; four-class support 13/18/243/37; no between-estimand subtraction.
- HR CV design: 11/14 canonical estimates inside repeated ranges; raw macro-F1 and sigmoid Brier/RPS outside.
- HR alias: historical 311, fit-free restricted 309, and refit 309 arms are separate; primary comparison is 309↔309.

## Language and claim-boundary checks

- No policy is described as leakage-free.
- No result is described as proving fairness, discrimination, causality, target equivalence, prospective validity, or deployment readiness.
- The contribution is operational integration, not a world-first or exhaustive-review claim.
- Calibration conclusions are metric-specific; empirical-prior ECE 0.0000 is not called perfect calibration.
- Release, DOI, licence, data rights, ethics/IRB, and author declarations are not inferred from scientific approval.

## Conditions before submission or release

The authors or relevant institutions must resolve every item in `AUTHOR_ACTION_REQUIRED.md`. Until then, the manuscript is scientifically revised but not submission-ready or release-authorized.
