# Bounded Novelty Positioning

## Claim

Within this 25-work, source-verified positioning set, no selected work reports the complete shared evidence contract used here.

This is a bounded comparison statement about the frozen positioning set. It is not a claim that no unobserved publication has related components, and it is not a checklist-based world-first claim.

## What the prior literature establishes

The evidence is distributed across separate traditions: same-dataset employee-performance prediction, HR governance and fairness scholarship, SHAP algorithms and explanation-validity tests, leakage/nested-evaluation methods, probability-calibration methods, and reproducibility/reporting frameworks. Those sources individually justify the components used here.

Across all 25 selected works, explicit `yes` counts are: leakage policy 2, nested CV 1, calibration 2, exact-fold explanations 0, SHAP stability 1, subgroup analysis 0, proxy reconstruction 0, second dataset 11, and complete artifact provenance 0.

Among the 9 closest HR empirical/XAI studies, the corresponding explicit `yes` counts are: leakage policy 0, nested CV 0, calibration 0, exact-fold explanations 0, SHAP stability 0, subgroup analysis 0, proxy reconstruction 0, second dataset 1, and complete artifact provenance 0.

These counts are descriptive of coded evidence, not quality scores. A `not_reported` value remains epistemically different from `no`.

## Mechanistic contribution

The contribution is a shared, hash-bound evidence contract connecting prediction-time feature policy, nested selection, exact held-out prediction and explanation identity, training-only calibration, explanation stability and faithfulness, subgroup/proxy claim boundaries, independent replication, and manuscript-number provenance.

The shared contract matters because it prevents evidence assembled in one analytical stage from silently changing identity before another stage cites it. Specifically, it is designed to prevent:

- explanation and evaluated-model identity mismatch
- calibration leakage from held-out outcomes
- confounding feature-policy effects with hyperparameter retuning
- unsupported fairness or discrimination inference from descriptive subgroup gaps
- conflating protected-attribute reconstructability with actual proxy use by the performance model
- untraceable or stale numerical claims in manuscript text

## Dataset-specific prior-art implication

Four verified publications use the exact INX data. The closest registered INX work uses an 80/20 split and a separate validation-curve presentation; another selects the highest accuracy across several train/test proportions. The exact HRDataset_v14 comparator reports a single 80/20 split and does not document removal of the target-alias field `PerfScoreID`. These are design differences, not grounds to dismiss the studies; they explain why the present contribution must be stated as an evidence-identity mechanism rather than another accuracy comparison.

## Prohibited overclaims

- world first
- first ever
- all leakage is eliminated
- fairness is established
- causal employee-performance drivers are identified
- deployment readiness is established

Later manuscript prose must also preserve the existing boundaries: SHAP is model attribution rather than causation; subgroup results are descriptive rather than proof of fairness/discrimination; HRDataset_v14 is an independently trained mapped-target sensitivity rather than locked transport; and public source presence does not establish licence or authenticity.
