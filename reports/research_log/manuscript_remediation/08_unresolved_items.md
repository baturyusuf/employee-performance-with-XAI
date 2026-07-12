# Unresolved Items

The canonical engineering/evidence build is complete. The remaining items require author judgement, external verification, or separately approved paid work; they are not silent pipeline defects.

## Author or External Review Required

1. **Leakage terminology/title.** The code reports exact exclusions, common-fold deltas, and leakage-sensitivity indices. Whether the manuscript/title says *leakage-safe*, *leakage-aware*, or *leakage-reduced* remains the author's decision; this remediation did not edit the manuscript.
2. **Dataset source/licence authenticity.** Dataset hashes, mirror URLs, mappings, support, and repository dates are recorded. Retrieval/authenticity/licence/citation fields that cannot be established by code remain `manual_review_required`; no legal or provenance judgement was assumed.
3. **Canonical real-LLM regeneration.** All 80 selected cases now have complete canonical evidence, but no paid canonical LLM batch was authorised or run. Any manuscript claim about real-LLM faithfulness under the new canonical evidence contract requires a separately costed and explicitly approved batch.
4. **Raw INX workbook provenance.** The canonical CSV is executable and hashed. The historical `.xls` workbook was not independently parsed in the current environment because `xlrd` is absent; the clean-environment dependency specification now declares it for future source-equivalence review.
5. **Actual OpenAI charge for the safety incident.** The preserved repository ledger estimates USD 0.1096371 for 24 unintended pre-safeguard test calls. The provider billing dashboard is authoritative and should be checked by the user.

## Historical Limits That Cannot Be Repaired Retrospectively

- The exact dirty working tree that generated the June historical candidate artifacts cannot be reconstructed from the recorded commit.
- Historical files lacking a config hash cannot be proven compatible and remain indexed as `historical_not_admitted_to_canonical_package`.
- The historical paid 80-case LLM output cannot acquire missing case evidence retroactively and is not part of the canonical package.
- Two remediation build attempts are intentionally preserved with `status=failed`; neither is referenced by `latest` or admitted to the successful final manifest.

## Claim/Use Blockers That Remain by Design

- No autonomous HR decision use.
- No deployment-readiness claim.
- No fairness guarantee or causal SHAP/counterfactual claim.
- No direct employee-performance validation claim from attrition, turnover, or restricted-target evidence.
- No locked INX-model transport claim for HRDataset_v14; the verified safe overlap contains only three features and the canonical result is independent target replication.
