# Final Readiness Report - Interim Checkpoint

Assessment date: 2026-07-14

Current state: Units 1A through 2G and V2-029 have tested implementation commits; V2-029 is local checkpoint `9c60353`. The full Unit 2G verified-real-data stage completed from clean source commit `17a3dcd` and passed independent atomic/scientific validation. It remains noncanonical stage evidence; no clean complete v2 scientific rebuild exists.

This is a live interim assessment, not a final completion declaration. `../finalization/CURRENT_STATUS.md` is the interruption-resilient status source.

## Engineering readiness

**Not ready.** Actual-input/side-input binding, core/supplementary isolation, shared-fold benchmarking, exact-model OOF SHAP, matched policy ablation, cross-fitted sigmoid calibration, support-aware subgroup/proxy contracts, conservative HRDataset replication, and closed-world lock-aware completion/promotion are implemented and tested. Unit 2G passes closed-world hashes, model/calibrator replay, OOF, bootstrap, SHAP, support and claim-boundary validation. V2-029 additionally pins raw-OOF subgroup source identity and raw-margin SHAP units; its expanded external gate passes 80 tests. The current full gate passes 656 pytest tests with 3 skips and 11 subtests plus 178 unittest tests with 2 skips. Figures/tables, dependency locking, CI/release workflows, remaining supplementary implementation and final all-stage integration remain open.

## Scientific readiness

**Not ready.** The verified noncanonical four-model trial did not trigger the baseline superiority gate, so XGBoost remains the predeclared XAI reference. The full HRDataset policy-A stage completed with 10 outer x 5 inner tuning, exact-model OOF SHAP and 5,000-draw uncertainty. Raw conservative macro-F1 is `0.666355`; sigmoid improves log loss/Brier but lowers macro-F1 and predicts no class-4 argmax cases; the temporality-restricted audit shows a large performance loss. These are stage-validation findings, not frozen manuscript claims. Department proxy estimation is unsupported because singleton-class training support is insufficient. No same-config complete canonical core package exists.

## Reproducibility readiness

**Not ready.** Scoped inputs, side inputs and cache identities pass real-input preflight. Same-run sibling cleanliness, exclusive locks, exact commands, semantic receipts, closed-world inventories and pointer promotion are fail-closed. Unit 2G demonstrated atomic closed-world publication on real data, but its outer input manifest is provisional and it has no package-level completion receipt. No clean cache-disabled all-stage v2 package exists, so the release contract has not yet been demonstrated on a complete core/supplementary package.

## Data/provenance readiness

**Blocked.** Pinned user-provided/acquisition contracts pass for current local inputs, but raw-data distribution and source/licence authenticity still require the accepted sanitized-publication workflow and manual verification. The 65.4 MB noncanonical Unit 2G package is preserved locally and removed from local tip `b7b2ad3` under D5. Its normal push timed out, so the remote tip still points to `e25f403`; that commit also remains in history because rewriting/force-push is prohibited. No ambiguous licence/source judgement has been invented.

## Ethics readiness

**Blocked by explicit manual item.** Institution, ethics/IRB unit, application/reference number and application date remain unknown because the supplied values were placeholders. The declared scope is pre-existing secondary HR data with no recruitment, intervention, contact or human evaluation.

## Manuscript-support readiness

**Not ready.** Unit 2G stage findings are logged, but no canonical v2 source-of-truth results, final tables, final figures or claim matrix exists. Explicit raw-OOF subgroup source identity and raw-margin SHAP units are implemented and tested, but have not yet been emitted by a canonical rebuild. Manuscript editing remains prohibited.

## Remaining blockers

See `02_issue_register.csv` and `08_manual_submission_blockers.md`. D1-D5, cross-fitted sigmoid, HRDataset policy A and the Option-A replacement figure plan are accepted. The real HRDataset stage and artifact audit are complete; V2-029 is code/test complete; the tested D5 cleanup remains unpushed due Git authentication timeout. Next are the frozen core figure-plan contract, remaining supplementary science, generator implementation, dependency/CI contracts, sanitized publication transition and final integration. Ethics and source/licence authenticity remain external manual gates; claim-matrix author approval occurs only after the final package exists.

## Final recommendation

**NO-GO for submission.** Continue the authorized implementation. Existing v1, trial and Unit 2G stage-validation artifacts must not be promoted or supply frozen v2 manuscript numbers. The canonical entrypoint must remain fail-closed until all retained core and supplementary stages can be regenerated and verified under one clean commit/config/run identity.
