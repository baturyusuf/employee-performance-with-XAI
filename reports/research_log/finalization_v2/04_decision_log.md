# Decision Log

## Fixed by User

| ID | Decision | Status |
| --- | --- | --- |
| F-001 | Study identity is leakage-aware XAI audit protocol. | accepted |
| F-002 | LLM/chatbot are excluded from core paper and core canonical pipeline. | accepted |
| F-003 | XGBoost primary; three baselines; XAI required only for XGBoost. | accepted |
| F-004 | Counterfactual is supplementary-only heuristic search if retained. | accepted |
| F-005 | HRDataset_v14 is core external mapped-target replication; IBM/turnover are supplementary. | accepted |
| F-006 | Unverified raw data must not be distributed in the publication repository. | accepted |
| F-007 | Sigmoid is predeclared primary calibration. | accepted |
| F-008 | Use leakage-aware terminology. | accepted |
| F-009 | Ethics information must be user/institution supplied; never invented. | accepted |
| F-010 | No paid API call is allowed. | accepted |

## User Decisions — 2026-07-13

| ID | Accepted decision | Consequence |
| --- | --- | --- |
| D1 | Multinomial Logistic Regression + Random Forest + LightGBM | Compare all three with XGBoost on the same primary-policy folds and metric contract. |
| D2 | Restrained nested tuning | Use shared 10-fold outer CV with 5-fold inner tuning within each outer train partition, versioned search spaces, and exact fold model/parameter records. Do not tune each leakage policy independently. |
| D3 | User-provided local files for INX, HRDataset_v14, IBM and Turnover; sanitized publication repository/bundle | Pin expected SHA/schema/support. Do not delete local raw files. The development repository remains historical; prepare a sanitized export without inheriting raw-data history. |
| D4 | Ethics/IRB application pending | Engineering may proceed; submission remains blocked. The provided institution/unit/reference/date strings were placeholders, not verified metadata, and must not be published as facts. |
| D5 | Small artifacts in Git; full evidence package in GitHub Release/Zenodo; pointer-only `latest` | Do not publish a release yet. Add an export/upload-ready package and release workflow only; actual publication requires explicit approval. |

## Additional Accepted Gates

- If any baseline exceeds XGBoost under paired OOF bootstrap and the confidence interval for baseline-minus-XGBoost excludes zero, stop before model-reference finalization and ask whether XGBoost remains the predeclared XAI reference or the complete XAI pipeline moves to the better model.
- If a local dataset is absent, automatic acquisition may use only a URL explicitly approved in the acquisition manifest. The downloaded bytes must match the pinned SHA-256, schema, row count and target distribution. Any mismatch fails closed, produces a comparison report and requires user direction. Do not try unapproved mirrors.
- Checkpoint commits are authorized only on `finalization/leakage-aware-v2`, after relevant tests pass. No force-push, merge, release publication, history alteration or new scientific-protocol decision is authorized.

## Superseded Decision Gate — Unit 2B Primary Metric (resolved 2026-07-13)

At the preceding checkpoint, D2 did not yet specify the single metric that defines inner candidate selection and the baseline-over-XGBoost stop gate. The implementation at that historical checkpoint kept both fields null and blocked before data/model execution. The options below are retained as decision provenance; the accepted resolution follows immediately afterward.

- Option A (recommended): `macro_f1` for both inner selection and the paired baseline gate. This directly prioritizes all three imbalanced classes.
- Option B: `quadratic_weighted_kappa` for both. This prioritizes ordinal disagreement distance.
- Option C: `macro_f1` for inner selection and macro-F1 plus QWK as multiplicity-controlled co-primary gate metrics. This is more complex and changes the gate implementation/config schema.

Affected files after user decision: `configs/model_grid.yaml`, `configs/manuscript_final.yaml`, benchmark/gate tests, scientific-input hash and all real benchmark artifacts. No option was inferred or executed before the explicit resolution below.

## Accepted Unit 2B Metric and Fold Correction — 2026-07-13

- User selected macro-F1 as the sole primary metric for both inner-CV hyperparameter selection and the baseline-versus-XGBoost stop gate.
- QWK is a reported secondary ordinal metric and may affect selection only as a predeclared tie-breaker among macro-F1 candidates that are tied or practically indistinguishable.
- The gate triggers only when the baseline-minus-XGBoost macro-F1 point estimate is positive **and** the paired OOF bootstrap 95% CI lower bound is greater than zero.
- User corrected the approved nested design to 10 outer × 5 inner folds. The prior 10×3 implementation/preflight is invalid for the approved real benchmark and will not be reused.
- Operational definition recorded and fail-closed before model fitting: candidates within an inclusive absolute inner-mean macro-F1 difference of `0.001` from the best candidate enter the QWK tie-break pool; highest inner-mean QWK wins, then lowest candidate index. This narrow deterministic threshold is an implementation assumption disclosed to the user before execution.

Expected fit count: 310 Logistic Regression fits plus 410 each for Random Forest, LightGBM and XGBoost = 1,540 fits including outer refits. Estimated local runtime: approximately 10–25 minutes. No paid or network service is involved.
