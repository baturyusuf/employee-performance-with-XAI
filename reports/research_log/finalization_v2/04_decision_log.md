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
| D2 | Restrained nested tuning | Use shared 10-fold outer CV, inner tuning within each outer train partition, versioned search spaces, and exact fold model/parameter records. Do not tune each leakage policy independently. |
| D3 | User-provided local files for INX, HRDataset_v14, IBM and Turnover; sanitized publication repository/bundle | Pin expected SHA/schema/support. Do not delete local raw files. The development repository remains historical; prepare a sanitized export without inheriting raw-data history. |
| D4 | Ethics/IRB application pending | Engineering may proceed; submission remains blocked. The provided institution/unit/reference/date strings were placeholders, not verified metadata, and must not be published as facts. |
| D5 | Small artifacts in Git; full evidence package in GitHub Release/Zenodo; pointer-only `latest` | Do not publish a release yet. Add an export/upload-ready package and release workflow only; actual publication requires explicit approval. |

## Additional Accepted Gates

- If any baseline exceeds XGBoost under paired OOF bootstrap and the confidence interval for baseline-minus-XGBoost excludes zero, stop before model-reference finalization and ask whether XGBoost remains the predeclared XAI reference or the complete XAI pipeline moves to the better model.
- If a local dataset is absent, automatic acquisition may use only a URL explicitly approved in the acquisition manifest. The downloaded bytes must match the pinned SHA-256, schema, row count and target distribution. Any mismatch fails closed, produces a comparison report and requires user direction. Do not try unapproved mirrors.
- Checkpoint commits are authorized only on `finalization/leakage-aware-v2`, after relevant tests pass. No force-push, merge, release publication, history alteration or new scientific-protocol decision is authorized.
