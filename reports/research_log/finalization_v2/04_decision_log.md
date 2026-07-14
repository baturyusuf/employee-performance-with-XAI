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

## Unit 2D Matched Policy Consequence — 2026-07-13

Accepted D2 already states “Do not tune each leakage policy independently.” Unit 2D therefore holds the primary policy's fold-selected XGBoost parameter schedule fixed across all six exact policies and uses the exact shared outer folds. The primary policy reuses the prediction-producing benchmark OOF rows; the five other policies are refit only on each outer-training partition. This is reported as matched feature-access sensitivity conditional on primary-policy hyperparameter selection, not as a fully optimized policy leaderboard. The information-rich full-feature comparator is diagnostic/audit-only and not guaranteed to be an empirical upper bound.

All policy uncertainty uses the same 5,000 paired, outer-fold-plus-class-stratified sample-level resample plan as the benchmark. Fold mean/SD/min/max are descriptive only; no t interval, Wilcoxon, Holm-adjusted rejection or policy stop gate is permitted. This followed D2 and did not require another user decision.

## Pending Calibration Training Decision — 2026-07-13

Read-only audit proved that the legacy calibration stage selects methods from outer-test metrics, regenerates incompatible folds, uses legacy fixed parameters and fold-t intervals. Three implementation choices remain:

1. recommended five-inner-fold OOF cross-fitted sigmoid calibration refits inside the calibration stage, then application to exact persisted outer-fold XGBoost probabilities (about 2–3 minutes);
2. the same cross-fitted predictions captured and persisted by the benchmark stage, avoiding calibration-stage refits but requiring a broader benchmark artifact-contract change and final benchmark regeneration;
3. single 20% outer-training calibration holdout with a separately fitted base model trained on only 80% of the outer-training rows (about 1.5–2 minutes; scientifically simpler but less directly comparable).

Outer test data is evaluation-only in every design. Isotonic is excluded from core selection. Implementation is paused for this material protocol choice.

## Accepted Calibration Option A — 2026-07-13

The user selected five-inner-fold cross-fitted sigmoid calibration. Within each outer-training partition, the exact persisted shared inner-fold assignments must generate one raw probability row per outer-training sample from a model that did not train on that sample. The fold-specific sigmoid calibrator is fitted only on those inner-OOF probabilities and labels. It is then applied to the exact selected benchmark outer-fold XGBoost model's untouched outer-test probabilities; that benchmark model is never refit or replaced.

Outer-test features, labels, predictions and metrics are evaluation-only. They cannot influence hyperparameter tuning, calibrator fitting, calibration-method selection or threshold selection. Sigmoid is predeclared primary, raw is the comparator, isotonic is excluded from the core stage, and class labels use the fixed argmax rule rather than a selected threshold. Every calibrator must record the corresponding outer fold, selected candidate, benchmark model SHA-256, model-set/fold/run/config/scientific-input identities and cross-fit evidence hash.

## Observed Unit 2B Gate Outcome — 2026-07-13

Trial `benchmark-10x5-20260713-6a80074` completed all 1,540 fits and 5,000 paired OOF bootstrap draws. No baseline had both a positive macro-F1 point advantage over XGBoost and a paired 95% CI lower bound above zero. The gate did not trigger, so the approved plan continues with XGBoost as the predeclared XAI reference. This does not suppress secondary results: Random Forest had the highest QWK, but QWK was not a gate metric.

## Unit 2C-A Exact-Model SHAP Engineering Decision — 2026-07-13

- Canonical SHAP loads each exact prediction-producing XGBoost outer-fold pipeline and performs no model/preprocessor fit or split generation.
- OOF replay uses the same run/config/scientific-input/fold identity and `1e-12` probability tolerance in the canonical consumer.
- One-hot grouping requires fitted feature-name lineage and exact transformed-index ownership; no name-parsing or positional fallback is allowed.
- The 45 fold pairs are dependent descriptive comparisons. Mean, SD, median and range are reported; confidence intervals are explicitly inapplicable unless independent repeated-CV units are introduced later.
- The historical trial remains immutable and can verify the reader at its prior `1e-6` probability contract, but its old probabilities/config and missing nested one-hot feature-name lineage prohibit canonical SHAP reuse. Benchmark and SHAP must be regenerated together under the final clean commit.

This is an implementation/provenance resolution of the already approved same-OOF-model XAI scope. It does not change the user-selected model, target, metric, dataset or claim scope and therefore required no new user decision.

## Unit 2E Reproducibility and Execution Timing Decision - 2026-07-13

- The calibrated-probability implementation uses the warning-free scikit-learn `>=1.8,<1.9` L2 contract (`l1_ratio=0.0`) under a one-thread solver limit. Config, requirements, calibrator parameters and protocol hashes record this contract.
- Persisted benchmark probabilities are validated within their upstream float32 simplex tolerance but are not silently renormalized; sigmoid outputs are explicitly normalized. CSV readback uses round-trip float parsing so parameters and probabilities replay exactly.
- Calibration uncertainty is frozen to the same 5,000 paired sample-level draws, 95% confidence, `(outer_fold, y_true)` strata and linear percentile quantiles as the benchmark.
- All upstream files and model bytes are re-read/replayed immediately before publication. A mutation during the 50-fit run blocks atomic rename.
- No historical hash is patched or relabelled. Current config `d755ecc3...` and historical benchmark config `7e70bf66...` are intentionally incompatible.
- Another 1,540-fit benchmark is deferred until fairness/external/figure and dependency side inputs are frozen. Running it earlier would knowingly create another noncanonical package after later config changes. This is an execution-order safeguard, not a change to the accepted model/folds/metric/calibration protocol.

## Unit 2G Conservative External Policy Consequences - 2026-07-13

- Policy A is executable without changing the mapped target: 311 rows and class support 31/243/37 remain intact.
- The primary model has exactly seven features. Salary, State, Zip, RecruitmentSource, direct department/position/status/marriage/diversity aliases, sensitive fields, identifiers and raw dates are forbidden. Proxy-rich and temporality variants are explicitly audit-only.
- Two invalid negative date-derived tenure values are set missing under a schema-count contract; raw dates remain excluded and fold-local preprocessing performs imputation. This is deterministic data-quality handling, not an outcome or feature-policy relaxation.
- Department proxy reconstructability cannot be estimated under exact outer folds because of a singleton class. No class merge/drop or alternative estimand is authorized; the result remains explicit insufficient support.
- Scope and scientific-input hashes are recomputed inside the stage, with exact dataset and side-input key equality. HEAD, source tree and worktree cleanliness are actual checks, not trusted manifest labels.
- No new user decision was required because these choices enforce the accepted conservative policy, fixed statistical protocol and fail-closed provenance boundary.

## Unit 2G Final Trust and Figure-Scope Consequences - 2026-07-13

- The verified raw HR schema spelling `DateofHire` is the single case-insensitive forbidden-name representative. `DateOfHire` is not a second contract entry; the case-insensitive scan still excludes either spelling. This repairs an internal alias contradiction without changing the approved raw-date exclusion.
- Exact-fold diagnostic replay uses the same `aligned_predict_proba` implementation as OOF generation. The `1e-12` replay tolerance remains unchanged; no tolerance was relaxed to accommodate XGBoost float32 output.
- A pre-existing lock is never deleted or taken over automatically. A human must verify and archive/remove a stale lock. This conservative choice prevents duplicate writers and does not change scientific scope.
- Complete reuse and promotion require lock absence and `release_ready:true`; the builder's own final check alone may inspect the package while holding the exact owned token. Promotion remains a separate pointer-only operation after both scopes pass.
- Under the unattended Option-A rule, the replacement main-figure plan is fixed as: study design/leakage-aware pipeline; feature-policy trade-off; four-model comparison; predeclared sigmoid calibration; global grouped OOF SHAP; SHAP stability; and HRDataset_v14 mapped-target replication summary. Deterministic representative local OOF explanations remain table/supplementary evidence. LLM/chatbot/agent dashboards remain excluded. This resolves the plan decision only; figure generation is still pending.

## Unit 2G Recovery and Publication Consequences - 2026-07-14

- Independent closed-world and scientific replay audits validate the completed atomic external stage under its recorded source identity. Its provisional outer input manifest and absent package-level manifests correctly prevent canonical promotion. The stage is retained; no duplicate Unit 2G rerun is scientifically warranted.
- Sigmoid is still the predeclared calibration method, but reporting separates probability quality from class decisions. Log loss/Brier improve while macro-F1 falls and class 4 receives zero sigmoid argmax predictions. No blanket calibration-improvement claim is permitted.
- The temporality-restricted audit's large performance loss is a primary limitation. Engagement/attendance features remain timing-unverified; no causal, prescriptive or deployment interpretation follows.
- The validated stage exposed two explicit metadata improvements for its future canonical regeneration: subgroup evidence must name raw OOF/source identity, and SHAP summaries must name raw-margin attribution units. Neither changes the validated numerical computation.
- The full noncanonical package was accidentally pushed in `e25f403`. D5 is restored at the current tip through a normal index-only forward removal while every local file is preserved. History is not rewritten; historical blob retention remains documented.
