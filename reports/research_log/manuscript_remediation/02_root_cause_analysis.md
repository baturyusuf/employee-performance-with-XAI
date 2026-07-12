# Root-Cause Analysis

## Executive Diagnosis

The repository contains substantial research functionality, but it evolved as a sequence of independently executed stages rather than as one immutable evidence build. The central failure is not a single incorrect metric; it is the absence of a run contract that binds policy semantics, data, folds, model settings, code identity, configuration, artifacts, claims, and hashes.

## Root Cause 1: Feature Policy Is Contextual Rather Than Canonical

`configs/feature_sets.yaml` names the primary policy, while `split_features_and_target(..., drop_sensitive=True)` independently removes Gender and MaritalStatus. Age is encoded in the feature-set drop list. Legacy grouped SHAP has a third hard-coded definition that removes only salary hike, attrition, and optionally department. As a result, the same string can mean different predictor sets depending on caller flags and module.

Evidence:

- `configs/feature_sets.yaml`: primary policy omits Gender and MaritalStatus from its explicit `drop` list.
- `src/data/preprocess.py`: `drop_sensitive=True` removes `SETTINGS.fairness_sensitive_columns` separately.
- `reports/model_card/hr_xai_model_card.md`: claims Gender and MaritalStatus are part of the primary exclusions.
- `src/explainability/xgboost_grouped_shap.py`: owns a hard-coded policy dictionary that omits Age.
- Historical primary-named grouped SHAP metadata and importance include Age.

Required correction: a policy name must resolve to one explicit feature list without caller-dependent flags. Audit-only sensitive fields remain available in the raw audit frame but never enter primary-model preprocessing, SHAP, or reason codes.

## Root Cause 2: Stage Outputs Preceded Commit/Configuration Freezing

The June 5 registry captured commit `6f9df0a`, but the relevant new source/config/output files were uncommitted at run time and were committed as `c703039` minutes later. Registry commit identity therefore points to a tree that cannot generate the outputs. Stage metadata does not contain a config content hash or source-tree-dirty digest.

Required correction: refuse a manuscript build when provenance requirements are unmet; record Git commit and dirty state, canonical config hash, dataset hashes, package versions, command, timestamps, seeds, stage status/failure, and every output hash in one run manifest.

## Root Cause 3: Reports Are Composed Across Independent Run Scopes

Calibration, SHAP stability, local reason codes, counterfactuals, fairness, model selection, external experiments, LLM evaluation, agents, guardrails, and the evidence manifest each have different timestamps/run identifiers or no run identifier. The final manifest is internally hash-correct for the files it lists, but it adopts the 80-case LLM run as its source identity and then references ML evidence generated elsewhere.

Required correction: every canonical stage must run beneath one `reports/manuscript_final/<run_id>/` root, inherit one config hash, and publish stage metadata tied to the same manifest. Cached stages may be reused only if their input contract hash matches.

## Root Cause 4: Metric Generation Is Task-Agnostic

The shared classification metric implementation applies ordinal measures to any numeric labels. For a 0/1 task, `abs(y_true-y_pred)>1` can never occur, so severe-error rate is always 0.0. That computational fact is mistakenly serialized as a scientific result. External report generators then place binary and ordinal tasks in a common comparison table.

Required correction: task type must determine metric applicability before calculation and serialization. Binary related tasks receive binary metrics; ordinal performance tasks receive ordinal metrics; restricted 3/4 robustness is explicitly non-comparable to the 2/3/4 primary task.

## Root Cause 5: Explanation and Counterfactual Case Generation Uses In-Sample Models

Final local reason codes and counterfactual cases fit on the full INX dataset and predict the same observations. Counterfactual scale estimates and desired-class prototype pools also use the full dataset. Representative-case selection therefore benefits from in-sample confidence, and the validity denominator is only four upward-eligible cases per policy.

Required correction: preserve fold assignments, use the fold-specific model trained without the evaluated case, derive scale/domain/prototype information from the fold training portion only, and aggregate over a defensible OOF case population with explicit denominators and uncertainty.

## Root Cause 6: LLM Completeness Was Not a Real-Run Gate

The expanded sampler starts with ten INX cases having saved reason codes, then fills the requested quota from OOF predictions. For fill cases it constructs `CompleteCaseEvidence` objects containing report-level defaults and a source string that local SHAP/reason-code evidence is unavailable. The run still calls the real LLM, and the faithfulness checker can score compliance with that incomplete evidence. Consequently, perfect text compliance coexists with 30 `evidence_missing` readiness outcomes.

Required correction: schema completeness and evidence readiness must be separate from response compliance. A real-run preflight must block incomplete required evidence unless missingness is a predeclared experimental stratum. Existing paid outputs remain historical; no new paid calls are authorized by remediation work.

## Root Cause 7: Deterministic Safety Suite Mirrors Rule Vocabulary

The guardrail prompts are embedded next to the evaluation runner and use direct English phrases such as fire, promote, rank, ignore warnings, and sensitive attributes. This is useful unit coverage but not a broad adversarial evaluation. Perfect results are predictable when prompts strongly overlap deterministic routing vocabulary.

The routing order compounds the coverage problem: safe allow-list expressions are evaluated before unsafe expressions, so mixed-intent prompts can be incorrectly allowed when they contain both safe audit language and a prohibited decision request.

Required correction: version prompts independently, expand semantic and multilingual strata, add retrieval/conflict failure modes, and report category denominators and Wilson intervals with a non-comprehensive-safety limitation.

## Root Cause 8: Support Context Is Lost During Fairness Reporting

Low-level outputs contain group rows and warnings, but summary composition selects maximum gaps without carrying minimum group support, valid bootstrap replicate count, or an instability category. This makes gaps of 1.0 visually dominant even when driven by sparse strata.

Required correction: the manuscript-facing row is invalid unless support, interval, bootstrap validity, audit category, and limitation fields travel with the gap.

## Root Cause 9: Dataset Provenance Is Narrative and Incomplete

Dataset cards contain useful URLs and claim boundaries, but not all required structured fields. Source mirrors and licensing cannot be authenticated by code alone. Missing dates/licence/citation status currently do not block packaging.

Required correction: validate completeness automatically while representing ambiguous source/licence authenticity as `manual_review_required`. Do not silently promote a mirror to an authenticated original source.

## Root Cause 10: Tests Validate Components, Not the Scientific Contract

The baseline suite passes 100 tests, but those tests largely confirm component behavior. They do not scan canonical artifacts for forbidden features, enforce task metric applicability, prove OOF counterfactual construction, block incomplete real-LLM cases, require dataset-card fields, or bind files to one run/config hash.

Required correction: add acceptance tests at the artifact boundary as well as unit tests. A scientific build is complete only when both executable behavior and generated evidence pass.

## Root Cause 11: API-Key Presence Was Mistaken for Cost Authorization

During Phase 8 focused testing, the deterministic guardrail runner constructed `GuardrailedChatEngine`, which constructed `GovernedExplainer` without an explicit runtime config. `GovernedExplainer` loaded environment settings; the client factory's `auto` provider selected OpenAI whenever a machine API key existed, even though `require_real_llm` was false. Safe prompt cases routed through governed explanation generation and triggered real calls.

The preserved usage ledger records 24 OpenAI Chat Completions requests for case 528 from 2026-07-12T14:15:19Z through 14:18:12Z: 136,392 input tokens, 123,648 cached input tokens, 20,179 output tokens, 156,571 total tokens, and estimated cost USD 0.1096371. These calls were not approved and are not scientific evidence.

Correction applied: `GovernedExplainer` defaults to the offline stub, and client-factory `auto` mode remains offline unless `require_real_llm=True` is explicitly supplied. A 30-test regression run with the machine key available left the usage-log hash unchanged.

## Historical Evidence Classification

- `reports/leakage_safe/**` grouped SHAP and diagnostics containing Age under the current primary policy name: incompatible historical evidence.
- June 5 `reports/*/final_candidates/**`: informative historical candidate evidence, but not canonical because source/config identity is incomplete and stages do not share one run ID.
- `reports/external_validation/**`: informative historical external/related-task evidence, but metric applicability and some role labels are incompatible.
- `reports/llm_explanations/**` and `reports/agent_audits/**`: historical real-LLM technical evidence; the 80-case run is not complete-case manuscript evidence because 30 cases lack required local evidence.
- `reports/manuscript_assets/final_evidence_manifest/**`: hash-valid historical package centered on the LLM run, not a full canonical scientific evidence package.

No historical output is deleted or silently rewritten by this classification.

## RCA-012 — First Canonical Run SHAP Family-Order Failure

- Symptom: run `manuscript_final_20260712T175019Z_c664ef152ff3` stopped in the SHAP stage with `Raw feature-family order changed across SHAP folds`.
- Root cause: `ColumnTransformer` emits numeric and categorical blocks, whereas the source dataframe preserves an interleaved raw-column order. The grouped feature sets were identical, but the stage incorrectly treated an order-only difference as feature-policy drift.
- Fix: require unique identical feature-family sets, compute an explicit grouped-axis permutation into canonical raw-column order, and continue to fail on any missing or extra family.
- Verification: five focused tests passed and a full ten-fold real-data SHAP diagnostic regenerated the complete global/class/local/stability/Figure 6/Figure 7 package.
- Scientific impact: the failed run did not complete or become `latest`; its failed manifest is preserved. No failed-run artifact is admitted to the final package.

## RCA-013 — Second Canonical Run Case-Selection Identity Failure

- Symptom: run `manuscript_final_20260712T175251Z_c664ef152ff3` completed policy, calibration, SHAP, all-eligible counterfactual, external, fairness, and provenance stages, then the CompleteCaseEvidence preflight rejected `shap/representative_cases.csv` because it lacked `run_id` and `config_hash` columns.
- Root cause: the representative-case selector copied prediction values into a new row dictionary but omitted canonical identity fields. A secondary integration mismatch also existed between the external stage's documented `model_predictions.csv`/`performance_metrics.csv` names and provisional LLM fixture aliases.
- Fix: require one run/config/policy identity in the selection input, propagate it into every selected row, and resolve one explicit external schema name/alias while rejecting ambiguous duplicate aliases.
- Verification: eight selector/LLM unit tests passed; a real-data 80-case diagnostic reported 80 requested, 80 selected, 80 complete, 0 incomplete, and API execution disallowed. The seven-role offline deterministic audit then completed with the usage ledger unchanged.
- Scientific impact: the second run remains `status=failed` and is not promoted or reused. Its completed actionability protocol is retained only as failed-run diagnostic evidence.
