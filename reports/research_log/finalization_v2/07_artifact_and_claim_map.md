# Artifact and Claim Map — Baseline

## Status Rule

No file under `reports/manuscript_final/latest` is admitted as final v2 evidence. It may be used only as historical diagnostic context until a clean-current-commit v2 build regenerates the result from explicitly bound inputs.

| Historical area | Current path | Baseline finding | v2 disposition |
| --- | --- | --- | --- |
| Policy ablation | `reports/manuscript_final/latest/policy/` | Common folds and exact primary exclusions exist; input binding and uncertainty are invalid for v2 | Regenerate core |
| Calibration | `reports/manuscript_final/latest/calibration/` | Nested split exists; primary method selected from outer-test results | Regenerate with predeclared sigmoid |
| SHAP | `reports/manuscript_final/latest/shap/` | OOF fold models/grouping useful; dependent-pair t-CI and absolute paths invalid | Refactor and regenerate core |
| Fairness/proxy | `reports/manuscript_final/latest/fairness/` | Support-aware subgroup bootstrap useful; proxy uncertainty/input contract incomplete | Refactor and regenerate core |
| HRDataset | `reports/manuscript_final/latest/external/hrdataset_v14/` | Correct claim boundary; mapping side input not bound | Regenerate core after D3 |
| IBM/turnover | `reports/manuscript_final/latest/external/` | Roles bounded but mixed into core | Regenerate supplementary only |
| Counterfactual | `reports/manuscript_final/latest/counterfactual/` | OOF search exists; terminology/modes/sensitivity violate v2 scope | Supplementary refactor only |
| LLM/agents/chatbot | `reports/manuscript_final/latest/llm/`, `chatbot/` | Historical offline diagnostics | Exclude from core and paper claims |
| Figures 1-7 | `reports/manuscript_final/latest/figures/` | Run-bound but wrong v2 scope for Figures 1-4 | Replace complete core figure set |
| Manifest | `reports/manuscript_final/latest/run_manifest.json` | Old dirty commit, incomplete entrypoint, wrong actual input, absolute path | Reject and rebuild |

## Claim Freeze

No numeric manuscript claim is frozen. The v2 `manuscript_support/results_source_of_truth.md` and `claim_to_artifact_matrix.csv` must be generated only after the clean release run and approved by the user before manuscript editing.

## v2 Scope Contract — Unit 2A

| Accepted scope | Exact datasets | Planned evidence | Explicitly excluded |
| --- | --- | --- | --- |
| Core | INX primary; HRDataset_v14 | shared folds, four-model benchmark, leakage ablation, sigmoid calibration, OOF SHAP, subgroup/proxy diagnostics, HR mapped-target replication, core tables/figures | heuristic search, IBM/Turnover, historical artifacts, LLM, chatbot, agent audits |
| Supplementary | INX primary; IBM performance/attrition; Employee Turnover | heuristic model-scenario search, restricted-target/related-task robustness, supplementary tables/cards | HR replication, historical artifacts, LLM, chatbot, agent audits |

Both scopes are intentionally `release_ready=false`. No v2 scientific artifact or numerical claim has been admitted. Scoped manifest and cache contracts are implementation evidence, not manuscript results.

## Unit 2B Infrastructure Status

| New component | Repository source | Current claim status |
| --- | --- | --- |
| Shared 10×5 folds | `src/experiments/shared_folds.py`; `reports/manuscript_final/trials/benchmark-10x5-20260713-6a80074/core/shared_folds/` | Clean-start trial artifact verified; downstream consumers have not yet adopted it |
| Four-model restrained nested benchmark | `src/experiments/manuscript_model_benchmark.py`; trial `model_benchmarks/` | Real trial complete and hash-verified; noncanonical decision evidence pending final all-stage rebuild |
| Paired OOF bootstrap | `src/models/oof_bootstrap.py`; trial `model_summary.csv`, `paired_model_differences.csv` | 5,000 draws complete; model gate evidence valid, manuscript result not yet frozen |
| Baseline reference stop gate | trial `baseline_xgboost_gate.json`, `run_manifest.json` | Observed gate outcome `false`; no model-reference user decision required |

The earlier in-memory 10×5 preflight hashes remain noncanonical. The clean-start real trial is preserved under its versioned trial root and supports the XGBoost-reference gate decision, but `canonical_release_eligible=false`; its numbers are not frozen manuscript claims until the complete core package is regenerated from one final clean commit after warning and downstream-stage fixes.

## Unit 2D Policy Evidence Contract

No current canonical policy artifact exists. The tested stage declares the following outputs under the future same-run `core/policy/` directory; each must carry the benchmark run/config/scientific-input/fold/model-set identity and must be generated anew after an exact current-code benchmark replay:

| Declared output | Evidence role | Current claim status |
| --- | --- | --- |
| `oof_predictions.csv` | Exactly one OOF prediction per INX sample and policy | Implementation/test contract only |
| `fold_metrics.csv` | Ten-fold descriptive variability; not population inference | Implementation/test contract only |
| `policy_metric_intervals.csv` | Raw 5,000-draw sample-level bootstrap intervals | Implementation/test contract only |
| `policy_summary.csv`; `manuscript_policy_table.csv` | OOF estimates, intervals, explicit denominators and role boundaries | Implementation/test contract only |
| `policy_pairwise_tests.csv`; `leakage_sensitivity_index.csv` | Pointwise paired policy differences and declared leakage-sensitivity contrasts | Implementation/test contract only; no multiplicity-adjusted rejection claim |
| `policy_feature_contract.csv`; `policy_hyperparameter_schedule.csv`; `policy_fit_receipts.csv` | Exact exclusions, feature lineage, fold-selected parameter reuse and fit denominators | Implementation/test contract only |
| `figure_leakage_policy_tradeoff_source.csv`; PNG; SVG | Reproducible policy trade-off figure | Implementation/test contract only |
| `policy_interpretation.md`; `policy_metadata.json` | Diagnostic/audit-only boundaries and provenance | Implementation/test contract only |

The primary policy must reuse the exact benchmark XGBoost OOF rows after `1e-12` replay. Five non-primary policies use the same outer folds and the primary-selected XGBoost candidate for each fold; they are not independently tuned. Therefore the comparison is matched feature-access sensitivity conditional on the primary selection schedule, not a fully optimized leaderboard. The full-feature comparator is diagnostic and nondeployable.

## Unit 2E Calibration Evidence Contract

No historical calibration file is admitted and no new canonical calibration artifact exists yet. Option A is implemented and independently reviewed; the future same-run `core/sigmoid_calibration/` package declares:

| Declared output | Evidence role | Current claim status |
| --- | --- | --- |
| `calibration_training_oof_predictions.csv`; `calibration_fit_receipts.csv` | Exactly-once outer-training cross-fit rows and 50 isolated inner-fit receipts | Implementation/real-fold diagnostic only |
| `sigmoid_calibrator_parameters.csv`; `calibrator_model_relationships.csv` | Replayable ten-fold sigmoid coefficients and exact benchmark-model bindings | Implementation/test contract only |
| `calibration_predictions.csv`; `calibration_fold_metrics.csv` | 1,200 raw plus 1,200 sigmoid OOF rows; fold variability descriptive only | Implementation/test contract only |
| `calibration_metric_intervals.csv`; `calibration_paired_differences.csv`; `bootstrap_metadata.json` | Frozen 5,000-draw paired 95% OOF uncertainty and benchmark resample identity | Implementation/test contract only |
| `calibration_bins.csv`; class reliability PNG/SVG; calibration summary PNG/SVG/source | Ten-bin class reliability and manuscript-figure source evidence | Implementation/test contract only |
| `calibration_protocol.json`; `predeclared_method_rationale.md`; `calibration_validation.json`; `calibration_metadata.json` | Fixed sigmoid/no-selection contract, warning and hash/provenance validation | Implementation/test contract only |

The raw rows must equal current-run benchmark OOF values exactly and every sigmoid calibrator must be trained only from the corresponding outer-training cross-fit rows. Current config `d755ecc3...` rejects historical benchmark config `7e70bf66...`; therefore no numerical calibration claim is admitted until the benchmark and calibration run under one final frozen identity and the persisted validator/manifest pass.
