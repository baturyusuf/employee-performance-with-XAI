# Root Cause Analysis

## RC-1 — Two Competing Data Contracts

The manifest/config contract points at raw files, while legacy convenience functions implement a global interim-first rule. Scientific stages imported the convenience loader instead of receiving an explicit validated dataset object. Result: hashes describe one file and models consume another.

## RC-2 — Provenance Added Around, Not Through, Scientific Stages

Run/config identities were wrapped around existing stage functions, but stage APIs do not accept an immutable input/fold/side-input contract. They can independently discover mappings, global settings, and local paths. This produces incomplete cache keys and absolute metadata paths.

## RC-3 — Publication Metrics Reused Fold-Descriptive Utilities

A fold-mean t interval helper was reused for performance, calibration, proxy and SHAP summaries. It is suitable only as descriptive fold variability, not primary sample-level uncertainty. Ten-fold Wilcoxon testing similarly replaced the required paired OOF bootstrap.

## RC-4 — Calibration Comparison Was Conflated With Selection

The implementation evaluated raw/sigmoid/isotonic on outer test folds and then ranked the same test results to choose the winner. A comparative table is legitimate; using that table to choose the primary method is not. The fixed v2 contract predeclares sigmoid.

## RC-5 — Scope Accretion

LLM, multi-agent, chatbot, counterfactual, related-task external datasets, governance dashboards and model evidence accumulated in one canonical orchestrator. The result is not a focused leakage-aware XAI protocol and carries unnecessary dependencies, artifacts and claims.

## RC-6 — Repository Used as an Artifact Store

Windows symlink fallback copies the entire package to `latest`; failed runs and diagnostic LLM/SHAP directories were also committed. This created hundreds of megabytes of duplicates. Raw data was committed before licence/source verification was resolved.

## RC-7 — Local Validation Without Release CI

Tests focused on unit/shape/identity behavior but did not assert actual consumed input, command completion, no absolute paths, no network, predeclared calibration, or valid uncertainty. No GitHub Actions workflow or clean-clone release gate exists.

## RC-8 — Independent Fold and Model Ownership

Policy ablation, calibration, SHAP and subgroup/proxy modules each evolved as complete standalone experiments. They independently generate folds and instantiate an XGBoost model from a fixed config block, so a common seed does not prove identical sample membership and the SHAP model need not be the model that produced the benchmark OOF prediction. Unit 2B introduces a persisted shared-fold contract and per-outer-fold selected/fitted model index, but every downstream consumer must be converted before paired comparisons or exact OOF explanations are admissible.

## RC-9 — Unspecified Primary Model-Comparison Metric

D2 fixed nested tuning but did not define whether candidate selection and the baseline-over-XGBoost gate use macro-F1, QWK or a co-primary rule. Inferring this choice after observing results would introduce researcher degrees of freedom. The benchmark therefore keeps both metric fields null and fails before data/model execution until the user predeclares the rule.
