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

## RC-10 - Subgroup/Proxy Stage Recomputed Instead of Reusing Evidence

The legacy manuscript subgroup/proxy module remained a standalone experiment after shared-fold and policy contracts were introduced. It independently created outer folds, refit fixed-parameter XGBoost models, created a second proxy-task splitter, and wrapped only run/config labels around those results. The builder did not supply verified upstream stage directories or the scientific-input identity. As a result, a common seed was mistaken for a common scientific contract: 1,091 of 1,200 historical fairness fold assignments differ from the verified trial assignment.

The same stage reused the fold-mean Student-t helper for proxy confidence intervals, omitted a proxy OOF table and resample hash, permitted a lower bootstrap count through its CLI, silently skipped absent audit attributes, serialized absolute paths and wrote non-atomically. It also fit two nominal proxy policies whose post-target-removal feature sets are mechanically identical. Correcting this requires evidence reuse and explicit equivalence, not relabelling the old outputs.

Implementation review exposed two additional consequences of the same standalone design. First, class-specific support (especially precision's predicted-positive denominator) can differ by policy, but legacy paired gap subtraction did not freeze a common group estimand; it could subtract max-min gaps computed over different group sets. Second, a declared batch size was never used: string-valued group arrays were expanded across all 5,000x1,200 bootstrap selections, creating a concrete several-hundred-MiB avoidable memory spike. Unit 2F therefore fixes eligibility on the complete OOF sample, intersects eligible groups for each paired policy/attribute/metric/class estimand, and executes compact integer-coded bootstrap calculations in deterministic batches.

Final review found that department reconstructability was still labelled with a restricted performance task schema and did not expose rare-class support in the manuscript-facing row; the label mapping also lacked scientific identity. This could imply performance-task comparability and conceal two observed zero-support fold/class cells. Unit 2F now uses a dedicated nominal proxy-diagnostic schema, binds target mapping to the run/config/scientific/fold/model/data identities, and reports overall and fold-level target support plus the conditional fixed-model inference scope. Boolean seeds, config-independent watchlists, positional audit joins and persisted feature-count drift are also rejected.
