# Canonical Evidence Claim Boundaries

Evidence scope: `core`  
Run ID: `canonical_v2_20260714T221501Z_483f96f`  
Config hash: `51415c2ce68c89d9ce2b042b0a7a811fe3e98180b72460006f9d0465f6bf49b7`

The package is research-grade analysis only and does not authorize autonomous HR use.

## Supported when the referenced stages and uncertainty checks pass

- Paired out-of-fold leakage-policy sensitivity under the declared feature contracts; the full-feature result is a diagnostic upper bound only.
- Contextual comparison of the primary XGBoost model with the three predeclared predictive baselines on shared folds.
- Predeclared sigmoid probability calibration evaluated only on outer test folds.
- Grouped out-of-fold SHAP attribution and descriptive fold stability; attribution is not causality.
- Support-aware subgroup and proxy-risk diagnostics; sensitive-feature removal is not a fairness guarantee.
- HRDataset_v14 independent mapped-target performance replication; not locked-model transport.

## Unsupported or prohibited

- Autonomous hiring, firing, promotion, ranking, compensation, or other HR decisions.
- Human usefulness, trust, usability, deployment readiness, legal compliance or a fairness guarantee.
- Causal interpretation of model attribution.
- Verified dataset licence, source authenticity or ethics approval until the recorded manual gates are closed.
