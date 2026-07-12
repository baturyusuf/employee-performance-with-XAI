# OOF Counterfactual Actionability Interpretation

Run ID: `counterfactual_cost_benchmark_20260712`  
Config hash: `738ec00ae6d2b64494c8e00f74fa3a6e32afd4d4afbff54c10900020287358fc`

All validity estimates use fold-specific models. Each evaluated case is excluded from model fitting, prototype selection, scale estimation, and domain construction. Desired-class prototypes come only from that case's outer training fold.

## Results by Intervention Mode

- `employee_only`: 0/10 valid (0.0000; Wilson 95% CI 0.0000–0.2775); total OOF cases 11.
- `employee_manager`: 1/10 valid (0.1000; Wilson 95% CI 0.0179–0.4042); total OOF cases 11.
- `organization_allowed`: 2/10 valid (0.2000; Wilson 95% CI 0.0567–0.5098); total OOF cases 11.
- `diagnostic_full_default`: 1/10 valid (0.1000; Wilson 95% CI 0.0179–0.4042); total OOF cases 11.
- `no_salary`: 2/10 valid (0.2000; Wilson 95% CI 0.0567–0.5098); total OOF cases 11.

## Claim Boundaries

Validity means only that a constrained model input scenario changed the fold-specific model prediction to the desired or a higher class.
Counterfactuals are not causal findings, guaranteed feasible interventions, employee prescriptions, or autonomous HR recommendations.
Employee, manager, and organisation modes must be interpreted separately. Diagnostic full-default results are an upper-bound diagnostic and may include immutable/history features; they are never actionable evidence.
Prototype values remain within observed training-fold domains and relational tenure constraints, but observational plausibility does not establish real-world feasibility.
