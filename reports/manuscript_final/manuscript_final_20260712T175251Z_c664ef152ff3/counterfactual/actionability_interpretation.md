# OOF Counterfactual Actionability Interpretation

Run ID: `manuscript_final_20260712T175251Z_c664ef152ff3`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`

All validity estimates use fold-specific models. Each evaluated case is excluded from model fitting, prototype selection, scale estimation, and domain construction. Desired-class prototypes come only from that case's outer training fold.

## Results by Intervention Mode

- `employee_only`: 28/1196 valid (0.0234; Wilson 95% CI 0.0162–0.0336); total OOF cases 1200.
- `employee_manager`: 352/1196 valid (0.2943; Wilson 95% CI 0.2692–0.3208); total OOF cases 1200.
- `organization_allowed`: 359/1196 valid (0.3002; Wilson 95% CI 0.2749–0.3267); total OOF cases 1200.
- `diagnostic_full_default`: 297/1196 valid (0.2483; Wilson 95% CI 0.2247–0.2736); total OOF cases 1200.
- `no_salary`: 359/1196 valid (0.3002; Wilson 95% CI 0.2749–0.3267); total OOF cases 1200.

## Claim Boundaries

Validity means only that a constrained model input scenario changed the fold-specific model prediction to the desired or a higher class.
Counterfactuals are not causal findings, guaranteed feasible interventions, employee prescriptions, or autonomous HR recommendations.
Employee, manager, and organisation modes must be interpreted separately. Diagnostic full-default results are an upper-bound diagnostic and may include immutable/history features; they are never actionable evidence.
Prototype values remain within observed training-fold domains and relational tenure constraints, but observational plausibility does not establish real-world feasibility.
