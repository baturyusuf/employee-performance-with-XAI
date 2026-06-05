# Counterfactual Actionability Interpretation

Counterfactuals are intervention hypotheses, not employee prescriptions. Validity means the fitted model prediction changes under constrained feature modifications; it does not imply feasibility, causality, fairness, or recommended employee behavior.

## Required Answers

### Can employees realistically change the model prediction through employee-only features?
For the primary candidate, employee-only validity is 0.0000. Low validity should be interpreted as limited employee-side recourse, not as an implementation failure.

### Are performance-class shifts mostly dependent on managerial or organisational variables?
For the primary candidate, employee+manager validity is 0.2500, while organization-allowed validity is 0.2500. A gain after adding manager/organisation controls means recourse depends on workplace context.

### Which counterfactual modes are valid but not practically actionable?
`full_default` may change immutable or historical variables and is a diagnostic upper bound. `organization_allowed` may be valid for workforce planning but is not an employee prescription.

### How should the paper phrase counterfactual explanations responsibly?
Use: 'Under constrained feature changes, the model prediction would change if these workplace/context variables were different.' Do not write 'the employee should change X' unless employee-only validity is strong and the changed feature is genuinely employee-controllable.
