# External OOF reason code: minority_true_class

Sample index: `53`; outer fold: `10`; true/predicted: `2`/`2`.

Grouped SHAP is model attribution for an exact OOF model, not causality.

Feature availability and timing must be verified for the intended prediction time; an attribution does not establish temporal or intervention validity.

Research-grade descriptive evidence only; no autonomous HR decision or employee prescription.

| Rank | Feature | Value | Grouped SHAP | Governance | Temporality |
|---:|---|---|---:|---|---|
| 1 | DaysLateLast30 | 5 | 5.287646 | operational_history_or_window | timing_unverified_history_or_window |
| 2 | EngagementSurvey | 2.0 | 0.809331 | employee_reported_context | timing_unverified_contemporaneous |
| 3 | ExperienceYearsAtThisCompany | 4.808 | -0.159548 | derived_tenure_context | derived_at_last_review_timing_unverified_negative_durations_set_missing |
| 4 | Absences | 16 | -0.140273 | operational_history_or_window | timing_unverified_history_or_window |
| 5 | EmpJobRole | Production Technician I | 0.007919 | operational_proxy_context | recorded_role_context_timing_unverified |
| 6 | EmpJobSatisfaction | 3 | -0.001647 | employee_reported_context | timing_unverified_contemporaneous |
| 7 | SpecialProjectsCount | 0 | 0.000000 | operational_history_or_window | timing_unverified_history_or_window |
