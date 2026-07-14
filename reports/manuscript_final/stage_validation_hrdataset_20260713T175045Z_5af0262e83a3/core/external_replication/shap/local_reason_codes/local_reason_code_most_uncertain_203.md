# External OOF reason code: most_uncertain

Sample index: `203`; outer fold: `3`; true/predicted: `3`/`4`.

Grouped SHAP is model attribution for an exact OOF model, not causality.

Feature availability and timing must be verified for the intended prediction time; an attribution does not establish temporal or intervention validity.

Research-grade descriptive evidence only; no autonomous HR decision or employee prescription.

| Rank | Feature | Value | Grouped SHAP | Governance | Temporality |
|---:|---|---|---:|---|---|
| 1 | ExperienceYearsAtThisCompany | 5.487 | -0.332300 | derived_tenure_context | derived_at_last_review_timing_unverified_negative_durations_set_missing |
| 2 | Absences | 4 | 0.316717 | operational_history_or_window | timing_unverified_history_or_window |
| 3 | EngagementSurvey | 3.6 | 0.207827 | employee_reported_context | timing_unverified_contemporaneous |
| 4 | DaysLateLast30 | 0 | 0.117911 | operational_history_or_window | timing_unverified_history_or_window |
| 5 | EmpJobRole | Production Technician I | -0.070448 | operational_proxy_context | recorded_role_context_timing_unverified |
| 6 | SpecialProjectsCount | 0 | -0.040097 | operational_history_or_window | timing_unverified_history_or_window |
| 7 | EmpJobSatisfaction | 5 | -0.027966 | employee_reported_context | timing_unverified_contemporaneous |
