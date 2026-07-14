# External OOF reason code: correct_low_confidence

Sample index: `49`; outer fold: `1`; true/predicted: `3`/`3`.

Grouped SHAP is model attribution for an exact OOF model, not causality.

Feature availability and timing must be verified for the intended prediction time; an attribution does not establish temporal or intervention validity.

Research-grade descriptive evidence only; no autonomous HR decision or employee prescription.

| Rank | Feature | Value | Grouped SHAP | Governance | Temporality |
|---:|---|---|---:|---|---|
| 1 | ExperienceYearsAtThisCompany | 0.966 | 0.700281 | derived_tenure_context | derived_at_last_review_timing_unverified_negative_durations_set_missing |
| 2 | EngagementSurvey | 4.2 | -0.378215 | employee_reported_context | timing_unverified_contemporaneous |
| 3 | DaysLateLast30 | 0 | 0.308282 | operational_history_or_window | timing_unverified_history_or_window |
| 4 | Absences | 9 | 0.195334 | operational_history_or_window | timing_unverified_history_or_window |
| 5 | EmpJobRole | Production Technician II | -0.118217 | operational_proxy_context | recorded_role_context_timing_unverified |
| 6 | EmpJobSatisfaction | 5 | 0.090096 | employee_reported_context | timing_unverified_contemporaneous |
| 7 | SpecialProjectsCount | 0 | -0.038289 | operational_history_or_window | timing_unverified_history_or_window |
