# External OOF reason code: incorrect_high_confidence

Sample index: `172`; outer fold: `6`; true/predicted: `4`/`3`.

Grouped SHAP is model attribution for an exact OOF model, not causality.

Feature availability and timing must be verified for the intended prediction time; an attribution does not establish temporal or intervention validity.

Research-grade descriptive evidence only; no autonomous HR decision or employee prescription.

| Rank | Feature | Value | Grouped SHAP | Governance | Temporality |
|---:|---|---|---:|---|---|
| 1 | ExperienceYearsAtThisCompany | 5.484 | 1.021644 | derived_tenure_context | derived_at_last_review_timing_unverified_negative_durations_set_missing |
| 2 | Absences | 19 | 0.725652 | operational_history_or_window | timing_unverified_history_or_window |
| 3 | EngagementSurvey | 4.2 | 0.402090 | employee_reported_context | timing_unverified_contemporaneous |
| 4 | DaysLateLast30 | 0 | 0.395034 | operational_history_or_window | timing_unverified_history_or_window |
| 5 | EmpJobSatisfaction | 4 | 0.294587 | employee_reported_context | timing_unverified_contemporaneous |
| 6 | EmpJobRole | Production Technician II | 0.158995 | operational_proxy_context | recorded_role_context_timing_unverified |
| 7 | SpecialProjectsCount | 0 | 0.049025 | operational_history_or_window | timing_unverified_history_or_window |
