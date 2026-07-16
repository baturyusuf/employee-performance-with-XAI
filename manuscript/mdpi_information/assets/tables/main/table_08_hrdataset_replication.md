# HRDataset_v14 mapped-target replication

Independent mapped-target replication. It is not locked-model transport, target equivalence, fairness evidence, or deployment validation.

| external_policy | role | feature_count | mapped_n | mapped_support | macro_f1_display | macro_f1_interval_display | qwk_display | qwk_interval_display | ordinal_mae_display | severe_error_rate_display | macro_f1_difference_vs_primary_display | raw_log_loss_display | sigmoid_log_loss_display | raw_brier_display | sigmoid_brier_display | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative_primary | canonical_external_primary | 7 | 311 | 2=31; 3=243; 4=37 | 0.6664 | [0.6281, 0.7047] | 0.5412 | [0.4851, 0.5988] | 0.1961 | 0.0000 | N/A | 0.5363 | 0.4224 | 0.3013 | 0.2352 | independently trained mapped-target replication |
| department_including_audit | audit_or_sensitivity_only | 8 | 311 | 2=31; 3=243; 4=37 | 0.6679 | [0.6300, 0.7056] | 0.5455 | [0.4914, 0.6025] | 0.1929 | 0.0000 | 0.0016 | N/A | N/A | N/A | N/A | audit/sensitivity only; cannot replace primary |
| job_role_free_audit | audit_or_sensitivity_only | 6 | 311 | 2=31; 3=243; 4=37 | 0.6790 | [0.6387, 0.7197] | 0.5606 | [0.5030, 0.6202] | 0.1865 | 0.0000 | 0.0126 | N/A | N/A | N/A | N/A | audit/sensitivity only; cannot replace primary |
| proxy_rich_audit | audit_or_sensitivity_only | 10 | 311 | 2=31; 3=243; 4=37 | 0.6527 | [0.6165, 0.6902] | 0.5361 | [0.4842, 0.5915] | 0.1897 | 0.0000 | -0.0136 | N/A | N/A | N/A | N/A | audit/sensitivity only; cannot replace primary |
| temporality_restricted_audit | audit_or_sensitivity_only | 2 | 311 | 2=31; 3=243; 4=37 | 0.3003 | [0.2691, 0.3334] | -0.0730 | [-0.1642, 0.0171] | 0.4019 | 0.0289 | -0.3660 | N/A | N/A | N/A | N/A | audit/sensitivity only; cannot replace primary |
