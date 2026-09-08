# Table 8. HRDataset_v14 replication and data-quality results

| Result | Value | Boundary |
| --- | --- | --- |
| Raw XGBoost mean macro-F1 | 0.6531 | Five fixed 5×5 repetitions; independently trained |
| Sigmoid XGBoost mean QWK | 0.6044 | Metric-specific calibration effect |
| Mapped class-2 support | 31 | Study mapping; no construct equivalence |
| Effective missing cells | 215 | 1.92% under declared rule |
| Rules with findings | 3 of 29 | Declared-rule scope only |
| Anomaly occurrences | 6 | No row-level uniqueness claim |
