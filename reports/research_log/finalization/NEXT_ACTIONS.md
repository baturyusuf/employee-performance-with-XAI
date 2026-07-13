# Next Actions

1. Finish recording and independently checking trial `benchmark-10x5-20260713-6a80074`, including warnings and the non-triggered macro-F1 gate.
2. Normalize aligned model probabilities in float64 to machine precision; add warning-free/argmax-invariance tests. Handle the verified LightGBM feature-name warning without hiding unrelated warnings.
3. Run focused and full regression gates and checkpoint the warning-hygiene fix; preserve the completed trial unchanged.
4. Refactor leakage-policy ablation to consume the same persisted 10-fold assignment and apply paired sample-level bootstrap differences.
5. Refactor predeclared sigmoid calibration to use each exact outer-train partition and keep outer-test labels out of method selection.
6. Generate OOF grouped SHAP from the exact selected XGBoost fold model/preprocessor and shared assignment, then update subgroup/proxy consumers.
7. Run the complete core build only after all remaining core stages are release-ready; regenerate every authoritative metric under one final clean commit/config/run identity.
8. Before release claims, make source-tree verification EOL-stable, finish dependency lock/CI, and package large artifacts under D5 without duplicating them in Git.

No model-reference user decision is required because the predeclared macro-F1 gate did not trigger. Do not edit the manuscript, call APIs, publish, push or merge.
