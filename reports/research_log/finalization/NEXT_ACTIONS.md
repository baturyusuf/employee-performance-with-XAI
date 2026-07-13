# Next Actions

1. Stage/checkpoint the fully tested Unit 2C-0 warning-hygiene implementation while leaving the immutable trial package unstaged.
2. Add a reusable benchmark artifact reader that validates run/config/scientific-input/fold identities and all 10 XGBoost model path/hash/size records.
3. Refactor OOF SHAP to load only each sample's exact prediction-producing outer-fold pipeline, prohibit model fitting, verify predictions against benchmark OOF evidence, and carry model hashes/lineage.
4. Replace dependent fold-pair t confidence intervals with descriptive SHAP stability distributions unless independent repeated-CV units are later introduced.
5. Refactor leakage-policy ablation to consume the shared 10-fold assignment and apply paired sample-level bootstrap differences.
6. Resolve the calibration development/split protocol decision before implementing predeclared sigmoid calibration; outer-test method selection remains prohibited.
7. Update subgroup/proxy/external consumers, then run the complete core build under one final clean commit/config/run identity.
8. Before release claims, make source-tree verification EOL-stable, finish dependency lock/CI, and package large artifacts under D5 without duplicating them in Git.

No model-reference user decision is required because the predeclared macro-F1 gate did not trigger. Do not edit the manuscript, call APIs, publish, push or merge.
