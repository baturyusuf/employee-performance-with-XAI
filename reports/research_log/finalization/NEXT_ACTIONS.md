# Next Actions

1. Checkpoint the fully tested Unit 2C-A exact-model OOF SHAP implementation while leaving the historical trial package unstaged.
2. Record the leakage-policy ablation root cause and refactor it to consume the same 10-fold assignment with paired sample-level OOF bootstrap differences.
3. Decide whether policy variants reuse the primary fold-selected XGBoost parameters or receive nested selection only if that choice materially changes scientific scope/compute; do not silently mix protocols.
4. Resolve the calibration development/split protocol decision before implementing predeclared sigmoid calibration; outer-test method selection remains prohibited.
5. Update subgroup/proxy/external consumers, then regenerate benchmark+SHAP together in the complete core build under one final clean commit/config/run identity.
6. Build the revised core-only figures/tables and claim matrix from that same run; historical SHAP/LLM/chatbot artifacts remain excluded.
7. Before release claims, make source-tree verification EOL-stable, finish dependency lock/CI, and package large artifacts under D5 without duplicating them in Git.

No model-reference user decision is required because the predeclared macro-F1 gate did not trigger. Do not edit the manuscript, call APIs, publish, push or merge.
