# Artifact and Claim Map — Baseline

## Status Rule

No file under `reports/manuscript_final/latest` is admitted as final v2 evidence. It may be used only as historical diagnostic context until a clean-current-commit v2 build regenerates the result from explicitly bound inputs.

| Historical area | Current path | Baseline finding | v2 disposition |
| --- | --- | --- | --- |
| Policy ablation | `reports/manuscript_final/latest/policy/` | Common folds and exact primary exclusions exist; input binding and uncertainty are invalid for v2 | Regenerate core |
| Calibration | `reports/manuscript_final/latest/calibration/` | Nested split exists; primary method selected from outer-test results | Regenerate with predeclared sigmoid |
| SHAP | `reports/manuscript_final/latest/shap/` | OOF fold models/grouping useful; dependent-pair t-CI and absolute paths invalid | Refactor and regenerate core |
| Fairness/proxy | `reports/manuscript_final/latest/fairness/` | Support-aware subgroup bootstrap useful; proxy uncertainty/input contract incomplete | Refactor and regenerate core |
| HRDataset | `reports/manuscript_final/latest/external/hrdataset_v14/` | Correct claim boundary; mapping side input not bound | Regenerate core after D3 |
| IBM/turnover | `reports/manuscript_final/latest/external/` | Roles bounded but mixed into core | Regenerate supplementary only |
| Counterfactual | `reports/manuscript_final/latest/counterfactual/` | OOF search exists; terminology/modes/sensitivity violate v2 scope | Supplementary refactor only |
| LLM/agents/chatbot | `reports/manuscript_final/latest/llm/`, `chatbot/` | Historical offline diagnostics | Exclude from core and paper claims |
| Figures 1-7 | `reports/manuscript_final/latest/figures/` | Run-bound but wrong v2 scope for Figures 1-4 | Replace complete core figure set |
| Manifest | `reports/manuscript_final/latest/run_manifest.json` | Old dirty commit, incomplete entrypoint, wrong actual input, absolute path | Reject and rebuild |

## Claim Freeze

No numeric manuscript claim is frozen. The v2 `manuscript_support/results_source_of_truth.md` and `claim_to_artifact_matrix.csv` must be generated only after the clean release run and approved by the user before manuscript editing.
