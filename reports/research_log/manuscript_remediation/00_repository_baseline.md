# Repository Baseline for Manuscript Remediation

Baseline captured on 2026-07-12 in the `Europe/Istanbul` timezone before any scientific-code modification. The repository was inspected read-only except for creation of the files in this research-log directory.

## Git State

- Repository: `https://github.com/baturyusuf/employee-performance-with-XAI`
- Branch: `main`
- Commit: `18347488bdb4eed60f115ceeff70c420071ceef0`
- Upstream: `origin/main`; ahead 0, behind 0
- Initial and post-test worktree: clean
- Latest commit timestamp: 2026-06-29 11:13:56 +03:00
- No `AGENTS.md` file was present in the repository.
- The manuscript at `manuscript/mdpi_information/main.md` was inspected but was not modified.

## Execution Environment

- Operating system: Windows 11 Pro, build 10.0.26200, 64 bit
- Repository interpreter: `myenv/Scripts/python.exe`
- Python: 3.14.0, 64 bit, MSC v.1944
- pip: 25.2
- Installed distributions: 131
- `pip check`: no broken requirements found
- Bare `python` is not usable in the current shell because the first PATH entry resolves to a broken Windows Store alias. Canonical commands must use the repository interpreter or a documented environment bootstrap.

Key installed versions:

| Package | Version |
| --- | --- |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| shap | 0.51.0 |
| matplotlib | 3.10.8 |
| joblib | 1.5.3 |
| imbalanced-learn | 0.14.1 |
| catboost | 1.2.10 |
| lightgbm | 4.6.0 |
| openai | 2.43.0 |
| openai-agents | 0.17.6 |
| pydantic | 2.13.4 |
| pytest | 9.1.1 |

The canonicalized `pip freeze` snapshot has SHA-256 `c687ec313f9885905f6defe4d49c4b09ff02cef429caccbdc0e7fabb6bcf9294`. Modules absent from the environment include `yaml`, `xlrd`, `openpyxl`, `seaborn`, and `statsmodels`. `environment.yml` is empty, while `requirements.txt` declares only the OpenAI packages and `python-dotenv`; the scientific environment is therefore not declared reproducibly.

## Dataset Baseline

The raw INX CSV uses a semicolon delimiter. Delimiter-aware inspection gives 1200 rows and 28 columns; a naive default `pandas.read_csv` call incorrectly reports one column. The raw CSV and validated interim CSV are logically identical in columns, shape, values, and cell contents, although their byte hashes differ because their serialization differs.

| Dataset file | Rows x columns | SHA-256 | Baseline role/status |
| --- | ---: | --- | --- |
| `data/raw/inx_employee_performance.csv` | 1200 x 28 | `b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a` | canonical INX raw CSV candidate |
| `data/raw/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls` | unavailable | `d7d224e7c8f50693e0aba7621e56a55a84b0c2c156ff95afd2b156daa5d0f003` | historical raw workbook; unreadable because `xlrd` is absent |
| `data/interim/inx_employee_performance_validated.csv` | 1200 x 28 | `afc97e8b8aba4bae1e1ca62e17c8c0c18b73f2c6e53c846ffbee2d0d4c495a7e` | validated INX input used by current loaders |
| `data/external/hrdataset_v14/raw.csv` | 311 x 36 | `cb19996755c93c0a8d6527f59da4701c80aef65eff854906546dce286249813c` | independent external performance-target replication |
| `data/external/ibm_hr_analytics/raw.csv` | 1470 x 35 | `a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659` | restricted target robustness and optional related attrition task |
| `data/external/employee_turnover/raw.csv` | 14999 x 10 | `2510e274a90547f34c7b0db5a4ab70282c2710eb54252f14921bf980b81a928c` | related turnover task only |

Target support verified from repository data:

- INX `PerformanceRating`: class 2 = 194, class 3 = 874, class 4 = 132.
- HRDataset_v14 raw `PerformanceScore`: PIP = 13, Needs Improvement = 18, Fully Meets = 243, Exceeds = 37. The declared mapping yields class 2 = 31, class 3 = 243, class 4 = 37.
- IBM performance: class 3 = 1244, class 4 = 226; it is not a 2/3/4 replication.
- IBM attrition: No = 1233, Yes = 237.
- Employee Turnover: 0 = 11428, 1 = 3571.

Processed-file hashes were also captured during the audit. These files are historical preprocessing outputs and are not accepted into the future manuscript package without a matching run contract:

| Path | Rows x columns | SHA-256 |
| --- | ---: | --- |
| `data/processed/X_train_raw.csv` | 960 x 24 | `e96e709fb45b91299bcda42986e0e65ee50084a4ebc73b2db0b98b4259964a90` |
| `data/processed/X_test_raw.csv` | 240 x 24 | `b10189c200ce46fcd6314ac10abe58eb0005cbd4f82ab049cd082cc8fc278afc` |
| `data/processed/X_train_preprocessed.csv` | 960 x 56 | `262dd0ae5235e7549b54f969d8097a1eecfcc02635bd4d0e062aafda9630e3e5` |
| `data/processed/X_test_preprocessed.csv` | 240 x 56 | `715b4c4e29abd2776f0e15b5ca265ca7bb3362b1eb53fc146103054aa336f133` |
| `data/processed/y_train.csv` | 960 x 1 | `ea0ab8a02ca6a028f2b84eeb7ead2f54627a91c9a830e710854e96acb0f61daa` |
| `data/processed/y_test.csv` | 240 x 1 | `540688f8a2750ba2013552b528637b466c3d5d7dee75c3bf41ed81eb8ee57ee5` |

## Baseline Tests

- Test files: 32
- Collected tests: 100
- `pytest`: 100 passed in 4.67 seconds with cache disabled; an independent repeat passed 100 tests in 4.13 seconds.
- `unittest discover`: 100 tests passed in 1.367 seconds.
- Worktree remained clean after tests.

These passing tests establish only the pre-remediation behavior. The acceptance tests requested for canonical policy consistency, task-aware metrics, OOF counterfactuals, complete LLM evidence, dataset cards, figures, and a full evidence manifest do not yet exist.

## Existing Artifact Snapshot

The required report roots contain 331 files in total. Each root digest below hashes a sorted inventory of relative path, byte size, UTC modification time, and individual SHA-256. Individual key-file hashes and timestamps are recorded in `07_artifact_manifest.csv`.

| Artifact root | Files | Oldest UTC | Newest UTC | Snapshot SHA-256 |
| --- | ---: | --- | --- | --- |
| `reports/model_selection` | 5 | 2026-06-05T15:49:30Z | 2026-06-05T16:10:05Z | `9942c9bec1caddbdadb904f9cc34d27d22e42bc6a50f5181b82cf7f6f2fb1c1b` |
| `reports/xai/final_candidates` | 26 | 2026-06-05T16:09:29Z | 2026-06-23T11:19:45Z | `20e35e0202ec0083e3e32cee34fe351f348377a034902a01d2c0d0d74a3676af` |
| `reports/counterfactuals/final_candidates` | 4 | 2026-06-05T16:09:52Z | 2026-06-05T16:09:52Z | `35217a64599e0909163199d550988279c4e2a498e6d8c7045d23eef082c6fb47` |
| `reports/calibration/final_candidates` | 15 | 2026-06-05T16:08:56Z | 2026-06-05T16:08:58Z | `a0c5f7dbd2435957740c53f860d18231009edc0c392e29edcbb46e1c65b34619` |
| `reports/llm_explanations` | 40 | 2026-06-22T11:12:35Z | 2026-06-25T09:23:42Z | `d1d83b97f5305b75bc12fbe063d7c6a3ec412eb258f852922567f504e2d46b30` |
| `reports/agent_audits` | 31 | 2026-06-22T11:12:35Z | 2026-06-25T09:23:42Z | `59e4e13596a5f562edc09045c6c9fe693b8812ffee988aa2552ad90bd48ce0ce` |
| `reports/chatbot_eval` | 5 | 2026-06-23T11:27:58Z | 2026-06-24T13:51:29Z | `e15290f539d432670e577312adba3564a46092eaa82ef3081e9cf6b5ca963952` |
| `reports/governance_reports` | 9 | 2026-06-22T11:12:37Z | 2026-06-25T09:23:53Z | `4545aa32ef73dc5c3945f6652835c4be9f646f08b31bf4caf3f99d68b1bd4daa` |
| `reports/external_validation` | 192 | 2026-06-24T07:15:58Z | 2026-06-25T09:23:53Z | `761a10df373e72983b88122a04137930156c27939564cb1c25ead90f59e4d3e7` |
| `reports/manuscript_assets/final_evidence_manifest` | 4 | 2026-06-25T09:26:03Z | 2026-06-25T09:26:03Z | `1ac1ef064183136f59c7c7207af565cf5273b688cf47476259876960a4a8ea97` |

The existing 29-entry evidence manifest currently verifies all 29 files it names: no missing file and no hash mismatch. That does not make it a canonical scientific manifest because it is anchored to the 80-case LLM run, omits a canonical configuration hash and scientific run identity, and combines scientific artifacts created under other provenance histories.

## Provenance and Compatibility Findings

1. The generated model card states that the primary candidate excludes `Age`, `Gender`, `MaritalStatus`, `EmpLastSalaryHikePercent`, `Attrition`, `EmpDepartment`, `EmpNumber`, and `PerformanceRating`. `configs/feature_sets.yaml` excludes Age and the leakage/group fields but does not encode Gender or MaritalStatus in that policy. Final scripts obtain the additional removal through a separate `drop_sensitive=True` call. The policy therefore has two definitions rather than one canonical definition.
2. `src/explainability/xgboost_grouped_shap.py` has its own hard-coded policy dictionary that omits Age, Gender, and MaritalStatus. Historical output at `reports/leakage_safe/xgboost_no_salary_hike_no_attrition_no_department/shap/global_grouped_shap_importance.csv` includes Age, and the associated metadata lists Age among model features. It is incompatible with the current primary-policy claim and must be historical-only.
3. June 5 final-candidate registry rows record Git commit `6f9df0a85d26bf1519634001357c8f91340a740c`. The generating source/config files did not exist in that commit and were committed shortly afterward in `c703039`. The recorded commit therefore describes a dirty working-tree base, not the exact generating tree. No config hash was recorded, so exact reconstruction is impossible.
4. Calibration, SHAP stability, counterfactual, model selection, and model-card artifacts were generated at different stage run IDs and later combined. Their metadata contains seeds and selected package versions but no common run ID, dataset hash, config hash, complete code identity, output hashes, or failure state.
5. Full-feature and reduced-policy evidence comes from different output trees and different policy eras. The current dashboard cannot establish that all comparisons used one fold contract.
6. Binary attrition/turnover reports contain ordinal MAE, QWK, adjacent accuracy, and severe-error rate 0.0. Severe error is structurally impossible for labels 0/1 under `abs(y_true-y_pred)>1`; reporting 0.0 implies applicability and is incompatible with the requested task schema.
7. `configs/external_validation.yaml` and generated metadata use the phrase `direct external performance validation` for HRDataset_v14. The defensible role is independent external performance-target replication because no locked INX model is transported. The three-feature cross-dataset infeasibility result is correctly documented.
8. Counterfactual evaluation fits and evaluates a model on the full dataset, uses desired-class prototypes from the same full dataset, selects five in-sample representative cases, and estimates validity from four eligible cases. These rates are optimistic and do not support population-level validity claims.
9. The 80-case LLM run contains 30 `evidence_missing` supervisor results. INX cases beyond ten existing reason-code examples are explicitly filled with report-level evidence and `partial_missing_local_shap`; real-run preflight did not block this state.
10. Guardrail testing uses 50 unsafe and 25 safe hard-coded English prompts. It includes direct jailbreak keywords but lacks the required multilingual, obfuscated, retrieval-failure, conflicting-evidence, and stronger indirect-attack strata. Existing perfect rates need Wilson intervals and bounded-coverage wording.
11. External fairness summaries headline gaps of 1.0 without presenting subgroup support and valid bootstrap replicate counts in the manuscript-facing row. This is not support-aware reporting.
12. Dataset cards omit retrieval date, licence status, citation fields, and explicit unresolved manual verification fields. Public-mirror authenticity and licensing remain manual-review items.
13. Existing manuscript figures are a different six-figure set and are not generated under a single scientific run contract matching the requested Figures 1–7.
14. The LLM final-80 configuration directly references the legacy INX SHAP representative-case tree under `reports/leakage_safe/xgboost_no_salary_hike_no_attrition_no_department/`, whose earlier grouped-SHAP metadata includes Age. This is a live stale-artifact dependency, not merely an unused historical file.
15. Artifact lineage spans at least commits `6f9df0a` (June 5 final-candidate stages recorded against an uncommitted tree), `c703039` (supplemental reason codes), `c318d93` (external evidence), `861f271` (80-case LLM and deterministic guardrail run), and `9cf48fb` (refreshed summaries and evidence manifest).
16. The deterministic chatbot guard evaluates safe allow-list patterns before unsafe patterns. A composite prompt containing safe audit vocabulary and a prohibited HR-decision request can therefore bypass refusal; the current suite does not exercise this conflict.
17. The LLM evidence schema's hard-coded forbidden-feature list omits Gender, MaritalStatus, and EmpDepartment, so it cannot enforce the generated model card's primary policy.
18. The INX fairness package has useful support and bootstrap fields, but its generator defaults to 500 replicates while `configs/evaluation.yaml` declares 5000. External subgroup audit code treats high-cardinality raw continuous fields as grouping attributes, producing thousands of low-support warnings and no manuscript-ready uncertainty context.
19. `reports/chatbot_eval/guardrail_evaluation.md` is a stale 22-prompt predecessor with mojibake that coexists with the current 75-prompt summary. The aggregate guardrail CI file incorrectly labels deterministic chatbot results as real-LLM output.

## Baseline Disposition

All pre-existing scientific outputs are now classified as historical or diagnostic until a canonical run proves compatibility. Historically important files will be preserved. No old result will be silently overwritten, and no current numeric result will be copied into the new manuscript package without regeneration by executable code under the canonical contract.
