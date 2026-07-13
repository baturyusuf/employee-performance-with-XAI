# Current Status

- Current phase: Unit 2C-0 probability/feature-warning hygiene is implemented and fully tested; preparing its checkpoint before exact-model OOF SHAP integration.
- Last completed task: canonical preprocessing now emits dense named pandas features; aligned probabilities are clipped only within the existing numerical bound and row-normalized in float64; label-only inner selection skips probability computation. Real fold-1 replay matched all historical labels without warnings.
- Work currently in progress: persist Unit 2C-0 tests/diagnostics, checkpoint the fix, then implement an exact selected-XGBoost fold-model artifact reader and OOF SHAP consumer.
- Files modified but not finalized: `configs/manuscript_final.yaml`, `src/models/canonical_models.py`, `src/experiments/manuscript_model_benchmark.py`, two focused test files, and persistent logs. The immutable untracked trial directory still contains 54 files/91,820,515 bytes and is not edited/staged.
- Latest tests: Unit 2C-0 focused 63 passed; full pytest 350 passed plus 4 subtests with 2 historical skips; unittest 162 passed with 2 skips; compileall/diff/manuscript/secret gates passed.
- Real benchmark result: XGBoost macro-F1 `0.621021` (95% CI `0.597319–0.644690`). Baseline-minus-XGBoost macro-F1 differences were Logistic Regression `-0.114800` (`-0.147597–-0.083224`), Random Forest `-0.028681` (`-0.049949–-0.008049`) and LightGBM `-0.015533` (`-0.038121–0.006382`). No baseline met the positive-point plus positive-CI-lower gate.
- Secondary result: QWK was XGBoost `0.567602`, LightGBM `0.588329`, Random Forest `0.631678`, Logistic Regression `0.371011`. QWK is reported but is not gate-eligible under the user-approved protocol.
- Known failures/open risks: the completed noncanonical trial retains its historical warnings and old source hash, by design; canonical probability metrics must be regenerated under the new code. Core/supplementary remain not release-ready; downstream shared-model adoption and the calibration/figures/tables/CI/lock/licence/history/EOL blockers remain.
- Exact next action: stage/checkpoint Unit 2C-0, then record and implement Unit 2C-A so OOF SHAP loads/hash-validates the exact selected XGBoost fold pipelines and never refits.
- Decisions awaiting the user: none from the model gate; verified ethics metadata later; acquisition mismatch only if triggered.
- Paid API/network calls in this phase: zero; the standalone trial and the recorded real fold-1 replay each installed an explicit process-local TCP/UDP/DNS denial guard.
- Manuscript edits: none.

The trial is real decision evidence but explicitly `canonical_release_eligible=false`; final manuscript numbers still require the clean complete core rebuild.
