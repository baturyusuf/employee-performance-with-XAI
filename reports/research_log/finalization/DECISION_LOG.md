# Persistent Decision Log

The fixed decisions are recorded in `../finalization_v2/00_scope_and_fixed_decisions.md` and `../finalization_v2/04_decision_log.md`.

Accepted on 2026-07-13:

- D1: Logistic Regression + Random Forest + LightGBM.
- D2: restrained nested tuning with outer-test isolation.
- D3: user-provided pinned files for all datasets; sanitized publication export; only approved-manifest URL acquisition when local data is absent.
- D4: application pending; institution/unit/reference/date remain unknown because supplied strings were placeholders.
- D5: small artifacts in Git; full package prepared for Release/Zenodo; pointer-only latest; no publication yet.
- Dedicated branch/commit permission: `finalization/leakage-aware-v2`, tested checkpoints only.
- Stop gate: if a baseline's paired OOF bootstrap advantage over XGBoost has a CI excluding zero, request user direction before choosing the XAI reference model.

Still prohibited without explicit approval: force-push, merge, release publication, Git history alteration, unapproved mirror use, or new scientific-protocol choices.
