# Current Status

Date: 2026-07-14

- Branch: `finalization/leakage-aware-v2`.
- Canonical technical evidence: complete, strictly validated, and atomically promoted.
- Canonical run: `canonical_v2_20260714T221501Z_483f96f`.
- Clean generation commit: `483f96fdbaab16cb0f32d03d9dbe676a759af44a`.
- Config hash: `51415c2ce68c89d9ce2b042b0a7a811fe3e98180b72460006f9d0465f6bf49b7`.
- Source-tree hash: `f1e358e99914563305428cece1b1595bc76a58643184407ec5b222162d650332`.
- Core scientific-input hash: `06c507bee525ea1daca43b61249764007d4d8baaa05c9333f23446ea723ce160`.
- Supplementary scientific-input hash: `caffb945d15f990e3a789e9707f7a8a9115be31fecbbd705822994a10cfaf151`.
- Publication pointer: `reports/manuscript_final/latest/pointer.json`, the sole file under `latest`, SHA-256 `ecd3990187112891c3f1d3bb41bb7939392576e8df7904cae46baf2106397219`.
- Compact canonical receipt: `../finalization_v2/15_canonical_evidence_receipt.json`.

## Acceptance evidence

- Core build: exit 0 in 1,641.749 seconds; 351 closed-world manifest records, 354 total files, 224,171,937 bytes.
- Supplementary build: exit 0 in 1,926.755 seconds; 188 closed-world manifest records, 191 total files, 222,423,549 bytes.
- Promotion: exit 0 in 5.968 seconds.
- Strict post-promotion validation: exit 0 in 5.677 seconds.
- All ten core and four supplementary stage commands are complete with return code 0.
- Seven final figures exist as PNG/SVG pairs with seven source CSVs and seven captions.
- Fourteen source tables exist: eleven core and three supplementary.
- Complete package scans found zero machine-absolute paths, active `leakage-safe` terms, secret patterns, raw dataset files, or lock/partial/staging directories.
- Scientific runtime receipts record zero attempted/successful network operations and zero paid API calls.
- The journal manuscript was not modified.

## Unit 2G

The original stage-validation run `stage_validation_hrdataset_20260713T175045Z_5af0262e83a3` remains preserved and noncanonical. The canonical run regenerated HRDataset_v14 from the current frozen inputs and independently replayed all acceptance invariants:

- raw SHA-256 `cb19996755c93c0a8d6527f59da4701c80aef65eff854906546dce286249813c`;
- parsed-cell SHA-256 `e6d5bb365c3548e9185f4c24f20737210cde62c758d6e1d1ce3462846f4c0983`;
- schema mapping `65c40fecc75235f57209cff6231d7386e4279ebedca9390f53ad2bb67ae63788`;
- feature policy `e08625831afd057d1102d0fc42e06d56c9cd51a5632d04780aafe880888d78e5`;
- 311 exactly-once cases with mapped support 31/243/37 and 10 outer x 5 inner folds;
- 400 candidate fits, 50 persisted outer models, 50 calibration fits, and 5,000 bootstrap draws;
- zero maximum outer-model, sigmoid, or grouped-SHAP replay error;
- exact-fold grouped OOF SHAP with 6,531 rows and maximum additivity error `4.468303814064711e-06`;
- forbidden features absent from primary SHAP and reason-code outputs;
- support-aware subgroup/proxy and source-table contracts passed.

## Test state

- Focused final cycle: 103 passed.
- Complete pytest: 752 passed, 2 skipped, 11 subtests passed in 139.36 seconds.
- Unittest: 179 tests passed, 1 skipped in 9.487 seconds.
- Compileall: exit 0 in 0.131 seconds.
- Repository CI gate passed at clean generation commit with 1,734 tracked files, 33 issue rows, 30 README links, and zero raw/environment/large/secret/machine-path findings.

## Remaining blockers

There is no remaining executable P0/P1 scientific or engineering blocker for the technical evidence package. Submission remains blocked by manual/external items:

1. Ethics/IRB institution, unit, application/reference number, and date remain pending.
2. Dataset source authenticity, licence, citation, and redistribution approval require manual verification.
3. Public Git history still contains previously committed raw/noncanonical evidence; remediation requires a separately authorized history/publication strategy because force-push and history rewriting were prohibited.
4. Claim-matrix approval, manuscript writing/formatting, journal submission metadata, release/DOI publication, and any public evidence upload remain separate manual tasks.

Do not begin manuscript writing in this technical-finalization chat.
