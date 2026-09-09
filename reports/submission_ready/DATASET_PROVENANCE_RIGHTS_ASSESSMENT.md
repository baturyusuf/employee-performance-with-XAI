# Dataset Provenance and Rights Assessment

**Assessment date:** 2026-09-09

This assessment separates analytical byte identity from upstream authenticity and legal redistribution rights. Public availability is not treated as permission to redistribute.

| Dataset | Claimed upstream source | Verified source? | Exact-byte chain? | Licence / rights evidence | Reuse permitted? | Redistribution permitted? | Manuscript-safe wording |
| --- | --- | --- | --- | --- | --- | --- | --- |
| INX Future Inc. Employee Performance | IABAC Project 10281 / historical `data.iabac.org` project download | **Partially verified.** An IABAC project PDF identifies the same Project 10281 and directs users to the named V1.8 XLS dataset. | **Local workbook-to-canonical-byte equivalence is hash-bound in the repository, but the repository does not establish that the local workbook was downloaded from the authoritative IABAC endpoint.** | The IABAC project document states that reproduction of its material requires IABAC permission and that rights are reserved. No dataset-specific redistribution licence has been verified for the exact study bytes. | Analysis use in this study has occurred, but a general third-party reuse licence for the exact dataset is **not established by the available evidence**. | **Not established. Do not redistribute raw INX employee-level bytes without permission/licence evidence.** | `The INX table was analyzed as a public cross-sectional benchmark. Repository hashes establish the analytical byte identity used in this study, but authoritative source-to-byte provenance and redistribution permission remain unresolved; raw employee-level data are therefore excluded from the release/submission package.` |
| HRDataset_v14 | Human Resources Data Set by Dr. Rich Huebner and Dr. Carla Patalano, Kaggle, Version 14 | **Yes for dataset identity/source description.** The authors' Kaggle data card identifies Version 14, the two original authors, 311 employee records, and the teaching purpose. | **No complete chain from the repository's local SHA-256 to a byte downloaded directly from the verified Kaggle source has been established in the current repository contract.** The recorded local source is a public GitHub mirror. | The verified Kaggle data card states **CC BY-NC-ND 4.0** and says use is licensed for learning/teaching; sharing should follow that licence. | **Restricted.** Use must remain consistent with the licence and its non-commercial/no-derivatives terms; journal-publication implications should be confirmed if needed. | **Potentially restricted and not cleared for repository redistribution in this project.** Do not assume that a public mirror authorizes redistribution of the study copy. | `HRDataset_v14 is a synthetic teaching dataset authored by Carla Patalano and Rich Huebner and distributed on Kaggle under CC BY-NC-ND 4.0. The study uses a local hash-bound copy, but the exact source-to-byte acquisition chain and redistribution clearance for the repository copy are not treated as resolved; raw rows are excluded from the submission package.` |

## Evidence already present in the repository

The acquisition contract records both analytical files as user-provided local inputs. It explicitly marks INX as `repository_local_copy_upstream_source_unverified` and HRDataset_v14 as `public_mirror_recorded_upstream_authenticity_unverified`; both carry `manual_review_required` licence status. The contract disables automatic download and does not define an approved download URL.

## Submission consequences

1. **Raw INX data:** exclude from GitHub release, journal supplement, and archive until IABAC/source permission is established.
2. **Raw HRDataset_v14 data:** exclude from this project's release unless the authors/rightsholders confirm that the intended sharing is compatible with the verified CC BY-NC-ND 4.0 terms and the exact acquisition chain is documented.
3. Hashes, schemas, aggregate statistics, code, derived publication tables, and qualified source locators may be reported subject to ordinary copyright/data-protection constraints; this assessment does not grant legal permission.
4. The manuscript title uses `Traceable`, not `Reproducible`, because the analytical evidence chain is strong but the upstream acquisition/release chain is not fully closed.

## Remaining blocker

**Dataset rights are not release-ready.** This is a legal/provenance blocker, not a reason to rerun the scientific experiments.