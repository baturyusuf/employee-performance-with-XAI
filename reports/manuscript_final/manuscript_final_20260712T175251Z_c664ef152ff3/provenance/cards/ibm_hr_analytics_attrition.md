# Dataset Card: IBM HR Analytics Employee Attrition and Performance (attrition related-task binding)

Run ID: `manuscript_final_20260712T175251Z_c664ef152ff3`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`
Provenance config hash: `f8349018704f719e9a2455322f983e3896e1f52bc4a93f7da5ee52d0a3a6cb64`

## File identity

- Raw file: `data/external/ibm_hr_analytics/raw.csv`
- SHA-256: `a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659`
- Shape: 1470 rows x 35 columns

## Role and claim boundary

- Role: `related_task_transfer`
- Task type: `binary_attrition_transfer`
- Allowed claim: Related attrition-task transfer evidence only; not employee-performance validation.

## Target mapping and observed support

Raw target: `Attrition`. Binary related-task mapping; attrition is not employee performance.

- `No`: n=1233 -> `0`
- `Yes`: n=237 -> `1`

## Source, mirror, and licence status

- Retrieval URL: `https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/master/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- Retrieval date: `manual_review_required`
- Known source/mirror status: `public_mirror_recorded_upstream_authenticity_unverified`
- Source-authenticity status: `manual_review_required`
- Licence: `manual_review_required`
- Licence verification: `manual_review_required`
- Citation/source: IBM HR Analytics Employee Attrition and Performance dataset; repository records the listed GitHub raw-file mirror, not an independently authenticated original source.
- Citation verification: `manual_review_required`

## Unresolved manual verification

- Confirm the actual retrieval date rather than treating the repository-add date as retrieval evidence.
- Confirm the authoritative original source and mirror chain.
- Confirm the dataset licence and reuse terms.
- Confirm the authoritative bibliographic citation.

Automated hashing and schema checks do not authenticate the upstream source or determine legal reuse rights. `manual_review_required` fields must be resolved by the author or another authorised reviewer before making stronger provenance or licence claims.
