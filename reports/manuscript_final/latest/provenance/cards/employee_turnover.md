# Dataset Card: Human Resources Analytics / Employee Turnover

Run ID: `manuscript_final_20260712T181754Z_c664ef152ff3`  
Config hash: `c664ef152ff3a9eef89c53a403b9c2e6f677340bea0307b02d97d07cc54bdfc3`
Provenance config hash: `f8349018704f719e9a2455322f983e3896e1f52bc4a93f7da5ee52d0a3a6cb64`

## File identity

- Raw file: `data/external/employee_turnover/raw.csv`
- SHA-256: `2510e274a90547f34c7b0db5a4ab70282c2710eb54252f14921bf980b81a928c`
- Shape: 14999 rows x 10 columns

## Role and claim boundary

- Role: `related_task_transfer`
- Task type: `binary_turnover_transfer`
- Allowed claim: Related turnover-task transfer evidence only; not employee-performance validation.

## Target mapping and observed support

Raw target: `left`. Identity mapping for the binary related turnover target; turnover is not employee performance.

- `0`: n=11428 -> `0`
- `1`: n=3571 -> `1`

## Source, mirror, and licence status

- Retrieval URL: `https://raw.githubusercontent.com/ucg8j/kaggle_HR/master/HR_comma_sep.csv`
- Retrieval date: `manual_review_required`
- Known source/mirror status: `public_mirror_recorded_upstream_authenticity_unverified`
- Source-authenticity status: `manual_review_required`
- Licence: `manual_review_required`
- Licence verification: `manual_review_required`
- Citation/source: Human Resources Analytics / Employee Turnover dataset; repository records the listed GitHub raw-file mirror, not an independently authenticated original source.
- Citation verification: `manual_review_required`

## Unresolved manual verification

- Confirm the actual retrieval date rather than treating the repository-add date as retrieval evidence.
- Confirm the authoritative original source and mirror chain.
- Confirm the dataset licence and reuse terms.
- Confirm the authoritative bibliographic citation.

Automated hashing and schema checks do not authenticate the upstream source or determine legal reuse rights. `manual_review_required` fields must be resolved by the author or another authorised reviewer before making stronger provenance or licence claims.
