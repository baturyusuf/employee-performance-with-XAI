# Dataset Provenance, License, and Redistribution Report

A checksum proves identity of bytes already held; it does not prove authorship, authenticity, consent, license ownership, or redistribution permission. A public mirror and its repository license, when present, are not treated as authority for third-party dataset rights.

## inx_employee_performance

- Scope/shape: `core_primary`, 1200 rows × 28 columns.
- Primary record: https://iabac.org/exam/p2/CDS_Project_2_INX_Future_Emp_Data_V1.6.pdf
- Source/authenticity status: `partial_official_record_local_byte_chain_unverified`.
- Synthetic/fictitious evidence: `official_project_brief_states_not_actual_organization`.
- License status: `restrictive_notice_verified_permission_scope_requires_review`; identifier: `IABAC-all-rights-reserved`.
- License evidence: The official project brief states all rights reserved and that reproduction of the material requires IABAC permission. Whether and how that notice applies to redistribution of the linked workbook requires author/legal confirmation.
- Redistribution status: `blocked_pending_written_permission`.
- Citation status: `primary_project_record_verified_no_dataset_doi`.
- Proposed citation: IABAC. Employee Performance Analysis: INX Future Inc., Project 10281, document CDS_Project_2_INX_Future_Emp_Data_V1.6 (2017). https://iabac.org/exam/p2/CDS_Project_2_INX_Future_Emp_Data_V1.6.pdf (accessed 2026-09-07).
- Public-package rule: Exclude both raw files; publish hashes, schema, acquisition instructions, and aggregate evidence only.
- Required resolution: Obtain written redistribution permission or an authoritative dataset-specific license and independently bind the local workbook bytes to the authorized source.

## hrdataset_v14

- Scope/shape: `core_replication`, 311 rows × 36 columns.
- Primary record: https://www.kaggle.com/datasets/rhuebner/human-resources-data-set
- Source/authenticity status: `partial_author_record_and_exact_mirror_chain_no_author_file_hash`.
- Synthetic/fictitious evidence: `author_record_states_created_for_fictitious_company`.
- License status: `author_record_verified_conditional`; identifier: `CC-BY-NC-ND-4.0`.
- License evidence: The authors' Kaggle description grants learning/teaching use and directs sharing under CC BY-NC-ND 4.0. The license permits sharing with attribution for noncommercial use but prohibits distribution of adapted material.
- Redistribution status: `blocked_pending_exact_author_file_binding_and_nc_nd_review`.
- Citation status: `author_record_verified_no_dataset_doi`.
- Proposed citation: Huebner, R.; Patalano, C. Human Resources Data Set, Version 14. Kaggle. https://www.kaggle.com/datasets/rhuebner/human-resources-data-set (accessed 2026-09-07).
- Public-package rule: Exclude raw data until the exact authors' version is byte-bound and noncommercial/no-derivatives redistribution compliance is approved; publish hashes, acquisition instructions, schema, and aggregates only.
- Required resolution: Bind the local bytes to an author-controlled v14 artifact and obtain author/legal confirmation that the intended archive and any transformations satisfy attribution, noncommercial, and no-derivatives conditions.

## ibm_hr_analytics

- Scope/shape: `supplementary_performance_and_attrition`, 1470 rows × 35 columns.
- Primary record: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- Source/authenticity status: `partial_kaggle_record_and_exact_mirror_chain_no_ibm_archive`.
- Synthetic/fictitious evidence: `kaggle_record_states_fictional_and_created_by_ibm_data_scientists`.
- License status: `platform_record_verified_rights_chain_requires_review`; identifier: `ODbL-1.0-and-DbCL-1.0`.
- License evidence: Kaggle metadata labels the database as Open Database and its contents as Database Contents. ODbL/DbCL attribution, notice, and share-alike obligations require review, and the uploader-to-IBM rights chain is not independently verified.
- Redistribution status: `blocked_pending_rights_chain_and_odbl_dbcl_compliance`.
- Citation status: `platform_record_verified_no_dataset_doi`.
- Proposed citation: IBM HR Analytics Employee Attrition & Performance [fictional dataset]. Kaggle dataset record maintained by pavansubhash. https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset (accessed 2026-09-07).
- Public-package rule: Exclude raw data pending IBM/uploader rights-chain and ODbL/DbCL compliance review; publish hashes, acquisition instructions, schema, and aggregates only.
- Required resolution: Locate an authoritative IBM archival source or obtain rights-chain confirmation and document the exact attribution/share-alike notices before any raw redistribution.

## employee_turnover

- Scope/shape: `supplementary_related_task`, 14999 rows × 10 columns.
- Primary record: https://www.kaggle.com/datasets/ludobenistant/hr-analytics
- Source/authenticity status: `unresolved_exact_mirror_only_original_record_unavailable`.
- Synthetic/fictitious evidence: `authoritative_secondary_sas_documentation_states_simulated`.
- License status: `unverified`; identifier: `not verified`.
- License evidence: No accessible authoritative license statement was verified. Licenses attached to third-party copies conflict and do not establish upstream rights.
- Redistribution status: `blocked_no_authoritative_license`.
- Citation status: `unresolved_original_record_unavailable`.
- Proposed citation: Human Resources Analytics / HR_comma_sep.csv. Historical Kaggle locator: https://www.kaggle.com/datasets/ludobenistant/hr-analytics (unavailable when checked 2026-09-07); provenance corroborated only by SAS documentation.
- Public-package rule: Exclude raw data; publish hashes, the immutable mirror locator, schema, aggregate results, and the unresolved source/license warning only.
- Required resolution: Identify the original author/owner, recover an authoritative license and citation record, and bind the local bytes to that record before any redistribution.

## Cross-dataset decision

No raw dataset is approved for redistribution. The current reproducibility boundary is hashes, schemas, acquisition instructions, and aggregate evidence. This is a conservative publication decision rather than a legal conclusion.
