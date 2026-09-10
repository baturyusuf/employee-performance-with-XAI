# HRDataset_v14 synthetic-disclosure audit

## Finding

HRDataset_v14 must be described as a **publicly available synthetic teaching dataset representing a fictitious organizational setting**. It is not an observed independent company cohort.

The source trail supports this wording:

- The dataset creator's [HRDataset_v14 codebook](https://rpubs.com/rhuebner/hrd_cb_v14) describes the data as synthetic and created for a graduate HR case study.
- The creator's [Kaggle dataset page](https://www.kaggle.com/datasets/rhuebner/human-resources-data-set) describes a fictitious company and identifies the resource as a synthetic teaching dataset.

Sources were checked on 2026-09-10. This audit follows the creator-controlled documentation rather than inferring realism from the table's employee-like records.

## Required manuscript language

Use this source-bounded description at first mention:

> We used HRDataset_v14, a publicly available synthetic teaching dataset created for a graduate HR case study and representing a fictitious organizational setting.

Describe its analytical role as **secondary protocol sensitivity on a synthetic HR teaching dataset**. Its results show how the evaluation protocol behaves under another feature schema and target formulation.

The Discussion and Limitations must state:

> Because HRDataset_v14 is synthetic, its results demonstrate protocol portability and target-formulation sensitivity rather than real-world cross-organizational transport.

## Section audit

| Location | Required change |
| --- | --- |
| Abstract | Identify the secondary analysis as synthetic protocol sensitivity; remove “cross-dataset replication.” |
| Introduction/contribution list | Describe protocol portability under a second synthetic schema. |
| Dataset description and Table 2 | State creator, teaching purpose, fictitious setting, and synthetic status. |
| HR methods | Retain separate fitting and target mappings; label the exercise methodological sensitivity. |
| Figure 7 caption | State synthetic dataset and exclude real-world transport interpretation. |
| Results | Report numerical results as descriptive sensitivity evidence. |
| Discussion | State that synthetic results do not demonstrate cross-organizational transport. |
| Limitations | Treat the absence of a real external cohort as a study limitation. |
| Conclusion | Bound the contribution to protocol portability and target-formulation sensitivity. |

## Prohibited interpretations

Do not use “external validation,” “independent real-world validation,” “cross-organizational validation,” “real-world replication,” or “external generalizability evidence” for HRDataset_v14. Synthetic status does not grant redistribution permission; the existing data-rights assessment remains unresolved.

