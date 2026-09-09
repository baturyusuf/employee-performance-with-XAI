# Reviewer C — ESWA editorial-fit simulation

Review date: 2026-09-09. Reviewer: a separate literature/editorial audit agent from the manuscript editor. This agent also prepared the fresh literature comparison, so this is independent manuscript-editor review, not an assertion of independence from its own literature work.

Reviewed artifact: `reports/submission_eswa/manuscript/main.md`, SHA-256 `5f7ede45efc9d32e5cee78ba0c9a130e66fbf9a7766f93f0f56ac4ae0b1e64e1`. Source Markdown was reviewed; this report does not claim to have inspected a compiled PDF, supplement rendering or the Editorial Manager PDF. Subsequent manuscript edits require a delta check.

## Editorial question and provisional decision

**Why publish this in Expert Systems with Applications?** The defensible answer is a concrete contribution to testing intelligent HR prediction systems: the manuscript demonstrates how a favorable aggregate evaluation can coexist with failure on a consequential class, and makes the model, probabilities, explanations and permissible claims traceable to the same evaluated system. Intelligent-system testing and HR management are within the [journal's official scope](https://shop.elsevier.com/journals/expert-systems-with-applications/0957-4174).

At the reviewed revision, I would request editorial clarification before considering external review. The strongest result is visible and the non-deployment boundaries are unusually clear, but the current related-work argument remains too close to “we assembled good practices.” The supplied literature update can address this without a new experiment. A completed internal simulation cannot promise journal acceptance.

## Findings and disposition

| ID | Finding | Classification | Required response without new experiments |
| --- | --- | --- | --- |
| C1 | Sections 2.3–2.6 and 5.7 claim integration, but do not yet compare the operational contribution with recent ESWA calibrated-explanation/stability work or REFORMS reporting guidance. This leaves the closest novelty objection unanswered. | VALID | Integrate specific 2024–2026 comparisons and explain the added executable linkage and observed interactions. Avoid “first” or “no existing work” language. Add the bounded comparison table to supplementary material if it improves reviewability. |
| C2 | Section 2.1 says four exact-INX studies, but the accompanying performance-study citation cluster contains the three identifiable direct performance studies; Abu-Faty et al., the fourth INX user, appears only in the subsequent churn cluster. | VALID | Name the four studies or explicitly separate three direct performance studies from the adjacent INX use. Preserve the distinction between the task labels; do not convert churn evidence into ordinal-performance validation. |
| C3 | “Operational protocol” is a credible scope claim, but another lab's successful use has not been demonstrated. HRDataset_v14 only partially replicates the protocol and the main empirical data are small public tables. | PARTIALLY VALID | Describe reusable inputs, checks and outputs through Table 1, Algorithm 1 and supplement, and keep reuse/generalization language bounded. No new dataset or user study is requested. |
| C4 | The abstract's strongest operational result—high QWK with near-zero highest-rating recall—is a persuasive ESWA contribution because it changes interpretation of a candidate system. | NOT SUPPORTED as a rejection objection | Retain the adverse finding and the class-level qualifier. Do not dilute it into a generic XAI overview or universal-best-model statement. |
| C5 | Unresolved source-to-byte provenance, data rights, ethics/consent and author declarations remain material submission barriers even though the scientific narrative is careful. | VALID | Preserve explicit internal author-action gates. Complete documentary determinations before submission; do not replace missing facts with reassuring boilerplate. |
| C6 | The absence of a newly proposed learning algorithm is itself sufficient reason to reject from ESWA. | NOT SUPPORTED | Official scope explicitly includes testing and management of intelligent systems; defend the operational testing contribution. Scope inclusion does not settle novelty or evidential sufficiency. |

## Strengths to retain

The title foregrounds an intelligent prediction system. Table 1 and Algorithm 1 make the audit process visible in the main paper. The abstract preserves selection sensitivity, extreme-class failure, the timing/information confound and metric-specific calibration. The Discussion turns those findings into concrete evaluation requirements while distinguishing them from sufficient conditions for deployment. The limits on recorded ratings, timestamps, subgroup inference, prospective validity and cross-dataset transport remain scientifically appropriate.

## Closure criterion

C1 and C2 can close through a targeted editorial revision and citation check. C3 remains a stated evidential limit. C5 remains open until author and documentary decisions are completed. No reviewer objection in this report authorizes a new scientific experiment, journal submission, licence choice, data release or acceptance claim.
