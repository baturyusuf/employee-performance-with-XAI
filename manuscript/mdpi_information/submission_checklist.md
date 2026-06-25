# Submission Checklist

## Scientific Claim Boundaries

- [x] No autonomous HR decision claim is made.
- [x] No hiring, firing, promotion, salary, discipline, or applicant-screening recommendation is made.
- [x] SHAP is described as attribution, not causality.
- [x] Counterfactuals are described as model scenarios, not causal interventions.
- [x] Counterfactuals are not presented as employee prescriptions.
- [x] No fairness guarantee or unbiased-model claim is made.
- [x] Department removal is not presented as proof of fairness.
- [x] IBM attrition and Employee Turnover are not described as direct employee-performance validation.
- [x] IBM performance is described as restricted target-space robustness.
- [x] Stub/dry-run outputs are not cited as manuscript-grade real LLM evidence.
- [x] Final readiness remains `not_ready`.

## Evidence Traceability

- [x] Numerical performance results trace to `reports/external_validation/external_validation_summary.md` and `reports/manuscript_assets/external_validation_tables.md`.
- [x] Final 80-case real LLM-agent results trace to `reports/llm_explanations/llm_agent_eval_summary.csv` and `reports/llm_explanations/eval_case_manifest.csv`.
- [x] Readiness components trace to `reports/governance_reports/gxair_component_dashboard.csv`.
- [x] Readiness blocker explanation traces to `reports/manuscript_assets/final_evidence_manifest/readiness_not_ready_explanation.md`.
- [x] Generated table CSV/Markdown files are saved under `manuscript/mdpi_information/tables/`.
- [x] Generated figure SVG/PNG files are saved under `manuscript/mdpi_information/figures/`.
- [x] Figure and table manifests list source data files.

## Author and Submission Fields

- [ ] Department, address, ORCID, and submission metadata fields are completed.
- [ ] Funding statement is completed.
- [ ] Institutional review board statement is completed.
- [ ] Informed consent statement is completed.
- [ ] Conflicts of interest statement is completed.
- [ ] Acknowledgments are completed.
- [ ] CRediT author-contribution roles are confirmed by the authors.
- [ ] Dataset provenance and licensing are independently checked against original providers.
- [ ] MDPI template files are obtained from the current official source and used for final formatting.

## References and Formatting

- [x] References were selected from publisher, proceedings, arXiv, book, or official standards pages where possible.
- [x] No DOI was intentionally invented for references without a verified DOI.
- [ ] Authors perform final reference metadata and MDPI numerical style verification.
- [ ] LaTeX compiles under the official MDPI template.
- [ ] Figures and tables are checked visually in the compiled PDF.

## GenAI Disclosure

- [x] A controlled generative-AI disclosure is included.
- [x] The disclosure does not imply that generative AI produced or validated empirical results.
- [x] The disclosure states that no generative AI system was used as the predictive model.
