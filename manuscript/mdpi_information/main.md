# Beyond Accuracy in HR Analytics: An LLM-Assisted Multi-Agent XAI Governance Framework for Leakage-Safe Employee Performance Decision Support

**Article type:** Original Research Article  
**Target journal:** *Information* (MDPI)  
**Authors:** Muhammed Yusuf Batur^1,* and Mehmet Göktürk^2  
**Affiliations:**  
^1 Rumeli University, AUTHOR_TO_COMPLETE; myusuf.batur@rumeli.edu.tr  
^2 Gebze Technical University, AUTHOR_TO_COMPLETE; gokturk@gtu.edu.tr  
* Correspondence: myusuf.batur@rumeli.edu.tr; AUTHOR_TO_COMPLETE

## Abstract

Human-resource analytics systems are often evaluated mainly by predictive accuracy, although employee-performance decision support also requires governance evidence about leakage, explanation reliability, calibration, fairness and proxy risk, and the practical actionability of counterfactual scenarios. This study presents a leakage-aware, SHAP-based, calibration-aware, fairness/proxy-audited, actionability-constrained governance framework for employee-performance decision support. The predictive component remains a tabular XGBoost model; large language models are not used for prediction. Instead, SHAP, calibration, subgroup, proxy-risk, leakage, and counterfactual evidence are serialized into structured case-level evidence that is interpreted by a governed OpenAI explanation layer and audited by deterministic multi-agent governance components. The primary internal research model is `no_salary_hike_no_attrition_no_department + XGBoost`; full-feature models are retained only as leakage-warning upper-bound baselines. External evidence includes HRDataset_v14 as an independent replication dataset with `PerformanceScore` mapped to the 2/3/4 target scale, while IBM and turnover datasets are treated only as restricted target-space or related-task robustness evidence. A final real OpenAI LLM-agent evaluation covered 80 cases, with 40 INX cases and 40 HRDataset_v14 cases using `gpt-5.4-mini`. Automated technical checks reported no unsupported, forbidden, or missing-warning failures in that run. Nevertheless, final readiness remains `not_ready` because high-severity proxy-risk and counterfactual-actionability blockers remain unresolved. The framework is therefore positioned as research-grade decision support, not as a deployment-ready or autonomous HR decision system.

**Keywords:** HR analytics; explainable artificial intelligence; decision support systems; model governance; data leakage; SHAP; counterfactual explanations; large language models; multi-agent systems; algorithmic fairness

## 1. Introduction

Predictive analytics in human-resource settings can affect domains where errors, proxy effects, and overconfident explanations carry organizational and ethical risk. Prior work on artificial intelligence in human resources emphasizes that data-driven HR systems face small-data constraints, accountability requirements, possible employee reactions, and limitations of purely technical optimization [@tambe2019aihr; @leichtdeobald2019hr]. For this reason, employee-performance prediction should not be assessed only by classification metrics. A responsible decision-support pipeline must also document whether the model uses outcome-proximal variables, whether probabilities are calibrated, whether subgroup performance or proxy signals create risks, whether explanations are stable, and whether counterfactual outputs describe practically actionable changes or merely technical model scenarios.

This manuscript reports a code-side evidence package for an LLM-assisted multi-agent XAI governance framework. The framework uses XGBoost as the predictive model [@chen2016xgboost], SHAP as attribution evidence [@lundberg2017shap; @lundberg2020tree], calibration metrics [@guo2017calibration], subgroup and proxy-risk audits [@mehrabi2021bias; @barocas2023fairness], and counterfactual actionability checks [@wachter2018counterfactual; @karimi2022recourse]. A governed LLM layer then converts structured evidence into explanation JSON, and deterministic governance agents audit leakage, fairness/proxy risk, calibration, SHAP stability, counterfactual actionability, explanation compliance, and supervisor readiness. The LLM does not predict employee performance and is not used to invent evidence.

The contribution is not a claim that XGBoost alone is a sufficient employee-performance system. Instead, the contribution is an auditable information-systems architecture for evidence-constrained HR analytics governance. The framework demonstrates how accuracy, leakage control, calibration, XAI, counterfactual actionability, LLM explanation faithfulness, guardrail behavior, and final readiness can be tracked together. The final repository snapshot is versioned as `v0.3-real-llm-governance-evidence`, and the manuscript tables and figures in this directory are generated from checked-in reports rather than new experiments.

## 2. Related Work

### 2.1 HR Analytics and Algorithmic Decision Support

HR analytics increasingly supports workforce planning, performance review, retention analysis, and other organizational decisions. However, HR data are often cross-sectional, historically biased, sparse for minority subgroups, and entangled with managerial processes. Tambe et al. identify the gap between the promise and reality of AI in HR management, including small data sets, fairness constraints, and adverse employee reactions [@tambe2019aihr]. Leicht-Deobald et al. argue that algorithm-based HR decision-making can affect personal integrity and organizational control [@leichtdeobald2019hr]. These concerns motivate decision-support framing rather than autonomous HR decision claims.

### 2.2 Explainable Artificial Intelligence for Decision Support

XAI methods such as LIME and SHAP were developed to make black-box predictions more inspectable [@ribeiro2016lime; @lundberg2017shap]. For tree ensembles, TreeSHAP provides efficient attribution estimates [@lundberg2020tree]. However, explanation methods should be evaluated in context and should not be mistaken for causal analysis [@doshi2017rigorous; @molnar2022interpretable]. In this project, SHAP is used as attribution evidence only.

### 2.3 Leakage and Outcome-Proximal Features

Data leakage occurs when model inputs encode information that would not be available or appropriate at decision time. In HR data, outcome-proximal fields such as salary-hike percentages or attrition outcomes can inflate apparent predictive performance. The current framework therefore treats full-feature models as leakage-warning upper-bound baselines only and selects a primary candidate that excludes salary-hike, attrition, department, identity, and direct sensitive/audit-only fields.

### 2.4 Fairness, Proxy Risk, and Subgroup Auditing

Fairness cannot be established by simply removing direct group variables. Proxy variables may reconstruct sensitive or organizational group membership, and subgroup metrics remain sensitive to support thresholds [@mehrabi2021bias; @barocas2023fairness]. Selbst et al. further warn against abstracting fairness from sociotechnical context [@selbst2019abstraction]. The framework therefore reports subgroup and proxy-risk diagnostics as risk evidence rather than as proof of fairness.

### 2.5 Counterfactual Explanations and Actionability

Counterfactual explanations can describe what changes would alter a model output [@wachter2018counterfactual], but actionable recourse is a stronger requirement than technical validity [@karimi2022recourse]. In HR performance settings, many changes may depend on managers, organizational assignments, review timing, or policy decisions. The framework therefore classifies counterfactuals by actionability mode and avoids employee-prescription language.

### 2.6 LLMs and Agents for Governed Explanations

LLMs can help structure complex evidence into readable explanations, but hallucination and unsupported claims remain central risks [@ji2023hallucination]. Documentation frameworks such as model cards and datasheets support traceability and intended-use limits [@mitchell2019modelcards; @gebru2021datasheets]. The present framework constrains the LLM to a structured evidence schema, then audits generated explanations through rule-based faithfulness checks and specialist governance agents.

## 3. Materials and Methods

### 3.1 Task Definition and Intended Use

The task is employee-performance decision support for research and governance review. The system may be used by researchers, HR analysts, model auditors, and governance reviewers to inspect predictive and explanatory evidence. It must not be used for autonomous hiring, firing, promotion, compensation, discipline, applicant screening, or individual employment decisions.

### 3.2 Datasets and External Validation Roles

Dataset roles are defined before interpretation. The INX dataset is the internal primary benchmark. HRDataset_v14 is the independent external replication dataset because it contains `PerformanceScore`, which is mapped to the 2/3/4 performance target scale. IBM HR Analytics performance is restricted target-space robustness because the audited `PerformanceRating` values are restricted to classes 3 and 4. IBM Attrition and Employee Turnover are related HR risk task-transfer evidence only. Table 1 and Figure 2 summarize the claim boundaries.

**Table 1 source:** `manuscript/mdpi_information/tables/table1_dataset_roles.csv` and `.md`.  
**Figure 2 source:** `manuscript/mdpi_information/figures/figure2_dataset_roles_scope.svg` and `.png`.

### 3.3 Feature Governance and Feature-Set Policies

The primary research model is `no_salary_hike_no_attrition_no_department + XGBoost`. The main comparison baseline is `no_salary_hike_no_attrition + XGBoost`, and the strict proxy-sensitivity baseline is `no_salary_hike_no_attrition_no_department_no_job_role + XGBoost`. Full-feature models are historical leakage-warning upper-bound baselines only. Table 2 summarizes the governance interpretation of each policy.

### 3.4 Predictive Modelling Protocol

The predictive model family is XGBoost. The model is evaluated with classification and ordinal metrics: macro-F1, balanced accuracy, weighted-F1 where available, quadratic weighted kappa, ordinal mean absolute error, severe error rate, log loss, Brier score, and expected calibration error. External validation outputs are generated under `reports/external_validation/`; cross-dataset INX-to-HRDataset transportability is reported as infeasible or too limited because only three department-free safe common features overlap.

### 3.5 SHAP Attribution and Explanation Stability

SHAP evidence is used for local and grouped feature attribution. Explanation stability is summarized with grouped top-k overlap and rank-stability indicators. The manuscript does not treat SHAP as causal evidence.

### 3.6 Calibration Diagnostics

Calibration is assessed with log loss, multiclass Brier score, and ECE. The governance reports frame probabilities as approximate model confidence rather than objective correctness.

### 3.7 Fairness and Proxy-Risk Audit

The framework audits available subgroup variables and proxy reconstructability. Department removal is treated as a policy choice, not proof that fairness risk has been solved. The primary model still contains EmpJobRole, which is documented as a department-proxy risk.

### 3.8 Counterfactual Actionability Protocol

Counterfactual outputs are classified by actionability mode: employee-actionable, manager-actionable, organization-actionable, ethically risky, not actionable, or unavailable. Technical validity is distinguished from practical actionability.

### 3.9 Governed LLM Explanation Layer

The governed LLM layer consumes only structured `CompleteCaseEvidence` JSON. It produces short and detailed explanations, warnings, evidence references, unsupported-evidence flags, human-review reminders, SHAP non-causality reminders, and non-autonomous-HR-use reminders. The final 80-case run used the real OpenAI path with `gpt-5.4-mini`; stub/dry-run outputs are excluded from manuscript-grade real LLM evidence.

### 3.10 Multi-Agent Governance Audit

Specialist agents audit leakage risk, fairness/proxy risk, calibration reliability, SHAP stability, counterfactual actionability, explanation compliance, and supervisor readiness. The agents audit evidence and warnings; they do not make HR decisions.

### 3.11 G-XAIR Readiness Dashboard

G-XAIR is implemented as a component readiness dashboard, not as a universal ethical score. Components include performance adequacy, leakage robustness, explanation stability, calibration reliability, fairness robustness, counterfactual actionability, proxy risk, LLM faithfulness, chatbot guardrails, and external validation robustness.

### 3.12 Reproducibility and Evidence Manifest

The final evidence package is bound by `reports/manuscript_assets/final_evidence_manifest/`, including Markdown, CSV, and JSON manifests with row counts, hashes, manuscript-grade status, claim role, and run scope. The final evidence snapshot is tagged `v0.3-real-llm-governance-evidence`.

## 4. Results

### 4.1 Internal and External Performance Evidence

Table 3 reports performance metrics from the checked-in external validation summaries. The INX primary model achieved macro-F1 0.59881, balanced accuracy 0.62606, QWK 0.637613, log loss 0.455092, Brier 0.260802, and ECE 0.063846. HRDataset_v14 achieved macro-F1 0.638943, balanced accuracy 0.656693, QWK 0.5945, log loss 0.550791, Brier 0.256523, and ECE 0.092454. IBM performance, IBM attrition, and Employee Turnover are included only under their restricted or related-task claim boundaries.

**Table 3 source:** `manuscript/mdpi_information/tables/table3_performance_metrics.csv`.  
**Figure 3 source:** `manuscript/mdpi_information/figures/figure3_performance_comparison.svg` and `.png`.

### 4.2 External Validation and Related-Task Boundaries

HRDataset_v14 is the only external dataset in this package that supports independent replication on a directly mappable performance target. IBM performance robustness has a restricted target space, while IBM attrition and Employee Turnover are related HR risk tasks. The high Employee Turnover performance should not be interpreted as employee-performance validation because the target is turnover.

### 4.3 Real LLM-Agent Governance Evaluation

The final real OpenAI LLM-agent evaluation covered 80 cases: 40 INX primary cases and 40 HRDataset_v14 cases. Run mode was `real`, `real_llm_used=True`, and the model was `gpt-5.4-mini`. Automated technical checks reported faithfulness pass rate 1.0, unsupported claim rate 0.0, forbidden claim rate 0.0, missing warning rate 0.0, parsing success rate 1.0, and agent compliance pass rate 1.0 for the final run. These values are technical evaluation results for the stated run scope and do not imply deployment readiness.

**Table 4 source:** `manuscript/mdpi_information/tables/table4_final_llm_agent_eval.csv`.  
**Figure 5 source:** `manuscript/mdpi_information/figures/figure5_llm_agent_summary.svg` and `.png`.

### 4.4 Chatbot Guardrail and Explanation Compliance

The final evidence manifest links chatbot guardrail outputs with 50 unsafe prompts and 25 safe prompts. The automated guardrail summary reports unsafe refusal rate 1.0 and safe audit-answer rate 1.0 for the evaluated prompt suite. This result supports technical guardrail behavior for the suite, not exhaustive coverage of all adversarial prompts.

### 4.5 G-XAIR Readiness Components

Table 5 and Figure 4 summarize the component dashboard. Performance adequacy and fairness robustness are warning-level components; leakage robustness, explanation stability, calibration reliability, LLM faithfulness, and chatbot guardrails pass under the available evidence. Counterfactual actionability and proxy risk fail with high severity.

### 4.6 Why the Final Readiness Label Remains Not Ready

The final readiness label remains `not_ready`. The two principal blockers are Proxy Risk Penalty and Counterfactual Actionability. Proxy risk remains high because Department can still be reconstructed from remaining fields; removing Department does not establish fairness. Counterfactual actionability remains weak because technically valid scenarios often require manager or organization-controlled changes rather than employee-controllable actions. Figure 6 visualizes these blockers.

## 5. Discussion

### 5.1 Accuracy Is Not Sufficient for HR Decision Support

The evidence package shows why HR analytics should be evaluated as a governed information system rather than a single predictive model. A model can produce useful macro-F1 and QWK values while still retaining proxy-risk, actionability, calibration, and governance limitations.

### 5.2 LLM Faithfulness Does Not Imply Deployment Readiness

The final LLM-agent run did not detect unsupported or forbidden claims, but this is a bounded automated evaluation. It does not replace human-subject evaluation, legal review, organizational validation, or monitoring after deployment. The LLM layer improves evidence communication only within the constraints of the supplied evidence schema.

### 5.3 Department Exclusion Is Not a Fairness Guarantee

The primary policy excludes Department, but EmpJobRole remains a documented proxy-risk concern. This supports the broader fairness literature: group-variable removal can reduce direct use but does not eliminate indirect inference or structural effects.

### 5.4 Valid Counterfactuals Are Not Necessarily Actionable

Counterfactual validity is a model property, not an employee instruction. In this project, low employee-only validity and manager/organization dependencies motivate a high-severity readiness blocker. The manuscript therefore frames counterfactuals as actionability-audited model scenarios only.

### 5.5 Implications for Responsible HR Information Systems

The framework can support responsible HR analytics research by making the evidence chain explicit: data and feature policies, model metrics, explanation stability, calibration, subgroup/proxy audits, counterfactual actionability, governed LLM explanations, agent audits, guardrails, and final readiness. The approach is designed to prevent overclaiming, not to justify automated HR decisions.

## 6. Limitations

The evidence relies on public cross-sectional datasets with provenance and licensing that should be independently verified before submission. HRDataset_v14 is a small external replication dataset. IBM performance uses restricted target classes 3 and 4, and attrition/turnover datasets are not direct performance validation. Cross-dataset INX-to-HRDataset transfer is limited by weak safe-feature overlap. Automated LLM-agent and chatbot checks are technical evaluations, not human-subject studies. The final readiness label remains `not_ready`, and the system is not deployment ready.

## 7. Conclusions

This manuscript draft presents an LLM-assisted multi-agent XAI governance framework for leakage-safe employee-performance decision support. The framework integrates XGBoost prediction, SHAP attribution, calibration diagnostics, fairness/proxy audits, counterfactual actionability review, governed LLM explanations, multi-agent audits, chatbot guardrails, and a readiness dashboard. The final 80-case real OpenAI evaluation supports evidence-constrained LLM explanation behavior for INX and HRDataset_v14, but the readiness label remains `not_ready` because proxy-risk and counterfactual-actionability blockers remain unresolved. The system should therefore be interpreted as a research-grade governance and decision-support framework, not as an operational HR decision system.

## Supplementary Materials

Supplementary files include the final evidence manifest, generated tables, generated figures, and repository reports under the tagged GitHub snapshot `v0.3-real-llm-governance-evidence`.

## Author Contributions

Conceptualization, AUTHOR_TO_COMPLETE; methodology, AUTHOR_TO_COMPLETE; software, AUTHOR_TO_COMPLETE; validation, AUTHOR_TO_COMPLETE; formal analysis, AUTHOR_TO_COMPLETE; investigation, AUTHOR_TO_COMPLETE; resources, AUTHOR_TO_COMPLETE; data curation, AUTHOR_TO_COMPLETE; writing--original draft preparation, AUTHOR_TO_COMPLETE; writing--review and editing, AUTHOR_TO_COMPLETE; supervision, AUTHOR_TO_COMPLETE. All authors have read and agreed to the published version of the manuscript. AUTHOR_TO_COMPLETE

## Funding

AUTHOR_TO_COMPLETE

## Institutional Review Board Statement

AUTHOR_TO_COMPLETE

## Informed Consent Statement

AUTHOR_TO_COMPLETE

## Data Availability Statement

The source code, configuration files, generated governance reports, and manuscript-support evidence package are available at https://github.com/baturyusuf/employee-performance-with-XAI under the tag `v0.3-real-llm-governance-evidence`. The project uses public HR datasets; readers should consult the original dataset providers and dataset cards for provenance and licensing details. Stub/dry-run outputs are retained for reproducibility and testing only and are excluded from manuscript-grade real LLM evidence.

## Acknowledgments

AUTHOR_TO_COMPLETE

## Conflicts of Interest

AUTHOR_TO_COMPLETE

## Use of Generative AI and AI-Assisted Technologies

During manuscript preparation, generative AI tools were used for language refinement, structural drafting assistance, and consistency checking under author supervision. The authors reviewed, edited, and verified all scientific claims, numerical results, tables, figures, and references. No generative AI system was used as the predictive model in the reported experiments.

## References

See `references.bib`.
