# Editor-ready literature integration

These paragraphs add literature positioning only. They neither change the frozen scientific findings nor propose a new experiment. Citation keys correspond to `VERIFIED_ADDITIONS.bib`.

## Related-work bridge

Recent work in *Expert Systems with Applications* treats explanation reliability as a substantive evaluation problem. Calibrated explanations attach uncertainty information to probability estimates and feature contributions [@lofstrom2024calibrated], while stability measures test how local feature rankings respond to small data changes [@sepulveda2025enhancing]. In HR analytics, an organizational attrition case study uses SHAP to examine both the magnitude and direction of feature contributions [@manafivarkiani2025predicting]. Rule-oriented regression further illustrates how explicit, testable conditions can expose unwanted patterns in training data [@rass2026statistically]. These studies address complementary reliability questions; their outcomes, perturbations and intended uses should not be treated as interchangeable with ordinal employee-performance evaluation.

## Gap paragraph

The methodological gap addressed here is the traceable connection between audit questions that can produce conflicting conclusions about the same prediction system. Reporting guidance such as REFORMS already calls for transparent validation and reproducibility [@kapoor2024reforms], and recent XAI research examines calibration and explanation stability. Our contribution is an executable integration for ordinal employee-performance prediction: fixed evidence identities connect selection-objective sensitivity, information-availability policies, class-level and ordinal errors, probability reliability, explanation tests, and subgroup/proxy diagnostics. This makes disagreements between aggregate scores and specific failure modes inspectable. It is an application-grounded audit contribution, rather than a claim that the individual methods are new or that the protocol has established deployment safety.

## Governance boundary

Organizational case studies show that awareness of AI ethics does not ensure comprehensive governance in HR analytics [@bargil2024ai]. Accordingly, our technical diagnostics are inputs to accountable human review. They do not resolve data rights, employee consent, organizational legitimacy, or the consequences of acting on a prediction. General guidance on machine-learning pitfalls similarly motivates explicit limits on selection, evaluation and interpretation [@lones2024avoiding].

## Optional broad survey citation

The recent ESWA survey [@abusitta2024survey] can support a short statement that XAI methods differ in scope, assumptions and limitations. It should not substitute for the specific calibrated-explanation, stability and HR case-study comparisons above. Omit it if the Introduction already has adequate general XAI coverage; citation count is not the objective.

## Recommended six keywords

Employee performance prediction; Explainable artificial intelligence; Intelligent decision support; Ordinal classification; Model evaluation; Data leakage.

The keywords connect the actual outcome and audit scope with observed ESWA terminology. They avoid “fairness,” “unbiased,” “causal,” and “deployment-ready,” because the frozen findings do not establish those properties. The recommended set is editorial judgment, not a claim of term-frequency analysis. See `KEYWORD_COMPARISON.csv` for exact publisher keyword lists.
