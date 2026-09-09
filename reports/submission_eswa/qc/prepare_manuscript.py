"""Presentation-only ESWA adaptation of the frozen, approved manuscript."""
from pathlib import Path
import csv, hashlib, json, re, shutil

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/submission_eswa'
SOURCE = ROOT / 'manuscript/mdpi_information'
TITLE = 'Auditing Intelligent Employee-Performance Prediction Systems Beyond Accuracy: A Traceable XAI Protocol'
ABSTRACT = '''Intelligent human-resource prediction systems can appear reliable under aggregate scores while failing on consequential rating categories or relying on information whose availability is uncertain. Evaluating these systems requires a consistent connection between predictions, probability quality, explanations, and the claims made about them. We present an integrated audit protocol for ordinal employee-performance prediction that combines nested model selection, explicit information policies, ordinal and per-class evaluation, training-only calibration, exact-model explanations, stability and deletion diagnostics, support-aware subgroup and proxy analysis, and traceable evidence. The protocol is examined on a primary cross-sectional employee table and through partial cross-dataset protocol replication. Benchmark leaders differed across classification, ordinal, and probability metrics. Changing the selection objective altered 37 of 60 model-fold candidate choices. Under quadratic-weighted-kappa selection, nominal XGBoost attained kappa 0.6418 while its highest-rating recall fell to 0.0076; Random Forest also failed to recognize that rating despite strong aggregate ordinal scores. In the nominal-XGBoost information-policy sensitivity experiment, restricting timing-uncertain information reduced macro-F1 by 0.1937 after independent retuning. This contrast also changes feature count and does not isolate a temporal effect. Calibration findings depended on the metric, and subgroup support and target mapping further limited interpretation. The contribution is an operational evaluation protocol that makes such conflicting evidence visible and binds conclusions to the evaluated system. The cross-sectional results support bounded system-audit claims; they do not establish prospective performance, fairness certification, or deployment readiness.'''
KEYWORDS = 'Explainable artificial intelligence; Human resource analytics; Ordinal classification; Model auditing; Data leakage; Decision support'

def table(rows):
    return '\n'.join(['| ' + ' | '.join(rows[0]) + ' |', '| ' + ' | '.join(['---']*len(rows[0])) + ' |'] + ['| ' + ' | '.join(map(str,r)) + ' |' for r in rows[1:]])

def main():
    for p in ['manuscript','supplement','figures','highlights','cover_letter','declarations','reproducibility','qc','portal','literature']:
        (OUT/p).mkdir(parents=True,exist_ok=True)
    original = (SOURCE/'main.md').read_text(encoding='utf-8-sig')
    body = original[original.index('## 1. Introduction'):]
    intro = '''## 1. Introduction

Intelligent human-resource decision support increasingly uses employee records to predict organizational outcomes. Performance-rating models are a particularly demanding case: the labels are ordered, their extreme categories can be sparse, and the records reflect organizational decisions as well as employee characteristics. HR scholarship documents the resulting accountability, personal-integrity, and employee-response concerns [@tambe2019artificial; @leichtdeobald2019challenges; @kochling2020discriminated; @giermindl2021dark]. Reliable system testing must therefore connect a predictive score to the information and decision conditions under which it was obtained.

An aggregate score can conceal an important operational failure. In the present benchmark, Random Forest achieved strong quadratic weighted kappa and ordinal mean absolute error while failing to recognize the highest recorded rating. Favorable probability metrics can likewise coexist with a worse calibration summary, and stable feature rankings need not establish causal explanation. These conflicts make evaluation of an intelligent prediction system a question of jointly interpreting several kinds of evidence.

Employee-performance studies have established interest in classifier comparisons on INX and related organizational data [@archana2019application; @lather2019prediction; @li2021employee; @patel2022ranker; @adeniyi2022comparison; @nayem2024unbiased; @putri2026klasifikasi]. Prediction, explanation, leakage, calibration, and subgroup analyses provide useful individual checks. The methodological need addressed here is to perform those checks under a shared held-out prediction contract, so that the model being explained, the probabilities being calibrated, and the system being evaluated remain identifiable.

We ask: How does the assessment of an intelligent employee-performance prediction system change when selection objective, information availability, ordinal severity, calibration, explanation reliability, and subgroup/proxy diagnostics are audited jointly?

The study makes four contributions:

- An operational audit protocol connects information availability, nested selection, held-out prediction, and permissible system claims.
- A common benchmark and selection-objective sensitivity analysis expose disagreement between aggregate ordinal performance and extreme-class recognition.
- Fixed-schedule and independently retuned information policies separate feature-access sensitivity from the additional effects of within-policy selection.
- Calibration, explanation stability and deletion, support-aware subgroup/proxy diagnostics, and partial cross-dataset protocol replication are linked through traceable aggregate evidence.

The target is the recorded organizational rating, not a validated measure of capability or productivity. The intended application is research and system audit; prospective validity, causal determinants, fairness certification, and autonomous employment decisions are outside the supported claims.

'''
    body = intro + body[body.index('## 2. Related Work'):]
    body = body.replace('P0 excludes only identifier and target and is a diagnostic upper bound.', 'P0 excludes only identifier and target and is an information-rich diagnostic comparator; its superiority is not guaranteed.')
    body = body.replace('P0–P5 are evaluated using', 'The nominal-XGBoost information-policy sensitivity experiment evaluates P0–P5 using')
    body = body.replace('HRDataset_v14 uses a separate seven-feature conservative policy and models trained anew on that dataset.', 'HRDataset_v14 is a partial cross-dataset protocol replication with models trained anew on a separate seven-feature conservative policy: `EmpJobRole`, `EngagementSurvey`, `EmpJobSatisfaction`, `SpecialProjectsCount`, `DaysLateLast30`, `Absences`, and `ExperienceYearsAtThisCompany`. Feature-observation and rating-decision timestamps are unavailable, so temporal availability is not established.')
    body = body.replace('Independent protocol replication', 'Partial cross-dataset protocol replication').replace('independent replication, and manuscript-number provenance', 'partial cross-dataset protocol replication, and numerical evidence provenance')
    body = body.replace('These are different estimands;', 'These are different estimands;')
    body = body.replace('### 3.5 Nested benchmark', 'Hard class predictions were obtained using the unmodified argmax of class probabilities. No class-specific threshold optimization was performed. Label-only baselines use their declared hard-label rules.\n\n### 3.5 Nested benchmark')
    body = body.replace('### 3.12 Evidence identity and claim control\n\n' + body.split('### 3.12 Evidence identity and claim control\n\n')[1].split('\n\n## 4. Results')[0], '''### 3.12 Evidence identity and claim control

Each aggregate evidence package records input identities, preprocessing and feature contracts, source hashes, and an inventory. Every reported numerical claim is linked to a particular source row and its stored value in the Supplementary Evidence Ledger. Model, probability, explanation, and evaluation identities are checked before results are combined. This supports traceability of the reported evaluation while keeping employee records and fitted objects outside the distributed scientific supplement.

### 3.13 Research software assistance

Generative AI tools assisted research coding and documentation. The authors retain responsibility for the implementation, selection of analyses, and interpretation. Scientific outputs were checked through persisted input and model identities, independent replay or recomputation, and automated validation. [AUTHOR CONFIRMATION REQUIRED: complete the verified tool/provider/model inventory and describe the human code-review process before submission.]''')
    # User explicitly requests these inapplicable published cells be suppressed.
    for baseline in ['Stratified baseline','Ordinal-median baseline','Majority baseline']:
        body = re.sub(r'^\| '+re.escape(baseline)+r' \|.*$', lambda m:'| '+' | '.join([c.strip() for c in m.group(0).strip('|').split('|')][:-2]+['—','—'])+' |', body, flags=re.M)
    body = body.replace('### 4.2 Selection-objective sensitivity', 'Hard-label baselines are not treated as probabilistic comparators. Their probability metrics are therefore omitted; the training-only empirical class-prior predictor supplies the probability reference.\n\n### 4.2 Selection-objective sensitivity')
    body = body.replace('The empirical-prior probability reference had', 'See Supplementary Table S4 for precision, recall, F1 and support for ratings 2/3/4.\n\nThe empirical class-prior probability reference had')
    body = body.replace('The restriction from P3 Primary Leakage-Aware', 'In the nominal-XGBoost information-policy sensitivity experiment, the restriction from P3 Primary Leakage-Aware')
    body = body.replace('Because source timestamps are unavailable, these are information-policy sensitivities rather than prospective validation.', 'P3, P4, and P5 contain 20, 13, and 6 features, respectively. The contrasts jointly change timing assumptions, information content, and feature count; they do not isolate timing-risk removal. Because source timestamps are unavailable, these are information-policy sensitivities rather than prospective validation. They do not estimate the same deterioration for every model family.')
    body = body.replace('The large P3→P4 deterioration under both fixed and retuned schedules shows', 'The nominal-XGBoost information-policy sensitivity experiment shows a large P3→P4 deterioration under both fixed and retuned schedules. This indicates')
    body = body.replace('P4/P5 are semantically stricter but timestamps', 'The P3/P4/P5 feature counts also change from 20 to 13 to 6, confounding timing restrictions with information loss. P4/P5 are semantically stricter but timestamps')
    # Make all simultaneous intervals/support visible in a compact publication table.
    with (ROOT/'reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv').open(encoding='utf-8-sig',newline='') as f:
        subrows=list(csv.DictReader(f))
    rows=[['Attribute','Metric','Gap','Simultaneous interval','Eligible/declared']]
    names={'macro_f1':'Macro-F1','quadratic_weighted_kappa':'QWK','ordinal_mae':'MAE'}
    for r in subrows:
        if r['metric'] in names:
            rows.append([r['attribute'],names[r['metric']],f"{float(r['gap_max_minus_min']):.4f}",f"[{float(r['simultaneous_ci_low']):.4f}, {float(r['simultaneous_ci_high']):.4f}]",r['eligible_group_count']+'/'+r['declared_group_count']])
    start=body.index('| Attribute | Macro-F1 gap')
    end=body.index('\n\n### 4.7',start)
    body=body[:start]+table(rows)+'\n\nIntervals are simultaneous exploratory intervals conditional on the fixed models, fold identities, and support eligibility; the minimum group support is 30. Full group counts and class-denominator flags are in Supplementary Table S6. This is a descriptive audit, not fairness certification.'+body[end:]
    body=body.replace('Complete group endpoints, intervals, support, and denominator flags appear in the Supplementary Evidence Ledger.', 'Simultaneous exploratory intervals and eligibility appear in Table 8; Supplementary Table S6 gives group endpoints, support, and denominator flags.')
    body=body.replace('### 5.6 Relation to prior work', '''### 5.6 Implications for intelligent-system design and evaluation

The audit identifies checks that should accompany consideration of an intelligent HR prediction system. Evaluation should retain per-class precision, recall, F1, and support beside aggregate ordinal criteria, with explicit attention to consequential extreme ratings. Feature availability should be defined for the intended decision context and verified against observation times before prospective claims are made. A change in policy should be evaluated separately from retuning, because both can change the conclusion.

Probability evaluation should report which event and scoring rule improved and preserve adverse findings. Explanations should be bound to the exact prediction-producing model and assessed separately for numerical validity, stability, and model-level deletion behavior. Subgroup diagnostics require visible support and uncertainty; they do not certify fairness. Finally, an evidence lineage connecting inputs, fitted systems, outputs, and reported claims permits another investigator to audit how a conclusion was obtained. These are evaluation requirements, not sufficient conditions for deployment.

### 5.7 Relation to prior work''')
    body=body.replace('Fourth, samples are modest', 'The analysis program developed after earlier examinations of the same datasets. Additional sensitivity specifications were frozen before their execution, but the overall research program was not preregistered. Nested cross-validation controls model-selection bias within the stated procedure; it does not eliminate researcher-level adaptive-analysis risk.\n\nFourth, samples are modest')
    body=re.sub(r'Finally, institutional review,.*?They cannot be inferred from analysis\.', 'Source-rights, ethical applicability, and consent determinations require documentary resolution before submission or reuse. The statistical audit does not establish those permissions.',body,flags=re.S)
    # Keep scientific supplement only; declarations remain visibly unresolved in author-review draft.
    body=body[:body.index('## Supplementary Materials')]+'''## Supplementary Materials

The scientific supplement contains feature contracts (Table S1), preprocessing and hyperparameter settings (Table S2), cross-validation and seed specifications (Table S3), per-class metrics (Table S4), full audit gates (Table S5), subgroup uncertainty and support (Table S6), confusion matrices, and a numerical evidence ledger. It excludes employee records and fitted models.

## Ethics and informed consent

[AUTHOR AND INSTITUTION CONFIRMATION REQUIRED: insert the documented ethics determination and linked consent applicability, including committee or institutional authority, reference and date.]

## Data availability

Aggregate scientific tables, feature schemas, and numerical evidence identifiers accompany this submission draft. Employee-level INX and HRDataset_v14 records are not redistributed because authoritative source-to-file provenance and the applicable redistribution permissions remain unresolved. Software and configuration release terms and a durable code location require rightsholder approval before public distribution. [AUTHOR CONFIRMATION REQUIRED: approve the final data and code availability statement and any repository link.]

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During preparation of this manuscript, the authors used OpenAI ChatGPT and Codex for drafting and revision assistance, literature organization, and document preparation. Research coding assistance is described in Methods. [AUTHOR CONFIRMATION REQUIRED: verify the complete tools and purposes, and attest to human review, editing and responsibility for the submitted work.]

## References
'''
    # Protocol gate table is first in reading order; renumber original tables.
    body=re.sub(r'Table (\d+)',lambda m:'Table '+str(int(m.group(1))+1),body)
    gate_rows=[['Stage','Input and audit','Failure condition','Permissible interpretation'],['Information','Feature timing and availability contract','Timing unverified','Sensitivity only'],['Selection','Nested training-only model selection','Test-informed selection','Evaluation invalid'],['Prediction','Held-out probabilities and per-class ordinal metrics','Extreme-class failure','Bound aggregate claim'],['Calibration','Cross-fitted training probabilities','Test-fit calibration','Calibration invalid'],['Explanation','Exact-fold model and attribution checks','Model identity mismatch','No explanation claim'],['Subgroup','Held-out outputs and support-aware intervals','Insufficient denominators','Descriptive or not estimated']]
    gates='''**Table 1. Operational audit gates and permitted claims**

'''+table(gate_rows)+'''

**Algorithm 1. Traceable audit protocol**

Input: dataset, ordered target, feature-information contract, candidate-model registry, selection rule, and fixed split identities.

1. Define the target, intended decision context, information policy, and claim limits.
2. Construct outer evaluation folds and inner training-only selection folds.
3. Fit preprocessing and choose candidates using only the relevant training partition.
4. Refit the selected model on outer-training data and predict its untouched outer-test fold exactly once.
5. Generate selected-candidate inner cross-fitted training probabilities; fit the predeclared calibrator on these probabilities and apply it to outer-test probabilities.
6. Evaluate aggregate ordinal scores, probability quality, and per-class errors using the declared decision rule.
7. Bind each held-out explanation to its exact prediction-producing fold model.
8. Assess explanation aggregation, ranking stability, and deletion behavior separately.
9. Evaluate subgroup support, exploratory uncertainty, and the distinct proxy questions.
10. Record source identities and bound each reported claim to its evidence.

Output: a traceable report of the evaluated system and the claims its evidence permits. Supplementary Table S5 provides the complete audit-gate specification. Figure 1 shows the separation between training operations and held-out assessment.

'''
    body=body.replace('### 3.2 Datasets, targets, and data quality',gates+'### 3.2 Datasets, targets, and data quality')
    # Six top-level result units; subordinate analyses retain all source text/numbers.
    body=body.replace('### 4.1 Metric-specific benchmark leaders and extreme-class behavior','### 4.1 Main intelligent-system benchmark')
    body=body.replace('### 4.2 Selection-objective sensitivity','### 4.2 Selection objective and extreme-class failure')
    body=body.replace('### 4.3 Repetition variability and ranking','#### Repetition variability and ranking')
    body=body.replace('### 4.4 P3→P4 timing/information sensitivity','### 4.3 Information-policy sensitivity')
    body=body.replace('### 4.5 Explanation stability and calibration','### 4.4 Calibration and explanation reliability')
    body=body.replace('### 4.6 Subgroup results across all prespecified attributes','### 4.5 Subgroup and proxy diagnostics')
    body=body.replace('### 4.7 Proxy diagnostics','#### Proxy diagnostics')
    body=body.replace('### 4.8 HR target-mapping and CV-design sensitivity','### 4.6 Partial cross-dataset protocol replication')
    body=body.replace('### 4.9 HR target-alias and data-quality sensitivity','#### Target-alias and data-quality sensitivity')
    # Copy only individually referenced public aggregate figure assets.
    figure_manifest=[]
    def copy_fig(m):
        caption,path=m.groups(); number=int(re.search(r'Figure (\d+)',caption).group(1))
        src=SOURCE/path
        target=OUT/'figures'/f'figure_{number:02d}.png'
        shutil.copyfile(src,target)
        if src.with_suffix('.svg').exists(): shutil.copyfile(src.with_suffix('.svg'),target.with_suffix('.svg'))
        figure_manifest.append({'figure':number,'source':src.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'destination':target.relative_to(OUT).as_posix()})
        return '!['+caption+'](../figures/'+target.name+')'
    body=re.sub(r'!\[(Figure \d+\..+)\]\(([^)]+)\)',copy_fig,body)
    body=body.replace('Figure 3. Fixed-schedule feature-policy sensitivity.', 'Figure 3. Nominal-XGBoost information-policy sensitivity under fixed and independently retuned schedules.')
    body=body.replace('Figure 7. HRDataset_v14 mapped-target replication.', 'Figure 7. Partial cross-dataset protocol replication on HRDataset_v14 with the retained three-class mapping.')
    manuscript='# '+TITLE+'\n\n## Abstract\n\n'+ABSTRACT+'\n\n**Keywords:** '+KEYWORDS+'\n\n'+body
    (OUT/'manuscript/main.md').write_text(manuscript,encoding='utf-8')
    shutil.copyfile(SOURCE/'references.bib',OUT/'manuscript/references.bib')
    (OUT/'qc/ORIGINAL_FIGURE_BINDINGS.json').write_text(json.dumps(figure_manifest,indent=2)+'\n')
    (OUT/'qc/SOURCE_PRESERVATION.json').write_text(json.dumps({'base_commit':'eb5ada99172ffbe8fbedefae8caae213a73586c2','claim_digest':'751208d036597461606bd02d68bfa9e642a4aa5ecf5df2df3fd7e46b99084aea','source_files':{str(p.relative_to(ROOT).as_posix()):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*') if p.is_file()}},indent=2)+'\n')
    (OUT/'qc/TITLE_APPROVAL.json').write_text(json.dumps({'proposed_title':TITLE,'status':'pending_author_approval','author_approvals':{}},indent=2)+'\n')
    print(json.dumps({'abstract_words':len(ABSTRACT.split()),'keywords':len(KEYWORDS.split(';')),'figures':len(figure_manifest),'main_written':True}))

if __name__=='__main__': main()
