"""Generate reviewable literature artifacts from the fresh public metadata receipts."""
import csv, html, json, pathlib, re, unicodedata
from audit_metadata import OUT, ROOT

existing=json.loads((OUT/'existing_reference_metadata.json').read_text(encoding='utf-8'))
additions=json.loads((OUT/'addition_metadata.json').read_text(encoding='utf-8'))
for row in additions:
    if row['key']=='tursunbayeva2024ai': row['key']='bargil2024ai'
(OUT/'addition_metadata.json').write_text(json.dumps(additions,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def year(m):
    return (m.get('published-print') or m.get('published') or m.get('issued'))['date-parts'][0][0]

def bib(row):
    m=row['registered_metadata']
    fields={'author':' and '.join(a['family']+', '+a.get('given','') for a in m['author']),
      'title':html.unescape(': '.join(m['title']+m.get('subtitle',[]))),
      'journal':html.unescape(m['container-title'][0]),'year':str(year(m)),
      'volume':m.get('volume'),'number':m.get('issue'),
      'pages':m.get('article-number') or m.get('page'),'doi':m['DOI'],
      'url':'https://doi.org/'+m['DOI']}
    return '@article{'+row['key']+',\n'+',\n'.join('  '+k+' = {'+v.replace('&',r'\&')+'}' for k,v in fields.items() if v)+'\n}\n'

(OUT/'VERIFIED_ADDITIONS.bib').write_text('\n'.join(bib(r) for r in additions),encoding='utf-8')

support={
 'archana2019application':('same-INX classifier comparison','Publisher abstract identifies INX; page naming differs from structured metadata; retain visible published initials pending name-order resolution.'),
 'lather2019prediction':('employee-performance classifier literature','Metadata independently verified; detailed dataset and validation design not newly established from full text.'),
 'li2021employee':('organizational employee-performance classification','Official proceedings PDF pp. 6870-6876; logistic regression, decision tree and naive Bayes; three internal HR tables.'),
 'patel2022ranker':('same-INX employee-performance classification','Publisher full text previously linked; current metadata verifies study identity. Detailed design comparison is not a fresh replication.'),
 'adeniyi2022comparison':('HRDataset performance comparison','Publisher full text: 311 records/36 fields; ANN/RF/DT and 80:20 split. Publisher title is authoritative over a conflicting Crossref title deposit.'),
 'nayem2024unbiased':('employee-performance application','Publisher abstract: Bangladesh employee dataset and classifiers; title wording does not establish unbiasedness.'),
 'abufaty2025integrating':('adjacent churn/ranking and same-INX use','Publisher abstract/full-text record supports adjacent task; cannot transfer churn claims to ordinal performance.'),
 'putri2026klasifikasi':('same-INX random-forest application','Current publisher metadata confirms 2026 work; detailed design is limited to accessible publisher evidence.'),
 'chaudhary2025integrated':('adjacent HR churn explanations','Publisher abstract supports logistic regression/random forest with SHAP, not ordinal performance validation.'),
 'tambe2019artificial':('HR-AI organizational constraints','Conceptual HR guidance; not empirical evidence of fairness or deployment benefit.'),
 'leichtdeobald2019challenges':('personal-integrity boundaries in HR','Normative framing; not a predictive benchmark.'),
 'kochling2020discriminated':('HR algorithmic fairness limitations','Systematic HR review; does not certify fairness of this manuscript system.'),
 'giermindl2021dark':('risks of people analytics','Volume 31(3) is the 2022 version of record; 2021 is the online-first year.'),
 'lundberg2017unified':('SHAP additive attribution foundation','Official NeurIPS abstract verifies method; additive attribution is not causality.'),
 'lundberg2020local':('Tree SHAP foundation','Exact tree attribution foundation; does not itself establish explanation stability.'),
 'alvarezmelis2018robustness':('explanation sensitivity precedent','Official arXiv record; keep explicitly a preprint, not a peer-reviewed journal article.'),
 'yeh2019infidelity':('quantitative infidelity and sensitivity','Official proceedings metadata gives Pradeep K. Ravikumar; the existing entry omitted K.'),
 'slack2020fooling':('adversarial limits of post-hoc explainers','Crossref subtitle supplies full title; explanation plausibility cannot certify unbiased behavior.'),
 'cawley2010overfitting':('selection/evaluation separation','JMLR model-selection paper; supports nested validation rationale.'),
 'kaufman2012leakage':('leakage definition and avoidance','Crossref subtitle supplies full title; no verification of feature timestamps in the present data.'),
 'kapoor2023leakage':('leakage/reproducibility audit rationale','General methodological precedent; not unique to HR.'),
 'guo2017calibration':('post-hoc calibration precedent','PMLR source; supports probability-quality evaluation, not a universal calibration improvement claim.'),
 'vaicenavicius2019evaluating':('multidimensional calibration','Official PMLR metadata lists Thomas Schön; existing Thomas B. Schön is an initial variant requiring source-consistent formatting.'),
 'mitchell2019model':('model reporting and intended use','Reporting framework; manuscript should distinguish reporting from immutable artifact lineage.'),
 'pineau2021improving':('reproducibility reporting','Official JMLR metadata strips some accents and space before subtitle; these are normalization variants, not another paper.'),
 'mccullagh1980regression':('proportional-odds foundation','Supports cumulative-link model structure; does not establish assumption validity on this data.'),
 'frank2001simple':('ordinal threshold decomposition','Supports ordinal construction; does not guarantee extreme-class recognition.'),
 'cardoso2011measuring':('ordinal evaluation','Ordinal metric rationale; not support for a universally optimal criterion.'),
 'cohen1968weighted':('weighted kappa','Weighted agreement definition; QWK result remains manuscript empirical evidence.'),
 'epstein1969scoring':('ranked probability score','Ranked-category probability scoring foundation.'),
 'gneiting2007strictly':('proper scoring rules','Proper scoring rule framework, not a guarantee that a particular fitted model is calibrated.'),
 'breiman2001random':('Random Forest algorithm','Method attribution only; does not support empirical superiority on this study.'),
 'chen2016xgboost':('XGBoost algorithm','Full title joins Crossref title and subtitle; method attribution only.'),
 'ke2017lightgbm':('LightGBM algorithm','Official NeurIPS proceedings confirm author order and paper identity; method attribution only.')}

def tags(row):
    d={}
    for tag in row.get('primary_metadata_tags',[]):
        n=re.search(r'(?:name|property)="([^"]+)"',tag); v=re.search(r'content="([^"]*)"',tag)
        if n and v: d.setdefault(n[1],[]).append(html.unescape(v[1]))
    return d

rows=[]
manuscript=(ROOT/'manuscript/mdpi_information/main.md').read_text(encoding='utf-8')
for r in existing:
    e=r['source_bib']; m=r.get('registered_metadata',{}); t=tags(r); k=r['key']
    title=html.unescape(': '.join(m.get('title',[])+m.get('subtitle',[]))) if m else '; '.join(t.get('citation_title',[]))
    authors='; '.join(a.get('given','')+' '+a.get('family','') for a in m.get('author',[])) if m else '; '.join(t.get('citation_author',[]))
    venue=html.unescape('; '.join(m.get('container-title',[]))) if m else '; '.join(t.get('citation_journal_title',t.get('citation_inbook_title',[])))
    checked_year=str(year(m)) if m else '; '.join(t.get('citation_publication_date',t.get('citation_date',[])))
    status='verified_metadata'; action='Use sentence-case title and consistent publication-type fields.'
    if k=='adeniyi2022comparison':
        status='publisher_crossref_discrepancy_resolved_to_publisher'; action='Retain existing publisher full-text title; do not replace with truncated/different Crossref title.'
    elif k=='giermindl2021dark':
        status='year_correction_required'; action='Change bibliographic year to 2022 for volume 31(3), pages 410-435; key may remain stable.'
    elif k=='archana2019application':
        status='author_name_order_and_asserted_doi_unresolved'; action='Visible publisher page uses Archana B.; Sharma S.; Singh S.; Rafsan A.; structured metadata expands differently. Do not infer surnames or add an unregistered DOI.'
    elif k=='yeh2019infidelity':
        status='author_initial_correction_required'; action='Use Pradeep K. Ravikumar to match official proceedings metadata.'
    elif k=='vaicenavicius2019evaluating':
        status='author_initial_variant'; action='Official PMLR metadata lists Thomas Schön; use this name form consistently.'
    if k=='li2021employee':
        title='Employee Performance Prediction using Different Supervised Classifiers'; authors='Merry Grace T. Li; Macrina Lazo; Ariel Kelly Balan; Joel de Goma'; venue='Proceedings of the 11th Annual International Conference on Industrial Engineering and Operations Management'; checked_year='2021'
    notice='no_update_notice_in_returned_crossref_metadata; not a comprehensive retraction clearance' if m else 'not_verified_by_a_retraction_registry; no clean-status assertion'
    role,note=support[k]
    rows.append({'citation_key':k,'source_title':e['title'],'verified_record_title':title,'verified_record_authors':authors,
      'verified_record_year':checked_year,'verified_record_venue':venue,'volume':m.get('volume','; '.join(t.get('citation_volume',[]))),
      'issue':m.get('issue','; '.join(t.get('citation_issue',[]))),'pages_or_article':m.get('article-number') or m.get('page') or ('; '.join(t.get('citation_firstpage',[]))+'-'+ '; '.join(t.get('citation_lastpage',[]))).strip('-'),
      'doi':e.get('doi',''),'doi_status':'registered_crossref_200' if m else 'not_in_existing_bib; no absence-of-DOI claim',
      'title_status':status,'authors_year_venue_status':status,'duplicate':'no_duplicate_key_or_doi','cited_in_source_manuscript':'yes' if '@'+k in manuscript else 'no',
      'claim_support_role':role,'claim_support_limit':note,'correction_retraction_status':notice,'required_action':action,
      'source_url':r['source_url'],'additional_primary_source':'https://paradigmplus.itiud.org/volume3/number3/adeniyi/' if k=='adeniyi2022comparison' else e.get('url',''),
      'checked_at_utc':r['checked_at_utc']})

qc=ROOT/'reports/submission_eswa/qc'; qc.mkdir(exist_ok=True)
with (qc/'REFERENCE_AUDIT.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)

# Remove only the ten full-page working snapshots created by this audit; receipts retain hashes and metadata.
snapshot_root=(OUT/'primary_records').resolve()
for r in existing:
    p=(snapshot_root/(r['key']+'.html')).resolve()
    if p.parent!=snapshot_root: raise RuntimeError('Unsafe snapshot path')
    if p.exists(): p.unlink()

print(f'Generated {len(rows)} reference audit rows and {len(additions)} verified bibliography additions.')
