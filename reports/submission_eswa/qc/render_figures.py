"""Re-render existing aggregate evidence; no statistical or model execution."""
from pathlib import Path
import csv, json, hashlib, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/submission_eswa/figures'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':300})
COLORS=['#1c5e7c','#b65422','#454545','#6e5a88']
manifest=[]
def read(path):
    return list(csv.DictReader((ROOT/path).open(encoding='utf-8-sig',newline='')))
def save(fig,n,source,caption):
    fig.savefig(OUT/f'figure_{n:02}.png',bbox_inches='tight',pad_inches=.14)
    fig.savefig(OUT/f'figure_{n:02}.pdf',bbox_inches='tight',pad_inches=.14,metadata={'Author':'','Title':f'Figure {n}'})
    fig.savefig(OUT/f'figure_{n:02}.svg',bbox_inches='tight',pad_inches=.14)
    plt.close(fig)
    (OUT/f'figure_{n:02}_caption.txt').write_text(caption+'\n',encoding='utf-8')
    manifest.append({'figure':n,'sources':[{'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in source],'caption':caption})

# Protocol diagram is a conventional vector drawing of the stated method.
fig,ax=plt.subplots(figsize=(8.2,7)); ax.set(xlim=(0,10),ylim=(0,10)); ax.axis('off')
def box(x,y,w,h,label,color='#edf3f6',size=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.08',edgecolor='#39434a',facecolor=color,lw=1))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size)
def arrow(a,b): ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#39434a','lw':1.3})
box(2.9,8.8,4.2,.65,'Employee records and ordered rating')
box(2.9,7.65,4.2,.65,'Information policy and target contract')
arrow((5,8.8),(5,8.3))
box(.2,5.8,4.3,1.2,'TRAINING PARTITIONS\nPreprocessing and nested selection\nCross-fitted calibrator',size=10)
box(5.5,5.8,4.3,1.2,'UNTOUCHED OUTER-TEST PARTITION\nExactly-once held-out probabilities\nDeclared argmax decision rule',color='#fff3e9',size=9.5)
arrow((4.2,7.65),(2.4,7)); arrow((5.8,7.65),(7.6,7)); arrow((4.5,6.4),(5.5,6.4))
ax.text(5,6.85,'Apply only',ha='center',fontsize=8)
for x,label in [(0.2,'Ordinal and\nper-class errors'),(3.65,'Probability quality\nand calibration'),(7.1,'Exact-model XAI\nstability and deletion')]:
    box(x,3.8,2.7,1,label,color='#eef4ed'); arrow((7.5,5.8),(x+1.35,4.8))
box(2.9,2.2,4.2,.8,'Support-aware subgroup / proxy audit')
for x in [1.55,5,8.45]:arrow((x,3.8),(5,3))
box(2.1,.55,5.8,.9,'Bounded intelligent-system claim\nlinked to auditable source evidence',color='#edf0f5')
arrow((5,2.2),(5,1.45))
ax.text(5,.04,'No outer-test outcome enters selection, preprocessing or calibration fitting.',ha='center',fontsize=9)
save(fig,1,['reports/submission_eswa/manuscript/main.md'],'Audit protocol for intelligent employee-performance prediction. Training-only operations are isolated from held-out assessment. The diagram was drawn programmatically with AI-assisted coding and checked against the reported procedure.')

models=['cumulative_threshold_xgboost','xgboost','lightgbm','random_forest','logistic_regression','proportional_odds_logistic','stratified_baseline','ordinal_median_baseline','majority_baseline']
labels=['Cumulative XGBoost','Nominal XGBoost','LightGBM','Random Forest','Multinomial logistic','Proportional odds','Stratified','Ordinal median','Majority']
src='reports/research_log/major_revision_v3/phase1b_ordinal_benchmark/aggregate_metrics.csv'; data=read(src)
actual=sorted(set(r['model_name'] for r in data)); print('Benchmark systems:',actual)
# Registry spelling is explicit, never a fuzzy scientific join.
aliases={'logistic_regression':'logistic_regression','proportional_odds_logistic':'proportional_odds_logistic'}
fig,axes=plt.subplots(1,3,figsize=(9.4,4.8),sharey=True)
for ax,metric,title in zip(axes,['macro_f1','quadratic_weighted_kappa','ordinal_mae'],['Macro-F1 (higher better)','QWK (higher better)','MAE (lower better)']):
    vals=[]
    for m in models:
        hits=[r for r in data if r['model_name']==m and r['metric']==metric]
        if len(hits)!=1: raise ValueError((m,metric,len(hits)))
        vals.append(float(hits[0]['value']))
    ax.barh(range(9),vals,color=[COLORS[0] if i<6 else '#aeb8bd' for i in range(9)])
    ax.set_title(title,fontsize=10); ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
    ax.set_yticks(range(9),labels); ax.set_xlim(0,max(vals)*1.22)
    for i,v in enumerate(vals):ax.text(v+.008,i,f'{v:.3f}',va='center',fontsize=8)
axes[0].invert_yaxis();fig.tight_layout(w_pad=1)
save(fig,2,[src],'Nine-system P3 benchmark under macro-F1 selection. Metric-specific leaders differ; the hard-label baselines are not probability comparators.')

src='reports/research_log/major_revision_v3/phase1d_policy_retuning/aggregate_metrics.csv'; data=read(src)
fig,axes=plt.subplots(1,3,figsize=(9.2,3.4))
estimands=sorted(set(r['estimand'] for r in data)); print('Policy estimands:',estimands)
for ax,metric,title in zip(axes,['macro_f1','quadratic_weighted_kappa','ordinal_mae'],['Macro-F1','QWK','Ordinal MAE']):
    for j,e in enumerate(estimands):
        vals=[float(next(r['value'] for r in data if r['estimand']==e and r['metric']==metric and r['policy_id']==f'P{k}')) for k in range(6)]
        ax.plot(range(6),vals,['o-','s--'][j],color=COLORS[j],label='Fixed schedule' if e.startswith('fixed') else 'Within-policy retuning',ms=4)
    ax.set_xticks(range(6),[f'P{k}\n(n={v})' for k,v in enumerate([26,24,21,20,13,6])],fontsize=8);ax.set_title(title);ax.set_ylim(0,1 if metric!='ordinal_mae' else .7);ax.grid(alpha=.2)
axes[1].legend(loc='upper right',fontsize=8,frameon=False);fig.tight_layout()
save(fig,3,[src],'Nominal-XGBoost information-policy sensitivity under fixed and independently retuned schedules. Counts are retained features. P0 is an information-rich diagnostic comparator; P3/P4/P5 changes confound timing restrictions with feature count and information loss. No prospective validity is established.')

src='manuscript/mdpi_information/assets/source_data/figures/figure_05_global_grouped_shap_source.csv'; data=read(src); print('SHAP fields',data[0])
# Figure 4 preserves the canonical source plot; only convert its public vector to PDF below.

src='reports/research_log/major_revision_v3/phase2a_shap_stability_faithfulness/stability_summary.csv';data=[r for r in read(src) if r['top_k']=='5'];print('Stability types',[r['stability_type'] for r in data])
fig,axes=plt.subplots(1,2,figsize=(8.5,3.4))
for ax,key,label in zip(axes,['jaccard','spearman'],['Top-5 Jaccard','All-feature Spearman']):
    for i,r in enumerate(data):
        mean=float(r[key+'_mean']);lo=float(r[key+'_min']);hi=float(r[key+'_max'])
        ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=COLORS[i],capsize=4)
    ax.set_yticks(range(len(data)),[r['stability_type'].replace('canonical_outer_fold_pair','Outer-fold pairs').replace('model_seed_pair','Model-seed pairs').replace('training_resample_pair','Training-resample pairs').replace('_',' ') for r in data],fontsize=9)
    ax.set_xlim(.7,1.025);ax.set_title(label);ax.grid(axis='x',alpha=.2)
fig.tight_layout()
save(fig,5,[src],'Explanation ranking stability across outer folds, model seeds and outer-training resamples. Points are means and bars are observed ranges across dependent pairs, not confidence intervals.')

src='reports/research_log/major_revision_v3/phase2b_calibration_diagnostics/calibration_metric_summary.csv';data=read(src)
fig,axes=plt.subplots(1,2,figsize=(8.7,3.7))
for ax,metrics,labs in [(axes[0],['nll_log_loss','multiclass_brier','ranked_probability_score'],['Log loss','Brier','RPS']),(axes[1],['top_label_ece','macro_classwise_ece','mean_cumulative_ece'],['Top-label','Classwise','Cumulative'])]:
    for j,method in enumerate(['raw','sigmoid']):
        row=next(r for r in data if r['method']==method); ax.bar([i+(j-.5)*.3 for i in range(3)],[float(row[m]) for m in metrics],width=.28,label=method.capitalize(),color=COLORS[j],hatch='' if j==0 else '//')
    ax.set_xticks(range(3),labs,fontsize=9);ax.set_ylim(0,.65 if ax==axes[0] else .125);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_title('Proper probability scores (lower better)');axes[1].set_title('Calibration errors (lower better)');axes[0].legend(frameon=False);fig.tight_layout()
save(fig,6,[src],'Raw and cross-fitted sigmoid probability diagnostics for P3 nominal XGBoost. Sigmoid improves several scores but worsens the top-label calibration-error point estimate. These retrospective diagnostics do not validate future probabilities.')

src='reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/variability_summary.csv'; data=read(src);print('HR systems',sorted(set(r['system'] for r in data)))
fig,axes=plt.subplots(1,2,figsize=(8.7,3.6))
for ax,form,title in zip(axes,['primary_three_class','raw_order_four_class'],['Retained three-class target','Distinct four-class target']):
    for j,system in enumerate(['xgboost_raw','xgboost_sigmoid']):
        vals=[]
        for metric in ['macro_f1','quadratic_weighted_kappa']:
            r=next(r for r in data if r['formulation_id']==form and r['system']==system and r['metric']==metric)
            vals.append((float(r['mean']),float(r['minimum']),float(r['maximum'])))
        ax.errorbar([i+(j-.5)*.1 for i in range(2)],[v[0] for v in vals],yerr=[[v[0]-v[1] for v in vals],[v[2]-v[0] for v in vals]],fmt=['o','s'][j],color=COLORS[j],capsize=4,label=['Raw','Sigmoid'][j])
    ax.set_xticks([0,1],['Macro-F1','QWK']);ax.set_xlim(-.45,1.45);ax.set_ylim(.4,.78);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=9)
fig.tight_layout()
save(fig,7,[src],'Partial cross-dataset protocol replication on HRDataset_v14. Points and bars show five-repetition means and observed ranges, not confidence intervals. Three- and four-class target formulations are different estimands and must not be compared as an improvement sequence.')
(OUT/'FIGURE_SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
