
from __future__ import annotations
import argparse, json, math, tempfile, time, warnings
from itertools import combinations
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, cohen_kappa_score, roc_auc_score, average_precision_score
from src.data.preprocess import build_sklearn_preprocessor, load_validated_or_raw_data, split_features_and_target, run_preprocessing, get_feature_groups
from src.models.train_catboost import prepare_catboost_inputs
from src.explainability.counterfactuals import DEFAULT_ACTIONABLE_FEATURES, generate_counterfactual_candidates, predict_single_sample
from src.utils.config import SETTINGS
from sklearn.base import BaseEstimator, ClassifierMixin, clone

class LabelEncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        objective: str = "multi:softprob",
        eval_metric: str = "mlogloss",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=self.objective,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            num_class=len(self.classes_),
        )

        self.model_.fit(X, y_encoded)
        return self

    def predict(self, X):
        y_encoded_pred = self.model_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(y_encoded_pred)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


warnings.filterwarnings("ignore")

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def save_json(d: dict, p: Path):
    def conv(o):
        if isinstance(o, dict): return {str(k):conv(v) for k,v in o.items()}
        if isinstance(o,(list,tuple)): return [conv(v) for v in o]
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,(np.floating,)): return float(o)
        if isinstance(o,(np.bool_,)): return bool(o)
        if isinstance(o,np.ndarray): return o.tolist()
        return o
    ensure_dir(p.parent); p.write_text(json.dumps(conv(d),indent=2,ensure_ascii=False),encoding="utf-8")
def sf(x):
    try:
        y=float(x); return None if math.isnan(y) or math.isinf(y) else y
    except Exception: return None
def metric_pack(y_true,y_pred,y_proba=None,labels=None):
    y_true=np.asarray(y_true,dtype=int); y_pred=np.asarray(y_pred,dtype=int)
    labels=labels or sorted(np.unique(np.r_[y_true,y_pred]).tolist())
    out={"accuracy":sf(accuracy_score(y_true,y_pred)),"balanced_accuracy":sf(balanced_accuracy_score(y_true,y_pred)),
         "macro_f1":sf(f1_score(y_true,y_pred,average="macro",zero_division=0)),"weighted_f1":sf(f1_score(y_true,y_pred,average="weighted",zero_division=0)),
         "macro_precision":sf(precision_score(y_true,y_pred,average="macro",zero_division=0)),"weighted_precision":sf(precision_score(y_true,y_pred,average="weighted",zero_division=0)),
         "macro_recall":sf(recall_score(y_true,y_pred,average="macro",zero_division=0)),"weighted_recall":sf(recall_score(y_true,y_pred,average="weighted",zero_division=0)),
         "ordinal_mae":sf(mean_absolute_error(y_true,y_pred)),"quadratic_weighted_kappa":sf(cohen_kappa_score(y_true,y_pred,weights="quadratic")),
         "adjacent_accuracy":sf(np.mean(np.abs(y_true-y_pred)<=1)),"severe_error_rate":sf(np.mean(np.abs(y_true-y_pred)>1))}
    if y_proba is not None:
        try:
            ybin=label_binarize(y_true,classes=labels)
            if ybin.shape[1]==y_proba.shape[1]:
                out["roc_auc_ovr_macro"]=sf(roc_auc_score(ybin,y_proba,average="macro",multi_class="ovr"))
                out["roc_auc_ovr_weighted"]=sf(roc_auc_score(ybin,y_proba,average="weighted",multi_class="ovr"))
                out["pr_auc_macro"]=sf(average_precision_score(ybin,y_proba,average="macro"))
                out["pr_auc_weighted"]=sf(average_precision_score(ybin,y_proba,average="weighted"))
        except Exception:
            out["roc_auc_ovr_macro"]=out["roc_auc_ovr_weighted"]=out["pr_auc_macro"]=out["pr_auc_weighted"]=None
    return out
def summarize(df, groups):
    nums=[c for c in df.columns if c not in groups+["fold"] and pd.api.types.is_numeric_dtype(df[c])]
    rows=[]
    for k,g in df.groupby(groups,dropna=False):
        if not isinstance(k,tuple): k=(k,)
        row={c:v for c,v in zip(groups,k)}; row["n_folds"]=len(g)
        for m in nums:
            vals=pd.to_numeric(g[m],errors="coerce").dropna()
            row[m+"_mean"]=float(vals.mean()) if len(vals) else np.nan
            row[m+"_std"]=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

def build_models(random_state: int = 42, drop_sensitive: bool = True):
    def pre():
        return build_sklearn_preprocessor(drop_sensitive=drop_sensitive)

    models = {
        "logreg": Pipeline([
            ("pre", pre()),
            ("model", LogisticRegression(
                max_iter=3000,
                random_state=random_state,
            )),
        ]),

        "logreg_balanced": Pipeline([
            ("pre", pre()),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),

        "rf": Pipeline([
            ("pre", pre()),
            ("model", RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),

        "rf_balanced": Pipeline([
            ("pre", pre()),
            ("model", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),

        "svm_rbf": Pipeline([
            ("pre", pre()),
            ("model", SVC(
                kernel="rbf",
                probability=True,
                random_state=random_state,
            )),
        ]),

        "knn_5": Pipeline([
            ("pre", pre()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),

        "sgd_log_loss": Pipeline([
            ("pre", pre()),
            ("model", SGDClassifier(
                loss="log_loss",
                max_iter=2000,
                random_state=random_state,
            )),
        ]),

        "mlp": Pipeline([
            ("pre", pre()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1200,
                early_stopping=True,
                random_state=random_state,
            )),
        ]),

        "voting_soft": Pipeline([
            ("pre", pre()),
            ("model", VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(
                        max_iter=3000,
                        random_state=random_state,
                    )),
                    ("svc", SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=random_state,
                    )),
                    ("sgd", SGDClassifier(
                        loss="log_loss",
                        max_iter=2000,
                        random_state=random_state,
                    )),
                ],
                voting="soft",
                weights=[1, 3, 1],
                n_jobs=-1,
            )),
        ]),

        "stacking": Pipeline([
            ("pre", pre()),
            ("model", StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression(
                        max_iter=3000,
                        random_state=random_state,
                    )),
                    ("svc", SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=random_state,
                    )),
                    ("rf", RandomForestClassifier(
                        n_estimators=200,
                        random_state=random_state,
                        n_jobs=-1,
                    )),
                ],
                final_estimator=LogisticRegression(
                    max_iter=3000,
                    random_state=random_state,
                ),
                stack_method="predict_proba",
                passthrough=True,
                n_jobs=-1,
            )),
        ]),
    }

    try:
        import xgboost  # noqa: F401

        models["xgboost"] = Pipeline([
            ("pre", pre()),
            ("model", LabelEncodedXGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=-1,
            )),
        ])
    except Exception as exc:
        print(f"XGBoost skipped: {exc}")

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = Pipeline([
            ("pre", pre()),
            ("model", LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                random_state=random_state,
                verbose=-1,
            )),
        ])
    except Exception as exc:
        print(f"LightGBM skipped: {exc}")

    try:
        from interpret.glassbox import ExplainableBoostingClassifier

        models["ebm"] = Pipeline([
            ("pre", pre()),
            ("model", ExplainableBoostingClassifier(
                random_state=random_state,
            )),
        ])
    except Exception as exc:
        print(f"EBM skipped: {exc}")

    return models

def catboost_pred(Xtr,ytr,Xte,seed=42,drop_sensitive=True,balanced=False):
    Xtrc,Xtec,features,catidx=prepare_catboost_inputs(Xtr,Xte,drop_sensitive=drop_sensitive)
    cw=None
    if balanced:
        counts=ytr.value_counts().sort_index(); n=len(ytr); k=len(counts)
        cw=[float(n/(k*counts.loc[c])) for c in sorted(counts.index)]
    m=CatBoostClassifier(loss_function="MultiClass",eval_metric="TotalF1",iterations=300,learning_rate=.05,depth=6,random_seed=seed,verbose=False,class_weights=cw)
    m.fit(Pool(Xtrc,ytr,cat_features=catidx,feature_names=features))
    pool=Pool(Xtec,cat_features=catidx,feature_names=features)
    return m.predict(pool).flatten().astype(int),m.predict_proba(pool)
def load_xy(drop_sensitive=True):
    df=load_validated_or_raw_data()
    return split_features_and_target(df,drop_sensitive=drop_sensitive)
def cv_benchmark(n_splits=10,seed=42,drop_sensitive=True):
    out=SETTINGS.reports_dir/"advanced_experiments"/"cv_benchmark"; ensure_dir(out)
    X,y=load_xy(drop_sensitive); labels=sorted(y.unique())
    skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed); rows=[]; mset=build_models(seed,drop_sensitive)
    for fold,(tr,te) in enumerate(skf.split(X,y),1):
        Xtr,Xte=X.iloc[tr].copy(),X.iloc[te].copy(); ytr,yte=y.iloc[tr].copy(),y.iloc[te].copy()
        for name,model in mset.items():
            try:
                model.fit(Xtr,ytr); yp=model.predict(Xte); proba=model.predict_proba(Xte) if hasattr(model,"predict_proba") else None
                rows.append({"model":name,"fold":fold,**metric_pack(yte,yp,proba,labels)}); print(f"[{fold}] {name}")
            except Exception as e: rows.append({"model":name,"fold":fold,"error":str(e)}); print(f"[{fold}] {name} FAIL {e}")
        for name,bal in [("catboost",False),("catboost_balanced",True)]:
            try:
                yp,proba=catboost_pred(Xtr,ytr,Xte,seed,drop_sensitive,bal)
                rows.append({"model":name,"fold":fold,**metric_pack(yte,yp,proba,labels)}); print(f"[{fold}] {name}")
            except Exception as e: rows.append({"model":name,"fold":fold,"error":str(e)}); print(f"[{fold}] {name} FAIL {e}")
    df=pd.DataFrame(rows); sm=summarize(df,["model"])
    df.to_csv(out/"cv_fold_metrics.csv",index=False); sm.to_csv(out/"cv_summary_metrics.csv",index=False); save_json({"n_splits":n_splits,"drop_sensitive":drop_sensitive},out/"metadata.json")
    print(sm.sort_values("macro_f1_mean",ascending=False).head(15).to_string(index=False))

def imbalance_benchmark(n_splits=10,seed=42,drop_sensitive=True):
    out=SETTINGS.reports_dir/"advanced_experiments"/"imbalance"; ensure_dir(out)
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.combine import SMOTETomek
        from imblearn.under_sampling import RandomUnderSampler
        samplers={"none":None,"smote":SMOTE(random_state=seed),"adasyn":ADASYN(random_state=seed),"smote_tomek":SMOTETomek(random_state=seed),"random_under":RandomUnderSampler(random_state=seed)}
    except Exception as e:
        print("imbalanced-learn aktif değil:",e); ImbPipeline=None; samplers={"none":None}
    X,y=load_xy(drop_sensitive); labels=sorted(y.unique())
    bases={"lr":LogisticRegression(max_iter=3000,random_state=seed),"lr_bal":LogisticRegression(max_iter=3000,class_weight="balanced",random_state=seed),"rf":RandomForestClassifier(n_estimators=300,random_state=seed,n_jobs=-1),"rf_bal":RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=seed,n_jobs=-1)}
    rows=[]; skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    for fold,(tr,te) in enumerate(skf.split(X,y),1):
        Xtr,Xte=X.iloc[tr].copy(),X.iloc[te].copy(); ytr,yte=y.iloc[tr].copy(),y.iloc[te].copy()
        for sname,sampler in samplers.items():
            for mname,model in bases.items():
                try:
                    pipe=Pipeline([("pre",build_sklearn_preprocessor(drop_sensitive)),("model",model)]) if sampler is None else ImbPipeline([("pre",build_sklearn_preprocessor(drop_sensitive)),("sampler",sampler),("model",model)])
                    pipe.fit(Xtr,ytr); yp=pipe.predict(Xte); proba=pipe.predict_proba(Xte)
                    rows.append({"sampler":sname,"model":mname,"fold":fold,**metric_pack(yte,yp,proba,labels)})
                except Exception as e: rows.append({"sampler":sname,"model":mname,"fold":fold,"error":str(e)})
    df=pd.DataFrame(rows); sm=summarize(df,["sampler","model"])
    df.to_csv(out/"imbalance_fold_metrics.csv",index=False); sm.to_csv(out/"imbalance_summary_metrics.csv",index=False); print(sm.sort_values("macro_f1_mean",ascending=False).head(20).to_string(index=False))
def subset_pool(X,y,features,drop_sensitive):
    fg=get_feature_groups(drop_sensitive); cat=[c for c in fg["categorical"] if c in features]
    X=X[features].copy()
    for c in cat: X[c]=X[c].astype("string").fillna("__MISSING__")
    for c in [c for c in features if c not in cat]: X[c]=pd.to_numeric(X[c],errors="coerce")
    return Pool(X,label=y,cat_features=[features.index(c) for c in cat],feature_names=features)
def ablation(n_splits=10,seed=42,drop_sensitive=True):
    out=SETTINGS.reports_dir/"advanced_experiments"/"ablation"; ensure_dir(out)
    X,y=load_xy(drop_sensitive)
    sets={"all_features":X.columns.tolist(),"satisfaction_only":[c for c in ["EmpEnvironmentSatisfaction","EmpJobSatisfaction","EmpRelationshipSatisfaction","EmpWorkLifeBalance","EmpJobInvolvement"] if c in X.columns],"career_tenure_only":[c for c in ["TotalWorkExperienceInYears","ExperienceYearsAtThisCompany","ExperienceYearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager","NumCompaniesWorked","TrainingTimesLastYear"] if c in X.columns],"compensation_only":[c for c in ["EmpHourlyRate","EmpLastSalaryHikePercent"] if c in X.columns],"role_org_only":[c for c in ["EmpDepartment","EmpJobRole","EmpJobLevel","BusinessTravelFrequency","OverTime","Attrition"] if c in X.columns]}
    sp=SETTINGS.reports_dir/"xai"/"global_shap_importance.csv"
    if sp.exists():
        ranked=pd.read_csv(sp)["feature"].tolist()
        for k in [5,10,15,20]: sets[f"shap_top_{k}"]=[c for c in ranked[:k] if c in X.columns]
    rows=[]; labels=sorted(y.unique()); skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    for name,features in sets.items():
        if not features: continue
        for fold,(tr,te) in enumerate(skf.split(X[features],y),1):
            Xtr,Xte=X.iloc[tr].copy(),X.iloc[te].copy(); ytr,yte=y.iloc[tr].copy(),y.iloc[te].copy()
            model=CatBoostClassifier(loss_function="MultiClass",iterations=300,learning_rate=.05,depth=6,random_seed=seed,verbose=False)
            model.fit(subset_pool(Xtr,ytr,features,drop_sensitive))
            pool=subset_pool(Xte,None,features,drop_sensitive)
            rows.append({"feature_set":name,"n_features":len(features),"fold":fold,**metric_pack(yte,model.predict(pool).flatten().astype(int),model.predict_proba(pool),labels)})
    df=pd.DataFrame(rows); sm=summarize(df,["feature_set","n_features"])
    df.to_csv(out/"ablation_fold_metrics.csv",index=False); sm.to_csv(out/"ablation_summary_metrics.csv",index=False); save_json({"feature_sets":sets},out/"metadata.json"); print(sm.sort_values("macro_f1_mean",ascending=False).to_string(index=False))
def holm(pv):
    p=pd.Series(pv,dtype=float); out=pd.Series(index=p.index,dtype=float); run=0; order=p.sort_values().index; m=len(order)
    for r,i in enumerate(order,1): run=max(run,min((m-r+1)*p.loc[i],1.0)); out.loc[i]=run
    return out
def stats(metric="macro_f1"):
    out=SETTINGS.reports_dir/"advanced_experiments"/"statistical_tests"; ensure_dir(out)
    path=SETTINGS.reports_dir/"advanced_experiments"/"cv_benchmark"/"cv_fold_metrics.csv"
    df=pd.read_csv(path)[["model","fold",metric]].dropna(); piv=df.pivot_table(index="fold",columns="model",values=metric,aggfunc="mean").dropna(axis=1)
    rng=np.random.default_rng(42); ci=[]
    for m in piv.columns:
        vals=piv[m].values; boots=[rng.choice(vals,size=len(vals),replace=True).mean() for _ in range(3000)]
        ci.append({"model":m,"mean":float(vals.mean()),"std":float(vals.std(ddof=1)),"ci_low":float(np.quantile(boots,.025)),"ci_high":float(np.quantile(boots,.975))})
    pairs=[]
    try:
        from scipy.stats import wilcoxon
        for a,b in combinations(piv.columns,2):
            st,p=wilcoxon(piv[a].values,piv[b].values,zero_method="wilcox")
            pairs.append({"model_a":a,"model_b":b,"mean_diff":float(piv[a].mean()-piv[b].mean()),"p_value":float(p)})
        pdf=pd.DataFrame(pairs); pdf["p_holm"]=holm(pdf["p_value"]); pdf["significant_holm"]=pdf["p_holm"]<.05
    except Exception as e: pdf=pd.DataFrame({"error":[str(e)]})
    cdf=pd.DataFrame(ci).sort_values("mean",ascending=False); cdf.to_csv(out/f"{metric}_bootstrap_ci.csv",index=False); pdf.to_csv(out/f"{metric}_pairwise_tests.csv",index=False); print(cdf.to_string(index=False))
def complexity(seed=42,repeats=20):
    out=SETTINGS.reports_dir/"advanced_experiments"/"complexity"; ensure_dir(out)
    prep=run_preprocessing(drop_sensitive=True,random_state=seed); Xtr,Xte,ytr=prep["X_train_raw"],prep["X_test_raw"],prep["y_train"]; rows=[]
    for name,model in build_models(seed,True).items():
        try:
            t=time.perf_counter(); model.fit(Xtr,ytr); train=time.perf_counter()-t
            t=time.perf_counter(); [model.predict(Xte) for _ in range(repeats)]; pred=(time.perf_counter()-t)/repeats
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/"m.joblib"; joblib.dump(model,p); size=p.stat().st_size/1024/1024
            rows.append({"model":name,"train_total_s":train,"predict_total_s":pred,"predict_per_sample_ms":pred/len(Xte)*1000,"throughput_samples_s":len(Xte)/pred,"model_size_mb":size})
        except Exception as e: rows.append({"model":name,"error":str(e)})
    df=pd.DataFrame(rows); df.to_csv(out/"complexity_report.csv",index=False); print(df.to_string(index=False))
def formal_fairness(attrs="Gender,MaritalStatus,EmpDepartment,EducationBackground,BusinessTravelFrequency"):
    out=SETTINGS.reports_dir/"fairness"/"formal"; ensure_dir(out)
    mp=SETTINGS.artifacts_dir/"catboost"/"catboost_model.cbm"; sp=SETTINGS.artifacts_dir/"catboost"/"run_summary.json"
    model=CatBoostClassifier(); model.load_model(str(mp)); summ=json.loads(sp.read_text(encoding="utf-8"))
    prep=run_preprocessing(test_size=summ.get("test_size",.2),random_state=summ.get("random_state",42),drop_sensitive=summ.get("drop_sensitive",False))
    Xmod,y=prep["X_test_raw"],prep["y_test"].astype(int)
    Xcb,_,features,catidx=prepare_catboost_inputs(Xmod,Xmod.copy(),drop_sensitive=summ.get("drop_sensitive",False)); pool=Pool(Xcb,y,cat_features=catidx,feature_names=features)
    pred=pd.Series(model.predict(pool).flatten().astype(int),index=Xmod.index); proba=pd.DataFrame(model.predict_proba(pool),columns=[int(c) for c in model.classes_],index=Xmod.index)
    full=load_validated_or_raw_data(); Xfull,yfull=split_features_and_target(full,drop_sensitive=False); _,Xfullte,_,_=train_test_split(Xfull,yfull,test_size=summ.get("test_size",.2),random_state=summ.get("random_state",42),stratify=yfull)
    rows=[]; gaps=[]; labels=sorted([int(c) for c in model.classes_])
    for attr in [a.strip() for a in attrs.split(",") if a.strip()]:
        if attr not in Xfullte.columns: continue
        groups=Xfullte[attr].astype("string").fillna("__MISSING__")
        for cls in labels:
            tmp=[]
            for gv in sorted(groups.unique()):
                mask=groups==gv; yt=(y.loc[mask]==cls).astype(int); yp=(pred.loc[mask]==cls).astype(int)
                tp=int(((yt==1)&(yp==1)).sum()); fp=int(((yt==0)&(yp==1)).sum()); tn=int(((yt==0)&(yp==0)).sum()); fn=int(((yt==1)&(yp==0)).sum())
                row={"attribute":attr,"group_value":str(gv),"class_label":cls,"n_samples":int(mask.sum()),"positive_prediction_rate":(tp+fp)/max(len(yt),1),"tpr_equal_opportunity":tp/max(tp+fn,1),"fpr":fp/max(fp+tn,1),"precision":tp/max(tp+fp,1),"mean_predicted_probability":float(proba.loc[mask,cls].mean())}
                rows.append(row); tmp.append(row)
            t=pd.DataFrame(tmp)
            for m in ["positive_prediction_rate","tpr_equal_opportunity","fpr","precision","mean_predicted_probability"]: gaps.append({"attribute":attr,"class_label":cls,"metric":m,"max_gap":float(t[m].max()-t[m].min())})
    pd.DataFrame(rows).to_csv(out/"formal_fairness_group_metrics.csv",index=False); pd.DataFrame(gaps).to_csv(out/"formal_fairness_disparity_summary.csv",index=False); print(pd.DataFrame(gaps).head(40).to_string(index=False))
def cf_eval(max_samples=100,max_features_changed=3,top_k=5):
    out=SETTINGS.reports_dir/"advanced_experiments"/"counterfactual_evaluation"; ensure_dir(out)
    mp=SETTINGS.artifacts_dir/"catboost"/"catboost_model.cbm"; sp=SETTINGS.artifacts_dir/"catboost"/"run_summary.json"
    model=CatBoostClassifier(); model.load_model(str(mp)); summ=json.loads(sp.read_text(encoding="utf-8"))
    prep=run_preprocessing(test_size=summ.get("test_size",.2),random_state=summ.get("random_state",42),drop_sensitive=summ.get("drop_sensitive",False))
    Xtr,Xte,ytr,yte=prep["X_train_raw"],prep["X_test_raw"],prep["y_train"].astype(int),prep["y_test"].astype(int); drop=summ.get("drop_sensitive",False); classes=sorted([int(c) for c in model.classes_])
    rows=[]; allc=[]
    for idx in Xte.index:
        pr=predict_single_sample(model,Xte.loc[[idx]],drop); pc=int(pr["predicted_class"])
        if pc>=max(classes): continue
        desired=min([c for c in classes if c>pc]); base=float(pr["probabilities"].get(desired,0))
        cands=generate_counterfactual_candidates(model,Xte.loc[[idx]],pc,desired,Xtr,ytr,drop,[f for f in DEFAULT_ACTIONABLE_FEATURES if f in Xte.columns],max_features_changed,top_k)
        best=cands[0] if cands else None
        rows.append({"sample_index":int(idx),"true_class":int(yte.loc[idx]),"predicted_class":pc,"desired_class":desired,"base_desired_probability":base,"valid_counterfactual_found":bool(cands),"best_desired_probability":float(best["desired_class_probability"]) if best else np.nan,"probability_gain":float(best["desired_class_probability"]-base) if best else np.nan,"best_cost":float(best["cost"]) if best else np.nan,"best_num_changes":int(best["num_changes"]) if best else np.nan})
        for r,c in enumerate(cands,1): allc.append({"sample_index":int(idx),"rank":r,"desired_class":desired,"prob":float(c["desired_class_probability"]),"cost":float(c["cost"]),"num_changes":int(c["num_changes"]),"changed_features":", ".join([x["feature"] for x in c["changes"]])})
        if len(rows)>=max_samples: break
    df=pd.DataFrame(rows); ac=pd.DataFrame(allc); df.to_csv(out/"counterfactual_evaluation_summary_by_sample.csv",index=False); ac.to_csv(out/"counterfactual_candidates_all.csv",index=False)
    meta={"n_samples":len(df),"validity_rate":float(df["valid_counterfactual_found"].mean()) if len(df) else np.nan,"mean_probability_gain":float(df["probability_gain"].mean()) if len(df) else np.nan,"mean_best_cost":float(df["best_cost"].mean()) if len(df) else np.nan,"mean_best_num_changes":float(df["best_num_changes"].mean()) if len(df) else np.nan}
    save_json(meta,out/"counterfactual_evaluation_metadata.json"); print(meta)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--task",choices=["cv","imbalance","ablation","stats","complexity","fairness","cf","all"],default="all"); p.add_argument("--n-splits",type=int,default=10); p.add_argument("--random-state",type=int,default=42); p.add_argument("--metric",default="macro_f1"); p.add_argument("--max-samples",type=int,default=100); p.add_argument("--drop-sensitive",action="store_true"); a=p.parse_args()
    drop=True
    if a.task in ["cv","all"]: cv_benchmark(a.n_splits,a.random_state,drop)
    if a.task in ["imbalance","all"]: imbalance_benchmark(a.n_splits,a.random_state,drop)
    if a.task in ["ablation","all"]: ablation(a.n_splits,a.random_state,drop)
    if a.task in ["stats","all"]: stats(a.metric)
    if a.task in ["complexity","all"]: complexity(a.random_state)
    if a.task in ["fairness","all"]: formal_fairness()
    if a.task in ["cf","all"]: cf_eval(a.max_samples)
if __name__=="__main__": main()
