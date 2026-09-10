"""Repeat macro-F1-versus-QWK selection on the five frozen Phase 1C splits."""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.canonical_loader import sha256_file
from src.experiments.manuscript_model_benchmark import select_candidate_index
from src.experiments.repeated_nested_cv_v3 import (
    TUNED_MODEL_NAMES, _fit_or_fail, _pipeline, _prepare_inputs,
)
from src.governance.manuscript_contract import source_tree_hash
from src.governance.offline_runtime import enforce_offline_runtime
from src.governance.repeated_nested_cv_run_validator_v3 import (
    _rebuild_fold_artifacts, validate_repeated_nested_cv_run_v3,
)
from src.models.canonical_models import aligned_predict_proba
from src.models.ordinal_evaluation_v3 import ordinal_evaluation_bundle_v3
from src.utils.config_loader import PROJECT_ROOT

DEFAULT_CONTRACT = Path("configs/eswa_repeated_selection_objective_v1.json")
DEFAULT_OUTPUT_ROOT = Path("reports/major_revision_round2_runs")
LABELS = (2, 3, 4)

class ESWARepeatedSelectionError(RuntimeError): pass
def _require(value: bool, message: str) -> None:
    if not value: raise ESWARepeatedSelectionError(message)
def _load(path: Path) -> dict[str, Any]:
    value=json.loads(path.read_text(encoding="utf-8")); _require(isinstance(value,dict),f"Invalid JSON object: {path}"); return value
def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def _git_identity() -> dict[str,str]:
    def git(*args: str) -> str:
        return subprocess.run(["git",*args],cwd=PROJECT_ROOT,check=True,capture_output=True,text=True).stdout.strip()
    status=git("status","--porcelain","--untracked-files=all"); _require(not status,f"Scientific run requires clean worktree: {status.splitlines()[:5]}")
    return {"commit":git("rev-parse","HEAD"),"branch":git("rev-parse","--abbrev-ref","HEAD")}

def validate_contract(path: Path=DEFAULT_CONTRACT) -> tuple[dict[str,Any],dict[str,str]]:
    c=_load(path); _require(c.get("contract_id")=="eswa_repeated_selection_objective_v1","Contract ID drifted")
    _require(tuple(c.get("ordered_labels",()))==LABELS,"Class order drifted")
    _require(tuple(c.get("models",()))==tuple(TUNED_MODEL_NAMES),"Model registry drifted")
    repeated=Path(c["repeated_design_contract"]["path"]); _require(sha256_file(repeated)==c["repeated_design_contract"]["sha256"],"Repeated design contract hash drifted")
    source=Path(c["phase1c_source_run"]["directory"])
    mapping={"candidate_search_results.csv":"candidate_search_results_sha256","fold_contracts.json":"fold_contracts_sha256","oof_predictions.csv":"macro_f1_oof_predictions_sha256","selected_hyperparameters.csv":"selected_hyperparameters_sha256","stage_metadata.json":"stage_metadata_sha256"}
    hashes={}
    for name,key in mapping.items():
        actual=sha256_file(source/name); _require(actual==c["phase1c_source_run"][key],f"Phase 1C source hash drifted: {name}"); hashes[name]=actual
    regimes=c["selection_regimes"]
    _require(regimes["macro_f1"]["primary_metric"]=="macro_f1" and regimes["macro_f1"]["tie_break_metric"]=="quadratic_weighted_kappa","Macro-F1 regime drifted")
    _require(regimes["qwk"]["primary_metric"]=="quadratic_weighted_kappa" and regimes["qwk"]["tie_break_metric"]=="macro_f1","QWK regime drifted")
    _require(float(regimes["macro_f1"]["practical_tie_tolerance"])==float(regimes["qwk"]["practical_tie_tolerance"])==0.001,"Tie tolerance drifted")
    return c,hashes

def build_schedule(candidates: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    required={"repetition","outer_fold","model","candidate_index","parameters_json","inner_macro_f1_mean","inner_qwk_mean","outer_test_used_for_selection","selected_by_protocol"}
    _require(required.issubset(candidates.columns),"Candidate source schema incomplete")
    candidates=candidates[candidates["model"].isin(TUNED_MODEL_NAMES)].copy()
    candidates["outer_test_used_for_selection"] = candidates["outer_test_used_for_selection"].astype(str).str.lower().eq("true")
    _require(not candidates["outer_test_used_for_selection"].any(),"Outer test entered source candidate selection")
    rows=[]; changes=[]
    for (rep,fold,model), scoped in candidates.groupby(["repetition","outer_fold","model"],sort=True):
        scoped=scoped.sort_values("candidate_index").reset_index(drop=True); selected={}
        for regime,primary,secondary in (("macro_f1","inner_macro_f1_mean","inner_qwk_mean"),("qwk","inner_qwk_mean","inner_macro_f1_mean")):
            pos=select_candidate_index(scoped[primary].astype(float).tolist(),scoped[secondary].astype(float).tolist(),practical_tie_tolerance=0.001,better_direction="higher")
            row=scoped.iloc[pos]; selected[regime]=int(row["candidate_index"])
            rows.append({"repetition":int(rep),"outer_fold":int(fold),"model":str(model),"selection_objective":regime,"selected_candidate_index":int(row["candidate_index"]),"selected_candidate_parameters_json":str(row["parameters_json"]),"selected_primary_mean":float(row[primary]),"selected_tie_break_mean":float(row[secondary]),"practical_tie_tolerance":0.001,"outer_test_used_for_selection":False})
        historical=scoped.loc[scoped["selected_by_protocol"].astype(str).str.lower().eq("true"),"candidate_index"].astype(int).tolist()
        _require(historical==[selected["macro_f1"]],f"Historical selection mismatch: rep={rep} fold={fold} model={model}")
        changes.append({"repetition":int(rep),"outer_fold":int(fold),"model":str(model),"macro_f1_candidate_index":selected["macro_f1"],"qwk_candidate_index":selected["qwk"],"selected_candidate_changed":selected["macro_f1"]!=selected["qwk"]})
    schedule=pd.DataFrame(rows).sort_values(["repetition","selection_objective","model","outer_fold"]).reset_index(drop=True)
    change=pd.DataFrame(changes).sort_values(["repetition","model","outer_fold"]).reset_index(drop=True)
    _require(len(schedule)==300 and len(change)==150,"Selection schedule coverage drifted")
    return schedule,change

def _folds(source: Path, macro_oof: pd.DataFrame):
    result={}
    records=json.loads((source/"fold_contracts.json").read_text(encoding="utf-8"))
    _require(isinstance(records,list) and len(records)==5,"Phase 1C fold-contract inventory drifted")
    for record in records:
        rep=int(record["repetition"]); scoped=macro_oof[macro_oof["repetition"]==rep]
        artifact=_rebuild_fold_artifacts(record,scoped)
        seed={"repetition":rep,"outer_seed":int(record["outer_seed"]),"inner_seed":int(record["inner_seed"]),"model_seed":int(record["model_seed"])}
        result[rep]=(artifact,seed)
    return result

def _validate_oof(oof: pd.DataFrame, folds: Mapping[int,Any], samples: int) -> None:
    _require(len(oof)==5*2*6*samples,"OOF row count drifted")
    for rep,(artifact,_) in folds.items():
        reference=artifact.outer_assignments.set_index("sample_index")[["outer_fold","y_true"]].astype(int).sort_index()
        for (regime,model), rows in oof[oof["repetition"]==rep].groupby(["selection_objective","model"]):
            rows=rows.set_index("sample_index").sort_index(); _require(len(rows)==samples and not rows.index.duplicated().any(),f"OOF coverage drifted: {rep}/{regime}/{model}")
            _require(rows[["outer_fold","y_true"]].astype(int).equals(reference),f"Fold identity drifted: {rep}/{regime}/{model}")
            p=rows[["prob_class_2","prob_class_3","prob_class_4"]].to_numpy(float)
            _require(np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,rtol=0,atol=1e-9),"Probability contract failed")
            _require(np.array_equal(np.asarray(LABELS)[np.argmax(p,axis=1)],rows["y_pred"].astype(int)),"Argmax/class-order contract failed")

def _summaries(oof: pd.DataFrame, schedule: pd.DataFrame, metric_names: Sequence[str]) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    result=[]; per_class=[]
    for (rep,regime,model), rows in oof.groupby(["repetition","selection_objective","model"],sort=True):
        rows=rows.sort_values("sample_index"); bundle=ordinal_evaluation_bundle_v3(rows["y_true"].astype(int),rows["y_pred"].astype(int),rows[["prob_class_2","prob_class_3","prob_class_4"]].to_numpy(float),labels=LABELS,dataset_key="inx_primary",model_name=str(model))
        record={"repetition":int(rep),"model":model,"selection_objective":regime}
        record.update({name:float(bundle["aggregate_metrics"][name]) for name in metric_names})
        for item in bundle["per_class_metrics"]:
            label=int(item["class_label"]); record[f"rating_{label}_recall"]=float(item["recall"]); record[f"rating_{label}_f1"]=float(item["f1"]); per_class.append({"repetition":int(rep),"selection_objective":regime,**item})
        scoped=schedule[(schedule["repetition"]==rep)&(schedule["model"]==model)&(schedule["selection_objective"]==regime)].sort_values("outer_fold")
        record["selected_candidate_ids_json"]=json.dumps(scoped["selected_candidate_index"].astype(int).tolist(),separators=(",",":")); result.append(record)
    results=pd.DataFrame(result).sort_values(["repetition","model","selection_objective"]).reset_index(drop=True)
    comparisons=[]; directions={"macro_f1":"higher","balanced_accuracy":"higher","quadratic_weighted_kappa":"higher","ordinal_mae":"lower","ranked_probability_score":"lower","nll_log_loss":"lower","multiclass_brier":"lower"}
    for rep, rows in results.groupby("repetition",sort=True):
        changed=[]; leaders=[]; orderings=[]
        for metric,direction in directions.items():
            orders={}
            for regime in ("macro_f1","qwk"):
                scoped=rows[rows["selection_objective"]==regime].sort_values([metric,"model"],ascending=[direction=="lower",True],kind="mergesort"); orders[regime]=scoped["model"].tolist()
            if orders["macro_f1"][0]!=orders["qwk"][0]: leaders.append(metric)
            if orders["macro_f1"]!=orders["qwk"]: orderings.append(metric)
        rec={"repetition":int(rep),"candidate_changes":int(0),"metric_leader_changes":len(leaders),"full_ordering_changes":len(orderings),"leader_changed_metrics_json":json.dumps(leaders,separators=(",",":")),"ordering_changed_metrics_json":json.dumps(orderings,separators=(",",":"))}
        for model,prefix in (("xgboost","nominal_xgb"),("cumulative_threshold_xgboost","cumulative_xgb"),("random_forest","rf")):
            p=rows[rows["model"]==model].set_index("selection_objective")
            rec[f"{prefix}_delta_qwk"]=float(p.loc["qwk","quadratic_weighted_kappa"]-p.loc["macro_f1","quadratic_weighted_kappa"])
            if model=="xgboost": rec[f"{prefix}_delta_mae"]=float(p.loc["qwk","ordinal_mae"]-p.loc["macro_f1","ordinal_mae"])
            rec[f"{prefix}_delta_rating4_recall"]=float(p.loc["qwk","rating_4_recall"]-p.loc["macro_f1","rating_4_recall"])
        comparisons.append(rec)
    return results,pd.DataFrame(per_class),pd.DataFrame(comparisons)

def run(contract_path: Path, output_dir: Path, run_id: str) -> dict[str,Any]:
    with enforce_offline_runtime() as offline:
        identity=_git_identity(); c,source_hashes=validate_contract(contract_path); repeated_path=Path(c["repeated_design_contract"]["path"])
        repeated,receipt,canonical,features,exclusions,target,nominal,ordinal=_prepare_inputs(repeated_path)
        source=Path(c["phase1c_source_run"]["directory"]); phase1c_validation=validate_repeated_nested_cv_run_v3(source); candidates=pd.read_csv(source/"candidate_search_results.csv"); macro=pd.read_csv(source/"oof_predictions.csv")
        schedule,changes=build_schedule(candidates); contract_hash=sha256_file(contract_path)
        implementation=[Path(__file__).relative_to(PROJECT_ROOT),contract_path]
        scientific={"git_identity":identity,"source_tree_hash":source_tree_hash(PROJECT_ROOT),"contract_sha256":contract_hash,"source_hashes":source_hashes,"implementation_hashes":{p.as_posix():sha256_file(p) for p in implementation},"dataset_sha256":canonical.receipt["actual_sha256"]}; scientific_hash=_digest(scientific)
        folds=_folds(source,macro)
        definitions={**{name:nominal["models"][name] for name in TUNED_MODEL_NAMES if name in nominal["models"]},**{name:ordinal["ordinal_models"][name] for name in TUNED_MODEL_NAMES if name in ordinal["ordinal_models"]}}
        macro=macro[macro["model"].isin(TUNED_MODEL_NAMES)].copy(); macro["selection_objective"]="macro_f1"; macro["evidence_source"]="hash_bound_phase1c_macro_f1_oof_reuse"
        macro=macro.drop(columns=["selected_candidate_index"]).merge(schedule[schedule["selection_objective"]=="macro_f1"][["repetition","outer_fold","model","selected_candidate_index"]],on=["repetition","outer_fold","model"],validate="many_to_one")
        qwk=[]
        for rep,(artifact,seed) in folds.items():
            outer=artifact.outer_assignments
            for model in TUNED_MODEL_NAMES:
                for fold in range(1,6):
                    chosen=schedule[(schedule["repetition"]==rep)&(schedule["outer_fold"]==fold)&(schedule["model"]==model)&(schedule["selection_objective"]=="qwk")].iloc[0]
                    test_ids=outer.loc[outer["outer_fold"].astype(int)==fold,"sample_index"].astype(int).tolist(); train_ids=outer.loc[outer["outer_fold"].astype(int)!=fold,"sample_index"].astype(int).tolist()
                    candidate=json.loads(chosen["selected_candidate_parameters_json"]); pipe=_pipeline(model,features.loc[train_ids],fixed_parameters=definitions[model]["fixed_params"],candidate_parameters=candidate,random_state=int(seed["model_seed"]),forbidden_features=exclusions)
                    _fit_or_fail(pipe,features.loc[train_ids],target.loc[train_ids],context=f"ESWA repeated QWK rep={rep} model={model} fold={fold}")
                    probability=aligned_predict_proba(pipe,features.loc[test_ids],labels=LABELS); prediction=np.asarray(LABELS)[np.argmax(probability,axis=1)]
                    for pos,sample in enumerate(test_ids): qwk.append({"run_id":run_id,"repetition":rep,"outer_seed":int(seed["outer_seed"]),"inner_seed":int(seed["inner_seed"]),"model_seed":int(seed["model_seed"]),"fold_contract_hash":artifact.contract["fold_contract_hash"],"evidence_source":"eswa_repeated_qwk_selected_outer_refit","model":model,"sample_index":sample,"outer_fold":fold,"y_true":int(target.loc[sample]),"y_pred":int(prediction[pos]),"selected_candidate_index":int(chosen["selected_candidate_index"]),"prob_class_2":float(probability[pos,0]),"prob_class_3":float(probability[pos,1]),"prob_class_4":float(probability[pos,2]),"selection_objective":"qwk"})
        combined=pd.concat([macro,pd.DataFrame(qwk)],ignore_index=True).sort_values(["repetition","selection_objective","model","sample_index"]).reset_index(drop=True); _validate_oof(combined,folds,len(target))
        results,per_class,comparisons=_summaries(combined,schedule,c["report_metrics"])
        comparisons["candidate_changes"]=comparisons["repetition"].map(changes.groupby("repetition")["selected_candidate_changed"].sum().astype(int))
        _require(not output_dir.exists(),f"Output exists: {output_dir}"); output_dir.parent.mkdir(parents=True,exist_ok=True); staging=output_dir.parent/f".{output_dir.name}.staging.{uuid.uuid4().hex}"; staging.mkdir()
        frames={"candidate_evidence.csv":candidates,"selection_schedule.csv":schedule,"selected_candidate_changes.csv":changes,"oof_predictions.csv":combined,"repeated_selection_results.csv":results,"per_class_metrics.csv":per_class,"repetition_comparisons.csv":comparisons}
        for name,frame in frames.items(): frame.to_csv(staging/name,index=False)
        metadata={"schema_version":1,"stage":"eswa_repeated_selection_objective_v1","status":"complete","run_id":run_id,"created_at_utc":datetime.now(timezone.utc).isoformat(),"git_identity":identity,"scientific_input_sha256":scientific_hash,"scientific_inputs":scientific,"phase1c_validation":phase1c_validation,"repetitions":5,"outer_folds_per_repetition":5,"inner_folds":5,"models":list(TUNED_MODEL_NAMES),"selection_objectives":["macro_f1","qwk"],"historical_phase1c_model_fit_count":5725,"qwk_outer_model_fit_count":150,"total_lineage_model_fit_count":5875,"new_inner_model_fit_count":0,"macro_f1_oof_reused":True,"folds_reconstructed_from_phase1c_contracts":True,"outer_test_used_for_selection":False,"seed_or_repetition_selected_from_results":False,"employee_level_outputs_publication_authorized":False,"runtime_policy":offline.receipt(),"network_calls":0,"paid_api_calls":0,"output_hashes":{name:sha256_file(staging/name) for name in frames}}
        (staging/"stage_metadata.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n",encoding="utf-8"); _require(_git_identity()==identity,"Git identity changed during run"); _require(source_tree_hash(PROJECT_ROOT)==scientific["source_tree_hash"],"Source tree changed during run"); os.replace(staging,output_dir)
        return {"status":"complete","run_id":run_id,"output_dir":output_dir.as_posix(),"qwk_outer_model_fit_count":150,"oof_rows":len(combined),"scientific_input_sha256":scientific_hash,"network_calls":0,"paid_api_calls":0}

def main(argv: Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--contract",type=Path,default=DEFAULT_CONTRACT); parser.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT_ROOT); parser.add_argument("--run-id"); parser.add_argument("--preflight-only",action="store_true"); args=parser.parse_args(argv)
    if args.preflight_only:
        c,h=validate_contract(args.contract); print(json.dumps({"status":"passed","models":c["models"],"repetitions":5,"outer_folds":5,"new_model_fits":0,"source_hashes":h},indent=2)); return 0
    _require(bool(args.run_id),"--run-id required"); print(json.dumps(run(args.contract,args.output_root/args.run_id/"repeated_selection_objective",args.run_id),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
