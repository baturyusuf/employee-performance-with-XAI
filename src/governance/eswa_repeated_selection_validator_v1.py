"""Independently validate the ESWA repeated selection-objective run."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from src.data.canonical_loader import sha256_file
from src.experiments.eswa_repeated_selection_objective_v1 import DEFAULT_CONTRACT, LABELS, TUNED_MODEL_NAMES, validate_contract
from src.governance.repeated_nested_cv_run_validator_v3 import _rebuild_fold_artifacts, validate_repeated_nested_cv_run_v3
from src.models.ordinal_evaluation_v3 import ordinal_evaluation_bundle_v3

class ESWARepeatedSelectionValidationError(RuntimeError): pass
def _require(value: bool,message: str)->None:
    if not value: raise ESWARepeatedSelectionValidationError(message)
def _frame(path: Path)->pd.DataFrame: return pd.read_csv(path,low_memory=False)
def _independent_select(scoped: pd.DataFrame,primary: str,secondary: str)->int:
    ordered=scoped.sort_values("candidate_index").reset_index(drop=True); p=ordered[primary].to_numpy(float); s=ordered[secondary].to_numpy(float)
    _require(np.isfinite(p).all() and np.isfinite(s).all(),"Candidate means are non-finite")
    best=float(p.max()); eligible=np.flatnonzero(best-p <= 0.001 + np.finfo(float).eps*max(1.0,abs(best))*8)
    second=float(s[eligible].max()); winners=eligible[s[eligible]==second]
    return int(ordered.iloc[int(winners[0])]["candidate_index"])

def validate_run(run_dir: Path,contract_path: Path=DEFAULT_CONTRACT)->dict[str,Any]:
    root=Path(run_dir); c,_=validate_contract(contract_path); source=Path(c["phase1c_source_run"]["directory"])
    phase1c=validate_repeated_nested_cv_run_v3(source)
    metadata=json.loads((root/"stage_metadata.json").read_text(encoding="utf-8")); _require(metadata["stage"]=="eswa_repeated_selection_objective_v1" and metadata["status"]=="complete","Stage metadata drifted")
    _require(metadata["qwk_outer_model_fit_count"]==150 and metadata["new_inner_model_fit_count"]==0,"Fit-count contract drifted")
    _require(metadata["outer_test_used_for_selection"] is False and metadata["network_calls"]==metadata["paid_api_calls"]==0,"Safety/selection boundary drifted")
    for name,digest in metadata["output_hashes"].items(): _require(sha256_file(root/name)==digest,f"Output hash drifted: {name}")
    candidates=_frame(root/"candidate_evidence.csv"); schedule=_frame(root/"selection_schedule.csv"); changes=_frame(root/"selected_candidate_changes.csv"); oof=_frame(root/"oof_predictions.csv"); results=_frame(root/"repeated_selection_results.csv"); per_class=_frame(root/"per_class_metrics.csv"); comparisons=_frame(root/"repetition_comparisons.csv")
    _require(len(candidates)==1100 and len(schedule)==300 and len(changes)==150,"Candidate/schedule grids drifted")
    _require(not candidates["outer_test_used_for_selection"].astype(bool).any() and not schedule["outer_test_used_for_selection"].astype(bool).any(),"Outer test entered selection")
    expected=[]
    for (rep,fold,model),scoped in candidates.groupby(["repetition","outer_fold","model"],sort=True):
        for regime,primary,secondary in (("macro_f1","inner_macro_f1_mean","inner_qwk_mean"),("qwk","inner_qwk_mean","inner_macro_f1_mean")):
            expected.append((int(rep),int(fold),str(model),regime,_independent_select(scoped,primary,secondary)))
    observed=[tuple(x) for x in schedule[["repetition","outer_fold","model","selection_objective","selected_candidate_index"]].itertuples(index=False,name=None)]
    _require(sorted(expected)==sorted(observed),"Independent schedule recomputation failed")
    source_oof=_frame(source/"oof_predictions.csv"); source_tuned=source_oof[source_oof["model"].isin(TUNED_MODEL_NAMES)].copy()
    macro=oof[oof["selection_objective"]=="macro_f1"].copy(); compare_cols=["repetition","outer_seed","inner_seed","model_seed","fold_contract_hash","model","sample_index","outer_fold","y_true","y_pred","prob_class_2","prob_class_3","prob_class_4"]
    a=source_tuned[compare_cols].sort_values(["repetition","model","sample_index"]).reset_index(drop=True); b=macro[compare_cols].sort_values(["repetition","model","sample_index"]).reset_index(drop=True)
    identity_columns=[name for name in compare_cols if not name.startswith("prob_class_")]
    _require(a[identity_columns].astype(str).equals(b[identity_columns].astype(str)),"Macro-F1 OOF identity is not an exact Phase 1C reuse")
    macro_reuse_max=float(np.max(np.abs(a[["prob_class_2","prob_class_3","prob_class_4"]].to_numpy(float)-b[["prob_class_2","prob_class_3","prob_class_4"]].to_numpy(float))))
    _require(macro_reuse_max<=1e-15,"Macro-F1 OOF serialization exceeded the fixed machine tolerance")
    records=json.loads((source/"fold_contracts.json").read_text(encoding="utf-8")); fold_map={}
    for record in records:
        rep=int(record["repetition"]); fold_map[rep]=_rebuild_fold_artifacts(record,source_oof[source_oof["repetition"]==rep])
    _require(len(oof)==72000,"Dual-regime OOF row count drifted")
    for rep,artifact in fold_map.items():
        identity=artifact.outer_assignments.set_index("sample_index")[["outer_fold","y_true"]].astype(int).sort_index()
        for (regime,model),rows in oof[oof["repetition"]==rep].groupby(["selection_objective","model"]):
            rows=rows.set_index("sample_index").sort_index(); _require(len(rows)==1200 and not rows.index.duplicated().any(),"Exactly-once OOF failed")
            _require(rows[["outer_fold","y_true"]].astype(int).equals(identity),"Persisted fold identity mismatch")
            p=rows[["prob_class_2","prob_class_3","prob_class_4"]].to_numpy(float); _require(np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,rtol=0,atol=1e-9),"Probability validation failed")
            _require(np.array_equal(np.asarray(LABELS)[np.argmax(p,axis=1)],rows["y_pred"].to_numpy(int)),"Class order/argmax failed")
            selected=schedule[(schedule["repetition"]==rep)&(schedule["model"]==model)&(schedule["selection_objective"]==regime)].set_index("outer_fold")["selected_candidate_index"].astype(int)
            _require(rows["selected_candidate_index"].astype(int).equals(rows["outer_fold"].map(selected).astype(int)),"Candidate lineage failed")
    unchanged=changes[~changes["selected_candidate_changed"].astype(bool)]; replay_max=0.0; replay_label_changes=0
    for row in unchanged.itertuples():
        scoped=oof[(oof["repetition"]==row.repetition)&(oof["outer_fold"]==row.outer_fold)&(oof["model"]==row.model)]
        left=scoped[scoped["selection_objective"]=="macro_f1"].sort_values("sample_index"); right=scoped[scoped["selection_objective"]=="qwk"].sort_values("sample_index")
        replay_max=max(replay_max,float(np.max(np.abs(left[["prob_class_2","prob_class_3","prob_class_4"]].to_numpy(float)-right[["prob_class_2","prob_class_3","prob_class_4"]].to_numpy(float)))))
        replay_label_changes+=int(np.sum(left["y_pred"].to_numpy(int)!=right["y_pred"].to_numpy(int)))
    _require(len(unchanged)==58 and replay_label_changes==0 and replay_max<=1e-15,"Unchanged-candidate replay exceeded tolerance")
    metric_names=c["report_metrics"]; recomputed=[]; classes=[]
    for (rep,regime,model),rows in oof.groupby(["repetition","selection_objective","model"],sort=True):
        rows=rows.sort_values("sample_index"); bundle=ordinal_evaluation_bundle_v3(rows["y_true"].astype(int),rows["y_pred"].astype(int),rows[["prob_class_2","prob_class_3","prob_class_4"]].to_numpy(float),labels=LABELS,dataset_key="inx_primary",model_name=str(model)); rec={"repetition":int(rep),"model":model,"selection_objective":regime}; rec.update({m:float(bundle["aggregate_metrics"][m]) for m in metric_names})
        for item in bundle["per_class_metrics"]:
            label=int(item["class_label"]); rec[f"rating_{label}_recall"]=float(item["recall"]); rec[f"rating_{label}_f1"]=float(item["f1"]); classes.append({"repetition":int(rep),"selection_objective":regime,**item})
        scoped=schedule[(schedule["repetition"]==rep)&(schedule["model"]==model)&(schedule["selection_objective"]==regime)].sort_values("outer_fold"); rec["selected_candidate_ids_json"]=json.dumps(scoped["selected_candidate_index"].astype(int).tolist(),separators=(",",":")); recomputed.append(rec)
    expected_results=pd.DataFrame(recomputed).sort_values(["repetition","model","selection_objective"]).reset_index(drop=True)
    results=results.sort_values(["repetition","model","selection_objective"]).reset_index(drop=True); _require(list(results.columns)==list(expected_results.columns),"Result columns drifted")
    for col in results.columns:
        if pd.api.types.is_numeric_dtype(expected_results[col]): _require(np.allclose(results[col].to_numpy(float),expected_results[col].to_numpy(float),rtol=0,atol=1e-14),f"Metric recomputation failed: {col}")
        else: _require(results[col].astype(str).equals(expected_results[col].astype(str)),f"Result identity failed: {col}")
    expected_pc=pd.DataFrame(classes).sort_values(["repetition","selection_objective","model_name","class_label"]).reset_index(drop=True); actual_pc=per_class.sort_values(["repetition","selection_objective","model_name","class_label"]).reset_index(drop=True)
    _require(expected_pc.shape==actual_pc.shape and np.allclose(expected_pc[["precision","recall","f1","support"]],actual_pc[["precision","recall","f1","support"]],rtol=0,atol=1e-14),"Per-class recomputation failed")
    _require(len(results)==60 and len(per_class)==180 and len(comparisons)==5,"Published summary grid drifted")
    _require(int(changes["selected_candidate_changed"].sum())==92,"Candidate-change count drifted")
    return {"status":"PASS_INDEPENDENT_RECOMPUTATION","run_id":metadata["run_id"],"generation_commit":metadata["git_identity"]["commit"],"phase1c_generation_commit":phase1c["generation_commit"],"candidate_rows":len(candidates),"schedule_rows":len(schedule),"candidate_changes":92,"macro_reuse_max_abs_serialization_difference":macro_reuse_max,"unchanged_candidate_cells":58,"unchanged_replay_max_abs_probability_difference":replay_max,"oof_rows":len(oof),"result_rows":len(results),"per_class_rows":len(per_class),"repetitions":5,"models":6,"qwk_outer_fits":150,"new_inner_fits":0,"network_calls":0,"paid_api_calls":0}

def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("run_dir",type=Path); parser.add_argument("--contract",type=Path,default=DEFAULT_CONTRACT); args=parser.parse_args(); print(json.dumps(validate_run(args.run_dir,args.contract),indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
