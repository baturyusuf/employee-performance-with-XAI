"""Publish aggregate-only repeated selection evidence after independent validation."""
from pathlib import Path
from hashlib import sha256
import json, shutil, sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.governance.eswa_repeated_selection_validator_v1 import validate_run
BASE=ROOT/"reports/submission_eswa"
RUN=ROOT/"reports/major_revision_round2_runs/eswa_selection_v1_20260910T000000Z_914e1e0/repeated_selection_objective"
OUT=BASE/"repeated_selection"
receipt=validate_run(RUN)
if OUT.exists():
    for path in OUT.iterdir():
        if path.is_file(): path.unlink()
else: OUT.mkdir(parents=True)

files=["candidate_evidence.csv","selection_schedule.csv","selected_candidate_changes.csv","repeated_selection_results.csv","per_class_metrics.csv","repetition_comparisons.csv","stage_metadata.json"]
for name in files: shutil.copyfile(RUN/name,OUT/name)
shutil.copyfile(RUN/"repeated_selection_results.csv",BASE/"REPEATED_SELECTION_RESULTS.csv")
(OUT/"independent_validation_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")

results=pd.read_csv(RUN/"repeated_selection_results.csv")
comparisons=pd.read_csv(RUN/"repetition_comparisons.csv")
def means(model):
    rows=results[results.model.eq(model)].groupby("selection_objective")[["macro_f1","quadratic_weighted_kappa","ordinal_mae","rating_4_recall"]].mean()
    return rows
xgb=means("xgboost"); cumulative=means("cumulative_threshold_xgboost"); rf=means("random_forest")
lines=[
"# Repeated selection-objective sensitivity",
"",
"## Design",
"",
"The analysis reuses the five exact Phase 1C repeated 5×5 nested-CV split identities, its 1,100 hash-bound inner-candidate records, and the six trained-system registries without adding a candidate or seed. Macro-F1 selection uses QWK as the tie-break; QWK selection swaps those roles. Both use the same inclusive 0.001 primary practical-tie pool and lowest-index final tie rule. The outer test partition is evaluation-only.",
"",
"The 36,000 tuned-model macro-F1-selected OOF rows are reused from the independently validated Phase 1C run. Exactly 150 new QWK-selected outer fits were performed: five repetitions × five outer folds × six models. No inner fit or baseline fit was added. The scientific lineage is 5,725 prior Phase 1C fits plus 150 new fits. Employee-level OOF rows remain local and are not published.",
"",
"## Result",
"",
"The selection-objective trade-off is **strongly replicated for nominal and cumulative-threshold XGBoost across all five split identities, with model-specific heterogeneity**. This classification is based on direction consistency, not a confidence interval: nominal XGBoost gained QWK and improved ordinal MAE in 5/5 repetitions while rating-4 recall fell in 5/5; cumulative-threshold XGBoost gained QWK in 5/5 while QWK-selected rating-4 recall was zero in 5/5. Random Forest changed much less and already had near-zero rating-4 recall under both regimes. The effect is therefore persistent for the boosting systems that drive the canonical contrast, but it is not universal across models and its magnitude varies by split.",
"",
"| System | Selection | Mean macro-F1 | Mean QWK | Mean ordinal MAE | Mean rating-4 recall |",
"| --- | --- | ---: | ---: | ---: | ---: |",
]
for label,table in [("Nominal XGBoost",xgb),("Cumulative-threshold XGBoost",cumulative),("Random Forest",rf)]:
    for regime in ["macro_f1","qwk"]:
        row=table.loc[regime]; lines.append(f"| {label} | {regime} | {row.macro_f1:.4f} | {row.quadratic_weighted_kappa:.4f} | {row.ordinal_mae:.4f} | {row.rating_4_recall:.4f} |")
lines += ["","## Repetition-level direction checks","", "| Quantity | Positive | Negative | Zero | Mean delta | Range |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
for column,label in [("nominal_xgb_delta_qwk","Nominal XGB ΔQWK"),("nominal_xgb_delta_mae","Nominal XGB ΔMAE"),("nominal_xgb_delta_rating4_recall","Nominal XGB Δrating-4 recall"),("cumulative_xgb_delta_qwk","Cumulative XGB ΔQWK"),("cumulative_xgb_delta_rating4_recall","Cumulative XGB Δrating-4 recall"),("rf_delta_qwk","RF ΔQWK"),("rf_delta_rating4_recall","RF Δrating-4 recall")]:
    values=comparisons[column]; lines.append(f"| {label} | {(values>0).sum()}/5 | {(values<0).sum()}/5 | {(values==0).sum()}/5 | {values.mean():.4f} | {values.min():.4f} to {values.max():.4f} |")
lines += ["","Deltas are QWK-selected minus macro-F1-selected. Negative MAE is an improvement; negative rating-4 recall is deterioration.","","Candidate selections changed in 15, 20, 17, 20, and 20 of 30 model-fold opportunities across repetitions (92/150 overall). Leaders changed for 7, 5, 7, 4, and 7 of seven reported metrics, while the full six-model ordering changed for all seven metrics in every repetition. These are separate descriptive facts, not a composite materiality flag.","","## Claim boundary","","The canonical 10×5 result remains a valid result for its persisted split identity. The repeated analysis strengthens the directional robustness of the XGBoost selection trade-off while showing model and magnitude heterogeneity. It does not show that QWK selection always harms the highest class, that the effect is universal, or that five repetitions are independent inferential samples.","","Exact aggregate results are in `REPEATED_SELECTION_RESULTS.csv`; fold-level selected candidate IDs are in `repeated_selection/selection_schedule.csv`. The independent receipt reports 72,000 locally validated OOF rows, 58 unchanged-candidate replay cells, zero prediction-label differences in those cells, and a maximum floating-point serialization difference of 3.39×10⁻²¹."]
(BASE/"REPEATED_SELECTION_OBJECTIVE_SENSITIVITY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

manifest=[]
for path in sorted(OUT.iterdir()):
    if path.name=="manifest.json" or not path.is_file(): continue
    manifest.append({"path":path.name,"bytes":path.stat().st_size,"sha256":sha256(path.read_bytes()).hexdigest(),"contains_employee_level_rows":False})
(OUT/"manifest.json").write_text(json.dumps({"schema_version":1,"package":"eswa_repeated_selection_objective_v1","historical_claim_digest":"751208d036597461606bd02d68bfa9e642a4aa5ecf5df2df3fd7e46b99084aea","files":manifest},indent=2,sort_keys=True)+"\n",encoding="utf-8")
print({"validation":receipt["status"],"files":len(manifest)+1,"candidate_changes":int(comparisons.candidate_changes.sum())})
