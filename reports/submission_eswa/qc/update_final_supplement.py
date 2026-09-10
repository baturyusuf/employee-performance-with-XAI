"""Publish the frozen repeated-selection summary into the anonymous supplement."""
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "reports/submission_eswa"
SOURCE = BASE / "repeated_selection/repeated_selection_results.csv"
COMPARE = BASE / "repeated_selection/repetition_comparisons.csv"
TABLE = BASE / "supplement/tables/Table_S9.csv"
MAIN = BASE / "supplement/main.md"

shutil.copyfile(SOURCE, TABLE)
results = pd.read_csv(SOURCE)
comparisons = pd.read_csv(COMPARE)
models = ["xgboost", "cumulative_threshold_xgboost", "random_forest"]
labels = {
    "xgboost": "Nominal XGBoost",
    "cumulative_threshold_xgboost": "Cumulative-threshold XGBoost",
    "random_forest": "Random Forest",
}

marker = "# Table S9. Repeated selection-objective sensitivity"
text = MAIN.read_text(encoding="utf-8-sig")
if marker in text:
    text = text.split(marker, 1)[0].rstrip() + "\n\n"

lines = [
    marker,
    "",
    "This analysis reuses the exact five Phase 1C repeated 5×5 nested-CV split identities, preprocessing, candidate registries, seeds, inclusive 0.001 tolerance, and deterministic tie logic. Macro-F1-selected OOF predictions were reused. QWK selection added 150 outer fits and no inner or baseline fits; outer-test outcomes remained evaluation-only. Values below are descriptive means over the five fixed repetitions.",
    "",
    "| System | Selection | Macro-F1 | Balanced acc. | QWK | MAE | RPS | Log loss | Brier | Rating-4 recall |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for model in models:
    for objective in ["macro_f1", "qwk"]:
        row = results[(results.model == model) & (results.selection_objective == objective)].mean(numeric_only=True)
        lines.append(
            f"| {labels[model]} | {objective.replace('_', '-')} | {row.macro_f1:.4f} | {row.balanced_accuracy:.4f} | "
            f"{row.quadratic_weighted_kappa:.4f} | {row.ordinal_mae:.4f} | {row.ranked_probability_score:.4f} | "
            f"{row.nll_log_loss:.4f} | {row.multiclass_brier:.4f} | {row.rating_4_recall:.4f} |"
        )
lines += [
    "",
    "Candidate selections changed in 15, 20, 17, 20, and 20 of 30 model-fold cells (92/150 overall). Full six-model ordering changed for all seven compared aggregate metrics in every repetition. Nominal XGBoost gained QWK and improved MAE while rating-4 recall fell in 5/5 repetitions. Cumulative-threshold XGBoost gained QWK in 5/5 and had zero QWK-selected rating-4 recall in 5/5. Random Forest gained QWK in 4/5 with one zero change; its rating-4 recall fell in 2/5 and was unchanged in 3/5.",
    "",
    "The exact 60 repetition × model × selection rows, including all required aggregate and per-class metrics plus the five selected candidate IDs, are supplied in `tables/Table_S9.csv`. Direction counts are not confidence intervals. Repetitions reuse the same observations, and the effect is not claimed to be universal.",
]
MAIN.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")
print({"table_rows": len(results), "candidate_changes": int(comparisons.candidate_changes.sum())})
