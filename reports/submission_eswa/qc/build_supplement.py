"""Build a readable anonymous supplement plus exact machine-readable tables."""
from pathlib import Path
import csv
import shutil

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "reports/submission_eswa"
OUT = BASE / "supplement/main.md"
TABLES = BASE / "supplement/tables"
TABLES.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def md_table(rows, fields, labels=None):
    labels = labels or fields
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        values = [str(row.get(field, "")).replace("|", "\\|").replace("_", " ") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def copy_table(source, target):
    shutil.copyfile(ROOT / source, TABLES / target)


s1_path = "reports/research_log/major_revision_round2/METHOD_REPRODUCIBILITY_TABLES/TABLE_S1_FEATURE_AVAILABILITY.csv"
s2_path = "reports/research_log/major_revision_round2/METHOD_REPRODUCIBILITY_TABLES/TABLE_S2_PREPROCESSING_HYPERPARAMETER_SEARCH.csv"
s3_path = "reports/research_log/major_revision_round2/METHOD_REPRODUCIBILITY_TABLES/TABLE_S3_CV_SEED_CONTRACT.csv"
s4_path = "reports/research_log/major_revision_round2/selection_objective_sensitivity/per_class_metrics.csv"
s6_path = "reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv"
s7_path = "reports/research_log/major_revision_round2/selection_objective_sensitivity/confusion_matrix.csv"
s8_path = "reports/submission_eswa/literature/COMPARISON_TABLE.csv"
for source, target in [(s1_path,"Table_S1.csv"),(s2_path,"Table_S2.csv"),(s3_path,"Table_S3.csv"),
                       (s4_path,"Table_S4.csv"),(s6_path,"Table_S6.csv"),(s7_path,"Table_S7.csv"),
                       (s8_path,"Table_S8.csv")]:
    copy_table(source, target)

parts = [
    "# Supplementary material",
    "This anonymous scientific supplement reports method contracts and aggregate evidence for the accompanying manuscript. It contains no author identity, employee-level record, fitted model, approval history, internal review, or development diary. Exact, unabridged versions of Tables S1--S4 and S6--S8 are supplied as CSV files in the `tables` directory.",
]

# S1: compact governance matrix; exact semantic descriptions and justifications remain in CSV.
s1 = read_csv(s1_path)
timing = {"not_applicable_identifier":"Identifier", "plausibly_pre_existing_timestamp_unverified":"Pre-existing?", "timing_uncertain":"Uncertain", "unknown_may_postdate_decision":"May postdate"}
risk = {"identifier":"Identifier", "sensitive_attribute":"Sensitive", "ordinary_predictor":"Ordinary", "organizational_proxy":"Org. proxy", "timing_uncertain":"Timing", "outcome_proximal":"Outcome-near"}
s1_governance = []
s1_matrix = []
for row in s1:
    s1_governance.append({"feature":row["feature_name"], "timing":timing.get(row["timing_status"], row["timing_status"]), "risk":risk.get(row["risk_type"], row["risk_type"])})
    item = {"feature": row["feature_name"]}
    for policy in ["P0","P1","P2","P3","P4","P5"]:
        item[policy] = "I" if row[policy] == "Included" else "E"
    s1_matrix.append(item)
parts += ["# Table S1. Feature-availability and governance matrix",
          "`I` denotes included and `E` excluded. A question mark marks plausible pre-existence without a verified row-level timestamp. Full semantic descriptions, governance classes, justifications, and timestamp caveats appear in `tables/Table_S1.csv`.",
          "## Table S1a. Timing and risk classification",
          md_table(s1_governance,["feature","timing","risk"],["Feature","Timing","Risk"]),
          "## Table S1b. Policy inclusion matrix",
          md_table(s1_matrix,["feature","P0","P1","P2","P3","P4","P5"],["Feature","P0","P1","P2","P3","P4","P5"])]

# S2: compact search registry, with exact JSON grids in CSV.
s2 = read_csv(s2_path)
s2_compact = []
for row in s2:
    s2_compact.append({"section":row["section"], "item":row["display_name"], "candidates":row["candidate_count"],
                       "fit":"Training partitions only", "primary":row["selection_primary"],
                       "tie":row["selection_tie_break"], "test":"Evaluation only"})
parts += ["# Table S2. Preprocessing and search contract",
          "All preprocessing and candidate selection are confined to the current training partition. Exact estimator paths, fixed parameters, candidate grids, tolerances, and failure rules appear in `tables/Table_S2.csv`.",
          md_table(s2_compact,["section","item","candidates","fit","primary","tie","test"],["Type","Component","Candidates","Fit scope","Primary","Tie break","Outer test"])]

# S3: compact CV/seed overview; exact schedules and reuse boundaries in CSV.
s3 = read_csv(s3_path)
s3_compact=[]
analysis_labels = {
    "Canonical INX six-model benchmark":"Canonical benchmark",
    "INX repeated nested-CV sensitivity":"Repeated-CV sensitivity",
    "Policy fixed-schedule sensitivity":"Fixed-policy sensitivity",
    "Policy independently retuned sensitivity":"Retuned-policy sensitivity",
    "Selection-objective sensitivity":"Selection-objective sensitivity",
    "Canonical sigmoid calibration diagnostics":"Sigmoid calibration",
    "HRDataset target-formulation/CV sensitivity":"HR target/CV sensitivity",
    "HR target-alias matched-sample sensitivity":"HR target-alias sensitivity",
}
def fold_label(value):
    value = value.lower()
    if "10" in value: return "10-fold stratified"
    if "5" in value: return "5-fold stratified"
    return "Persisted folds"
for row in s3:
    s3_compact.append({"analysis":analysis_labels.get(row["analysis"],row["analysis"]), "population":row["population_or_policy"], "reps":row["repetitions"],
                       "outer":fold_label(row["outer_design"]), "inner":fold_label(row["inner_design"]), "test":"Evaluation only"})
parts += ["# Table S3. Cross-validation overview",
          "Exact seed schedules, fold identities, out-of-fold coverage, calibration isolation, selection-score sources, and reuse/refit boundaries appear in `tables/Table_S3.csv`.",
          md_table(s3_compact,["analysis","population","reps","outer","inner","test"],["Analysis","Population","n","Outer design","Inner design","Outer test"])]

s4 = read_csv(s4_path)
for row in s4:
    for field in ["precision","recall","f1"]:
        row[field] = f"{float(row[field]):.4f}"
parts += ["# Table S4. Per-class performance by selection objective",
          md_table(s4,["selection_objective","model_name","class_label","precision","recall","f1","support"],["Selection","System","Rating","Precision","Recall","F1","Support"]),
          "Hard predictions use the declared argmax rule with no class-specific threshold optimization. The exact table is `tables/Table_S4.csv`."]

gates = [
    {"stage":"Information","input":"Features and intended context","audit":"Availability/timing contract","failure":"Timing unverified","claim":"Sensitivity only"},
    {"stage":"Selection","input":"Outer-training partitions","audit":"Nested candidate selection","failure":"Test-informed selection","claim":"Evaluation invalid"},
    {"stage":"Prediction","input":"Untouched outer test","audit":"OOF probabilities; ordinal/class metrics","failure":"Extreme-class collapse","claim":"Bound aggregate claim"},
    {"stage":"Calibration","input":"Cross-fitted training probabilities","audit":"Predeclared sigmoid fit","failure":"Outer-test fit/selection","claim":"Calibration invalid"},
    {"stage":"Explanation","input":"Prediction-producing fold model","audit":"Identity, additivity, stability, deletion","failure":"Model mismatch","claim":"No explanation claim"},
    {"stage":"Subgroup/proxy","input":"Held-out outputs/categories","audit":"Support, interval, distinct proxy questions","failure":"Unsupported denominator","claim":"Descriptive/not estimated"},
    {"stage":"Replication","input":"Separately fitted HR systems","audit":"Target, folds, calibration, support","failure":"Construct/transport conflation","claim":"Partial replication"},
    {"stage":"Evidence","input":"Aggregate source rows","audit":"Selector, value, hash, qualifier","failure":"Unresolved/stale number","claim":"No numerical claim"},
]
parts += ["# Table S5. Complete audit gates", md_table(gates,["stage","input","audit","failure","claim"],["Stage","Input","Audit","Failure","Permissible claim"])]

s6 = read_csv(s6_path)
for row in s6:
    for field in ["gap_max_minus_min","simultaneous_ci_low","simultaneous_ci_high"]:
        row[field] = f"{float(row[field]):.4f}"
parts += ["# Table S6. Subgroup gap uncertainty and support",
          "Intervals are 95% simultaneous exploratory intervals based on the studentized maximum absolute bootstrap deviation over all estimable prespecified P3 attribute, support-threshold, and metric gap cells. Eligibility is fixed before resampling; fitted-model variability is excluded. These diagnostics do not certify fairness."]
for attribute in dict.fromkeys(row["attribute"] for row in s6):
    group = [row for row in s6 if row["attribute"] == attribute]
    parts += [f"## {attribute}", md_table(group,["metric","gap_max_minus_min","minimum_group","maximum_group","simultaneous_ci_low","simultaneous_ci_high"],["Metric","Gap","Minimum group","Maximum group","Simultaneous low","Simultaneous high"])]
parts += ["The exact support counts, pointwise intervals, resample counts, source identities, and claim boundaries appear in `tables/Table_S6.csv`."]

s7 = read_csv(s7_path)
parts += ["# Table S7. Selection-objective confusion matrices",
          md_table(s7,["selection_objective","model_name","true_label","predicted_label","count"],["Selection","System","True","Predicted","Count"]),
          "The exact table, including the dataset key, is `tables/Table_S7.csv`."]

s8 = read_csv(s8_path)
parts += ["# Table S8. Bounded literature comparison"]
for row in s8:
    parts.append(f"## {row['Prior paper']}\n\n**Problem and evidence:** {row['Problem']}; {row['Dataset']}. **Method and XAI:** {row['Method']}; {row['XAI']}. **Boundary relative to this study:** {row['Difference from ours']} {row['Evidence boundary']}")
parts += ["The full comparison dimensions and official-source links appear in `tables/Table_S8.csv`.",
          "# Numerical evidence ledger",
          "The machine-readable ledger and its twenty anonymous aggregate source tables are supplied in the `evidence` directory. `SOURCE_INDEX.csv` records every anonymous source hash. The ledger preserves exact values, selectors, rounding, required qualifiers, and prohibited interpretations for the 99 active numerical claims."]

OUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
print({"s1":len(s1),"s2":len(s2),"s3":len(s3),"s4":len(s4),"s6":len(s6),"s7":len(s7),"s8":len(s8),"output":str(OUT)})
