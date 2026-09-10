"""Build the additive ESWA final claim boundary without altering Round 2 history."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "reports" / "submission_eswa"
OLD = ROOT / "reports" / "research_log" / "major_revision_round2" / "round2_claim_matrix" / "ROUND2_CLAIM_MATRIX.csv"
OUT = BASE / "ESWA_FINAL_CLAIM_MATRIX.csv"
HISTORICAL_DIGEST = "751208d036597461606bd02d68bfa9e642a4aa5ecf5df2df3fd7e46b99084aea"


def file_hash(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


with OLD.open(encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    fields = list(reader.fieldnames or [])
    rows = list(reader)

# R2N005 remains historically intact in its source matrix. In the final boundary it
# is superseded by an exact system-identity formulation.
for row in rows:
    if row["claim_id"] == "R2N005":
        row["disposition"] = "modified"
        row["active_for_rewrite"] = "False"
        row["replacement_claim_ids"] = "ESWAF004"


def claim(
    claim_id: str,
    disposition: str,
    active: bool,
    section: str,
    component: str,
    claim_type: str,
    support: str,
    text: str,
    source: str,
    selector: str = "{}",
    exact: str = "",
    display: str = "",
    places: str = "",
    scope: str = "",
    qualifier: str = "",
    prohibited: str = "",
    prior: str = "",
) -> dict[str, str]:
    row = {field: "" for field in fields}
    row.update(
        claim_id=claim_id,
        prior_claim_id=prior,
        disposition=disposition,
        active_for_rewrite=str(active),
        manuscript_section=section,
        component=component,
        claim_type=claim_type,
        support_level=support,
        approval_status="authorized_final_revision_boundary",
        proposed_claim=text,
        source_path=source,
        source_sha256=file_hash(source) if source else "",
        source_selector=selector,
        exact_value=exact,
        display_value=display,
        rounding_places=places,
        display_scale="1",
        evidence_scope=scope,
        mandatory_qualifier=qualifier,
        prohibited_overclaim=prohibited,
    )
    return row


repeat_results = "reports/submission_eswa/repeated_selection/repeated_selection_results.csv"
repeat_compare = "reports/submission_eswa/repeated_selection/repetition_comparisons.csv"
repeat_meta = "reports/submission_eswa/repeated_selection/stage_metadata.json"
subgroup_source = "reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use/subgroup_gap_sensitivity.csv"

rows.extend(
    [
        claim("ESWAF001", "new", True, "Abstract; Methods; Discussion; Limitations; Conclusion", "hrdataset_role", "narrative", "source_documented_boundary", "HRDataset_v14 is a publicly available synthetic teaching dataset created for a graduate HR case study and representing a fictitious organization.", "reports/submission_eswa/HRDATASET_SYNTHETIC_DISCLOSURE_AUDIT.md", scope="Upstream-described dataset role.", qualifier="Treat it as synthetic teaching data.", prohibited="Do not present it as an observed company cohort or real-world external validation."),
        claim("ESWAF002", "new", True, "Abstract; Results; Discussion", "hrdataset_role", "narrative", "bounded_synthesis", "The HRDataset_v14 analysis demonstrates protocol portability and target-formulation sensitivity under another synthetic schema, not real-world cross-organizational transport.", "reports/submission_eswa/HRDATASET_SYNTHETIC_DISCLOSURE_AUDIT.md", scope="Separately fitted models on a synthetic secondary dataset.", qualifier="Feature, target, population, and parameter differences remain explicit.", prohibited="Do not claim external validation, cross-organizational generalization, or deployment evidence."),
        claim("ESWAF003", "new", True, "Methods; Supplement", "hrdataset_role", "narrative", "terminology_control", "The preferred label is secondary protocol sensitivity on a synthetic HR teaching dataset.", "reports/submission_eswa/HRDATASET_SYNTHETIC_DISCLOSURE_AUDIT.md", qualifier="Use consistent synthetic-data wording throughout."),
        claim("ESWAF004", "modified", True, "Methods 3.10; subgroup Results; Table 9; Discussion; Supplement S6", "phase2c", "narrative", "exact_identity", "The primary subgroup results are P3 nominal-XGBoost subgroup diagnostics; P3 is the feature policy and nominal XGBoost is the fitted-system family.", subgroup_source, selector='{"system_id":"no_salary_hike_no_attrition_no_department","v3_system_label":"P3_PRIMARY_LEAKAGE_AWARE"}', scope="Persisted exactly-once OOF nominal-XGBoost predictions under P3.", qualifier="Do not generalize the displayed gaps to all six trained systems.", prohibited="Do not use P3 alone as though it were a model identity.", prior="R2N005"),
        claim("ESWAF005", "new", True, "Methods; repeated sensitivity Results", "repeated_selection", "narrative", "procedural_boundary", "The repeated selection-objective sensitivity reused the exact five Phase 1C repeated 5×5 nested-CV split identities, candidate registries, preprocessing, seeds, tolerance, and tie logic.", repeat_meta, scope="Six trained systems; no new split, candidate, or seed.", qualifier="Outer-test outcomes remained evaluation-only.", prohibited="Do not imply a new seed search or altered grid."),
        claim("ESWAF006", "new", True, "Methods; repeated sensitivity Results", "repeated_selection", "numerical", "direct_exact", "The repeated sensitivity added 150 QWK-selected outer fits and no new inner or baseline fits; 36,000 macro-F1-selected OOF rows were reused.", repeat_meta, exact="150", display="150", places="0", scope="5 repetitions × 5 outer folds × 6 trained systems.", qualifier="The total scientific lineage is 5,875 fits including 5,725 prior Phase 1C fits.", prohibited="Do not report reused fits as newly executed."),
        claim("ESWAF007", "new", True, "Repeated sensitivity Results", "repeated_selection", "numerical", "direct_exact", "Selection objective changed 92 of 150 model-fold candidate choices across the five repeated designs.", repeat_compare, selector='{"aggregate":"sum(candidate_changes)"}', exact="92", display="92/150", places="0", scope="Six systems × five folds × five repetitions.", qualifier="Counts are descriptive across fixed split identities.", prohibited="Do not treat cells as independent inferential observations."),
        claim("ESWAF008", "new", True, "Repeated sensitivity Results", "repeated_selection", "numerical", "direct_exact", "The complete six-model ordering changed for all seven aggregate metrics in every one of the five repeated designs.", repeat_compare, selector='{"full_ordering_changes":7,"repetitions":5}', exact="5", display="5/5", places="0", scope="Seven aggregate metrics reported in the comparison table.", qualifier="Leader changes and full-ordering changes are separate summaries.", prohibited="Do not collapse ordering changes into a universal materiality claim."),
        claim("ESWAF009", "new", True, "Abstract; repeated sensitivity Results; Discussion", "repeated_selection", "narrative", "direct_descriptive", "For nominal XGBoost, QWK selection increased QWK and reduced ordinal MAE in all five repeated designs while rating-4 recall fell in all five.", repeat_results, selector='{"model":"xgboost","delta":"qwk_minus_macro_f1","repetitions":5}', scope="Five prespecified repeated split identities.", qualifier="Direction consistency is descriptive and magnitudes vary by split.", prohibited="Do not claim that QWK optimization always destroys class-4 recall."),
        claim("ESWAF010", "new", True, "Repeated sensitivity Results", "repeated_selection", "numerical", "direct_descriptive", "Nominal-XGBoost mean QWK changed from 0.5833 under macro-F1 selection to 0.6377 under QWK selection, while mean rating-4 recall changed from 0.1894 to 0.0167.", repeat_results, selector='{"model":"xgboost","groupby":"selection_objective","mean_over_repetition":true}', scope="Means over five fixed repetitions.", qualifier="These are descriptive means, not confidence estimates.", prohibited="Do not infer a population-average causal effect."),
        claim("ESWAF011", "new", True, "Repeated sensitivity Results; Discussion", "repeated_selection", "narrative", "direct_descriptive", "For cumulative-threshold XGBoost, QWK increased in all five repeated designs and QWK-selected rating-4 recall was zero in all five.", repeat_results, selector='{"model":"cumulative_threshold_xgboost","repetitions":5}', scope="Five prespecified repeated split identities.", qualifier="This is system-specific and descriptive.", prohibited="Do not generalize the result to every model or future dataset."),
        claim("ESWAF012", "new", True, "Repeated sensitivity Results; Discussion", "repeated_selection", "narrative", "direct_descriptive", "Random Forest showed smaller and less uniform changes: QWK increased in four repetitions and was unchanged in one; rating-4 recall fell in two and was unchanged in three.", repeat_compare, selector='{"columns":["rf_delta_qwk","rf_delta_rating4_recall"]}', scope="Five prespecified repeated split identities.", qualifier="Random Forest already had near-zero rating-4 recall under both regimes.", prohibited="Do not describe the repeated effect as homogeneous across models."),
        claim("ESWAF013", "new", True, "Abstract; Discussion; Conclusion", "repeated_selection", "narrative", "bounded_synthesis", "The macro-F1-versus-QWK trade-off is strongly replicated for nominal and cumulative-threshold XGBoost across the five split identities, with model-specific heterogeneity.", "reports/submission_eswa/REPEATED_SELECTION_OBJECTIVE_SENSITIVITY.md", scope="Directional robustness within the fixed repeated design.", qualifier="The classification is based on direction consistency, not an inferential interval.", prohibited="Do not call the effect universal or the repetitions independent population samples."),
    ]
)

for index, text in enumerate(
    [
        "PROHIBITED: HRDataset_v14 provides external validation.",
        "PROHIBITED: Cross-organizational generalization was demonstrated.",
        "PROHIBITED: Synthetic data validate real-world deployment.",
        "PROHIBITED: The selection-objective effect is universal.",
        "PROHIBITED: QWK optimization always destroys rating-4 recall.",
        "PROHIBITED: P3 is leakage-free or prospectively valid.",
        "PROHIBITED: A subgroup gap proves discrimination or legal noncompliance.",
        "PROHIBITED: JobRole causes department bias.",
        "PROHIBITED: SHAP identifies causal drivers.",
        "PROHIBITED: Calibrated probabilities are prospectively valid.",
        "PROHIBITED: The protocol is proven deployment-ready or first-ever.",
    ],
    start=1,
):
    rows.append(claim(f"ESWAP{index:03d}", "prohibited", False, "All", "final_boundary", "narrative", "prohibited", text, "reports/submission_eswa/ESWA_FINAL_REVISION_PLAN.md", prohibited=text))

with OUT.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
active = sum(row["active_for_rewrite"] == "True" for row in rows)
counts: dict[str, int] = {}
for row in rows:
    counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1

(BASE / "ESWA_FINAL_CLAIM_DIGEST.txt").write_text(
    f"schema=eswa_final_claim_boundary_v1\n"
    f"historical_round2_digest={HISTORICAL_DIGEST}\n"
    f"historical_round2_matrix={OLD.relative_to(ROOT).as_posix()}\n"
    f"historical_round2_matrix_sha256={file_hash(OLD.relative_to(ROOT).as_posix())}\n"
    f"final_matrix=reports/submission_eswa/ESWA_FINAL_CLAIM_MATRIX.csv\n"
    f"final_matrix_sha256={digest}\n"
    f"row_count={len(rows)}\nactive_count={active}\n",
    encoding="utf-8",
)

lines = [
    "# ESWA final claim matrix",
    "",
    f"Final matrix SHA-256: `{digest}`",
    "",
    f"The immutable approved Round 2 digest remains `{HISTORICAL_DIGEST}`. Its 129-row source matrix is unchanged. This final matrix copies that history, marks the broad P3 subgroup wording as modified in this final boundary, and adds exact synthetic-data, system-identity, repeated-selection, and prohibited-claim rows.",
    "",
    "## Counts",
    "",
    "| Disposition | Rows |",
    "| --- | ---: |",
]
for key in sorted(counts):
    lines.append(f"| {key} | {counts[key]} |")
lines += ["", f"Active claims: **{active}** of **{len(rows)}**.", "", "## Rewrite gate", "", "Only rows with `active_for_rewrite=True` may support affirmative manuscript language. Prohibited rows and each active row's mandatory qualifier govern the final rewrite. The repeated analysis bounds rather than replaces the canonical 10×5 finding."]
(BASE / "ESWA_FINAL_CLAIM_MATRIX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print({"rows": len(rows), "active": active, "counts": counts, "digest": digest})
