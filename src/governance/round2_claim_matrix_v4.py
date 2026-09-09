"""Freeze and validate the pending-approval Round 2 claim matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import uuid
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CONFIG = Path("configs/round2_claim_matrix_v4.json")
ROUND2_DIR = Path("reports/research_log/major_revision_round2")
OUTPUT_DIR = ROUND2_DIR / "round2_claim_matrix"
HISTORICAL_CLAIM_DIR = Path("reports/research_log/major_revision_v3/phase5a_claim_matrix")
HISTORICAL_MANUSCRIPT_DIR = Path("reports/research_log/major_revision_v3/phase5b_manuscript")
MANIFEST_NAME = "manifest.json"
EXPECTED_OUTPUTS = frozenset(
    {
        "README.md",
        "ROUND2_CLAIM_MATRIX.csv",
        "ROUND2_CLAIM_MATRIX.md",
        "ROUND2_SOURCE_REGISTER.csv",
        "RESULTS_COMPARISON_ROUND2.csv",
        "DERIVED_SUMMARIES.csv",
        "ROUND2_CROSS_SOURCE_AUDIT.md",
        "ROUND2_CLAIM_DIGEST.txt",
        "APPROVAL_REQUEST.md",
        "approval_record.json",
        "provenance_receipt.json",
        MANIFEST_NAME,
    }
)
MIRRORED_FILES = (
    "ROUND2_CLAIM_MATRIX.md",
    "ROUND2_SOURCE_REGISTER.csv",
    "RESULTS_COMPARISON_ROUND2.csv",
    "ROUND2_CROSS_SOURCE_AUDIT.md",
    "ROUND2_CLAIM_DIGEST.txt",
)
DISPOSITIONS = frozenset({"retained", "modified", "superseded", "new", "prohibited"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class Round2ClaimMatrixError(RuntimeError):
    """Raised when the Round 2 claim boundary is incomplete or has drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Round2ClaimMatrixError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _csv_bytes(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return stream.getvalue().encode("utf-8")


def _load_csv(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"Source is absent: {path.as_posix()}.")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def _git_is_clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def _display(value: str, places: int, scale: str = "1") -> str:
    number = Decimal(value) * Decimal(scale)
    quantum = Decimal(1).scaleb(-places)
    return format(number.quantize(quantum, rounding=ROUND_HALF_EVEN), f".{places}f")


def _load_config(path: Path = CONFIG) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 2, "Round 2 config schema drifted.")
    _require(payload.get("phase") == "round2_claim_freeze", "Round 2 config phase drifted.")
    _require(payload.get("snapshot_date") == "2026-09-08", "Round 2 snapshot date drifted.")
    _require(payload.get("status") in {"pending_user_approval", "approved_for_rewrite"}, "Invalid Round 2 status.")
    approval = payload.get("user_approval", {})
    approved = approval.get("status") == "approved"
    _require(approval.get("status") in {"pending", "approved"}, "Invalid user-approval state.")
    _require(approved == (payload["status"] == "approved_for_rewrite"), "Approval and package states disagree.")
    controls = payload.get("controls", {})
    for field in (
        "manuscript_editing_authorized",
        "bibliography_editing_authorized",
        "reviewer_response_editing_authorized",
    ):
        _require(controls.get(field) is approved, f"{field} does not match approval state.")
    for field in (
        "release_authorized",
        "tag_creation_authorized",
        "doi_minting_authorized",
        "raw_data_publication_authorized",
        "network_required",
    ):
        _require(controls.get(field) is False, f"Unsafe Round 2 control: {field}.")
    _require(controls.get("paid_api_calls") == 0, "Paid API calls are prohibited.")
    plan = Path(payload["round2_plan"]["path"])
    _require(_sha256(plan) == payload["round2_plan"]["sha256"], "Round 2 plan hash drifted.")
    _require(len(payload.get("global_prohibited_assertions", [])) >= 15, "Prohibited assertions are incomplete.")
    return payload


def _source_row(path: Path, selector: Mapping[str, str]) -> tuple[int, dict[str, str]]:
    matches = [
        (index, row)
        for index, row in enumerate(_load_csv(path), start=2)
        if all(row.get(key) == value for key, value in selector.items())
    ]
    _require(len(matches) == 1, f"Selector {selector!r} matched {len(matches)} rows in {path.as_posix()}.")
    return matches[0]


def _derived_rows() -> list[dict[str, str]]:
    ranking_path = ROUND2_DIR / "selection_objective_sensitivity/ranking_changes.csv"
    candidate_path = ROUND2_DIR / "selection_objective_sensitivity/selected_candidate_changes.csv"
    cv_path = Path("reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/cv_design_sensitivity.csv")
    hr_candidate_path = ROUND2_DIR / "hr_target_alias_sensitivity/selected_hyperparameters.csv"
    policy_path = Path("reports/research_log/major_revision_v3/phase1d_policy_retuning/headline_policy_comparison.csv")

    ranking = _load_csv(ranking_path)
    candidates = _load_csv(candidate_path)
    cv_rows = _load_csv(cv_path)
    hr_candidates = _load_csv(hr_candidate_path)
    policies = {row["policy_id"]: row for row in _load_csv(policy_path)}

    rows: list[dict[str, str]] = []

    def add(summary_id: str, category: str, value: Any, denominator: Any, source: Path, derivation: str) -> None:
        rows.append(
            {
                "summary_id": summary_id,
                "category": category,
                "value": str(value),
                "denominator": str(denominator),
                "source_path": source.as_posix(),
                "source_sha256": _sha256(source),
                "derivation": derivation,
            }
        )

    add("selection_leader_changed", "selection", sum(row["leader_changed"] == "True" for row in ranking), len(ranking), ranking_path, "count True in leader_changed")
    add("selection_full_ordering_changed", "selection", sum(row["full_ordering_changed"] == "True" for row in ranking), len(ranking), ranking_path, "count True in full_ordering_changed")
    add("selection_candidate_changed", "selection", sum(row["selected_candidate_changed"] == "True" for row in candidates), len(candidates), candidate_path, "count True in selected_candidate_changed")
    add("hr_cv_inside", "hr_cv_design", sum(row["ten_fold_inside_repeated_range"] == "True" for row in cv_rows), len(cv_rows), cv_path, "count True in ten_fold_inside_repeated_range")
    add("hr_cv_outside", "hr_cv_design", sum(row["ten_fold_inside_repeated_range"] == "False" for row in cv_rows), len(cv_rows), cv_path, "count False in ten_fold_inside_repeated_range")
    add("hr_alias_candidate_changed", "hr_alias", sum(row["selected_candidate_changed"] == "True" for row in hr_candidates), len(hr_candidates), hr_candidate_path, "count True in selected_candidate_changed")

    for left, right in (("P3", "P4"), ("P4", "P5")):
        for estimand in ("fixed", "retuned"):
            for metric in ("macro_f1", "quadratic_weighted_kappa", "ordinal_mae"):
                column = f"{estimand}_{metric}"
                value = Decimal(policies[right][column]) - Decimal(policies[left][column])
                add(
                    f"{left.lower()}_{right.lower()}_{estimand}_{metric}_delta",
                    "timing_information",
                    str(value),
                    1,
                    policy_path,
                    f"{right}.{column} minus {left}.{column}",
                )
    return rows


def _legacy_claims(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    historical = _load_csv(HISTORICAL_CLAIM_DIR / "CLAIM_MATRIX.csv")
    dispositions = config["legacy_dispositions"]
    _require(set(dispositions) <= {row["claim_id"] for row in historical}, "Unknown historical disposition id.")
    claims: list[dict[str, Any]] = []
    for row in historical:
        disposition = dispositions.get(row["claim_id"], "retained")
        _require(disposition in {"retained", "modified", "superseded"}, "Invalid legacy disposition.")
        item = dict(row)
        item.update(
            {
                "prior_claim_id": row["claim_id"],
                "disposition": disposition,
                "active_for_rewrite": "True" if disposition == "retained" else "False",
                "approval_status": "pending_user_approval",
                "replacement_claim_ids": "",
            }
        )
        claims.append(item)
    replacements = {
        "C002": "R2N001",
        "C004": "R2N004",
        "C006": "R2N002",
        "C007": "R2N005",
        "C008": "R2N006 | R2N007 | R2N008",
        "C010": "R2N010",
        "C013": "R2N012",
    }
    for claim in claims:
        claim["replacement_claim_ids"] = replacements.get(claim["claim_id"], "")
    return claims


def _new_claims(derived_path: Path, derived_sha: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []

    def base(claim_id: str, section: str, component: str, disposition: str, proposed: str, source: Path, scope: str, qualifier: str, prohibited: str) -> dict[str, Any]:
        return {
            "claim_id": claim_id,
            "prior_claim_id": "",
            "disposition": disposition,
            "active_for_rewrite": "False" if disposition == "prohibited" else "True",
            "manuscript_section": section,
            "component": component,
            "claim_type": "narrative",
            "support_level": "procedural_boundary" if disposition == "prohibited" else "bounded_synthesis",
            "approval_status": "pending_user_approval",
            "proposed_claim": proposed,
            "source_path": source.as_posix(),
            "source_sha256": _sha256(source),
            "source_selector": "",
            "source_row_number": "",
            "source_column": "",
            "exact_value": "",
            "display_value": "",
            "rounding_places": "",
            "display_scale": "",
            "evidence_anchor": "",
            "evidence_scope": scope,
            "mandatory_qualifier": qualifier,
            "prohibited_overclaim": prohibited,
            "replacement_claim_ids": "",
        }

    def narrative(claim_id: str, section: str, component: str, proposed: str, source: Path, anchor: str, scope: str, qualifier: str, prohibited: str) -> None:
        item = base(claim_id, section, component, "new", proposed, source, scope, qualifier, prohibited)
        text = source.read_text(encoding="utf-8")
        _require(anchor in text, f"Narrative anchor drifted for {claim_id}.")
        item["evidence_anchor"] = anchor
        claims.append(item)

    def numerical(claim_id: str, section: str, component: str, proposed_template: str, source: Path, selector: Mapping[str, str], column: str, places: int, scope: str, qualifier: str, prohibited: str, scale: str = "1") -> None:
        row_number, row = _source_row(source, selector)
        _require(column in row, f"Missing source column for {claim_id}: {column}.")
        exact = row[column]
        shown = _display(exact, places, scale)
        item = base(claim_id, section, component, "new", proposed_template.format(value=shown), source, scope, qualifier, prohibited)
        item.update(
            {
                "claim_type": "numerical",
                "support_level": "direct_exact",
                "source_selector": json.dumps(selector, sort_keys=True, separators=(",", ":")),
                "source_row_number": str(row_number),
                "source_column": column,
                "exact_value": exact,
                "display_value": shown,
                "rounding_places": str(places),
                "display_scale": scale,
            }
        )
        claims.append(item)

    selection_report = ROUND2_DIR / "SELECTION_OBJECTIVE_SENSITIVITY.md"
    baseline_report = ROUND2_DIR / "PROBABILITY_BASELINE_REPORT.md"
    class_report = ROUND2_DIR / "PER_CLASS_EXTREME_CLASS_REPORT.md"
    plan = ROUND2_DIR / "ROUND2_PLAN.md"
    hr_report = ROUND2_DIR / "HR_MAPPING_SENSITIVITY_REPORT.md"
    methods = ROUND2_DIR / "METHOD_REPRODUCIBILITY_TABLES/METHODS_NOTES.md"
    novelty = ROUND2_DIR / "LITERATURE_V4/NOVELTY_BOUNDARY.md"

    narrative("R2N001", "Results — selection-objective sensitivity", "round2_selection", "Selection sensitivity is reported through separate leader changes, full-ordering changes, selected-candidate changes, and metric-specific effect magnitudes; it is not collapsed into one binary material-dependence verdict.", selection_report, "These are separate descriptive facts, not one binary material-dependence classification.", "Prespecified macro-F1 versus QWK selection under the common P3 folds and candidate registries.", "Each diagnostic remains separate and descriptive.", "Do not convert the diagnostics into a single binary claim or treat a lower-order permutation alone as material dependence.")
    narrative("R2N002", "Results — probability baselines", "round2_probability_baseline", "The outer-training empirical-prior comparator contextualizes probability metrics but does not establish calibration quality or deployment performance.", baseline_report, "The empirical-prior values provide a meaningful non-informative probability comparator.", "Fold-specific empirical priors derived only from each outer-training partition.", "The comparator is naive, sample-conditional, and not selected against outer-test data.", "Do not interpret ECE zero as perfect general calibration or the baseline as a deployable model.")
    narrative("R2N003", "Results and discussion — extreme class", "round2_extreme_class", "Strong aggregate ordinal scores can coexist with complete failure to identify rating 4, so per-class behavior must accompany aggregate rankings.", class_report, "producing rating-4 recall and F1 of zero", "P3 exactly-once OOF predictions for the four focal trained systems.", "This is observed metric behavior under one sample and protocol.", "Do not infer a causal explanation for the extreme-class failure.")
    narrative("R2N004", "Results — information-policy sensitivity", "round2_timing", "The P3→P4 and P4→P5 contrasts are central timing/information sensitivities under fixed and retuned estimands; neither is a causal feature-removal effect or prospective validation.", plan, "Move the P3→P4 and P4→P5 changes into the central Results/Discussion/Conclusion story", "Matched policy contrasts on the frozen INX folds.", "P4/P5 are prospective-plausibility sensitivities because feature timestamps are unavailable.", "Do not call P3 confirmed leakage or P4/P5 leakage-free prospective evidence.")
    narrative("R2N005", "Results — subgroup diagnostics", "round2_subgroup", "All six prespecified P3 subgroup attributes are reported at the support threshold with endpoints and exploratory intervals, including unfavorable cells.", plan, "Publish all six attributes, including the large department gap", "P3 maximum-minus-minimum gaps for macro-F1, QWK, and ordinal MAE at the declared support threshold.", "The table is descriptive and support-dependent.", "Do not claim fairness, discrimination, protected-class effects, or legal compliance.")
    narrative("R2N006", "Results — HR target construction", "round2_hr_mapping", "The retained three-class and raw-order four-class HR formulations are different estimands and are compared side by side without subtraction or improvement language.", hr_report, "These values describe different target estimands.", "Five repeated 5×5 nested-CV runs for raw and sigmoid XGBoost within each formulation.", "Target construction and metric choice jointly condition interpretation.", "Do not claim construct equivalence or cross-formulation improvement.")
    narrative("R2N007", "Results — HR CV design", "round2_hr_cv", "The 10×5 versus repeated-5×5 HR comparison is descriptive CV-design sensitivity, not an equivalence test or robustness certification.", hr_report, "The 11/14 result does not authorize an unqualified claim of robustness to CV design.", "Fourteen point-estimate versus empirical-range checks under the retained three-class formulation.", "Repeated ranges are not confidence intervals.", "Do not describe the design comparison as statistical equivalence or unqualified robustness.")
    narrative("R2N008", "Results — HR target aliases", "round2_hr_alias", "The primary target-alias comparison restricts canonical predictions and exclusion/refit predictions to the identical 309-row population; the separate 311→309 contrast is only a fit-free sample-removal effect.", hr_report, "The primary training/data-rule comparison is **canonical predictions restricted to 309 rows versus exclusion/refit predictions on those identical 309 rows**.", "Repetition-1 raw-XGBoost sensitivity with two prespecified audited disagreements removed.", "Only the matched 309↔309 contrast isolates training/data-rule sensitivity.", "Do not attribute the 311→309 change to refitting or claim equivalence.")
    narrative("R2N009", "Methods — reproducibility", "round2_methods", "The Methods contract fully enumerates feature governance, preprocessing, candidate grids, tie rules, fold identities, seed identities, and refit boundaries.", methods, "cumulative-threshold XGBoost independently fits binary XGBoost tasks for each observed training threshold", "Publication-facing S1–S3 tables bound to governed source hashes.", "The description does not add a new experiment or change the frozen estimands.", "Do not omit outer-test isolation, tie tolerance, or timestamp caveats.")
    narrative("R2N010", "Discussion — novelty positioning", "round2_literature", "The contribution is not the novelty of any single component, but their operational integration into a traceable evaluation contract.", novelty, "The contribution is not the novelty of any single component, but their operational integration into a traceable evaluation contract.", "Additive source-verified Round 2 core-method bibliography and bounded positioning statement.", "This is a positioning statement, not an exhaustive literature claim.", "Do not state world-first, first-ever, or exhaustive novelty.")
    narrative("R2N011", "Abstract — estimand boundary", "round2_abstract", "The abstract distinguishes single frozen canonical INX OOF values from HR means across five repetitions and does not imply that they share an estimand.", plan, "Use the explicit-design alternative", "Round 2 abstract reporting rule.", "INX and HR values must carry their respective design qualifiers.", "Do not pool, subtract, or directly rank the two dataset estimands.")
    narrative("R2N012", "Approval gate", "round2_gate", "The complete Round 2 claim digest requires explicit digest-specific user approval before any manuscript, bibliography, or reviewer-response rewrite.", plan, "No approval is inferred from a generic instruction to continue.", "Pre-rewrite governance gate for the Round 2 claim matrix.", "Approval must identify the exact new digest.", "Do not infer approval from a generic continue instruction.")

    # Explicit prohibited rows make the most consequential non-claims reviewable.
    for claim_id, proposed, anchor in (
        ("R2P001", "PROHIBITED: Treat selection sensitivity as one binary claim that ranking materially depends on the objective.", "These are separate descriptive facts, not one binary material-dependence classification."),
        ("R2P002", "PROHIBITED: Attribute the HR 311→309 contrast to model refitting.", "These changes must not be attributed to retraining."),
        ("R2P003", "PROHIBITED: Describe score differences between the three-class and four-class HR formulations as improvements.", "are not subtracted or labelled as improvements"),
        ("R2P004", "PROHIBITED: Describe P4 or P5 as timestamp-verified prospective validation.", "P4 is never timestamp-verified prospective evidence"),
        ("R2P005", "PROHIBITED: Treat subgroup gaps as proof of fairness or discrimination.", "The table is a descriptive diagnostic, not evidence of fairness"),
    ):
        source = selection_report if claim_id == "R2P001" else hr_report if claim_id in {"R2P002", "R2P003"} else plan
        item = base(claim_id, "Global boundary", "round2_prohibited", "prohibited", proposed, source, "Explicit Round 2 non-claim.", "This language is excluded from every manuscript component.", proposed.replace("PROHIBITED: ", ""))
        _require(anchor in source.read_text(encoding="utf-8"), f"Prohibition anchor drifted for {claim_id}.")
        item["evidence_anchor"] = anchor
        claims.append(item)

    derived = Path(derived_path)
    _require(derived_sha == _sha256(derived), "Derived-summary bytes drifted before claim construction.")
    numerical("R2E001", "Results — selection-objective sensitivity", "round2_selection", "Metric leaders changed for {value} of 9 reported metrics.", derived, {"summary_id": "selection_leader_changed"}, "value", 0, "Nine model-ranking metrics under macro-F1 and QWK selection.", "Leader change is distinct from full-ordering change.", "Do not convert this count into a binary materiality verdict.")
    numerical("R2E002", "Results — selection-objective sensitivity", "round2_selection", "The full model ordering changed for {value} of 9 reported metrics.", derived, {"summary_id": "selection_full_ordering_changed"}, "value", 0, "Nine model-ranking metrics under macro-F1 and QWK selection.", "A full-ordering change may be confined to lower ranks.", "Do not equate every permutation with material dependence.")
    numerical("R2E003", "Results — selection-objective sensitivity", "round2_selection", "The selected candidate changed in {value} of 60 model-by-fold selections.", derived, {"summary_id": "selection_candidate_changed"}, "value", 0, "Six trained models across ten shared outer folds.", "This count concerns candidate identities, not independent test repetitions.", "Do not interpret it as a probability of future selection change.")

    effects = ROUND2_DIR / "selection_objective_sensitivity/metric_effect_magnitudes.csv"
    numerical("R2E004", "Results — selection-objective sensitivity", "round2_selection", "With QWK selection, nominal XGBoost achieved QWK {value}.", effects, {"metric": "quadratic_weighted_kappa", "model": "xgboost"}, "qwk_selection_value", 4, "Exactly-once P3 OOF predictions under QWK-based candidate selection.", "Leadership is metric-, regime-, and protocol-specific.", "Do not call XGBoost universally best.")
    numerical("R2E005", "Results — selection-objective sensitivity", "round2_selection", "For cumulative-threshold XGBoost, QWK changed by {value} under QWK rather than macro-F1 selection.", effects, {"metric": "quadratic_weighted_kappa", "model": "cumulative_threshold_xgboost"}, "qwk_minus_macro_f1_selection", 4, "Signed within-model OOF score difference between the two selection regimes.", "This is a descriptive sensitivity without an uncertainty interval.", "Do not call the change statistically significant.")

    per_class = ROUND2_DIR / "selection_objective_sensitivity/per_class_metrics.csv"
    numerical("R2E006", "Results — extreme-class behavior", "round2_extreme_class", "Under canonical macro-F1 selection, Random Forest rating-4 recall was {value}.", per_class, {"selection_objective": "macro_f1", "dataset_key": "inx_primary", "model_name": "random_forest", "class_label": "4"}, "recall", 4, "Rating-4 class on the 1,200-row P3 exactly-once OOF sample.", "Aggregate QWK and MAE remain separate metrics.", "Do not imply adequate extreme-class detection from aggregate ordinal scores.")
    numerical("R2E007", "Results — extreme-class behavior", "round2_extreme_class", "Cumulative-threshold XGBoost rating-4 recall was {value} under macro-F1 selection.", per_class, {"selection_objective": "macro_f1", "dataset_key": "inx_primary", "model_name": "cumulative_threshold_xgboost", "class_label": "4"}, "recall", 4, "Rating-4 class on the 1,200-row P3 exactly-once OOF sample.", "This value is selection-regime specific.", "Do not attribute the value to a causal mechanism.")
    numerical("R2E008", "Results — extreme-class behavior", "round2_extreme_class", "Cumulative-threshold XGBoost rating-4 recall was {value} under QWK selection.", per_class, {"selection_objective": "qwk", "dataset_key": "inx_primary", "model_name": "cumulative_threshold_xgboost", "class_label": "4"}, "recall", 4, "Rating-4 class on the 1,200-row P3 exactly-once OOF sample.", "This value is selection-regime specific.", "Do not treat aggregate QWK improvement as uniform classwise improvement.")

    prior = ROUND2_DIR / "selection_objective_sensitivity/empirical_prior_metrics.csv"
    numerical("R2E009", "Results — probability baselines", "round2_probability_baseline", "The outer-training empirical-prior baseline had RPS {value}.", prior, {"model": "outer_training_empirical_prior", "metric": "ranked_probability_score"}, "value", 4, "Exactly-once fold-specific empirical-prior probabilities.", "Lower RPS is better; this baseline is descriptive.", "Do not call this a calibrated deployable model.")
    numerical("R2E010", "Results — probability baselines", "round2_probability_baseline", "The outer-training empirical-prior baseline had log loss {value}.", prior, {"model": "outer_training_empirical_prior", "metric": "nll_log_loss"}, "value", 4, "Exactly-once fold-specific empirical-prior probabilities.", "Lower log loss is better; this baseline is descriptive.", "Do not claim statistical superiority without an interval or test.")

    timing_specs = (
        ("R2E011", "p3_p4_fixed_macro_f1_delta", "Under the fixed schedule, P3→P4 changed macro-F1 by {value}."),
        ("R2E012", "p3_p4_retuned_macro_f1_delta", "Under independent retuning, P3→P4 changed macro-F1 by {value}."),
        ("R2E013", "p3_p4_fixed_quadratic_weighted_kappa_delta", "Under the fixed schedule, P3→P4 changed QWK by {value}."),
        ("R2E014", "p3_p4_retuned_quadratic_weighted_kappa_delta", "Under independent retuning, P3→P4 changed QWK by {value}."),
        ("R2E015", "p4_p5_fixed_quadratic_weighted_kappa_delta", "Under the fixed schedule, P4→P5 changed QWK by {value}."),
        ("R2E016", "p4_p5_retuned_quadratic_weighted_kappa_delta", "Under independent retuning, P4→P5 changed QWK by {value}."),
    )
    for claim_id, summary_id, sentence in timing_specs:
        numerical(claim_id, "Results — information-policy sensitivity", "round2_timing", sentence, derived, {"summary_id": summary_id}, "value", 4, "Adjacent policy contrast on the shared INX folds.", "Fixed and retuned estimands answer different descriptive questions.", "Do not interpret the contrast as causal or timestamp-verified prospective performance.")

    policy = Path("reports/research_log/major_revision_v3/phase1d_policy_retuning/headline_policy_comparison.csv")
    for claim_id, policy_id, label in (
        ("R2E017", "P0", "Information-Rich Diagnostic"),
        ("R2E018", "P3", "Primary Leakage-Aware"),
        ("R2E019", "P4", "Prospective-Plausibility"),
        ("R2E020", "P5", "Strict Proxy-Reduced Prospective-Plausibility"),
    ):
        numerical(claim_id, "Results — information-policy sensitivity", "round2_timing", f"Retuned macro-F1 for {label} was {{value}}.", policy, {"policy_id": policy_id}, "retuned_macro_f1", 4, "Policy-specific retuning on the shared INX folds.", "The label is manuscript-facing; timestamps remain unavailable.", "Do not describe P4/P5 as prospective validation.")

    subgroup = ROUND2_DIR / "SUBGROUP_MANUSCRIPT_SUMMARY.csv"
    metric_label = {"macro_f1": "macro-F1", "quadratic_weighted_kappa": "QWK", "ordinal_mae": "ordinal MAE"}
    for index, row in enumerate(_load_csv(subgroup), start=1):
        numerical(
            f"R2E{100 + index:03d}",
            "Results — subgroup diagnostics",
            "round2_subgroup",
            f"For {row['attribute']}, the P3 {metric_label[row['metric']]} maximum-minus-minimum gap was {{value}}.",
            subgroup,
            {"policy_id": row["policy_id"], "attribute": row["attribute"], "metric": row["metric"]},
            "gap_max_minus_min",
            4,
            "Groups meeting the declared n≥30 support rule, with endpoints retained in the source row.",
            "This is a descriptive support-dependent gap with exploratory intervals.",
            "Do not infer fairness, discrimination, or legal compliance.",
        )

    target = Path("reports/research_log/major_revision_v3/phase3b_data_quality/target_distribution.csv")
    support_specs = (
        ("R2E201", "canonical_mapped", "2", "The retained three-class HR label 2 support was {value}."),
        ("R2E202", "canonical_mapped", "3", "The retained three-class HR label 3 support was {value}."),
        ("R2E203", "canonical_mapped", "4", "The retained three-class HR label 4 support was {value}."),
        ("R2E204", "raw", "PIP", "The raw-order HR PIP support was {value}."),
        ("R2E205", "raw", "Needs Improvement", "The raw-order HR Needs Improvement support was {value}."),
        ("R2E206", "raw", "Fully Meets", "The raw-order HR Fully Meets support was {value}."),
        ("R2E207", "raw", "Exceeds", "The raw-order HR Exceeds support was {value}."),
    )
    for claim_id, view, label, sentence in support_specs:
        numerical(claim_id, "Methods — HR target construction", "round2_hr_mapping", sentence, target, {"dataset_key": "hrdataset_v14", "target_view": view, "target_label": label}, "count", 0, "Observed HRDataset_v14 target support.", "The mapping is a study estimand and does not establish equivalence with INX labels.", "Do not treat mapped categories as validated construct-equivalent outcomes.")

    variability = Path("reports/research_log/major_revision_v3/phase3a_hrdataset_sensitivity/variability_summary.csv")
    hr_metric_specs = (
        ("R2E208", "primary_three_class", "xgboost_raw", "macro_f1", "Three-class raw-XGBoost mean macro-F1 was {value}."),
        ("R2E209", "primary_three_class", "xgboost_raw", "quadratic_weighted_kappa", "Three-class raw-XGBoost mean QWK was {value}."),
        ("R2E210", "primary_three_class", "xgboost_sigmoid", "macro_f1", "Three-class sigmoid-XGBoost mean macro-F1 was {value}."),
        ("R2E211", "primary_three_class", "xgboost_sigmoid", "quadratic_weighted_kappa", "Three-class sigmoid-XGBoost mean QWK was {value}."),
        ("R2E212", "raw_order_four_class", "xgboost_raw", "macro_f1", "Four-class raw-XGBoost mean macro-F1 was {value}."),
        ("R2E213", "raw_order_four_class", "xgboost_raw", "quadratic_weighted_kappa", "Four-class raw-XGBoost mean QWK was {value}."),
        ("R2E214", "raw_order_four_class", "xgboost_sigmoid", "macro_f1", "Four-class sigmoid-XGBoost mean macro-F1 was {value}."),
        ("R2E215", "raw_order_four_class", "xgboost_sigmoid", "quadratic_weighted_kappa", "Four-class sigmoid-XGBoost mean QWK was {value}."),
    )
    for claim_id, formulation, system, metric, sentence in hr_metric_specs:
        numerical(claim_id, "Results — HR target construction", "round2_hr_mapping", sentence, variability, {"formulation_id": formulation, "system": system, "metric": metric}, "mean", 4, "Mean across five prespecified 5×5 nested-CV repetitions.", "The value is formulation-, system-, and metric-specific; ranges are not confidence intervals.", "Do not subtract across formulations or call either target construct-equivalent.")

    numerical("R2E216", "Results — HR CV design", "round2_hr_cv", "Of the 14 HR CV-design comparisons, {value} fell inside the repeated-run ranges.", derived, {"summary_id": "hr_cv_inside"}, "value", 0, "Fourteen retained-mapping point-estimate versus empirical-range checks.", "Observed repetition ranges are not confidence intervals.", "Do not call the result an equivalence test.")
    numerical("R2E217", "Results — HR CV design", "round2_hr_cv", "Of the 14 HR CV-design comparisons, {value} fell outside the repeated-run ranges.", derived, {"summary_id": "hr_cv_outside"}, "value", 0, "Fourteen retained-mapping point-estimate versus empirical-range checks.", "Observed repetition ranges are not confidence intervals.", "Do not claim unqualified robustness.")

    deltas = ROUND2_DIR / "hr_target_alias_sensitivity/comparison_deltas.csv"
    alias_specs = (
        ("R2E218", "matched_refit_effect", "macro_f1", "The matched 309↔309 exclusion/refit macro-F1 change was {value}."),
        ("R2E219", "matched_refit_effect", "quadratic_weighted_kappa", "The matched 309↔309 exclusion/refit QWK change was {value}."),
        ("R2E220", "matched_refit_effect", "ordinal_mae", "The matched 309↔309 exclusion/refit ordinal-MAE change was {value}."),
        ("R2E221", "matched_refit_effect", "ranked_probability_score", "The matched 309↔309 exclusion/refit RPS change was {value}."),
        ("R2E222", "matched_refit_effect", "nll_log_loss", "The matched 309↔309 exclusion/refit log-loss change was {value}."),
        ("R2E223", "matched_refit_effect", "multiclass_brier", "The matched 309↔309 exclusion/refit Brier-score change was {value}."),
        ("R2E224", "matched_refit_effect", "ece_confidence", "The matched 309↔309 exclusion/refit ECE change was {value}."),
        ("R2E225", "sample_removal_effect", "macro_f1", "The fit-free 311→309 sample-removal macro-F1 change was {value}."),
        ("R2E226", "sample_removal_effect", "quadratic_weighted_kappa", "The fit-free 311→309 sample-removal QWK change was {value}."),
        ("R2E227", "sample_removal_effect", "ordinal_mae", "The fit-free 311→309 sample-removal ordinal-MAE change was {value}."),
        ("R2E228", "sample_removal_effect", "ranked_probability_score", "The fit-free 311→309 sample-removal RPS change was {value}."),
    )
    for claim_id, comparison, metric, sentence in alias_specs:
        numerical(claim_id, "Results — HR target aliases", "round2_hr_alias", sentence, deltas, {"comparison_id": comparison, "metric": metric}, "right_minus_left", 6, "Repetition-1 raw-XGBoost target-alias sensitivity.", "Matched refit and fit-free sample-removal contrasts are distinct estimands.", "Do not attribute sample removal to refitting or claim equivalence.")
    numerical("R2E229", "Results — HR target aliases", "round2_hr_alias", "Candidate selection changed in {value} of 5 HR exclusion/refit outer folds.", derived, {"summary_id": "hr_alias_candidate_changed"}, "value", 0, "Repetition-1 raw-XGBoost exclusion/refit candidate schedules.", "The count is descriptive and fold-specific.", "Do not interpret it as population uncertainty.")

    return claims


def _claim_fields() -> list[str]:
    return [
        "claim_id", "prior_claim_id", "disposition", "active_for_rewrite", "replacement_claim_ids",
        "manuscript_section", "component", "claim_type", "support_level", "approval_status", "proposed_claim",
        "source_path", "source_sha256", "source_selector", "source_row_number", "source_column", "exact_value",
        "display_value", "rounding_places", "display_scale", "evidence_anchor", "evidence_scope",
        "mandatory_qualifier", "prohibited_overclaim",
    ]


def _validate_claims(claims: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(row["claim_id"]) for row in claims]
    _require(len(ids) == len(set(ids)), "Claim ids are duplicated.")
    _require(set(row["disposition"] for row in claims) == DISPOSITIONS, "All five dispositions must be represented.")
    _require(sum(row["prior_claim_id"] != "" for row in claims) == 45, "Historical claim coverage drifted.")
    for row in claims:
        _require(row["disposition"] in DISPOSITIONS, f"Invalid disposition for {row['claim_id']}.")
        _require(row["approval_status"] in {"pending_user_approval", "approved"}, "Claim approval state is invalid.")
        source = Path(str(row["source_path"]))
        _require(source.is_file(), f"Claim source is absent: {source.as_posix()}.")
        _require(_sha256(source) == row["source_sha256"], f"Source hash drifted for {row['claim_id']}.")
        if row["claim_type"] == "numerical":
            selector = json.loads(str(row["source_selector"]))
            row_number, source_row = _source_row(source, selector)
            _require(str(row_number) == str(row["source_row_number"]), f"Source row drifted for {row['claim_id']}.")
            _require(source_row[row["source_column"]] == row["exact_value"], f"Exact value drifted for {row['claim_id']}.")
            expected = _display(str(row["exact_value"]), int(row["rounding_places"]), str(row["display_scale"]))
            _require(expected == row["display_value"], f"Display value drifted for {row['claim_id']}.")
            _require(expected in row["proposed_claim"], f"Displayed value is absent for {row['claim_id']}.")
        else:
            anchor = str(row["evidence_anchor"])
            _require(len(anchor) >= 20 and anchor in source.read_text(encoding="utf-8"), f"Narrative anchor drifted for {row['claim_id']}.")


def _claim_digest(claims: Sequence[Mapping[str, Any]]) -> str:
    canonical = [{field: row.get(field, "") for field in _claim_fields() if field != "approval_status"} for row in claims]
    return _sha256_bytes(_json_bytes({"schema_version": 2, "claims": canonical}))


def _matrix_md(claims: Sequence[Mapping[str, Any]], digest: str) -> str:
    approved = all(row["approval_status"] == "approved" for row in claims)
    status = (
        "Status: **APPROVED FOR ROUND 2 REWRITE**. These active claims are the sole scientific boundary for manuscript, bibliography, and reviewer-response drafting."
        if approved
        else "Status: **PENDING DIGEST-SPECIFIC USER APPROVAL**. No manuscript, bibliography, or reviewer-response edit is authorized."
    )
    lines = [
        "# Round 2 Claim Matrix",
        "",
        status,
        "",
        f"Claim-set SHA-256: `{digest}`",
        "",
        "Historical rows marked `modified` or `superseded` are retained for audit but are not active rewrite language. `prohibited` rows are explicit non-claims.",
        "",
    ]
    for disposition in ("retained", "modified", "superseded", "new", "prohibited"):
        selected = [row for row in claims if row["disposition"] == disposition]
        lines.extend([f"## {disposition.title()} ({len(selected)})", ""])
        for row in selected:
            source_detail = f"row {row['source_row_number']}, `{row['source_column']}`" if row["claim_type"] == "numerical" else "text anchor"
            lines.extend(
                [
                    f"### {row['claim_id']}",
                    "",
                    str(row["proposed_claim"]),
                    "",
                    f"- Active for rewrite: `{str(row['active_for_rewrite']).lower()}`",
                    f"- Evidence: `{row['source_path']}` ({source_detail}; SHA-256 `{row['source_sha256']}`).",
                    f"- Required qualifier: {row['mandatory_qualifier']}",
                    f"- Prohibited overclaim: {row['prohibited_overclaim']}",
                    "",
                ]
            )
    return "\n".join(lines)


def _source_register(claims: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for claim in claims:
        grouped.setdefault(str(claim["source_path"]), []).append(claim)
    rows = []
    for source_path, source_claims in sorted(grouped.items()):
        path = Path(source_path)
        rows.append(
            {
                "source_path": source_path,
                "source_sha256": _sha256(path),
                "source_size_bytes": path.stat().st_size,
                "source_format": path.suffix.lower().lstrip("."),
                "claim_count": len(source_claims),
                "active_claim_count": sum(row["active_for_rewrite"] == "True" for row in source_claims),
                "claim_ids": " | ".join(str(row["claim_id"]) for row in source_claims),
            }
        )
    return rows


def _validate_historical_phase5b() -> int:
    manifest = json.loads((HISTORICAL_MANUSCRIPT_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    _require(manifest.get("claim_set_sha256") == "1664b188df14135d3b3de8642de3d080c25a3fbadc440615c2bd04b2f14ddabe", "Historical Phase 5B digest drifted.")
    for item in manifest["files"]:
        path = Path(item["path"])
        _require(path.is_file(), f"Historical Phase 5B file is absent: {path.as_posix()}.")
        _require(_sha256(path) == item["sha256"], f"Historical Phase 5B bytes drifted: {path.as_posix()}.")
    return len(manifest["files"])


def _cross_audit(claims: Sequence[Mapping[str, Any]], digest: str) -> str:
    from src.governance.claim_matrix_v3 import validate_claim_matrix_package_v3

    legacy_receipt = validate_claim_matrix_package_v3(HISTORICAL_CLAIM_DIR)
    _require(legacy_receipt["claim_set_sha256"] == "1664b188df14135d3b3de8642de3d080c25a3fbadc440615c2bd04b2f14ddabe", "Historical Phase 5A digest drifted.")
    phase5b_file_count = _validate_historical_phase5b()
    phase5b_rows = {row["claim_id"]: row for row in _load_csv(HISTORICAL_MANUSCRIPT_DIR / "RESULTS_COMPARISON.csv")}
    legacy_numeric = [row for row in claims if row["prior_claim_id"] and row["claim_type"] == "numerical"]
    _require(len(legacy_numeric) == len(phase5b_rows) == 32, "Historical numerical comparison coverage drifted.")
    for row in legacy_numeric:
        prior = phase5b_rows[row["claim_id"]]
        for field in ("proposed_claim", "exact_value", "display_value", "source_path", "source_sha256"):
            _require(row[field] == prior[field], f"Phase 5B cross-source mismatch: {row['claim_id']} {field}.")
    active_numeric = [row for row in claims if row["active_for_rewrite"] == "True" and row["claim_type"] == "numerical"]
    return "\n".join(
        [
            "# Round 2 Cross-Source Consistency Audit",
            "",
            "Status: **PASSED**",
            "",
            f"- Round 2 claim-set SHA-256: `{digest}`",
            f"- Historical Phase 5A claim digest verified: `{legacy_receipt['claim_set_sha256']}`",
            f"- Historical Phase 5A numerical claims matched to Phase 5B comparison rows: `{len(legacy_numeric)}`",
            f"- Historical Phase 5B manifest-bound files independently rehashed: `{phase5b_file_count}`",
            f"- Active Round 2 numerical claims independently resolved to one exact source row/value/hash: `{len(active_numeric)}`",
            "- Round 2 selection, timing, subgroup, HR mapping/CV, and matched target-alias sources were checked through the same row-selector validator.",
            "- Historical Phase 5A and Phase 5B packages were read-only inputs and their tracked bytes were not changed.",
            "- Manuscript, bibliography, and reviewer-response editing remain unauthorized pending explicit approval of the digest above.",
            "",
        ]
    )


def _build_outputs(generation_commit: str, config_path: Path = CONFIG) -> tuple[dict[str, bytes], dict[str, Any]]:
    config = _load_config(config_path)
    derived_rows = _derived_rows()
    derived_fields = list(derived_rows[0])
    derived_bytes = _csv_bytes(derived_fields, derived_rows)
    temporary_dir = ROUND2_DIR / f".round2-claim-source-{uuid.uuid4().hex}"
    temporary_dir.mkdir()
    try:
        derived_path = temporary_dir / "DERIVED_SUMMARIES.csv"
        derived_path.write_bytes(derived_bytes)
        claims = _legacy_claims(config) + _new_claims(derived_path, _sha256_bytes(derived_bytes))
        claim_approval_status = "approved" if config["user_approval"]["status"] == "approved" else "pending_user_approval"
        for claim in claims:
            claim["approval_status"] = claim_approval_status
        # Replace the temporary path with its final governed path without changing source bytes.
        final_derived = OUTPUT_DIR / "DERIVED_SUMMARIES.csv"
        for claim in claims:
            if claim["source_path"] == derived_path.as_posix():
                claim["source_path"] = final_derived.as_posix()
        _validate_claims_except_derived_path(claims, final_derived, derived_bytes)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    digest = _claim_digest(claims)
    source_rows = _source_register_with_virtual_derived(claims, final_derived, derived_bytes)
    numerical_rows = [
        {
            "claim_id": row["claim_id"],
            "prior_claim_id": row["prior_claim_id"],
            "disposition": row["disposition"],
            "component": row["component"],
            "proposed_claim": row["proposed_claim"],
            "exact_value": row["exact_value"],
            "display_value": row["display_value"],
            "evidence_scope": row["evidence_scope"],
            "mandatory_qualifier": row["mandatory_qualifier"],
            "prohibited_overclaim": row["prohibited_overclaim"],
            "source_path": row["source_path"],
            "source_sha256": row["source_sha256"],
            "source_selector": row["source_selector"],
            "source_row_number": row["source_row_number"],
            "source_column": row["source_column"],
        }
        for row in claims
        if row["active_for_rewrite"] == "True" and row["claim_type"] == "numerical"
    ]
    audit = _cross_audit_virtual(claims, digest, final_derived, derived_bytes)
    approval = config["user_approval"]
    if approval["status"] == "approved":
        _require(approval["approved_claim_set_sha256"] == digest, "Approved digest does not match the claim set.")
    else:
        _require(all(approval.get(field) is None for field in ("approved_by", "approved_at_utc", "approved_claim_set_sha256")), "Pending approval fields must be null.")
    counts = {value: sum(row["disposition"] == value for row in claims) for value in sorted(DISPOSITIONS)}
    provenance = {
        "schema_version": 2,
        "package_kind": "round2_claim_matrix",
        "generation_commit": generation_commit,
        "claim_set_sha256": digest,
        "historical_claim_set_sha256": config["historical_claim_set_sha256"],
        "claim_count": len(claims),
        "numerical_claim_count": sum(row["claim_type"] == "numerical" for row in claims),
        "narrative_claim_count": sum(row["claim_type"] == "narrative" for row in claims),
        "active_claim_count": sum(row["active_for_rewrite"] == "True" for row in claims),
        "disposition_counts": counts,
        "source_file_count": len(source_rows),
        "approval_status": approval["status"],
        "manuscript_editing_authorized": config["controls"]["manuscript_editing_authorized"],
        "bibliography_editing_authorized": config["controls"]["bibliography_editing_authorized"],
        "reviewer_response_editing_authorized": config["controls"]["reviewer_response_editing_authorized"],
        "network_calls": 0,
        "paid_api_calls": 0,
        "status": "passed_pending_user_approval" if approval["status"] == "pending" else "passed_approved_for_rewrite",
    }
    approval_boundary = (
        "The recorded digest-specific approval authorizes manuscript, bibliography, and reviewer-response drafting under the active rows in this package. It does not authorize release, tag, DOI, raw-data publication, or resolution of licence, ethics, and author-declaration blockers."
        if approval["status"] == "approved"
        else "The package does not authorize manuscript, bibliography, or reviewer-response edits. A generic instruction to continue is not approval of this digest."
    )
    readme = "\n".join(
        [
            "# Round 2 Claim-Matrix Package",
            "",
            f"Claim-set SHA-256: `{digest}`",
            f"Generation commit: `{generation_commit}`",
            f"Status: `{provenance['status']}`",
            "",
            "This additive package preserves the historical Phase 5A claim bytes, classifies every historical and Round 2 item, binds every numerical claim to one exact aggregate source row/value/hash, and binds every narrative or prohibited item to a hashed text anchor.",
            "",
            approval_boundary,
            "",
        ]
    )
    approval_request = (
        "\n".join(
            [
                "# Round 2 Digest-Specific Approval Record",
                "",
                f"Claim-set SHA-256: `{digest}`",
                f"Claims: `{len(claims)}`; active rewrite claims: `{provenance['active_claim_count']}`",
                "Current decision: `approved`",
                f"Approved by: `{approval['approved_by']}`",
                f"Approved at UTC: `{approval['approved_at_utc']}`",
                "",
                str(approval["approval_statement"]),
                "",
            ]
        )
        if approval["status"] == "approved"
        else "\n".join(
            [
                "# Round 2 Digest-Specific Approval Request",
                "",
                f"Claim-set SHA-256: `{digest}`",
                f"Claims: `{len(claims)}`; active rewrite claims: `{provenance['active_claim_count']}`",
                "Current decision: `pending`",
                "",
                "Please explicitly approve or reject this exact digest as the sole Round 2 claim boundary. Until that decision is recorded, `manuscript/mdpi_information/main.md`, `main.tex`, `references.bib`, and the Round 2 reviewer response remain outside the authorized edit scope.",
                "",
            ]
        )
    )
    approval_record = {
        "schema_version": 2,
        "decision": approval["status"],
        "claim_set_sha256": digest,
        "approved_claim_set_sha256": approval["approved_claim_set_sha256"],
        "approved_by": approval["approved_by"],
        "approved_at_utc": approval["approved_at_utc"],
        "approval_statement": approval["approval_statement"],
        "manuscript_editing_authorized": config["controls"]["manuscript_editing_authorized"],
    }
    outputs = {
        "README.md": readme.encode("utf-8"),
        "ROUND2_CLAIM_MATRIX.csv": _csv_bytes(_claim_fields(), claims),
        "ROUND2_CLAIM_MATRIX.md": _matrix_md(claims, digest).encode("utf-8"),
        "ROUND2_SOURCE_REGISTER.csv": _csv_bytes(list(source_rows[0]), source_rows),
        "RESULTS_COMPARISON_ROUND2.csv": _csv_bytes(list(numerical_rows[0]), numerical_rows),
        "DERIVED_SUMMARIES.csv": derived_bytes,
        "ROUND2_CROSS_SOURCE_AUDIT.md": audit.encode("utf-8"),
        "ROUND2_CLAIM_DIGEST.txt": (digest + "\n").encode("utf-8"),
        "APPROVAL_REQUEST.md": approval_request.encode("utf-8"),
        "approval_record.json": _json_bytes(approval_record),
        "provenance_receipt.json": _json_bytes(provenance),
    }
    return outputs, provenance


def _validate_claims_except_derived_path(claims: Sequence[Mapping[str, Any]], derived_path: Path, derived_bytes: bytes) -> None:
    ids = [str(row["claim_id"]) for row in claims]
    _require(len(ids) == len(set(ids)), "Claim ids are duplicated.")
    _require(set(row["disposition"] for row in claims) == DISPOSITIONS, "All five dispositions must be represented.")
    _require(sum(row["prior_claim_id"] != "" for row in claims) == 45, "Historical claim coverage drifted.")
    virtual_rows = list(csv.DictReader(io.StringIO(derived_bytes.decode("utf-8"))))
    for row in claims:
        source = Path(str(row["source_path"]))
        is_derived = source == derived_path
        _require(row["source_sha256"] == (_sha256_bytes(derived_bytes) if is_derived else _sha256(source)), f"Source hash drifted for {row['claim_id']}.")
        if row["claim_type"] == "numerical":
            selector = json.loads(str(row["source_selector"]))
            source_rows = virtual_rows if is_derived else _load_csv(source)
            matches = [(i, item) for i, item in enumerate(source_rows, 2) if all(item.get(k) == v for k, v in selector.items())]
            _require(len(matches) == 1, f"Numerical selector drifted for {row['claim_id']}.")
            row_number, source_row = matches[0]
            _require(str(row_number) == str(row["source_row_number"]), f"Source row drifted for {row['claim_id']}.")
            _require(source_row[row["source_column"]] == row["exact_value"], f"Exact value drifted for {row['claim_id']}.")
            expected = _display(str(row["exact_value"]), int(row["rounding_places"]), str(row["display_scale"]))
            _require(expected == row["display_value"] and expected in row["proposed_claim"], f"Display drifted for {row['claim_id']}.")
        else:
            _require(row["evidence_anchor"] in source.read_text(encoding="utf-8"), f"Narrative anchor drifted for {row['claim_id']}.")


def _source_register_with_virtual_derived(claims: Sequence[Mapping[str, Any]], derived_path: Path, derived_bytes: bytes) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for claim in claims:
        grouped.setdefault(str(claim["source_path"]), []).append(claim)
    rows = []
    for source_path, source_claims in sorted(grouped.items()):
        path = Path(source_path)
        is_derived = path == derived_path
        rows.append(
            {
                "source_path": source_path,
                "source_sha256": _sha256_bytes(derived_bytes) if is_derived else _sha256(path),
                "source_size_bytes": len(derived_bytes) if is_derived else path.stat().st_size,
                "source_format": path.suffix.lower().lstrip("."),
                "claim_count": len(source_claims),
                "active_claim_count": sum(row["active_for_rewrite"] == "True" for row in source_claims),
                "claim_ids": " | ".join(str(row["claim_id"]) for row in source_claims),
            }
        )
    return rows


def _cross_audit_virtual(claims: Sequence[Mapping[str, Any]], digest: str, derived_path: Path, derived_bytes: bytes) -> str:
    # Derived bytes are already independently replayed by _validate_claims_except_derived_path.
    _require(_sha256_bytes(derived_bytes) == next(row["source_sha256"] for row in claims if Path(row["source_path"]) == derived_path), "Derived source hash mismatch.")
    return _cross_audit(claims, digest)


def _manifest(outputs: Mapping[str, bytes], generation_commit: str, config_sha: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "package_kind": "round2_claim_matrix",
        "generation_commit": generation_commit,
        "config_sha256": config_sha,
        "file_count_excluding_manifest": len(outputs),
        "files": [
            {"path": name, "sha256": _sha256_bytes(content), "size_bytes": len(content)}
            for name, content in sorted(outputs.items())
        ],
    }


def export_round2_claim_matrix(
    output_dir: Path = OUTPUT_DIR,
    *,
    generation_commit: str | None = None,
    require_clean_git: bool = True,
    replace_existing: bool = False,
) -> dict[str, Any]:
    if require_clean_git:
        _require(_git_is_clean(), "Round 2 claim export requires a clean Git worktree.")
    commit = generation_commit or _git_head()
    _require(GIT_SHA_RE.fullmatch(commit) is not None or commit == "test", "Invalid generation commit.")
    outputs, provenance = _build_outputs(commit)
    outputs[MANIFEST_NAME] = _json_bytes(_manifest(outputs, commit, _sha256(CONFIG)))
    output = Path(output_dir)
    _require(replace_existing or not output.exists(), f"Output already exists: {output.as_posix()}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for name, content in outputs.items():
            path = staging / name
            with path.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        if output.exists():
            backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
            output.replace(backup)
            try:
                staging.replace(output)
            except BaseException:
                backup.replace(output)
                raise
            shutil.rmtree(backup)
        else:
            staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for name in MIRRORED_FILES:
        destination = ROUND2_DIR / name
        destination.write_bytes((output / name).read_bytes())
    return validate_round2_claim_matrix(output)


def validate_round2_claim_matrix(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    _require(output.is_dir(), f"Round 2 claim package is absent: {output.as_posix()}.")
    names = {path.name for path in output.iterdir() if path.is_file()}
    _require(names == EXPECTED_OUTPUTS, "Round 2 claim package inventory drifted.")
    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    commit = manifest["generation_commit"]
    expected, provenance = _build_outputs(commit)
    expected_manifest = _manifest(expected, commit, _sha256(CONFIG))
    _require(manifest == expected_manifest, "Round 2 manifest content drifted.")
    for name, content in expected.items():
        _require((output / name).read_bytes() == content, f"Round 2 derived file drifted: {name}.")
    for name in MIRRORED_FILES:
        _require((ROUND2_DIR / name).read_bytes() == (output / name).read_bytes(), f"Round 2 top-level mirror drifted: {name}.")
    return {
        "status": provenance["status"],
        "package_dir": output.as_posix(),
        "generation_commit": commit,
        "claim_set_sha256": provenance["claim_set_sha256"],
        "claim_count": provenance["claim_count"],
        "active_claim_count": provenance["active_claim_count"],
        "numerical_claim_count": provenance["numerical_claim_count"],
        "narrative_claim_count": provenance["narrative_claim_count"],
        "disposition_counts": provenance["disposition_counts"],
        "source_file_count": provenance["source_file_count"],
        "manuscript_editing_authorized": provenance["manuscript_editing_authorized"],
        "bibliography_editing_authorized": provenance["bibliography_editing_authorized"],
        "reviewer_response_editing_authorized": provenance["reviewer_response_editing_authorized"],
        "paid_api_calls": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--generation-commit")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = (
        validate_round2_claim_matrix(args.output_dir)
        if args.validate_only
        else export_round2_claim_matrix(
            args.output_dir,
            generation_commit=args.generation_commit,
            require_clean_git=not args.allow_dirty,
            replace_existing=args.replace_output,
        )
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
