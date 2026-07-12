"""Fail-closed evidence readiness checks for governed LLM evaluation.

The governed explainer is allowed to interpret only the evidence supplied to it.
Consequently, text-compliance scores are not meaningful evidence of readiness when
the case record itself is incomplete.  This module keeps those two questions
separate and provides the gate used before any paid/real LLM request loop.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.core.io_utils import write_json
from src.core.reporting import markdown_table, write_markdown
from src.governance.manuscript_contract import (
    DEFAULT_CONFIG_PATH,
    forbidden_feature_mentions,
    load_manuscript_config,
    primary_policy_name,
)
from src.llm.evidence_schema import CompleteCaseEvidence
from src.utils.config_loader import PROJECT_ROOT
from src.utils.experiment_registry import utc_now_iso


PREFLIGHT_SCHEMA_VERSION = 1
COMPLETE_CASE_STRATUM = "complete_case_evidence"


class EvidencePreflightError(RuntimeError):
    """Raised before real API execution when selected evidence is not ready."""


@dataclass(frozen=True)
class CaseEvidenceReadiness:
    """Machine-readable validation result for one selected case."""

    case_id: str
    dataset_name: str
    feature_policy: str
    complete: bool
    missing_fields: tuple[str, ...]
    invalid_fields: tuple[str, ...]
    forbidden_features: Mapping[str, tuple[str, ...]]
    readiness_status: str = COMPLETE_CASE_STRATUM
    evidence_stratum: str = COMPLETE_CASE_STRATUM

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["missing_fields"] = list(self.missing_fields)
        payload["invalid_fields"] = list(self.invalid_fields)
        payload["forbidden_features"] = {
            name: list(values) for name, values in self.forbidden_features.items()
        }
        return payload


def _nested_value(payload: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _is_missing(value: Any, *, require_non_empty: bool = True) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return require_non_empty and not value.strip()
    if isinstance(value, (list, tuple, set, frozenset, Mapping)):
        return require_non_empty and len(value) == 0
    if isinstance(value, float) and not math.isfinite(value):
        return True
    return False


def _feature_names(payload: Mapping[str, Any]) -> list[str]:
    shap = payload.get("shap")
    if not isinstance(shap, Mapping):
        return []
    names: list[str] = []
    for key in ("top_positive_features", "top_negative_features"):
        rows = shap.get(key, [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping) and row.get("feature") is not None:
                    names.append(str(row["feature"]))
    for key in ("grouped_shap_values", "class_specific_shap_values"):
        values = shap.get(key, {})
        if isinstance(values, Mapping):
            names.extend(str(name) for name in values)
    counterfactual = payload.get("counterfactual")
    if isinstance(counterfactual, Mapping):
        changed = counterfactual.get("changed_features", [])
        if isinstance(changed, list):
            names.extend(str(name) for name in changed)
    return names


def _source_file_errors(sources: Any, *, project_root: Path) -> list[str]:
    """Identify source entries that claim a concrete artifact which is absent.

    Free-text provenance notes remain permitted.  Entries ending in a common
    artifact suffix are treated as file references and must exist.
    """

    if not isinstance(sources, list):
        return []
    errors: list[str] = []
    artifact_suffixes = {".csv", ".json", ".jsonl", ".md", ".parquet", ".yaml", ".yml"}
    for index, source in enumerate(sources):
        if not isinstance(source, str) or not source.strip():
            errors.append(f"evidence_sources[{index}]")
            continue
        candidate = Path(source)
        if candidate.suffix.lower() not in artifact_suffixes:
            continue
        resolved = candidate if candidate.is_absolute() else project_root / candidate
        if not resolved.is_file():
            errors.append(f"evidence_sources[{index}]:missing_file:{source}")
    return errors


def validate_complete_case_evidence(
    evidence: CompleteCaseEvidence,
    *,
    canonical_config: Mapping[str, Any] | None = None,
    canonical_config_path: str | Path = DEFAULT_CONFIG_PATH,
    project_root: str | Path = PROJECT_ROOT,
) -> CaseEvidenceReadiness:
    """Validate required case fields and canonical-primary feature exclusions.

    Optional scientific quantities may legitimately be unavailable (for example,
    no valid counterfactual may exist), so completeness is based on presence of a
    governed section and its interpretation/warning fields rather than requiring a
    fabricated numeric value.
    """

    if not isinstance(evidence, CompleteCaseEvidence):
        raise TypeError("evidence must be a CompleteCaseEvidence instance")
    payload = evidence.to_dict()
    required_non_empty = (
        "prediction.case_id",
        "prediction.dataset_name",
        "prediction.predicted_class",
        "prediction.true_class",
        "prediction.class_probabilities",
        "prediction.confidence",
        "prediction.model_name",
        "prediction.feature_policy",
        "prediction.leakage_safe_status",
        "shap",
        "shap.grouped_shap_values",
        "shap.class_specific_shap_values",
        "shap.shap_stability_summary",
        "shap.explanation_stability_warning",
        "fairness",
        "fairness.audited_groups",
        "fairness.proxy_risk_warnings",
        "calibration",
        "calibration.calibration_warning",
        "calibration.probability_interpretation",
        "counterfactual",
        "counterfactual.counterfactual_mode",
        "counterfactual.actionability_label",
        "counterfactual.warning",
        "leakage",
        "leakage.feature_policy",
        "leakage.leakage_warning",
        "governance",
        "governance.intended_use",
        "governance.prohibited_use",
        "governance.model_card_summary",
        "governance.deployment_status",
        "governance.required_warnings",
        "evidence_sources",
    )
    missing = sorted(
        path
        for path in required_non_empty
        if _is_missing(_nested_value(payload, path), require_non_empty=True)
    )

    invalid: list[str] = []
    probabilities = _nested_value(payload, "prediction.class_probabilities")
    if isinstance(probabilities, Mapping) and probabilities:
        for label, value in probabilities.items():
            try:
                probability = float(value)
            except (TypeError, ValueError):
                invalid.append(f"prediction.class_probabilities.{label}:not_numeric")
                continue
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                invalid.append(f"prediction.class_probabilities.{label}:outside_[0,1]")
        predicted = _nested_value(payload, "prediction.predicted_class")
        if predicted is not None and str(predicted) not in {str(label) for label in probabilities}:
            invalid.append("prediction.class_probabilities:predicted_class_missing")

    confidence = _nested_value(payload, "prediction.confidence")
    if confidence is not None:
        try:
            numeric_confidence = float(confidence)
        except (TypeError, ValueError):
            invalid.append("prediction.confidence:not_numeric")
        else:
            if not math.isfinite(numeric_confidence) or not 0.0 <= numeric_confidence <= 1.0:
                invalid.append("prediction.confidence:outside_[0,1]")

    top_positive = _nested_value(payload, "shap.top_positive_features")
    top_negative = _nested_value(payload, "shap.top_negative_features")
    if payload.get("shap") is not None:
        if not isinstance(top_positive, list) or not isinstance(top_negative, list):
            invalid.append("shap.top_feature_lists:not_lists")
        elif not top_positive and not top_negative:
            missing.append("shap.top_positive_features|shap.top_negative_features")

    calibration_values = [
        _nested_value(payload, "calibration.log_loss"),
        _nested_value(payload, "calibration.brier_score"),
        _nested_value(payload, "calibration.expected_calibration_error"),
        _nested_value(payload, "calibration.ece"),
    ]
    if payload.get("calibration") is not None and all(_is_missing(value) for value in calibration_values):
        missing.append("calibration.at_least_one_reliability_metric")

    governance = payload.get("governance")
    if isinstance(governance, Mapping) and governance.get("human_review_required") is not True:
        invalid.append("governance.human_review_required:must_be_true")

    leakage = payload.get("leakage")
    prediction = payload.get("prediction")
    if isinstance(leakage, Mapping) and isinstance(prediction, Mapping):
        if leakage.get("feature_policy") != prediction.get("feature_policy"):
            invalid.append("leakage.feature_policy:does_not_match_prediction")

    invalid.extend(
        _source_file_errors(payload.get("evidence_sources"), project_root=Path(project_root).resolve())
    )

    config = dict(canonical_config) if canonical_config is not None else load_manuscript_config(canonical_config_path)
    policy = str(_nested_value(payload, "prediction.feature_policy") or "")
    forbidden: dict[str, tuple[str, ...]] = {}
    if policy == primary_policy_name(config):
        mentions = forbidden_feature_mentions(_feature_names(payload), config)
        forbidden = {name: tuple(values) for name, values in mentions.items()}

    missing_tuple = tuple(sorted(set(missing)))
    invalid_tuple = tuple(sorted(set(invalid)))
    complete = not missing_tuple and not invalid_tuple and not forbidden
    return CaseEvidenceReadiness(
        case_id=str(_nested_value(payload, "prediction.case_id") or ""),
        dataset_name=str(_nested_value(payload, "prediction.dataset_name") or ""),
        feature_policy=policy,
        complete=complete,
        missing_fields=missing_tuple,
        invalid_fields=invalid_tuple,
        forbidden_features=forbidden,
        readiness_status=COMPLETE_CASE_STRATUM if complete else "incomplete_evidence",
        evidence_stratum=COMPLETE_CASE_STRATUM if complete else "incomplete_evidence",
    )


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence_level: float = 0.95,
) -> tuple[float | None, float | None]:
    """Return a Wilson score interval without requiring SciPy."""

    if isinstance(successes, bool) or isinstance(total, bool):
        raise TypeError("successes and total must be integer counts")
    if int(successes) != successes or int(total) != total:
        raise TypeError("successes and total must be integer counts")
    successes = int(successes)
    total = int(total)
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson counts require 0 <= successes <= total")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if total == 0:
        return None, None
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = successes / total
    denominator = 1.0 + (z * z) / total
    center = (proportion + (z * z) / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((proportion * (1.0 - proportion) / total) + (z * z) / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def readiness_distribution(
    statuses: Iterable[str],
    *,
    confidence_level: float = 0.95,
) -> list[dict[str, Any]]:
    """Summarize evidence-readiness strata with one-vs-rest Wilson intervals."""

    values = [str(status) for status in statuses]
    counts = Counter(values)
    total = len(values)
    rows: list[dict[str, Any]] = []
    for status, count in sorted(counts.items()):
        ci_low, ci_high = wilson_interval(count, total, confidence_level=confidence_level)
        rows.append(
            {
                "readiness_status": status,
                "count": count,
                "denominator": total,
                "rate": count / total if total else None,
                "wilson_ci_low": ci_low,
                "wilson_ci_high": ci_high,
                "confidence_level": confidence_level,
            }
        )
    return rows


def _missing_field_counts(results: Sequence[CaseEvidenceReadiness]) -> tuple[dict[str, int], dict[str, int]]:
    exact: Counter[str] = Counter()
    category_cases: dict[str, set[str]] = {}
    for result in results:
        fields = list(result.missing_fields) + list(result.invalid_fields)
        fields.extend(f"forbidden_feature.{name}" for name in result.forbidden_features)
        for field in fields:
            exact[field] += 1
            category = field.split(".", 1)[0].split(":", 1)[0]
            category_cases.setdefault(category, set()).add(result.case_id)
    category_counts = {name: len(case_ids) for name, case_ids in sorted(category_cases.items())}
    return dict(sorted(exact.items())), category_counts


def _resolve_missing_stratum(
    raw_config: Any,
    incomplete_case_ids: set[str],
) -> tuple[set[str], dict[str, Any], list[str]]:
    if raw_config is None:
        return set(), {"configured": False, "enabled": False}, []
    if not isinstance(raw_config, Mapping):
        return set(), {"configured": True, "enabled": False}, [
            "missing_evidence_stratum must be a mapping"
        ]
    enabled = raw_config.get("enabled") is True
    summary = {
        "configured": True,
        "enabled": enabled,
        "name": raw_config.get("name"),
        "include_in_primary_compliance_metrics": raw_config.get(
            "include_in_primary_compliance_metrics"
        ),
    }
    if not enabled:
        return set(), summary, []

    errors: list[str] = []
    name = raw_config.get("name")
    if not isinstance(name, str) or not name.strip() or name == COMPLETE_CASE_STRATUM:
        errors.append("enabled missing_evidence_stratum requires a distinct non-empty name")
    if raw_config.get("include_in_primary_compliance_metrics") is not False:
        errors.append(
            "missing-evidence cases must set include_in_primary_compliance_metrics=false"
        )
    raw_case_ids = raw_config.get("case_ids", [])
    if not isinstance(raw_case_ids, list) or any(not isinstance(value, str) for value in raw_case_ids):
        errors.append("missing_evidence_stratum.case_ids must be a list of case-id strings")
        raw_case_ids = []
    declared = set(raw_case_ids)
    if raw_config.get("include_all_incomplete_cases") is True:
        declared.update(incomplete_case_ids)
    if not declared and incomplete_case_ids:
        errors.append(
            "enabled missing_evidence_stratum must declare case_ids or include_all_incomplete_cases=true"
        )
    unknown = declared - incomplete_case_ids
    if unknown:
        errors.append(f"missing_evidence_stratum declares non-incomplete/unknown cases: {sorted(unknown)}")
    allowed = declared & incomplete_case_ids
    summary["declared_case_ids"] = sorted(declared)
    summary["allowed_incomplete_case_ids"] = sorted(allowed)
    return allowed, summary, errors


def _diagnostic_category(result: CaseEvidenceReadiness, evidence_note: str) -> str:
    if result.complete:
        return "complete_case_evidence"
    if result.forbidden_features:
        return "canonical_policy_contract_violation"
    if any("missing_file" in field for field in result.invalid_fields):
        return "referenced_artifact_missing_generator_or_packaging_defect"
    note = evidence_note.casefold()
    if result.missing_fields and (
        "reason-code" in note or "local shap" in note or "local reason" in note
    ):
        return "local_evidence_generation_coverage_gap"
    if result.invalid_fields:
        return "evidence_schema_or_generator_defect"
    return "required_evidence_missing_unclassified"


def build_evidence_preflight_report(
    evidence_items: Sequence[Mapping[str, Any] | CompleteCaseEvidence],
    *,
    run_id: str,
    run_mode: str,
    missing_evidence_stratum: Mapping[str, Any] | None = None,
    canonical_config_path: str | Path = DEFAULT_CONFIG_PATH,
    confidence_level: float = 0.95,
    requested_case_count: int | None = None,
) -> dict[str, Any]:
    """Build the full selected-case readiness report used by the real-run gate."""

    normalized_mode = str(run_mode).lower()
    if normalized_mode not in {"dry_run", "real"}:
        raise ValueError("run_mode must be 'dry_run' or 'real'")
    canonical_config = load_manuscript_config(canonical_config_path)
    results: list[CaseEvidenceReadiness] = []
    evidence_notes: list[str] = []
    for item in evidence_items:
        evidence = item if isinstance(item, CompleteCaseEvidence) else item.get("evidence")
        if not isinstance(evidence, CompleteCaseEvidence):
            raise TypeError("Each evidence item must be CompleteCaseEvidence or contain an 'evidence' value")
        evidence_notes.append(
            "" if isinstance(item, CompleteCaseEvidence) else str(item.get("notes", ""))
        )
        results.append(
            validate_complete_case_evidence(
                evidence,
                canonical_config=canonical_config,
                canonical_config_path=canonical_config_path,
            )
        )

    case_ids = [result.case_id for result in results]
    duplicate_ids = sorted(case_id for case_id, count in Counter(case_ids).items() if count > 1)
    incomplete_ids = {result.case_id for result in results if not result.complete}
    allowed_incomplete, stratum_summary, stratum_errors = _resolve_missing_stratum(
        missing_evidence_stratum,
        incomplete_ids,
    )
    configured_name = str(stratum_summary.get("name") or "missing_evidence_stratum")

    case_rows: list[dict[str, Any]] = []
    statuses: list[str] = []
    blocked_ids: list[str] = []
    for result, evidence_note in zip(results, evidence_notes):
        row = result.to_dict()
        if result.complete:
            status = COMPLETE_CASE_STRATUM
            stratum = COMPLETE_CASE_STRATUM
            eligible = True
        elif result.case_id in allowed_incomplete and not stratum_errors:
            status = "incomplete_evidence_explicit_stratum"
            stratum = configured_name
            eligible = True
        else:
            status = "incomplete_evidence_blocking"
            stratum = "unassigned_incomplete_evidence"
            eligible = False
            blocked_ids.append(result.case_id)
        row.update(
            {
                "readiness_status": status,
                "evidence_stratum": stratum,
                "eligible_for_real_api": eligible,
                "diagnostic_category": _diagnostic_category(result, evidence_note),
                "evidence_notes": evidence_note,
            }
        )
        statuses.append(status)
        case_rows.append(row)

    complete_count = sum(result.complete for result in results)
    selected_count = len(results)
    requested = selected_count if requested_case_count is None else int(requested_case_count)
    if requested < 0:
        raise ValueError("requested_case_count must be non-negative")
    if selected_count > requested:
        raise ValueError("requested_case_count cannot be smaller than the selected evidence count")
    incomplete_count = selected_count - complete_count
    selection_shortfall = max(0, requested - selected_count)
    exact_counts, category_counts = _missing_field_counts(results)
    complete_ci = wilson_interval(complete_count, requested, confidence_level=confidence_level)
    blocking_reasons = list(stratum_errors)
    if requested == 0:
        blocking_reasons.append("no evaluation cases were requested")
    if selected_count != requested:
        blocking_reasons.append(
            f"selected case count {selected_count} does not equal configured request {requested}"
        )
    if duplicate_ids:
        blocking_reasons.append(f"duplicate selected case IDs: {duplicate_ids}")
    if blocked_ids:
        blocking_reasons.append(
            f"incomplete evidence is not assigned to an explicit separate stratum: {sorted(blocked_ids)}"
        )
    evidence_gate_passed = not blocking_reasons
    real_api_allowed = normalized_mode == "real" and evidence_gate_passed
    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_mode": normalized_mode,
        "generated_at": utc_now_iso(),
        "cases_requested": requested,
        "cases_selected": selected_count,
        "cases_complete": complete_count,
        "cases_incomplete": incomplete_count,
        "cases_not_selected": selection_shortfall,
        "cases_in_missing_evidence_stratum": len(allowed_incomplete),
        "cases_blocked_from_real_api": len(blocked_ids),
        "complete_case_rate": complete_count / requested if requested else None,
        "complete_case_rate_wilson_ci_low": complete_ci[0],
        "complete_case_rate_wilson_ci_high": complete_ci[1],
        "confidence_level": confidence_level,
        "missing_fields": exact_counts,
        "missing_fields_by_category": category_counts,
        "duplicate_case_ids": duplicate_ids,
        "missing_evidence_stratum": stratum_summary,
        "readiness_distribution": readiness_distribution(
            statuses, confidence_level=confidence_level
        ),
        "diagnostic_distribution": dict(
            sorted(Counter(row["diagnostic_category"] for row in case_rows).items())
        ),
        "evidence_gate_passed": evidence_gate_passed,
        "preflight_passed": evidence_gate_passed,
        "api_execution_allowed": real_api_allowed,
        "real_api_execution_allowed": real_api_allowed,
        "blocking_reasons": blocking_reasons,
        "cases": case_rows,
    }
    return report


def enforce_real_llm_preflight(report: Mapping[str, Any]) -> None:
    """Raise before client construction/request loops if a real run is blocked."""

    if str(report.get("run_mode", "")).lower() != "real":
        return
    if report.get("real_api_execution_allowed") is True:
        return
    reasons = report.get("blocking_reasons") or ["evidence readiness gate did not pass"]
    raise EvidencePreflightError(
        "Real LLM evaluation blocked by CompleteCaseEvidence preflight: "
        + "; ".join(str(reason) for reason in reasons)
    )


def _safe_run_directory_name(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("._") or "unnamed_run"


def write_evidence_preflight_report(
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write versioned JSON, case CSV, and human-readable preflight artifacts."""

    root = Path(output_dir) / "preflight" / _safe_run_directory_name(str(report.get("run_id", "")))
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "llm_evidence_preflight.json"
    csv_path = root / "llm_evidence_preflight_cases.csv"
    md_path = root / "llm_evidence_preflight.md"
    write_json(json_path, dict(report))

    case_rows = []
    for raw in report.get("cases", []):
        row = dict(raw)
        for key in ("missing_fields", "invalid_fields", "forbidden_features"):
            row[key] = json.dumps(row.get(key), sort_keys=True, ensure_ascii=True)
        case_rows.append(row)
    pd.DataFrame(case_rows).to_csv(csv_path, index=False)

    readiness_df = pd.DataFrame(report.get("readiness_distribution", []))
    missing_df = pd.DataFrame(
        [
            {"field_or_error": field, "case_count": count}
            for field, count in report.get("missing_fields", {}).items()
        ]
    )
    diagnostic_df = pd.DataFrame(
        [
            {"diagnostic_category": category, "case_count": count}
            for category, count in report.get("diagnostic_distribution", {}).items()
        ]
    )
    lines = [
        "# CompleteCaseEvidence Preflight",
        "",
        f"Run ID: `{report.get('run_id')}`",
        f"Run mode: `{report.get('run_mode')}`",
        f"Cases requested: {report.get('cases_requested')}",
        f"Cases selected: {report.get('cases_selected')}",
        f"Cases complete: {report.get('cases_complete')}",
        f"Cases incomplete: {report.get('cases_incomplete')}",
        f"Cases not selected: {report.get('cases_not_selected')}",
        f"Real API execution allowed: `{report.get('real_api_execution_allowed')}`",
        "",
        "## Readiness Distribution",
        "",
        *markdown_table(readiness_df),
        "",
        "## Missing or Invalid Evidence",
        "",
        *markdown_table(missing_df),
        "",
        "## Diagnostic Classification",
        "",
        *markdown_table(diagnostic_df),
        "",
        "## Blocking Reasons",
        "",
    ]
    blocking = report.get("blocking_reasons", [])
    lines.extend([f"- {reason}" for reason in blocking] or ["No evidence-readiness blockers."])
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "Text faithfulness/compliance and evidence completeness are separate outcomes. "
            "A perfect compliance rate cannot repair missing case evidence, and a finite all-pass "
            "sample does not establish a zero population failure probability.",
        ]
    )
    write_markdown(md_path, lines)
    return {"preflight_json": json_path, "preflight_cases_csv": csv_path, "preflight_md": md_path}


__all__ = [
    "COMPLETE_CASE_STRATUM",
    "CaseEvidenceReadiness",
    "EvidencePreflightError",
    "build_evidence_preflight_report",
    "enforce_real_llm_preflight",
    "readiness_distribution",
    "validate_complete_case_evidence",
    "wilson_interval",
    "write_evidence_preflight_report",
]
