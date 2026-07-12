"""Build canonical, case-complete evidence for offline LLM readiness checks.

This stage is deliberately provider-free.  It binds case evidence to manuscript
run/config identities, selects the predeclared INX and HRDataset_v14 cases, and
writes a deterministic completeness preflight.  It does *not* construct an LLM
client and cannot execute a paid API request.

The input directories are outputs of canonical experiment stages.  INX evidence
uses nested-calibration OOF predictions, class-specific OOF grouped SHAP values,
support-aware fairness/proxy results, and OOF counterfactual results.  External
case evidence is accepted only when local SHAP rows identify their OOF fold and
agree with the OOF prediction for the case; legacy full-fit external SHAP output
is therefore rejected rather than mixed with OOF predictions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.core.io_utils import ensure_dir, safe_float, write_json, write_jsonl
from src.data.external_adapters import (
    audit_attribute_columns,
    build_feature_columns,
    load_external_dataset,
    role_columns,
)
from src.experiments.final_evidence_common import align_proba, predict_labels_from_proba
from src.experiments.final_shap_stability import (
    get_group_mapping,
    group_shap_values,
    normalize_shap_values,
)
from src.experiments.leakage_safe_cv import (
    LabelEncodedXGBClassifier,
    infer_columns,
    make_preprocessor,
)
from src.governance.manuscript_contract import (
    DEFAULT_CONFIG_PATH,
    canonical_config_hash,
    load_manuscript_config,
    primary_excluded_features,
    primary_policy_name,
    sha256_file,
)
from src.llm.evidence_preflight import build_evidence_preflight_report
from src.llm.evidence_schema import (
    CalibrationEvidence,
    CompleteCaseEvidence,
    CounterfactualEvidence,
    FairnessEvidence,
    GovernanceEvidence,
    LeakageEvidence,
    PredictionEvidence,
    ShapEvidence,
)


class ManuscriptCaseEvidenceError(RuntimeError):
    """Raised when canonical case evidence cannot be assembled consistently."""


@dataclass(frozen=True)
class StageInputs:
    """Canonical stage roots consumed by :func:`run`."""

    inx_shap_dir: Path
    calibration_dir: Path
    counterfactual_dir: Path
    fairness_dir: Path
    policy_dir: Path
    hrdataset_dir: Path

    def directories(self) -> tuple[Path, ...]:
        return (
            self.inx_shap_dir,
            self.calibration_dir,
            self.counterfactual_dir,
            self.fairness_dir,
            self.policy_dir,
            self.hrdataset_dir,
        )


@dataclass(frozen=True)
class SelectedCase:
    sample_index: int
    sampling_reason: str


def _required_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise ManuscriptCaseEvidenceError(f"Required canonical evidence artifact is missing: {path}")
    return path


def _read_required_csv(root: Path, relative: str) -> tuple[Path, pd.DataFrame]:
    path = _required_file(root, relative)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas supplies the detail
        raise ManuscriptCaseEvidenceError(f"Cannot read canonical evidence artifact {path}: {exc}") from exc
    if frame.empty:
        raise ManuscriptCaseEvidenceError(f"Canonical evidence artifact is empty: {path}")
    return path, frame


def _read_required_csv_schema_alias(
    root: Path,
    canonical_name: str,
    *aliases: str,
) -> tuple[Path, pd.DataFrame]:
    """Read one explicitly versioned schema name without silently choosing ambiguity."""

    candidates = [root / name for name in (canonical_name, *aliases) if (root / name).is_file()]
    if not candidates:
        expected = ", ".join((canonical_name, *aliases))
        raise ManuscriptCaseEvidenceError(
            f"Required canonical evidence artifact is missing under {root}; expected one of: {expected}."
        )
    if len(candidates) > 1:
        raise ManuscriptCaseEvidenceError(
            f"Ambiguous canonical evidence schema under {root}; multiple aliases exist: {candidates}."
        )
    path = candidates[0]
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas supplies the detail
        raise ManuscriptCaseEvidenceError(f"Cannot read canonical evidence artifact {path}: {exc}") from exc
    if frame.empty:
        raise ManuscriptCaseEvidenceError(f"Canonical evidence artifact is empty: {path}")
    return path, frame


def _assert_columns(frame: pd.DataFrame, columns: Iterable[str], *, path: Path) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ManuscriptCaseEvidenceError(f"Artifact {path} is missing required columns: {missing}")


def _assert_identity(
    frame: pd.DataFrame,
    *,
    path: Path,
    run_id: str,
    config_hash: str,
    required: bool,
) -> None:
    """Reject incompatible canonical tables while allowing legacy-free externals.

    The canonical INX stages always carry both fields.  External files are bound
    through hashes in this stage's metadata and may additionally carry identities;
    if either identity column is present it must be complete and compatible.
    """

    has_run = "run_id" in frame.columns
    has_config = "config_hash" in frame.columns
    if required and not (has_run and has_config):
        raise ManuscriptCaseEvidenceError(
            f"Canonical artifact {path} must contain run_id and config_hash columns."
        )
    if has_run != has_config:
        raise ManuscriptCaseEvidenceError(
            f"Artifact {path} contains only one of run_id/config_hash; identity is incomplete."
        )
    if not has_run:
        return
    observed_runs = set(frame["run_id"].dropna().astype(str).unique())
    observed_hashes = set(frame["config_hash"].dropna().astype(str).unique())
    if observed_runs != {run_id} or observed_hashes != {config_hash}:
        raise ManuscriptCaseEvidenceError(
            f"Artifact identity mismatch for {path}: run_id={sorted(observed_runs)}, "
            f"config_hash={sorted(observed_hashes)}; expected {run_id}/{config_hash}."
        )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _record_dict(row: pd.Series) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in row.to_dict().items()}


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in frame.columns if str(column).startswith("prob_class_")]
    if not columns:
        raise ManuscriptCaseEvidenceError("Prediction artifact contains no prob_class_<label> columns.")
    return sorted(columns)


def _prepare_predictions(frame: pd.DataFrame, *, path: Path) -> pd.DataFrame:
    _assert_columns(frame, ["sample_index", "y_true", "y_pred"], path=path)
    probabilities = _probability_columns(frame)
    out = frame.copy()
    out["sample_index"] = out["sample_index"].astype(int)
    out["confidence"] = out[probabilities].apply(pd.to_numeric, errors="coerce").max(axis=1)
    out["correct"] = out["y_true"].astype(int) == out["y_pred"].astype(int)
    if out["sample_index"].duplicated().any():
        duplicates = sorted(out.loc[out["sample_index"].duplicated(False), "sample_index"].unique())
        raise ManuscriptCaseEvidenceError(f"Prediction artifact has duplicate case rows: {duplicates[:10]}")
    return out


def _select_cases(
    predictions: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
    representative: pd.DataFrame | None = None,
    local_values: pd.DataFrame | None = None,
) -> list[SelectedCase]:
    """Deterministically select risk-aware cases and fill across target strata."""

    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if sample_size > len(predictions):
        raise ManuscriptCaseEvidenceError(
            f"Requested {sample_size} cases but only {len(predictions)} prediction rows are available."
        )
    available = set(predictions["sample_index"].astype(int))
    selected: dict[int, list[str]] = {}

    def add(index: Any, reason: str) -> None:
        if len(selected) >= sample_size:
            return
        try:
            sample_index = int(index)
        except (TypeError, ValueError):
            return
        if sample_index not in available:
            return
        if sample_index in selected:
            if reason not in selected[sample_index]:
                selected[sample_index].append(reason)
        else:
            selected[sample_index] = [reason]

    if representative is not None and not representative.empty and "sample_index" in representative:
        reason_column = next(
            (column for column in ("case_type", "case", "sampling_reason") if column in representative),
            None,
        )
        for row in representative.itertuples(index=False):
            reason = str(getattr(row, reason_column)) if reason_column else "canonical_representative"
            add(getattr(row, "sample_index"), reason)

    correct = predictions[predictions["correct"]]
    incorrect = predictions[~predictions["correct"]]
    if not correct.empty:
        add(correct.sort_values(["confidence", "sample_index"], ascending=[False, True]).iloc[0]["sample_index"], "correct_high_confidence")
        add(correct.sort_values(["confidence", "sample_index"], ascending=[True, True]).iloc[0]["sample_index"], "correct_low_confidence")
    if not incorrect.empty:
        add(incorrect.sort_values(["confidence", "sample_index"], ascending=[False, True]).iloc[0]["sample_index"], "misclassification_high_confidence")
        add(incorrect.sort_values(["confidence", "sample_index"], ascending=[True, True]).iloc[0]["sample_index"], "misclassification_low_confidence")
    uncertain = predictions[predictions["confidence"] < 0.60]
    if not uncertain.empty:
        add(uncertain.sort_values(["confidence", "sample_index"]).iloc[0]["sample_index"], "low_confidence_uncertain")
    for label in sorted(predictions["y_true"].dropna().astype(int).unique()):
        group = predictions[predictions["y_true"].astype(int) == label]
        if not group.empty:
            add(group.sort_values(["confidence", "sample_index"], ascending=[False, True]).iloc[0]["sample_index"], f"true_class_{label}_coverage")

    if local_values is not None and not local_values.empty:
        _assert_columns(local_values, ["sample_index", "abs_grouped_shap_value"], path=Path("local SHAP table"))
        concentration = (
            local_values.assign(
                abs_value=pd.to_numeric(local_values["abs_grouped_shap_value"], errors="coerce")
            )
            .groupby("sample_index")["abs_value"]
            .agg(["max", "sum"])
            .reset_index()
        )
        concentration["concentration"] = concentration["max"] / concentration["sum"].replace(0.0, pd.NA)
        concentration = concentration.dropna(subset=["concentration"])
        if not concentration.empty:
            add(
                concentration.sort_values(["concentration", "sample_index"], ascending=[False, True]).iloc[0]["sample_index"],
                "strong_shap_attribution_concentration",
            )

    remaining = predictions[~predictions["sample_index"].isin(selected)].copy()
    if len(selected) < sample_size and not remaining.empty:
        # A seeded priority per class gives a reproducible stratified round-robin fill.
        remaining = remaining.sample(frac=1.0, random_state=seed)
        groups = {
            int(label): group["sample_index"].astype(int).tolist()
            for label, group in remaining.groupby(remaining["y_true"].astype(int), sort=True)
        }
        while len(selected) < sample_size and any(groups.values()):
            for label in sorted(groups):
                if groups[label] and len(selected) < sample_size:
                    add(groups[label].pop(0), f"stratified_seeded_fill_class_{label}")

    if len(selected) != sample_size:
        raise ManuscriptCaseEvidenceError(
            f"Risk-aware selection produced {len(selected)} cases; expected {sample_size}."
        )
    return [
        SelectedCase(sample_index=index, sampling_reason=";".join(reasons))
        for index, reasons in selected.items()
    ]


def _top_local_shap(
    local_values: pd.DataFrame,
    *,
    sample_index: int,
    predicted_class: int,
    top_k: int,
    external: bool = False,
    expected_fold: int | None = None,
) -> ShapEvidence:
    _assert_columns(
        local_values,
        ["sample_index", "feature", "grouped_shap_value", "abs_grouped_shap_value"],
        path=Path("local grouped SHAP values"),
    )
    case = local_values[local_values["sample_index"].astype(int) == sample_index].copy()
    if "class_label" in case.columns:
        case = case[case["class_label"].astype(int) == predicted_class]
    if case.empty:
        raise ManuscriptCaseEvidenceError(
            f"No class-specific local SHAP evidence for case {sample_index}, class {predicted_class}."
        )
    if external:
        if "fold" not in case.columns:
            raise ManuscriptCaseEvidenceError(
                "External local SHAP lacks an OOF fold column. Legacy full-fit SHAP must not be "
                "combined with OOF predictions."
            )
        if "predicted_class" in case.columns:
            observed = set(case["predicted_class"].dropna().astype(int).unique())
            if observed != {predicted_class}:
                raise ManuscriptCaseEvidenceError(
                    f"External local SHAP/prediction mismatch for case {sample_index}: "
                    f"SHAP model classes={sorted(observed)}, OOF prediction={predicted_class}."
                )
        if expected_fold is not None and set(case["fold"].dropna().astype(int).unique()) != {expected_fold}:
            raise ManuscriptCaseEvidenceError(
                f"External local SHAP fold mismatch for case {sample_index}; expected fold {expected_fold}."
            )
        if "evaluation_scope" in case.columns and not case["evaluation_scope"].astype(str).str.contains(
            "out_of_fold|oof", case=False, regex=True
        ).all():
            raise ManuscriptCaseEvidenceError(
                f"External local SHAP has a non-OOF evaluation scope for case {sample_index}."
            )
        if "prediction_identity_verified" in case.columns and not case[
            "prediction_identity_verified"
        ].map(_as_bool).all():
            raise ManuscriptCaseEvidenceError(
                f"External local SHAP prediction identity is unverified for case {sample_index}."
            )
    case["grouped_shap_value"] = pd.to_numeric(case["grouped_shap_value"], errors="coerce")
    case["abs_grouped_shap_value"] = pd.to_numeric(case["abs_grouped_shap_value"], errors="coerce")
    case = case.dropna(subset=["feature", "grouped_shap_value", "abs_grouped_shap_value"])
    case = case.sort_values(["abs_grouped_shap_value", "feature"], ascending=[False, True])
    if case.empty:
        raise ManuscriptCaseEvidenceError(f"Local SHAP evidence contains no finite values for case {sample_index}.")

    def rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in frame.head(top_k).itertuples(index=False):
            item: dict[str, Any] = {
                "feature": str(row.feature),
                "grouped_shap_value": float(row.grouped_shap_value),
            }
            for key in ("control_type", "sensitive_or_proxy", "leakage_risk", "governance_notes"):
                if hasattr(row, key):
                    item[key] = _jsonable(getattr(row, key))
            result.append(item)
        return result

    positive = case[case["grouped_shap_value"] > 0]
    negative = case[case["grouped_shap_value"] < 0]
    if positive.empty and negative.empty:
        # Preserve a genuine zero attribution without pretending it supports/opposes.
        neutral = rows(case.head(1))
        neutral[0]["direction"] = "neutral_zero_attribution"
        positive_rows, negative_rows = neutral, []
    else:
        positive_rows, negative_rows = rows(positive), rows(negative)
    grouped = {
        str(row.feature): float(row.grouped_shap_value)
        for row in case.itertuples(index=False)
    }
    return ShapEvidence(
        top_positive_features=positive_rows,
        top_negative_features=negative_rows,
        grouped_shap_values=grouped,
        class_specific_shap_values=dict(grouped),
        shap_stability_summary={},  # supplied by the calling dataset-specific builder
        explanation_stability_warning=(
            "SHAP values are model attributions, not causal effects; remaining features may "
            "encode sensitive or organisational proxies."
        ),
    )


def _fairness_table_path(root: Path) -> Path:
    candidates = (
        "manuscript_fairness_proxy_table.csv",
        "fairness_proxy_manuscript_table.csv",
        "fairness_proxy_table.csv",
        "manuscript_fairness_table.csv",
    )
    for name in candidates:
        path = root / name
        if path.is_file():
            return path
    for path in sorted(root.glob("*.csv")):
        try:
            columns = set(pd.read_csv(path, nrows=2).columns)
        except Exception:
            continue
        if {"attribute", "metric"}.issubset(columns) and columns.intersection(
            {"gap", "point_estimate", "max_gap"}
        ):
            return path
    raise ManuscriptCaseEvidenceError(
        f"No manuscript fairness/proxy table with attribute, metric, and gap fields found in {root}."
    )


def _fairness_evidence(
    frame: pd.DataFrame,
    *,
    policy: str,
    audited_groups: Sequence[str],
) -> FairnessEvidence:
    subset = frame.copy()
    for column in ("policy", "feature_policy", "feature_set"):
        if column in subset.columns:
            candidate = subset[subset[column].astype(str) == policy]
            if not candidate.empty:
                subset = candidate
            break
    if subset.empty:
        raise ManuscriptCaseEvidenceError(f"Fairness/proxy table has no rows for policy {policy}.")
    gap_column = next((name for name in ("gap", "point_estimate", "max_gap") if name in subset), None)
    if gap_column is None:
        raise ManuscriptCaseEvidenceError("Fairness/proxy table does not contain a disparity gap column.")
    gaps: dict[str, Any] = {}
    intervals: dict[str, Any] = {}
    for row in subset.itertuples(index=False):
        attribute = str(getattr(row, "attribute", "unknown"))
        metric = str(getattr(row, "metric", "unknown"))
        class_label = getattr(row, "class_label", None)
        class_part = "overall" if class_label is None or pd.isna(class_label) else f"class_{class_label}"
        key = f"{attribute}.{metric}.{class_part}"
        gaps[key] = safe_float(getattr(row, gap_column))
        if hasattr(row, "ci_low") or hasattr(row, "ci_high"):
            intervals[key] = {
                "ci_low": safe_float(getattr(row, "ci_low", None)),
                "ci_high": safe_float(getattr(row, "ci_high", None)),
                "valid_bootstrap_samples": _jsonable(getattr(row, "valid_bootstrap_samples", None)),
            }
    warnings: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        category = str(row.get("interpretation_category", "")).casefold()
        minimum = safe_float(row.get("minimum_subgroup_support", row.get("min_group_support")))
        if any(token in category for token in ("insufficient", "unstable", "low_support", "limited")) or (
            minimum is not None and minimum < 30
        ):
            warnings.append(_record_dict(row))
    return FairnessEvidence(
        audited_groups=list(audited_groups),
        subgroup_metrics={"n_report_rows": int(len(subset))},
        disparity_gaps=gaps,
        bootstrap_ci=intervals,
        low_support_warnings=warnings,
        proxy_risk_warnings=[
            "Subgroup gaps are descriptive support-aware audit evidence, not proof of discrimination or fairness.",
            "Department reconstructability is proxy-risk evidence, not proof of causal or discriminatory model use.",
            "Removing sensitive or group variables does not establish fairness.",
        ],
    )


def _calibration_method(
    comparison: pd.DataFrame,
    metadata_path: Path | None,
) -> str:
    if metadata_path is not None and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        selected = metadata.get("selected_method")
        if selected:
            return str(selected)
    if "selected" in comparison.columns:
        selected_rows = comparison[comparison["selected"].map(_as_bool)]
        if len(selected_rows) == 1:
            return str(selected_rows.iloc[0]["method"])
    if "selection_rank_sum" in comparison.columns:
        return str(comparison.sort_values(["selection_rank_sum", "method"]).iloc[0]["method"])
    raise ManuscriptCaseEvidenceError("Canonical calibration comparison does not identify one selected method.")


def _governance_evidence(*, dataset_name: str, role: str, allowed_claim: str) -> GovernanceEvidence:
    return GovernanceEvidence(
        intended_use=(
            f"Research-grade decision support and governance review for {dataset_name}; {allowed_claim}"
        ),
        prohibited_use=(
            "No autonomous hiring, firing, promotion, compensation, disciplinary, ranking, "
            "or individual employee-evaluation decisions."
        ),
        model_card_summary=(
            f"Dataset role: {role}. The performance model is XGBoost; any later LLM may only "
            "interpret this structured evidence and is never the predictor or decision maker."
        ),
        deployment_status="research_only_decision_support_not_deployed",
        required_warnings=[
            "Human review is required.",
            "SHAP is attribution, not causality.",
            "Counterfactuals are model scenarios, not employee prescriptions.",
            "Removing sensitive or group variables does not establish fairness.",
            "Not for autonomous HR decisions.",
        ],
        human_review_required=True,
    )


def _sources(*paths: Path) -> list[str]:
    return [str(path.resolve()) for path in paths]


def _build_inx_cases(
    *,
    inputs: StageInputs,
    settings: Mapping[str, Any],
    run_id: str,
    config_hash: str,
    sample_size: int,
    seed: int,
) -> tuple[list[CompleteCaseEvidence], list[dict[str, Any]], set[Path]]:
    calibration_predictions_path, calibration_predictions = _read_required_csv(
        inputs.calibration_dir, "calibration_predictions.csv"
    )
    calibration_comparison_path, calibration_comparison = _read_required_csv(
        inputs.calibration_dir, "calibration_method_comparison.csv"
    )
    calibration_metadata_path = inputs.calibration_dir / "calibration_metadata.json"
    local_path, local_values = _read_required_csv(inputs.inx_shap_dir, "local_grouped_shap_values.csv")
    stability_path, stability = _read_required_csv(inputs.inx_shap_dir, "shap_stability_summary.csv")
    representative_path, representative = _read_required_csv(inputs.inx_shap_dir, "representative_cases.csv")
    cf_cases_path, cf_cases = _read_required_csv(inputs.counterfactual_dir, "actionability_by_case.csv")
    cf_summary_path, cf_summary = _read_required_csv(inputs.counterfactual_dir, "actionability_summary.csv")
    policy_summary_path, policy_summary = _read_required_csv(inputs.policy_dir, "policy_summary.csv")
    lsi_path, lsi = _read_required_csv(inputs.policy_dir, "leakage_sensitivity_index.csv")
    fairness_path = _fairness_table_path(inputs.fairness_dir)
    fairness_table = pd.read_csv(fairness_path)
    if fairness_table.empty:
        raise ManuscriptCaseEvidenceError(f"Canonical fairness table is empty: {fairness_path}")

    bound_tables = (
        (calibration_predictions_path, calibration_predictions),
        (calibration_comparison_path, calibration_comparison),
        (local_path, local_values),
        (stability_path, stability),
        (representative_path, representative),
        (cf_cases_path, cf_cases),
        (cf_summary_path, cf_summary),
        (policy_summary_path, policy_summary),
        (lsi_path, lsi),
        (fairness_path, fairness_table),
    )
    for path, frame in bound_tables:
        _assert_identity(frame, path=path, run_id=run_id, config_hash=config_hash, required=True)

    policy = primary_policy_name({"manuscript_final": dict(settings)})
    selected_method = _calibration_method(
        calibration_comparison,
        calibration_metadata_path if calibration_metadata_path.is_file() else None,
    )
    predictions = calibration_predictions[
        calibration_predictions["method"].astype(str) == selected_method
    ].copy()
    if "policy" in predictions:
        predictions = predictions[predictions["policy"].astype(str) == policy]
    predictions = _prepare_predictions(predictions, path=calibration_predictions_path)
    selected = _select_cases(
        predictions,
        sample_size=sample_size,
        seed=seed,
        representative=representative,
        local_values=local_values,
    )

    stability_top_k = int(settings.get("shap", {}).get("stability", {}).get("top_k", 10))
    stable_rows = stability[stability["top_k"].astype(int) == stability_top_k]
    if len(stable_rows) != 1:
        raise ManuscriptCaseEvidenceError(
            f"Expected one SHAP stability row for top_k={stability_top_k}; found {len(stable_rows)}."
        )
    stability_payload = {
        key: _jsonable(value)
        for key, value in _record_dict(stable_rows.iloc[0]).items()
        if key not in {"run_id", "config_hash", "policy"}
    }
    calibration_rows = calibration_comparison[
        calibration_comparison["method"].astype(str) == selected_method
    ]
    if len(calibration_rows) != 1:
        raise ManuscriptCaseEvidenceError(
            f"Expected one calibration comparison row for {selected_method}; found {len(calibration_rows)}."
        )
    calibration_row = calibration_rows.iloc[0]
    fairness = _fairness_evidence(
        fairness_table,
        policy=policy,
        audited_groups=settings.get("governance_fields", {}).get("fairness_audit_fields", []),
    )
    summary_policy = policy_summary[policy_summary["policy"].astype(str) == policy]
    summary_full = policy_summary[
        policy_summary["policy"].astype(str) == "full_feature_upper_bound"
    ]
    lsi_policy = lsi[
        (lsi["policy"].astype(str) == policy) & (lsi["metric"].astype(str) == "macro_f1")
    ]
    if len(summary_policy) != 1 or len(summary_full) != 1 or len(lsi_policy) != 1:
        raise ManuscriptCaseEvidenceError("Canonical policy/leakage tables do not have unique primary rows.")
    excluded = list(primary_excluded_features({"manuscript_final": dict(settings)}))
    dataset_spec = settings.get("datasets", {}).get("inx_primary", {})
    governance = _governance_evidence(
        dataset_name="inx_primary",
        role=str(dataset_spec.get("role", "primary_internal_oof")),
        allowed_claim=str(dataset_spec.get("allowed_claim", "Internal OOF evidence only.")),
    )
    probability_columns = _probability_columns(predictions)
    evidence: list[CompleteCaseEvidence] = []
    manifest: list[dict[str, Any]] = []
    for selection in selected:
        prediction_rows = predictions[predictions["sample_index"] == selection.sample_index]
        if len(prediction_rows) != 1:
            raise ManuscriptCaseEvidenceError(f"Missing unique INX prediction for {selection.sample_index}.")
        row = prediction_rows.iloc[0]
        predicted_class = int(row["y_pred"])
        probabilities = {
            column.removeprefix("prob_class_"): float(row[column])
            for column in probability_columns
        }
        shap = _top_local_shap(
            local_values,
            sample_index=selection.sample_index,
            predicted_class=predicted_class,
            top_k=int(settings.get("shap", {}).get("local", {}).get("top_k_reason_codes", 10)),
        )
        shap.shap_stability_summary = dict(stability_payload)
        cf_rows = cf_cases[
            (cf_cases["sample_index"].astype(int) == selection.sample_index)
            & (cf_cases["intervention_mode"].astype(str) == "employee_only")
        ]
        if len(cf_rows) != 1:
            raise ManuscriptCaseEvidenceError(
                f"Expected one employee_only OOF counterfactual row for INX case {selection.sample_index}; "
                f"found {len(cf_rows)}."
            )
        cf = cf_rows.iloc[0]
        eligible = _as_bool(cf.get("eligible_for_upward_shift"))
        valid = _as_bool(cf.get("valid"))
        changed = [
            value for value in str(cf.get("changed_features", "")).split(";") if value and value.casefold() != "nan"
        ]
        if not eligible:
            actionability_label = "not_eligible_already_predicted_highest_class"
        elif valid:
            actionability_label = "valid_model_scenario_found_not_prescriptive"
        else:
            actionability_label = "no_valid_model_scenario_found"
        case_id = f"inx_primary_{policy}_{selection.sample_index}"
        item = CompleteCaseEvidence(
            prediction=PredictionEvidence(
                case_id=case_id,
                predicted_class=predicted_class,
                true_class=int(row["y_true"]),
                class_probabilities=probabilities,
                confidence=float(row["confidence"]),
                uncertainty_flag=bool(float(row["confidence"]) < 0.60),
                model_name="xgboost",
                feature_policy=policy,
                leakage_safe_status="canonical_primary_nested_calibration_oof",
                dataset_name="inx_primary",
            ),
            shap=shap,
            fairness=fairness,
            calibration=CalibrationEvidence(
                log_loss=safe_float(calibration_row.get("nll_log_loss_mean")),
                brier_score=safe_float(calibration_row.get("multiclass_brier_mean")),
                expected_calibration_error=safe_float(calibration_row.get("ece_confidence_mean")),
                calibration_warning=(
                    f"{selected_method} probabilities are nested-OOF research confidence estimates, "
                    "not autonomous decision thresholds."
                ),
            ),
            counterfactual=CounterfactualEvidence(
                counterfactual_mode="employee_only",
                validity=float(valid) if eligible else None,
                changed_features=changed if valid else [],
                probability_gain=safe_float(cf.get("probability_gain")) if valid else None,
                proximity_cost=safe_float(cf.get("cost")) if valid else None,
                actionability_label=actionability_label,
                failed_reason=(
                    "" if valid else str(cf.get("failure_reason") or "No valid OOF scenario was identified.")
                ),
                warning="OOF counterfactual model scenario only; not a causal finding or employee prescription.",
                desired_class=int(cf["desired_class"]) if not pd.isna(cf.get("desired_class")) else None,
            ),
            leakage=LeakageEvidence(
                feature_policy=policy,
                excluded_leakage_features=excluded,
                full_feature_score=safe_float(summary_full.iloc[0].get("macro_f1_mean")),
                leakage_safe_score=safe_float(summary_policy.iloc[0].get("macro_f1_mean")),
                leakage_sensitivity_index=safe_float(lsi_policy.iloc[0].get("index_mean")),
                leakage_warning=(
                    "The full-feature score is a diagnostic leakage-warning upper bound only; "
                    "feature removal does not establish fairness or causality."
                ),
            ),
            governance=governance,
            evidence_sources=_sources(
                calibration_predictions_path,
                calibration_comparison_path,
                local_path,
                stability_path,
                cf_cases_path,
                cf_summary_path,
                fairness_path,
                policy_summary_path,
                lsi_path,
            ),
        )
        evidence.append(item)
        manifest.append(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "dataset_name": "inx_primary",
                "case_id": case_id,
                "sample_index": selection.sample_index,
                "sampling_reason": selection.sampling_reason,
                "true_class": int(row["y_true"]),
                "predicted_class": predicted_class,
                "confidence": float(row["confidence"]),
                "correct": bool(row["correct"]),
                "feature_policy": policy,
                "evidence_available": "complete_canonical_case_evidence",
            }
        )
    used_paths = {path for path, _ in bound_tables}
    if calibration_metadata_path.is_file():
        used_paths.add(calibration_metadata_path)
    return evidence, manifest, used_paths


def _split_columns(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [part for part in str(value).split(";") if part and part.casefold() != "nan"]


def _external_fairness(
    *,
    audited_groups: Sequence[str],
    disparity: pd.DataFrame | None = None,
    warnings: pd.DataFrame | None = None,
    policy: str = "department_free",
) -> FairnessEvidence:
    """Represent external audit scope without inventing uncomputed disparities."""

    subset = pd.DataFrame() if disparity is None else disparity.copy()
    for column in ("feature_set", "policy", "feature_policy"):
        if column in subset:
            candidate = subset[subset[column].astype(str) == policy]
            if not candidate.empty:
                subset = candidate
            break
    gap_column = next(
        (name for name in ("max_gap", "gap", "point_estimate") if name in subset.columns),
        None,
    )
    gaps: dict[str, Any] = {}
    if gap_column:
        for row in subset.itertuples(index=False):
            key = (
                f"{getattr(row, 'attribute', 'unknown')}."
                f"{getattr(row, 'metric', 'unspecified')}."
                f"{getattr(row, 'class_label', 'overall')}"
            )
            gaps[key] = safe_float(getattr(row, gap_column))
    warning_rows = [] if warnings is None or warnings.empty else [
        _record_dict(row) for _, row in warnings.iterrows()
    ]
    status = (
        "support_aware_external_disparities_available"
        if not subset.empty and gaps
        else "not_estimated_in_canonical_external_replication_stage"
    )
    return FairnessEvidence(
        audited_groups=list(audited_groups),
        subgroup_metrics={
            "status": status,
            "n_report_rows": int(len(subset)),
            "interpretation": "Mapped sensitive/proxy groups are retained for audit scope only.",
        },
        disparity_gaps=gaps,
        bootstrap_ci={},
        low_support_warnings=warning_rows,
        proxy_risk_warnings=[
            "External subgroup gaps, when present, are descriptive and support-dependent; absence of a number is not evidence of fairness.",
            "Removing direct sensitive/group variables does not eliminate proxy risk.",
        ],
    )


def _xgb_parameters(settings: Mapping[str, Any], seed: int) -> dict[str, Any]:
    model = settings.get("model", {})
    raw = model.get("xgboost", {}) if isinstance(model, Mapping) else {}
    if not isinstance(raw, Mapping):
        raise ManuscriptCaseEvidenceError("manuscript_final.model.xgboost must be a mapping.")
    parameters = dict(raw)
    parameters.pop("random_state_seed", None)
    parameters["random_state"] = seed
    return parameters


def _generate_external_oof_local_shap(
    *,
    settings: Mapping[str, Any],
    predictions: pd.DataFrame,
    selections: Sequence[SelectedCase],
    policy: str,
    run_id: str,
    config_hash: str,
    output_path: Path,
) -> pd.DataFrame:
    """Regenerate selected HR case attributions with each case's OOF model.

    The canonical external stage stores OOF predictions but intentionally omits
    XAI.  Replaying its declared fold/model contract here prevents the legacy
    full-fit-SHAP/OOF-prediction mismatch while limiting SHAP computation to the
    predeclared selected cases.
    """

    import shap

    dataset = load_external_dataset("hrdataset_v14", target_kind="primary")
    features = build_feature_columns(dataset, policy)
    forbidden = sorted(
        set(features).intersection(
            set(role_columns(dataset, "id"))
            | set(role_columns(dataset, "leakage"))
            | set(role_columns(dataset, "sensitive"))
        )
    )
    if forbidden:
        raise ManuscriptCaseEvidenceError(
            f"Forbidden mapped HR features reached OOF SHAP generation: {forbidden}"
        )
    selected_indices = {case.sample_index for case in selections}
    prediction_subset = predictions[predictions["sample_index"].isin(selected_indices)].copy()
    if len(prediction_subset) != len(selected_indices):
        raise ManuscriptCaseEvidenceError("Selected HR OOF prediction rows are incomplete.")
    X = dataset.canonical.loc[:, features].copy()
    y = dataset.canonical[dataset.target_column].astype(int)
    labels = [int(value) for value in settings.get("target", {}).get("labels", [2, 3, 4])]
    cv = settings.get("evaluation", {}).get("cv", {})
    requested_splits = int(cv.get("n_splits", 10))
    effective_splits = min(requested_splits, int(y.value_counts().min()))
    seed_value = cv.get("seed", "cv")
    seed = int(seed_value) if isinstance(seed_value, int) else int(settings.get("seeds", {})[seed_value])
    splitter = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=bool(cv.get("shuffle", True)),
        random_state=seed,
    )
    parameters = _xgb_parameters(settings, seed)
    rows: list[dict[str, Any]] = []
    for fold, (train_positions, test_positions) in enumerate(splitter.split(X, y), start=1):
        fold_indices = set(int(index) for index in X.index[test_positions])
        selected_in_fold = sorted(selected_indices.intersection(fold_indices))
        if not selected_in_fold:
            continue
        stored = prediction_subset[prediction_subset["sample_index"].isin(selected_in_fold)].copy()
        if set(stored["fold"].astype(int)) != {fold}:
            raise ManuscriptCaseEvidenceError(
                f"Stored HR fold identity does not match reconstructed fold {fold}."
            )
        X_train = X.iloc[train_positions]
        pipeline = Pipeline(
            [
                ("preprocessor", make_preprocessor(X_train)),
                ("model", LabelEncodedXGBClassifier(**parameters)),
            ]
        )
        pipeline.fit(X_train, y.iloc[train_positions])
        X_selected = X.loc[selected_in_fold]
        classifier = pipeline.named_steps["model"]
        probabilities = align_proba(
            pipeline.predict_proba(X_selected), classifier.classes_, labels
        )
        predicted = predict_labels_from_proba(probabilities, labels)
        stored = stored.set_index("sample_index").loc[selected_in_fold]
        if list(predicted.astype(int)) != stored["y_pred"].astype(int).tolist():
            raise ManuscriptCaseEvidenceError(
                f"Replayed HR OOF model predictions disagree with canonical stage in fold {fold}."
            )
        stored_probabilities = stored[[f"prob_class_{label}" for label in labels]].to_numpy(dtype=float)
        if not np.allclose(probabilities, stored_probabilities, rtol=1e-6, atol=1e-7):
            maximum = float(np.max(np.abs(probabilities - stored_probabilities)))
            raise ManuscriptCaseEvidenceError(
                f"Replayed HR OOF probabilities disagree with canonical stage in fold {fold}; "
                f"max absolute difference={maximum:.3g}."
            )
        preprocessor = pipeline.named_steps["preprocessor"]
        numeric, categorical = infer_columns(X_train)
        group_names, mapping = get_group_mapping(preprocessor, numeric, categorical)
        if set(group_names) != set(features) or len(group_names) != len(features):
            raise ManuscriptCaseEvidenceError(
                "HR OOF SHAP raw feature families do not match the external policy features."
            )
        transformed = preprocessor.transform(X_selected)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        raw = shap.TreeExplainer(classifier.model_).shap_values(transformed)
        normalized = normalize_shap_values(
            raw,
            n_samples=len(X_selected),
            n_features=transformed.shape[1],
            n_classes=len(labels),
        )
        grouped = group_shap_values(normalized, group_names, mapping)
        for row_position, sample_index in enumerate(selected_in_fold):
            predicted_class = int(predicted[row_position])
            class_index = labels.index(predicted_class)
            for feature_index, feature in enumerate(group_names):
                value = float(grouped[row_position, class_index, feature_index])
                rows.append(
                    {
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "dataset_name": "hrdataset_v14",
                        "policy": policy,
                        "sample_index": sample_index,
                        "fold": fold,
                        "class_label": predicted_class,
                        "predicted_class": predicted_class,
                        "feature": feature,
                        "grouped_shap_value": value,
                        "abs_grouped_shap_value": abs(value),
                        "evaluation_scope": "out_of_fold_selected_case",
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty or frame["sample_index"].nunique() != len(selected_indices):
        raise ManuscriptCaseEvidenceError("Selected HR OOF SHAP replay did not cover every selected case.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def _build_external_cases(
    *,
    inputs: StageInputs,
    settings: Mapping[str, Any],
    run_id: str,
    config_hash: str,
    sample_size: int,
    seed: int,
    policy: str,
    generated_output_dir: Path,
) -> tuple[list[CompleteCaseEvidence], list[dict[str, Any]], set[Path]]:
    prediction_path, raw_predictions = _read_required_csv_schema_alias(
        inputs.hrdataset_dir,
        "model_predictions.csv",
        "oof_predictions.csv",
    )
    metrics_path, metrics = _read_required_csv_schema_alias(
        inputs.hrdataset_dir,
        "performance_metrics.csv",
        "policy_summary.csv",
    )
    policy_path, policy_audit = _read_required_csv(inputs.hrdataset_dir, "feature_policy_audit.csv")
    support_path, support = _read_required_csv(inputs.hrdataset_dir, "target_support.csv")
    representative_path = inputs.hrdataset_dir / "representative_cases.csv"
    representative = pd.read_csv(representative_path) if representative_path.is_file() else pd.DataFrame()
    local_path = inputs.hrdataset_dir / "shap" / policy / "local_grouped_shap_values.csv"
    local_values = pd.read_csv(local_path) if local_path.is_file() else pd.DataFrame()
    fairness_path = inputs.hrdataset_dir / "fairness_proxy" / "fairness_disparity_summary.csv"
    disparity = pd.read_csv(fairness_path) if fairness_path.is_file() else pd.DataFrame()
    warning_path = inputs.hrdataset_dir / "fairness_proxy" / "small_group_warnings.csv"
    fairness_warnings = pd.read_csv(warning_path) if warning_path.is_file() else pd.DataFrame()
    actionability_path = inputs.hrdataset_dir / "actionability_summary.csv"
    actionability = pd.read_csv(actionability_path) if actionability_path.is_file() else pd.DataFrame()

    # External stages are bound by exact upstream hashes below.  If the external
    # runner already emits run/config fields, enforce them as well.
    for path, frame in (
        (prediction_path, raw_predictions),
        (metrics_path, metrics),
        (policy_path, policy_audit),
        (support_path, support),
    ):
        _assert_identity(frame, path=path, run_id=run_id, config_hash=config_hash, required=True)
    for path, frame in (
        (representative_path, representative),
        (local_path, local_values),
        (fairness_path, disparity),
        (warning_path, fairness_warnings),
        (actionability_path, actionability),
    ):
        if path.is_file():
            _assert_identity(frame, path=path, run_id=run_id, config_hash=config_hash, required=True)

    predictions = raw_predictions[raw_predictions["policy"].astype(str) == policy].copy()
    predictions = _prepare_predictions(predictions, path=prediction_path)
    selected = _select_cases(
        predictions,
        sample_size=sample_size,
        seed=seed,
        representative=(
            representative[representative["policy"].astype(str) == policy]
            if not representative.empty and "policy" in representative
            else representative
        ),
        local_values=(
            local_values[local_values["policy"].astype(str) == policy]
            if not local_values.empty and "policy" in local_values
            else local_values
        ),
    )
    if local_values.empty:
        local_path = generated_output_dir / "hrdataset_v14_oof_local_grouped_shap_values.csv"
        local_policy = _generate_external_oof_local_shap(
            settings=settings,
            predictions=predictions,
            selections=selected,
            policy=policy,
            run_id=run_id,
            config_hash=config_hash,
            output_path=local_path,
        )
    else:
        local_policy = (
            local_values[local_values["policy"].astype(str) == policy]
            if "policy" in local_values
            else local_values
        )
    metric_rows = metrics[metrics["policy"].astype(str) == policy]
    audit_rows = policy_audit[policy_audit["policy"].astype(str) == policy]
    if len(metric_rows) != 1 or len(audit_rows) != 1:
        raise ManuscriptCaseEvidenceError(
            f"External policy {policy} must have unique metric and feature-audit rows."
        )
    metric_row = metric_rows.iloc[0]
    audit_row = audit_rows.iloc[0]
    external_dataset = load_external_dataset("hrdataset_v14", target_kind="primary")
    fairness = _external_fairness(
        audited_groups=audit_attribute_columns(external_dataset),
        disparity=disparity,
        warnings=fairness_warnings,
        policy=policy,
    )
    dataset_spec = settings.get("datasets", {}).get("hrdataset_v14", {})
    governance = _governance_evidence(
        dataset_name="hrdataset_v14",
        role=str(dataset_spec.get("role", "independent_external_performance_target_replication")),
        allowed_claim=str(
            dataset_spec.get(
                "allowed_claim",
                "Independent external performance-target replication; not locked-model transport.",
            )
        ),
    )
    probability_columns = _probability_columns(predictions)
    excluded = sorted(
        set(_split_columns(audit_row.get("excluded_leakage_columns")))
        | set(_split_columns(audit_row.get("excluded_sensitive_columns")))
    )
    evidence: list[CompleteCaseEvidence] = []
    manifest: list[dict[str, Any]] = []
    for selection in selected:
        prediction_rows = predictions[predictions["sample_index"] == selection.sample_index]
        if len(prediction_rows) != 1:
            raise ManuscriptCaseEvidenceError(f"Missing unique HRDataset_v14 prediction for {selection.sample_index}.")
        row = prediction_rows.iloc[0]
        predicted_class = int(row["y_pred"])
        shap = _top_local_shap(
            local_policy,
            sample_index=selection.sample_index,
            predicted_class=predicted_class,
            top_k=int(settings.get("shap", {}).get("local", {}).get("top_k_reason_codes", 10)),
            external=True,
            expected_fold=int(row["fold"]),
        )
        shap.shap_stability_summary = {
            "evaluation_scope": "case-specific OOF attribution",
            "cross_fold_rank_stability_available": False,
            "limitation": "External case evidence does not claim INX fold-ranking stability transfers to this dataset.",
        }
        probabilities = {
            column.removeprefix("prob_class_"): float(row[column])
            for column in probability_columns
        }
        case_id = f"hrdataset_v14_{policy}_{selection.sample_index}"
        item = CompleteCaseEvidence(
            prediction=PredictionEvidence(
                case_id=case_id,
                predicted_class=predicted_class,
                true_class=int(row["y_true"]),
                class_probabilities=probabilities,
                confidence=float(row["confidence"]),
                uncertainty_flag=bool(float(row["confidence"]) < 0.60),
                model_name="xgboost",
                feature_policy=policy,
                leakage_safe_status="independent_external_replication_oof_dataset_specific_model",
                dataset_name="hrdataset_v14",
            ),
            shap=shap,
            fairness=fairness,
            calibration=CalibrationEvidence(
                log_loss=safe_float(metric_row.get("nll_log_loss", metric_row.get("log_loss"))),
                brier_score=safe_float(metric_row.get("multiclass_brier")),
                expected_calibration_error=safe_float(
                    metric_row.get("ece_confidence", metric_row.get("ece"))
                ),
                calibration_warning=(
                    "External OOF probabilities are dataset-specific diagnostic confidence estimates; "
                    "they are not transported INX probabilities or autonomous thresholds."
                ),
            ),
            counterfactual=CounterfactualEvidence(
                counterfactual_mode="external_case_level_counterfactual_not_evaluated",
                validity=None,
                changed_features=[],
                probability_gain=None,
                proximity_cost=None,
                actionability_label=str(
                    (
                        actionability[actionability["policy"].astype(str) == policy].iloc[0].get(
                            "actionability_status",
                            "external_case_level_actionability_not_evaluated",
                        )
                        if not actionability.empty and "policy" in actionability
                        and len(actionability[actionability["policy"].astype(str) == policy])
                        else "external_case_level_actionability_not_evaluated"
                    )
                ),
                failed_reason=(
                    "No case-level OOF counterfactual search was performed for the external replication; "
                    "no external counterfactual validity claim is supplied."
                ),
                warning="No employee prescription is supported; counterfactuals would be model scenarios only.",
            ),
            leakage=LeakageEvidence(
                feature_policy=policy,
                excluded_leakage_features=excluded,
                full_feature_score=None,
                leakage_safe_score=safe_float(metric_row.get("macro_f1")),
                leakage_sensitivity_index=None,
                leakage_warning=(
                    "Dataset-specific external policy excludes mapped target, identifier, leakage-risk, "
                    "and sensitive/audit-only fields; this is replication, not locked INX-model transport."
                ),
            ),
            governance=governance,
            evidence_sources=_sources(
                prediction_path,
                metrics_path,
                policy_path,
                local_path,
                support_path,
            ),
        )
        evidence.append(item)
        manifest.append(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "dataset_name": "hrdataset_v14",
                "case_id": case_id,
                "sample_index": selection.sample_index,
                "sampling_reason": selection.sampling_reason,
                "true_class": int(row["y_true"]),
                "predicted_class": predicted_class,
                "confidence": float(row["confidence"]),
                "correct": bool(row["correct"]),
                "feature_policy": policy,
                "evidence_available": "complete_canonical_case_evidence",
            }
        )
    used_paths = {
        prediction_path,
        metrics_path,
        policy_path,
        local_path,
        support_path,
    }
    used_paths.update(
        path
        for path in (representative_path, fairness_path, warning_path, actionability_path)
        if path.is_file()
    )
    return evidence, manifest, used_paths


def _write_preflight(
    report: Mapping[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    json_path = output_dir / "preflight_report.json"
    csv_path = output_dir / "preflight_cases.csv"
    md_path = output_dir / "preflight_report.md"
    write_json(json_path, dict(report))
    rows: list[dict[str, Any]] = []
    for raw in report.get("cases", []):
        row = dict(raw)
        for key in ("missing_fields", "invalid_fields", "forbidden_features"):
            row[key] = json.dumps(row.get(key), sort_keys=True, ensure_ascii=True)
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    lines = [
        "# Canonical CompleteCaseEvidence Preflight",
        "",
        f"Run ID: `{report.get('run_id')}`  ",
        f"Config hash: `{report.get('config_hash')}`  ",
        f"Execution intent: `{report.get('execution_intent')}`",
        "",
        f"- Cases requested: {report.get('cases_requested')}",
        f"- Cases selected: {report.get('cases_selected')}",
        f"- Cases complete: {report.get('cases_complete')}",
        f"- Cases incomplete: {report.get('cases_incomplete')}",
        f"- Complete-case rate: {report.get('complete_case_rate')}",
        f"- Wilson interval: [{report.get('complete_case_rate_wilson_ci_low')}, {report.get('complete_case_rate_wilson_ci_high')}]",
        f"- API call attempted: `{report.get('api_call_attempted')}`",
        f"- Real API execution allowed: `{report.get('real_api_execution_allowed')}`",
        "",
        "## Missing Fields by Category",
        "",
    ]
    missing = report.get("missing_fields_by_category", {})
    lines.extend([f"- `{key}`: {value}" for key, value in sorted(missing.items())] or ["No missing evidence fields."])
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This artifact establishes deterministic evidence readiness only. It contains no generated LLM text, "
            "does not report LLM faithfulness/compliance, and does not authorize a paid API run. Any real batch "
            "requires separate explicit user approval and the existing real-run approval/preflight boundary.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "preflight_report": json_path,
        "preflight_cases": csv_path,
        "preflight_markdown": md_path,
    }


def run(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    output_dir: str | Path,
    run_id: str,
    inx_shap_dir: str | Path | None = None,
    calibration_dir: str | Path | None = None,
    counterfactual_dir: str | Path | None = None,
    fairness_dir: str | Path | None = None,
    policy_dir: str | Path | None = None,
    hrdataset_dir: str | Path | None = None,
    shap_dir: str | Path | None = None,
    external_dir: str | Path | None = None,
    config_hash: str | None = None,
    inx_sample_size: int | None = None,
    hr_sample_size: int | None = None,
    hr_policy: str = "department_free",
    require_complete: bool = True,
    real_api_execution: bool = False,
) -> dict[str, Path]:
    """Build and validate the canonical 40+40 case evidence package offline.

    ``hrdataset_dir`` may be the canonical external stage root or its
    ``hrdataset_v14`` task subdirectory.  ``shap_dir``/``external_dir`` are
    aliases used by the integrated manuscript orchestrator.  This function has
    no real execution path; passing ``real_api_execution=True`` is rejected.
    """

    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if real_api_execution:
        raise ManuscriptCaseEvidenceError(
            "This stage is deterministic preflight-only and cannot execute a real LLM/API batch."
        )
    config = load_manuscript_config(config_path)
    settings = config["manuscript_final"]
    expected_hash = canonical_config_hash(config)
    config_hash = config_hash or expected_hash
    if config_hash != expected_hash:
        raise ManuscriptCaseEvidenceError(
            f"Provided config hash {config_hash} does not match canonical config {expected_hash}."
        )
    output = ensure_dir(Path(output_dir))
    resolved_shap = Path(inx_shap_dir or shap_dir) if (inx_shap_dir or shap_dir) else None
    resolved_external = Path(hrdataset_dir or external_dir) if (hrdataset_dir or external_dir) else None
    if resolved_external is not None and (resolved_external / "hrdataset_v14").is_dir():
        resolved_external = resolved_external / "hrdataset_v14"
    resolved_policy = Path(policy_dir) if policy_dir is not None else (
        resolved_shap.parent / "policy" if resolved_shap is not None else None
    )
    required_directories = {
        "inx_shap_dir": resolved_shap,
        "calibration_dir": Path(calibration_dir) if calibration_dir is not None else None,
        "counterfactual_dir": Path(counterfactual_dir) if counterfactual_dir is not None else None,
        "fairness_dir": Path(fairness_dir) if fairness_dir is not None else None,
        "policy_dir": resolved_policy,
        "hrdataset_dir": resolved_external,
    }
    missing_arguments = sorted(name for name, path in required_directories.items() if path is None)
    if missing_arguments:
        raise ManuscriptCaseEvidenceError(
            f"Canonical stage directory arguments are missing: {missing_arguments}"
        )
    inputs = StageInputs(
        inx_shap_dir=required_directories["inx_shap_dir"],  # type: ignore[arg-type]
        calibration_dir=required_directories["calibration_dir"],  # type: ignore[arg-type]
        counterfactual_dir=required_directories["counterfactual_dir"],  # type: ignore[arg-type]
        fairness_dir=required_directories["fairness_dir"],  # type: ignore[arg-type]
        policy_dir=required_directories["policy_dir"],  # type: ignore[arg-type]
        hrdataset_dir=required_directories["hrdataset_dir"],  # type: ignore[arg-type]
    )
    missing_directories = [str(path) for path in inputs.directories() if not path.is_dir()]
    if missing_directories:
        raise ManuscriptCaseEvidenceError(
            f"Canonical stage directories are missing: {missing_directories}"
        )
    scope = settings.get("llm_agent_evaluation", {}).get("scope", {})
    inx_count = int(scope.get("inx_primary", 40) if inx_sample_size is None else inx_sample_size)
    hr_count = int(scope.get("hrdataset_v14", 40) if hr_sample_size is None else hr_sample_size)
    seed = int(settings.get("seeds", {}).get("llm_sampling", 42))

    inx_evidence, inx_manifest, inx_paths = _build_inx_cases(
        inputs=inputs,
        settings=settings,
        run_id=run_id,
        config_hash=config_hash,
        sample_size=inx_count,
        seed=seed,
    )
    hr_evidence, hr_manifest, external_paths = _build_external_cases(
        inputs=inputs,
        settings=settings,
        run_id=run_id,
        config_hash=config_hash,
        sample_size=hr_count,
        seed=seed,
        policy=hr_policy,
        generated_output_dir=output,
    )
    evidence = [*inx_evidence, *hr_evidence]
    manifest = [*inx_manifest, *hr_manifest]
    requested = inx_count + hr_count
    report = build_evidence_preflight_report(
        [{"evidence": item, "notes": "canonical offline manuscript evidence"} for item in evidence],
        run_id=run_id,
        run_mode="dry_run",
        canonical_config_path=config_path,
        requested_case_count=requested,
    )
    report.update(
        {
            "config_hash": config_hash,
            "execution_intent": "deterministic_preflight_only_no_llm_calls",
            "api_call_attempted": False,
            "real_api_execution_allowed": False,
            "api_execution_allowed": False,
            "paid_api_approval_recorded": False,
            "next_step_requires_explicit_paid_api_approval": True,
            "dataset_requested_counts": {"inx_primary": inx_count, "hrdataset_v14": hr_count},
            "n_cases_requested": report["cases_requested"],
            "n_cases_selected": report["cases_selected"],
            "n_cases_complete": report["cases_complete"],
            "n_cases_incomplete": report["cases_incomplete"],
            "evidence_jsonl_schema": "run_id/config_hash/selection/evidence",
        }
    )
    evidence_path = output / "complete_case_evidence.jsonl"
    manifest_path = output / "case_manifest.csv"
    metadata_path = output / "case_evidence_metadata.json"
    manifest_lookup = {row["case_id"]: row for row in manifest}
    write_jsonl(
        evidence_path,
        (
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "dataset_name": item.prediction.dataset_name,
                "case_id": item.prediction.case_id,
                "sampling_reason": manifest_lookup[item.prediction.case_id]["sampling_reason"],
                "evidence": item.to_dict(),
            }
            for item in evidence
        ),
    )
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    preflight_paths = _write_preflight(report, output_dir=output)
    upstream_paths = sorted(inx_paths | external_paths, key=lambda path: str(path))
    dataset_counts = dict(Counter(item.prediction.dataset_name for item in evidence))
    write_json(
        metadata_path,
        {
            "stage": "canonical_llm_case_evidence_preflight",
            "run_id": run_id,
            "config_hash": config_hash,
            "execution_mode": "deterministic_preflight_only",
            "api_call_attempted": False,
            "real_api_execution_allowed": False,
            "paid_api_approval_recorded": False,
            "requested_case_count": requested,
            "complete_case_count": report["cases_complete"],
            "dataset_counts": dataset_counts,
            "external_local_shap_requirement": "case-specific OOF fold and OOF-prediction agreement",
            "upstream_artifacts": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in upstream_paths
            ],
            "outputs": {
                "complete_case_evidence": str(evidence_path),
                "case_manifest": str(manifest_path),
                **{key: str(value) for key, value in preflight_paths.items()},
            },
            "claim_boundary": (
                "Evidence readiness only; no LLM faithfulness claim, no API execution, and no autonomous HR use."
            ),
        },
    )
    outputs = {
        "complete_case_evidence": evidence_path,
        "case_manifest": manifest_path,
        "metadata": metadata_path,
        **preflight_paths,
    }
    if require_complete and (
        report["cases_selected"] != requested
        or report["cases_complete"] != requested
        or report["cases_incomplete"] != 0
    ):
        raise ManuscriptCaseEvidenceError(
            "Canonical LLM evidence preflight is incomplete; reports were written for diagnosis: "
            f"requested={requested}, selected={report['cases_selected']}, "
            f"complete={report['cases_complete']}, incomplete={report['cases_incomplete']}."
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical CompleteCaseEvidence and a deterministic no-API preflight."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash", default=None)
    parser.add_argument("--inx-shap-dir", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--counterfactual-dir", required=True)
    parser.add_argument("--fairness-dir", required=True)
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--hrdataset-dir", required=True)
    parser.add_argument("--inx-sample-size", type=int, default=None)
    parser.add_argument("--hr-sample-size", type=int, default=None)
    parser.add_argument("--hr-policy", default="department_free")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(
        arguments.config,
        output_dir=arguments.output_dir,
        run_id=arguments.run_id,
        config_hash=arguments.config_hash,
        inx_shap_dir=arguments.inx_shap_dir,
        calibration_dir=arguments.calibration_dir,
        counterfactual_dir=arguments.counterfactual_dir,
        fairness_dir=arguments.fairness_dir,
        policy_dir=arguments.policy_dir,
        hrdataset_dir=arguments.hrdataset_dir,
        inx_sample_size=arguments.inx_sample_size,
        hr_sample_size=arguments.hr_sample_size,
        hr_policy=arguments.hr_policy,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2, sort_keys=True))


__all__ = ["ManuscriptCaseEvidenceError", "StageInputs", "run"]
