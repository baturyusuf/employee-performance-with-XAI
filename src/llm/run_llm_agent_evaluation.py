from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.agents.run_governance_audit import run_batch_from_records
from src.chatbot.run_guardrail_eval import run as run_chatbot_guardrail_eval
from src.core.io_utils import object_hash, read_csv_optional, write_jsonl
from src.core.reporting import markdown_table, write_markdown
from src.core.run_registry import RunRegistryEntry, append_run_entry
from src.data.external_adapters import load_external_dataset
from src.data.external_audit import dataset_report_dir
from src.experiments.external_validation import build_external_case_evidence
from src.llm.evidence_schema import (
    CalibrationEvidence,
    CompleteCaseEvidence,
    CounterfactualEvidence,
    FairnessEvidence,
    GovernanceEvidence,
    LeakageEvidence,
    PredictionEvidence,
)
from src.llm.faithfulness_checker import check_faithfulness_categories
from src.llm.governed_explainer import GovernedExplainer
from src.llm.runtime_config import LLMRuntimeConfig
from src.utils.config import SETTINGS
from src.utils.config_loader import load_config
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


DEFAULT_CONFIG = "configs/llm_agent_eval.yaml"


def run(config_path: str = DEFAULT_CONFIG) -> Dict[str, Path]:
    config = load_config(config_path)
    settings = config.get("llm_agent_eval", config)
    output_dir = Path(settings.get("output_dir", "reports/llm_explanations"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{settings.get('run_id_prefix', 'llm_agent_eval')}_{utc_now_iso()}"
    run_mode = str(settings.get("run_mode", "dry_run")).lower()
    prompt_version = str(settings.get("prompt_version", "governed_evidence_v2"))
    agent_version = str(settings.get("agent_system_version", "deterministic_agents_v2"))
    retry_policy = settings.get("retry_policy", {})
    rate_limit = settings.get("rate_limit", {})

    runtime_config = runtime_from_settings(settings, run_mode)
    manifest_rows, evidence_items = build_manifest_and_evidence(settings, run_id=run_id)
    manifest_path = output_dir / "eval_case_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    explainer = GovernedExplainer(runtime_config=runtime_config)
    explanation_rows: List[Dict[str, Any]] = []
    faithfulness_rows: List[Dict[str, Any]] = []

    for item in evidence_items:
        evidence = item["evidence"]
        evidence_dict = evidence.to_dict()
        evidence_hash = object_hash(evidence_dict)
        parsing_errors = ""
        try:
            response = generate_with_retries(
                explainer,
                evidence,
                max_retries=int(retry_policy.get("max_retries", 0)),
                retry_backoff_seconds=float(retry_policy.get("retry_backoff_seconds", 1.0)),
                sleep_after_success_seconds=rate_limit_sleep_seconds(rate_limit) if run_mode == "real" else 0.0,
            )
            parsed_success, parsing_errors = validate_explanation_payload(response)
        except Exception as exc:
            if run_mode == "real":
                raise
            response = {
                "case_id": evidence.prediction.case_id,
                "short_explanation": "",
                "detailed_explanation": "",
                "warnings": [],
                "unsupported_claims_detected": [],
                "requires_human_review": True,
            }
            parsed_success = False
            parsing_errors = str(exc)
        response_hash = object_hash(response)
        categorized = check_faithfulness_categories(response, evidence_dict, parsing_error=parsing_errors if not parsed_success else "")
        faithfulness_rows.append(
            categorized.to_eval_row(
                run_id=run_id,
                dataset_name=evidence.prediction.dataset_name,
                case_id=evidence.prediction.case_id,
                evidence_hash=evidence_hash,
                response_hash=response_hash,
                notes=item.get("notes", ""),
            )
        )
        explanation_rows.append(
            {
                "run_id": run_id,
                "dataset_name": evidence.prediction.dataset_name,
                "case_id": evidence.prediction.case_id,
                "model_name": evidence.prediction.model_name,
                "feature_policy": evidence.prediction.feature_policy,
                "llm_provider": runtime_config.provider,
                "llm_model": runtime_config.model,
                "run_mode": run_mode,
                "prompt_version": prompt_version,
                "agent_system_version": agent_version,
                "evidence_hash": evidence_hash,
                "response_hash": response_hash,
                "short_explanation": response.get("short_explanation", ""),
                "detailed_explanation": response.get("detailed_explanation", ""),
                "warnings": response.get("warnings", []),
                "evidence_references": evidence.evidence_sources,
                "unsupported_evidence_flags": response.get("unsupported_claims_detected", []),
                "raw_response": response,
                "structured_evidence": evidence_dict,
                "parsed_success": parsed_success,
                "parsing_errors": parsing_errors,
                "created_at": utc_now_iso(),
            }
        )

    explanations_path = output_dir / "governed_explanations.jsonl"
    faithfulness_path = output_dir / "faithfulness_eval.csv"
    faithfulness_summary_path = output_dir / "faithfulness_eval_summary.md"
    write_jsonl(explanations_path, explanation_rows)
    faithfulness_df = pd.DataFrame(faithfulness_rows)
    faithfulness_df.to_csv(faithfulness_path, index=False)
    write_faithfulness_summary(faithfulness_df, faithfulness_summary_path)

    agent_outputs: Dict[str, Path] = {}
    if settings.get("evaluation_flags", {}).get("run_agent_audit", True):
        agent_output_dir = Path(settings.get("agent_output_dir", SETTINGS.reports_dir / "agent_audits"))
        agent_outputs = run_batch_from_records(
            explanation_rows,
            run_id=run_id,
            output_dir=agent_output_dir,
            config_path=config_path,
        )

    guardrail_outputs: Dict[str, Path] = {}
    if settings.get("evaluation_flags", {}).get("run_chatbot_guardrail_tests", True):
        guardrail_outputs = run_chatbot_guardrail_eval("configs/chatbot_guardrail_eval.yaml")

    summary_paths = write_integrated_summary(
        run_id=run_id,
        run_mode=run_mode,
        runtime_config=runtime_config,
        prompt_version=prompt_version,
        agent_version=agent_version,
        manifest_df=pd.DataFrame(manifest_rows),
        faithfulness_df=faithfulness_df,
        agent_csv=agent_outputs.get("csv"),
        guardrail_csv=guardrail_outputs.get("evaluation"),
        output_dir=output_dir,
    )

    outputs = {
        "manifest": manifest_path,
        "governed_explanations": explanations_path,
        "faithfulness_eval": faithfulness_path,
        "faithfulness_summary": faithfulness_summary_path,
        **{f"agent_{key}": value for key, value in agent_outputs.items()},
        **{f"chatbot_{key}": value for key, value in guardrail_outputs.items()},
        **summary_paths,
    }
    append_run_entry(
        RunRegistryEntry(
            run_id=run_id,
            command=f"python -m src.llm.run_llm_agent_evaluation --config {config_path}",
            config_path=config_path,
            dataset="multiple",
            model="xgboost",
            feature_policy="multiple",
            llm_provider=runtime_config.provider,
            llm_model=runtime_config.model,
            seed=str(settings.get("seed", 42)),
            output_files=[str(path) for path in outputs.values()],
        )
    )
    append_registry_row(
        {
            "run_id": run_id,
            "date_time": utc_now_iso(),
            "git_commit_if_available": get_git_commit(),
            "script": "python -m src.llm.run_llm_agent_evaluation",
            "config": config_path,
            "feature_set": "multiple",
            "model": f"{runtime_config.provider}:{runtime_config.model}",
            "seed": settings.get("seed", 42),
            "cv_strategy": "not_applicable",
            "primary_metrics": summarize_faithfulness_metrics(faithfulness_df),
            "output_dir": str(output_dir),
            "notes": f"Config-driven LLM-agent evaluation. run_mode={run_mode}. Dry-run outputs are not manuscript-grade real LLM evidence.",
            "decision_status": "candidate" if run_mode == "real" else "dry_run_stub",
        }
    )
    return outputs


def rate_limit_sleep_seconds(rate_limit: Dict[str, Any]) -> float:
    configured_sleep = float(rate_limit.get("sleep_between_requests_seconds", 0.0))
    rpm = float(rate_limit.get("requests_per_minute", 0) or 0)
    rpm_sleep = 60.0 / rpm if rpm > 0 else 0.0
    return max(configured_sleep, rpm_sleep)


def generate_with_retries(
    explainer: GovernedExplainer,
    evidence: CompleteCaseEvidence,
    *,
    max_retries: int,
    retry_backoff_seconds: float,
    sleep_after_success_seconds: float,
) -> Dict[str, Any]:
    attempt = 0
    while True:
        try:
            response = explainer.generate(evidence)
            if sleep_after_success_seconds > 0:
                time.sleep(sleep_after_success_seconds)
            return response
        except Exception:
            if attempt >= max_retries:
                raise
            attempt += 1
            time.sleep(retry_backoff_seconds * attempt)


def runtime_from_settings(settings: Dict[str, Any], run_mode: str) -> LLMRuntimeConfig:
    llm = settings.get("llm", {})
    if run_mode == "dry_run":
        return LLMRuntimeConfig(
            provider="offline",
            model=str(llm.get("model", "offline_stub_llm")),
            temperature=float(llm.get("temperature", 0.0)),
            max_tokens=int(llm.get("max_tokens", 1200)),
            require_real_llm=False,
        )
    if run_mode != "real":
        raise ValueError("llm_agent_eval.run_mode must be 'dry_run' or 'real'")
    provider = str(llm.get("provider", "openai"))
    if provider != "openai":
        raise ValueError("Real manuscript-grade LLM evaluation currently requires provider='openai'")
    return LLMRuntimeConfig(
        provider="openai",
        model=str(llm.get("model", "gpt-5.4-mini")),
        temperature=float(llm.get("temperature", 0.0)),
        max_tokens=int(llm.get("max_tokens", 1200)),
        require_real_llm=True,
    )


def build_manifest_and_evidence(settings: Dict[str, Any], *, run_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    seed = int(settings.get("seed", 42))
    manifest_rows: List[Dict[str, Any]] = []
    evidence_items: List[Dict[str, Any]] = []
    for ds in settings.get("datasets", []):
        if not ds.get("enabled", True):
            continue
        if ds.get("source") == "internal":
            rows, items = build_internal_dataset_cases(ds, run_id=run_id, seed=seed)
        elif ds.get("source") == "external":
            rows, items = build_external_dataset_cases(ds, run_id=run_id, seed=seed)
        else:
            raise ValueError(f"Unknown dataset source for {ds.get('dataset_name')}: {ds.get('source')}")
        manifest_rows.extend(rows)
        evidence_items.extend(items)
    return manifest_rows, evidence_items


def build_internal_dataset_cases(ds: Dict[str, Any], *, run_id: str, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dataset_name = str(ds["dataset_name"])
    feature_policy = str(ds["feature_policy"])
    prediction_path = Path(ds["prediction_path"])
    if not prediction_path.exists():
        raise FileNotFoundError(f"Required internal prediction file missing: {prediction_path}")
    predictions = pd.read_csv(prediction_path)
    predictions = predictions[(predictions["feature_set"] == feature_policy) & (predictions["method"] == "sigmoid")].copy()
    if predictions.empty:
        raise RuntimeError(f"No internal predictions found for {dataset_name}/{feature_policy} in {prediction_path}")
    predictions["confidence"] = predictions[[col for col in predictions.columns if col.startswith("prob_class_")]].max(axis=1)
    predictions["correct"] = predictions["y_true"].astype(int) == predictions["y_pred"].astype(int)

    representative = read_csv_optional(Path(ds.get("shap_representative_path", "")))
    reason_code_dir = Path(ds.get("reason_code_dir", ""))
    direct_reason_cases = reason_code_case_ids(reason_code_dir)
    selected = select_risk_aware_cases(
        predictions,
        sample_size=int(ds.get("sample_size", 30)),
        seed=seed,
        representative=representative,
        representative_case_col="case",
        extra_available_cases=direct_reason_cases,
    )
    base = CompleteCaseEvidence.from_reports(case_id=next(iter(direct_reason_cases), None))
    rows: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []
    for selected_row in selected:
        sample_index = int(selected_row["sample_index"])
        pred_row = predictions[predictions["sample_index"] == sample_index].iloc[0]
        evidence_case_id = f"{dataset_name}_{feature_policy}_{sample_index}"
        has_reason = str(sample_index) in direct_reason_cases
        if has_reason:
            evidence = CompleteCaseEvidence.from_reports(case_id=str(sample_index))
            evidence.prediction = make_prediction_evidence(dataset_name, evidence_case_id, feature_policy, pred_row, ds["model_name"])
        else:
            evidence = CompleteCaseEvidence(
                prediction=make_prediction_evidence(dataset_name, evidence_case_id, feature_policy, pred_row, ds["model_name"]),
                shap=None,
                fairness=base.fairness,
                calibration=base.calibration,
                counterfactual=base.counterfactual,
                leakage=base.leakage,
                governance=base.governance,
                evidence_sources=[str(prediction_path), "local SHAP/reason-code evidence unavailable for this sampled case"],
            )
        evidence_available = "full" if has_reason else "partial_missing_local_shap"
        rows.append(
            manifest_row(
                run_id=run_id,
                dataset_name=dataset_name,
                case_id=evidence.prediction.case_id,
                pred_row=pred_row,
                feature_policy=feature_policy,
                model_name=str(ds["model_name"]),
                sampling_reason=str(selected_row["sampling_reason"]),
                evidence_available=evidence_available,
                notes="" if has_reason else "Local reason-code/SHAP evidence unavailable; report-level evidence retained.",
            )
        )
        items.append({"evidence": evidence, "notes": rows[-1]["notes"]})
    return rows, items


def build_external_dataset_cases(ds: Dict[str, Any], *, run_id: str, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    display_dataset_name = str(ds["dataset_name"])
    external_dataset_name = str(ds.get("external_dataset_name", display_dataset_name))
    target_kind = str(ds.get("target_kind", "primary"))
    feature_policy = str(ds["feature_policy"])
    report_dir = Path(ds.get("report_dir") or dataset_report_dir(external_dataset_name, target_kind))
    prediction_path = report_dir / "model_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Required external prediction file missing: {prediction_path}")
    predictions = pd.read_csv(prediction_path)
    predictions = predictions[predictions["policy"] == feature_policy].copy()
    if predictions.empty:
        raise RuntimeError(f"No external predictions found for {display_dataset_name}/{feature_policy} in {prediction_path}")

    representative = read_csv_optional(report_dir / "representative_cases.csv")
    selected = select_risk_aware_cases(
        predictions,
        sample_size=int(ds.get("sample_size", 10)),
        seed=seed,
        representative=representative[representative["policy"] == feature_policy] if not representative.empty else representative,
        representative_case_col="case_type",
        extra_available_cases=set(),
        shap_local_path=report_dir / "shap" / feature_policy / "local_grouped_shap_values.csv",
    )
    dataset = load_external_dataset(external_dataset_name, target_kind=target_kind)
    rows: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []
    for selected_row in selected:
        sample_index = int(selected_row["sample_index"])
        pred_row = predictions[predictions["sample_index"] == sample_index].iloc[0]
        evidence = build_external_case_evidence(
            dataset=dataset,
            output_dir=report_dir,
            policy=feature_policy,
            sample_index=sample_index,
            case_type=str(selected_row["sampling_reason"]),
        )
        if display_dataset_name != external_dataset_name:
            evidence.prediction.dataset_name = display_dataset_name
            evidence.prediction.case_id = evidence.prediction.case_id.replace(external_dataset_name, display_dataset_name, 1)
        rows.append(
            manifest_row(
                run_id=run_id,
                dataset_name=display_dataset_name,
                case_id=evidence.prediction.case_id,
                pred_row=pred_row,
                feature_policy=feature_policy,
                model_name=str(ds["model_name"]),
                sampling_reason=str(selected_row["sampling_reason"]),
                evidence_available="full_external_report_evidence",
                notes=str(ds.get("role", "")),
            )
        )
        items.append({"evidence": evidence, "notes": rows[-1]["notes"]})
    return rows, items


def make_prediction_evidence(
    dataset_name: str,
    case_id: str,
    feature_policy: str,
    pred_row: pd.Series,
    model_name: str,
) -> PredictionEvidence:
    probabilities = {
        col.replace("prob_class_", ""): float(pred_row[col])
        for col in pred_row.index
        if str(col).startswith("prob_class_")
    }
    return PredictionEvidence(
        dataset_name=dataset_name,
        case_id=case_id,
        predicted_class=int(pred_row["y_pred"]),
        true_class=int(pred_row["y_true"]),
        class_probabilities=probabilities,
        confidence=float(pred_row.get("confidence", max(probabilities.values()) if probabilities else 0.0)),
        correctness=bool(pred_row.get("correct", int(pred_row["y_true"]) == int(pred_row["y_pred"]))),
        uncertainty_flag=bool(float(pred_row.get("confidence", max(probabilities.values()) if probabilities else 0.0)) < 0.60),
        model_name=model_name,
        feature_policy=feature_policy,
        leakage_safe_status="final_candidate_leakage_safe" if dataset_name == "inx_primary" else "external_research_candidate",
    )


def select_risk_aware_cases(
    predictions: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
    representative: pd.DataFrame,
    representative_case_col: str,
    extra_available_cases: set[str],
    shap_local_path: Path | None = None,
) -> List[Dict[str, Any]]:
    selected: Dict[int, str] = {}

    def add(sample_index: Any, reason: str) -> None:
        idx = int(sample_index)
        if idx in selected:
            if reason not in selected[idx]:
                selected[idx] = f"{selected[idx]};{reason}"
            return
        if len(selected) >= sample_size:
            return
        if idx in set(predictions["sample_index"].astype(int)):
            selected[idx] = reason

    if representative is not None and not representative.empty:
        for row in representative.itertuples(index=False):
            reason = str(getattr(row, representative_case_col, "representative_case"))
            add(getattr(row, "sample_index"), reason)

    for case_id in sorted(extra_available_cases):
        add(case_id, "existing_reason_code_evidence")

    correct = predictions[predictions["correct"].astype(bool)]
    wrong = predictions[~predictions["correct"].astype(bool)]
    if not correct.empty:
        add(correct.sort_values("confidence", ascending=False).iloc[0]["sample_index"], "correct_high_confidence")
        add(correct.sort_values("confidence", ascending=True).iloc[0]["sample_index"], "correct_low_confidence")
    if not wrong.empty:
        add(wrong.sort_values("confidence", ascending=False).iloc[0]["sample_index"], "misclassification_high_confidence")
        add(wrong.sort_values("confidence", ascending=True).iloc[0]["sample_index"], "misclassification_low_confidence")

    uncertain = predictions[predictions["confidence"].astype(float) < 0.60]
    if not uncertain.empty:
        add(uncertain.sort_values("confidence", ascending=True).iloc[0]["sample_index"], "low_confidence_uncertain")

    for target in sorted(predictions["y_true"].dropna().unique()):
        subset = predictions[predictions["y_true"] == target].sort_values("confidence", ascending=False)
        if not subset.empty:
            add(subset.iloc[0]["sample_index"], f"class_{int(target)}_coverage")

    if shap_local_path is not None and shap_local_path.exists():
        shap_df = pd.read_csv(shap_local_path)
        concentration = (
            shap_df.assign(abs_value=shap_df["abs_grouped_shap_value"].astype(float))
            .groupby("sample_index")["abs_value"]
            .agg(["max", "sum"])
            .reset_index()
        )
        concentration["concentration"] = concentration["max"] / concentration["sum"].replace(0, pd.NA)
        concentration = concentration.dropna(subset=["concentration"]).sort_values("concentration", ascending=False)
        if not concentration.empty:
            add(concentration.iloc[0]["sample_index"], "strong_shap_attribution_concentration")

    remaining = predictions[~predictions["sample_index"].astype(int).isin(selected.keys())].copy()
    if len(selected) < sample_size and not remaining.empty:
        shuffled = remaining.sample(frac=1.0, random_state=seed)
        for _, row in shuffled.groupby("y_true", group_keys=False).head(max(1, sample_size)).iterrows():
            add(row["sample_index"], "stratified_seeded_fill")
            if len(selected) >= sample_size:
                break
    if len(selected) < sample_size and not remaining.empty:
        for row in remaining.sample(frac=1.0, random_state=seed + 1).itertuples(index=False):
            add(getattr(row, "sample_index"), "seeded_fill")
            if len(selected) >= sample_size:
                break

    return [{"sample_index": idx, "sampling_reason": reason} for idx, reason in selected.items()]


def manifest_row(
    *,
    run_id: str,
    dataset_name: str,
    case_id: str,
    pred_row: pd.Series,
    feature_policy: str,
    model_name: str,
    sampling_reason: str,
    evidence_available: str,
    notes: str,
) -> Dict[str, Any]:
    confidence = float(pred_row.get("confidence", 0.0))
    correct = bool(pred_row.get("correct", int(pred_row["y_true"]) == int(pred_row["y_pred"])))
    return {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "case_id": case_id,
        "target": int(pred_row["y_true"]),
        "predicted_class": int(pred_row["y_pred"]),
        "confidence": confidence,
        "correctness": correct,
        "uncertainty_flag": bool(confidence < 0.60),
        "sampling_reason": sampling_reason,
        "feature_policy": feature_policy,
        "model_name": model_name,
        "evidence_available": evidence_available,
        "notes": notes,
    }


def reason_code_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for file in path.glob("*.json"):
        stem = file.stem
        suffix = stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            out.add(suffix)
    return out


def validate_explanation_payload(payload: Dict[str, Any]) -> Tuple[bool, str]:
    required = ["short_explanation", "detailed_explanation", "warnings", "requires_human_review"]
    missing = [key for key in required if key not in payload]
    if missing:
        return False, f"missing required keys: {missing}"
    if not isinstance(payload.get("warnings"), list):
        return False, "warnings must be a list"
    return True, ""


def write_faithfulness_summary(df: pd.DataFrame, path: Path) -> None:
    summary = summarize_faithfulness_metrics(df)
    by_dataset = df.groupby("dataset_name").agg(
        n_cases=("case_id", "count"),
        faithfulness_pass_rate=("faithfulness_pass", "mean"),
        mean_faithfulness_score=("faithfulness_score", "mean"),
        missing_warning_rate=("missing_warning_count", lambda s: float((s > 0).mean())),
    ).reset_index()
    failures = df[df["faithfulness_pass"] == False].head(10)  # noqa: E712
    lines = [
        "# Faithfulness Evaluation Summary",
        "",
        *[f"- {key}: {value}" for key, value in summary.items()],
        "",
        "## Per-Dataset Breakdown",
        "",
        *markdown_table(by_dataset),
        "",
        "## Examples of Failures",
        "",
    ]
    if failures.empty:
        lines.append("No faithfulness failures detected.")
    else:
        for row in failures.itertuples(index=False):
            lines.append(f"- `{row.case_id}`: {row.notes}")
    write_markdown(path, lines)


def summarize_faithfulness_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "n_cases": 0,
            "faithfulness_pass_rate": 0.0,
            "mean_faithfulness_score": 0.0,
            "unsupported_claim_rate": 0.0,
            "forbidden_claim_rate": 0.0,
            "missing_warning_rate": 0.0,
            "parsing_success_rate": 0.0,
        }
    forbidden_cols = [
        "causal_claim_count",
        "autonomous_decision_claim_count",
        "fairness_overclaim_count",
        "employee_prescription_count",
    ]
    return {
        "n_cases": int(len(df)),
        "faithfulness_pass_rate": float(df["faithfulness_pass"].mean()),
        "mean_faithfulness_score": float(df["faithfulness_score"].mean()),
        "unsupported_claim_rate": float(((df["unsupported_metric_count"] + df["unsupported_feature_count"]) > 0).mean()),
        "forbidden_claim_rate": float((df[forbidden_cols].sum(axis=1) > 0).mean()),
        "missing_warning_rate": float((df["missing_warning_count"] > 0).mean()),
        "parsing_success_rate": float((df["parsing_error"].fillna("") == "").mean()),
    }


def write_integrated_summary(
    *,
    run_id: str,
    run_mode: str,
    runtime_config: LLMRuntimeConfig,
    prompt_version: str,
    agent_version: str,
    manifest_df: pd.DataFrame,
    faithfulness_df: pd.DataFrame,
    agent_csv: Path | None,
    guardrail_csv: Path | None,
    output_dir: Path,
) -> Dict[str, Path]:
    metrics = summarize_faithfulness_metrics(faithfulness_df)
    agent_df = pd.read_csv(agent_csv) if agent_csv is not None and agent_csv.exists() else pd.DataFrame()
    guardrail_df = pd.read_csv(guardrail_csv) if guardrail_csv is not None and guardrail_csv.exists() else pd.DataFrame()
    supervisor = agent_df[agent_df["agent_name"] == "SupervisorGovernanceAgent"] if not agent_df.empty else pd.DataFrame()
    readiness = supervisor["status"].value_counts().rename_axis("readiness_status").reset_index(name="count") if not supervisor.empty else pd.DataFrame()
    explanation_agent = agent_df[agent_df["agent_name"] == "ExplanationComplianceAgent"] if not agent_df.empty else pd.DataFrame()
    agent_compliance_pass_rate = float((explanation_agent["status"] == "pass").mean()) if not explanation_agent.empty else 0.0

    unsafe = guardrail_df[guardrail_df["expected_behavior"] == "refuse_with_safe_alternative"] if not guardrail_df.empty else pd.DataFrame()
    safe = guardrail_df[guardrail_df["expected_behavior"] == "answer_with_governance_warnings"] if not guardrail_df.empty else pd.DataFrame()
    summary_row = {
        "run_id": run_id,
        "run_mode": run_mode,
        "real_llm_used": run_mode == "real",
        "n_cases": int(len(manifest_df)),
        "n_datasets": int(manifest_df["dataset_name"].nunique()) if not manifest_df.empty else 0,
        "llm_model_used": runtime_config.model,
        "llm_provider": runtime_config.provider,
        "prompt_version": prompt_version,
        "agent_system_version": agent_version,
        **metrics,
        "chatbot_unsafe_refusal_rate": float(unsafe["refused"].mean()) if not unsafe.empty else 0.0,
        "chatbot_safe_answer_rate": float(safe["pass"].mean()) if not safe.empty else 0.0,
        "agent_compliance_pass_rate": agent_compliance_pass_rate,
        "supervisor_readiness_distribution": readiness.to_dict(orient="records"),
    }
    csv_path = output_dir / "llm_agent_eval_summary.csv"
    md_path = output_dir / "llm_agent_eval_summary.md"
    pd.DataFrame([summary_row]).to_csv(csv_path, index=False)

    by_dataset = faithfulness_df.groupby("dataset_name").agg(
        n_cases=("case_id", "count"),
        faithfulness_pass_rate=("faithfulness_pass", "mean"),
        mean_faithfulness_score=("faithfulness_score", "mean"),
    ).reset_index() if not faithfulness_df.empty else pd.DataFrame()
    if run_mode == "real":
        limitation_lines = [
            "- This run used the real OpenAI-backed governed explanation path.",
            "- Automated LLM, faithfulness, agent, and chatbot checks do not replace human evaluation or legal/governance review.",
            "- Future larger or second-stage real LLM batches still require explicit approval because they incur API cost.",
        ]
    else:
        limitation_lines = [
            "- Dry-run outputs use the deterministic offline stub and are not manuscript-grade real LLM evidence.",
            "- Real OpenAI execution requires explicit approval and API-key configuration.",
            "- Automated checks do not replace human evaluation or legal/governance review.",
        ]
    lines = [
        "# LLM-Agent Evaluation Summary",
        "",
        f"Run ID: `{run_id}`",
        f"Run mode: `{run_mode}`",
        f"Real LLM used: `{run_mode == 'real'}`",
        f"LLM provider/model: `{runtime_config.provider}` / `{runtime_config.model}`",
        f"Prompt version: `{prompt_version}`",
        f"Agent system version: `{agent_version}`",
        "",
        "## Summary Metrics",
        "",
        *[f"- {key}: {value}" for key, value in summary_row.items() if key not in {"supervisor_readiness_distribution"}],
        "",
        "## Per-Dataset Breakdown",
        "",
        *markdown_table(by_dataset),
        "",
        "## Supervisor Readiness Distribution",
        "",
        *markdown_table(readiness),
        "",
        "## Limitations",
        "",
        *limitation_lines,
    ]
    write_markdown(md_path, lines)
    return {"llm_agent_eval_summary_csv": csv_path, "llm_agent_eval_summary_md": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configurable LLM-agent evaluation.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(args.config))
