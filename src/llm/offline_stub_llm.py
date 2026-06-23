from __future__ import annotations

from typing import Any, Dict, List

from src.llm.llm_client import LLMClient


def _fmt_prob(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "unavailable"


class OfflineStubLLM(LLMClient):
    """Deterministic governed explanation generator used for tests/reproducibility."""

    def generate_json(self, system_prompt: str, user_prompt: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        prediction = evidence.get("prediction", {})
        shap = evidence.get("shap") or {}
        fairness = evidence.get("fairness") or {}
        calibration = evidence.get("calibration") or {}
        counterfactual = evidence.get("counterfactual") or {}
        leakage = evidence.get("leakage") or {}
        governance = evidence.get("governance", {})

        case_id = str(prediction.get("case_id", "unknown"))
        predicted_class = prediction.get("predicted_class", "unavailable")
        confidence = _fmt_prob(prediction.get("confidence"))
        top_pos = [item.get("feature", "unknown") for item in shap.get("top_positive_features", [])[:3]]
        top_neg = [item.get("feature", "unknown") for item in shap.get("top_negative_features", [])[:3]]
        shap_summary = ", ".join(top_pos) if top_pos else "SHAP evidence unavailable"
        negative_summary = ", ".join(top_neg) if top_neg else "opposing SHAP evidence unavailable"

        warnings: List[Dict[str, str]] = [
            {
                "type": "causality",
                "severity": "high",
                "message": "SHAP is attribution, not causality.",
            },
            {
                "type": "deployment",
                "severity": "high",
                "message": "This is decision support only and is not for autonomous HR decisions.",
            },
            {
                "type": "deployment",
                "severity": "high",
                "message": "Prediction requires human review.",
            },
        ]

        if leakage:
            warnings.append(
                {
                    "type": "leakage",
                    "severity": "high",
                    "message": leakage.get("leakage_warning", "Leakage evidence is unavailable."),
                }
            )
        if fairness:
            for message in fairness.get("proxy_risk_warnings", []):
                warnings.append({"type": "proxy", "severity": "high", "message": message})
        if calibration:
            warnings.append(
                {
                    "type": "calibration",
                    "severity": "medium",
                    "message": calibration.get("calibration_warning", "Calibration evidence is unavailable."),
                }
            )
        if counterfactual:
            warnings.append(
                {
                    "type": "actionability",
                    "severity": "high",
                    "message": counterfactual.get("warning", "Counterfactual evidence is unavailable."),
                }
            )

        short = (
            f"For case {case_id}, the XGBoost model predicted performance class {predicted_class} "
            f"with approximate confidence {confidence}. The largest model attributions were associated "
            f"with {shap_summary}."
        )
        detailed = (
            f"{short} Opposing attributions included {negative_summary}. "
            "These attributions are not causal effects. "
            f"Calibration evidence: log loss={_fmt_prob(calibration.get('log_loss'))}, "
            f"Brier={_fmt_prob(calibration.get('brier_score'))}, "
            f"ECE={_fmt_prob(calibration.get('expected_calibration_error'))}. "
            "Fairness/proxy evidence indicates that department removal does not prove fairness and JobRole may proxy Department. "
            "Counterfactual evidence should be treated as model-level scenarios, not employee instructions. "
            f"Intended use: {governance.get('intended_use', 'unavailable')}"
        )
        return {
            "case_id": case_id,
            "short_explanation": short,
            "detailed_explanation": detailed,
            "warnings": warnings,
            "unsupported_claims_detected": [],
            "requires_human_review": True,
        }

