from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from src.llm.evidence_preflight import (
    EvidencePreflightError,
    build_evidence_preflight_report,
    enforce_real_llm_preflight,
)
from src.llm.run_llm_agent_evaluation import run, run_preflight
from tests.test_llm_case_evidence_completeness import complete_evidence


def test_real_runner_blocks_incomplete_evidence_before_client_construction(tmp_path: Path) -> None:
    output_dir = tmp_path / "llm_outputs"
    config_path = tmp_path / "real_eval.json"
    config_path.write_text(
        json.dumps(
            {
                "llm_agent_eval": {
                    "run_id_prefix": "preflight_block_test",
                    "run_mode": "real",
                    "output_dir": str(output_dir),
                    "llm": {"provider": "openai", "model": "paid-model-must-not-be-called"},
                    "evaluation_flags": {
                        "run_agent_audit": False,
                        "run_chatbot_guardrail_tests": False,
                    },
                    "datasets": [
                        {
                            "dataset_name": "inx_primary",
                            "enabled": True,
                            "source": "internal",
                            "sample_size": 1,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    evidence = replace(complete_evidence("blocked_case"), shap=None)
    manifest = {
        "run_id": "patched_at_runtime",
        "dataset_name": "inx_primary",
        "case_id": "blocked_case",
        "target": 3,
        "predicted_class": 3,
        "confidence": 0.7,
        "correctness": True,
        "uncertainty_flag": False,
        "sampling_reason": "unit_test",
        "feature_policy": evidence.prediction.feature_policy,
        "model_name": "xgboost",
        "evidence_available": "partial_missing_local_shap",
        "notes": "unit test",
    }

    with patch(
        "src.llm.run_llm_agent_evaluation.build_manifest_and_evidence",
        return_value=([manifest], [{"evidence": evidence, "notes": "unit test"}]),
    ), patch("src.llm.run_llm_agent_evaluation.GovernedExplainer") as explainer:
        with pytest.raises(EvidencePreflightError, match="blocked"):
            run(str(config_path))
        explainer.assert_not_called()

    reports = list(output_dir.glob("preflight/*/llm_evidence_preflight.json"))
    assert len(reports) == 1
    preflight = json.loads(reports[0].read_text(encoding="utf-8"))
    assert preflight["cases_incomplete"] == 1
    assert preflight["real_api_execution_allowed"] is False
    assert preflight["blocking_reasons"]


def test_real_preflight_allows_only_explicit_separate_missing_evidence_stratum() -> None:
    incomplete = replace(complete_evidence("missing_case"), shap=None)
    blocked = build_evidence_preflight_report(
        [{"evidence": incomplete}],
        run_id="blocked",
        run_mode="real",
    )
    with pytest.raises(EvidencePreflightError):
        enforce_real_llm_preflight(blocked)

    explicitly_separated = build_evidence_preflight_report(
        [{"evidence": incomplete}],
        run_id="separate_stratum",
        run_mode="real",
        missing_evidence_stratum={
            "enabled": True,
            "name": "declared_missing_evidence_handling",
            "case_ids": ["missing_case"],
            "include_in_primary_compliance_metrics": False,
        },
    )
    enforce_real_llm_preflight(explicitly_separated)
    assert explicitly_separated["real_api_execution_allowed"] is True
    assert explicitly_separated["cases"][0]["evidence_stratum"] == "declared_missing_evidence_handling"
    assert explicitly_separated["cases"][0]["complete"] is False


def test_preflight_only_mode_never_constructs_client(tmp_path: Path) -> None:
    output_dir = tmp_path / "preflight_only"
    config_path = tmp_path / "preflight_only.json"
    config_path.write_text(
        json.dumps(
            {
                "llm_agent_eval": {
                    "run_id_prefix": "preflight_only_test",
                    "run_mode": "real",
                    "output_dir": str(output_dir),
                    "datasets": [
                        {
                            "dataset_name": "inx_primary",
                            "enabled": True,
                            "source": "internal",
                            "sample_size": 1,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    evidence = replace(complete_evidence("preflight_only_case"), shap=None)
    manifest = {
        "case_id": "preflight_only_case",
        "dataset_name": "inx_primary",
    }
    with patch(
        "src.llm.run_llm_agent_evaluation.build_manifest_and_evidence",
        return_value=([manifest], [{"evidence": evidence, "notes": "unit test"}]),
    ), patch("src.llm.run_llm_agent_evaluation.GovernedExplainer") as explainer:
        outputs = run_preflight(str(config_path))
        explainer.assert_not_called()

    report = json.loads(outputs["preflight_json"].read_text(encoding="utf-8"))
    assert report["execution_intent"] == "preflight_only_no_llm_calls"
    assert report["api_call_attempted"] is False
    assert report["real_api_execution_allowed"] is False
