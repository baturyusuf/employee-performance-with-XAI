from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments import build_manuscript_evidence as manuscript_build
from src.utils.config_loader import load_config


CONFIG_PATH = Path("configs/manuscript_final.yaml")

EXPECTED_CORE_STAGE_ORDER = (
    "shared_folds",
    "model_benchmarks",
    "policy_ablation",
    "sigmoid_calibration",
    "oof_shap",
    "subgroup_proxy",
    "external_replication",
    "dataset_cards",
    "core_tables",
    "core_figures",
)
EXPECTED_SUPPLEMENTARY_STAGE_ORDER = (
    "heuristic_counterfactual",
    "external_robustness",
    "dataset_cards",
    "supplementary_tables",
)
EXPECTED_CORE_DATASETS = {"inx_primary", "hrdataset_v14"}
EXPECTED_SUPPLEMENTARY_DATASETS = {
    "inx_primary",
    "ibm_hr_analytics",
    "ibm_hr_analytics_attrition",
    "employee_turnover",
}
PROHIBITED_ACCEPTED_STAGE_TOKENS = ("llm", "chatbot", "agent", "openai")


def _scope_settings() -> dict:
    settings = load_config(CONFIG_PATH)["manuscript_final"]
    scopes = settings.get("evidence_scopes")
    assert isinstance(scopes, dict), "Canonical config must declare evidence_scopes."
    assert set(scopes) == {"core", "supplementary"}
    return scopes


def test_code_and_config_publish_the_same_explicit_stage_graphs() -> None:
    scopes = _scope_settings()

    assert manuscript_build.CORE_STAGE_ORDER == EXPECTED_CORE_STAGE_ORDER
    assert manuscript_build.SUPPLEMENTARY_STAGE_ORDER == EXPECTED_SUPPLEMENTARY_STAGE_ORDER
    assert tuple(scopes["core"]["stages"]) == EXPECTED_CORE_STAGE_ORDER
    assert tuple(scopes["supplementary"]["stages"]) == EXPECTED_SUPPLEMENTARY_STAGE_ORDER
    assert len(set(manuscript_build.CORE_STAGE_ORDER)) == len(manuscript_build.CORE_STAGE_ORDER)
    assert len(set(manuscript_build.SUPPLEMENTARY_STAGE_ORDER)) == len(
        manuscript_build.SUPPLEMENTARY_STAGE_ORDER
    )


def test_scope_dataset_contracts_match_fixed_scientific_roles() -> None:
    scopes = _scope_settings()

    assert set(scopes["core"]["dataset_keys"]) == EXPECTED_CORE_DATASETS
    assert set(scopes["supplementary"]["dataset_keys"]) == EXPECTED_SUPPLEMENTARY_DATASETS
    assert "hrdataset_v14" not in scopes["supplementary"]["dataset_keys"]
    assert {
        "ibm_hr_analytics",
        "ibm_hr_analytics_attrition",
        "employee_turnover",
    }.isdisjoint(scopes["core"]["dataset_keys"])


def test_accepted_stage_graphs_exclude_llm_chatbot_agents_and_api_stages() -> None:
    for scope_name, stage_order in {
        "core": manuscript_build.CORE_STAGE_ORDER,
        "supplementary": manuscript_build.SUPPLEMENTARY_STAGE_ORDER,
    }.items():
        folded = " ".join(stage_order).casefold()
        for token in PROHIBITED_ACCEPTED_STAGE_TOKENS:
            assert token not in folded, f"{scope_name} graph contains prohibited token {token!r}"

    core = set(manuscript_build.CORE_STAGE_ORDER)
    supplementary = set(manuscript_build.SUPPLEMENTARY_STAGE_ORDER)
    assert "heuristic_counterfactual" not in core
    assert "external_robustness" not in core
    assert {"heuristic_counterfactual", "external_robustness"}.issubset(supplementary)


@pytest.mark.parametrize("scope_name", ["core", "supplementary"])
def test_incomplete_scope_graphs_fail_closed_before_artifact_execution(scope_name: str) -> None:
    scopes = _scope_settings()
    assert scopes[scope_name]["release_ready"] is False

    with pytest.raises(manuscript_build.ManuscriptBuildError, match="release.ready|not ready"):
        manuscript_build.validate_scope_release_ready(
            scopes,
            scope_name,
        )
