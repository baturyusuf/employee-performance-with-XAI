from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.governance.core_figure_contract import (
    CORE_FIGURE_KEYS,
    CORE_FIGURE_PROHIBITED_TOKENS,
    CoreFigureContractError,
    expected_core_figure_plan,
    validate_core_figure_plan,
)
from src.governance.manuscript_contract import (
    DEFAULT_CONFIG_PATH,
    ManuscriptConfigError,
    evidence_scope_contract,
    load_manuscript_config,
    validate_manuscript_config,
)


def _canonical_context() -> tuple[dict, list[str], str]:
    config = load_manuscript_config(DEFAULT_CONFIG_PATH)
    settings = config["manuscript_final"]
    core = settings["evidence_scopes"]["core"]
    return config, list(core["stages"]), str(core["blocking_reason"])


def test_canonical_core_figure_plan_is_exact_and_release_blocking() -> None:
    config, core_stages, core_blocking_reason = _canonical_context()
    figures = config["manuscript_final"]["figures"]

    assert figures == expected_core_figure_plan()
    assert tuple(figures["definitions"]) == CORE_FIGURE_KEYS
    assert figures["release_ready"] is False
    assert config["manuscript_final"]["evidence_scopes"]["core"]["release_ready"] is False
    assert core_stages[-1] == "core_figures"
    assert core_blocking_reason.strip()


def test_canonical_plan_contains_no_excluded_core_subject() -> None:
    config, _, _ = _canonical_context()
    text = repr(config["manuscript_final"]["figures"]).casefold()
    assert not [token for token in CORE_FIGURE_PROHIBITED_TOKENS if token in text]


@pytest.mark.parametrize(
    "mutation",
    [
        "remove_figure",
        "add_figure",
        "renumber",
        "legacy_subject",
        "absolute_source",
        "traversal_source",
        "wrong_stage_prefix",
        "duplicate_source",
        "figure_release_ready",
        "core_release_ready",
    ],
)
def test_figure_plan_drift_fails_closed(mutation: str) -> None:
    config, core_stages, core_blocking_reason = _canonical_context()
    figures = deepcopy(config["manuscript_final"]["figures"])
    core_release_ready = False

    if mutation == "remove_figure":
        figures["definitions"].pop("figure_7")
    elif mutation == "add_figure":
        figures["definitions"]["figure_8"] = deepcopy(
            figures["definitions"]["figure_7"]
        )
    elif mutation == "renumber":
        figures["definitions"]["figure_2"]["number"] = 7
    elif mutation == "legacy_subject":
        figures["definitions"]["figure_1"]["title"] = "LLM governance dashboard"
    elif mutation == "absolute_source":
        figures["definitions"]["figure_2"]["sources"][0]["path"] = (
            "C:/private/policy.csv"
        )
    elif mutation == "traversal_source":
        figures["definitions"]["figure_2"]["sources"][0]["path"] = (
            "policy_ablation/../stale.csv"
        )
    elif mutation == "wrong_stage_prefix":
        figures["definitions"]["figure_2"]["sources"][0]["path"] = (
            "model_benchmarks/figure_leakage_policy_tradeoff_source.csv"
        )
    elif mutation == "duplicate_source":
        source = deepcopy(figures["definitions"]["figure_3"]["sources"][0])
        figures["definitions"]["figure_3"]["sources"].append(source)
    elif mutation == "figure_release_ready":
        figures["release_ready"] = True
    elif mutation == "core_release_ready":
        core_release_ready = True
    else:  # pragma: no cover - guarded by the parameter list
        raise AssertionError(mutation)

    with pytest.raises(CoreFigureContractError):
        validate_core_figure_plan(
            figures,
            core_stages=core_stages,
            core_scope_release_ready=core_release_ready,
            core_scope_blocking_reason=core_blocking_reason,
        )


def test_general_config_validation_wires_the_figure_contract() -> None:
    config, _, _ = _canonical_context()
    mutated = deepcopy(config)
    mutated["manuscript_final"]["figures"]["definitions"]["figure_6"][
        "claim_boundary"
    ] = "Independent fold-pair population confidence interval."

    with pytest.raises(ManuscriptConfigError, match="frozen leakage-aware core plan"):
        validate_manuscript_config(mutated)


def test_declared_sources_are_portable_and_precede_core_figures() -> None:
    config, core_stages, _ = _canonical_context()
    definitions = config["manuscript_final"]["figures"]["definitions"]
    core_figure_index = core_stages.index("core_figures")
    allowed_stages = {"run_inputs", *core_stages[:core_figure_index]}

    for definition in definitions.values():
        seen: set[str] = set()
        for source in definition["sources"]:
            path = source["path"]
            assert source["stage"] in allowed_stages
            assert path.startswith(f"{source['stage']}/")
            assert not Path(path).is_absolute()
            assert "\\" not in path
            assert not any(part in {".", ".."} for part in path.split("/"))
            assert path not in seen
            seen.add(path)


def test_core_scope_contract_still_has_no_figure_runner_claim() -> None:
    config, _, _ = _canonical_context()
    contract = evidence_scope_contract(config, "core")
    assert contract["stages"][-1] == "core_figures"
    assert config["manuscript_final"]["figures"]["release_ready"] is False
