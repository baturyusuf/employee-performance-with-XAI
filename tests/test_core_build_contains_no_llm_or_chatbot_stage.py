from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.experiments import build_manuscript_evidence as manuscript_build
from src.utils.config_loader import PROJECT_ROOT, load_config


FORBIDDEN_IMPORT_PREFIXES = (
    "src.llm",
    "src.chatbot",
    "src.agents",
    "openai",
    "agents",
)
REQUIRED_CORE_FORBIDDEN_OUTPUT_PREFIXES = {
    "heuristic_counterfactual/",
    "counterfactual/",
    "external_robustness/",
    "external/ibm_performance/",
    "external/ibm_attrition/",
    "external/employee_turnover/",
    "llm/",
    "chatbot/",
    "agents/",
    "agent_audits/",
    "historical/",
}


def _module_path(module_name: str) -> Path | None:
    candidate = PROJECT_ROOT.joinpath(*module_name.split("."))
    module_file = candidate.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = candidate / "__init__.py"
    return package_file if package_file.is_file() else None


def _reachable_import_edges(start_module: str) -> list[tuple[str, str, str, int]]:
    """Inspect imports recursively, including imports nested inside functions."""

    visited: set[str] = set()
    edges: list[tuple[str, str, str, int]] = []

    def visit(module_name: str) -> None:
        if module_name in visited:
            return
        visited.add(module_name)
        path = _module_path(module_name)
        if path is None:
            return
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module]
            for dependency in imported:
                edges.append(
                    (
                        module_name,
                        dependency,
                        path.relative_to(PROJECT_ROOT).as_posix(),
                        int(getattr(node, "lineno", 0)),
                    )
                )
                if dependency.startswith("src."):
                    visit(dependency)

    visit(start_module)
    return edges


def test_core_entrypoint_static_import_graph_has_no_llm_chatbot_agent_or_openai_path() -> None:
    edges = _reachable_import_edges("src.experiments.build_manuscript_evidence")
    forbidden = [
        edge
        for edge in edges
        if any(
            edge[1] == prefix or edge[1].startswith(prefix + ".")
            for prefix in FORBIDDEN_IMPORT_PREFIXES
        )
    ]

    assert not forbidden, "Core-reachable prohibited imports:\n" + "\n".join(
        f"{source} -> {dependency} ({path}:{line})"
        for source, dependency, path, line in forbidden
    )

def test_core_scope_declares_all_legacy_and_out_of_scope_output_prefixes_forbidden() -> None:
    settings = load_config("configs/manuscript_final.yaml")["manuscript_final"]
    core = settings["evidence_scopes"]["core"]
    observed = {str(value).replace("\\", "/") for value in core["forbidden_output_prefixes"]}

    assert REQUIRED_CORE_FORBIDDEN_OUTPUT_PREFIXES.issubset(observed)


@pytest.mark.parametrize(
    "relative_path",
    sorted(REQUIRED_CORE_FORBIDDEN_OUTPUT_PREFIXES),
)
def test_core_artifact_contract_rejects_out_of_scope_paths(relative_path: str) -> None:
    candidate = relative_path + "unexpected_scientific_artifact.csv"

    with pytest.raises(manuscript_build.ManuscriptBuildError, match="scope|forbidden"):
        manuscript_build.validate_scope_artifact_paths("core", [candidate])


def test_core_artifact_contract_accepts_only_in_scope_examples() -> None:
    manuscript_build.validate_scope_artifact_paths(
        "core",
        [
            "shared_folds/fold_assignments.csv",
            "model_benchmarks/model_summary.csv",
            "policy_ablation/policy_summary.csv",
            "sigmoid_calibration/reliability_bins.csv",
            "oof_shap/global_grouped_shap_importance.csv",
            "subgroup_proxy/subgroup_diagnostics.csv",
            "external_replication/hrdataset_v14/performance_metrics.csv",
            "dataset_cards/dataset_cards.json",
            "core_tables/table_manifest.csv",
            "core_figures/figure_manifest.csv",
        ],
    )
