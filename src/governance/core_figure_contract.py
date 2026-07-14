"""Fail-closed contract for the v2 manuscript core-figure plan.

This module freezes figure scope and lineage declarations.  Production
generation and closed-world validation are implemented separately; readiness
here authorizes execution but is not evidence that canonical figures exist.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


class CoreFigureContractError(ValueError):
    """Raised when the canonical core-figure plan is incomplete or drifts."""


CORE_FIGURE_PLAN_VERSION = "leakage_aware_core_figures_v2"
CORE_FIGURE_KEYS = tuple(f"figure_{number}" for number in range(1, 8))
CORE_FIGURE_IDENTITY_FIELDS = (
    "run_id",
    "config_hash",
    "scientific_input_hash",
    "source_tree_hash",
)
CORE_FIGURE_PROHIBITED_TOKENS = (
    "llm",
    "chatbot",
    "agent",
    "gxair",
    "counterfactual",
    "actionability",
    "local_reason_code",
    "ibm",
    "turnover",
    "historical",
)
CORE_FIGURE_BLOCKING_REASON = (
    "Production core-figure generation and package validation are implemented; "
    "canonical artifacts remain pending the final clean real-data build."
)


_EXPECTED_DEFINITIONS: dict[str, dict[str, Any]] = {
    "figure_1": {
        "number": 1,
        "figure_id": "study_design_and_leakage_aware_xai_pipeline",
        "output_stem": "figure_1_study_design_leakage_aware_pipeline",
        "title": "Study design and leakage-aware XAI audit pipeline",
        "figure_type": "directed_protocol_diagram",
        "sources": [
            {
                "stage": "run_inputs",
                "path": "run_inputs/canonical_config_snapshot.yaml",
            },
            {
                "stage": "run_inputs",
                "path": "run_inputs/input_contract.json",
            },
            {
                "stage": "shared_folds",
                "path": "shared_folds/fold_contract.json",
            },
        ],
        "source_data_filename": "figure_1_study_design_leakage_aware_pipeline_source.csv",
        "caption_filename": "figure_1_study_design_leakage_aware_pipeline_caption.md",
        "claim_boundary": (
            "Protocol architecture only; not deployment readiness or autonomous HR "
            "decision support."
        ),
    },
    "figure_2": {
        "number": 2,
        "figure_id": "feature_policy_leakage_ablation_tradeoff",
        "output_stem": "figure_2_feature_policy_tradeoff",
        "title": "Feature-policy and leakage-risk ablation trade-off",
        "figure_type": "paired_policy_tradeoff",
        "sources": [
            {
                "stage": "policy_ablation",
                "path": "policy_ablation/figure_leakage_policy_tradeoff_source.csv",
            }
        ],
        "source_data_filename": "figure_2_feature_policy_tradeoff_source.csv",
        "caption_filename": "figure_2_feature_policy_tradeoff_caption.md",
        "claim_boundary": (
            "The full-feature system is a diagnostic information-rich comparator; "
            "policy differences are leakage-risk sensitivity, not causal effects."
        ),
    },
    "figure_3": {
        "number": 3,
        "figure_id": "primary_xgboost_vs_three_baselines",
        "output_stem": "figure_3_xgboost_vs_baselines",
        "title": "Primary XGBoost versus three predeclared baselines",
        "figure_type": "model_comparison_interval_plot",
        "sources": [
            {
                "stage": "model_benchmarks",
                "path": "model_benchmarks/model_summary.csv",
            },
            {
                "stage": "model_benchmarks",
                "path": "model_benchmarks/paired_model_differences.csv",
            },
            {
                "stage": "model_benchmarks",
                "path": "model_benchmarks/baseline_xgboost_gate.json",
            },
        ],
        "source_data_filename": "figure_3_xgboost_vs_baselines_source.csv",
        "caption_filename": "figure_3_xgboost_vs_baselines_caption.md",
        "claim_boundary": (
            "Shared-fold predeclared OOF benchmark context only; not a universal model "
            "leaderboard."
        ),
    },
    "figure_4": {
        "number": 4,
        "figure_id": "predeclared_cross_fitted_sigmoid_calibration",
        "output_stem": "figure_4_cross_fitted_sigmoid_calibration",
        "title": "Predeclared cross-fitted sigmoid calibration",
        "figure_type": "reliability_and_metric_panel",
        "sources": [
            {
                "stage": "sigmoid_calibration",
                "path": "sigmoid_calibration/calibration_bins.csv",
            },
            {
                "stage": "sigmoid_calibration",
                "path": "sigmoid_calibration/calibration_method_comparison.csv",
            },
            {
                "stage": "sigmoid_calibration",
                "path": "sigmoid_calibration/calibration_metric_intervals.csv",
            },
            {
                "stage": "sigmoid_calibration",
                "path": "sigmoid_calibration/calibration_paired_differences.csv",
            },
            {
                "stage": "sigmoid_calibration",
                "path": "sigmoid_calibration/calibration_figure_source.json",
            },
        ],
        "source_data_filename": "figure_4_cross_fitted_sigmoid_calibration_source.csv",
        "caption_filename": "figure_4_cross_fitted_sigmoid_calibration_caption.md",
        "claim_boundary": (
            "Sigmoid is predeclared for probability calibration; probability quality "
            "and argmax predictive performance are reported separately."
        ),
    },
    "figure_5": {
        "number": 5,
        "figure_id": "global_grouped_oof_shap",
        "output_stem": "figure_5_global_grouped_oof_shap",
        "title": "Global grouped out-of-fold SHAP attribution",
        "figure_type": "horizontal_grouped_shap_bar",
        "sources": [
            {
                "stage": "oof_shap",
                "path": "oof_shap/global_grouped_shap_importance.csv",
            },
            {
                "stage": "oof_shap",
                "path": "oof_shap/shap_metadata.json",
            },
        ],
        "source_data_filename": "figure_5_global_grouped_oof_shap_source.csv",
        "caption_filename": "figure_5_global_grouped_oof_shap_caption.md",
        "claim_boundary": (
            "Grouped OOF SHAP is model attribution in the declared output space, not "
            "causality or employee prescription."
        ),
    },
    "figure_6": {
        "number": 6,
        "figure_id": "grouped_oof_shap_stability",
        "output_stem": "figure_6_oof_shap_stability",
        "title": "Grouped out-of-fold SHAP stability across outer folds",
        "figure_type": "descriptive_stability_panel",
        "sources": [
            {
                "stage": "oof_shap",
                "path": "oof_shap/fold_feature_rankings.csv",
            },
            {
                "stage": "oof_shap",
                "path": "oof_shap/shap_stability_pairwise.csv",
            },
            {
                "stage": "oof_shap",
                "path": "oof_shap/shap_stability_summary.csv",
            },
            {
                "stage": "oof_shap",
                "path": "oof_shap/shap_metadata.json",
            },
        ],
        "source_data_filename": "figure_6_oof_shap_stability_source.csv",
        "caption_filename": "figure_6_oof_shap_stability_caption.md",
        "claim_boundary": (
            "Fold-pair stability is descriptive because fold pairs are dependent; no "
            "population confidence interval is claimed."
        ),
    },
    "figure_7": {
        "number": 7,
        "figure_id": "hrdataset_v14_mapped_target_replication_summary",
        "output_stem": "figure_7_hrdataset_v14_mapped_target_replication",
        "title": "HRDataset_v14 independent mapped-target replication",
        "figure_type": "mapped_target_replication_summary",
        "sources": [
            {
                "stage": "external_replication",
                "path": "external_replication/target_support.csv",
            },
            {
                "stage": "external_replication",
                "path": "external_replication/raw_metric_intervals.csv",
            },
            {
                "stage": "external_replication",
                "path": "external_replication/calibration_metric_intervals.csv",
            },
            {
                "stage": "external_replication",
                "path": "external_replication/calibration_paired_differences.csv",
            },
            {
                "stage": "external_replication",
                "path": "external_replication/policy_pairwise_differences.csv",
            },
            {
                "stage": "external_replication",
                "path": "external_replication/external_replication_metadata.json",
            },
        ],
        "source_data_filename": "figure_7_hrdataset_v14_mapped_target_replication_source.csv",
        "caption_filename": "figure_7_hrdataset_v14_mapped_target_replication_caption.md",
        "claim_boundary": (
            "Independent mapped-target replication only; not locked model transport, "
            "universal external validation, fairness, causality, or deployment evidence."
        ),
    },
}


def expected_core_figure_plan() -> dict[str, Any]:
    """Return a mutable copy of the frozen v2 core-figure plan."""

    return {
        "plan_version": CORE_FIGURE_PLAN_VERSION,
        "scope": "core",
        "release_ready": True,
        "blocking_reason": CORE_FIGURE_BLOCKING_REASON,
        "publication_dpi": 300,
        "output_formats": ["png", "svg"],
        "font_family": "DejaVu Sans",
        "identity_fields": list(CORE_FIGURE_IDENTITY_FIELDS),
        "source_hash_algorithm": "sha256",
        "source_data_subdirectory": "source_data",
        "caption_subdirectory": "captions",
        "definitions": deepcopy(_EXPECTED_DEFINITIONS),
    }


def _portable_source_path(value: Any, *, stage: str, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or value.endswith("/")
    ):
        raise CoreFigureContractError(f"{context} must be a portable file path.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value in {".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
        or len(path.parts) < 2
        or path.parts[0] != stage
    ):
        raise CoreFigureContractError(
            f"{context} must be a stage-relative portable file below {stage!r}."
        )
    return value


def _validate_definition_structure(
    definitions: Mapping[str, Any],
    *,
    core_stages: Sequence[str],
) -> None:
    if tuple(definitions) != CORE_FIGURE_KEYS:
        raise CoreFigureContractError(
            "Core figures must define exactly figure_1 through figure_7 in order."
        )
    if "core_figures" not in core_stages:
        raise CoreFigureContractError("The core stage graph must contain core_figures.")
    figure_stage_index = core_stages.index("core_figures")
    if figure_stage_index != len(core_stages) - 1:
        raise CoreFigureContractError("core_figures must be the final core stage.")
    allowed_sources = {"run_inputs", *core_stages[:figure_stage_index]}

    for key, raw_definition in definitions.items():
        if not isinstance(raw_definition, Mapping):
            raise CoreFigureContractError(f"{key} must be a mapping.")
        sources = raw_definition.get("sources")
        if not isinstance(sources, list) or not sources:
            raise CoreFigureContractError(f"{key}.sources must be a non-empty list.")
        seen_paths: set[str] = set()
        for index, raw_source in enumerate(sources):
            context = f"{key}.sources[{index}]"
            if not isinstance(raw_source, Mapping) or set(raw_source) != {"stage", "path"}:
                raise CoreFigureContractError(
                    f"{context} must contain exactly stage and path."
                )
            stage = raw_source.get("stage")
            if not isinstance(stage, str) or stage not in allowed_sources:
                raise CoreFigureContractError(
                    f"{context}.stage must precede core_figures in the core graph."
                )
            path = _portable_source_path(raw_source.get("path"), stage=stage, context=context)
            if path in seen_paths:
                raise CoreFigureContractError(f"{key} contains duplicate source path {path!r}.")
            seen_paths.add(path)


def _recursive_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return "\n".join(
            item
            for key, child in value.items()
            for item in (_recursive_text(key), _recursive_text(child))
        )
    if isinstance(value, (list, tuple)):
        return "\n".join(_recursive_text(item) for item in value)
    return value if isinstance(value, str) else ""


def validate_core_figure_plan(
    figures: Mapping[str, Any],
    *,
    core_stages: Sequence[str],
    core_scope_release_ready: Any,
    core_scope_blocking_reason: Any,
) -> None:
    """Validate exact v2 figure scope, sources, ordering, and release boundary."""

    if not isinstance(figures, Mapping):
        raise CoreFigureContractError("manuscript_final.figures must be a mapping.")
    expected = expected_core_figure_plan()
    definitions = figures.get("definitions")
    if not isinstance(definitions, Mapping):
        raise CoreFigureContractError("figures.definitions must be a mapping.")
    _validate_definition_structure(definitions, core_stages=tuple(core_stages))

    normalized_text = _recursive_text(figures).casefold()
    prohibited = sorted(
        token for token in CORE_FIGURE_PROHIBITED_TOKENS if token in normalized_text
    )
    if prohibited:
        raise CoreFigureContractError(
            f"Core figure plan contains prohibited scope tokens: {prohibited}."
        )
    if dict(figures) != expected:
        raise CoreFigureContractError(
            "manuscript_final.figures differs from the frozen leakage-aware core plan."
        )
    if core_scope_release_ready is not True:
        raise CoreFigureContractError(
            "The implemented core figure stage requires release_ready=true before execution."
        )
    if not isinstance(core_scope_blocking_reason, str) or not core_scope_blocking_reason.strip():
        raise CoreFigureContractError(
            "The core scope requires a non-empty canonical-execution readiness note."
        )
