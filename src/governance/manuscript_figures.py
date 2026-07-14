"""Legacy v1 governance figures retained outside the canonical v2 core build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

from src.core.io_utils import ensure_dir
from src.governance.manuscript_contract import canonical_config_hash, load_manuscript_config


FIGURE_STEMS = {
    1: "figure_1_governance_architecture",
    2: "figure_2_structured_evidence_flow",
    3: "figure_3_multi_agent_audit",
    4: "figure_4_gxair_readiness_dashboard",
    5: "figure_5_calibration_ordinal_error",
    6: "figure_6_global_grouped_shap",
    7: "figure_7_local_reason_code",
}


class ManuscriptFigureError(RuntimeError):
    """Raised when manuscript figure sources or outputs are incomplete."""


def _save(fig: plt.Figure, output_dir: Path, stem: str, *, dpi: int, run_id: str, config_hash: str) -> Dict[str, Path]:
    description = f"run_id={run_id}; config_hash={config_hash}"
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=dpi, metadata={"Description": description})
    fig.savefig(svg, format="svg", metadata={"Title": stem.replace("_", " "), "Description": description})
    plt.close(fig)
    return {"png": png, "svg": svg}


def _box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    fontsize: float = 9,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor=facecolor,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = "#555555") -> None:
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=color))


def _graph_source(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    source_dir: Path,
    stem: str,
    *,
    run_id: str,
    config_hash: str,
) -> Dict[str, Path]:
    node_frame = pd.DataFrame(nodes)
    edge_frame = pd.DataFrame(edges)
    node_frame.insert(0, "config_hash", config_hash)
    node_frame.insert(0, "run_id", run_id)
    edge_frame.insert(0, "config_hash", config_hash)
    edge_frame.insert(0, "run_id", run_id)
    node_path = source_dir / f"{stem}_nodes.csv"
    edge_path = source_dir / f"{stem}_edges.csv"
    node_frame.to_csv(node_path, index=False)
    edge_frame.to_csv(edge_path, index=False)
    return {"nodes": node_path, "edges": edge_path}


def figure_1(
    output_dir: Path,
    source_dir: Path,
    *,
    run_id: str,
    config_hash: str,
    dpi: int,
) -> Dict[str, Path]:
    nodes = [
        {"node_id": "data", "label": "Versioned datasets\n+ provenance + roles", "layer": "inputs", "x": 0.04, "y": 0.68},
        {"node_id": "policy", "label": "Canonical policy\n+ task schema + folds", "layer": "inputs", "x": 0.04, "y": 0.36},
        {"node_id": "model", "label": "OOF XGBoost\n+ fold-safe preprocessing", "layer": "prediction", "x": 0.29, "y": 0.52},
        {"node_id": "calibration", "label": "Nested\n+ calibration", "layer": "evidence", "x": 0.53, "y": 0.76},
        {"node_id": "shap", "label": "Grouped SHAP\n+ global/local/stability", "layer": "evidence", "x": 0.53, "y": 0.55},
        {"node_id": "fairness", "label": "Fairness + proxy\n+ support + uncertainty", "layer": "evidence", "x": 0.53, "y": 0.34},
        {"node_id": "counterfactual", "label": "OOF counterfactual\n+ actionability", "layer": "evidence", "x": 0.53, "y": 0.13},
        {"node_id": "case", "label": "CompleteCaseEvidence\n+ validation gate", "layer": "governance", "x": 0.76, "y": 0.55},
        {"node_id": "llm", "label": "Governed explanation\n+ LLM is not predictor", "layer": "governance", "x": 0.76, "y": 0.31},
        {"node_id": "audit", "label": "Deterministic agents\n+ guardrails + readiness", "layer": "governance", "x": 0.76, "y": 0.07},
    ]
    edges = [
        {"source": "data", "target": "model", "label": "validated inputs"},
        {"source": "policy", "target": "model", "label": "one contract"},
        *[{"source": "model", "target": target, "label": "OOF evidence"} for target in ("calibration", "shap", "fairness", "counterfactual")],
        *[{"source": source, "target": "case", "label": "structured evidence"} for source in ("calibration", "shap", "fairness", "counterfactual")],
        {"source": "case", "target": "llm", "label": "complete only"},
        {"source": "llm", "target": "audit", "label": "deterministic checks"},
    ]
    sources = _graph_source(nodes, edges, source_dir, FIGURE_STEMS[1], run_id=run_id, config_hash=config_hash)
    fig, axis = plt.subplots(figsize=(14, 7.6))
    fig.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.07)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    colors = {"inputs": "#E9F2F9", "prediction": "#D7EBE8", "evidence": "#FFF0D9", "governance": "#EDE3F5"}
    positions: Dict[str, tuple[float, float]] = {}
    for node in nodes:
        width, height = (0.18, 0.15) if node["layer"] != "evidence" else (0.18, 0.13)
        _box(axis, node["x"], node["y"], width, height, node["label"], facecolor=colors[node["layer"]])
        positions[node["node_id"]] = (node["x"], node["y"])
    _arrow(axis, (0.22, 0.755), (0.29, 0.62))
    _arrow(axis, (0.22, 0.435), (0.29, 0.57))
    for y in (0.825, 0.615, 0.405, 0.195):
        _arrow(axis, (0.47, 0.59), (0.53, y))
        _arrow(axis, (0.71, y), (0.76, 0.62))
    _arrow(axis, (0.85, 0.55), (0.85, 0.46))
    _arrow(axis, (0.85, 0.31), (0.85, 0.22))
    axis.text(0.5, 0.96, "Figure 1. HR-XAI governance architecture under one run contract", ha="center", fontsize=15, weight="bold")
    axis.text(0.5, 0.015, "Research-grade decision support only — no autonomous HR decisions", ha="center", fontsize=9, color="#8B1E3F")
    outputs = _save(fig, output_dir, FIGURE_STEMS[1], dpi=dpi, run_id=run_id, config_hash=config_hash)
    return {**outputs, **{f"source_{key}": value for key, value in sources.items()}}


def figure_2(
    output_dir: Path,
    source_dir: Path,
    *,
    run_id: str,
    config_hash: str,
    dpi: int,
) -> Dict[str, Path]:
    nodes = [
        {"node_id": "prediction", "label": "Raw OOF\nprediction", "layer": "model", "x": 0.03, "y": 0.67},
        {"node_id": "probabilities", "label": "Calibrated\nprobabilities", "layer": "evidence", "x": 0.20, "y": 0.78},
        {"node_id": "local_shap", "label": "Local grouped\nSHAP", "layer": "evidence", "x": 0.20, "y": 0.59},
        {"node_id": "global_shap", "label": "Global + stable\nSHAP context", "layer": "evidence", "x": 0.20, "y": 0.40},
        {"node_id": "fairness", "label": "Fairness + proxy\ncontext", "layer": "evidence", "x": 0.20, "y": 0.21},
        {"node_id": "counterfactual", "label": "Counterfactual\nactionability", "layer": "evidence", "x": 0.20, "y": 0.02},
        {"node_id": "complete", "label": "CompleteCaseEvidence\nschema + hashes", "layer": "gate", "x": 0.49, "y": 0.45},
        {"node_id": "explanation", "label": "Governed\nexplanation", "layer": "output", "x": 0.72, "y": 0.58},
        {"node_id": "checker", "label": "Deterministic\nfaithfulness checker", "layer": "output", "x": 0.72, "y": 0.31},
    ]
    edges = [
        {"source": "prediction", "target": item, "label": "case identity"}
        for item in ("probabilities", "local_shap", "global_shap", "fairness", "counterfactual")
    ] + [
        {"source": item, "target": "complete", "label": "required field"}
        for item in ("probabilities", "local_shap", "global_shap", "fairness", "counterfactual")
    ] + [
        {"source": "complete", "target": "explanation", "label": "preflight pass"},
        {"source": "explanation", "target": "checker", "label": "claims + warnings"},
        {"source": "checker", "target": "explanation", "label": "block/flag"},
    ]
    sources = _graph_source(nodes, edges, source_dir, FIGURE_STEMS[2], run_id=run_id, config_hash=config_hash)
    fig, axis = plt.subplots(figsize=(14, 7.4))
    fig.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.07)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    palette = {"model": "#D7EBE8", "evidence": "#FFF0D9", "gate": "#EDE3F5", "output": "#E9F2F9"}
    for node in nodes:
        _box(axis, node["x"], node["y"], 0.18, 0.13, node["label"], facecolor=palette[node["layer"]])
    for y in (0.845, 0.655, 0.465, 0.275, 0.085):
        _arrow(axis, (0.21, 0.735), (0.20, y))
        _arrow(axis, (0.38, y), (0.49, 0.515))
    _arrow(axis, (0.67, 0.515), (0.72, 0.645))
    _arrow(axis, (0.81, 0.58), (0.81, 0.44))
    axis.text(0.5, 0.96, "Figure 2. Structured SHAP-to-LLM evidence flow", ha="center", fontsize=15, weight="bold")
    axis.text(0.5, 0.015, "Incomplete required evidence blocks real execution; the LLM cannot invent missing scientific evidence.", ha="center", fontsize=9)
    outputs = _save(fig, output_dir, FIGURE_STEMS[2], dpi=dpi, run_id=run_id, config_hash=config_hash)
    return {**outputs, **{f"source_{key}": value for key, value in sources.items()}}


def figure_3(
    config: Mapping[str, Any],
    output_dir: Path,
    source_dir: Path,
    *,
    run_id: str,
    config_hash: str,
    dpi: int,
) -> Dict[str, Path]:
    settings = config["manuscript_final"]
    roles = settings["llm_agent_evaluation"]["deterministic_agent_roles"]
    rows = []
    for role, definition in roles.items():
        rows.append(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "agent_role": role,
                "evidence_scope": definition["evidence_scope"],
                "authority": definition["authority"],
            }
        )
    source_path = source_dir / "figure_3_agent_roles.csv"
    pd.DataFrame(rows).to_csv(source_path, index=False)
    supervisor = next((row for row in rows if "Supervisor" in row["agent_role"]), rows[-1])
    specialists = [row for row in rows if row is not supervisor]
    fig, axis = plt.subplots(figsize=(14, 8.2))
    fig.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.07)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    columns = 3
    for index, row in enumerate(specialists):
        column = index % columns
        line = index // columns
        x = 0.06 + column * 0.31
        y = 0.67 - line * 0.29
        label = f"{row['agent_role']}\n\n{row['evidence_scope']}\n\nAuthority: audit/warning only"
        _box(axis, x, y, 0.25, 0.20, label, facecolor="#E9F2F9", fontsize=7.7)
        _arrow(axis, (x + 0.125, y), (0.5, 0.24))
    _box(
        axis,
        0.34,
        0.06,
        0.32,
        0.17,
        f"{supervisor['agent_role']}\n\n{supervisor['evidence_scope']}\n\nNo HR-decision authority",
        facecolor="#EDE3F5",
        fontsize=8.5,
    )
    axis.text(0.5, 0.96, "Figure 3. Deterministic multi-agent governance audit structure", ha="center", fontsize=15, weight="bold")
    outputs = _save(fig, output_dir, FIGURE_STEMS[3], dpi=dpi, run_id=run_id, config_hash=config_hash)
    return {**outputs, "source_roles": source_path}


def default_readiness_matrix(run_dir: Path, *, run_id: str, config_hash: str) -> pd.DataFrame:
    definitions = [
        ("Performance adequacy", "policy/policy_summary.csv", "research_only", "Model utility does not establish deployment fitness."),
        ("Leakage robustness", "policy/leakage_sensitivity_index.csv", "pass_with_limits", "Full-feature evidence is diagnostic only."),
        ("Calibration reliability", "calibration/calibration_method_comparison.csv", "pass_with_limits", "Probabilities remain approximate."),
        ("Explanation stability", "shap/shap_stability_summary.csv", "pass_with_limits", "SHAP is attribution, not causality."),
        ("Fairness and subgroup support", "fairness/manuscript_fairness_proxy_table.csv", "warning", "Audits do not prove fairness."),
        ("Proxy risk", "fairness/proxy_policy_comparison.csv", "warning", "Reconstructability is risk evidence only."),
        ("Counterfactual actionability", "counterfactual/actionability_summary.csv", "warning", "Scenarios are not employee prescriptions."),
        ("LLM evidence readiness", "llm/preflight_report.json", "research_only", "Compliance is conditional on complete evidence."),
        ("Chatbot guardrails", "chatbot/category_summary.csv", "pass_with_limits", "Deterministic suite is not comprehensive safety proof."),
        ("External evidence", "external/external_dataset_roles.csv", "research_only", "Replication and related tasks have distinct claim boundaries."),
        ("Data provenance", "provenance/dataset_cards.json", "manual_review_required", "Licence/source authenticity requires manual review."),
    ]
    rows = []
    for component, relative_path, interpretation, limitation in definitions:
        path = run_dir / relative_path
        rows.append(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "component": component,
                "evidence_file": relative_path,
                "evidence_state": "present" if path.is_file() and path.stat().st_size > 0 else "missing",
                "interpretation_status": interpretation if path.is_file() and path.stat().st_size > 0 else "evidence_missing",
                "limitation": limitation,
                "autonomous_hr_ready": False,
            }
        )
    return pd.DataFrame(rows)


def figure_4(
    readiness: pd.DataFrame,
    output_dir: Path,
    source_dir: Path,
    *,
    run_id: str,
    config_hash: str,
    dpi: int,
) -> Dict[str, Path]:
    required = {"component", "evidence_state", "interpretation_status", "limitation"}
    missing = sorted(required.difference(readiness.columns))
    if missing:
        raise ManuscriptFigureError(f"Readiness matrix lacks columns: {missing}")
    source = readiness.copy()
    source["run_id"] = run_id
    source["config_hash"] = config_hash
    source_path = source_dir / "figure_4_readiness_status_matrix.csv"
    source.to_csv(source_path, index=False)
    colors = {
        "pass_with_limits": "#2A9D8F",
        "research_only": "#4C78A8",
        "warning": "#F2A541",
        "manual_review_required": "#9C6ADE",
        "evidence_missing": "#C44E52",
    }
    fig_height = max(6.5, 0.55 * len(source) + 1.6)
    fig, axis = plt.subplots(figsize=(13.5, fig_height))
    fig.subplots_adjust(left=0.24, right=0.98, top=0.92, bottom=0.10)
    positions = list(range(len(source)))
    axis.barh(
        positions,
        [1] * len(source),
        color=[colors.get(value, "#999999") for value in source["interpretation_status"]],
        height=0.62,
    )
    axis.set_yticks(positions, source["component"])
    axis.invert_yaxis()
    axis.set_xlim(0, 1.7)
    axis.set_xticks([])
    axis.set_title("Figure 4. G-XAIR component readiness status matrix (no composite score)", fontsize=14, weight="bold")
    for index, row in source.reset_index(drop=True).iterrows():
        axis.text(0.03, index, str(row["interpretation_status"]), va="center", color="white", weight="bold", fontsize=8)
        axis.text(1.03, index, str(row["limitation"]), va="center", fontsize=7.5)
    axis.text(
        0,
        len(source) + 0.2,
        "All components remain research-only; none authorizes autonomous HR decisions. Missing/manual-review evidence prevents packaging readiness.",
        fontsize=9,
        color="#8B1E3F",
    )
    outputs = _save(fig, output_dir, FIGURE_STEMS[4], dpi=dpi, run_id=run_id, config_hash=config_hash)
    return {**outputs, "source_matrix": source_path}


def generate_architecture_figures(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    run_dir: str | Path,
    run_id: str,
    config_hash: str | None = None,
    readiness: pd.DataFrame | None = None,
) -> Dict[str, Path]:
    config = load_manuscript_config(config_path)
    legacy_llm = config["manuscript_final"].get("llm_agent_evaluation")
    if not isinstance(legacy_llm, Mapping):
        raise ManuscriptFigureError(
            "Legacy governance Figures 1-4 require an explicit legacy configuration and are "
            "excluded from the canonical leakage-aware core figure scope."
        )
    config_hash = config_hash or canonical_config_hash(config)
    output = ensure_dir(Path(output_dir))
    sources = ensure_dir(output / "source_data")
    dpi = int(config["manuscript_final"]["figures"].get("publication_dpi", 300))
    run_root = Path(run_dir)
    readiness = readiness if readiness is not None else default_readiness_matrix(
        run_root,
        run_id=run_id,
        config_hash=config_hash,
    )
    outputs: Dict[str, Path] = {}
    for number, generated in (
        (1, figure_1(output, sources, run_id=run_id, config_hash=config_hash, dpi=dpi)),
        (2, figure_2(output, sources, run_id=run_id, config_hash=config_hash, dpi=dpi)),
        (3, figure_3(config, output, sources, run_id=run_id, config_hash=config_hash, dpi=dpi)),
        (4, figure_4(readiness, output, sources, run_id=run_id, config_hash=config_hash, dpi=dpi)),
    ):
        outputs.update({f"figure_{number}_{key}": value for key, value in generated.items()})
    return outputs


def validate_all_seven_figures(figure_dir: str | Path) -> Dict[str, Any]:
    root = Path(figure_dir)
    missing: list[str] = []
    empty: list[str] = []
    for number, stem in FIGURE_STEMS.items():
        for suffix in (".png", ".svg"):
            path = root / f"{stem}{suffix}"
            if not path.is_file():
                missing.append(str(path))
            elif path.stat().st_size == 0:
                empty.append(str(path))
    if missing or empty:
        raise ManuscriptFigureError(f"Figure package invalid; missing={missing}, empty={empty}")
    return {"status": "passed", "figure_count": 7, "formats": ["png", "svg"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate legacy v1 governance figures 1-4.")
    parser.add_argument("--config", default="configs/manuscript_final.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            {
                key: str(value)
                for key, value in generate_architecture_figures(
                    arguments.config,
                    output_dir=arguments.output_dir,
                    run_dir=arguments.run_dir,
                    run_id=arguments.run_id,
                    config_hash=arguments.config_hash,
                ).items()
            },
            indent=2,
            sort_keys=True,
        )
    )
