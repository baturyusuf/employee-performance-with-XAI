"""Deterministic compact export of canonical manuscript-support assets.

This module never fits a model or recomputes scientific evidence.  It verifies
the immutable canonical package, copies or presentation-renders compact assets,
and emits complete source maps for manuscript insertion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from src.core.atomic_publish import atomic_replace_directory
from src.governance.manuscript_contract import source_tree_hash


class ManuscriptAssetExportError(RuntimeError):
    """Raised when canonical-source or compact-export validation fails."""


RUN_ID = "canonical_v2_20260714T221501Z_483f96f"
GENERATION_COMMIT = "483f96fdbaab16cb0f32d03d9dbe676a759af44a"
CONFIG_HASH = "51415c2ce68c89d9ce2b042b0a7a811fe3e98180b72460006f9d0465f6bf49b7"
SOURCE_TREE_HASH = "f1e358e99914563305428cece1b1595bc76a58643184407ec5b222162d650332"
CORE_SCIENTIFIC_INPUT_HASH = (
    "06c507bee525ea1daca43b61249764007d4d8baaa05c9333f23446ea723ce160"
)
SUPPLEMENTARY_SCIENTIFIC_INPUT_HASH = (
    "caffb945d15f990e3a789e9707f7a8a9115be31fecbbd705822994a10cfaf151"
)
METRIC_SCHEMA_HASH = "98ae57b622a56983192975c5bff94374ed51322f13c76af633c3c96ceb198cdd"
CANONICAL_RELATIVE_ROOT = Path("reports/manuscript_final") / RUN_ID
ASSET_RELATIVE_ROOT = Path("manuscript/mdpi_information/assets")
CANONICAL_RECEIPT = Path(
    "reports/research_log/finalization_v2/15_canonical_evidence_receipt.json"
)
CANONICAL_POINTER = Path("reports/manuscript_final/latest/pointer.json")
DISPLAY_DECIMALS = 4
ROUNDING_RULE = "decimal ROUND_HALF_EVEN to 4 places; integer denominators unrounded"


MAIN_FIGURE_EXPORTS: tuple[dict[str, Any], ...] = (
    {
        "manuscript_number": 1,
        "stem": "figure_01_audit_protocol",
        "canonical_number": 1,
        "status": "post_canonical_presentation_rendering",
    },
    {
        "manuscript_number": 2,
        "stem": "figure_02_model_benchmark",
        "canonical_number": 3,
        "status": "byte_for_byte_copy",
    },
    {
        "manuscript_number": 3,
        "stem": "figure_03_feature_policy_sensitivity",
        "canonical_number": 2,
        "status": "byte_for_byte_copy",
    },
    {
        "manuscript_number": 4,
        "stem": "figure_04_calibration",
        "canonical_number": 4,
        "status": "byte_for_byte_copy",
    },
    {
        "manuscript_number": 5,
        "stem": "figure_05_global_grouped_shap",
        "canonical_number": 5,
        "status": "byte_for_byte_copy",
    },
    {
        "manuscript_number": 6,
        "stem": "figure_06_shap_stability",
        "canonical_number": 6,
        "status": "byte_for_byte_copy",
    },
    {
        "manuscript_number": 7,
        "stem": "figure_07_hrdataset_replication",
        "canonical_number": 7,
        "status": "byte_for_byte_copy",
    },
)

TABLE_SPECS: tuple[tuple[int, str, str], ...] = (
    (1, "table_01_datasets", "Core datasets, targets, support, and analytical roles"),
    (2, "table_02_feature_governance", "Prespecified feature-governance policies"),
    (3, "table_03_nested_benchmark", "Ten-by-five nested OOF four-model benchmark"),
    (4, "table_04_feature_policy_sensitivity", "Matched-fold feature-policy sensitivity"),
    (5, "table_05_calibration", "Raw and cross-fitted sigmoid probability metrics"),
    (6, "table_06_shap_attribution_stability", "Grouped SHAP attribution and stability"),
    (7, "table_07_subgroup_proxy_diagnostics", "Subgroup and proxy diagnostics"),
    (8, "table_08_hrdataset_replication", "HRDataset_v14 mapped-target replication"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fields = ordered
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, float_precision="round_trip")
    frame["_canonical_csv_row"] = [str(index + 2) for index in range(len(frame))]
    return frame


def _safe_relative(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ManuscriptAssetExportError(f"Path escapes repository: {path}") from exc
    if any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts):
        raise ManuscriptAssetExportError(f"Unsafe repository-relative path: {relative}")
    return relative


def _decimal(value: Any) -> Decimal | None:
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise ManuscriptAssetExportError(f"Expected a finite decimal, got {value!r}.") from exc
    if not parsed.is_finite():
        raise ManuscriptAssetExportError(f"Expected a finite decimal, got {value!r}.")
    return parsed


def _format_decimal(value: Any, decimals: int = DISPLAY_DECIMALS) -> str:
    parsed = _decimal(value)
    if parsed is None:
        return "N/A"
    quantum = Decimal(1).scaleb(-decimals)
    return format(parsed.quantize(quantum, rounding=ROUND_HALF_EVEN), f".{decimals}f")


def _invert_triplet(value: str, low: str, high: str) -> tuple[str, str, str]:
    parsed = _decimal(value)
    parsed_low = _decimal(low)
    parsed_high = _decimal(high)
    if parsed is None or parsed_low is None or parsed_high is None:
        return "N/A", "N/A", "N/A"
    return str(-parsed), str(-parsed_high), str(-parsed_low)


def _markdown(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], title: str, note: str) -> str:
    lines = [f"# {title}", "", note, "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = [str(row.get(column, "")).replace("\n", " ").replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _assert_identity(payload: Mapping[str, Any], *, scope: str | None = None) -> None:
    expected_scientific = (
        CORE_SCIENTIFIC_INPUT_HASH if scope in {None, "core"} else SUPPLEMENTARY_SCIENTIFIC_INPUT_HASH
    )
    expected = {
        "run_id": RUN_ID,
        "config_hash": CONFIG_HASH,
        "source_tree_hash": SOURCE_TREE_HASH,
    }
    if scope is not None:
        expected["scientific_input_hash"] = expected_scientific
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ManuscriptAssetExportError(
                f"Canonical identity mismatch for {field}: {payload.get(field)!r} != {value!r}."
            )


def validate_canonical_source(project_root: Path, canonical_root: Path) -> dict[str, Any]:
    """Verify both closed-world manifests and every declared canonical byte."""

    root = project_root.resolve()
    source = canonical_root.resolve()
    expected_source = (root / CANONICAL_RELATIVE_ROOT).resolve()
    if source != expected_source or not source.is_dir() or source.is_symlink():
        raise ManuscriptAssetExportError("Canonical source must be the exact regular configured run root.")

    receipt_path = root / CANONICAL_RECEIPT
    pointer_path = root / CANONICAL_POINTER
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    generation = receipt.get("generation", {})
    _assert_identity(generation)
    if generation.get("git_commit") != GENERATION_COMMIT:
        raise ManuscriptAssetExportError("Canonical receipt generation commit mismatch.")
    _assert_identity(pointer)
    if pointer.get("git_commit") != GENERATION_COMMIT or pointer.get("relative_target") != f"../{RUN_ID}":
        raise ManuscriptAssetExportError("Canonical pointer does not target the expected immutable run.")
    if receipt.get("status") != "passed_and_promoted" or pointer.get("status") != "complete":
        raise ManuscriptAssetExportError("Canonical receipt or pointer is not complete.")

    scope_results: dict[str, Any] = {}
    for scope, expected_count, expected_scientific in (
        ("core", 351, CORE_SCIENTIFIC_INPUT_HASH),
        ("supplementary", 188, SUPPLEMENTARY_SCIENTIFIC_INPUT_HASH),
    ):
        scope_root = source / scope
        final_json = scope_root / "final_evidence_manifest.json"
        final_csv = scope_root / "final_evidence_manifest.csv"
        run_manifest_path = scope_root / "run_manifest.json"
        manifest = json.loads(final_json.read_text(encoding="utf-8"))
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        _assert_identity(manifest, scope=scope)
        _assert_identity(run_manifest, scope=scope)
        if manifest.get("git_commit") != GENERATION_COMMIT or run_manifest.get("git_commit") != GENERATION_COMMIT:
            raise ManuscriptAssetExportError(f"{scope} generation commit mismatch.")
        if manifest.get("n_files") != expected_count or len(manifest.get("files", [])) != expected_count:
            raise ManuscriptAssetExportError(f"{scope} manifest record count mismatch.")
        pointer_scope = pointer["scopes"][scope]
        receipt_scope = receipt["artifact_inventory"][scope]
        for path, expected_hash in (
            (final_json, pointer_scope["final_evidence_manifest"]["sha256"]),
            (run_manifest_path, pointer_scope["run_manifest"]["sha256"]),
        ):
            if _sha256(path) != expected_hash:
                raise ManuscriptAssetExportError(f"Canonical manifest hash mismatch: {path}")
        if _sha256(final_json) != receipt_scope["final_evidence_manifest_sha256"]:
            raise ManuscriptAssetExportError(f"Canonical receipt {scope} final-manifest mismatch.")
        if _sha256(run_manifest_path) != receipt_scope["run_manifest_sha256"]:
            raise ManuscriptAssetExportError(f"Canonical receipt {scope} run-manifest mismatch.")

        seen: set[str] = set()
        total_bytes = 0
        for record in manifest["files"]:
            _assert_identity(record, scope=scope)
            if record.get("git_commit") != GENERATION_COMMIT:
                raise ManuscriptAssetExportError(f"{scope} artifact commit mismatch.")
            relative = str(record.get("path", ""))
            pure = PurePosixPath(relative)
            if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
                raise ManuscriptAssetExportError(f"Unsafe canonical artifact path: {relative}")
            if relative in seen:
                raise ManuscriptAssetExportError(f"Duplicate canonical artifact path: {relative}")
            seen.add(relative)
            artifact = scope_root / Path(*pure.parts)
            if not artifact.is_file() or artifact.is_symlink():
                raise ManuscriptAssetExportError(f"Missing or link-like canonical artifact: {artifact}")
            if artifact.stat().st_size != int(record["size_bytes"]) or _sha256(artifact) != record["sha256"]:
                raise ManuscriptAssetExportError(f"Canonical artifact byte mismatch: {artifact}")
            total_bytes += artifact.stat().st_size
        actual = {
            p.relative_to(scope_root).as_posix()
            for p in scope_root.rglob("*")
            if p.is_file()
        }
        expected_physical = seen | {
            "final_evidence_manifest.csv",
            "final_evidence_manifest.json",
            "run_manifest.json",
        }
        if actual != expected_physical:
            raise ManuscriptAssetExportError(
                f"{scope} closed-world inventory mismatch: missing={sorted(expected_physical-actual)}, "
                f"extra={sorted(actual-expected_physical)}"
            )
        if len(actual) != int(receipt_scope["actual_files"]):
            raise ManuscriptAssetExportError(f"{scope} physical file count differs from receipt.")
        if sum((scope_root / path).stat().st_size for path in actual) != int(receipt_scope["actual_bytes"]):
            raise ManuscriptAssetExportError(f"{scope} physical byte count differs from receipt.")
        if _sha256(final_csv) == "":  # pragma: no cover - defensive nonempty assertion
            raise ManuscriptAssetExportError("Unreachable empty manifest digest.")
        scope_results[scope] = {
            "manifest_records": len(seen),
            "physical_files": len(actual),
            "physical_bytes": sum((scope_root / path).stat().st_size for path in actual),
            "scientific_input_hash": expected_scientific,
        }

    if source_tree_hash(root) != SOURCE_TREE_HASH:
        raise ManuscriptAssetExportError("Current scientific source-tree bytes differ from the canonical hash.")
    changed = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            f"{GENERATION_COMMIT}..HEAD",
            "--",
            "src",
            "configs",
            "requirements.txt",
            "requirements-core.txt",
            "requirements-supplementary.txt",
            "requirements-legacy-optional.txt",
            "requirements-dev.txt",
            "constraints/py314-lock.txt",
            "environment.yml",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if changed:
        raise ManuscriptAssetExportError(
            "Scientific source/config changed after canonical generation: " + changed.replace("\n", ", ")
        )
    return {"status": "passed", "run_id": RUN_ID, "scopes": scope_results}


@dataclass
class ExportContext:
    project_root: Path
    canonical_root: Path
    output_root: Path
    export_timestamp: str
    lineage: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    ledger: list[dict[str, str]] = field(default_factory=list)
    table_outputs: dict[int, dict[str, Any]] = field(default_factory=dict)
    figure_rows: list[dict[str, Any]] = field(default_factory=list)

    def relative(self, path: Path) -> str:
        try:
            within_output = path.resolve().relative_to(self.output_root.resolve())
        except ValueError:
            pass
        else:
            return (ASSET_RELATIVE_ROOT / within_output).as_posix()
        return _safe_relative(path, self.project_root)

    def bind(self, output: Path, sources: Iterable[Path]) -> None:
        self.lineage[self.relative(output)] = [
            {"path": self.relative(source), "sha256": _sha256(source)} for source in sources
        ]

    def source_ref(self, table_path: Path, row: Mapping[str, Any], column: str) -> dict[str, str]:
        return {
            "canonical_source_path": self.relative(table_path),
            "source_row_key": f"csv_row={row.get('_canonical_csv_row','')};"
            f"record={row.get('source_record_type','')};source_row={row.get('source_row_number','')}",
            "source_column": column,
            "source_sha256": _sha256(table_path),
        }

    def add_measure(
        self,
        output_row: dict[str, Any],
        *,
        prefix: str,
        value: Any,
        low: Any = "",
        high: Any = "",
        source_table: Path,
        source_row: Mapping[str, Any],
        metric: str,
        manuscript_table: str,
        claim_id: str,
        dataset: str,
        model: str,
        policy: str,
        denominator: Any,
        uncertainty_method: str,
        direction: str,
    ) -> None:
        full = str(value).strip() or "N/A"
        lower = str(low).strip() or "N/A"
        upper = str(high).strip() or "N/A"
        output_row[f"{prefix}_full_precision"] = full
        output_row[f"{prefix}_display"] = _format_decimal(full)
        output_row[f"{prefix}_ci_lower_full_precision"] = lower
        output_row[f"{prefix}_ci_upper_full_precision"] = upper
        output_row[f"{prefix}_interval_display"] = (
            "N/A" if lower == "N/A" or upper == "N/A" else f"[{_format_decimal(lower)}, {_format_decimal(upper)}]"
        )
        ref = self.source_ref(source_table, source_row, metric)
        self.ledger.append(
            {
                "claim_id": claim_id,
                "manuscript_section": "Results",
                "object_type": "numerical_result",
                "manuscript_table_or_figure": manuscript_table,
                "metric_or_statement": metric,
                "full_precision_value": full,
                "displayed_value": output_row[f"{prefix}_display"],
                "interval_lower": lower,
                "interval_upper": upper,
                "denominator": str(denominator).strip() or "N/A",
                "dataset": dataset,
                "model": model,
                "policy": policy,
                "uncertainty_method": uncertainty_method,
                **ref,
                "run_id": RUN_ID,
                "config_hash": CONFIG_HASH,
                "scientific_input_hash": (
                    SUPPLEMENTARY_SCIENTIFIC_INPUT_HASH
                    if "supplementary" in str(source_table).replace("\\", "/")
                    else CORE_SCIENTIFIC_INPUT_HASH
                ),
                "source_tree_hash": SOURCE_TREE_HASH,
                "verification_status": f"verified;direction={direction};rounding={ROUNDING_RULE}",
            }
        )


def _box(ax: Any, x: float, y: float, w: float, h: float, text: str, color: str, *, fontsize: float = 8.5) -> None:
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=color,
        edgecolor="#26384a",
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow(ax: Any, start: tuple[float, float], end: tuple[float, float], *, color: str = "#26384a", label: str = "") -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": color})
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.025, label, ha="center", va="bottom", fontsize=7.3, color=color)


def _render_figure_1(png_path: Path, svg_path: Path) -> None:
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[0.32, 0.68])
    ax_a = fig.add_subplot(grid[0])
    ax_b = fig.add_subplot(grid[1])
    for ax in (ax_a, ax_b):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax_a.text(0.0, 0.97, "A. Conceptual rationale", fontsize=13, fontweight="bold", va="top")
    concepts = [
        (0.11, "Predictive\nperformance alone", "#dbe9f6"),
        (0.37, "Unresolved risks\nleakage · calibration · explanation stability\nsubgroup/proxy · external reproducibility", "#f9e2ae"),
        (0.64, "Integrated leakage-aware\nXAI audit", "#cce8d8"),
        (0.89, "Bounded evidence\n+ explicit prohibited claims", "#ded5ef"),
    ]
    for x, text, color in concepts:
        _box(ax_a, x, 0.48, 0.205, 0.43, text, color, fontsize=9.3)
    for left, right in zip(concepts, concepts[1:]):
        _arrow(ax_a, (left[0] + 0.105, 0.48), (right[0] - 0.105, 0.48))

    ax_b.text(0.0, 0.985, "B. Exact technical pipeline", fontsize=13, fontweight="bold", va="top")
    ax_b.text(0.012, 0.80, "TRAINING-ONLY FLOW", fontsize=9, fontweight="bold", color="#175b87")
    ax_b.text(0.012, 0.48, "UNTOUCHED OUTER-TEST FLOW", fontsize=9, fontweight="bold", color="#a34b16")
    ax_b.text(0.012, 0.17, "AUDIT / REPORTING FLOW", fontsize=9, fontweight="bold", color="#28643c")
    ax_b.axhspan(0.62, 0.93, color="#eaf3fa", alpha=0.65, zorder=-5)
    ax_b.axhspan(0.34, 0.59, color="#fff0df", alpha=0.65, zorder=-5)
    ax_b.axhspan(0.02, 0.30, color="#eaf6ed", alpha=0.65, zorder=-5)

    nodes = {
        "inputs": (0.08, 0.72, "Verified dataset bytes\nschema · target"),
        "policy": (0.24, 0.72, "Prespecified\nfeature policies"),
        "folds": (0.40, 0.72, "Shared 10 outer folds\n5-fold inner selection"),
        "model": (0.57, 0.72, "Exact persisted\nouter-fold XGBoost"),
        "calfit": (0.74, 0.72, "Cross-fitted sigmoid fit\nouter-training only"),
        "outer": (0.40, 0.46, "Untouched\nouter-test cases"),
        "prediction": (0.57, 0.46, "Outer-test prediction\nraw probabilities"),
        "sigmoid": (0.74, 0.46, "Predeclared sigmoid\nprobabilities"),
        "shap": (0.55, 0.19, "Grouped SHAP from the\nsame persisted fold model"),
        "stability": (0.70, 0.19, "SHAP stability"),
        "diagnostics": (0.86, 0.19, "Subgroup/proxy diagnostics\n+ HR mapped-target replication"),
        "outputs": (0.91, 0.46, "Source tables · figures\nmanifests · claim boundaries"),
    }
    for key, (x, y, text) in nodes.items():
        if y > 0.6:
            color = "#cfe5f5"
        elif y > 0.3:
            color = "#f8d9b9"
        else:
            color = "#cfe9d7"
        width = 0.14
        if key == "stability":
            width = 0.12
        if key == "diagnostics":
            width = 0.18
        _box(ax_b, x, y, width, 0.13, text, color, fontsize=7.8)

    _arrow(ax_b, (0.15, 0.72), (0.17, 0.72))
    _arrow(ax_b, (0.31, 0.72), (0.33, 0.72))
    _arrow(ax_b, (0.47, 0.72), (0.50, 0.72), label="outer training")
    _arrow(ax_b, (0.64, 0.72), (0.67, 0.72), label="inner-OOF probabilities")
    _arrow(ax_b, (0.40, 0.655), (0.40, 0.525), color="#a34b16", label="held out")
    _arrow(ax_b, (0.47, 0.46), (0.50, 0.46), color="#a34b16")
    _arrow(ax_b, (0.57, 0.655), (0.57, 0.525), label="same model")
    _arrow(ax_b, (0.64, 0.46), (0.67, 0.46), color="#a34b16")
    _arrow(ax_b, (0.74, 0.655), (0.74, 0.525), label="apply only")
    _arrow(ax_b, (0.57, 0.395), (0.55, 0.255), color="#28643c", label="exact prediction/model identity")
    _arrow(ax_b, (0.62, 0.19), (0.63, 0.19), color="#28643c")
    _arrow(ax_b, (0.76, 0.19), (0.77, 0.19), color="#28643c")
    _arrow(ax_b, (0.81, 0.46), (0.84, 0.46), color="#28643c")
    _arrow(ax_b, (0.86, 0.255), (0.89, 0.395), color="#28643c")
    ax_b.text(
        0.5,
        0.005,
        "No outer-test path enters preprocessing fitting, model selection, calibration fitting, or threshold selection.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#8d2f23",
        fontweight="bold",
    )
    fig.suptitle("Leakage-aware XAI audit: rationale and isolated evidence flow", fontsize=16, fontweight="bold")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    creator = "matplotlib deterministic manuscript presentation rendering"
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": creator},
    )
    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": creator},
    )
    plt.close(fig)


def _validate_svg(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    lowered = text.casefold()
    if "<!entity" in lowered or "<script" in lowered or re.search(r"(?:href|src)=[\"'](?:https?:|file:|//)", lowered):
        raise ManuscriptAssetExportError(f"Unsafe SVG content: {path}")
    if re.search(r"[A-Za-z]:[\\/]Users[\\/]", text):
        raise ManuscriptAssetExportError(f"Machine-absolute path in SVG: {path}")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ManuscriptAssetExportError(f"Invalid SVG XML: {path}") from exc
    if not root.tag.endswith("svg"):
        raise ManuscriptAssetExportError(f"SVG root element is invalid: {path}")
    return "passed_no_script_entity_external_resource_or_machine_path"


def _figure_1_source_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    conceptual = [
        (1, "predictive_performance", "Predictive performance alone", "Does not resolve audit validity."),
        (2, "unresolved_risks", "Unresolved audit risks", "Leakage, calibration, stability, subgroup/proxy, and replication risks."),
        (3, "integrated_audit", "Integrated leakage-aware XAI audit", "Prespecified and identity-bound audit stages."),
        (4, "bounded_evidence", "Bounded evidence and prohibited claims", "No causal, fairness-proof, deployment, or autonomous-HR claim."),
    ]
    technical = [
        (1, "inputs", "Verified dataset bytes/schema/target", "run_inputs/input_contract.json"),
        (2, "feature_policies", "Prespecified feature policies", "policy_ablation/policy_feature_contract.csv"),
        (3, "nested_selection", "Shared 10 outer x 5 inner folds", "shared_folds/fold_contract.json"),
        (4, "outer_model", "Exact persisted outer-fold XGBoost", "model_benchmarks/outer_model_receipts.csv"),
        (5, "outer_test", "Untouched outer-test prediction", "model_benchmarks/oof_predictions.csv"),
        (6, "sigmoid", "Cross-fitted sigmoid calibration", "sigmoid_calibration/calibration_protocol.json"),
        (7, "same_model_shap", "Grouped SHAP from the same fold model", "oof_shap/shap_metadata.json"),
        (8, "stability", "SHAP stability", "oof_shap/shap_stability_summary.csv"),
        (9, "diagnostics", "Subgroup/proxy and mapped-target replication", "subgroup_proxy;external_replication"),
        (10, "publication", "Tables, figures, manifests, claim boundaries", "core_tables;core_figures"),
    ]
    for order, node, label, detail in conceptual:
        rows.append({"panel": "A_conceptual_rationale", "order": order, "node_id": node, "label": label, "detail": detail})
    for order, node, label, detail in technical:
        rows.append({"panel": "B_exact_technical_pipeline", "order": order, "node_id": node, "label": label, "detail": detail})
    for row in rows:
        row.update(
            {
                "run_id": RUN_ID,
                "config_hash": CONFIG_HASH,
                "scientific_input_hash": CORE_SCIENTIFIC_INPUT_HASH,
                "source_tree_hash": SOURCE_TREE_HASH,
            }
        )
    return rows


MAIN_CAPTION_BOUNDARIES = {
    1: "Panel A states the conceptual rationale; Panel B distinguishes training-only and untouched outer-test paths. The diagram is a post-canonical presentation rendering of the frozen protocol and contains no newly computed result.",
    2: "Intervals and baseline-minus-XGBoost contrasts use the predeclared paired sample-level OOF bootstrap. The superiority gate is interpreted only under its macro-F1 rule; it is not a universal leaderboard.",
    3: "Audit-only policies are feature-access sensitivity systems and must not be presented as alternative primary models or causal interventions.",
    4: "Calibration concerns probability reliability. It does not automatically improve argmax classification performance or authorize decision thresholds.",
    5: "Grouped OOF SHAP values are model attributions in raw-margin space from the same persisted outer-fold model that produced each prediction; they are not causal effects.",
    6: "The 45 outer-fold pairs are dependent descriptive comparisons and do not support a population confidence interval.",
    7: "HRDataset_v14 is an independently trained mapped-target replication, not locked-model transport, target equivalence, or universal external validation.",
}

MAIN_ALT_TEXT = {
    1: "Two-panel audit diagram. Panel A moves from predictive performance through unresolved audit risks to an integrated leakage-aware audit and bounded claims. Panel B separates training-only model and calibrator fitting from untouched outer-test prediction, with grouped SHAP explicitly tied to the same persisted fold model.",
    2: "Interval plot comparing XGBoost with logistic regression, random forest, and LightGBM on shared out-of-fold benchmark metrics, with paired uncertainty and gate context.",
    3: "Feature-policy sensitivity plot comparing the prespecified primary system with audit and sensitivity policies on matched folds; audit-only systems are visually distinguished.",
    4: "Calibration figure showing raw and cross-fitted sigmoid reliability and probability-quality metrics, separately from classification outcomes.",
    5: "Horizontal ranking of global mean absolute grouped out-of-fold SHAP values for the primary XGBoost policy, with governance annotations.",
    6: "Descriptive stability panels for grouped SHAP rankings across dependent outer-fold pairs, including top-k overlap and Spearman summaries.",
    7: "Summary of HRDataset_v14 mapped-target support, independently trained policy performance, calibration, and policy sensitivity with bounded replication language.",
}


def _export_main_figures(ctx: ExportContext) -> None:
    figure_root = ctx.canonical_root / "core/core_figures"
    manifest_path = figure_root / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _assert_identity(manifest, scope="core")
    if manifest.get("n_figures") != 7 or manifest.get("status") != "complete":
        raise ManuscriptAssetExportError("Canonical figure manifest is incomplete.")
    by_number = {int(row["number"]): row for row in manifest["figures"]}
    mapping_rows: list[dict[str, Any]] = []
    for spec in MAIN_FIGURE_EXPORTS:
        manuscript_number = int(spec["manuscript_number"])
        canonical_number = int(spec["canonical_number"])
        source_record = by_number[canonical_number]
        canonical_png = figure_root / source_record["png_path"]
        canonical_svg = figure_root / source_record["svg_path"]
        canonical_source_data = figure_root / source_record["source_data_path"]
        canonical_caption = figure_root / source_record["caption_path"]
        stem = str(spec["stem"])
        output_png = ctx.output_root / "figures/main" / f"{stem}.png"
        output_svg = ctx.output_root / "figures/main" / f"{stem}.svg"
        output_source = ctx.output_root / "source_data/figures" / f"{stem}_source.csv"
        output_caption = ctx.output_root / "figures/captions" / f"figure_{manuscript_number:02d}_caption.md"
        output_alt = ctx.output_root / "figures/alt_text" / f"figure_{manuscript_number:02d}_alt_text.txt"

        if manuscript_number == 1:
            _render_figure_1(output_png, output_svg)
            _write_csv(output_source, _figure_1_source_rows())
        else:
            output_png.parent.mkdir(parents=True, exist_ok=True)
            output_source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(canonical_png, output_png)
            shutil.copyfile(canonical_svg, output_svg)
            shutil.copyfile(canonical_source_data, output_source)
        canonical_caption_text = canonical_caption.read_text(encoding="utf-8").strip()
        _write_text(
            output_caption,
            f"# Figure {manuscript_number} caption\n\n{canonical_caption_text}\n\n"
            f"**Manuscript boundary.** {MAIN_CAPTION_BOUNDARIES[manuscript_number]}",
        )
        _write_text(output_alt, MAIN_ALT_TEXT[manuscript_number])
        with Image.open(output_png) as image:
            width, height = image.size
            dpi_value = image.info.get("dpi", (0.0, 0.0))
            dpi = min(float(dpi_value[0]), float(dpi_value[1])) if dpi_value else 0.0
        svg_result = _validate_svg(output_svg)
        if width < 1800 or height < 1000 or dpi < 299.0:
            raise ManuscriptAssetExportError(f"Figure {manuscript_number} fails journal raster QA.")
        status = str(spec["status"])
        if status == "byte_for_byte_copy" and (
            _sha256(output_png) != _sha256(canonical_png) or _sha256(output_svg) != _sha256(canonical_svg)
        ):
            raise ManuscriptAssetExportError(f"Figure {manuscript_number} copy hash parity failed.")
        sources = [canonical_png, canonical_svg, canonical_source_data, canonical_caption, manifest_path]
        for output in (output_png, output_svg, output_source, output_caption, output_alt):
            ctx.bind(output, sources)
        mapping_rows.append(
            {
                "manuscript_figure_number": manuscript_number,
                "manuscript_stem": stem,
                "canonical_figure_number": canonical_number,
                "canonical_figure_id": source_record["figure_id"],
                "canonical_png_path": ctx.relative(canonical_png),
                "canonical_png_sha256": _sha256(canonical_png),
                "exported_png_path": ctx.relative(output_png),
                "exported_png_sha256": _sha256(output_png),
                "canonical_svg_path": ctx.relative(canonical_svg),
                "canonical_svg_sha256": _sha256(canonical_svg),
                "exported_svg_path": ctx.relative(output_svg),
                "exported_svg_sha256": _sha256(output_svg),
                "canonical_source_data_path": ctx.relative(canonical_source_data),
                "canonical_source_data_sha256": _sha256(canonical_source_data),
                "exported_source_data_path": ctx.relative(output_source),
                "exported_source_data_sha256": _sha256(output_source),
                "caption_path": ctx.relative(output_caption),
                "alt_text_path": ctx.relative(output_alt),
                "width_px": width,
                "height_px": height,
                "dpi": f"{dpi:.4f}",
                "svg_validation": svg_result,
                "export_status": status,
                "run_id": RUN_ID,
                "config_hash": CONFIG_HASH,
                "scientific_input_hash": CORE_SCIENTIFIC_INPUT_HASH,
                "source_tree_hash": SOURCE_TREE_HASH,
            }
        )
    mapping_path = ctx.output_root / "manifests/figure_number_mapping.csv"
    _write_csv(mapping_path, mapping_rows)
    ctx.bind(mapping_path, [manifest_path])
    ctx.figure_rows.extend(mapping_rows)


def _render_supplementary_figures(ctx: ExportContext) -> None:
    table8 = ctx.canonical_root / "core/core_tables/table_08_support_aware_subgroup_diagnostics.csv"
    subgroup = _read_csv(table8)
    gaps = subgroup[(subgroup["source_record_type"] == "fairness_disparity_uncertainty")].copy()
    gaps["estimand"] = gaps["attribute"] + " | " + gaps["metric"] + gaps["class_label"].map(lambda x: f" | class {x}" if x else "")
    gaps["gap_numeric"] = pd.to_numeric(gaps["gap"], errors="coerce")
    matrix = gaps.pivot(index="estimand", columns="policy", values="gap_numeric")
    matrix = matrix.reindex(sorted(matrix.index))
    fig_height = max(12.0, 0.27 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(11, fig_height), constrained_layout=True)
    masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    image = ax.imshow(masked, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xticks(range(len(matrix.columns)), labels=[str(x).replace("_", "\n") for x in matrix.columns], rotation=25, ha="right")
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index, fontsize=6.5)
    ax.set_title("Figure S1. Full support-aware subgroup gap inventory")
    ax.set_xlabel("Feature policy")
    ax.set_ylabel("Supported estimand; blank cells are not estimable/support-eligible")
    fig.colorbar(image, ax=ax, label="Largest supported descriptive gap")
    for y, x in zip(*np.where(np.ma.getmaskarray(masked))):
        ax.text(x, y, "×", ha="center", va="center", color="black", fontsize=7)
    s1_png = ctx.output_root / "figures/supplementary/figure_s1_subgroup_heatmap.png"
    s1_svg = ctx.output_root / "figures/supplementary/figure_s1_subgroup_heatmap.svg"
    s1_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(s1_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(s1_svg, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    s1_source = ctx.output_root / "source_data/figures/figure_s1_subgroup_heatmap_source.csv"
    _write_csv(s1_source, gaps.drop(columns=["gap_numeric"]).to_dict(orient="records"))

    class_paths = [ctx.canonical_root / f"core/oof_shap/class_{label}_grouped_shap_importance.csv" for label in (2, 3, 4)]
    class_frames = [_read_csv(path) for path in class_paths]
    class_frame = pd.concat(class_frames, ignore_index=True)
    class_frame["rank_numeric"] = pd.to_numeric(class_frame["rank"], errors="raise")
    top_features = list(
        class_frame[class_frame["rank_numeric"] <= 10]
        .groupby("feature")["mean_abs_grouped_shap"]
        .apply(lambda values: pd.to_numeric(values, errors="raise").max())
        .sort_values(ascending=False)
        .head(12)
        .index
    )
    plot = class_frame[class_frame["feature"].isin(top_features)].copy()
    pivot = plot.pivot(index="feature", columns="class_label", values="mean_abs_grouped_shap").astype(float).fillna(0.0)
    pivot = pivot.loc[top_features[::-1]]
    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    y = np.arange(len(pivot.index)); width = 0.24
    colors = ["#0072B2", "#E69F00", "#009E73"]; hatches = ["///", "...", "\\\\"]
    for index, label in enumerate(("2", "3", "4")):
        ax.barh(y + (index - 1) * width, pivot.get(label, pd.Series(0.0, index=pivot.index)), width, label=f"Class {label}", color=colors[index], hatch=hatches[index], edgecolor="black", linewidth=0.4)
    ax.set_yticks(y, labels=pivot.index)
    ax.set_xlabel("Mean absolute grouped SHAP (raw-margin units)")
    ax.set_title("Figure S2. Class-specific grouped OOF SHAP summary")
    ax.legend()
    s2_png = ctx.output_root / "figures/supplementary/figure_s2_class_grouped_shap.png"
    s2_svg = ctx.output_root / "figures/supplementary/figure_s2_class_grouped_shap.svg"
    fig.savefig(s2_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(s2_svg, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    s2_source = ctx.output_root / "source_data/figures/figure_s2_class_grouped_shap_source.csv"
    _write_csv(s2_source, class_frame.drop(columns=["rank_numeric"]).to_dict(orient="records"))

    budget_path = ctx.canonical_root / "supplementary/heuristic_counterfactual/heuristic_search_budget_sensitivity.csv"
    budget = _read_csv(budget_path)
    budget_order = {"restricted": 0, "primary": 1, "expanded": 2}
    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    markers = ["o", "s", "^", "D"]
    styles = ["-", "--", "-.", ":"]
    for index, (scope, frame) in enumerate(budget.groupby("candidate_feature_scope", sort=True)):
        ordered = frame.assign(order=frame["budget_id"].map(budget_order)).sort_values("order")
        x = ordered["order"].to_numpy(dtype=float)
        yv = pd.to_numeric(ordered["heuristic_search_success_rate"], errors="raise").to_numpy()
        low = pd.to_numeric(ordered["search_success_ci_low"], errors="raise").to_numpy()
        high = pd.to_numeric(ordered["search_success_ci_high"], errors="raise").to_numpy()
        ax.errorbar(x, yv, yerr=np.vstack([yv - low, high - yv]), marker=markers[index], linestyle=styles[index], color=colors[index], capsize=3, label=str(scope).replace("_", " "))
    ax.set_xticks([0, 1, 2], ["restricted", "primary", "expanded"])
    ax.set_ylabel("Heuristic search-success rate")
    ax.set_xlabel("Search budget")
    ax.set_title("Figure S3. Heuristic counterfactual-search budget sensitivity")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    s3_png = ctx.output_root / "figures/supplementary/figure_s3_heuristic_budget_sensitivity.png"
    s3_svg = ctx.output_root / "figures/supplementary/figure_s3_heuristic_budget_sensitivity.svg"
    fig.savefig(s3_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(s3_svg, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    s3_source = ctx.output_root / "source_data/figures/figure_s3_heuristic_budget_sensitivity_source.csv"
    _write_csv(s3_source, budget.drop(columns=["_canonical_csv_row"]).to_dict(orient="records"))

    supplemental = [
        ("s1", "Full support-aware subgroup diagnostic heatmap", s1_png, s1_svg, s1_source, [table8]),
        ("s2", "Class-specific grouped OOF SHAP summary", s2_png, s2_svg, s2_source, class_paths),
        ("s3", "Heuristic counterfactual-search budget sensitivity", s3_png, s3_svg, s3_source, [budget_path]),
    ]
    for label, title, png, svg, source_csv, sources in supplemental:
        with Image.open(png) as image_file:
            dpi_value = image_file.info.get("dpi", (0.0, 0.0))
            if min(dpi_value) < 299.0 or min(image_file.size) < 1000:
                raise ManuscriptAssetExportError(f"Supplementary figure {label} fails raster QA.")
        _validate_svg(svg)
        caption = ctx.output_root / "figures/captions" / f"figure_{label}_caption.md"
        alt = ctx.output_root / "figures/alt_text" / f"figure_{label}_alt_text.txt"
        boundary = {
            "s1": "Support-aware descriptive gaps only; blank cells are not zero and no fairness or discrimination conclusion is supported.",
            "s2": "Exact-fold grouped SHAP in raw-margin space; model attribution is not causality.",
            "s3": "Heuristic search-success sensitivity only; not causal recourse, feasibility, actionability, advice, or intervention evidence.",
        }[label]
        _write_text(caption, f"# Figure {label.upper()} caption\n\n{title}. {boundary}")
        _write_text(alt, f"{title}. Visual encodings include labels, markers, hatching, or explicit missing-cell symbols. {boundary}")
        for output in (png, svg, source_csv, caption, alt):
            ctx.bind(output, sources)


def _table_path(ctx: ExportContext, number: int) -> Path:
    names = {
        1: "table_01_dataset_roles_target_mappings_support.csv",
        2: "table_02_exact_primary_feature_policy.csv",
        3: "table_03_four_model_nested_benchmark.csv",
        4: "table_04_leakage_policy_sensitivity.csv",
        5: "table_05_cross_fitted_sigmoid_calibration.csv",
        6: "table_06_global_grouped_oof_shap.csv",
        7: "table_07_oof_shap_stability.csv",
        8: "table_08_support_aware_subgroup_diagnostics.csv",
        9: "table_09_department_proxy_reconstructability.csv",
        10: "table_10_hrdataset_v14_mapped_target_replication.csv",
    }
    return ctx.canonical_root / "core/core_tables" / names[number]


def _select_one(frame: pd.DataFrame, **conditions: str) -> dict[str, str]:
    selected = frame
    for column, value in conditions.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ManuscriptAssetExportError(f"Expected one row for {conditions}, found {len(selected)}.")
    return {str(key): str(value) for key, value in selected.iloc[0].to_dict().items()}


def _policy_difference(frame: pd.DataFrame, policy: str, primary: str, metric: str) -> tuple[dict[str, str], str, str, str]:
    rows = frame[(frame["source_record_type"] == "policy_pairwise_tests") & (frame["metric"] == metric)]
    rows = rows[((rows["system_a"] == policy) & (rows["system_b"] == primary)) | ((rows["system_a"] == primary) & (rows["system_b"] == policy))]
    if len(rows) != 1:
        raise ManuscriptAssetExportError(f"Missing exact policy difference for {policy}/{metric}.")
    row = {str(key): str(value) for key, value in rows.iloc[0].to_dict().items()}
    value, low, high = row["raw_difference_a_minus_b"], row["raw_difference_ci_low"], row["raw_difference_ci_high"]
    if row["system_a"] != policy:
        value, low, high = _invert_triplet(value, low, high)
    return row, value, low, high


def _finish_table(
    ctx: ExportContext,
    number: int,
    stem: str,
    title: str,
    rows: list[dict[str, Any]],
    preview_columns: Sequence[str],
    source_paths: Sequence[Path],
    note: str,
) -> None:
    output_csv = ctx.output_root / "tables/main" / f"{stem}.csv"
    output_md = ctx.output_root / "tables/main" / f"{stem}.md"
    source_map_path = ctx.output_root / "tables/main" / f"{stem}.source_map.json"
    _write_csv(output_csv, rows)
    _write_text(output_md, _markdown(rows, preview_columns, title, note))
    source_map = {
        "schema_version": 1,
        "manuscript_table": number,
        "title": title,
        "output_csv": ctx.relative(output_csv),
        "output_sha256": _sha256(output_csv),
        "rounding_rule": ROUNDING_RULE,
        "canonical_sources": [
            {"path": ctx.relative(path), "sha256": _sha256(path)} for path in source_paths
        ],
        "run_id": RUN_ID,
        "config_hash": CONFIG_HASH,
        "scientific_input_hash": CORE_SCIENTIFIC_INPUT_HASH,
        "source_tree_hash": SOURCE_TREE_HASH,
        "row_count": len(rows),
        "validation_status": "source_rows_and_display_values_verified",
    }
    _write_json(source_map_path, source_map)
    for output in (output_csv, output_md, source_map_path):
        ctx.bind(output, source_paths)
    ctx.table_outputs[number] = {
        "stem": stem,
        "title": title,
        "rows": rows,
        "csv": output_csv,
        "markdown": output_md,
        "source_map": source_map_path,
        "sources": list(source_paths),
    }


def _build_tables_1_2(ctx: ExportContext) -> None:
    table1 = _table_path(ctx, 1); frame1 = _read_csv(table1)
    cards = frame1[frame1["source_record_type"] == "dataset_cards"]
    rows1: list[dict[str, Any]] = []
    for _, row in cards.iterrows():
        rows1.append(
            {
                "dataset": row["dataset_id"],
                "canonical_name": row["canonical_name"],
                "analytical_role": row["role"],
                "task_type": row["task_type"],
                "target": row["target_column"],
                "target_definition": row["target_mapping"],
                "observed_support": row["target_mapping_support"],
                "n_rows": str(int(Decimal(row["row_count"]))),
                "allowed_interpretation": row["allowed_claim"],
                "licence_status": row["licence_verification_status"],
            }
        )
    _finish_table(ctx, 1, TABLE_SPECS[0][1], TABLE_SPECS[0][2], rows1, list(rows1[0]), [table1], "Observed support and analytical role only; source authenticity, licence, and target equivalence remain manual boundaries.")

    table2 = _table_path(ctx, 2); frame2 = _read_csv(table2)
    rows2: list[dict[str, Any]] = []
    for _, row in frame2.sort_values("policy_order", key=lambda values: pd.to_numeric(values)).iterrows():
        primary = row["role"] == "canonical_primary"
        audit = row["audit_only"].casefold() == "true"
        rows2.append(
            {
                "policy": row["policy"],
                "analytical_role": row["role"],
                "status": "primary" if primary else ("audit_only" if audit else "matched_sensitivity"),
                "feature_count": str(int(Decimal(row["n_features"]))),
                "retained_features": "; ".join(json.loads(row["feature_columns_json"])),
                "excluded_features": "; ".join(json.loads(row["excluded_features_json"])),
                "supported_interpretation": "prespecified primary" if primary else "feature-access sensitivity; not an alternative primary model",
            }
        )
    _finish_table(ctx, 2, TABLE_SPECS[1][1], TABLE_SPECS[1][2], rows2, list(rows2[0]), [table2], "Policies are prespecified governance contracts. Audit-only systems cannot replace the primary model.")


def _build_table_3(ctx: ExportContext) -> None:
    path = _table_path(ctx, 3); frame = _read_csv(path)
    summary = frame[frame["source_record_type"] == "model_summary"]
    differences = frame[frame["source_record_type"] == "paired_model_differences"]
    order = ["xgboost", "logistic_regression", "random_forest", "lightgbm"]
    rows: list[dict[str, Any]] = []
    metrics = {
        "macro_f1": "macro_f1",
        "quadratic_weighted_kappa": "qwk",
        "balanced_accuracy": "balanced_accuracy",
        "ordinal_mae": "ordinal_mae",
        "severe_error_rate": "severe_error_rate",
    }
    for model in order:
        out: dict[str, Any] = {"model": model, "analytical_role": "XGBoost reference" if model == "xgboost" else "predeclared baseline"}
        for metric, prefix in metrics.items():
            row = _select_one(summary, system_id=model, metric=metric)
            ctx.add_measure(out, prefix=prefix, value=row["point_estimate"], low=row["ci_low"], high=row["ci_high"], source_table=path, source_row=row, metric=metric, manuscript_table="Table 3", claim_id=f"T03_{model}_{metric}", dataset="inx_primary", model=model, policy="no_salary_hike_no_attrition_no_department", denominator=row["n_samples"], uncertainty_method=row["method"], direction=row["better_direction"])
        if model == "xgboost":
            out.update({"baseline_minus_xgboost_macro_f1_full_precision": "N/A", "baseline_minus_xgboost_macro_f1_display": "N/A", "baseline_minus_xgboost_macro_f1_ci_lower_full_precision": "N/A", "baseline_minus_xgboost_macro_f1_ci_upper_full_precision": "N/A", "baseline_minus_xgboost_macro_f1_interval_display": "N/A"})
        else:
            row = _select_one(differences, system_a=model, system_b="xgboost", metric="macro_f1")
            ctx.add_measure(out, prefix="baseline_minus_xgboost_macro_f1", value=row["raw_difference_a_minus_b"], low=row["raw_difference_ci_low"], high=row["raw_difference_ci_high"], source_table=path, source_row=row, metric="baseline_minus_xgboost_macro_f1", manuscript_table="Table 3", claim_id=f"T03_{model}_minus_xgboost_macro_f1", dataset="inx_primary", model=model, policy="no_salary_hike_no_attrition_no_department", denominator=row["n_samples"], uncertainty_method=row["method"], direction="higher")
        rows.append(out)
    preview = ["model", "analytical_role", "macro_f1_display", "macro_f1_interval_display", "qwk_display", "qwk_interval_display", "balanced_accuracy_display", "ordinal_mae_display", "severe_error_rate_display", "baseline_minus_xgboost_macro_f1_display", "baseline_minus_xgboost_macro_f1_interval_display"]
    _finish_table(ctx, 3, TABLE_SPECS[2][1], TABLE_SPECS[2][2], rows, preview, [path], "Paired sample-level OOF bootstrap, 5,000 draws. The superiority gate is macro-F1-specific and did not replace the predeclared XGBoost XAI reference.")


def _build_table_4(ctx: ExportContext) -> None:
    path = _table_path(ctx, 4); frame = _read_csv(path)
    summaries = frame[frame["source_record_type"] == "manuscript_policy_table"].copy()
    primary = "no_salary_hike_no_attrition_no_department"
    rows: list[dict[str, Any]] = []
    for _, series in summaries.iterrows():
        row = {str(k): str(v) for k, v in series.to_dict().items()}; policy = row["policy"]
        audit = row["audit_only"].casefold() == "true"
        out: dict[str, Any] = {
            "policy": policy,
            "role": row["role"],
            "status": "primary" if policy == primary else ("audit_only" if audit else "matched_sensitivity"),
            "feature_count": str(int(Decimal(row["n_features"]))),
            "excluded_features": "; ".join(json.loads(row["excluded_features_json"])),
            "supported_interpretation": "prespecified primary" if policy == primary else "matched feature-access sensitivity; not causal and not an alternative primary model",
        }
        for metric, prefix in (("macro_f1", "macro_f1"), ("quadratic_weighted_kappa", "qwk"), ("ordinal_mae", "ordinal_mae"), ("severe_error_rate", "severe_error_rate")):
            ctx.add_measure(out, prefix=prefix, value=row[f"{metric}_oof"], low=row[f"{metric}_ci_low"], high=row[f"{metric}_ci_high"], source_table=path, source_row=row, metric=metric, manuscript_table="Table 4", claim_id=f"T04_{policy}_{metric}", dataset="inx_primary", model="xgboost", policy=policy, denominator=row["n_samples"], uncertainty_method=row["confidence_interval_method"], direction="lower" if metric in {"ordinal_mae", "severe_error_rate"} else "higher")
        for metric, prefix in (("macro_f1", "macro_f1_difference_vs_primary"), ("quadratic_weighted_kappa", "qwk_difference_vs_primary")):
            if policy == primary:
                out.update({f"{prefix}_full_precision": "N/A", f"{prefix}_display": "N/A", f"{prefix}_ci_lower_full_precision": "N/A", f"{prefix}_ci_upper_full_precision": "N/A", f"{prefix}_interval_display": "N/A"})
            else:
                diff_row, value, low, high = _policy_difference(frame, policy, primary, metric)
                ctx.add_measure(out, prefix=prefix, value=value, low=low, high=high, source_table=path, source_row=diff_row, metric=f"{metric}_policy_minus_primary", manuscript_table="Table 4", claim_id=f"T04_{policy}_minus_primary_{metric}", dataset="inx_primary", model="xgboost", policy=policy, denominator=diff_row["n_samples"], uncertainty_method=diff_row["method"], direction="higher")
        rows.append(out)
    preview = ["policy", "role", "status", "feature_count", "macro_f1_display", "macro_f1_difference_vs_primary_display", "qwk_display", "qwk_difference_vs_primary_display", "ordinal_mae_display", "severe_error_rate_display", "supported_interpretation"]
    _finish_table(ctx, 4, TABLE_SPECS[3][1], TABLE_SPECS[3][2], rows, preview, [path, _table_path(ctx, 2)], "Same outer folds and selected candidate schedule. Audit-only policies are sensitivity models, not alternate primary systems.")


def _build_table_5(ctx: ExportContext) -> None:
    path = _table_path(ctx, 5); frame = _read_csv(path)
    intervals = frame[frame["source_record_type"] == "calibration_metric_intervals"]
    differences = frame[frame["source_record_type"] == "calibration_paired_differences"]
    rows: list[dict[str, Any]] = []
    for method in ("raw", "sigmoid"):
        out: dict[str, Any] = {"method": method, "fitting_scope": "not_applicable_raw_probabilities" if method == "raw" else "outer-training five-fold cross-fitted probabilities only", "outer_test_used_for_fitting": "False"}
        for metric, prefix in (("nll_log_loss", "log_loss"), ("multiclass_brier", "brier_score"), ("ece_confidence", "ece")):
            row = _select_one(intervals, system_id=method, metric=metric)
            ctx.add_measure(out, prefix=prefix, value=row["point_estimate"], low=row["ci_low"], high=row["ci_high"], source_table=path, source_row=row, metric=metric, manuscript_table="Table 5", claim_id=f"T05_{method}_{metric}", dataset="inx_primary", model="xgboost", policy="no_salary_hike_no_attrition_no_department", denominator=row["n_samples"], uncertainty_method=row["method"], direction="lower")
            diff_prefix = f"{prefix}_sigmoid_minus_raw"
            if method == "raw":
                out.update({f"{diff_prefix}_full_precision": "N/A", f"{diff_prefix}_display": "N/A", f"{diff_prefix}_ci_lower_full_precision": "N/A", f"{diff_prefix}_ci_upper_full_precision": "N/A", f"{diff_prefix}_interval_display": "N/A"})
            else:
                diff = _select_one(differences, system_a="sigmoid", system_b="raw", metric=metric)
                ctx.add_measure(out, prefix=diff_prefix, value=diff["raw_difference_a_minus_b"], low=diff["raw_difference_ci_low"], high=diff["raw_difference_ci_high"], source_table=path, source_row=diff, metric=f"{metric}_sigmoid_minus_raw", manuscript_table="Table 5", claim_id=f"T05_sigmoid_minus_raw_{metric}", dataset="inx_primary", model="xgboost", policy="no_salary_hike_no_attrition_no_department", denominator=diff["n_samples"], uncertainty_method=diff["method"], direction="lower")
        rows.append(out)
    preview = ["method", "log_loss_display", "log_loss_interval_display", "log_loss_sigmoid_minus_raw_display", "brier_score_display", "brier_score_interval_display", "brier_score_sigmoid_minus_raw_display", "ece_display", "ece_interval_display", "ece_sigmoid_minus_raw_display", "fitting_scope", "outer_test_used_for_fitting"]
    _finish_table(ctx, 5, TABLE_SPECS[4][1], TABLE_SPECS[4][2], rows, preview, [path], "Calibration concerns probability reliability. Sigmoid-minus-raw paired differences do not imply automatic classification improvement or deployment readiness.")


def _build_table_6(ctx: ExportContext) -> None:
    global_path = _table_path(ctx, 6); stability_path = _table_path(ctx, 7)
    global_frame = _read_csv(global_path); stability = _read_csv(stability_path)
    ranking_path = ctx.canonical_root / "core/oof_shap/fold_feature_rankings.csv"; rankings = _read_csv(ranking_path)
    rankings["rank_numeric"] = pd.to_numeric(rankings["rank"], errors="raise")
    frequency = rankings.assign(top10=rankings["rank_numeric"] <= 10).groupby("feature")["top10"].sum().to_dict()
    rows: list[dict[str, Any]] = []
    for _, series in global_frame.sort_values("rank", key=lambda values: pd.to_numeric(values)).iterrows():
        row = {str(k): str(v) for k, v in series.to_dict().items()}; out = {"panel": "A_attribution", "rank": row["rank"], "feature_family": row["feature"], "fold_top10_frequency": str(int(frequency.get(row["feature"], 0))), "governance_category": f"{row['control_type']}; sensitive_or_proxy={row['sensitive_or_proxy']}; leakage_risk={row['leakage_risk']}", "temporality_status": row["governance_notes"]}
        ctx.add_measure(out, prefix="mean_absolute_grouped_shap", value=row["mean_abs_grouped_shap"], source_table=global_path, source_row=row, metric="mean_abs_grouped_shap", manuscript_table="Table 6A", claim_id=f"T06A_rank_{row['rank']}_{row['feature']}", dataset="inx_primary", model="xgboost", policy=row["policy"], denominator="1200", uncertainty_method="descriptive_complete_oof_attribution", direction="descriptive")
        rows.append(out)
    pairwise = stability[stability["source_record_type"] == "shap_stability_pairwise"].copy()
    summaries = stability[stability["source_record_type"] == "shap_stability_summary"].copy()
    for _, series in summaries.sort_values("top_k", key=lambda values: pd.to_numeric(values)).iterrows():
        row = {str(k): str(v) for k, v in series.to_dict().items()}; subset = pairwise[pairwise["top_k"] == row["top_k"]]
        jaccard = pd.to_numeric(subset["top_k_jaccard"], errors="raise"); spearman = pd.to_numeric(subset["spearman_all_features"], errors="raise")
        out = {"panel": "B_stability", "top_k": row["top_k"], "fold_pair_count": row["n_fold_pairs"], "jaccard_median_full_precision": row["jaccard_median"], "jaccard_median_display": _format_decimal(row["jaccard_median"]), "jaccard_iqr_full_precision": f"{jaccard.quantile(0.25, interpolation='linear')};{jaccard.quantile(0.75, interpolation='linear')}", "jaccard_iqr_display": f"[{_format_decimal(jaccard.quantile(0.25))}, {_format_decimal(jaccard.quantile(0.75))}]", "jaccard_range_full_precision": f"{row['jaccard_min']};{row['jaccard_max']}", "jaccard_range_display": f"[{_format_decimal(row['jaccard_min'])}, {_format_decimal(row['jaccard_max'])}]", "spearman_median_full_precision": row["spearman_median"], "spearman_median_display": _format_decimal(row["spearman_median"]), "spearman_iqr_display": f"[{_format_decimal(spearman.quantile(0.25))}, {_format_decimal(spearman.quantile(0.75))}]", "spearman_range_display": f"[{_format_decimal(row['spearman_min'])}, {_format_decimal(row['spearman_max'])}]", "inference_status": "descriptive_dependent_fold_pairs_no_confidence_interval"}
        ctx.add_measure(out, prefix="jaccard_descriptive", value=row["jaccard_median"], source_table=stability_path, source_row=row, metric="top_k_jaccard_median", manuscript_table="Table 6B", claim_id=f"T06B_top_{row['top_k']}_jaccard", dataset="inx_primary", model="xgboost", policy=row["policy"], denominator=row["n_fold_pairs"], uncertainty_method="descriptive_dependent_fold_pairs_no_confidence_interval", direction="higher")
        rows.append(out)
    preview = ["panel", "rank", "feature_family", "mean_absolute_grouped_shap_display", "fold_top10_frequency", "governance_category", "top_k", "fold_pair_count", "jaccard_median_display", "jaccard_iqr_display", "jaccard_range_display", "spearman_median_display", "spearman_iqr_display", "spearman_range_display", "inference_status"]
    _finish_table(ctx, 6, TABLE_SPECS[5][1], TABLE_SPECS[5][2], rows, preview, [global_path, stability_path, ranking_path], "Panel A is raw-margin model attribution. Panel B summarizes dependent fold pairs descriptively; no population confidence interval is constructed.")


def _build_table_7(ctx: ExportContext) -> None:
    subgroup_path = _table_path(ctx, 8); proxy_path = _table_path(ctx, 9)
    subgroup = _read_csv(subgroup_path); proxy = _read_csv(proxy_path)
    gaps = subgroup[(subgroup["source_record_type"] == "fairness_disparity_uncertainty") & (subgroup["policy"] == "no_salary_hike_no_attrition_no_department") & (subgroup["gap"] != "") & (subgroup["estimate_status"] == "support_sufficient_descriptive_estimate")]
    rows: list[dict[str, Any]] = []
    for _, series in gaps.iterrows():
        row = {str(k): str(v) for k, v in series.to_dict().items()}; out = {"panel": "A_supported_subgroup", "audited_attribute": row["attribute"], "eligible_groups": row["included_groups_json"], "support_context": f"groups={row['n_groups_included']}; minimum_group_support={row['minimum_subgroup_support']}; metric_denominator>={row['minimum_metric_denominator']}", "metric": row["metric"], "class_label": row["class_label"] or "N/A", "support_status": row["estimate_status"], "headline_eligible": row["headline_eligible"]}
        ctx.add_measure(out, prefix="largest_supported_gap", value=row["gap"], low=row["ci_low"], high=row["ci_high"], source_table=subgroup_path, source_row=row, metric=f"{row['metric']}_subgroup_gap", manuscript_table="Table 7A", claim_id=f"T07A_{row['attribute']}_{row['metric']}_{row['class_label'] or 'overall'}", dataset="inx_primary", model="xgboost", policy=row["policy"], denominator=row["denominator"], uncertainty_method=row["bootstrap_method"], direction="descriptive")
        rows.append(out)
    intervals = proxy[proxy["source_record_type"] == "proxy_metric_intervals"]
    for _, series in intervals.iterrows():
        row = {str(k): str(v) for k, v in series.to_dict().items()}; out = {"panel": "B_proxy_reconstruction", "predictor_policy": row["system_id"], "metric": row["metric"], "target_class_support": row["proxy_target_class_counts_json"], "support_context": f"min overall={row['minimum_proxy_target_class_support']}; min nonzero outer-test={row['minimum_nonzero_outer_test_class_support']}; zero cells={row['zero_support_outer_test_cells']}", "estimation_status": "estimated_with_support_context", "interpretation": row["limitations"]}
        ctx.add_measure(out, prefix="proxy_metric", value=row["point_estimate"], low=row["ci_low"], high=row["ci_high"], source_table=proxy_path, source_row=row, metric=f"proxy_{row['metric']}", manuscript_table="Table 7B", claim_id=f"T07B_{row['system_id']}_{row['metric']}", dataset="inx_primary", model="proxy_diagnostic_xgboost", policy=row["system_id"], denominator=row["n_samples"], uncertainty_method="paired_stratified_percentile_bootstrap_5000_draws", direction="higher")
        rows.append(out)
    preview = ["panel", "audited_attribute", "eligible_groups", "metric", "class_label", "largest_supported_gap_display", "largest_supported_gap_interval_display", "predictor_policy", "proxy_metric_display", "proxy_metric_interval_display", "support_context", "support_status", "estimation_status"]
    _finish_table(ctx, 7, TABLE_SPECS[6][1], TABLE_SPECS[6][2], rows, preview, [subgroup_path, proxy_path], "Subgroup gaps and department reconstructability are descriptive proxy-risk diagnostics only. Non-estimable results must remain N/A, never zero.")


def _build_table_8(ctx: ExportContext) -> None:
    path = _table_path(ctx, 10); frame = _read_csv(path)
    metadata = _select_one(frame, source_record_type="external_replication_metadata")
    policy_order = json.loads(metadata["policy_order"]); primary = metadata["primary_policy"]
    raw = frame[frame["source_record_type"] == "raw_metric_intervals"]
    cal = frame[frame["source_record_type"] == "calibration_metric_intervals"]
    diffs = frame[frame["source_record_type"] == "policy_pairwise_differences"]
    support = frame[frame["source_record_type"] == "target_support"]
    support_text = "; ".join(
        f"{row.target_value}={int(Decimal(row['count']))}"
        for _, row in support[support["support_scale"] == "mapped"].iterrows()
    )
    feature_path = ctx.canonical_root / "core/external_replication/feature_policy_features.csv"; features = _read_csv(feature_path)
    feature_counts = features[features["included"].str.casefold() == "true"].groupby("policy")["feature"].count().to_dict()
    rows: list[dict[str, Any]] = []
    for policy in policy_order:
        out: dict[str, Any] = {"external_policy": policy, "role": "canonical_external_primary" if policy == primary else "audit_or_sensitivity_only", "feature_count": str(int(feature_counts.get(policy, 0))), "mapped_n": "311", "mapped_support": support_text, "interpretation": "independently trained mapped-target replication" if policy == primary else "audit/sensitivity only; cannot replace primary"}
        for metric, prefix in (("macro_f1", "macro_f1"), ("quadratic_weighted_kappa", "qwk"), ("ordinal_mae", "ordinal_mae"), ("severe_error_rate", "severe_error_rate")):
            row = _select_one(raw, system_id=policy, metric=metric)
            ctx.add_measure(out, prefix=prefix, value=row["point_estimate"], low=row["ci_low"], high=row["ci_high"], source_table=path, source_row=row, metric=metric, manuscript_table="Table 8", claim_id=f"T08_{policy}_{metric}", dataset="hrdataset_v14", model="independently_trained_xgboost", policy=policy, denominator=row["n_samples"], uncertainty_method=row["method"], direction=row["better_direction"])
        for metric, prefix in (("macro_f1", "macro_f1_difference_vs_primary"), ("quadratic_weighted_kappa", "qwk_difference_vs_primary")):
            if policy == primary:
                out.update({f"{prefix}_full_precision": "N/A", f"{prefix}_display": "N/A", f"{prefix}_ci_lower_full_precision": "N/A", f"{prefix}_ci_upper_full_precision": "N/A", f"{prefix}_interval_display": "N/A"})
            else:
                candidates = diffs[(diffs["metric"] == metric) & (((diffs["system_a"] == policy) & (diffs["system_b"] == primary)) | ((diffs["system_a"] == primary) & (diffs["system_b"] == policy)))]
                if len(candidates) != 1: raise ManuscriptAssetExportError(f"Missing external policy difference {policy}/{metric}")
                row = {str(k): str(v) for k, v in candidates.iloc[0].to_dict().items()}; value, low, high = row["raw_difference_a_minus_b"], row["raw_difference_ci_low"], row["raw_difference_ci_high"]
                if row["system_a"] != policy: value, low, high = _invert_triplet(value, low, high)
                ctx.add_measure(out, prefix=prefix, value=value, low=low, high=high, source_table=path, source_row=row, metric=f"{metric}_policy_minus_primary", manuscript_table="Table 8", claim_id=f"T08_{policy}_minus_primary_{metric}", dataset="hrdataset_v14", model="independently_trained_xgboost", policy=policy, denominator=row["n_samples"], uncertainty_method=row["method"], direction="higher")
        for method in ("raw", "sigmoid"):
            for metric, prefix in (("nll_log_loss", "log_loss"), ("multiclass_brier", "brier"), ("ece_confidence", "ece")):
                if policy == primary:
                    row = _select_one(cal, system_id=method, metric=metric)
                    ctx.add_measure(out, prefix=f"{method}_{prefix}", value=row["point_estimate"], low=row["ci_low"], high=row["ci_high"], source_table=path, source_row=row, metric=f"{method}_{metric}", manuscript_table="Table 8", claim_id=f"T08_{method}_{metric}", dataset="hrdataset_v14", model="independently_trained_xgboost", policy=primary, denominator=row["n_samples"], uncertainty_method=row["method"], direction="lower")
                else:
                    out.update({f"{method}_{prefix}_full_precision": "N/A", f"{method}_{prefix}_display": "N/A", f"{method}_{prefix}_ci_lower_full_precision": "N/A", f"{method}_{prefix}_ci_upper_full_precision": "N/A", f"{method}_{prefix}_interval_display": "N/A"})
        rows.append(out)
    preview = ["external_policy", "role", "feature_count", "mapped_n", "mapped_support", "macro_f1_display", "macro_f1_interval_display", "qwk_display", "qwk_interval_display", "ordinal_mae_display", "severe_error_rate_display", "macro_f1_difference_vs_primary_display", "raw_log_loss_display", "sigmoid_log_loss_display", "raw_brier_display", "sigmoid_brier_display", "interpretation"]
    _finish_table(ctx, 8, TABLE_SPECS[7][1], TABLE_SPECS[7][2], rows, preview, [path, feature_path], "Independent mapped-target replication. It is not locked-model transport, target equivalence, fairness evidence, or deployment validation.")


def _export_main_tables(ctx: ExportContext) -> None:
    _build_tables_1_2(ctx)
    _build_table_3(ctx)
    _build_table_4(ctx)
    _build_table_5(ctx)
    _build_table_6(ctx)
    _build_table_7(ctx)
    _build_table_8(ctx)


def _export_supplementary_tables(ctx: ExportContext) -> None:
    source_root = ctx.canonical_root / "supplementary/supplementary_tables"
    specs = [
        (11, "table_s11_heuristic_search_success", "table_11_heuristic_counterfactual_search_success.csv", ["candidate_feature_scope", "budget_id", "n_eligible_oof_cases", "n_search_successes", "heuristic_search_success_rate", "search_success_ci_low", "search_success_ci_high", "evidence_role"]),
        (12, "table_s12_task_bounded_robustness", "table_12_restricted_and_binary_task_evidence.csv", ["source_record_type", "task_key", "policy", "metric", "point_estimate", "ci_low", "ci_high", "claim_boundary"]),
        (13, "table_s13_claim_boundaries", "table_13_supplementary_reproducibility_and_claim_boundaries.csv", ["source_record_type", "task_type", "metric", "applicable", "not_applicable_reason", "claim_boundary"]),
    ]
    for number, stem, filename, columns in specs:
        source = source_root / filename; frame = _read_csv(source)
        if number == 11:
            selected = frame[frame["source_record_type"].isin(["heuristic_search_summary", "heuristic_search_budget_sensitivity"])]
        elif number == 12:
            selected = frame[frame["source_record_type"].isin(["task_strata_index", "ibm_restricted_target_performance_robustness", "ibm_attrition_task_transfer", "employee_turnover_task_transfer"])]
        else:
            selected = frame[frame["source_record_type"] == "claim_boundary"]
        rows = [{column: row.get(column, "") for column in columns} for row in selected.to_dict(orient="records")]
        output_csv = ctx.output_root / "tables/supplementary" / f"{stem}.csv"
        output_md = ctx.output_root / "tables/supplementary" / f"{stem}.md"
        source_map = ctx.output_root / "tables/supplementary" / f"{stem}.source_map.json"
        _write_csv(output_csv, rows, columns)
        _write_text(output_md, _markdown(rows, columns, f"Supplementary Table S{number - 10}", "Canonical supplementary evidence; task and claim boundaries remain explicit."))
        _write_json(source_map, {"schema_version": 1, "supplementary_table": f"S{number-10}", "output_csv": ctx.relative(output_csv), "source_path": ctx.relative(source), "source_sha256": _sha256(source), "run_id": RUN_ID, "config_hash": CONFIG_HASH, "scientific_input_hash": SUPPLEMENTARY_SCIENTIFIC_INPUT_HASH, "source_tree_hash": SOURCE_TREE_HASH, "rounding_rule": ROUNDING_RULE, "row_count": len(rows), "validation_status": "verified"})
        for output in (output_csv, output_md, source_map): ctx.bind(output, [source])


def _copy_canonical_source_tables(ctx: ExportContext) -> None:
    destinations: list[tuple[Path, Path]] = []
    for scope, directory in (("core", ctx.canonical_root / "core/core_tables"), ("supplementary", ctx.canonical_root / "supplementary/supplementary_tables")):
        for source in sorted(directory.glob("table_*.csv")):
            destination = ctx.output_root / "source_data/tables" / f"{scope}_{source.name}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if _sha256(source) != _sha256(destination):
                raise ManuscriptAssetExportError(f"Source-table copy parity failed: {source}")
            ctx.bind(destination, [source])
            destinations.append((source, destination))
    if len(destinations) != 14:
        raise ManuscriptAssetExportError("Expected exactly fourteen canonical source-table copies.")


def _handoff_files(ctx: ExportContext) -> None:
    handoff = ctx.output_root / "handoff"
    exact_results = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "generation_commit": GENERATION_COMMIT,
        "config_hash": CONFIG_HASH,
        "source_tree_hash": SOURCE_TREE_HASH,
        "core_scientific_input_hash": CORE_SCIENTIFIC_INPUT_HASH,
        "tables": {
            str(number): {
                "path": ctx.relative(value["csv"]),
                "sha256": _sha256(value["csv"]),
                "rows": value["rows"],
            }
            for number, value in ctx.table_outputs.items()
            if 3 <= number <= 8
        },
        "verification_status": "derived_only_from_verified_canonical_source_rows",
    }
    exact_path = handoff / "manuscript_exact_results.json"; _write_json(exact_path, exact_results)
    ctx.bind(exact_path, [item for number in range(3, 11) for item in ([_table_path(ctx, number)] if number in {3,4,5,6,7,8,9,10} else [])])

    ledger_fields = ["claim_id", "manuscript_section", "object_type", "manuscript_table_or_figure", "metric_or_statement", "full_precision_value", "displayed_value", "interval_lower", "interval_upper", "denominator", "dataset", "model", "policy", "uncertainty_method", "canonical_source_path", "source_row_key", "source_column", "source_sha256", "run_id", "config_hash", "scientific_input_hash", "source_tree_hash", "verification_status"]
    ledger_path = handoff / "manuscript_evidence_ledger.csv"; _write_csv(ledger_path, ctx.ledger, ledger_fields)
    ctx.bind(ledger_path, sorted({ctx.project_root / row["canonical_source_path"] for row in ctx.ledger}))

    figure_map_path = handoff / "figure_source_map.csv"; _write_csv(figure_map_path, ctx.figure_rows)
    ctx.bind(figure_map_path, [ctx.canonical_root / "core/core_figures/figure_manifest.json"])
    table_map_rows = []
    for number, value in sorted(ctx.table_outputs.items()):
        table_map_rows.append({"manuscript_table": number, "title": value["title"], "asset_csv": ctx.relative(value["csv"]), "asset_markdown": ctx.relative(value["markdown"]), "source_map": ctx.relative(value["source_map"]), "asset_sha256": _sha256(value["csv"]), "canonical_source_paths": ";".join(ctx.relative(path) for path in value["sources"]), "canonical_source_sha256_values": ";".join(_sha256(path) for path in value["sources"]), "run_id": RUN_ID, "config_hash": CONFIG_HASH, "scientific_input_hash": CORE_SCIENTIFIC_INPUT_HASH, "source_tree_hash": SOURCE_TREE_HASH})
    table_map_path = handoff / "table_source_map.csv"; _write_csv(table_map_path, table_map_rows)
    ctx.bind(table_map_path, [path for value in ctx.table_outputs.values() for path in value["sources"]])
    result_map_path = handoff / "result_source_map.csv"; _write_csv(result_map_path, ctx.ledger, ledger_fields)
    ctx.bind(result_map_path, sorted({ctx.project_root / row["canonical_source_path"] for row in ctx.ledger}))

    guide_lines = ["# Figure and table insertion guide", "", f"All assets derive from canonical run `{RUN_ID}` without a scientific rerun.", "", "## Main figures", ""]
    for row in ctx.figure_rows:
        guide_lines.append(f"- Figure {row['manuscript_figure_number']}: `{row['exported_png_path']}` or `{row['exported_svg_path']}`; caption `{row['caption_path']}`; source `{row['exported_source_data_path']}`. {MAIN_CAPTION_BOUNDARIES[int(row['manuscript_figure_number'])]}")
    guide_lines.extend(["", "## Main tables", ""])
    for number, value in sorted(ctx.table_outputs.items()):
        guide_lines.append(f"- Table {number}: `{ctx.relative(value['csv'])}`; preview `{ctx.relative(value['markdown'])}`; source map `{ctx.relative(value['source_map'])}`.")
    guide_lines.extend(["", "## Mandatory boundaries", "", "- Paired OOF intervals condition on observed samples and the fixed protocol.", "- SHAP is noncausal raw-margin model attribution.", "- Subgroup/proxy diagnostics do not establish fairness or discrimination.", "- HRDataset_v14 is independently trained mapped-target replication, not locked-model transport.", "- Heuristic counterfactual-search success is not recourse, actionability, feasibility, advice, or intervention evidence."])
    guide_path = handoff / "figure_table_insertion_guide.md"; _write_text(guide_path, "\n".join(guide_lines))
    ctx.bind(guide_path, [ctx.canonical_root / "core/canonical_claim_boundaries.md", ctx.canonical_root / "supplementary/canonical_claim_boundaries.md"])

    claim_path = handoff / "claim_boundary_handoff.md"
    _write_text(claim_path, """# Claim-boundary handoff

## Supported

- Within-dataset exactly-once OOF benchmark evidence under the frozen 10 x 5 protocol.
- Matched feature-access sensitivity conditional on the primary selection schedule.
- Predeclared cross-fitted sigmoid probability-quality evidence.
- Exact-fold grouped OOF SHAP attribution and descriptive stability.
- Support-aware descriptive subgroup/proxy diagnostics.
- Independently trained HRDataset_v14 mapped-target replication.

## Conditionally supported

- Paired bootstrap intervals condition on observed samples and the fixed training protocol.
- Audit-only policy values are sensitivities, not alternative primary systems.
- Proxy reconstruction is a proxy-risk diagnostic with explicit target support.

## Prohibited

- Causality, fairness proof, discrimination finding, actionability, employee advice, legal compliance, deployment readiness, autonomous HR decisions, target equivalence, or locked-model transport.

## Insufficient support and supplementary boundaries

- Any canonical `not_estimated` or support-ineligible result remains N/A, never zero.
- Counterfactual outputs are heuristic search-success evidence only.
- IBM/turnover tasks remain separate supplementary task-transfer or restricted-target evidence and are not directly comparable with the primary task.
""")
    ctx.bind(claim_path, [ctx.canonical_root / "core/canonical_claim_boundaries.md", ctx.canonical_root / "supplementary/canonical_claim_boundaries.md"])


def _asset_readme(ctx: ExportContext) -> None:
    lines = ["# Canonical manuscript-support assets", "", f"This compact tracked package derives from canonical run `{RUN_ID}` at generation commit `{GENERATION_COMMIT}`. No model, calibration, bootstrap, OOF, or SHAP scientific computation was rerun.", "", "The complete closed-world evidence package remains local and ignored. These publication copies do not replace it and contain no raw dataset or employee-level record.", "", "## Main figures", ""]
    for row in ctx.figure_rows:
        number = row["manuscript_figure_number"]
        lines.append(f"- Figure {number}: [`PNG`](figures/main/{row['manuscript_stem']}.png), [`SVG`](figures/main/{row['manuscript_stem']}.svg), [`caption`](figures/captions/figure_{int(number):02d}_caption.md), [`alt text`](figures/alt_text/figure_{int(number):02d}_alt_text.txt).")
    lines.extend(["", "## Main tables", ""])
    for number, value in sorted(ctx.table_outputs.items()):
        lines.append(f"- Table {number}: [`CSV`](tables/main/{value['stem']}.csv), [`preview`](tables/main/{value['stem']}.md), [`source map`](tables/main/{value['stem']}.source_map.json).")
    lines.extend(["", "## Handoff", "", "- [`Insertion guide`](handoff/figure_table_insertion_guide.md)", "- [`Exact results`](handoff/manuscript_exact_results.json)", "- [`Evidence ledger`](handoff/manuscript_evidence_ledger.csv)", "- [`Claim boundaries`](handoff/claim_boundary_handoff.md)", "- [`Asset manifest`](manifests/manuscript_asset_manifest.json)"])
    path = ctx.output_root / "README.md"; _write_text(path, "\n".join(lines)); ctx.bind(path, [ctx.project_root / CANONICAL_RECEIPT])


def _export_manifest(ctx: ExportContext, source_validation: Mapping[str, Any]) -> Path:
    manifest_path = ctx.output_root / "manifests/manuscript_asset_manifest.json"
    files = []
    for path in sorted((p for p in ctx.output_root.rglob("*") if p.is_file() and p != manifest_path), key=lambda item: item.relative_to(ctx.output_root).as_posix()):
        repo_relative = ctx.relative(path)
        sources = ctx.lineage.get(repo_relative, [])
        files.append({"path": repo_relative, "type": path.suffix.lower().lstrip(".") or "text", "size_bytes": path.stat().st_size, "sha256": _sha256(path), "source_paths": [item["path"] for item in sources], "source_sha256_values": [item["sha256"] for item in sources], "run_id": RUN_ID, "config_hash": CONFIG_HASH, "scientific_input_hash": SUPPLEMENTARY_SCIENTIFIC_INPUT_HASH if "/supplementary/" in repo_relative or "figure_s" in path.name else CORE_SCIENTIFIC_INPUT_HASH, "source_tree_hash": SOURCE_TREE_HASH, "export_timestamp": ctx.export_timestamp, "copy_or_render_status": "deterministic_export", "validation_outcome": "passed"})
    payload = {"schema_version": 1, "manifest_kind": "compact_manuscript_support_assets", "self_inventory_rule": "The manifest cannot contain its own cryptographic hash; every other exported file is inventoried.", "run_id": RUN_ID, "generation_commit": GENERATION_COMMIT, "config_hash": CONFIG_HASH, "source_tree_hash": SOURCE_TREE_HASH, "core_scientific_input_hash": CORE_SCIENTIFIC_INPUT_HASH, "supplementary_scientific_input_hash": SUPPLEMENTARY_SCIENTIFIC_INPUT_HASH, "export_timestamp": ctx.export_timestamp, "source_validation": source_validation, "rounding_rule": ROUNDING_RULE, "files": files, "file_count_excluding_manifest": len(files), "total_bytes_excluding_manifest": sum(row["size_bytes"] for row in files), "status": "passed"}
    _write_json(manifest_path, payload)
    return manifest_path


def validate_manuscript_asset_package(
    project_root: Path,
    asset_root: Path,
    *,
    require_published_path: bool = True,
) -> dict[str, Any]:
    root = project_root.resolve(); assets = asset_root.resolve()
    if (
        (require_published_path and assets != (root / ASSET_RELATIVE_ROOT).resolve())
        or not assets.is_dir()
        or assets.is_symlink()
    ):
        raise ManuscriptAssetExportError("Asset package must be the exact regular configured directory.")
    manifest_path = assets / "manifests/manuscript_asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed" or manifest.get("run_id") != RUN_ID or manifest.get("generation_commit") != GENERATION_COMMIT:
        raise ManuscriptAssetExportError("Asset manifest identity/status mismatch.")
    records = manifest.get("files", []); expected: set[str] = set()
    for record in records:
        relative = str(record.get("path", "")); pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise ManuscriptAssetExportError(f"Unsafe asset path: {relative}")
        try:
            asset_relative = pure.relative_to(PurePosixPath(ASSET_RELATIVE_ROOT.as_posix()))
        except ValueError as exc:
            raise ManuscriptAssetExportError(f"Asset is outside the publication prefix: {relative}") from exc
        path = assets / Path(*asset_relative.parts)
        try:
            path.resolve().relative_to(assets)
        except ValueError as exc:
            raise ManuscriptAssetExportError(f"Asset escapes package: {relative}") from exc
        if not path.is_file() or path.is_symlink() or path.stat().st_size != int(record["size_bytes"]) or _sha256(path) != record["sha256"]:
            raise ManuscriptAssetExportError(f"Asset byte mismatch: {relative}")
        expected.add(relative)
    actual = {
        (ASSET_RELATIVE_ROOT / ctx_path.relative_to(assets)).as_posix()
        for ctx_path in assets.rglob("*")
        if ctx_path.is_file() and ctx_path != manifest_path
    }
    if expected != actual or len(records) != int(manifest["file_count_excluding_manifest"]):
        raise ManuscriptAssetExportError("Asset manifest closed-world inventory mismatch.")
    def actual_path(relative: str) -> Path:
        suffix = PurePosixPath(relative).relative_to(PurePosixPath(ASSET_RELATIVE_ROOT.as_posix()))
        return assets / Path(*suffix.parts)

    total = sum(actual_path(path).stat().st_size for path in actual) + manifest_path.stat().st_size
    if total >= 50 * 1024 * 1024 or any(actual_path(path).stat().st_size >= 100 * 1024 * 1024 for path in actual):
        raise ManuscriptAssetExportError("Asset package violates size gates.")
    pngs = sorted((assets / "figures").rglob("*.png")); svgs = sorted((assets / "figures").rglob("*.svg"))
    if len(list((assets / "figures/main").glob("*.png"))) != 7 or len(list((assets / "figures/main").glob("*.svg"))) != 7:
        raise ManuscriptAssetExportError("Main figure numbering/inventory mismatch.")
    if len(list((assets / "tables/main").glob("table_*.csv"))) != 8:
        raise ManuscriptAssetExportError("Main table numbering/inventory mismatch.")
    for table_csv in sorted((assets / "tables/main").glob("table_*.csv")):
        table_rows = list(csv.DictReader(table_csv.open(encoding="utf-8", newline="")))
        for row in table_rows:
            for field_name, full_value in row.items():
                if not field_name.endswith("_full_precision"):
                    continue
                if any(token in field_name for token in ("_ci_lower_", "_ci_upper_", "_iqr_", "_range_")):
                    continue
                display_name = field_name.removesuffix("_full_precision") + "_display"
                if display_name not in row:
                    raise ManuscriptAssetExportError(
                        f"Display field is missing for {field_name} in {table_csv}."
                    )
                if full_value == "" and row[display_name] == "":
                    # Wide multi-panel tables leave fields from the other panel
                    # structurally empty; these are not missing numerical results.
                    continue
                if row[display_name] != _format_decimal(full_value):
                    raise ManuscriptAssetExportError(
                        f"Display rounding mismatch for {field_name} in {table_csv}."
                    )
            for field_name, interval_value in row.items():
                if not field_name.endswith("_interval_display"):
                    continue
                prefix = field_name.removesuffix("_interval_display")
                lower = row.get(f"{prefix}_ci_lower_full_precision", "N/A")
                upper = row.get(f"{prefix}_ci_upper_full_precision", "N/A")
                if interval_value == "" and lower == "" and upper == "":
                    continue
                expected_interval = (
                    "N/A"
                    if lower in {"", "N/A"} or upper in {"", "N/A"}
                    else f"[{_format_decimal(lower)}, {_format_decimal(upper)}]"
                )
                if interval_value != expected_interval:
                    raise ManuscriptAssetExportError(
                        f"Display interval mismatch for {prefix} in {table_csv}."
                    )
    for png in pngs:
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
            if min(image.size) < 1000 or min(dpi) < 299.0: raise ManuscriptAssetExportError(f"PNG QA failed: {png}")
    for svg in svgs: _validate_svg(svg)
    forbidden_suffixes = {".joblib", ".pkl", ".pickle", ".parquet", ".npy", ".npz", ".cbm", ".xlsx", ".xls"}
    if any(path.suffix.casefold() in forbidden_suffixes for path in assets.rglob("*") if path.is_file()):
        raise ManuscriptAssetExportError("Serialized model/data artifact found in compact package.")
    machine = re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.IGNORECASE)
    secret = re.compile(r"(?i)(api[_-]?key|access[_-]?token|password)\s*[:=]\s*[\"'][^\"']{8,}")
    for path in (p for p in assets.rglob("*") if p.is_file() and p.suffix.casefold() in {".md", ".txt", ".csv", ".json", ".svg"}):
        text = path.read_text(encoding="utf-8")
        if machine.search(text) or secret.search(text): raise ManuscriptAssetExportError(f"Portable-path/secret scan failed: {path}")
    prohibited_headers = {"employee_id", "employeeid", "employee_name", "employeename", "empnumber", "sample_index"}
    for csv_path in assets.rglob("*.csv"):
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if prohibited_headers.intersection(value.casefold() for value in header):
            raise ManuscriptAssetExportError(f"Employee-level identifier header found: {csv_path}")
    readme_text = (assets / "README.md").read_text(encoding="utf-8")
    local_links = [
        match.group(1).split("#", 1)[0]
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", readme_text)
        if not re.match(r"^(?:https?://|mailto:|#)", match.group(1))
    ]
    missing_links = [link for link in local_links if link and not (assets / link).exists()]
    if missing_links:
        raise ManuscriptAssetExportError(f"Asset README contains missing links: {missing_links}")
    mapping = pd.read_csv(assets / "manifests/figure_number_mapping.csv")
    if mapping["manuscript_figure_number"].tolist() != list(range(1, 8)) or mapping["canonical_figure_number"].tolist() != [1, 3, 2, 4, 5, 6, 7]:
        raise ManuscriptAssetExportError("Figure numbering reconciliation failed.")
    return {"status": "passed", "file_count_including_manifest": len(actual) + 1, "total_bytes": total, "main_png": 7, "main_svg": 7, "supplementary_png": len(pngs) - 7, "supplementary_svg": len(svgs) - 7, "main_tables": 8, "manifest_sha256": _sha256(manifest_path)}


def validate_export_source_parity(
    project_root: Path,
    canonical_root: Path,
    asset_root: Path,
) -> dict[str, Any]:
    """Verify every declared source hash while the ignored canonical package is local."""

    root = project_root.resolve()
    validate_canonical_source(root, canonical_root.resolve())
    manifest_path = asset_root.resolve() / "manifests/manuscript_asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    for record in manifest.get("files", []):
        sources = record.get("source_paths", [])
        hashes = record.get("source_sha256_values", [])
        if len(sources) != len(hashes):
            raise ManuscriptAssetExportError(
                f"Source path/hash cardinality mismatch for {record.get('path')}."
            )
        if not sources:
            raise ManuscriptAssetExportError(
                f"Exported file has no declared canonical lineage: {record.get('path')}."
            )
        for relative, expected_hash in zip(sources, hashes, strict=True):
            source = root / Path(*PurePosixPath(relative).parts)
            if not source.is_file() or _sha256(source) != expected_hash:
                raise ManuscriptAssetExportError(
                    f"Export source parity failed for {record.get('path')}: {relative}"
                )
            checked += 1
    return {"status": "passed", "asset_files": len(manifest["files"]), "source_bindings_checked": checked}


def export_manuscript_assets(project_root: Path, canonical_root: Path, output_root: Path) -> dict[str, Any]:
    root = project_root.resolve(); source = canonical_root.resolve(); destination = output_root.resolve()
    if destination != (root / ASSET_RELATIVE_ROOT).resolve():
        raise ManuscriptAssetExportError("Output root must be manuscript/mdpi_information/assets.")
    if os.path.lexists(destination):
        raise FileExistsError(f"Immutable manuscript asset destination already exists: {destination}")
    source_validation = validate_canonical_source(root, source)
    temporary = tempfile.TemporaryDirectory(prefix=".manuscript-assets-staging-", dir=destination.parent)
    staging = Path(temporary.name) / destination.name
    staging.mkdir(parents=True)
    ctx = ExportContext(root, source, staging, datetime.now(UTC).replace(microsecond=0).isoformat())
    primary_error: BaseException | None = None
    try:
        _export_main_figures(ctx)
        _render_supplementary_figures(ctx)
        _export_main_tables(ctx)
        _export_supplementary_tables(ctx)
        _copy_canonical_source_tables(ctx)
        _handoff_files(ctx)
        _asset_readme(ctx)
        _write_text(ctx.output_root / "tables/previews/README.md", "# Table previews\n\nMarkdown previews are colocated with their main or supplementary CSV files.")
        ctx.bind(ctx.output_root / "tables/previews/README.md", [root / CANONICAL_RECEIPT])
        validation_path = staging / "manifests/export_validation.json"
        _write_json(
            validation_path,
            {
                "schema_version": 1,
                "status": "passed_pre_manifest",
                "checks": [
                    "canonical_closed_world_source_hashes",
                    "figure_dimensions_and_svg_safety",
                    "copy_hash_parity",
                    "table_source_rows_and_display_rounding",
                    "figure_and_table_numbering",
                    "claim_boundaries",
                ],
                "run_id": RUN_ID,
                "generation_commit": GENERATION_COMMIT,
                "config_hash": CONFIG_HASH,
                "source_tree_hash": SOURCE_TREE_HASH,
            },
        )
        ctx.bind(validation_path, [root / CANONICAL_RECEIPT])
        manifest = _export_manifest(ctx, source_validation)
        validation = validate_manuscript_asset_package(
            root,
            staging,
            require_published_path=False,
        )
        atomic_replace_directory(staging, destination)
        temporary.cleanup()
        return {**validation, "asset_root": _safe_relative(destination, root), "manifest": _safe_relative(destination / "manifests/manuscript_asset_manifest.json", root)}
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if os.path.exists(temporary.name):
            try: temporary.cleanup()
            except Exception as cleanup_error:
                if primary_error is None: raise
                primary_error.add_note(f"Temporary export cleanup also failed: {cleanup_error!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export or validate compact canonical manuscript assets.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=CANONICAL_RELATIVE_ROOT)
    parser.add_argument("--output-root", type=Path, default=ASSET_RELATIVE_ROOT)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args(); root = args.project_root.resolve()
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    if args.validate_only:
        result = validate_manuscript_asset_package(root, output)
    else:
        source = args.source_root if args.source_root.is_absolute() else root / args.source_root
        result = export_manuscript_assets(root, source, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
