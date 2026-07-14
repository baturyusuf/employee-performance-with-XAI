from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest

from src.governance.core_figure_package import (
    CoreFigurePackageError,
    validate_core_figure_package,
)
from src.governance.manuscript_contract import canonical_config_hash, sha256_file
from src.utils.config_loader import load_config


IDENTITY = {
    "run_id": "core-figure-test-run",
    "config_hash": canonical_config_hash(load_config("configs/manuscript_final.yaml")),
    "scientific_input_hash": "b" * 64,
    "source_tree_hash": "c" * 64,
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, *, feature: str = "Signal") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "run_id,config_hash,scientific_input_hash,source_tree_hash,feature,value\n"
        f"{IDENTITY['run_id']},{IDENTITY['config_hash']},{IDENTITY['scientific_input_hash']},"
        f"{IDENTITY['source_tree_hash']},{feature},1\n",
        encoding="utf-8",
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_png(path: Path, width: int = 640, height: int = 480) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(pixels, level=9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _record(path: Path, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_stage_contract(run: Path, stage: str) -> None:
    root = run / stage
    outputs = [
        _record(path, path.relative_to(root).as_posix())
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "stage_contract.json"
    ]
    _write_json(
        root / "stage_contract.json",
        {
            "stage": stage,
            "status": "complete",
            "inventory_mode": "closed_world",
            "path_basis": "stage_relative",
            **IDENTITY,
            "outputs": outputs,
        },
    )


def _write_upstream_source(path: Path) -> None:
    if path.suffix == ".csv":
        _write_csv(path)
    elif path.suffix == ".json":
        _write_json(path, {**IDENTITY, "feature": "Signal", "value": 1})
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")


def _build_package(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    config = load_config("configs/manuscript_final.yaml")
    plan = config["manuscript_final"]["figures"]
    run = tmp_path / "run"
    run.mkdir()

    run_inputs = run / "run_inputs"
    snapshot = run_inputs / "canonical_config_snapshot.yaml"
    snapshot.parent.mkdir()
    snapshot.write_text("manuscript_final: {}\n", encoding="utf-8")
    snapshot_row = {
        "logical_name": "canonical_config",
        "input_kind": "canonical_config",
        "source_path": "configs/manuscript_final.yaml",
        "snapshot_path": "canonical_config_snapshot.yaml",
        "source_sha256": sha256_file(snapshot),
        "source_size_bytes": snapshot.stat().st_size,
        "snapshot_sha256": sha256_file(snapshot),
        "snapshot_size_bytes": snapshot.stat().st_size,
    }
    _write_json(
        run_inputs / "input_contract.json",
        {
            "schema_version": 1,
            "contract_kind": "manuscript_run_inputs",
            "status": "complete",
            "inventory_mode": "closed_world",
            "path_basis": "run_inputs_relative",
            **IDENTITY,
            "n_snapshots": 1,
            "snapshots": [snapshot_row],
        },
    )

    stage_paths: dict[str, list[Path]] = {}
    for definition in plan["definitions"].values():
        for source in definition["sources"]:
            if source["stage"] == "run_inputs":
                continue
            path = run / source["path"]
            if not path.exists():
                _write_upstream_source(path)
            stage_paths.setdefault(source["stage"], []).append(path)
    for stage in stage_paths:
        _write_stage_contract(run, stage)

    figures = run / "core_figures"
    figures.mkdir()
    manifest_rows: list[dict[str, object]] = []
    for key, definition in plan["definitions"].items():
        stem = definition["output_stem"]
        png = figures / f"{stem}.png"
        svg = figures / f"{stem}.svg"
        source_data = figures / plan["source_data_subdirectory"] / definition["source_data_filename"]
        caption = figures / plan["caption_subdirectory"] / definition["caption_filename"]
        _write_png(png)
        svg.parent.mkdir(parents=True, exist_ok=True)
        svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480">'
            '<rect width="640" height="480" fill="white"/></svg>\n',
            encoding="utf-8",
        )
        _write_csv(source_data)
        caption.parent.mkdir(parents=True, exist_ok=True)
        caption.write_text(
            "Fixture caption. " + "; ".join(f"{field}={value}" for field, value in IDENTITY.items()) + "\n",
            encoding="utf-8",
        )
        source_rows = []
        for source in definition["sources"]:
            path = run / source["path"]
            source_rows.append(
                {
                    "stage": source["stage"],
                    "path": source["path"],
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        manifest_rows.append(
            {
                "figure_key": key,
                "number": definition["number"],
                "figure_id": definition["figure_id"],
                "output_stem": stem,
                "png_path": png.relative_to(figures).as_posix(),
                "png_sha256": sha256_file(png),
                "png_size_bytes": png.stat().st_size,
                "png_width_px": 640,
                "png_height_px": 480,
                "svg_path": svg.relative_to(figures).as_posix(),
                "svg_sha256": sha256_file(svg),
                "svg_size_bytes": svg.stat().st_size,
                "svg_width_px": 640.0,
                "svg_height_px": 480.0,
                "source_data_path": source_data.relative_to(figures).as_posix(),
                "source_data_sha256": sha256_file(source_data),
                "source_data_size_bytes": source_data.stat().st_size,
                "caption_path": caption.relative_to(figures).as_posix(),
                "caption_sha256": sha256_file(caption),
                "caption_size_bytes": caption.stat().st_size,
                "sources": source_rows,
            }
        )
    manifest = {
        "schema_version": 1,
        "manifest_kind": "core_figure_package",
        "status": "complete",
        "inventory_mode": "closed_world",
        "path_basis": "core_figures_relative",
        "plan_version": plan["plan_version"],
        "stage": "core_figures",
        "scope": "core",
        "hash_algorithm": "sha256",
        "n_figures": 7,
        **IDENTITY,
        "figures": manifest_rows,
    }
    _write_json(figures / "figure_manifest.json", manifest)
    _write_stage_contract(run, "core_figures")
    return run, figures, config, manifest


def _validate(run: Path, figures: Path, config: dict) -> dict:
    return validate_core_figure_package(
        figures,
        run_root=run,
        config=config,
        **IDENTITY,
    )


def test_current_run_source_bound_seven_figure_package_passes(tmp_path: Path) -> None:
    run, figures, config, _ = _build_package(tmp_path)
    result = _validate(run, figures, config)
    assert result["figure_count"] == 7
    assert result["format_count"] == 14
    assert result["source_count"] == 24
    assert result["artifact_count_excluding_stage_contract"] == 29
    assert result["closed_world"] is True


def test_manual_figure_copy_without_manifest_and_stage_receipt_is_rejected(tmp_path: Path) -> None:
    run, figures, config, _ = _build_package(tmp_path)
    (figures / "figure_manifest.json").unlink()
    (figures / "stage_contract.json").unlink()
    with pytest.raises(CoreFigurePackageError, match="stage contract"):
        _validate(run, figures, config)


def test_missing_png_svg_pair_is_rejected(tmp_path: Path) -> None:
    run, figures, config, manifest = _build_package(tmp_path)
    (figures / manifest["figures"][0]["svg_path"]).unlink()
    with pytest.raises(CoreFigurePackageError, match="missing"):
        _validate(run, figures, config)


def test_wrong_run_upstream_stage_contract_is_rejected(tmp_path: Path) -> None:
    run, figures, config, _ = _build_package(tmp_path)
    receipt_path = run / "oof_shap" / "stage_contract.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["run_id"] = "wrong-run"
    _write_json(receipt_path, receipt)
    with pytest.raises(CoreFigurePackageError, match="identity mismatch"):
        _validate(run, figures, config)


def test_obsolete_numbered_preview_is_rejected_as_closed_world_extra(tmp_path: Path) -> None:
    run, figures, config, _ = _build_package(tmp_path)
    _write_png(figures / "figure_7_local_reason_code.png")
    _write_stage_contract(run, "core_figures")
    with pytest.raises(CoreFigurePackageError, match="obsolete"):
        _validate(run, figures, config)


def test_source_hash_mutation_is_rejected(tmp_path: Path) -> None:
    run, figures, config, _ = _build_package(tmp_path)
    source = run / "policy_ablation" / "figure_leakage_policy_tradeoff_source.csv"
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CoreFigurePackageError, match="(hash|size) mismatch"):
        _validate(run, figures, config)


def test_forbidden_primary_feature_in_figure_source_data_is_rejected(tmp_path: Path) -> None:
    run, figures, config, manifest = _build_package(tmp_path)
    row = manifest["figures"][6]
    source = figures / row["source_data_path"]
    _write_csv(source, feature="Salary")
    row["source_data_sha256"] = sha256_file(source)
    row["source_data_size_bytes"] = source.stat().st_size
    _write_json(figures / "figure_manifest.json", manifest)
    _write_stage_contract(run, "core_figures")
    with pytest.raises(CoreFigurePackageError, match="Forbidden.*feature"):
        _validate(run, figures, config)
