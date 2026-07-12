from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.experiments.build_manuscript_evidence import (
    StageContext,
    _stage_cache_valid,
    _write_stage_metadata,
)


def _context(tmp_path: Path) -> tuple[StageContext, Path]:
    run_dir = tmp_path / "reports" / "run"
    output = run_dir / "policy" / "fold_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("fold,macro_f1\n0,0.5\n", encoding="utf-8")
    config_path = tmp_path / "configs" / "manuscript_final.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "git_commit": "a" * 40,
        "source_tree_hash": "b" * 64,
        "config_hash": "c" * 64,
        "dataset_hashes": {
            "inx_primary": {"path": "data/raw/sample.csv", "sha256": "d" * 64}
        },
        "actual_input_receipts": {
            "inx_primary": {
                "actual_path": "data/raw/sample.csv",
                "actual_sha256": "d" * 64,
                "row_count": 3,
                "column_count": 2,
            }
        },
        "side_input_hashes": {
            "feature_taxonomy": {
                "path": "configs/feature_taxonomy.yaml",
                "sha256": "e" * 64,
                "size_bytes": 123,
            }
        },
        "scientific_input_hash": "f" * 64,
    }
    context = StageContext(
        config_path=config_path,
        config={},
        settings={},
        run_dir=run_dir,
        run_id="unit_1b_cache",
        config_hash="c" * 64,
        manifest=manifest,
    )
    return context, output


def _write_valid_contract(tmp_path: Path) -> StageContext:
    context, output = _context(tmp_path)
    _write_stage_metadata(
        context,
        "policy",
        [output],
        started_at="2026-07-13T00:00:00+00:00",
        elapsed_seconds=1.0,
    )
    assert _stage_cache_valid(context, "policy")
    return context


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("git_commit", "1" * 40),
        ("source_tree_hash", "2" * 64),
        ("dataset_hashes", {"inx_primary": {"path": "data/raw/sample.csv", "sha256": "3" * 64}}),
        (
            "actual_input_receipts",
            {
                "inx_primary": {
                    "actual_path": "data/raw/sample.csv",
                    "actual_sha256": "4" * 64,
                    "row_count": 3,
                    "column_count": 2,
                }
            },
        ),
        (
            "side_input_hashes",
            {
                "feature_taxonomy": {
                    "path": "configs/feature_taxonomy.yaml",
                    "sha256": "5" * 64,
                    "size_bytes": 123,
                }
            },
        ),
        ("scientific_input_hash", "6" * 64),
    ],
)
def test_cache_reuse_rejects_every_changed_scientific_identity_component(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    context = _write_valid_contract(tmp_path)
    changed_manifest = copy.deepcopy(context.manifest)
    changed_manifest[field] = replacement
    changed = replace(context, manifest=changed_manifest)

    assert not _stage_cache_valid(changed, "policy")


def test_cache_reuse_rejects_changed_config_identity(tmp_path: Path) -> None:
    context = _write_valid_contract(tmp_path)
    changed_manifest = copy.deepcopy(context.manifest)
    changed_manifest["config_hash"] = "7" * 64
    changed = replace(context, config_hash="7" * 64, manifest=changed_manifest)

    assert not _stage_cache_valid(changed, "policy")


def test_stage_contract_persists_complete_scientific_identity(tmp_path: Path) -> None:
    context = _write_valid_contract(tmp_path)
    contract_path = context.run_dir / "policy" / "stage_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    for field in (
        "git_commit",
        "source_tree_hash",
        "config_hash",
        "dataset_hashes",
        "actual_input_receipts",
        "side_input_hashes",
        "scientific_input_hash",
    ):
        assert contract[field] == context.manifest[field]


def test_cache_reuse_rejects_contract_missing_scientific_identity(tmp_path: Path) -> None:
    context = _write_valid_contract(tmp_path)
    contract_path = context.run_dir / "policy" / "stage_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.pop("side_input_hashes")
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    assert not _stage_cache_valid(context, "policy")
