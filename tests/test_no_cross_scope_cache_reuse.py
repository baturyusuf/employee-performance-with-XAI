from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.experiments.build_manuscript_evidence import (
    StageContext,
    _stage_cache_valid,
    _write_stage_metadata,
)


def _context(tmp_path: Path, evidence_scope: str) -> tuple[StageContext, Path]:
    run_dir = tmp_path / "reports" / "scope-cache"
    output = run_dir / "dataset_cards" / "dataset_cards.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"status": "test-contract-only"}\n', encoding="utf-8")
    config_path = tmp_path / "configs" / "manuscript_final.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}\n", encoding="utf-8")
    manifest: dict[str, Any] = {
        "evidence_scope": evidence_scope,
        "git_commit": "a" * 40,
        "source_tree_hash": "b" * 64,
        "config_hash": "c" * 64,
        "dataset_hashes": {"fixture": {"path": "data/fixture.csv", "sha256": "d" * 64}},
        "actual_input_receipts": {
            "fixture": {
                "actual_path": "data/fixture.csv",
                "actual_sha256": "d" * 64,
                "row_count": 1,
                "column_count": 2,
            }
        },
        "side_input_hashes": {
            "fixture_schema": {
                "path": "configs/fixture_schema.json",
                "sha256": "e" * 64,
                "size_bytes": 10,
            }
        },
        "scientific_input_hash": "f" * 64,
    }
    return (
        StageContext(
            config_path=config_path,
            config={},
            settings={},
            run_dir=run_dir,
            run_id="scope-cache-test",
            config_hash="c" * 64,
            manifest=manifest,
            evidence_scope=evidence_scope,
        ),
        output,
    )


def test_stage_contract_records_scope_and_same_scope_cache_is_eligible(tmp_path: Path) -> None:
    context, output = _context(tmp_path, "core")
    metadata = _write_stage_metadata(
        context,
        "dataset_cards",
        [output],
        started_at="2026-07-13T00:00:00+00:00",
        elapsed_seconds=0.0,
    )

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["evidence_scope"] == "core"
    assert _stage_cache_valid(context, "dataset_cards")


def test_core_cache_cannot_be_reused_by_supplementary_scope(tmp_path: Path) -> None:
    core, output = _context(tmp_path, "core")
    _write_stage_metadata(
        core,
        "dataset_cards",
        [output],
        started_at="2026-07-13T00:00:00+00:00",
        elapsed_seconds=0.0,
    )
    supplementary_manifest = dict(core.manifest)
    supplementary_manifest["evidence_scope"] = "supplementary"
    supplementary = replace(
        core,
        evidence_scope="supplementary",
        manifest=supplementary_manifest,
    )

    assert not _stage_cache_valid(supplementary, "dataset_cards")
