from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.experiments.build_manuscript_evidence import (
    ATOMIC_DIRECTORY_STAGE_RUNNERS,
    STAGE_ORPHAN_PREFIXES,
    STAGE_RUNNERS,
)
from src.experiments.manuscript_tables import (
    ManuscriptTableError,
    run,
    validate_table_package,
)
from src.governance.manuscript_contract import canonical_config_hash, manuscript_settings
from src.governance.table_contract import expected_table_plan, validate_table_plan_declaration
from src.models.task_schema import metric_schema_hash
from src.utils.config_loader import PROJECT_ROOT, load_config


CONFIG_PATH = PROJECT_ROOT / "configs" / "manuscript_final.yaml"
RUN_ID = "table-fixture-run"
SCIENTIFIC_INPUT_HASH = "a" * 64
SOURCE_TREE_HASH = "b" * 64


def _identity(config_hash: str) -> dict[str, str]:
    return {
        "run_id": RUN_ID,
        "config_hash": config_hash,
        "scientific_input_hash": SCIENTIFIC_INPUT_HASH,
        "source_tree_hash": SOURCE_TREE_HASH,
    }


def _build_sources(tmp_path: Path, scope: str) -> tuple[Path, str]:
    config = load_config(CONFIG_PATH)
    config_hash = canonical_config_hash(config)
    identity = _identity(config_hash)
    run_root = tmp_path / scope
    run_root.mkdir()
    sources = {
        str(source)
        for definition in expected_table_plan(scope).values()
        for source in definition["sources"]
    }
    for relative in sorted(sources):
        path = run_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "run_inputs/input_contract.json":
            path.write_text(json.dumps({**identity, "status": "complete"}) + "\n", encoding="utf-8")
        elif relative == "run_inputs/canonical_config_snapshot.yaml":
            shutil.copyfile(CONFIG_PATH, path)
        elif path.suffix == ".json":
            path.write_text(
                json.dumps({**identity, "status": "complete", "n_samples": 20}) + "\n",
                encoding="utf-8",
            )
        else:
            pd.DataFrame(
                [
                    {
                        **identity,
                        "dataset_key": "fixture_dataset",
                        "system_id": "fixture_model",
                        "metric": "macro_f1",
                        "point_estimate": 0.5,
                        "n_samples": 20,
                    }
                ]
            ).to_csv(path, index=False)
    return run_root, config_hash


@pytest.mark.parametrize("scope,stage", [("core", "core_tables"), ("supplementary", "supplementary_tables")])
def test_source_bound_table_package_is_atomic_and_closed_world(
    tmp_path: Path,
    scope: str,
    stage: str,
) -> None:
    run_root, config_hash = _build_sources(tmp_path, scope)
    outputs = run(
        CONFIG_PATH,
        run_root=run_root,
        output_dir=run_root / stage,
        scope=scope,
        run_id=RUN_ID,
        config_hash=config_hash,
        scientific_input_hash=SCIENTIFIC_INPUT_HASH,
        source_tree_hash=SOURCE_TREE_HASH,
    )

    result = validate_table_package(
        run_root / stage,
        run_root=run_root,
        config=load_config(CONFIG_PATH),
        scope=scope,
        run_id=RUN_ID,
        config_hash=config_hash,
        scientific_input_hash=SCIENTIFIC_INPUT_HASH,
        source_tree_hash=SOURCE_TREE_HASH,
    )

    assert result["table_count"] == len(expected_table_plan(scope))
    assert result["metric_schema_hash"] == metric_schema_hash()
    assert set(outputs) == {*expected_table_plan(scope), "manifest"}
    assert not list(run_root.glob(f"{stage}.__staging__.*"))
    for definition in expected_table_plan(scope).values():
        table = pd.read_csv(run_root / stage / definition["filename"], keep_default_na=False)
        assert set(table["run_id"]) == {RUN_ID}
        assert set(table["source_artifact"]).issubset(set(definition["sources"]))
        assert set(table["metric_schema_hash"]) == {metric_schema_hash()}


def test_wrong_source_identity_preserves_forensic_staging(tmp_path: Path) -> None:
    run_root, config_hash = _build_sources(tmp_path, "core")
    source = run_root / "model_benchmarks" / "model_summary.csv"
    frame = pd.read_csv(source)
    frame.loc[0, "run_id"] = "wrong-run"
    frame.to_csv(source, index=False)

    with pytest.raises(ManuscriptTableError, match="wrong run_id"):
        run(
            CONFIG_PATH,
            run_root=run_root,
            output_dir=run_root / "core_tables",
            scope="core",
            run_id=RUN_ID,
            config_hash=config_hash,
            scientific_input_hash=SCIENTIFIC_INPUT_HASH,
            source_tree_hash=SOURCE_TREE_HASH,
        )

    assert not (run_root / "core_tables").exists()
    assert len(list(run_root.glob("core_tables.__staging__.*"))) == 1


def test_validator_rejects_duplicated_source_receipt(tmp_path: Path) -> None:
    run_root, config_hash = _build_sources(tmp_path, "core")
    run(
        CONFIG_PATH,
        run_root=run_root,
        output_dir=run_root / "core_tables",
        scope="core",
        run_id=RUN_ID,
        config_hash=config_hash,
        scientific_input_hash=SCIENTIFIC_INPUT_HASH,
        source_tree_hash=SOURCE_TREE_HASH,
    )
    manifest_path = run_root / "core_tables" / "table_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["tables"] if len(item["sources"]) > 1)
    record["sources"][1] = dict(record["sources"][0])
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ManuscriptTableError, match="missing or duplicated"):
        validate_table_package(
            run_root / "core_tables",
            run_root=run_root,
            config=load_config(CONFIG_PATH),
            scope="core",
            run_id=RUN_ID,
            config_hash=config_hash,
            scientific_input_hash=SCIENTIFIC_INPUT_HASH,
            source_tree_hash=SOURCE_TREE_HASH,
        )


def test_table_plan_and_builder_registration_are_frozen() -> None:
    settings = manuscript_settings(load_config(CONFIG_PATH))
    validate_table_plan_declaration(settings["tables"])
    assert {"core_tables", "supplementary_tables"}.issubset(STAGE_RUNNERS)
    assert {"core_tables", "supplementary_tables"}.issubset(ATOMIC_DIRECTORY_STAGE_RUNNERS)
    assert STAGE_ORPHAN_PREFIXES["core_tables"] == ("core_tables.__staging__.",)
    assert STAGE_ORPHAN_PREFIXES["supplementary_tables"] == (
        "supplementary_tables.__staging__.",
    )
