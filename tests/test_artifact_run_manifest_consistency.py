from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.governance.manuscript_contract import (
    RunManifestError,
    canonical_config_hash,
    create_run_manifest,
    finalize_run_manifest,
    load_manuscript_config,
    register_artifact,
    validate_run_manifest,
    write_run_manifest,
)


class ArtifactRunManifestConsistencyTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, dict]:
        dataset = root / "data" / "sample.csv"
        dataset.parent.mkdir(parents=True)
        dataset.write_text("feature,target\n1,2\n", encoding="utf-8")

        config = copy.deepcopy(load_manuscript_config())
        for definition in config["manuscript_final"]["datasets"].values():
            definition["path"] = "data/sample.csv"
        acquisition_path = root / "configs" / "data_acquisition.yaml"
        acquisition_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
        dataset_names = list(config["manuscript_final"]["datasets"])
        acquisition_path.write_text(
            json.dumps(
                {
                    "data_acquisition": {
                        "schema_version": 1,
                        "physical_datasets": {
                            "fixture": {
                                "local_path": "data/sample.csv",
                                "expected_sha256": dataset_hash,
                                "expected_rows": 1,
                                "expected_column_count": 2,
                                "expected_columns": ["feature", "target"],
                                "format": "csv",
                                "delimiter": ",",
                                "encoding": "utf-8",
                                "target_profiles": {
                                    "main": {
                                        "raw_target": "target",
                                        "expected_distribution": {"2": 1},
                                    }
                                },
                                "automatic_download_allowed": False,
                            }
                        },
                        "logical_bindings": {
                            name: {
                                "physical_dataset": "fixture",
                                "target_profile": "main",
                            }
                            for name in dataset_names
                        },
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        provenance = config["manuscript_final"]["provenance"]
        provenance["data_acquisition_manifest"] = "configs/data_acquisition.yaml"
        provenance["scientific_side_inputs"] = {
            "data_acquisition_contract": "configs/data_acquisition.yaml"
        }
        config_path = root / "configs" / "manuscript_final.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        artifact = root / "reports" / "manuscript_final" / "test_run" / "table.csv"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("metric,value\nmacro_f1,0.5\n", encoding="utf-8")

        manifest = create_run_manifest(
            config_path,
            project_root=root,
            run_id="manuscript_final_test_contract",
            initial_command="python -m src.experiments.build_manuscript_evidence",
        )
        register_artifact(
            manifest,
            artifact,
            project_root=root,
            stage="policy_ablation",
            artifact_type="table_csv",
        )
        finalize_run_manifest(manifest, status="complete")
        return config_path, artifact, manifest

    def test_complete_manifest_validates_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path, _, manifest = self._fixture(root)
            self.assertEqual(manifest["config_hash"], canonical_config_hash(config_path))
            validated = validate_run_manifest(
                manifest,
                project_root=root,
                expected_config_hash=canonical_config_hash(config_path),
                require_complete=True,
            )
            self.assertEqual(validated["run_id"], "manuscript_final_test_contract")
            self.assertTrue(validated["dataset_hashes"])
            self.assertEqual(len(validated["output_files"]), 1)

            manifest_path = root / "reports" / "manuscript_final" / "test_run" / "run_manifest.json"
            write_run_manifest(
                manifest,
                manifest_path,
                project_root=root,
                require_complete=True,
            )
            from_disk = validate_run_manifest(
                manifest_path,
                project_root=root,
                require_complete=True,
            )
            self.assertEqual(from_disk["config_hash"], manifest["config_hash"])

    def test_artifact_with_different_config_hash_is_rejected_at_registration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, artifact, manifest = self._fixture(root)
            second_artifact = artifact.with_name("second.csv")
            second_artifact.write_text("metric,value\nmacro_f1,0.6\n", encoding="utf-8")
            with self.assertRaisesRegex(RunManifestError, "config_hash"):
                register_artifact(
                    manifest,
                    second_artifact,
                    project_root=root,
                    stage="calibration",
                    artifact_type="table_csv",
                    artifact_config_hash="0" * 64,
                )

    def test_manifest_rejects_missing_referenced_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, artifact, manifest = self._fixture(root)
            artifact.unlink()
            with self.assertRaisesRegex(RunManifestError, "artifact is missing"):
                validate_run_manifest(manifest, project_root=root, require_complete=True)

    def test_manifest_rejects_artifact_hash_or_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, artifact, manifest = self._fixture(root)
            artifact.write_text("metric,value\nmacro_f1,0.9\n", encoding="utf-8")
            manifest["output_files"][0]["config_hash"] = "f" * 64
            with self.assertRaises(RunManifestError) as raised:
                validate_run_manifest(manifest, project_root=root, require_complete=True)
            message = str(raised.exception)
            self.assertIn("config_hash does not match", message)
            self.assertIn("artifact hash mismatch", message)

    def test_manifest_rejects_config_changed_after_run_started(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path, _, manifest = self._fixture(root)
            changed = json.loads(config_path.read_text(encoding="utf-8"))
            changed["manuscript_final"]["seeds"]["master"] = 999
            config_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(RunManifestError, "config hash mismatch"):
                validate_run_manifest(manifest, project_root=root, require_complete=True)


if __name__ == "__main__":
    unittest.main()
