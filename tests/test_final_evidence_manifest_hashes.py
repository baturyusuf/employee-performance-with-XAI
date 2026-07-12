from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.build_manuscript_evidence import (
    ManuscriptBuildError,
    _update_latest_pointer,
    build_final_evidence_manifest,
    validate_final_evidence_manifest,
)


class FinalEvidenceManifestHashTests(unittest.TestCase):
    def test_generated_manifest_verifies_every_referenced_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "policy").mkdir()
            (root / "figures").mkdir()
            (root / "policy" / "fold_metrics.csv").write_text(
                "run_id,config_hash,fold\ncanonical-run,abc,1\n", encoding="utf-8"
            )
            (root / "figures" / "figure.svg").write_text(
                "<svg><!-- canonical-run --></svg>\n", encoding="utf-8"
            )
            outputs = build_final_evidence_manifest(
                root,
                run_id="canonical-run",
                config_hash="a" * 64,
            )
            result = validate_final_evidence_manifest(
                outputs["json"],
                run_dir=root,
                expected_run_id="canonical-run",
                expected_config_hash="a" * 64,
            )
            self.assertEqual(result["n_files"], 2)
            payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
            self.assertEqual({row["path"] for row in payload["files"]}, {
                "figures/figure.svg",
                "policy/fold_metrics.csv",
            })

    def test_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "stage" / "table.csv"
            evidence.parent.mkdir()
            evidence.write_text("metric,value\naccuracy,0.5\n", encoding="utf-8")
            outputs = build_final_evidence_manifest(
                root,
                run_id="canonical-run",
                config_hash="b" * 64,
            )
            evidence.write_text("metric,value\naccuracy,0.9\n", encoding="utf-8")
            with self.assertRaisesRegex(ManuscriptBuildError, "hash mismatch"):
                validate_final_evidence_manifest(
                    outputs["json"],
                    run_dir=root,
                    expected_run_id="canonical-run",
                    expected_config_hash="b" * 64,
                )

    def test_latest_canonical_manifest_verifies_when_present(self) -> None:
        latest = Path("reports/manuscript_final/latest")
        manifest = latest / "final_evidence_manifest.json"
        if not manifest.is_file():
            self.skipTest("Canonical end-to-end run has not been generated yet.")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        result = validate_final_evidence_manifest(
            manifest,
            run_dir=latest,
            expected_run_id=str(payload["run_id"]),
            expected_config_hash=str(payload["config_hash"]),
        )
        self.assertEqual(result["n_files"], payload["n_files"])

    def test_latest_pointer_exposes_the_complete_versioned_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "canonical-run"
            (run / "nested").mkdir(parents=True)
            (run / "nested" / "artifact.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            latest = _update_latest_pointer(run, root)
            self.assertTrue((latest / "nested" / "artifact.csv").is_file())


if __name__ == "__main__":
    unittest.main()
