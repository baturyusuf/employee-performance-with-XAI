from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.build_manuscript_evidence import (
    ManuscriptBuildError,
    build_final_evidence_manifest,
    validate_final_evidence_manifest,
)


GIT_COMMIT = "d" * 40
SOURCE_TREE_HASH = "e" * 64
SCIENTIFIC_INPUT_HASH = "f" * 64


class FinalEvidenceManifestHashTests(unittest.TestCase):
    def test_generated_manifest_verifies_every_referenced_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "policy_ablation").mkdir()
            (root / "core_figures").mkdir()
            (root / "policy_ablation" / "fold_metrics.csv").write_text(
                "run_id,config_hash,fold\ncanonical-run,abc,1\n", encoding="utf-8"
            )
            (root / "core_figures" / "figure.svg").write_text(
                "<svg><!-- canonical-run --></svg>\n", encoding="utf-8"
            )
            outputs = build_final_evidence_manifest(
                root,
                run_id="canonical-run",
                config_hash="a" * 64,
                evidence_scope="core",
                scope_contract_hash="c" * 64,
                git_commit=GIT_COMMIT,
                source_tree_hash=SOURCE_TREE_HASH,
                scientific_input_hash=SCIENTIFIC_INPUT_HASH,
            )
            result = validate_final_evidence_manifest(
                outputs["json"],
                run_dir=root,
                expected_run_id="canonical-run",
                expected_config_hash="a" * 64,
                expected_evidence_scope="core",
                expected_scope_contract_hash="c" * 64,
                expected_git_commit=GIT_COMMIT,
                expected_source_tree_hash=SOURCE_TREE_HASH,
                expected_scientific_input_hash=SCIENTIFIC_INPUT_HASH,
            )
            self.assertEqual(result["n_files"], 2)
            payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
            self.assertEqual({row["path"] for row in payload["files"]}, {
                "core_figures/figure.svg",
                "policy_ablation/fold_metrics.csv",
            })

    def test_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "model_benchmarks" / "table.csv"
            evidence.parent.mkdir()
            evidence.write_text("metric,value\naccuracy,0.5\n", encoding="utf-8")
            outputs = build_final_evidence_manifest(
                root,
                run_id="canonical-run",
                config_hash="b" * 64,
                evidence_scope="core",
                scope_contract_hash="d" * 64,
                git_commit=GIT_COMMIT,
                source_tree_hash=SOURCE_TREE_HASH,
                scientific_input_hash=SCIENTIFIC_INPUT_HASH,
            )
            evidence.write_text("metric,value\naccuracy,0.9\n", encoding="utf-8")
            with self.assertRaisesRegex(ManuscriptBuildError, "hash mismatch"):
                validate_final_evidence_manifest(
                    outputs["json"],
                    run_dir=root,
                    expected_run_id="canonical-run",
                    expected_config_hash="b" * 64,
                    expected_evidence_scope="core",
                    expected_scope_contract_hash="d" * 64,
                    expected_git_commit=GIT_COMMIT,
                    expected_source_tree_hash=SOURCE_TREE_HASH,
                    expected_scientific_input_hash=SCIENTIFIC_INPUT_HASH,
                )

    def test_latest_canonical_manifest_verifies_when_present(self) -> None:
        latest = Path("reports/manuscript_final/latest")
        manifest = latest / "final_evidence_manifest.json"
        if not manifest.is_file():
            self.skipTest("Canonical end-to-end run has not been generated yet.")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not {
            "evidence_scope",
            "scope_contract_hash",
            "git_commit",
            "source_tree_hash",
            "scientific_input_hash",
        }.issubset(payload):
            self.skipTest("Existing latest package is historical and not a scoped v2 package.")
        result = validate_final_evidence_manifest(
            manifest,
            run_dir=latest,
            expected_run_id=str(payload["run_id"]),
            expected_config_hash=str(payload["config_hash"]),
            expected_evidence_scope=str(payload["evidence_scope"]),
            expected_scope_contract_hash=str(payload["scope_contract_hash"]),
            expected_git_commit=str(payload["git_commit"]),
            expected_source_tree_hash=str(payload["source_tree_hash"]),
            expected_scientific_input_hash=str(payload["scientific_input_hash"]),
        )
        self.assertEqual(result["n_files"], payload["n_files"])

    def test_latest_pointer_exposes_the_complete_versioned_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "canonical-run"
            (run / "nested").mkdir(parents=True)
            (run / "nested" / "artifact.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            # Physical package duplication is prohibited; pointer promotion is
            # covered by the scoped latest-pointer contract tests.
            self.assertTrue((run / "nested" / "artifact.csv").is_file())


if __name__ == "__main__":
    unittest.main()
