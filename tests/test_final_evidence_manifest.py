from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.governance.final_evidence_manifest import (
    STUB_DRY_RUN_DISCLAIMER,
    file_sha256,
    readiness_blockers,
    row_count,
)


class FinalEvidenceManifestTests(unittest.TestCase):
    def test_row_count_and_hash_for_csv_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "evidence.csv"
            jsonl_path = root / "evidence.jsonl"
            csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
            jsonl_path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")

            self.assertEqual(row_count(csv_path, "csv"), "2")
            self.assertEqual(row_count(jsonl_path, "jsonl"), "2")
            self.assertNotEqual(file_sha256(csv_path), "missing")

    def test_readiness_blockers_include_proxy_and_counterfactual(self) -> None:
        dashboard = pd.DataFrame(
            [
                {
                    "component": "Proxy Risk Penalty",
                    "status": "fail",
                    "severity": "high",
                    "limitations": "Low score means high proxy risk.",
                },
                {
                    "component": "Counterfactual Actionability",
                    "status": "fail",
                    "severity": "high",
                    "limitations": "Counterfactuals are model scenarios.",
                },
            ]
        )
        blockers = readiness_blockers(dashboard)
        self.assertIn("Proxy Risk Penalty", blockers[0])
        self.assertIn("Counterfactual Actionability", blockers[1])

    def test_stub_dry_run_disclaimer_is_explicit(self) -> None:
        self.assertIn("not be cited as real LLM evidence", STUB_DRY_RUN_DISCLAIMER)
        self.assertIn("offline_stub_llm", STUB_DRY_RUN_DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
