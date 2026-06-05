from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.audit import run_data_audit


class DataAuditTests(unittest.TestCase):
    def test_run_data_audit_writes_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = run_data_audit(output_dir=Path(tmp), write_registry=False)

            for path in outputs.values():
                self.assertTrue(path.exists(), f"Missing expected audit output: {path}")

            self.assertIn("feature_temporality_audit", outputs)


if __name__ == "__main__":
    unittest.main()
