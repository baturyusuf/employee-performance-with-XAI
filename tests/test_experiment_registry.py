from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.utils.experiment_registry import REGISTRY_COLUMNS, append_registry_row, read_registry


class ExperimentRegistryTests(unittest.TestCase):
    def test_append_and_read_registry_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment_registry.csv"
            append_registry_row(
                {
                    "run_id": "unit_test_run",
                    "date_time": "2026-06-05T00:00:00Z",
                    "script": "tests",
                    "feature_set": "no_salary_hike_no_attrition",
                    "model": "xgboost",
                    "decision_status": "candidate",
                },
                registry_path=path,
            )

            rows = read_registry(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "unit_test_run")
            self.assertEqual(list(rows[0].keys()), REGISTRY_COLUMNS)


if __name__ == "__main__":
    unittest.main()
