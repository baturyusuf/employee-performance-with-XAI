from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.core.io_utils import object_hash, read_jsonl, write_jsonl
from src.core.reporting import markdown_table
from src.core.run_registry import RunRegistryEntry, append_run_entry, read_run_registry


class CoreReportingRegistryTests(unittest.TestCase):
    def test_hash_is_stable_for_dict_order(self) -> None:
        self.assertEqual(object_hash({"a": 1, "b": 2}), object_hash({"b": 2, "a": 1}))

    def test_jsonl_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            write_jsonl(path, [{"x": 1}, {"x": 2}])
            self.assertEqual(read_jsonl(path), [{"x": 1}, {"x": 2}])

    def test_markdown_table_escapes_pipe(self) -> None:
        table = markdown_table(pd.DataFrame([{"a": "x|y"}]))
        self.assertIn("x\\|y", "\n".join(table))

    def test_run_registry_writes_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_registry.csv"
            append_run_entry(RunRegistryEntry(run_id="r1", command="cmd", output_files=["a"]), path=path)
            rows = read_run_registry(path)
            self.assertEqual(rows[0]["run_id"], "r1")
            self.assertIn("output_files", rows[0])


if __name__ == "__main__":
    unittest.main()

