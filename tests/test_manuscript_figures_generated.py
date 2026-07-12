from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.governance.manuscript_figures import (
    ManuscriptFigureError,
    generate_architecture_figures,
    validate_all_seven_figures,
)


class ManuscriptFiguresGeneratedTests(unittest.TestCase):
    def test_figures_1_to_4_generate_png_svg_and_source_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            readiness = pd.DataFrame(
                [
                    {
                        "component": "Test component",
                        "evidence_state": "present",
                        "interpretation_status": "research_only",
                        "limitation": "Synthetic status used only to test rendering.",
                    }
                ]
            )
            outputs = generate_architecture_figures(
                "configs/manuscript_final.yaml",
                output_dir=root / "figures",
                run_dir=root,
                run_id="test-run",
                readiness=readiness,
            )
            for number in range(1, 5):
                for suffix in ("png", "svg"):
                    path = outputs[f"figure_{number}_{suffix}"]
                    self.assertTrue(path.is_file(), path)
                    self.assertGreater(path.stat().st_size, 0, path)
            source_files = list((root / "figures" / "source_data").glob("*.csv"))
            self.assertGreaterEqual(len(source_files), 6)

    def test_validator_rejects_incomplete_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ManuscriptFigureError):
                validate_all_seven_figures(directory)

    def test_latest_canonical_package_has_all_figures_when_present(self) -> None:
        latest = Path("reports/manuscript_final/latest/figures")
        if not latest.exists():
            self.skipTest("Canonical end-to-end run has not been generated yet.")
        result = validate_all_seven_figures(latest)
        self.assertEqual(result["figure_count"], 7)


if __name__ == "__main__":
    unittest.main()
