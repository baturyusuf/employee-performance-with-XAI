from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.governance.manuscript_figures import (
    ManuscriptFigureError,
    generate_architecture_figures,
    validate_all_seven_figures,
)


class ManuscriptFiguresGeneratedTests(unittest.TestCase):
    def test_legacy_figures_fail_before_writing_under_the_core_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ManuscriptFigureError, "Legacy governance Figures"):
                generate_architecture_figures(
                    "configs/manuscript_final.yaml",
                    output_dir=root / "figures",
                    run_dir=root,
                    run_id="test-run",
                )
            self.assertFalse((root / "figures").exists())

    def test_validator_rejects_incomplete_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ManuscriptFigureError):
                validate_all_seven_figures(directory)

    def test_latest_canonical_package_has_all_figures_when_present(self) -> None:
        latest = Path("reports/manuscript_final/latest/core/core_figures")
        if not latest.exists():
            self.skipTest("Canonical end-to-end run has not been generated yet.")
        result = validate_all_seven_figures(latest)
        self.assertEqual(result["figure_count"], 7)


if __name__ == "__main__":
    unittest.main()
