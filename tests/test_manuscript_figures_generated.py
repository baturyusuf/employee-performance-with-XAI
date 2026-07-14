from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.governance.manuscript_figures import (
    ManuscriptFigureError,
    generate_architecture_figures,
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

if __name__ == "__main__":
    unittest.main()
