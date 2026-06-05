from __future__ import annotations

import unittest

from src.models.evaluate import leakage_sensitivity_index


class LeakageSensitivityTests(unittest.TestCase):
    def test_lsi_and_absolute_drop(self) -> None:
        result = leakage_sensitivity_index(full_feature_score=0.90, leakage_safe_score=0.60)
        self.assertAlmostEqual(result["absolute_drop"], 0.30)
        self.assertAlmostEqual(result["lsi"], 1.0 / 3.0)

    def test_zero_full_score_returns_no_lsi(self) -> None:
        result = leakage_sensitivity_index(full_feature_score=0.0, leakage_safe_score=0.0)
        self.assertEqual(result["absolute_drop"], 0.0)
        self.assertIsNone(result["lsi"])


if __name__ == "__main__":
    unittest.main()
