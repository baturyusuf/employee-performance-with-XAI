from __future__ import annotations

import unittest

from src.experiments.proxy_analysis import cramers_v


class ProxyAnalysisTests(unittest.TestCase):
    def test_cramers_v_detects_perfect_categorical_association(self) -> None:
        feature = ["a", "a", "b", "b"]
        target = ["x", "x", "y", "y"]
        self.assertAlmostEqual(cramers_v(feature, target), 1.0)

    def test_cramers_v_returns_zero_for_constant_feature(self) -> None:
        feature = ["a", "a", "a", "a"]
        target = ["x", "x", "y", "y"]
        self.assertEqual(cramers_v(feature, target), 0.0)


if __name__ == "__main__":
    unittest.main()
