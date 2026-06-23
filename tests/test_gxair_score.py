from __future__ import annotations

import unittest

from src.governance.gxair_score import compute_component_scores


class GXAIRScoreTests(unittest.TestCase):
    def test_component_scores_are_exposed(self) -> None:
        scores = compute_component_scores()
        self.assertIn("LLM Governance Compliance Score", set(scores["component"]))
        self.assertTrue(((scores["score"] >= 0) & (scores["score"] <= 1)).all())


if __name__ == "__main__":
    unittest.main()

