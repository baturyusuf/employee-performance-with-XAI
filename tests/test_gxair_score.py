from __future__ import annotations

from pathlib import Path
import unittest

from src.governance.gxair_score import build_component_rows, compute_component_scores, final_readiness_label


class GXAIRScoreTests(unittest.TestCase):
    def test_component_scores_are_exposed(self) -> None:
        scores = compute_component_scores()
        self.assertIn("LLM Governance Compliance Score", set(scores["component"]))
        self.assertTrue(((scores["score"] >= 0) & (scores["score"] <= 1)).all())

    def test_missing_evidence_component_is_not_scored(self) -> None:
        rows = build_component_rows(
            {
                "final_candidate_dashboard": Path("missing_dashboard.csv"),
                "llm_agent_eval_summary": Path("missing_llm.csv"),
                "chatbot_guardrail_eval": Path("missing_guardrail.csv"),
                "external_validation_summary": Path("missing_external.md"),
            },
            "no_salary_hike_no_attrition_no_department",
        )
        self.assertTrue(any(row["status"] == "evidence_missing" for row in rows))
        import pandas as pd

        self.assertEqual(final_readiness_label(pd.DataFrame(rows)), "evidence_missing")


if __name__ == "__main__":
    unittest.main()
