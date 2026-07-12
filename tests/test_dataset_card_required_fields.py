from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.governance.dataset_cards import (
    DatasetCardValidationError,
    REQUIRED_CARD_FIELDS,
    build_dataset_cards,
    run,
    validate_dataset_card_record,
    validate_dataset_cards,
)


class DatasetCardRequiredFieldsTests(unittest.TestCase):
    def test_canonical_cards_cover_every_logical_dataset_role(self) -> None:
        cards = build_dataset_cards(run_id="card_test", config_hash="b" * 64)
        self.assertEqual(
            {card["dataset_id"] for card in cards},
            {
                "inx_primary",
                "hrdataset_v14",
                "ibm_hr_analytics",
                "ibm_hr_analytics_attrition",
                "employee_turnover",
            },
        )
        validate_dataset_cards(cards)
        for card in cards:
            self.assertTrue(set(REQUIRED_CARD_FIELDS).issubset(card))
            self.assertFalse(card["unmapped_observed_target_values"])
            self.assertEqual(
                sum(row["raw_count"] for row in card["target_mapping_support"]),
                card["row_count"],
            )
            self.assertEqual(card["licence_verification_status"], "manual_review_required")
            self.assertTrue(card["unresolved_manual_verification_items"])

    def test_blank_required_field_fails_validation(self) -> None:
        card = build_dataset_cards(run_id="card_test", config_hash="b" * 64)[0]
        broken = copy.deepcopy(card)
        broken["retrieval_url"] = ""
        errors = validate_dataset_card_record(broken)
        self.assertTrue(any("retrieval_url" in error for error in errors))
        with self.assertRaises(DatasetCardValidationError):
            validate_dataset_cards([broken])

    def test_generator_writes_machine_readable_cards_with_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs = run(
                output_dir=Path(temporary_directory),
                run_id="card_test",
                config_hash="b" * 64,
            )
            self.assertEqual(outputs["dataset_cards_json"].name, "dataset_cards.json")
            payload = json.loads(outputs["dataset_cards_json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], "card_test")
            self.assertEqual(payload["config_hash"], "b" * 64)
            self.assertEqual(len(payload["cards"]), 5)
            report = json.loads(outputs["validation_report"].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertFalse(report["manual_source_or_licence_authenticity_decisions_made"])


if __name__ == "__main__":
    unittest.main()
