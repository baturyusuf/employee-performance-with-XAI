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


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_AVAILABLE = all(
    (ROOT / relative).is_file()
    for relative in (
        "data/raw/inx_employee_performance.csv",
        "data/external/hrdataset_v14/raw.csv",
        "data/external/ibm_hr_analytics/raw.csv",
        "data/external/employee_turnover/raw.csv",
    )
)


@unittest.skipUnless(LOCAL_DATA_AVAILABLE, "requires all ignored local canonical datasets")
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
            metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["manuscript_config"], "configs/manuscript_final.yaml")
            self.assertEqual(metadata["provenance_config"], "configs/dataset_provenance.yaml")
            self.assertEqual(
                metadata["markdown_cards"],
                [
                    "cards/inx_primary.md",
                    "cards/hrdataset_v14.md",
                    "cards/ibm_hr_analytics.md",
                    "cards/ibm_hr_analytics_attrition.md",
                    "cards/employee_turnover.md",
                ],
            )
            self.assertNotIn(str(ROOT), json.dumps(metadata, sort_keys=True))

    def test_generator_respects_exact_scope_dataset_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs = run(
                output_dir=Path(temporary_directory),
                run_id="core_card_test",
                config_hash="c" * 64,
                dataset_keys=("inx_primary", "hrdataset_v14"),
            )
            payload = json.loads(outputs["dataset_cards_json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_keys"], ["inx_primary", "hrdataset_v14"])
            self.assertEqual(
                {card["dataset_id"] for card in payload["cards"]},
                {"inx_primary", "hrdataset_v14"},
            )
            self.assertFalse((Path(temporary_directory) / "cards" / "ibm_hr_analytics.md").exists())


if __name__ == "__main__":
    unittest.main()
