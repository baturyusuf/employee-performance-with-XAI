from __future__ import annotations

import unittest

from src.governance.warning_taxonomy import (
    MANDATORY_WARNING_IDS,
    add_mandatory_warning_records,
    normalize_warning_records,
    normalize_warning_text,
)


class WarningTaxonomyTests(unittest.TestCase):
    def test_full_feature_warning_maps_to_leakage_id(self) -> None:
        record = normalize_warning_text("Full-feature models are leakage-warning upper-bound baselines only.")
        self.assertEqual(record["warning_id"], "leakage.full_feature_upper_bound_only")
        self.assertEqual(record["category"], "leakage")

    def test_job_role_proxy_warning_maps_to_proxy_id(self) -> None:
        records = normalize_warning_records(["JobRole may proxy Department even after department removal."])
        self.assertEqual(records[0]["warning_id"], "proxy.jobrole_department_proxy")

    def test_mandatory_warnings_are_added_once(self) -> None:
        records = add_mandatory_warning_records(
            ["Prediction requires human review.", "Prediction requires human review."]
        )
        warning_ids = [item["warning_id"] for item in records]
        self.assertEqual(len(warning_ids), len(set(warning_ids)))
        for warning_id in MANDATORY_WARNING_IDS:
            self.assertIn(warning_id, warning_ids)


if __name__ == "__main__":
    unittest.main()
