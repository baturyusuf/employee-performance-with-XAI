from __future__ import annotations

import unittest

from src.utils.config_loader import load_config, load_feature_sets_config, load_feature_taxonomy_config


class ConfigLoadingTests(unittest.TestCase):
    def test_project_config_loads(self) -> None:
        config = load_config("project")
        self.assertEqual(config["project"]["target_column"], "PerformanceRating")
        self.assertEqual(config["project"]["id_column"], "EmpNumber")

    def test_feature_sets_config_loads(self) -> None:
        config = load_feature_sets_config()
        self.assertIn("no_salary_hike_no_attrition", config["feature_sets"])
        self.assertIn("full_feature_upper_bound", config["feature_sets"])

    def test_feature_taxonomy_has_unique_entries(self) -> None:
        config = load_feature_taxonomy_config()
        names = [row["feature_name"] for row in config["feature_taxonomy"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("EmpLastSalaryHikePercent", names)


if __name__ == "__main__":
    unittest.main()
