from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from src.experiments.external_validation import run_external_dataset_experiment


def run_all_robustness() -> Dict[str, Dict[str, Path]]:
    """Run non-HRDataset robustness experiments with conservative task labels."""
    return {
        "ibm_performance": run_external_dataset_experiment("ibm_hr_analytics", target_kind="primary"),
        "ibm_attrition": run_external_dataset_experiment("ibm_hr_analytics", target_kind="attrition"),
        "employee_turnover": run_external_dataset_experiment("employee_turnover", target_kind="primary"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external HR robustness experiments.")
    parser.add_argument(
        "--task",
        default="all",
        choices=["all", "ibm_performance", "ibm_attrition", "employee_turnover"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task == "all":
        print(run_all_robustness())
    elif args.task == "ibm_performance":
        print(run_external_dataset_experiment("ibm_hr_analytics", target_kind="primary"))
    elif args.task == "ibm_attrition":
        print(run_external_dataset_experiment("ibm_hr_analytics", target_kind="attrition"))
    elif args.task == "employee_turnover":
        print(run_external_dataset_experiment("employee_turnover", target_kind="primary"))
