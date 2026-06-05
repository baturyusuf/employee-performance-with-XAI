from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.data.load_data import load_raw_data
from src.data.validate_schema import validate_dataframe
from src.features.feature_sets import get_taxonomy_rows
from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


DATA_AUDIT_DIR = SETTINGS.reports_dir / "data_audit"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def build_schema_report(df: pd.DataFrame, expected_columns: List[str]) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append(
            {
                "feature_name": col,
                "observed": True,
                "expected": col in expected_columns,
                "pandas_dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "missing_count": int(df[col].isna().sum()),
                "missing_rate": float(df[col].isna().mean()),
                "n_unique": int(df[col].nunique(dropna=True)),
            }
        )

    observed = set(df.columns)
    for col in expected_columns:
        if col not in observed:
            rows.append(
                {
                    "feature_name": col,
                    "observed": False,
                    "expected": True,
                    "pandas_dtype": "",
                    "non_null_count": 0,
                    "missing_count": None,
                    "missing_rate": None,
                    "n_unique": None,
                }
            )

    return pd.DataFrame(rows).sort_values(["expected", "observed", "feature_name"], ascending=[False, False, True])


def build_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "feature_name": col,
            "missing_count": int(df[col].isna().sum()),
            "missing_rate": float(df[col].isna().mean()),
        }
        for col in df.columns
    ]
    return pd.DataFrame(rows).sort_values(["missing_count", "feature_name"], ascending=[False, True])


def build_target_distribution(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    counts = df[target_col].value_counts(dropna=False).sort_index()
    total = len(df)
    return pd.DataFrame(
        {
            "target_value": counts.index.astype(str),
            "count": counts.values.astype(int),
            "proportion": [float(count / total) for count in counts.values],
        }
    )


def build_duplicate_report(df: pd.DataFrame, id_col: str) -> Dict[str, Any]:
    duplicate_id_count = int(df[id_col].duplicated().sum()) if id_col in df.columns else None
    duplicate_row_count = int(df.duplicated().sum())
    return {
        "id_column": id_col,
        "duplicate_id_count": duplicate_id_count,
        "duplicate_row_count": duplicate_row_count,
        "row_count": int(len(df)),
    }


def build_taxonomy_review(df: pd.DataFrame) -> pd.DataFrame:
    taxonomy_rows = get_taxonomy_rows()
    observed_columns = set(df.columns)
    rows = []

    for row in taxonomy_rows:
        out = dict(row)
        out["observed_in_data"] = row["feature_name"] in observed_columns
        rows.append(out)

    taxonomy_features = {row["feature_name"] for row in taxonomy_rows}
    for col in sorted(observed_columns - taxonomy_features):
        rows.append(
            {
                "feature_name": col,
                "data_type": "unknown",
                "temporal_status": "unknown",
                "leakage_risk": "unknown",
                "control_type": "unknown",
                "sensitive_or_proxy": "unknown",
                "allowed_for_final_model": False,
                "notes": "Observed in data but missing from taxonomy.",
                "observed_in_data": True,
            }
        )

    return pd.DataFrame(rows)


def feature_temporality_audit_text(df: pd.DataFrame) -> str:
    target_counts = df[SETTINGS.target_col].value_counts().sort_index().to_dict()
    return f"""# Feature Temporality Audit

Date: 2026-06-05

Rows audited: {len(df)}

Target distribution: {target_counts}

## Summary

This audit classifies features by temporal and governance risk before model selection. It does not prove causal leakage. It identifies variables that are outcome-proximal, post-evaluation-risk, organisational grouping variables, or plausible proxies that must be tested through ablation, fairness analysis, proxy analysis, and explanation shift.

## High Leakage-Risk Features

### EmpLastSalaryHikePercent

`EmpLastSalaryHikePercent` is high leakage risk because salary increases are often determined after or during performance review cycles. If the model uses this field, performance can look artificially high because the feature may encode the outcome or a manager decision derived from the outcome. Full-feature models using this variable must be treated as upper-bound/leakage-warning baselines, not deployable HR decision-support models.

### Attrition

`Attrition` is high risk unless the prediction time is explicitly before attrition status is known. If attrition is observed after the performance period, it may encode downstream employee or organisational outcomes. The primary leakage-safe feature set excludes it.

## Organisational Group and Proxy Risk

### EmpDepartment

`EmpDepartment` is not necessarily a legally protected attribute by itself, but it encodes organisational group membership. Direct department dependence can create unequal error patterns across departments and can hide process differences between business units. The department-free feature set is therefore required as a fairness-aware sensitivity analysis.

### EmpJobRole

`EmpJobRole` may remain a proxy for department after `EmpDepartment` is removed. If job role predicts department well, a department-free model can still retain indirect department dependence. A proxy analysis should test whether remaining features, especially job role, reconstruct `EmpDepartment`.

## Concurrent Survey and Historical Decision Features

Satisfaction, involvement, overtime, work-life balance, years since promotion, and related features may be measured near the evaluation period or encode prior managerial decisions. They are not automatically removed, but reason codes and counterfactuals must avoid causal language and employee-blaming recommendations.

## Required Follow-Up

- Compare full-feature and leakage-safe feature sets.
- Compute Leakage Sensitivity Index for macro-F1 and QWK.
- Run proxy analysis for `EmpDepartment`.
- Review whether `Age` should be excluded from final model candidates.
- Use the taxonomy for counterfactual actionability constraints.
"""


def run_data_audit(output_dir: Path = DATA_AUDIT_DIR, write_registry: bool = True) -> Dict[str, Path]:
    ensure_dir(output_dir)

    raw_df = load_raw_data()
    df, validation_report = validate_dataframe(raw_df)

    outputs = {
        "schema_report": output_dir / "schema_report.csv",
        "target_distribution": output_dir / "target_distribution.csv",
        "missingness_report": output_dir / "missingness_report.csv",
        "duplicate_report": output_dir / "duplicate_report.json",
        "feature_taxonomy_review": output_dir / "feature_taxonomy_review.csv",
        "feature_temporality_audit": output_dir / "feature_temporality_audit.md",
    }

    build_schema_report(df, SETTINGS.expected_columns).to_csv(outputs["schema_report"], index=False)
    build_target_distribution(df, SETTINGS.target_col).to_csv(outputs["target_distribution"], index=False)
    build_missingness_report(df).to_csv(outputs["missingness_report"], index=False)
    save_json(build_duplicate_report(df, SETTINGS.id_col), outputs["duplicate_report"])
    build_taxonomy_review(df).to_csv(outputs["feature_taxonomy_review"], index=False)
    outputs["feature_temporality_audit"].write_text(feature_temporality_audit_text(df), encoding="utf-8")

    if write_registry:
        append_registry_row(
            {
                "run_id": "data_audit_20260605",
                "date_time": utc_now_iso(),
                "git_commit_if_available": get_git_commit(),
                "script": "python -m src.data.audit",
                "config": "configs/project.yaml; configs/feature_taxonomy.yaml",
                "feature_set": "not_applicable",
                "model": "not_applicable",
                "seed": "not_applicable",
                "cv_strategy": "not_applicable",
                "primary_metrics": {
                    "rows": validation_report.row_count,
                    "columns": validation_report.column_count,
                    "duplicate_id_count": validation_report.duplicate_id_count,
                    "null_target_count": validation_report.null_target_count,
                },
                "output_dir": str(output_dir.relative_to(SETTINGS.project_root)),
                "notes": "Generated schema, target distribution, missingness, duplicate, taxonomy review, and temporality audit outputs.",
                "decision_status": "candidate",
            }
        )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data and feature temporality audit reports.")
    parser.add_argument("--output-dir", type=Path, default=DATA_AUDIT_DIR)
    parser.add_argument("--no-registry", action="store_true", help="Do not append to experiment registry.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = run_data_audit(output_dir=args.output_dir, write_registry=not args.no_registry)
    print("Data audit outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
