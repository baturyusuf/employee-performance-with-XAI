"""Build and validate the Round 2 P3 manuscript-facing subgroup summary."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


SOURCE_ROOT = Path("reports/research_log/major_revision_v3/phase2c_subgroup_proxy_use")
DEFAULT_OUTPUT = Path("reports/research_log/major_revision_round2/SUBGROUP_MANUSCRIPT_SUMMARY.csv")
SYSTEM_ID = "no_salary_hike_no_attrition_no_department"
SYSTEM_LABEL = "P3_PRIMARY_LEAKAGE_AWARE"
SUPPORT_THRESHOLD = 30
ATTRIBUTES = (
    "Age",
    "Gender",
    "MaritalStatus",
    "BusinessTravelFrequency",
    "EmpDepartment",
    "EducationBackground",
)
ATTRIBUTE_LABELS = {
    "Age": "Age",
    "Gender": "Gender",
    "MaritalStatus": "Marital Status",
    "BusinessTravelFrequency": "Business Travel",
    "EmpDepartment": "Department",
    "EducationBackground": "Education Background",
}
METRICS = ("macro_f1", "quadratic_weighted_kappa", "ordinal_mae")
SOURCE_HASHES = {
    "subgroup_gap_sensitivity.csv": "d78755b695b3520d17f877c843aca6f53414905b9a71de3a8a92f529038ee891",
    "subgroup_metric_grid.csv": "6c8ef7a473d760a82492f3bb1a9abbd6d6d04d039e134c9ee3493acd6b60d6aa",
    "primary_gap_bootstrap_intervals.csv": "10f63819d37933d8c28662c4226acf4bbcdb7c272c13102bb08930ea9397c31a",
}


class SubgroupManuscriptSummaryV4Error(RuntimeError):
    """Raised when the source or extracted summary violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SubgroupManuscriptSummaryV4Error(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sources(source_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for name, expected in SOURCE_HASHES.items():
        path = source_root / name
        _require(path.is_file(), f"Missing subgroup source: {path.as_posix()}.")
        _require(_sha256(path) == expected, f"Subgroup source hash drifted: {name}.")
    return (
        pd.read_csv(source_root / "subgroup_gap_sensitivity.csv"),
        pd.read_csv(source_root / "subgroup_metric_grid.csv"),
        pd.read_csv(source_root / "primary_gap_bootstrap_intervals.csv"),
    )


def build_subgroup_manuscript_summary_v4(source_root: Path | str = SOURCE_ROOT) -> pd.DataFrame:
    root = Path(source_root)
    gaps, grid, intervals = _load_sources(root)
    keys = ["system_id", "support_threshold", "attribute", "metric"]
    wanted = {
        "system_id": SYSTEM_ID,
        "support_threshold": SUPPORT_THRESHOLD,
    }
    gap_rows = gaps.loc[
        (gaps["system_id"] == wanted["system_id"])
        & (gaps["support_threshold"] == wanted["support_threshold"])
        & gaps["attribute"].isin(ATTRIBUTES)
        & gaps["metric"].isin(METRICS)
    ].copy()
    interval_rows = intervals.loc[
        (intervals["system_id"] == wanted["system_id"])
        & (intervals["support_threshold"] == wanted["support_threshold"])
        & intervals["attribute"].isin(ATTRIBUTES)
        & intervals["metric"].isin(METRICS)
    ].copy()
    _require(len(gap_rows) == 18 and not gap_rows.duplicated(keys).any(), "Expected 18 unique P3 gap rows.")
    _require(len(interval_rows) == 18 and not interval_rows.duplicated(keys).any(), "Expected 18 unique P3 interval rows.")
    _require(set(gap_rows["v3_system_label"]) == {SYSTEM_LABEL}, "P3 system label drifted.")
    _require(set(gap_rows["status"]) == {"estimable_descriptive_gap"}, "A selected gap is not estimable.")

    merged = gap_rows.merge(
        interval_rows[
            keys
            + [
                "gap_max_minus_min",
                "pointwise_ci_low",
                "pointwise_ci_high",
                "simultaneous_ci_low",
                "simultaneous_ci_high",
                "n_resamples",
                "n_complete_familywise_draws",
                "status",
                "interval_scope",
                "eligibility_scope",
                "model_training_variability_included",
                "claim_boundary",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("_gap", "_interval"),
    )
    _require(merged["status_interval"].notna().all(), "An interval row failed to join.")
    _require(
        np.allclose(merged["gap_max_minus_min_gap"], merged["gap_max_minus_min_interval"], rtol=0.0, atol=1e-15),
        "Gap and interval point estimates differ.",
    )

    source_row_numbers: dict[tuple[str, str], tuple[int, int]] = {}
    recomputed: dict[tuple[str, str], tuple[str, float, str, float, int]] = {}
    for _, row in gap_rows.iterrows():
        attribute = str(row["attribute"])
        metric = str(row["metric"])
        eligible = grid.loc[
            (grid["system_id"] == SYSTEM_ID)
            & (grid["support_threshold"] == SUPPORT_THRESHOLD)
            & (grid["attribute"] == attribute)
            & (grid["metric"] == metric)
            & grid["eligible_for_gap"].astype(bool)
        ].copy()
        _require(len(eligible) == int(row["eligible_group_count"]), f"Eligible group count drifted for {attribute}/{metric}.")
        minimum = eligible.loc[eligible["point_estimate"].idxmin()]
        maximum = eligible.loc[eligible["point_estimate"].idxmax()]
        calculated_gap = float(maximum["point_estimate"] - minimum["point_estimate"])
        _require(np.isclose(calculated_gap, float(row["gap_max_minus_min"]), rtol=0.0, atol=1e-15), f"Gap failed source-grid replay for {attribute}/{metric}.")
        _require(str(minimum["group"]) == str(row["minimum_group"]), f"Minimum endpoint drifted for {attribute}/{metric}.")
        _require(str(maximum["group"]) == str(row["maximum_group"]), f"Maximum endpoint drifted for {attribute}/{metric}.")
        gap_index = int(gaps.index[(gaps[keys] == row[keys].values).all(axis=1)][0]) + 2
        interval_match = intervals.loc[(intervals[keys] == row[keys].values).all(axis=1)]
        interval_index = int(interval_match.index[0]) + 2
        source_row_numbers[(attribute, metric)] = (gap_index, interval_index)
        recomputed[(attribute, metric)] = (
            str(minimum["group"]),
            float(minimum["point_estimate"]),
            str(maximum["group"]),
            float(maximum["point_estimate"]),
            len(eligible),
        )

    attribute_order = {value: index for index, value in enumerate(ATTRIBUTES)}
    metric_order = {value: index for index, value in enumerate(METRICS)}
    merged["_attribute_order"] = merged["attribute"].map(attribute_order)
    merged["_metric_order"] = merged["metric"].map(metric_order)
    merged = merged.sort_values(["_attribute_order", "_metric_order"], kind="stable")

    records: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        key = (str(row["attribute"]), str(row["metric"]))
        minimum_group, minimum_value, maximum_group, maximum_value, eligible_count = recomputed[key]
        gap_row, interval_row = source_row_numbers[key]
        records.append(
            {
                "policy_id": "P3",
                "policy_label": "Primary Leakage-Aware",
                "support_threshold": SUPPORT_THRESHOLD,
                "attribute": ATTRIBUTE_LABELS[key[0]],
                "source_attribute": key[0],
                "metric": key[1],
                "higher_value_is_better": bool(row["higher_value_is_better"]),
                "gap_max_minus_min": float(row["gap_max_minus_min_gap"]),
                "minimum_group": minimum_group,
                "minimum_value": minimum_value,
                "maximum_group": maximum_group,
                "maximum_value": maximum_value,
                "eligible_group_count": eligible_count,
                "declared_group_count": int(row["declared_group_count"]),
                "gap_status": str(row["status_gap"]),
                "pointwise_ci_low": float(row["pointwise_ci_low"]),
                "pointwise_ci_high": float(row["pointwise_ci_high"]),
                "simultaneous_ci_low": float(row["simultaneous_ci_low"]),
                "simultaneous_ci_high": float(row["simultaneous_ci_high"]),
                "interval_status": str(row["status_interval"]),
                "interval_scope": str(row["interval_scope"]),
                "model_training_variability_included": bool(row["model_training_variability_included"]),
                "n_resamples": int(row["n_resamples"]),
                "n_complete_familywise_draws": int(row["n_complete_familywise_draws"]),
                "source_gap_row": gap_row,
                "source_interval_row": interval_row,
                "source_gap_sha256": SOURCE_HASHES["subgroup_gap_sensitivity.csv"],
                "source_grid_sha256": SOURCE_HASHES["subgroup_metric_grid.csv"],
                "source_interval_sha256": SOURCE_HASHES["primary_gap_bootstrap_intervals.csv"],
                "claim_boundary": str(row["claim_boundary"]),
            }
        )
    return pd.DataFrame.from_records(records)


def write_subgroup_manuscript_summary_v4(
    output_path: Path | str = DEFAULT_OUTPUT,
    source_root: Path | str = SOURCE_ROOT,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_subgroup_manuscript_summary_v4(source_root).to_csv(output, index=False, lineterminator="\n", float_format="%.17g")
    return output


def validate_subgroup_manuscript_summary_v4(
    output_path: Path | str = DEFAULT_OUTPUT,
    source_root: Path | str = SOURCE_ROOT,
) -> dict[str, object]:
    output = Path(output_path)
    _require(output.is_file(), f"Missing subgroup summary: {output.as_posix()}.")
    expected = build_subgroup_manuscript_summary_v4(source_root)
    observed = pd.read_csv(output)
    _require(list(observed.columns) == list(expected.columns), "Subgroup summary columns drifted.")
    _require(len(observed) == 18, "Subgroup summary row count drifted.")
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            _require(np.allclose(observed[column], expected[column], rtol=0.0, atol=1e-15), f"Numeric subgroup summary column drifted: {column}.")
        else:
            _require(observed[column].astype(str).tolist() == expected[column].astype(str).tolist(), f"Subgroup summary column drifted: {column}.")
    return {
        "status": "passed",
        "row_count": len(observed),
        "attribute_count": observed["attribute"].nunique(),
        "metric_count": observed["metric"].nunique(),
        "output_sha256": _sha256(output),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.validate_only:
        write_subgroup_manuscript_summary_v4(args.output, args.source_root)
    print(validate_subgroup_manuscript_summary_v4(args.output, args.source_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SubgroupManuscriptSummaryV4Error",
    "build_subgroup_manuscript_summary_v4",
    "validate_subgroup_manuscript_summary_v4",
    "write_subgroup_manuscript_summary_v4",
]
