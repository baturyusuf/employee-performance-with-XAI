"""Execute the aggregate-only Phase 3B quality audit for canonical core data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.canonical_loader import load_canonical_dataset, sha256_file
from src.data.external_adapters import build_feature_columns, load_external_dataset
from src.governance.data_quality_contract_v3 import (
    DATASET_KEYS,
    DEFAULT_DATA_QUALITY_CONTRACT,
    validate_data_quality_contract_v3,
)
from src.governance.manuscript_contract import source_tree_hash
from src.governance.offline_runtime import enforce_offline_runtime
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_LOCAL_RUN_ROOT = Path("reports/major_revision_v3_runs")
EXPECTED_LOCAL_FILES = frozenset(
    {
        "categorical_cardinality.csv",
        "cleaned_schema.csv",
        "column_profiles.csv",
        "dataset_summary.csv",
        "duplicate_audit.csv",
        "identifier_audit.csv",
        "manuscript_ready_data_quality.csv",
        "raw_schema.csv",
        "rule_anomaly_audit.csv",
        "stage_metadata.json",
        "target_distribution.csv",
    }
)


class DataQualityV3Error(RuntimeError):
    """Raised when Phase 3B execution violates an audit invariant."""


@dataclass(frozen=True)
class DataQualityV3Result:
    dataset_summary: pd.DataFrame
    column_profiles: pd.DataFrame
    categorical_cardinality: pd.DataFrame
    duplicate_audit: pd.DataFrame
    identifier_audit: pd.DataFrame
    rule_anomaly_audit: pd.DataFrame
    target_distribution: pd.DataFrame
    raw_schema: pd.DataFrame
    cleaned_schema: pd.DataFrame
    manuscript_ready_data_quality: pd.DataFrame


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataQualityV3Error(message)


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _schema_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"position": position, "name": str(column), "pandas_dtype": str(frame[column].dtype)}
        for position, column in enumerate(frame.columns, start=1)
    ]


def _schema_sha256(frame: pd.DataFrame) -> str:
    return _canonical_json_sha256(_schema_records(frame))


def _blank_mask(series: pd.Series) -> pd.Series:
    if not (
        pd.api.types.is_object_dtype(series.dtype)
        or isinstance(series.dtype, pd.StringDtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return pd.Series(False, index=series.index, dtype=bool)
    return series.astype("string").str.strip().eq("").fillna(False)


def _is_categorical(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series.dtype)
        or isinstance(series.dtype, pd.StringDtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series.dtype)
    )


def _profile_columns(dataset_key: str, raw: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    row_count = len(raw)
    for position, column in enumerate(raw.columns, start=1):
        series = raw[column]
        blank = _blank_mask(series)
        missing_count = int(series.isna().sum())
        blank_count = int(blank.sum())
        effective = series.mask(blank)
        nonmissing = effective.dropna()
        unique_count = int(nonmissing.nunique(dropna=True))
        if len(nonmissing):
            mode_count = int(nonmissing.value_counts(dropna=True).iloc[0])
            mode_share = float(mode_count / len(nonmissing))
        else:
            mode_count = 0
            mode_share = float("nan")
        numeric = pd.to_numeric(nonmissing, errors="coerce") if pd.api.types.is_numeric_dtype(series.dtype) else pd.Series(dtype=float)
        rows.append(
            {
                "dataset_key": dataset_key,
                "column_position": position,
                "column_name": str(column),
                "pandas_dtype": str(series.dtype),
                "row_count": row_count,
                "nonmissing_count": int(len(nonmissing)),
                "missing_count": missing_count,
                "blank_string_count": blank_count,
                "effective_missing_count": missing_count + blank_count,
                "effective_missing_percentage": float((missing_count + blank_count) / row_count * 100.0),
                "unique_nonmissing_count": unique_count,
                "mode_count_nonmissing": mode_count,
                "mode_share_nonmissing": mode_share,
                "is_constant_nonmissing": bool(unique_count == 1),
                "is_near_constant_nonmissing": bool(unique_count > 0 and mode_share >= threshold),
                "is_categorical": _is_categorical(series),
                "categorical_cardinality": unique_count if _is_categorical(series) else np.nan,
                "numeric_minimum": float(numeric.min()) if len(numeric) else np.nan,
                "numeric_maximum": float(numeric.max()) if len(numeric) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _duplicate_row(dataset_key: str, raw: pd.DataFrame, target: str) -> dict[str, Any]:
    predictor_columns = [column for column in raw.columns if column != target]
    exact_extra = int(raw.duplicated(keep="first").sum())
    exact_participating = int(raw.duplicated(keep=False).sum())
    predictor_extra = int(raw.duplicated(subset=predictor_columns, keep="first").sum())
    predictor_participating = raw.duplicated(subset=predictor_columns, keep=False)
    conflicting_groups = 0
    conflicting_rows = 0
    if bool(predictor_participating.any()):
        grouped = raw.loc[predictor_participating].groupby(
            predictor_columns, dropna=False, sort=False
        )[target]
        target_counts = grouped.nunique(dropna=False)
        conflicting_groups = int((target_counts > 1).sum())
        if conflicting_groups:
            conflict_keys = set(target_counts.loc[target_counts > 1].index.tolist())
            for key, group in grouped:
                if key in conflict_keys:
                    conflicting_rows += int(len(group))
    return {
        "dataset_key": dataset_key,
        "row_count": int(len(raw)),
        "exact_duplicate_extra_rows": exact_extra,
        "exact_duplicate_participating_rows": exact_participating,
        "predictor_duplicate_extra_rows": predictor_extra,
        "predictor_duplicate_participating_rows": int(predictor_participating.sum()),
        "conflicting_target_groups": conflicting_groups,
        "conflicting_target_rows": conflicting_rows,
        "raw_target_excluded_for_predictor_duplicate_test": target,
        "status": "finding" if any((exact_extra, predictor_extra, conflicting_groups)) else "passed",
    }


def _normalised_name_signal(column: str, tokens: Sequence[str]) -> bool:
    parts = [part.casefold() for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", column)]
    compact = column.casefold()
    return any(token.casefold() in parts for token in tokens) or compact.endswith("id")


def _exact_unit_sequence(series: pd.Series) -> bool:
    nonmissing = series.dropna()
    if len(nonmissing) < 2 or nonmissing.nunique(dropna=True) != len(nonmissing):
        return False
    numeric = pd.to_numeric(nonmissing, errors="coerce")
    if numeric.notna().all():
        values = numeric.to_numpy(dtype=float)
    else:
        extracted = nonmissing.astype(str).str.extract(r"(\d+)$", expand=False)
        if extracted.isna().any():
            return False
        values = extracted.astype(float).to_numpy()
    return bool(np.all(np.diff(values) == 1.0))


def _primary_features(
    inx: pd.DataFrame,
    hr_dataset: Any,
    contract: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    feature_contract = load_config(
        PROJECT_ROOT / contract["source_contracts"]["inx_feature_availability"]["path"]
    )
    p3 = next(item for item in feature_contract["policies"] if item["policy_id"] == "P3")
    inx_features = tuple(column for column in inx.columns if column not in set(p3["excluded_features"]))
    hr_features = tuple(build_feature_columns(hr_dataset, "conservative_primary"))
    return {"inx_primary": inx_features, "hrdataset_v14": hr_features}


def _identifier_rows(
    dataset_key: str,
    raw: pd.DataFrame,
    definition: Mapping[str, Any],
    primary_features: Sequence[str],
    tokens: Sequence[str],
    unique_threshold: float,
) -> list[dict[str, Any]]:
    declared = set(str(value) for value in definition["declared_identifier_columns"])
    rows: list[dict[str, Any]] = []
    for column in raw.columns:
        series = raw[column]
        support = int(series.notna().sum())
        ratio = float(series.nunique(dropna=True) / support) if support else 0.0
        name_signal = _normalised_name_signal(str(column), tokens)
        sequence_signal = _exact_unit_sequence(series)
        detected = bool(column in declared or sequence_signal or (name_signal and ratio >= unique_threshold))
        if not detected:
            continue
        aliases = {str(column)}
        if dataset_key == "hrdataset_v14" and column == "EmpID":
            aliases.add("EmpNumber")
        included_aliases = sorted(aliases.intersection(primary_features))
        rows.append(
            {
                "dataset_key": dataset_key,
                "column_name": str(column),
                "schema_stage": "raw",
                "declared_identifier": bool(column in declared),
                "identifier_name_signal": name_signal,
                "unique_ratio_nonmissing": ratio,
                "exact_row_order_unit_sequence": sequence_signal,
                "detected_identifier_candidate": detected,
                "primary_model_feature_alias_count": len(included_aliases),
                "primary_model_feature_status": "excluded" if not included_aliases else "included",
                "leakage_audit_status": "passed_excluded" if not included_aliases else "failed_identifier_in_primary_features",
                "interpretation": "identifier_or_name_field_not_a_model_predictor" if not included_aliases else "identifier_candidate_requires_removal",
            }
        )
    if dataset_key == "hrdataset_v14":
        generated_included = "ExternalSampleId" in primary_features
        rows.append(
            {
                "dataset_key": dataset_key,
                "column_name": "ExternalSampleId",
                "schema_stage": "cleaned_generated",
                "declared_identifier": True,
                "identifier_name_signal": True,
                "unique_ratio_nonmissing": 1.0,
                "exact_row_order_unit_sequence": True,
                "detected_identifier_candidate": True,
                "primary_model_feature_alias_count": int(generated_included),
                "primary_model_feature_status": "included" if generated_included else "excluded",
                "leakage_audit_status": "failed_identifier_in_primary_features" if generated_included else "passed_excluded",
                "interpretation": "generated_row_key_not_a_model_predictor" if not generated_included else "generated_row_key_requires_removal",
            }
        )
    return rows


def _audit_numeric_rules(dataset_key: str, raw: pd.DataFrame, rules: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in rules:
        column = str(rule["column"])
        numeric = pd.to_numeric(raw[column], errors="coerce")
        missing = numeric.isna()
        if "allowed" in rule:
            anomaly = (~missing) & (~numeric.isin([float(value) for value in rule["allowed"]]))
            expectation = "allowed_values_contract"
        else:
            anomaly = (~missing) & (
                (numeric < float(rule["minimum"])) | (numeric > float(rule["maximum"]))
            )
            expectation = f"inclusive_range_{rule['minimum']}_to_{rule['maximum']}"
        anomaly_count = int(anomaly.sum())
        checked = int((~missing).sum())
        rows.append(
            _rule_row(
                dataset_key,
                f"numeric_{column}",
                "numeric_domain",
                column,
                checked,
                int(missing.sum()),
                anomaly_count,
                expectation,
            )
        )
    return rows


def _compare(left: pd.Series, operator: str, right: pd.Series) -> pd.Series:
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">=":
        return left >= right
    if operator == ">":
        return left > right
    raise DataQualityV3Error(f"Unsupported relational operator: {operator}.")


def _rule_row(
    dataset_key: str,
    rule_id: str,
    rule_type: str,
    columns: str,
    checked_count: int,
    missing_or_skipped_count: int,
    anomaly_count: int,
    expectation: str,
) -> dict[str, Any]:
    return {
        "dataset_key": dataset_key,
        "rule_id": rule_id,
        "rule_type": rule_type,
        "columns": columns,
        "checked_count": checked_count,
        "missing_or_skipped_count": missing_or_skipped_count,
        "anomaly_count": anomaly_count,
        "anomaly_percentage_of_checked": float(anomaly_count / checked_count * 100.0) if checked_count else 0.0,
        "status": "finding" if anomaly_count else "passed",
        "expectation": expectation,
    }


def _audit_relations(dataset_key: str, raw: pd.DataFrame, rules: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in rules:
        left = pd.to_numeric(raw[str(rule["left"])], errors="coerce")
        right = pd.to_numeric(raw[str(rule["right"])], errors="coerce")
        comparable = left.notna() & right.notna()
        valid = _compare(left, str(rule["operator"]), right)
        anomalies = comparable & (~valid)
        rows.append(
            _rule_row(
                dataset_key,
                str(rule["rule_id"]),
                "numeric_relation",
                f"{rule['left']} {rule['operator']} {rule['right']}",
                int(comparable.sum()),
                int((~comparable).sum()),
                int(anomalies.sum()),
                "declared_cross_field_relation",
            )
        )
    return rows


def _parse_hr_dates(raw: pd.DataFrame, date_rules: Mapping[str, Any]) -> tuple[dict[str, pd.Series], list[dict[str, Any]]]:
    parsed: dict[str, pd.Series] = {}
    for column, rule in date_rules.items():
        parsed[column] = pd.to_datetime(raw[column], format=str(rule["format"]), errors="coerce")
    latest_context_date = pd.concat(
        [parsed["DateofHire"], parsed["DateofTermination"], parsed["LastPerformanceReview_Date"]],
        ignore_index=True,
    ).max()
    future_dob = parsed["DOB"] > latest_context_date
    parsed["DOB"] = parsed["DOB"].where(~future_dob, parsed["DOB"] - pd.DateOffset(years=100))
    rows: list[dict[str, Any]] = []
    for column, rule in date_rules.items():
        supplied = raw[column].notna() & (~_blank_mask(raw[column]))
        invalid = supplied & parsed[column].isna()
        rows.append(
            _rule_row(
                "hrdataset_v14",
                f"date_parse_{column}",
                "date_parse",
                column,
                int(supplied.sum()),
                int((~supplied).sum()),
                int(invalid.sum()),
                str(rule["format"]) + ("_with_declared_two_digit_year_normalization" if column == "DOB" else ""),
            )
        )
    return parsed, rows


def _audit_hr_relations(raw: pd.DataFrame, rules: Sequence[Mapping[str, Any]], parsed: Mapping[str, pd.Series]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in rules:
        left = parsed[str(rule["left"])]
        right = parsed[str(rule["right"])]
        comparable = left.notna() & right.notna()
        valid = _compare(left, str(rule["operator"]), right)
        anomaly = comparable & (~valid)
        rows.append(
            _rule_row(
                "hrdataset_v14",
                str(rule["rule_id"]),
                "date_relation",
                f"{rule['left']} {rule['operator']} {rule['right']}",
                int(comparable.sum()),
                int((~comparable).sum()),
                int(anomaly.sum()),
                "declared_temporal_relation",
            )
        )
    return rows


def _audit_hr_consistency(raw: pd.DataFrame, rules: Sequence[Mapping[str, Any]], parsed_dates: Mapping[str, pd.Series]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in rules:
        rule_type = str(rule["type"])
        rule_id = str(rule["rule_id"])
        if rule_type in {"text_code_mapping", "normalized_text_code_mapping"}:
            text_column = str(rule["text_column"])
            code_column = str(rule["code_column"])
            text = raw[text_column].astype("string").str.strip()
            expected = text.map(rule["mapping"])
            observed = pd.to_numeric(raw[code_column], errors="coerce")
            comparable = expected.notna() & observed.notna()
            anomaly = comparable & (expected.astype("Float64") != observed.astype("Float64"))
            columns = f"{text_column}->{code_column}"
            expectation = "declared_text_code_correspondence"
        elif rule_type == "boolean_text_mapping":
            text_column = str(rule["text_column"])
            code_column = str(rule["code_column"])
            text = raw[text_column].astype("string").str.strip()
            expected = pd.Series(
                np.where(text.eq(str(rule["positive_text"])), int(rule["positive_code"]), int(rule["negative_code"])),
                index=raw.index,
            )
            observed = pd.to_numeric(raw[code_column], errors="coerce")
            comparable = text.notna() & observed.notna()
            anomaly = comparable & (expected != observed)
            columns = f"{text_column}->{code_column}"
            expectation = "positive_text_flag_correspondence"
        elif rule_type == "termination_flag_date":
            observed = pd.to_numeric(raw["Termd"], errors="coerce")
            termination = parsed_dates["DateofTermination"]
            comparable = observed.notna()
            anomaly = comparable & (((observed == 1) & termination.isna()) | ((observed == 0) & termination.notna()))
            columns = "Termd<->DateofTermination"
            expectation = "terminated_iff_termination_date_present"
        elif rule_type == "code_to_text_function":
            code_column = str(rule["code_column"])
            text_column = str(rule["text_column"])
            complete = raw[[code_column, text_column]].dropna()
            anomaly_count = 0
            for _, group in complete.groupby(code_column, sort=False):
                counts = group[text_column].astype("string").str.strip().value_counts()
                anomaly_count += int(len(group) - int(counts.iloc[0]))
            rows.append(
                _rule_row(
                    "hrdataset_v14",
                    rule_id,
                    rule_type,
                    f"{code_column}->{text_column}",
                    int(len(complete)),
                    int(len(raw) - len(complete)),
                    anomaly_count,
                    "each_code_has_one_text_label_minority_rows_counted_as_findings",
                )
            )
            continue
        else:
            raise DataQualityV3Error(f"Unsupported HR consistency rule: {rule_type}.")
        rows.append(
            _rule_row(
                "hrdataset_v14",
                rule_id,
                rule_type,
                columns,
                int(comparable.sum()),
                int((~comparable).sum()),
                int(anomaly.sum()),
                expectation,
            )
        )
    return rows


def _schema_frame(dataset_key: str, frame: pd.DataFrame, stage: str, source_map: Mapping[str, str] | None = None) -> pd.DataFrame:
    digest = _schema_sha256(frame)
    rows: list[dict[str, Any]] = []
    source_map = source_map or {}
    for record in _schema_records(frame):
        column = str(record["name"])
        source = source_map.get(column, column)
        if stage == "cleaned" and column == "ExternalSampleId":
            transformation = "generated_zero_based_row_key_excluded_from_models"
        elif stage == "cleaned" and column == "ExperienceYearsAtThisCompany" and dataset_key == "hrdataset_v14":
            transformation = "derived_review_minus_hire_years_negative_values_set_missing"
        elif stage == "cleaned" and column == "PerformanceRating" and dataset_key == "hrdataset_v14":
            transformation = "mapped_from_raw_performance_score_to_ordered_2_3_4"
        elif source != column:
            transformation = "renamed_and_string_values_trimmed_where_applicable"
        else:
            transformation = "verified_and_string_values_trimmed_where_applicable" if stage == "cleaned" else "none_raw_profile"
        rows.append(
            {
                "dataset_key": dataset_key,
                "schema_stage": stage,
                "column_position": record["position"],
                "column_name": column,
                "pandas_dtype": record["pandas_dtype"],
                "source_column": source,
                "transformation": transformation,
                "missing_count": int(frame[column].isna().sum()),
                "schema_sha256": digest,
            }
        )
    return pd.DataFrame(rows)


def _target_rows(dataset_key: str, target_view: str, target_column: str, series: pd.Series) -> list[dict[str, Any]]:
    counts = series.value_counts(dropna=False, sort=False)
    total = len(series)
    rows: list[dict[str, Any]] = []
    for label, count in counts.items():
        label_text = "<NA>" if pd.isna(label) else str(label)
        if isinstance(label, (float, np.floating)) and float(label).is_integer():
            label_text = str(int(label))
        rows.append(
            {
                "dataset_key": dataset_key,
                "target_view": target_view,
                "target_column": target_column,
                "target_label": label_text,
                "count": int(count),
                "percentage": float(count / total * 100.0),
            }
        )
    return rows


def evaluate_data_quality_v3(contract: Mapping[str, Any]) -> DataQualityV3Result:
    """Compute aggregate profiles without editing or repairing source records."""

    sources = contract["source_contracts"]
    verified = {
        key: load_canonical_dataset(
            sources["canonical_config"]["path"],
            key,
            sources["acquisition_manifest"]["path"],
            allow_download=False,
        )
        for key in DATASET_KEYS
    }
    raw_frames = {key: value.frame.copy(deep=True) for key, value in verified.items()}
    hr_dataset = load_external_dataset(
        "hrdataset_v14",
        raw_frame=raw_frames["hrdataset_v14"],
        schema_mapping_path=PROJECT_ROOT / sources["hrdataset_schema_mapping"]["path"],
    )
    cleaned_frames = {
        "inx_primary": raw_frames["inx_primary"].copy(deep=True),
        "hrdataset_v14": hr_dataset.canonical.copy(deep=True),
    }
    primary_features = _primary_features(raw_frames["inx_primary"], hr_dataset, contract)
    policy = contract["audit_policy"]
    threshold = float(policy["near_constant_nonmissing_mode_share_threshold"])
    tokens = list(policy["identifier_name_tokens"])
    unique_threshold = float(policy["identifier_unique_ratio_threshold"])

    profile_frames: list[pd.DataFrame] = []
    duplicate_rows: list[dict[str, Any]] = []
    identifier_rows: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    raw_schema_frames: list[pd.DataFrame] = []
    cleaned_schema_frames: list[pd.DataFrame] = []

    for dataset_key in DATASET_KEYS:
        raw = raw_frames[dataset_key]
        definition = contract["datasets"][dataset_key]
        profile_frames.append(_profile_columns(dataset_key, raw, threshold))
        duplicate_rows.append(_duplicate_row(dataset_key, raw, str(definition["raw_target"])))
        identifier_rows.extend(
            _identifier_rows(
                dataset_key,
                raw,
                definition,
                primary_features[dataset_key],
                tokens,
                unique_threshold,
            )
        )
        rule_rows.extend(_audit_numeric_rules(dataset_key, raw, definition["numeric_rules"]))
        if dataset_key == "inx_primary":
            rule_rows.extend(_audit_relations(dataset_key, raw, definition["relational_rules"]))
        else:
            parsed_dates, date_rows = _parse_hr_dates(raw, definition["date_rules"])
            rule_rows.extend(date_rows)
            rule_rows.extend(_audit_hr_relations(raw, definition["relational_rules"], parsed_dates))
            rule_rows.extend(_audit_hr_consistency(raw, definition["consistency_rules"], parsed_dates))
        target_rows.extend(
            _target_rows(dataset_key, "raw", str(definition["raw_target"]), raw[str(definition["raw_target"])])
        )
        raw_schema_frames.append(_schema_frame(dataset_key, raw, "raw"))

    target_rows.extend(
        _target_rows(
            "hrdataset_v14",
            "canonical_mapped",
            "PerformanceRating",
            cleaned_frames["hrdataset_v14"]["PerformanceRating"],
        )
    )
    rename_map = load_config(PROJECT_ROOT / sources["hrdataset_schema_mapping"]["path"])["rename_columns"]
    inverse_hr_sources = {str(target): str(source) for source, target in rename_map.items()}
    inverse_hr_sources.update(
        {
            "ExternalSampleId": "<generated>",
            "ExperienceYearsAtThisCompany": "DateofHire+LastPerformanceReview_Date",
            "PerformanceRating": "PerformanceScore",
        }
    )
    cleaned_schema_frames.append(_schema_frame("inx_primary", cleaned_frames["inx_primary"], "cleaned"))
    cleaned_schema_frames.append(
        _schema_frame("hrdataset_v14", cleaned_frames["hrdataset_v14"], "cleaned", inverse_hr_sources)
    )

    profiles = pd.concat(profile_frames, ignore_index=True)
    duplicate = pd.DataFrame(duplicate_rows)
    identifiers = pd.DataFrame(identifier_rows).sort_values(["dataset_key", "schema_stage", "column_name"]).reset_index(drop=True)
    rules = pd.DataFrame(rule_rows)
    target_distribution = pd.DataFrame(target_rows)
    raw_schema = pd.concat(raw_schema_frames, ignore_index=True)
    cleaned_schema = pd.concat(cleaned_schema_frames, ignore_index=True)
    categorical = profiles.loc[profiles["is_categorical"].astype(bool), [
        "dataset_key",
        "column_position",
        "column_name",
        "row_count",
        "nonmissing_count",
        "effective_missing_count",
        "categorical_cardinality",
        "mode_count_nonmissing",
        "mode_share_nonmissing",
        "is_constant_nonmissing",
        "is_near_constant_nonmissing",
    ]].copy()

    summary_rows: list[dict[str, Any]] = []
    manuscript_rows: list[dict[str, Any]] = []
    for dataset_key in DATASET_KEYS:
        raw = raw_frames[dataset_key]
        scoped_profiles = profiles.loc[profiles["dataset_key"] == dataset_key]
        scoped_identifiers = identifiers.loc[identifiers["dataset_key"] == dataset_key]
        scoped_rules = rules.loc[rules["dataset_key"] == dataset_key]
        dup = duplicate.loc[duplicate["dataset_key"] == dataset_key].iloc[0]
        total_missing = int(scoped_profiles["effective_missing_count"].sum())
        anomaly_count = int(scoped_rules["anomaly_count"].sum())
        identifier_failures = int((scoped_identifiers["leakage_audit_status"] != "passed_excluded").sum())
        raw_hash = str(raw_schema.loc[raw_schema["dataset_key"] == dataset_key, "schema_sha256"].iloc[0])
        clean_hash = str(cleaned_schema.loc[cleaned_schema["dataset_key"] == dataset_key, "schema_sha256"].iloc[0])
        summary_rows.append(
            {
                "dataset_key": dataset_key,
                "raw_row_count": int(len(raw)),
                "raw_column_count": int(len(raw.columns)),
                "cleaned_row_count": int(len(cleaned_frames[dataset_key])),
                "cleaned_column_count": int(len(cleaned_frames[dataset_key].columns)),
                "raw_schema_sha256": raw_hash,
                "cleaned_schema_sha256": clean_hash,
                "effective_missing_cell_count": total_missing,
                "effective_missing_cell_percentage": float(total_missing / raw.size * 100.0),
                "columns_with_effective_missing": int((scoped_profiles["effective_missing_count"] > 0).sum()),
                "constant_column_count": int(scoped_profiles["is_constant_nonmissing"].sum()),
                "near_constant_column_count": int(scoped_profiles["is_near_constant_nonmissing"].sum()),
                "categorical_column_count": int(scoped_profiles["is_categorical"].sum()),
                "exact_duplicate_extra_rows": int(dup["exact_duplicate_extra_rows"]),
                "predictor_duplicate_extra_rows": int(dup["predictor_duplicate_extra_rows"]),
                "conflicting_target_groups": int(dup["conflicting_target_groups"]),
                "identifier_candidate_count": int(len(scoped_identifiers)),
                "identifier_candidates_in_primary_features": identifier_failures,
                "declared_rule_count": int(len(scoped_rules)),
                "rules_with_findings": int((scoped_rules["anomaly_count"] > 0).sum()),
                "summed_rule_anomaly_occurrences": anomaly_count,
                "source_rows_modified": 0,
                "audit_status": "findings_require_limitation" if anomaly_count or total_missing else "no_flagged_findings_under_declared_rules",
            }
        )
        if dataset_key == "inx_primary":
            headline = "No missing cells, duplicate rows, identifier leakage, or violations of the declared domain/tenure rules were detected."
        else:
            headline = "215 missing cells (207 termination dates and 8 manager IDs), two review-before-hire rows, two performance text/code mismatches, and two minority department code/text rows were detected; declared identifiers remain excluded."
        manuscript_rows.append(
            {
                "dataset": "INX (primary)" if dataset_key == "inx_primary" else "HRDataset_v14 (independent replication)",
                "rows": int(len(raw)),
                "raw_columns": int(len(raw.columns)),
                "cleaned_columns": int(len(cleaned_frames[dataset_key].columns)),
                "missing_cells_n_percent": f"{total_missing} ({total_missing / raw.size * 100.0:.2f}%)",
                "exact_duplicate_extra_rows": int(dup["exact_duplicate_extra_rows"]),
                "near_constant_columns": int(scoped_profiles["is_near_constant_nonmissing"].sum()),
                "identifier_candidates_in_primary_features": identifier_failures,
                "declared_rules_with_findings": int((scoped_rules["anomaly_count"] > 0).sum()),
                "headline_finding": headline,
                "construct_boundary": "Recorded organizational rating; not established as true capability, objective productivity, or future potential.",
            }
        )

    result = DataQualityV3Result(
        dataset_summary=pd.DataFrame(summary_rows),
        column_profiles=profiles,
        categorical_cardinality=categorical,
        duplicate_audit=duplicate,
        identifier_audit=identifiers,
        rule_anomaly_audit=rules,
        target_distribution=target_distribution,
        raw_schema=raw_schema,
        cleaned_schema=cleaned_schema,
        manuscript_ready_data_quality=pd.DataFrame(manuscript_rows),
    )
    _validate_result(result)
    return result


def _validate_result(result: DataQualityV3Result) -> None:
    _require(len(result.dataset_summary) == 2, "Dataset summary must contain two core datasets.")
    _require(len(result.column_profiles) == 64, "Raw column profile support drifted.")
    _require(len(result.duplicate_audit) == 2, "Duplicate audit support drifted.")
    _require(len(result.raw_schema) == 64, "Raw schema support drifted.")
    _require(len(result.cleaned_schema) == 67, "Cleaned schema support drifted.")
    _require(len(result.manuscript_ready_data_quality) == 2, "Manuscript-ready table support drifted.")
    _require(len(result.target_distribution) == 10, "Target-distribution support drifted.")
    _require(int(result.identifier_audit["primary_model_feature_alias_count"].sum()) == 0, "An identifier candidate entered a primary model feature set.")
    _require(int(result.dataset_summary["source_rows_modified"].sum()) == 0, "The audit must not repair source rows.")
    expected_anomalies = {
        ("hrdataset_v14", "review_on_or_after_hire"): 2,
        ("hrdataset_v14", "performance_text_matches_code"): 2,
        ("hrdataset_v14", "department_id_functionally_maps_to_text"): 2,
    }
    observed_findings = {
        (str(row.dataset_key), str(row.rule_id)): int(row.anomaly_count)
        for row in result.rule_anomaly_audit.itertuples(index=False)
        if int(row.anomaly_count) > 0
    }
    _require(observed_findings == expected_anomalies, f"Declared rule findings drifted: {observed_findings}.")
    inx = result.dataset_summary.loc[result.dataset_summary["dataset_key"] == "inx_primary"].iloc[0]
    hr = result.dataset_summary.loc[result.dataset_summary["dataset_key"] == "hrdataset_v14"].iloc[0]
    _require(int(inx["effective_missing_cell_count"]) == 0, "INX missing-cell result drifted.")
    _require(int(hr["effective_missing_cell_count"]) == 215, "HRDataset missing-cell result drifted.")


def preflight_data_quality_v3(
    *, contract_path: Path | str = DEFAULT_DATA_QUALITY_CONTRACT
) -> dict[str, Any]:
    """Validate the audit contract and source identities without writing output."""

    receipt = validate_data_quality_contract_v3(contract_path)
    return {
        **receipt,
        "planned_dataset_count": 2,
        "planned_raw_column_profiles": 64,
        "model_fit_calls": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def _clean_git_identity() -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DataQualityV3Error(f"Could not establish Git identity: {exc}") from exc
    _require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "Git HEAD is not a full lowercase digest.")
    _require(not status, f"Scientific execution requires a clean worktree: {status.splitlines()[:10]}.")
    return {"commit": commit, "branch": branch}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _run_impl(
    *,
    contract_path: Path,
    output_dir: Path,
    run_id: str,
    offline_state: Any,
) -> dict[str, Any]:
    git_identity = _clean_git_identity()
    receipt = validate_data_quality_contract_v3(contract_path)
    contract_full = contract_path if contract_path.is_absolute() else PROJECT_ROOT / contract_path
    contract = load_config(contract_full)
    _require(not output_dir.exists(), f"Output destination already exists: {output_dir}.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        implementation_paths = (
            Path("src/experiments/data_quality_v3.py"),
            Path("src/governance/data_quality_contract_v3.py"),
            Path("src/data/canonical_loader.py"),
            Path("src/data/external_adapters.py"),
        )
        source_tree = source_tree_hash(PROJECT_ROOT)
        scientific_inputs = {
            "git_identity": git_identity,
            "source_tree_hash": source_tree,
            "contract_sha256": receipt["contract_sha256"],
            "source_hashes": receipt["source_hashes"],
            "dataset_hashes": {
                key: load_canonical_dataset(
                    contract["source_contracts"]["canonical_config"]["path"],
                    key,
                    contract["source_contracts"]["acquisition_manifest"]["path"],
                    allow_download=False,
                ).receipt["actual_sha256"]
                for key in DATASET_KEYS
            },
            "implementation_hashes": {path.as_posix(): sha256_file(PROJECT_ROOT / path) for path in implementation_paths},
        }
        scientific_hash = _canonical_json_sha256(scientific_inputs)
        result = evaluate_data_quality_v3(contract)
        frames = {
            "categorical_cardinality.csv": result.categorical_cardinality,
            "cleaned_schema.csv": result.cleaned_schema,
            "column_profiles.csv": result.column_profiles,
            "dataset_summary.csv": result.dataset_summary,
            "duplicate_audit.csv": result.duplicate_audit,
            "identifier_audit.csv": result.identifier_audit,
            "manuscript_ready_data_quality.csv": result.manuscript_ready_data_quality,
            "raw_schema.csv": result.raw_schema,
            "rule_anomaly_audit.csv": result.rule_anomaly_audit,
            "target_distribution.csv": result.target_distribution,
        }
        for filename, frame in frames.items():
            frame.to_csv(staging / filename, index=False, lineterminator="\n")
        output_hashes = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        metadata = {
            "schema_version": 1,
            "stage": "core_data_quality_v3",
            "status": "complete",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "contract_sha256": receipt["contract_sha256"],
            "scientific_input_sha256": scientific_hash,
            "scientific_inputs": scientific_inputs,
            "git_identity": git_identity,
            "dataset_keys": list(DATASET_KEYS),
            "dataset_count": 2,
            "raw_row_counts": receipt["row_counts"],
            "raw_column_counts": receipt["raw_column_counts"],
            "cleaned_column_counts": receipt["cleaned_column_counts"],
            "raw_column_profile_count": len(result.column_profiles),
            "declared_rule_count": len(result.rule_anomaly_audit),
            "rules_with_findings": int((result.rule_anomaly_audit["anomaly_count"] > 0).sum()),
            "summed_rule_anomaly_occurrences": int(result.rule_anomaly_audit["anomaly_count"].sum()),
            "source_rows_modified": 0,
            "row_level_values_published": False,
            "raw_data_published": False,
            "model_fit_calls": 0,
            "runtime_policy": offline_state.receipt(),
            "network_calls": 0,
            "paid_api_calls": 0,
            "output_hashes": output_hashes,
        }
        _require(_clean_git_identity() == git_identity, "Git identity changed during execution.")
        _require(source_tree_hash(PROJECT_ROOT) == source_tree, "Scientific source tree changed during execution.")
        repeated = validate_data_quality_contract_v3(contract_path)
        _require(repeated["contract_sha256"] == receipt["contract_sha256"], "Data-quality contract changed during execution.")
        _write_json(staging / "stage_metadata.json", metadata)
        _require({path.name for path in staging.iterdir() if path.is_file()} == EXPECTED_LOCAL_FILES, "Phase 3B local output inventory drifted.")
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            for child in staging.iterdir():
                if child.is_file():
                    child.unlink()
            staging.rmdir()
        raise
    return {
        "status": "complete",
        "run_id": run_id,
        "output_dir": output_dir.as_posix(),
        "contract_sha256": receipt["contract_sha256"],
        "scientific_input_sha256": scientific_hash,
        "dataset_count": 2,
        "raw_column_profile_count": len(result.column_profiles),
        "declared_rule_count": len(result.rule_anomaly_audit),
        "rules_with_findings": int((result.rule_anomaly_audit["anomaly_count"] > 0).sum()),
        "model_fit_calls": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def run_data_quality_v3(
    *,
    output_dir: Path | str,
    run_id: str,
    contract_path: Path | str = DEFAULT_DATA_QUALITY_CONTRACT,
) -> dict[str, Any]:
    with enforce_offline_runtime() as offline_state:
        return _run_impl(
            contract_path=Path(contract_path),
            output_dir=Path(output_dir),
            run_id=run_id,
            offline_state=offline_state,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_DATA_QUALITY_CONTRACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_LOCAL_RUN_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    if arguments.preflight_only:
        print(json.dumps(preflight_data_quality_v3(contract_path=arguments.contract), indent=2, sort_keys=True))
        return
    commit = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = arguments.run_id or f"phase3b_v3_{timestamp}_{commit}"
    output_dir = arguments.output_root / run_id / "data_quality"
    print(
        json.dumps(
            run_data_quality_v3(
                output_dir=output_dir,
                run_id=run_id,
                contract_path=arguments.contract,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
