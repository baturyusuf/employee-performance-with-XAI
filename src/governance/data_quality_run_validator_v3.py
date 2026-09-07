"""Independent closed-world validator for a completed Phase 3B quality run.

The validator intentionally does not import the Phase 3B runner. It binds the
generation commit, reloads both canonical core datasets, and independently
reconstructs every published local table from the frozen audit contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
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
from src.governance.offline_runtime import validate_policy_receipt
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_DATA_QUALITY_RUN = Path(
    "reports/major_revision_v3_runs/"
    "phase3b_v3_20260907T154418Z_0d9643b/data_quality"
)
EXPECTED_FILES = frozenset(
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
OUTPUT_HASH_FILES = EXPECTED_FILES - {"stage_metadata.json"}
EXPECTED_IMPLEMENTATIONS = frozenset(
    {
        "src/experiments/data_quality_v3.py",
        "src/governance/data_quality_contract_v3.py",
        "src/data/canonical_loader.py",
        "src/data/external_adapters.py",
    }
)


class DataQualityRunValidationV3Error(RuntimeError):
    """Raised when persisted Phase 3B evidence is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataQualityRunValidationV3Error(message)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataQualityRunValidationV3Error(f"Could not read {path.as_posix()}: {exc}") from exc


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, float_precision="round_trip")
    except Exception as exc:
        raise DataQualityRunValidationV3Error(f"Could not parse {path.name}: {exc}") from exc


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_blob(commit: str, relative_path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DataQualityRunValidationV3Error(
            f"Could not resolve generation blob {commit}:{relative_path}: {exc}"
        ) from exc


def _assert_frame_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_columns: Sequence[str],
    context: str,
    atol: float = 1e-14,
) -> None:
    _require(set(observed.columns) == set(expected.columns), f"{context} schema drifted.")
    columns = list(expected.columns)
    left = observed.loc[:, columns].sort_values(list(sort_columns)).reset_index(drop=True)
    right = expected.loc[:, columns].sort_values(list(sort_columns)).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=atol,
        )
    except AssertionError as exc:
        raise DataQualityRunValidationV3Error(
            f"{context} does not match independent recomputation: {exc}"
        ) from exc


def _string_like(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series.dtype)
        or isinstance(series.dtype, pd.StringDtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _blank(series: pd.Series) -> pd.Series:
    if not _string_like(series):
        return pd.Series(False, index=series.index, dtype=bool)
    return series.astype("string").str.strip().eq("").fillna(False)


def _categorical(series: pd.Series) -> bool:
    return bool(_string_like(series) or pd.api.types.is_bool_dtype(series.dtype))


def _schema_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"position": index, "name": str(column), "pandas_dtype": str(frame[column].dtype)}
        for index, column in enumerate(frame.columns, start=1)
    ]


def _profiles(dataset_key: str, frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, column in enumerate(frame.columns, start=1):
        series = frame[column]
        blanks = _blank(series)
        effective = series.mask(blanks)
        values = effective.dropna()
        unique = int(values.nunique(dropna=True))
        mode_count = int(values.value_counts(dropna=True).max()) if len(values) else 0
        mode_share = float(mode_count / len(values)) if len(values) else float("nan")
        numeric = series if pd.api.types.is_numeric_dtype(series.dtype) else pd.Series(dtype=float)
        rows.append(
            {
                "dataset_key": dataset_key,
                "column_position": index,
                "column_name": str(column),
                "pandas_dtype": str(series.dtype),
                "row_count": len(frame),
                "nonmissing_count": len(values),
                "missing_count": int(series.isna().sum()),
                "blank_string_count": int(blanks.sum()),
                "effective_missing_count": int(series.isna().sum() + blanks.sum()),
                "effective_missing_percentage": float((series.isna().sum() + blanks.sum()) / len(frame) * 100.0),
                "unique_nonmissing_count": unique,
                "mode_count_nonmissing": mode_count,
                "mode_share_nonmissing": mode_share,
                "is_constant_nonmissing": unique == 1,
                "is_near_constant_nonmissing": bool(unique and mode_share >= threshold),
                "is_categorical": _categorical(series),
                "categorical_cardinality": unique if _categorical(series) else np.nan,
                "numeric_minimum": float(numeric.min()) if len(numeric) else np.nan,
                "numeric_maximum": float(numeric.max()) if len(numeric) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _duplicates(dataset_key: str, frame: pd.DataFrame, target: str) -> dict[str, Any]:
    predictors = [column for column in frame.columns if column != target]
    participating = frame.duplicated(subset=predictors, keep=False)
    conflicting_groups = 0
    conflicting_rows = 0
    if bool(participating.any()):
        groups = frame.loc[participating].groupby(predictors, dropna=False, sort=False)[target]
        for _, target_values in groups:
            if target_values.nunique(dropna=False) > 1:
                conflicting_groups += 1
                conflicting_rows += len(target_values)
    exact_extra = int(frame.duplicated(keep="first").sum())
    predictor_extra = int(frame.duplicated(subset=predictors, keep="first").sum())
    return {
        "dataset_key": dataset_key,
        "row_count": len(frame),
        "exact_duplicate_extra_rows": exact_extra,
        "exact_duplicate_participating_rows": int(frame.duplicated(keep=False).sum()),
        "predictor_duplicate_extra_rows": predictor_extra,
        "predictor_duplicate_participating_rows": int(participating.sum()),
        "conflicting_target_groups": conflicting_groups,
        "conflicting_target_rows": conflicting_rows,
        "raw_target_excluded_for_predictor_duplicate_test": target,
        "status": "finding" if exact_extra or predictor_extra or conflicting_groups else "passed",
    }


def _name_signal(column: str, tokens: Sequence[str]) -> bool:
    words = [part.casefold() for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", column)]
    return bool(any(token.casefold() in words for token in tokens) or column.casefold().endswith("id"))


def _sequential(series: pd.Series) -> bool:
    values = series.dropna()
    if len(values) < 2 or values.nunique() != len(values):
        return False
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        suffix = values.astype(str).str.extract(r"(\d+)$", expand=False)
        if suffix.isna().any():
            return False
        numeric = suffix.astype(float)
    return bool(np.array_equal(np.diff(numeric.to_numpy(float)), np.ones(len(numeric) - 1)))


def _primary_feature_sets(
    inx: pd.DataFrame,
    hr_dataset: Any,
    contract: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    availability = load_config(PROJECT_ROOT / contract["source_contracts"]["inx_feature_availability"]["path"])
    p3 = next(policy for policy in availability["policies"] if policy["policy_id"] == "P3")
    return {
        "inx_primary": tuple(column for column in inx.columns if column not in set(p3["excluded_features"])),
        "hrdataset_v14": tuple(build_feature_columns(hr_dataset, "conservative_primary")),
    }


def _identifiers(
    dataset_key: str,
    frame: pd.DataFrame,
    definition: Mapping[str, Any],
    primary_features: Sequence[str],
    tokens: Sequence[str],
    threshold: float,
) -> list[dict[str, Any]]:
    declared = set(definition["declared_identifier_columns"])
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        support = int(series.notna().sum())
        unique_ratio = float(series.nunique(dropna=True) / support) if support else 0.0
        naming = _name_signal(str(column), tokens)
        sequence = _sequential(series)
        candidate = bool(column in declared or sequence or (naming and unique_ratio >= threshold))
        if not candidate:
            continue
        aliases = {str(column)}
        if dataset_key == "hrdataset_v14" and column == "EmpID":
            aliases.add("EmpNumber")
        leaked = aliases.intersection(primary_features)
        rows.append(
            {
                "dataset_key": dataset_key,
                "column_name": str(column),
                "schema_stage": "raw",
                "declared_identifier": column in declared,
                "identifier_name_signal": naming,
                "unique_ratio_nonmissing": unique_ratio,
                "exact_row_order_unit_sequence": sequence,
                "detected_identifier_candidate": candidate,
                "primary_model_feature_alias_count": len(leaked),
                "primary_model_feature_status": "included" if leaked else "excluded",
                "leakage_audit_status": "failed_identifier_in_primary_features" if leaked else "passed_excluded",
                "interpretation": "identifier_candidate_requires_removal" if leaked else "identifier_or_name_field_not_a_model_predictor",
            }
        )
    if dataset_key == "hrdataset_v14":
        leaked = "ExternalSampleId" in primary_features
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
                "primary_model_feature_alias_count": int(leaked),
                "primary_model_feature_status": "included" if leaked else "excluded",
                "leakage_audit_status": "failed_identifier_in_primary_features" if leaked else "passed_excluded",
                "interpretation": "generated_row_key_requires_removal" if leaked else "generated_row_key_not_a_model_predictor",
            }
        )
    return rows


def _rule(
    dataset_key: str,
    rule_id: str,
    rule_type: str,
    columns: str,
    checked: int,
    skipped: int,
    anomalies: int,
    expectation: str,
) -> dict[str, Any]:
    return {
        "dataset_key": dataset_key,
        "rule_id": rule_id,
        "rule_type": rule_type,
        "columns": columns,
        "checked_count": checked,
        "missing_or_skipped_count": skipped,
        "anomaly_count": anomalies,
        "anomaly_percentage_of_checked": float(anomalies / checked * 100.0) if checked else 0.0,
        "status": "finding" if anomalies else "passed",
        "expectation": expectation,
    }


def _relation(left: pd.Series, operator: str, right: pd.Series) -> pd.Series:
    return {"<": left < right, "<=": left <= right, ">=": left >= right, ">": left > right}[operator]


def _numeric_and_relational_rules(
    dataset_key: str,
    frame: pd.DataFrame,
    definition: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in definition["numeric_rules"]:
        column = str(spec["column"])
        series = pd.to_numeric(frame[column], errors="coerce")
        available = series.notna()
        if "allowed" in spec:
            anomalies = available & (~series.isin([float(value) for value in spec["allowed"]]))
            expectation = "allowed_values_contract"
        else:
            anomalies = available & ((series < float(spec["minimum"])) | (series > float(spec["maximum"])))
            expectation = f"inclusive_range_{spec['minimum']}_to_{spec['maximum']}"
        rows.append(_rule(dataset_key, f"numeric_{column}", "numeric_domain", column, int(available.sum()), int((~available).sum()), int(anomalies.sum()), expectation))
    if dataset_key == "inx_primary":
        for spec in definition["relational_rules"]:
            left = pd.to_numeric(frame[str(spec["left"])], errors="coerce")
            right = pd.to_numeric(frame[str(spec["right"])], errors="coerce")
            available = left.notna() & right.notna()
            anomalies = available & (~_relation(left, str(spec["operator"]), right))
            rows.append(_rule(dataset_key, str(spec["rule_id"]), "numeric_relation", f"{spec['left']} {spec['operator']} {spec['right']}", int(available.sum()), int((~available).sum()), int(anomalies.sum()), "declared_cross_field_relation"))
    return rows


def _hr_special_rules(frame: pd.DataFrame, definition: Mapping[str, Any]) -> list[dict[str, Any]]:
    parsed = {
        column: pd.to_datetime(frame[column], format=str(spec["format"]), errors="coerce")
        for column, spec in definition["date_rules"].items()
    }
    latest = pd.concat(
        [parsed["DateofHire"], parsed["DateofTermination"], parsed["LastPerformanceReview_Date"]],
        ignore_index=True,
    ).max()
    parsed["DOB"] = parsed["DOB"].where(parsed["DOB"] <= latest, parsed["DOB"] - pd.DateOffset(years=100))
    rows: list[dict[str, Any]] = []
    for column, spec in definition["date_rules"].items():
        supplied = frame[column].notna() & (~_blank(frame[column]))
        invalid = supplied & parsed[column].isna()
        expectation = str(spec["format"]) + ("_with_declared_two_digit_year_normalization" if column == "DOB" else "")
        rows.append(_rule("hrdataset_v14", f"date_parse_{column}", "date_parse", column, int(supplied.sum()), int((~supplied).sum()), int(invalid.sum()), expectation))
    for spec in definition["relational_rules"]:
        left = parsed[str(spec["left"])]
        right = parsed[str(spec["right"])]
        available = left.notna() & right.notna()
        anomalies = available & (~_relation(left, str(spec["operator"]), right))
        rows.append(_rule("hrdataset_v14", str(spec["rule_id"]), "date_relation", f"{spec['left']} {spec['operator']} {spec['right']}", int(available.sum()), int((~available).sum()), int(anomalies.sum()), "declared_temporal_relation"))
    for spec in definition["consistency_rules"]:
        kind = str(spec["type"])
        rule_id = str(spec["rule_id"])
        if kind in {"text_code_mapping", "normalized_text_code_mapping"}:
            text_column, code_column = str(spec["text_column"]), str(spec["code_column"])
            expected = frame[text_column].astype("string").str.strip().map(spec["mapping"])
            observed = pd.to_numeric(frame[code_column], errors="coerce")
            available = expected.notna() & observed.notna()
            anomalies = available & (expected.astype("Float64") != observed.astype("Float64"))
            columns = f"{text_column}->{code_column}"
            expectation = "declared_text_code_correspondence"
        elif kind == "boolean_text_mapping":
            text_column, code_column = str(spec["text_column"]), str(spec["code_column"])
            text = frame[text_column].astype("string").str.strip()
            expected = pd.Series(np.where(text.eq(str(spec["positive_text"])), int(spec["positive_code"]), int(spec["negative_code"])), index=frame.index)
            observed = pd.to_numeric(frame[code_column], errors="coerce")
            available = text.notna() & observed.notna()
            anomalies = available & (expected != observed)
            columns = f"{text_column}->{code_column}"
            expectation = "positive_text_flag_correspondence"
        elif kind == "termination_flag_date":
            observed = pd.to_numeric(frame["Termd"], errors="coerce")
            available = observed.notna()
            anomalies = available & (((observed == 1) & parsed["DateofTermination"].isna()) | ((observed == 0) & parsed["DateofTermination"].notna()))
            columns = "Termd<->DateofTermination"
            expectation = "terminated_iff_termination_date_present"
        elif kind == "code_to_text_function":
            code_column, text_column = str(spec["code_column"]), str(spec["text_column"])
            complete = frame[[code_column, text_column]].dropna()
            anomalies_count = sum(
                len(group) - int(group[text_column].astype("string").str.strip().value_counts().max())
                for _, group in complete.groupby(code_column, sort=False)
            )
            rows.append(_rule("hrdataset_v14", rule_id, kind, f"{code_column}->{text_column}", len(complete), len(frame) - len(complete), int(anomalies_count), "each_code_has_one_text_label_minority_rows_counted_as_findings"))
            continue
        else:  # pragma: no cover - contract validation prevents this
            raise DataQualityRunValidationV3Error(f"Unsupported consistency rule: {kind}.")
        rows.append(_rule("hrdataset_v14", rule_id, kind, columns, int(available.sum()), int((~available).sum()), int(anomalies.sum()), expectation))
    return rows


def _schema_frame(
    dataset_key: str,
    frame: pd.DataFrame,
    stage: str,
    source_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    schema = _schema_rows(frame)
    digest = _canonical_hash(schema)
    source_map = source_map or {}
    rows: list[dict[str, Any]] = []
    for record in schema:
        column = str(record["name"])
        source = source_map.get(column, column)
        if stage == "cleaned" and column == "ExternalSampleId":
            transformation = "generated_zero_based_row_key_excluded_from_models"
        elif stage == "cleaned" and dataset_key == "hrdataset_v14" and column == "ExperienceYearsAtThisCompany":
            transformation = "derived_review_minus_hire_years_negative_values_set_missing"
        elif stage == "cleaned" and dataset_key == "hrdataset_v14" and column == "PerformanceRating":
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


def _targets(dataset_key: str, view: str, column: str, series: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, count in series.value_counts(dropna=False, sort=False).items():
        if pd.isna(value):
            label = "<NA>"
        elif isinstance(value, (float, np.floating)) and float(value).is_integer():
            label = str(int(value))
        else:
            label = str(value)
        rows.append(
            {
                "dataset_key": dataset_key,
                "target_view": view,
                "target_column": column,
                "target_label": label,
                "count": int(count),
                "percentage": float(count / len(series) * 100.0),
            }
        )
    return rows


def _independent_tables(contract: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
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
    raw = {key: verified[key].frame.copy(deep=True) for key in DATASET_KEYS}
    hr_dataset = load_external_dataset(
        "hrdataset_v14",
        raw_frame=raw["hrdataset_v14"],
        schema_mapping_path=PROJECT_ROOT / sources["hrdataset_schema_mapping"]["path"],
    )
    cleaned = {"inx_primary": raw["inx_primary"].copy(deep=True), "hrdataset_v14": hr_dataset.canonical.copy(deep=True)}
    features = _primary_feature_sets(raw["inx_primary"], hr_dataset, contract)
    policy = contract["audit_policy"]
    profiles = pd.concat(
        [_profiles(key, raw[key], float(policy["near_constant_nonmissing_mode_share_threshold"])) for key in DATASET_KEYS],
        ignore_index=True,
    )
    duplicate = pd.DataFrame(
        [_duplicates(key, raw[key], str(contract["datasets"][key]["raw_target"])) for key in DATASET_KEYS]
    )
    identifier_rows: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    raw_schemas: list[pd.DataFrame] = []
    for key in DATASET_KEYS:
        definition = contract["datasets"][key]
        identifier_rows.extend(
            _identifiers(
                key,
                raw[key],
                definition,
                features[key],
                policy["identifier_name_tokens"],
                float(policy["identifier_unique_ratio_threshold"]),
            )
        )
        rule_rows.extend(_numeric_and_relational_rules(key, raw[key], definition))
        if key == "hrdataset_v14":
            rule_rows.extend(_hr_special_rules(raw[key], definition))
        target = str(definition["raw_target"])
        target_rows.extend(_targets(key, "raw", target, raw[key][target]))
        raw_schemas.append(_schema_frame(key, raw[key], "raw"))
    target_rows.extend(_targets("hrdataset_v14", "canonical_mapped", "PerformanceRating", cleaned["hrdataset_v14"]["PerformanceRating"]))
    mapping = load_config(PROJECT_ROOT / sources["hrdataset_schema_mapping"]["path"])
    source_map = {str(target): str(source) for source, target in mapping["rename_columns"].items()}
    source_map.update({"ExternalSampleId": "<generated>", "ExperienceYearsAtThisCompany": "DateofHire+LastPerformanceReview_Date", "PerformanceRating": "PerformanceScore"})
    raw_schema = pd.concat(raw_schemas, ignore_index=True)
    cleaned_schema = pd.concat(
        [
            _schema_frame("inx_primary", cleaned["inx_primary"], "cleaned"),
            _schema_frame("hrdataset_v14", cleaned["hrdataset_v14"], "cleaned", source_map),
        ],
        ignore_index=True,
    )
    identifiers = pd.DataFrame(identifier_rows).sort_values(["dataset_key", "schema_stage", "column_name"]).reset_index(drop=True)
    rules = pd.DataFrame(rule_rows)
    categorical = profiles.loc[profiles["is_categorical"].astype(bool), [
        "dataset_key", "column_position", "column_name", "row_count", "nonmissing_count",
        "effective_missing_count", "categorical_cardinality", "mode_count_nonmissing",
        "mode_share_nonmissing", "is_constant_nonmissing", "is_near_constant_nonmissing",
    ]].copy()

    summaries: list[dict[str, Any]] = []
    manuscript: list[dict[str, Any]] = []
    for key in DATASET_KEYS:
        scoped_profiles = profiles.loc[profiles["dataset_key"] == key]
        scoped_ids = identifiers.loc[identifiers["dataset_key"] == key]
        scoped_rules = rules.loc[rules["dataset_key"] == key]
        dup = duplicate.loc[duplicate["dataset_key"] == key].iloc[0]
        missing = int(scoped_profiles["effective_missing_count"].sum())
        anomalies = int(scoped_rules["anomaly_count"].sum())
        identifier_failures = int((scoped_ids["leakage_audit_status"] != "passed_excluded").sum())
        raw_hash = str(raw_schema.loc[raw_schema["dataset_key"] == key, "schema_sha256"].iloc[0])
        cleaned_hash = str(cleaned_schema.loc[cleaned_schema["dataset_key"] == key, "schema_sha256"].iloc[0])
        summaries.append(
            {
                "dataset_key": key,
                "raw_row_count": len(raw[key]),
                "raw_column_count": len(raw[key].columns),
                "cleaned_row_count": len(cleaned[key]),
                "cleaned_column_count": len(cleaned[key].columns),
                "raw_schema_sha256": raw_hash,
                "cleaned_schema_sha256": cleaned_hash,
                "effective_missing_cell_count": missing,
                "effective_missing_cell_percentage": float(missing / raw[key].size * 100.0),
                "columns_with_effective_missing": int((scoped_profiles["effective_missing_count"] > 0).sum()),
                "constant_column_count": int(scoped_profiles["is_constant_nonmissing"].sum()),
                "near_constant_column_count": int(scoped_profiles["is_near_constant_nonmissing"].sum()),
                "categorical_column_count": int(scoped_profiles["is_categorical"].sum()),
                "exact_duplicate_extra_rows": int(dup["exact_duplicate_extra_rows"]),
                "predictor_duplicate_extra_rows": int(dup["predictor_duplicate_extra_rows"]),
                "conflicting_target_groups": int(dup["conflicting_target_groups"]),
                "identifier_candidate_count": len(scoped_ids),
                "identifier_candidates_in_primary_features": identifier_failures,
                "declared_rule_count": len(scoped_rules),
                "rules_with_findings": int((scoped_rules["anomaly_count"] > 0).sum()),
                "summed_rule_anomaly_occurrences": anomalies,
                "source_rows_modified": 0,
                "audit_status": "findings_require_limitation" if anomalies or missing else "no_flagged_findings_under_declared_rules",
            }
        )
        headline = (
            "No missing cells, duplicate rows, identifier leakage, or violations of the declared domain/tenure rules were detected."
            if key == "inx_primary"
            else "215 missing cells (207 termination dates and 8 manager IDs), two review-before-hire rows, two performance text/code mismatches, and two minority department code/text rows were detected; declared identifiers remain excluded."
        )
        manuscript.append(
            {
                "dataset": "INX (primary)" if key == "inx_primary" else "HRDataset_v14 (independent replication)",
                "rows": len(raw[key]),
                "raw_columns": len(raw[key].columns),
                "cleaned_columns": len(cleaned[key].columns),
                "missing_cells_n_percent": f"{missing} ({missing / raw[key].size * 100.0:.2f}%)",
                "exact_duplicate_extra_rows": int(dup["exact_duplicate_extra_rows"]),
                "near_constant_columns": int(scoped_profiles["is_near_constant_nonmissing"].sum()),
                "identifier_candidates_in_primary_features": identifier_failures,
                "declared_rules_with_findings": int((scoped_rules["anomaly_count"] > 0).sum()),
                "headline_finding": headline,
                "construct_boundary": "Recorded organizational rating; not established as true capability, objective productivity, or future potential.",
            }
        )
    return {
        "categorical_cardinality.csv": categorical,
        "cleaned_schema.csv": cleaned_schema,
        "column_profiles.csv": profiles,
        "dataset_summary.csv": pd.DataFrame(summaries),
        "duplicate_audit.csv": duplicate,
        "identifier_audit.csv": identifiers,
        "manuscript_ready_data_quality.csv": pd.DataFrame(manuscript),
        "raw_schema.csv": raw_schema,
        "rule_anomaly_audit.csv": rules,
        "target_distribution.csv": pd.DataFrame(target_rows),
    }


def _validate_metadata(run_dir: Path, metadata: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    _require(metadata.get("schema_version") == 1, "Metadata schema version drifted.")
    _require(metadata.get("stage") == "core_data_quality_v3", "Metadata stage drifted.")
    _require(metadata.get("status") == "complete", "Data-quality run is incomplete.")
    run_id = str(metadata.get("run_id", ""))
    _require(run_id == run_dir.parent.name, "Run id/path identity drifted.")
    git_identity = metadata.get("git_identity")
    _require(isinstance(git_identity, Mapping), "Git identity is absent.")
    commit = str(git_identity.get("commit", ""))
    _require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "Generation commit is invalid.")
    _require(run_id.endswith(commit[:7]), "Run id does not bind the generation commit.")
    _require(git_identity.get("branch") == "finalization/leakage-aware-v2", "Generation branch drifted.")
    _require(metadata.get("contract_sha256") == receipt["contract_sha256"], "Metadata contract hash drifted.")
    _require(metadata.get("dataset_keys") == list(DATASET_KEYS), "Metadata dataset scope drifted.")
    _require(metadata.get("dataset_count") == 2, "Metadata dataset count drifted.")
    _require(metadata.get("raw_row_counts") == receipt["row_counts"], "Metadata row counts drifted.")
    _require(metadata.get("raw_column_counts") == receipt["raw_column_counts"], "Metadata raw column counts drifted.")
    _require(metadata.get("cleaned_column_counts") == receipt["cleaned_column_counts"], "Metadata cleaned column counts drifted.")
    for key, expected in {
        "raw_column_profile_count": 64,
        "declared_rule_count": 52,
        "rules_with_findings": 3,
        "summed_rule_anomaly_occurrences": 6,
        "source_rows_modified": 0,
        "model_fit_calls": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
    }.items():
        _require(metadata.get(key) == expected, f"Metadata {key} drifted.")
    _require(metadata.get("row_level_values_published") is False, "Row-level publication flag drifted.")
    _require(metadata.get("raw_data_published") is False, "Raw-data publication flag drifted.")
    validate_policy_receipt(metadata.get("runtime_policy", {}))

    scientific_inputs = metadata.get("scientific_inputs")
    _require(isinstance(scientific_inputs, Mapping), "Scientific-input receipt is absent.")
    _require(metadata.get("scientific_input_sha256") == _canonical_hash(scientific_inputs), "Scientific-input hash drifted.")
    _require(scientific_inputs.get("contract_sha256") == receipt["contract_sha256"], "Scientific contract binding drifted.")
    _require(scientific_inputs.get("source_hashes") == receipt["source_hashes"], "Scientific source bindings drifted.")
    _require(scientific_inputs.get("git_identity") == git_identity, "Scientific Git identity drifted.")
    source_tree = str(scientific_inputs.get("source_tree_hash", ""))
    _require(bool(re.fullmatch(r"[0-9a-f]{64}", source_tree)), "Source-tree receipt is invalid.")
    expected_dataset_hashes = {
        "inx_primary": "b8deac0a615b97076622ae540f4cfd0d3c3f1e7acb83ba3ff6560470a9ccf60a",
        "hrdataset_v14": "cb19996755c93c0a8d6527f59da4701c80aef65eff854906546dce286249813c",
    }
    _require(scientific_inputs.get("dataset_hashes") == expected_dataset_hashes, "Scientific dataset bindings drifted.")
    implementations = scientific_inputs.get("implementation_hashes")
    _require(isinstance(implementations, Mapping) and set(implementations) == EXPECTED_IMPLEMENTATIONS, "Implementation inventory drifted.")
    for path, digest in implementations.items():
        _require(hashlib.sha256(_git_blob(commit, str(path))).hexdigest() == digest, f"Generation implementation blob drifted: {path}.")
    output_hashes = metadata.get("output_hashes")
    _require(isinstance(output_hashes, Mapping) and set(output_hashes) == OUTPUT_HASH_FILES, "Output-hash inventory drifted.")
    for name, digest in output_hashes.items():
        _require(sha256_file(run_dir / str(name)) == digest, f"Output hash drifted: {name}.")


def validate_data_quality_run_v3(
    run_dir: Path | str = DEFAULT_DATA_QUALITY_RUN,
    *,
    contract_path: Path | str = DEFAULT_DATA_QUALITY_CONTRACT,
) -> dict[str, Any]:
    """Validate a Phase 3B run and independently reconstruct all ten tables."""

    directory = Path(run_dir)
    _require(directory.is_dir(), f"Data-quality run directory is absent: {directory}.")
    inventory = {path.name for path in directory.iterdir() if path.is_file()}
    _require(inventory == EXPECTED_FILES, f"Data-quality closed-world inventory drifted: {sorted(inventory ^ EXPECTED_FILES)}.")
    receipt = validate_data_quality_contract_v3(contract_path)
    metadata = _load_json(directory / "stage_metadata.json")
    _require(isinstance(metadata, Mapping), "Stage metadata must contain an object.")
    _validate_metadata(directory, metadata, receipt)
    contract_full = Path(contract_path)
    if not contract_full.is_absolute():
        contract_full = PROJECT_ROOT / contract_full
    expected = _independent_tables(load_config(contract_full))
    sort_keys = {
        "categorical_cardinality.csv": ["dataset_key", "column_position"],
        "cleaned_schema.csv": ["dataset_key", "column_position"],
        "column_profiles.csv": ["dataset_key", "column_position"],
        "dataset_summary.csv": ["dataset_key"],
        "duplicate_audit.csv": ["dataset_key"],
        "identifier_audit.csv": ["dataset_key", "schema_stage", "column_name"],
        "manuscript_ready_data_quality.csv": ["dataset"],
        "raw_schema.csv": ["dataset_key", "column_position"],
        "rule_anomaly_audit.csv": ["dataset_key", "rule_id"],
        "target_distribution.csv": ["dataset_key", "target_view", "target_label"],
    }
    for name, expected_frame in expected.items():
        _assert_frame_equal(
            _read_csv(directory / name),
            expected_frame,
            sort_columns=sort_keys[name],
            context=name,
        )
    findings = expected["rule_anomaly_audit.csv"].loc[
        expected["rule_anomaly_audit.csv"]["anomaly_count"] > 0,
        ["dataset_key", "rule_id", "anomaly_count"],
    ].to_dict("records")
    return {
        "status": "passed",
        "run_id": metadata["run_id"],
        "generation_commit": metadata["git_identity"]["commit"],
        "contract_sha256": metadata["contract_sha256"],
        "scientific_input_sha256": metadata["scientific_input_sha256"],
        "file_count": len(EXPECTED_FILES),
        "table_count": len(expected),
        "dataset_count": 2,
        "raw_column_profile_count": len(expected["column_profiles.csv"]),
        "raw_schema_row_count": len(expected["raw_schema.csv"]),
        "cleaned_schema_row_count": len(expected["cleaned_schema.csv"]),
        "declared_rule_count": len(expected["rule_anomaly_audit.csv"]),
        "findings": findings,
        "identifier_candidates_in_primary_features": int(expected["dataset_summary.csv"]["identifier_candidates_in_primary_features"].sum()),
        "source_rows_modified": 0,
        "model_fit_calls": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_DATA_QUALITY_RUN)
    parser.add_argument("--contract", type=Path, default=DEFAULT_DATA_QUALITY_CONTRACT)
    arguments = parser.parse_args()
    print(
        json.dumps(
            validate_data_quality_run_v3(arguments.run_dir, contract_path=arguments.contract),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
