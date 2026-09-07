"""Fail-closed contract for the Phase 3B core-dataset quality audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.data.canonical_loader import load_canonical_dataset
from src.data.external_adapters import build_feature_columns, load_external_dataset
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_DATA_QUALITY_CONTRACT = Path("configs/data_quality_v3.json")
DATASET_KEYS = ("inx_primary", "hrdataset_v14")
EXPECTED_SOURCE_KEYS = {
    "canonical_config",
    "acquisition_manifest",
    "inx_feature_availability",
    "hrdataset_schema_mapping",
}
EXPECTED_TOP_LEVEL = {
    "schema_version",
    "contract_id",
    "scope",
    "source_contracts",
    "audit_policy",
    "datasets",
    "construct_boundary",
    "claim_boundary",
    "publication",
}


class DataQualityContractV3Error(RuntimeError):
    """Raised when the Phase 3B audit contract is incomplete or has drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataQualityContractV3Error(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataQualityContractV3Error(f"Could not read {path.as_posix()}: {exc}") from exc
    _require(isinstance(payload, dict), f"{path.name} must contain a JSON object.")
    return payload


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object.")
    return value


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DataQualityContractV3Error(f"Could not hash {path.as_posix()}: {exc}") from exc


def _validate_sources(contract: Mapping[str, Any]) -> dict[str, str]:
    sources = _mapping(contract.get("source_contracts"), "source_contracts")
    _require(set(sources) == EXPECTED_SOURCE_KEYS, "Data-quality source inventory drifted.")
    hashes: dict[str, str] = {}
    for name, raw_record in sources.items():
        record = _mapping(raw_record, f"source_contracts.{name}")
        _require(set(record) == {"path", "sha256"}, f"Source record drifted for {name}.")
        source_path = PROJECT_ROOT / str(record.get("path", ""))
        _require(source_path.is_file(), f"Bound data-quality source is absent: {name}.")
        actual = _sha256(source_path)
        _require(actual == record.get("sha256"), f"Bound source hash drifted: {name}.")
        hashes[name] = actual
    return hashes


def _validate_scope(contract: Mapping[str, Any]) -> None:
    scope = _mapping(contract.get("scope"), "scope")
    _require(scope.get("canonical_evidence_scope") == "core", "Audit scope must remain core.")
    _require(tuple(scope.get("dataset_keys", ())) == DATASET_KEYS, "Core dataset order or identity drifted.")
    _require(scope.get("supplementary_datasets_included") is False, "Supplementary datasets cannot be represented as core.")
    canonical = load_config(PROJECT_ROOT / contract["source_contracts"]["canonical_config"]["path"])
    declared = canonical.get("manuscript_final", {}).get("evidence_scopes", {}).get("core", {}).get("dataset_keys", [])
    _require(tuple(declared) == DATASET_KEYS, "Canonical core evidence-scope membership drifted.")


def _validate_policy(contract: Mapping[str, Any]) -> None:
    policy = _mapping(contract.get("audit_policy"), "audit_policy")
    exact = {
        "blank_string_policy": "strip_whitespace_then_count_empty_as_missing_for_audit_only",
        "missing_percentage_denominator": "all_rows",
        "near_constant_nonmissing_mode_share_threshold": 0.99,
        "identifier_unique_ratio_threshold": 0.98,
        "identifier_name_tokens": ["id", "identifier", "name", "number"],
        "identifier_leakage_test": "every_declared_or_detected_identifier_must_be_absent_from_the_declared_primary_model_feature_set",
        "schema_hash_definition": "sha256_of_canonical_json_ordered_position_name_pandas_dtype_records",
        "categorical_profile_definition": "object_string_category_or_boolean_columns_only",
        "raw_values_or_row_identifiers_may_be_published": False,
        "automatic_download_allowed": False,
        "network_calls_allowed": False,
        "paid_api_calls_allowed": False,
    }
    for key, expected in exact.items():
        _require(policy.get(key) == expected, f"audit_policy.{key} drifted.")
    duplicates = _mapping(policy.get("duplicate_definitions"), "audit_policy.duplicate_definitions")
    _require(
        set(duplicates)
        == {
            "exact_duplicate_extra_rows",
            "exact_duplicate_participating_rows",
            "predictor_duplicate_extra_rows",
            "conflicting_target_groups",
        },
        "Duplicate-definition inventory drifted.",
    )


def _validate_rule_inventory(dataset_key: str, definition: Mapping[str, Any], raw_columns: set[str]) -> None:
    target = definition.get("raw_target")
    _require(isinstance(target, str) and target in raw_columns, f"{dataset_key} raw target is absent.")
    identifiers = definition.get("declared_identifier_columns")
    _require(isinstance(identifiers, list) and identifiers, f"{dataset_key} identifier registry is absent.")
    _require(set(identifiers).issubset(raw_columns), f"{dataset_key} declares unknown identifier columns.")
    numeric_rules = definition.get("numeric_rules")
    _require(isinstance(numeric_rules, list) and numeric_rules, f"{dataset_key} numeric rules are absent.")
    seen: set[str] = set()
    for raw_rule in numeric_rules:
        rule = _mapping(raw_rule, f"{dataset_key}.numeric_rule")
        column = str(rule.get("column", ""))
        _require(column in raw_columns and column not in seen, f"{dataset_key} numeric rule column drifted: {column}.")
        seen.add(column)
        has_allowed = isinstance(rule.get("allowed"), list) and bool(rule["allowed"])
        has_range = "minimum" in rule or "maximum" in rule
        _require(has_allowed != has_range, f"{dataset_key}.{column} must define allowed values xor a range.")
        if has_range:
            _require("minimum" in rule and "maximum" in rule, f"{dataset_key}.{column} range is incomplete.")
            _require(float(rule["minimum"]) <= float(rule["maximum"]), f"{dataset_key}.{column} range is reversed.")
    relations = definition.get("relational_rules")
    _require(isinstance(relations, list) and relations, f"{dataset_key} relational rules are absent.")
    relation_ids: set[str] = set()
    for raw_rule in relations:
        rule = _mapping(raw_rule, f"{dataset_key}.relational_rule")
        rule_id = str(rule.get("rule_id", ""))
        _require(rule_id and rule_id not in relation_ids, f"{dataset_key} relational rule identity drifted.")
        relation_ids.add(rule_id)
        _require(rule.get("left") in raw_columns and rule.get("right") in raw_columns, f"{dataset_key}.{rule_id} references unknown columns.")
        _require(rule.get("operator") in {"<", "<=", ">=", ">"}, f"{dataset_key}.{rule_id} has an unsupported operator.")


def _validate_datasets(contract: Mapping[str, Any]) -> dict[str, Any]:
    definitions = _mapping(contract.get("datasets"), "datasets")
    _require(tuple(definitions) == DATASET_KEYS, "Dataset-definition order or identity drifted.")
    loaded = {
        key: load_canonical_dataset(
            contract["source_contracts"]["canonical_config"]["path"],
            key,
            contract["source_contracts"]["acquisition_manifest"]["path"],
            allow_download=False,
        )
        for key in DATASET_KEYS
    }
    for key in DATASET_KEYS:
        _validate_rule_inventory(key, _mapping(definitions[key], f"datasets.{key}"), set(loaded[key].frame.columns))

    inx = _mapping(definitions["inx_primary"], "datasets.inx_primary")
    _require(inx.get("primary_feature_policy") == "P3", "INX primary feature policy drifted.")
    _require(inx.get("cleaned_view") == "canonical_loader_verified_raw_frame", "INX cleaned-view definition drifted.")
    feature_contract = load_config(PROJECT_ROOT / contract["source_contracts"]["inx_feature_availability"]["path"])
    policies = feature_contract.get("policies", feature_contract.get("feature_availability", {}).get("policies", []))
    p3 = next((item for item in policies if item.get("policy_id") == "P3"), None)
    _require(isinstance(p3, Mapping), "INX P3 feature policy is absent.")
    inx_primary_features = [column for column in loaded["inx_primary"].frame.columns if column not in set(p3["excluded_features"])]
    _require("EmpNumber" not in inx_primary_features, "INX identifier leaked into P3 features.")

    hr_definition = _mapping(definitions["hrdataset_v14"], "datasets.hrdataset_v14")
    _require(hr_definition.get("primary_feature_policy") == "conservative_primary", "HRDataset primary feature policy drifted.")
    _require(hr_definition.get("cleaned_view") == "external_adapter_canonical_frame", "HRDataset cleaned-view definition drifted.")
    date_rules = _mapping(hr_definition.get("date_rules"), "hrdataset_v14.date_rules")
    _require(tuple(date_rules) == ("DOB", "DateofHire", "DateofTermination", "LastPerformanceReview_Date"), "HRDataset date-rule inventory drifted.")
    consistency = hr_definition.get("consistency_rules")
    _require(isinstance(consistency, list) and len(consistency) == 5, "HRDataset consistency-rule inventory drifted.")
    consistency_ids = [str(item.get("rule_id", "")) for item in consistency if isinstance(item, Mapping)]
    _require(len(consistency_ids) == len(set(consistency_ids)) == 5, "HRDataset consistency-rule identities drifted.")
    adapted = load_external_dataset(
        "hrdataset_v14",
        raw_frame=loaded["hrdataset_v14"].frame,
        schema_mapping_path=PROJECT_ROOT / contract["source_contracts"]["hrdataset_schema_mapping"]["path"],
    )
    hr_primary_features = build_feature_columns(adapted, "conservative_primary")
    canonical_identifier_aliases = {"Employee_Name", "EmpNumber", "ManagerName", "ManagerID", "ExternalSampleId"}
    _require(canonical_identifier_aliases.isdisjoint(hr_primary_features), "HRDataset identifier leaked into primary features.")
    return {
        "row_counts": {key: int(len(value.frame)) for key, value in loaded.items()},
        "raw_column_counts": {key: int(len(value.frame.columns)) for key, value in loaded.items()},
        "cleaned_column_counts": {"inx_primary": 28, "hrdataset_v14": int(len(adapted.canonical.columns))},
        "primary_feature_counts": {"inx_primary": len(inx_primary_features), "hrdataset_v14": len(hr_primary_features)},
    }


def _validate_boundaries(contract: Mapping[str, Any]) -> None:
    construct = _mapping(contract.get("construct_boundary"), "construct_boundary")
    _require(construct.get("target_construct") == "recorded_organizational_performance_rating", "Target construct drifted.")
    _require(
        construct.get("not_established_as") == ["true_employee_capability", "objective_productivity", "future_potential"],
        "Construct-validity exclusions drifted.",
    )
    _require(bool(str(construct.get("required_limitation", "")).strip()), "Construct limitation is absent.")
    claims = _mapping(contract.get("claim_boundary"), "claim_boundary")
    _require(claims and all(value is False for value in claims.values()), "Every data-quality claim boundary must remain false.")
    publication = _mapping(contract.get("publication"), "publication")
    _require(publication.get("required_report") == "DATA_QUALITY_REPORT.md", "Required report name drifted.")
    _require(publication.get("required_manuscript_table") == "manuscript_ready_data_quality.csv", "Required manuscript table drifted.")
    _require(publication.get("publish_row_level_values") is False, "Row-level publication is prohibited.")
    _require(publication.get("publish_raw_data") is False, "Raw-data publication is prohibited.")
    _require(publication.get("publish_only_aggregate_profiles_and_schemas") is True, "Aggregate-only publication is required.")


def validate_data_quality_contract_v3(
    contract_path: Path | str = DEFAULT_DATA_QUALITY_CONTRACT,
) -> dict[str, Any]:
    """Validate source identity, core scope, audit rules, and publication limits."""

    path = Path(contract_path)
    full = path if path.is_absolute() else PROJECT_ROOT / path
    contract = _load_json(full)
    _require(set(contract) == EXPECTED_TOP_LEVEL, "Data-quality top-level inventory drifted.")
    _require(contract.get("schema_version") == 1, "Data-quality schema_version drifted.")
    _require(contract.get("contract_id") == "core_dataset_data_quality_v3", "Data-quality contract_id drifted.")
    source_hashes = _validate_sources(contract)
    _validate_scope(contract)
    _validate_policy(contract)
    dataset_receipt = _validate_datasets(contract)
    _validate_boundaries(contract)
    return {
        "status": "passed",
        "contract_path": full.relative_to(PROJECT_ROOT).as_posix(),
        "contract_sha256": _sha256(full),
        "source_hashes": source_hashes,
        "dataset_keys": list(DATASET_KEYS),
        "dataset_count": len(DATASET_KEYS),
        **dataset_receipt,
        "model_fit_calls": 0,
        "network_calls": 0,
        "paid_api_calls": 0,
    }


__all__ = [
    "DATASET_KEYS",
    "DEFAULT_DATA_QUALITY_CONTRACT",
    "DataQualityContractV3Error",
    "validate_data_quality_contract_v3",
]
