"""Build the canonical HRDataset_v14 mapped-target replication evidence.

This module is the only core-paper orchestrator for HRDataset_v14.  It binds
the exact bytes consumed by the canonical loader to the scoped run manifest,
uses the frozen conservative feature contract, executes the nested out-of-fold
engine, derives exact-model diagnostics, and publishes one staged directory
only after late input revalidation.  Historical external-report directories
and the legacy external runner are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.canonical_loader import CanonicalDataset, load_canonical_dataset
from src.data.external_adapters import (
    ExternalDataset,
    build_feature_columns,
    feature_mapping_rows,
    load_external_dataset,
)
from src.experiments.hrdataset_replication_core import (
    HRDatasetReplicationProtocol,
    HRDatasetReplicationResult,
    evaluate_hrdataset_replication,
)
from src.experiments.hrdataset_replication_diagnostics import (
    ATTRIBUTION_WARNING,
    AuditAttributeSpec,
    FoldModelReference,
    OOFShapEvidence,
    ProxyReconstructabilityEvidence,
    RESEARCH_USE_WARNING,
    ReplicationIdentity,
    SubgroupDiagnosticsEvidence,
    TEMPORALITY_WARNING,
    compute_exact_oof_grouped_shap,
    compute_proxy_reconstructability,
    compute_support_aware_subgroup_diagnostics,
    feature_policy_contract_sha256,
    model_set_sha256,
)
from src.experiments.manuscript_calibration import calibration_bin_rows
from src.experiments.manuscript_model_benchmark import exact_primary_feature_frame
from src.experiments.shared_folds import write_shared_folds
from src.governance.external_replication_contract import (
    POLICY_ORDER,
    policy_exclusion_list,
    validate_external_replication_side_inputs,
    validate_external_replication_settings,
)
from src.governance.manuscript_contract import (
    ACTUAL_INPUT_IDENTITY_FIELDS,
    RunManifestError,
    canonical_config_hash,
    evidence_scope_contract,
    evidence_scope_contract_hash,
    load_manuscript_config,
    manuscript_settings,
    primary_excluded_features,
    sha256_file,
    scientific_input_hash as compute_scientific_input_hash,
    source_tree_hash as compute_source_tree_hash,
    utc_now_iso,
    validate_portable_run_id,
)
from src.utils.config_loader import PROJECT_ROOT, load_config


DATASET_KEY = "hrdataset_v14"
INX_DATASET_KEY = "inx_primary"
PRIMARY_TASK = "ordinal_multiclass_performance"
LABELS = (2, 3, 4)
MINIMUM_TRANSPORT_FEATURES = 5
REQUIRED_SIDE_INPUT_KEYS = frozenset(
    {
        "data_acquisition_contract",
        "dataset_provenance",
        "feature_policy_projection",
        "model_search_space",
        "external_hrdataset_v14_schema_mapping",
    }
)
PORTABLE_TEXT_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".md", ".txt"})
FORBIDDEN_CORE_PATH_TOKENS = frozenset(
    {"counterfactual", "chatbot", "llm", "agent_audit", "ibm", "turnover"}
)
ABSOLUTE_USER_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s\"']+", re.IGNORECASE),
    re.compile(r"/home/[^/\s\"']+", re.IGNORECASE),
)


class HRDatasetStageError(RuntimeError):
    """Raised when the canonical external-replication stage fails closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_json_scalar,
    ).encode("utf-8")


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HRDatasetStageError(f"{context} must be a mapping.")
    return value


def _require_sha256(name: str, value: Any) -> str:
    observed = str(value)
    if len(observed) != 64 or any(character not in "0123456789abcdef" for character in observed):
        raise HRDatasetStageError(f"{name} must be a lowercase SHA-256 digest.")
    return observed


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_scalar,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    if not isinstance(frame, pd.DataFrame):
        raise HRDatasetStageError(f"CSV output is not a dataframe: {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    if not path.is_file() or path.stat().st_size <= 0:
        raise HRDatasetStageError(f"CSV output is missing or empty: {path}.")
    return path


def _side_input_path(record: Mapping[str, Any], *, key: str) -> Path:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise HRDatasetStageError(f"Scientific side input {key!r} has no portable path.")
    path = Path(raw_path)
    if path.is_absolute():
        raise HRDatasetStageError(f"Scientific side input {key!r} contains an absolute path.")
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise HRDatasetStageError(f"Scientific side input {key!r} escapes the repository.") from exc
    return resolved


def _validate_side_input_records(
    records: Mapping[str, Mapping[str, Any]],
    *,
    required_keys: Iterable[str] = REQUIRED_SIDE_INPUT_KEYS,
) -> dict[str, dict[str, Any]]:
    missing = sorted(set(required_keys).difference(records))
    if missing:
        raise HRDatasetStageError(f"Scoped run manifest lacks required side inputs: {missing}.")
    validated: dict[str, dict[str, Any]] = {}
    for key, raw_record in sorted(records.items()):
        record = _require_mapping(raw_record, context=f"side input {key!r}")
        path = _side_input_path(record, key=key)
        if not path.is_file():
            raise HRDatasetStageError(f"Scientific side input is missing: {record.get('path')!r}.")
        digest = sha256_file(path)
        size = int(path.stat().st_size)
        if digest != record.get("sha256") or size != record.get("size_bytes"):
            raise HRDatasetStageError(f"Scientific side input {key!r} changed after manifest creation.")
        validated[key] = {
            "path": str(record["path"]).replace("\\", "/"),
            "sha256": digest,
            "size_bytes": size,
        }
    return validated


def _validate_scope_and_side_inputs(
    config: Mapping[str, Any],
    *,
    supplied_scope_contract_hash: str,
    records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    """Recompute the immutable core scope and verify its complete side-input set."""

    scope_contract = evidence_scope_contract(config, "core")
    observed_scope_hash = evidence_scope_contract_hash(config, "core")
    if observed_scope_hash != supplied_scope_contract_hash:
        raise HRDatasetStageError(
            "Supplied scope_contract_hash differs from the canonical core evidence scope."
        )
    expected_keys = set(scope_contract["side_input_keys"])
    observed_keys = set(records)
    if observed_keys != expected_keys:
        raise HRDatasetStageError(
            "Scoped side-input set differs from the canonical core evidence scope: "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}."
        )
    return scope_contract, _validate_side_input_records(
        records,
        required_keys=scope_contract["side_input_keys"],
    )


def _dataset_hash_records_from_verified_receipts(
    settings: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
    *,
    dataset_keys: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Reconstruct the exact dataset identity used by ``create_run_manifest``.

    The caller must first compare these receipts with fresh canonical-loader
    receipts.  Keeping the reconstruction here deterministic lets the stage
    independently verify the manifest's composite scientific-input hash.
    """

    expected_keys = set(dataset_keys)
    observed_keys = set(receipts)
    if observed_keys != expected_keys:
        raise HRDatasetStageError(
            "Scoped dataset receipt set differs from the canonical core evidence scope: "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}."
        )
    dataset_definitions = _require_mapping(settings.get("datasets"), context="datasets")
    records: dict[str, dict[str, Any]] = {}
    for dataset_key in dataset_keys:
        receipt = _require_mapping(
            receipts[dataset_key], context=f"actual input receipt {dataset_key!r}"
        )
        definition = _require_mapping(
            dataset_definitions.get(dataset_key), context=f"dataset {dataset_key!r}"
        )
        records[dataset_key] = {
            "path": receipt.get("actual_path"),
            "sha256": receipt.get("actual_sha256"),
            "size_bytes": receipt.get("size_bytes"),
            "row_count": receipt.get("row_count"),
            "column_count": receipt.get("column_count"),
            "schema_status": receipt.get("schema_status"),
            "target_column": receipt.get("target_column"),
            "target_distribution": receipt.get("target_distribution"),
            "role": definition.get("role", ""),
            "task_type": definition.get("task_type", ""),
        }
    return records


def _validate_scientific_identity(
    settings: Mapping[str, Any],
    *,
    config_hash: str,
    scope_contract_hash: str,
    scope_contract: Mapping[str, Sequence[str]],
    receipts: Mapping[str, Mapping[str, Any]],
    side_inputs: Mapping[str, Mapping[str, Any]],
    supplied_scientific_input_hash: str,
) -> dict[str, dict[str, Any]]:
    """Recompute and verify the composite scientific-input identity."""

    dataset_hashes = _dataset_hash_records_from_verified_receipts(
        settings,
        receipts,
        dataset_keys=scope_contract["dataset_keys"],
    )
    observed = compute_scientific_input_hash(
        config_hash=config_hash,
        scope_contract_hash=scope_contract_hash,
        dataset_hashes=dataset_hashes,
        side_input_hashes=side_inputs,
    )
    if observed != supplied_scientific_input_hash:
        raise HRDatasetStageError(
            "Supplied scientific_input_hash differs from the recomputed canonical inputs."
        )
    return dataset_hashes


def _validate_actual_receipt(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    dataset_key: str,
) -> dict[str, Any]:
    fields = (*ACTUAL_INPUT_IDENTITY_FIELDS, "size_bytes")
    differences = {
        field: {"expected": expected.get(field), "observed": observed.get(field)}
        for field in fields
        if expected.get(field) != observed.get(field)
    }
    if differences:
        raise HRDatasetStageError(
            f"Canonical loader receipt for {dataset_key!r} differs from the scoped run manifest: "
            + json.dumps(differences, sort_keys=True, ensure_ascii=True)
        )
    return {field: observed.get(field) for field in observed}


def _load_bound_datasets(
    config_path: Path,
    expected_receipts: Mapping[str, Mapping[str, Any]],
    *,
    acquisition_manifest_path: Path,
) -> tuple[CanonicalDataset, CanonicalDataset]:
    missing = sorted({INX_DATASET_KEY, DATASET_KEY}.difference(expected_receipts))
    if missing:
        raise HRDatasetStageError(f"Scoped run manifest lacks dataset receipts: {missing}.")
    inx = load_canonical_dataset(
        config_path,
        INX_DATASET_KEY,
        acquisition_manifest_path,
        allow_download=False,
    )
    hr = load_canonical_dataset(
        config_path,
        DATASET_KEY,
        acquisition_manifest_path,
        allow_download=False,
    )
    _validate_actual_receipt(inx.receipt, expected_receipts[INX_DATASET_KEY], dataset_key=INX_DATASET_KEY)
    _validate_actual_receipt(hr.receipt, expected_receipts[DATASET_KEY], dataset_key=DATASET_KEY)
    return inx, hr


def _feature_contract(
    dataset: ExternalDataset,
    external_settings: Mapping[str, Any],
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, str],
    dict[str, tuple[str, ...]],
    pd.DataFrame,
]:
    contract = _require_mapping(
        external_settings.get("feature_policy_contract"), context="external feature policy contract"
    )
    configured_order = tuple(str(value) for value in contract.get("reported_policy_order", ()))
    if configured_order != POLICY_ORDER:
        raise HRDatasetStageError("External policy order differs from the frozen contract.")
    metadata = _require_mapping(contract.get("policies"), context="external policy metadata")
    schema_variants = dataset.config.feature_policy_variants
    frames: dict[str, pd.DataFrame] = {}
    roles: dict[str, str] = {}
    forbidden: dict[str, tuple[str, ...]] = {}
    rows: list[dict[str, Any]] = []
    for policy in configured_order:
        columns = build_feature_columns(dataset, policy)
        if not columns or len(set(columns)) != len(columns):
            raise HRDatasetStageError(f"Policy {policy!r} has an empty or duplicate feature contract.")
        expected_exclusions = tuple(policy_exclusion_list(policy))
        observed_exclusions = tuple(str(value) for value in schema_variants[policy]["exclude_columns"])
        if observed_exclusions != expected_exclusions:
            raise HRDatasetStageError(f"Policy {policy!r} schema exclusions drifted.")
        policy_metadata = _require_mapping(metadata.get(policy), context=f"policy {policy!r}")
        frames[policy] = dataset.canonical.loc[:, columns].copy()
        roles[policy] = str(policy_metadata["role"])
        forbidden[policy] = expected_exclusions
        for position, feature in enumerate(columns, start=1):
            rows.append(
                {
                    "policy": policy,
                    "policy_role": roles[policy],
                    "audit_only": bool(policy_metadata["audit_only"]),
                    "raw_feature_order": position,
                    "feature": feature,
                    "included": True,
                    "exclusion_contract_sha256": _sha256_bytes(
                        _canonical_json_bytes(list(expected_exclusions))
                    ),
                }
            )
    primary = str(contract.get("primary_policy"))
    governance = _require_mapping(
        external_settings.get("feature_governance"), context="external feature governance"
    )
    expected_primary = tuple(str(value) for value in governance.get("exact_primary_feature_families", ()))
    if tuple(frames[primary].columns) != expected_primary:
        raise HRDatasetStageError(
            "Observed conservative-primary feature order differs from the exact governance contract."
        )
    return frames, roles, forbidden, pd.DataFrame(rows)


def _target_support(
    dataset: ExternalDataset,
    *,
    identity: Mapping[str, Any],
) -> pd.DataFrame:
    raw = dataset.raw[dataset.target_raw_column].astype("string").str.strip()
    mapped = dataset.canonical[dataset.target_column].astype(int)
    rows: list[dict[str, Any]] = []
    for scale, values in (("raw", raw), ("mapped", mapped)):
        counts = values.value_counts(dropna=False).sort_index()
        for label, count in counts.items():
            rows.append(
                {
                    **identity,
                    "support_scale": scale,
                    "target_column": dataset.target_raw_column if scale == "raw" else dataset.target_column,
                    "target_value": _json_scalar(label),
                    "count": int(count),
                    "proportion": float(count / len(values)),
                    "n_total": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def _derived_feature_quality(
    dataset: ExternalDataset,
    *,
    identity: Mapping[str, Any],
) -> pd.DataFrame:
    feature = "ExperienceYearsAtThisCompany"
    values = pd.to_numeric(dataset.canonical[feature], errors="coerce")
    return pd.DataFrame(
        [
            {
                **identity,
                "feature": feature,
                "source_column": "DateofHire",
                "reference_column": "LastPerformanceReview_Date",
                "invalid_duration_policy": "set_negative_to_missing",
                "expected_negative_source_duration_count": 2,
                "observed_missing_after_quality_rule": int(values.isna().sum()),
                "observed_negative_after_quality_rule": int((values.dropna() < 0).sum()),
                "raw_date_fields_used_as_model_inputs": False,
                "imputation_scope": "training_partition_only",
                "temporality_status": (
                    "derived_at_last_review_timing_unverified_negative_durations_set_missing"
                ),
                "claim_boundary": "derived_context_not_causal_not_actionable",
            }
        ]
    )


def _attach_diagnostic_identity(
    frame: pd.DataFrame,
    identity: ReplicationIdentity,
) -> pd.DataFrame:
    output = frame.copy()
    for field, value in identity.as_dict().items():
        if field in output.columns:
            observed = set(output[field].astype(str))
            if observed != {str(value)}:
                raise HRDatasetStageError(f"Upstream frame has conflicting {field}: {observed}.")
        else:
            output[field] = value
    return output


def _diagnostic_identity(
    result: HRDatasetReplicationResult,
    *,
    policy_frames: Mapping[str, pd.DataFrame],
    primary_policy: str,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    dataset_sha256: str,
    schema_mapping_sha256: str,
) -> tuple[ReplicationIdentity, list[FoldModelReference]]:
    primary_receipts = result.outer_model_receipts[
        result.outer_model_receipts["policy"].astype(str) == primary_policy
    ].copy()
    if len(primary_receipts) != 10 or primary_receipts["outer_fold"].duplicated().any():
        raise HRDatasetStageError("Primary external model receipts must contain exactly ten folds.")
    references: list[FoldModelReference] = []
    for row in primary_receipts.sort_values("outer_fold").itertuples(index=False):
        key = (primary_policy, int(row.outer_fold))
        if key not in result.fitted_outer_models:
            raise HRDatasetStageError(f"Missing in-memory primary model for fold {row.outer_fold}.")
        references.append(
            FoldModelReference(
                outer_fold=int(row.outer_fold),
                model_sha256=str(row.model_sha256),
                pipeline=result.fitted_outer_models[key],
            )
        )
    policy_features = {name: tuple(frame.columns) for name, frame in policy_frames.items()}
    identity = ReplicationIdentity(
        run_id=run_id,
        config_hash=config_hash,
        scientific_input_hash=scientific_input_hash,
        dataset_sha256=dataset_sha256,
        schema_mapping_sha256=schema_mapping_sha256,
        fold_contract_hash=str(result.folds.contract["fold_contract_hash"]),
        feature_policy_contract_sha256=feature_policy_contract_sha256(policy_features),
        model_set_sha256=model_set_sha256(references),
    )
    return identity, references


def _governance_mapping(external_settings: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    governance = _require_mapping(
        external_settings.get("feature_governance"), context="external feature governance"
    )
    features = _require_mapping(governance.get("features"), context="feature governance rows")
    warning = str(governance.get("model_scenario_only_warning", ""))
    if not warning:
        raise HRDatasetStageError("External feature governance lacks its model-scenario warning.")
    return {
        str(feature): {
            "governance_category": str(_require_mapping(row, context=f"governance {feature}")["category"]),
            "temporality_status": str(row["temporality_status"]),
            "proxy_watchlist": str(row["category"]) in {
                "operational_proxy_context",
                "derived_tenure_context",
            },
            "model_scenario_only_warning": warning,
        }
        for feature, row in features.items()
    }


def _audit_attribute_specs(external_settings: Mapping[str, Any]) -> list[AuditAttributeSpec]:
    subgroup = _require_mapping(
        external_settings.get("subgroup_diagnostics"), context="external subgroup diagnostics"
    )
    attributes = _require_mapping(subgroup.get("attributes"), context="external subgroup attributes")
    specs: list[AuditAttributeSpec] = []
    for category in ("protected_sensitive", "exploratory_operational"):
        definition = _require_mapping(attributes.get(category), context=f"subgroup {category}")
        if definition.get("type") != "categorical":
            raise HRDatasetStageError("HRDataset external subgroup attributes must remain categorical.")
        specs.extend(
            AuditAttributeSpec(name=str(feature), category=category, transform="categorical")
            for feature in definition.get("features", ())
        )
    return specs


def _compute_diagnostics(
    result: HRDatasetReplicationResult,
    dataset: ExternalDataset,
    policy_frames: Mapping[str, pd.DataFrame],
    external_settings: Mapping[str, Any],
    forbidden_by_policy: Mapping[str, Sequence[str]],
    identity: ReplicationIdentity,
    fold_models: Sequence[FoldModelReference],
) -> tuple[OOFShapEvidence, SubgroupDiagnosticsEvidence, ProxyReconstructabilityEvidence]:
    contract = _require_mapping(
        external_settings.get("feature_policy_contract"), context="external feature policy contract"
    )
    primary_policy = str(contract["primary_policy"])
    primary_features = policy_frames[primary_policy].copy()
    primary_features.insert(0, "sample_index", primary_features.index.astype(int))
    policy_feature_names = {name: tuple(frame.columns) for name, frame in policy_frames.items()}
    raw_oof = _attach_diagnostic_identity(result.raw_oof_predictions, identity)
    shap_oof = raw_oof[raw_oof["policy"].astype(str) == primary_policy].copy()
    shap = compute_exact_oof_grouped_shap(
        features=primary_features,
        fold_assignments=result.folds.outer_assignments,
        oof_predictions=shap_oof,
        fold_models=fold_models,
        policy_features=policy_feature_names,
        primary_policy=primary_policy,
        forbidden_features=forbidden_by_policy[primary_policy],
        identity=identity,
        labels=LABELS,
        feature_governance=_governance_mapping(external_settings),
        top_k=int(external_settings["shap"]["stability_top_k"]),
        attribution_unit=str(external_settings["shap"]["attribution_unit"]),
        additivity_output_space=str(external_settings["shap"]["additivity_output_space"]),
    )

    specs = _audit_attribute_specs(external_settings)
    audit_columns = [spec.name for spec in specs]
    audit = dataset.canonical.loc[:, audit_columns].copy()
    audit.insert(0, "sample_index", audit.index.astype(int))
    subgroup_settings = _require_mapping(
        external_settings.get("subgroup_diagnostics"), context="external subgroup diagnostics"
    )
    subgroup = compute_support_aware_subgroup_diagnostics(
        oof_predictions=shap_oof,
        fold_assignments=result.folds.outer_assignments,
        audit_frame=audit,
        attributes=specs,
        identity=identity,
        labels=LABELS,
        minimum_group_support=int(subgroup_settings["minimum_group_support"]),
        minimum_metric_denominator=int(subgroup_settings["minimum_class_metric_denominator"]),
        n_resamples=int(external_settings["uncertainty"]["n_resamples"]),
        confidence_level=float(external_settings["uncertainty"]["confidence_level"]),
        seed=int(external_settings["resolved_seeds"]["bootstrap"]),
        batch_size=200,
        probability_method=str(subgroup_settings["probability_method"]),
    )

    proxy_settings = _require_mapping(
        external_settings.get("proxy_diagnostics"), context="external proxy diagnostics"
    )
    proxy_target = str(proxy_settings["target"])
    proxy_audit = dataset.canonical.loc[:, [proxy_target]].copy()
    proxy_audit.insert(0, "sample_index", proxy_audit.index.astype(int))
    predictor_sources = tuple(str(value) for value in proxy_settings["predictor_policy_sources"])
    predictor_sets = {name: tuple(policy_frames[name].columns) for name in predictor_sources}
    union_features = list(dict.fromkeys(feature for name in predictor_sources for feature in predictor_sets[name]))
    proxy_features = dataset.canonical.loc[:, union_features].copy()
    proxy_features.insert(0, "sample_index", proxy_features.index.astype(int))
    proxy = compute_proxy_reconstructability(
        features=proxy_features,
        fold_assignments=result.folds.outer_assignments,
        audit_frame=proxy_audit,
        predictor_sets=predictor_sets,
        proxy_target=proxy_target,
        proxy_aliases=tuple(str(value) for value in proxy_settings["target_aliases"]),
        identity=identity,
        seed=int(external_settings["resolved_seeds"]["fairness"]),
        n_resamples=int(proxy_settings["bootstrap"]["n_resamples"]),
        confidence_level=float(proxy_settings["bootstrap"]["confidence_level"]),
        batch_size=200,
    )
    return shap, subgroup, proxy


def _transport_evidence(
    inx: CanonicalDataset,
    hr_features: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    inx_features = exact_primary_feature_frame(
        inx.frame,
        excluded_features=primary_excluded_features(config),
    )
    inx_names = set(map(str, inx_features.columns))
    hr_names = set(map(str, hr_features.columns))
    common = sorted(inx_names.intersection(hr_names))
    rows = pd.DataFrame(
        [
            {
                **identity,
                "feature": feature,
                "in_inx_canonical_primary": feature in inx_names,
                "in_hrdataset_conservative_primary": feature in hr_names,
                "common_safe_feature": feature in common,
            }
            for feature in sorted(inx_names.union(hr_names))
        ]
    )
    feasible = len(common) >= MINIMUM_TRANSPORT_FEATURES
    assessment = {
        **identity,
        "status": "schema_gate_satisfied_but_transport_not_run" if feasible else "infeasible_too_few_common_safe_features",
        "locked_inx_model_transported": False,
        "n_common_safe_features": len(common),
        "common_safe_features": common,
        "minimum_common_feature_gate": MINIMUM_TRANSPORT_FEATURES,
        "hr_target_labels_verified": list(LABELS),
        "target_mapping_changed_by_feature_policy": False,
        "claim": "schema_overlap_feasibility_only_not_locked_model_transport",
        "interpretation": (
            "Only three conservative common features are available; no scientifically defensible "
            "locked INX-model transport was attempted. HRDataset_v14 remains an independently "
            "trained mapped-target replication."
            if not feasible
            else "The schema gate alone is satisfied, but no locked INX model was transported."
        ),
    }
    return rows, assessment


def _calibration_reliability(
    result: HRDatasetReplicationResult,
    identity: ReplicationIdentity,
) -> pd.DataFrame:
    primary = result.protocol_metadata["primary_policy"]
    raw = result.raw_oof_predictions[result.raw_oof_predictions["policy"] == primary].sort_values(
        "sample_index"
    )
    sigmoid = result.calibrated_oof_predictions.sort_values("sample_index")
    if not np.array_equal(raw["sample_index"].to_numpy(), sigmoid["sample_index"].to_numpy()):
        raise HRDatasetStageError("Raw and sigmoid external OOF sample orders differ.")
    rows: list[dict[str, Any]] = []
    for method, frame in (("raw", raw), ("sigmoid", sigmoid)):
        rows.extend(
            calibration_bin_rows(
                frame["y_true"].to_numpy(int),
                frame[[f"prob_class_{label}" for label in LABELS]].to_numpy(float),
                LABELS,
                run_id=identity.run_id,
                config_hash=identity.config_hash,
                method=method,
                n_bins=10,
                identity=identity.as_dict(),
            )
        )
    return pd.DataFrame(rows)


def _write_models(
    staging: Path,
    result: HRDatasetReplicationResult,
) -> pd.DataFrame:
    receipts = result.outer_model_receipts.copy()
    artifact_paths: dict[tuple[str, int], str] = {}
    for (policy, outer_fold), payload in sorted(result.serialized_outer_models.items()):
        path = staging / "models" / str(policy) / f"outer_fold_{int(outer_fold):02d}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        digest = sha256_file(path)
        expected = receipts[
            (receipts["policy"].astype(str) == str(policy))
            & (receipts["outer_fold"].astype(int) == int(outer_fold))
        ]
        if len(expected) != 1 or digest != str(expected.iloc[0]["model_sha256"]):
            raise HRDatasetStageError(f"Persisted model hash differs for {policy}/fold {outer_fold}.")
        artifact_paths[(str(policy), int(outer_fold))] = path.relative_to(staging).as_posix()
    receipts["model_artifact_path"] = [
        artifact_paths[(str(row.policy), int(row.outer_fold))]
        for row in receipts.itertuples(index=False)
    ]
    return receipts


def _write_local_reason_codes(staging: Path, shap: OOFShapEvidence) -> list[Path]:
    output_dir = staging / "shap" / "local_reason_codes"
    paths: list[Path] = []
    for case in shap.representative_cases.itertuples(index=False):
        sample_index = int(case.sample_index)
        predicted_class = int(case.y_pred)
        scoped = shap.local_values[
            (shap.local_values["sample_index"].astype(int) == sample_index)
            & (shap.local_values["class_label"].astype(int) == predicted_class)
        ].sort_values(["within_case_class_abs_rank", "feature"], kind="stable")
        if scoped.empty:
            raise HRDatasetStageError(f"Representative SHAP case {sample_index} has no local rows.")
        slug = f"{str(case.case_type)}_{sample_index}"
        csv_path = _write_csv(output_dir / f"local_reason_code_{slug}.csv", scoped)
        payload = {
            "run_id": str(case.run_id),
            "config_hash": str(case.config_hash),
            "scientific_input_hash": str(case.scientific_input_hash),
            "dataset_sha256": str(case.dataset_sha256),
            "case_type": str(case.case_type),
            "selection_rule": str(case.selection_rule),
            "sample_index": sample_index,
            "outer_fold": int(case.outer_fold),
            "y_true": int(case.y_true),
            "y_pred": predicted_class,
            "confidence": float(case.confidence),
            "attribution_unit": str(shap.metadata["attribution_unit"]),
            "additivity_output_space": str(shap.metadata["additivity_output_space"]),
            "attribution_warning": ATTRIBUTION_WARNING,
            "temporality_warning": TEMPORALITY_WARNING,
            "research_use_warning": RESEARCH_USE_WARNING,
            "reason_codes": scoped.to_dict(orient="records"),
        }
        json_path = _write_json(output_dir / f"local_reason_code_{slug}.json", payload)
        md_path = output_dir / f"local_reason_code_{slug}.md"
        lines = [
            f"# External OOF reason code: {case.case_type}",
            "",
            f"Sample index: `{sample_index}`; outer fold: `{int(case.outer_fold)}`; true/predicted: `{int(case.y_true)}`/`{predicted_class}`.",
            "",
            ATTRIBUTION_WARNING,
            "",
            TEMPORALITY_WARNING,
            "",
            RESEARCH_USE_WARNING,
            "",
            f"Attribution unit: `{shap.metadata['attribution_unit']}`; additivity output space: `{shap.metadata['additivity_output_space']}`.",
            "",
            "| Rank | Feature | Value | Grouped SHAP (XGBoost raw-margin score) | Governance | Temporality |",
            "|---:|---|---|---:|---|---|",
        ]
        for row in scoped.itertuples(index=False):
            lines.append(
                f"| {int(row.within_case_class_abs_rank)} | {row.feature} | {row.feature_value} | "
                f"{float(row.grouped_shap_value):.6f} | {row.governance_category} | {row.temporality_status} |"
            )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        paths.extend((csv_path, json_path, md_path))
    return paths


def _all_files(root: Path, *, exclude: Iterable[str] = ()) -> list[Path]:
    excluded = set(exclude)
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not (
                len(path.relative_to(root).parts) == 1
                and path.relative_to(root).name in excluded
            )
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _write_artifact_manifest(staging: Path, identity: ReplicationIdentity) -> tuple[Path, Path]:
    excluded = {"artifact_manifest.csv", "artifact_manifest.json", "stage_contract.json"}
    records = [
        {
            "run_id": identity.run_id,
            "config_hash": identity.config_hash,
            "scientific_input_hash": identity.scientific_input_hash,
            "dataset_sha256": identity.dataset_sha256,
            "path": path.relative_to(staging).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in _all_files(staging, exclude=excluded)
    ]
    if not records:
        raise HRDatasetStageError("External replication produced no artifact records.")
    csv_path = _write_csv(staging / "artifact_manifest.csv", pd.DataFrame(records))
    json_path = _write_json(
        staging / "artifact_manifest.json",
        {
            **identity.as_dict(),
            "status": "complete",
            "hash_algorithm": "sha256",
            "self_excluded_files": sorted(excluded),
            "n_artifacts": len(records),
            "artifacts": records,
        },
    )
    return csv_path, json_path


def _write_atomic_stage_contract(
    staging: Path,
    final_output: Path,
    *,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    scope_contract_hash: str,
    git_commit: str,
    source_tree_hash: str,
    dataset_hashes: Mapping[str, Any],
    actual_input_receipts: Mapping[str, Any],
    side_input_hashes: Mapping[str, Any],
    started_at: str,
    elapsed_seconds: float,
) -> Path:
    """Write the builder-compatible closed-world receipt before atomic publish."""

    contract_name = "stage_contract.json"
    files = _all_files(staging, exclude={contract_name})
    if not files:
        raise HRDatasetStageError("Atomic external stage contract cannot be empty.")
    records: list[dict[str, Any]] = []
    final_root = final_output.resolve()
    for source in files:
        if source.stat().st_size <= 0:
            raise HRDatasetStageError(
                f"Atomic external stage contract cannot admit an empty artifact: {source.name}."
            )
        relative = source.relative_to(staging).as_posix()
        final_path = (final_root / relative).resolve()
        try:
            final_path.relative_to(final_root)
            final_path.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise HRDatasetStageError(
                f"Atomic external stage artifact escapes its final stage root: {relative}."
            ) from exc
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(source),
                "size_bytes": int(source.stat().st_size),
            }
        )
    return _write_json(
        staging / contract_name,
        {
            "stage": "external_replication",
            "status": "complete",
            "inventory_mode": "closed_world",
            "path_basis": "stage_relative",
            "run_id": run_id,
            "config_hash": config_hash,
            "evidence_scope": "core",
            "scope_contract_hash": scope_contract_hash,
            "git_commit": git_commit,
            "source_tree_hash": source_tree_hash,
            "dataset_hashes": dict(dataset_hashes),
            "actual_input_receipts": dict(actual_input_receipts),
            "side_input_hashes": dict(side_input_hashes),
            "scientific_input_hash": scientific_input_hash,
            "started_at": started_at,
            "ended_at": utc_now_iso(),
            "elapsed_seconds": float(elapsed_seconds),
            "outputs": records,
        },
    )


def _validate_portability_and_scope(staging: Path) -> None:
    errors: list[str] = []
    for path in _all_files(staging):
        relative = path.relative_to(staging).as_posix()
        lowered = relative.casefold()
        if any(token in lowered for token in FORBIDDEN_CORE_PATH_TOKENS):
            errors.append(f"forbidden core artifact path: {relative}")
        if path.stat().st_size <= 0:
            errors.append(f"empty artifact: {relative}")
        if path.suffix.casefold() in PORTABLE_TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            for pattern in ABSOLUTE_USER_PATH_PATTERNS:
                if pattern.search(text):
                    errors.append(f"absolute user path in {relative}")
    if errors:
        raise HRDatasetStageError("External artifact validation failed:\n- " + "\n- ".join(errors))


def _write_interpretation(
    staging: Path,
    result: HRDatasetReplicationResult,
    transport: Mapping[str, Any],
    proxy: ProxyReconstructabilityEvidence,
    identity: ReplicationIdentity,
) -> Path:
    primary = str(result.protocol_metadata["primary_policy"])
    intervals = result.raw_metric_intervals
    primary_rows = intervals[intervals["system_id"].astype(str) == primary].set_index("metric")
    sigmoid_rows = result.calibration_metric_intervals
    sigmoid = sigmoid_rows[sigmoid_rows["system_id"].astype(str) == "sigmoid"].set_index("metric")

    def metric_line(frame: pd.DataFrame, metric: str) -> str:
        row = frame.loc[metric]
        return f"{float(row.point_estimate):.6f} (95% CI {float(row.ci_low):.6f}–{float(row.ci_high):.6f})"

    proxy_status = str(proxy.status.iloc[0]["analysis_status"])
    lines = [
        "# HRDataset_v14 independent mapped-target replication",
        "",
        f"Run ID: `{identity.run_id}`  ",
        f"Config hash: `{identity.config_hash}`",
        "",
        "This stage independently trains the leakage-aware XGBoost protocol on HRDataset_v14. "
        "It is not locked INX-model transport or universal external validation.",
        "",
        "## Conservative primary result",
        "",
        f"- Raw macro-F1: {metric_line(primary_rows, 'macro_f1')}.",
        f"- Raw QWK: {metric_line(primary_rows, 'quadratic_weighted_kappa')}.",
        f"- Predeclared sigmoid macro-F1: {metric_line(sigmoid, 'macro_f1')}.",
        "- Fold summaries are descriptive variability only; sample-level intervals use 5,000 paired "
        "outer-fold/target-stratified bootstrap draws.",
        "",
        "## Claim boundaries",
        "",
        "- Engagement, satisfaction, project, lateness and attendance fields have unverified timing; "
        "the temporality-restricted audit is reported separately.",
        "- SHAP values are exact-fold model attribution, not causal effects.",
        "- Subgroup differences are support-aware descriptive diagnostics, not proof of fairness, "
        "discrimination, legal compliance or deployment readiness.",
        f"- Department reconstructability status is `{proxy_status}`; classes were not merged or dropped.",
        "- No employee advice, prescription, autonomous HR decision, or causal intervention is supported.",
        "- Dataset source authenticity and licence remain manual-review items.",
        "",
        "## Locked-model transport feasibility",
        "",
        f"Common conservative features: {transport['n_common_safe_features']} "
        f"({', '.join(transport['common_safe_features'])}). Status: `{transport['status']}`. "
        "No locked model was transported.",
    ]
    path = staging / "external_replication_interpretation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _late_revalidate(
    config_path: Path,
    *,
    config_hash: str,
    expected_receipts: Mapping[str, Mapping[str, Any]],
    expected_side_inputs: Mapping[str, Mapping[str, Any]],
    acquisition_manifest_path: Path,
    git_commit: str,
    expected_source_tree_hash: str,
    allowed_untracked_root: Path,
) -> dict[str, Any]:
    if canonical_config_hash(config_path) != config_hash:
        raise HRDatasetStageError("Canonical config changed during external replication.")
    validated_side_inputs = _validate_side_input_records(expected_side_inputs)
    inx, hr = _load_bound_datasets(
        config_path,
        expected_receipts,
        acquisition_manifest_path=acquisition_manifest_path,
    )
    _validate_source_identity_at_start(
        git_commit=git_commit,
        expected_source_tree_hash=expected_source_tree_hash,
        allowed_untracked_root=allowed_untracked_root,
    )
    current_commit = git_commit
    current_source_hash = compute_source_tree_hash(PROJECT_ROOT)
    return {
        "config_hash": config_hash,
        "actual_input_sha256": {
            INX_DATASET_KEY: inx.receipt["actual_sha256"],
            DATASET_KEY: hr.receipt["actual_sha256"],
        },
        "side_input_sha256": {
            key: record["sha256"] for key, record in validated_side_inputs.items()
        },
        "git_commit": current_commit,
        "source_tree_hash": current_source_hash,
        "tracked_worktree_clean_at_publication": True,
        "untracked_files_restricted_to_current_run_root": True,
        "status": "passed_before_atomic_publication",
    }


def _validate_source_identity_at_start(
    *,
    git_commit: str,
    expected_source_tree_hash: str,
    allowed_untracked_root: str | Path,
) -> None:
    """Verify source identity while allowing only builder-owned current-run files.

    The top-level builder already requires an entirely clean worktree before it
    creates the run manifest.  By the time this seventh core stage starts, its
    upstream stages are necessarily untracked files under the current run root.
    Tracked changes and untracked files anywhere else remain fatal.
    """

    allowed_root = Path(allowed_untracked_root).resolve()
    try:
        allowed_root.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise HRDatasetStageError("Current-run output root must remain inside the repository.") from exc

    try:
        current_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()
        untracked_output = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HRDatasetStageError(f"Cannot validate Git source identity at run start: {exc}") from exc
    if current_commit != git_commit:
        raise HRDatasetStageError("Git HEAD differs from the scoped run manifest at run start.")
    if tracked_status:
        raise HRDatasetStageError("Canonical HRDataset publication requires a clean worktree at run start.")
    disallowed_untracked: list[str] = []
    for relative in (value for value in untracked_output.split("\0") if value):
        candidate = (PROJECT_ROOT / relative).resolve()
        try:
            candidate.relative_to(allowed_root)
        except ValueError:
            disallowed_untracked.append(relative.replace("\\", "/"))
    if disallowed_untracked:
        raise HRDatasetStageError(
            "Canonical HRDataset publication requires a clean worktree at run start; "
            "untracked files outside the current run root were found: "
            f"{sorted(disallowed_untracked)}."
        )
    if compute_source_tree_hash(PROJECT_ROOT) != expected_source_tree_hash:
        raise HRDatasetStageError("Source tree differs from the scoped run manifest at run start.")


def _validate_output_contract(
    settings: Mapping[str, Any],
    *,
    run_id: str,
    output: Path,
) -> Path:
    """Bind publication to the exact builder-owned core run directory."""

    try:
        validate_portable_run_id(run_id)
    except RunManifestError as exc:
        raise HRDatasetStageError(str(exc)) from exc
    output_settings = _require_mapping(settings.get("output"), context="output")
    raw_root = output_settings.get("root")
    if not isinstance(raw_root, str) or not raw_root.strip() or Path(raw_root).is_absolute():
        raise HRDatasetStageError("Canonical output.root must be a repository-relative path.")
    package_root = (PROJECT_ROOT / raw_root).resolve()
    try:
        package_root.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise HRDatasetStageError("Canonical output.root escapes the repository.") from exc
    run_root = (package_root / run_id / "core").resolve()
    try:
        relative_run_root = run_root.relative_to(package_root)
    except ValueError as exc:
        raise HRDatasetStageError("Canonical run root escapes output.root.") from exc
    if not relative_run_root.parts or relative_run_root.parts[0] != run_id:
        raise HRDatasetStageError("Canonical run root is not an exact child of output.root.")
    expected_output = run_root / "external_replication"
    if output != expected_output:
        raise HRDatasetStageError(
            "HRDataset output path differs from the canonical builder-owned run contract: "
            f"expected={expected_output.relative_to(PROJECT_ROOT).as_posix()!r}."
        )
    return run_root


def run(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    run_id: str,
    config_hash: str,
    scientific_input_hash: str,
    expected_actual_input_receipts: Mapping[str, Mapping[str, Any]],
    expected_side_input_hashes: Mapping[str, Mapping[str, Any]],
    git_commit: str,
    source_tree_hash: str,
    scope_contract_hash: str,
    expected_git_worktree_dirty: bool,
) -> dict[str, Any]:
    """Execute and atomically publish one canonical HRDataset replication stage."""

    stage_started_at = utc_now_iso()
    stage_started_perf = time.perf_counter()
    config_path = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    if not str(run_id).strip():
        raise HRDatasetStageError("run_id must be non-empty.")
    for name, value in (
        ("config_hash", config_hash),
        ("scientific_input_hash", scientific_input_hash),
        ("source_tree_hash", source_tree_hash),
        ("scope_contract_hash", scope_contract_hash),
    ):
        _require_sha256(name, value)
    if not str(git_commit).strip():
        raise HRDatasetStageError("git_commit must be non-empty.")
    if expected_git_worktree_dirty is not False:
        raise HRDatasetStageError(
            "Canonical HRDataset publication requires a clean worktree at run start."
        )
    config = load_manuscript_config(config_path)
    settings = manuscript_settings(config)
    run_root = _validate_output_contract(settings, run_id=str(run_id), output=output)
    _validate_source_identity_at_start(
        git_commit=git_commit,
        expected_source_tree_hash=source_tree_hash,
        allowed_untracked_root=run_root,
    )
    if canonical_config_hash(config) != config_hash:
        raise HRDatasetStageError("Supplied config hash differs from the canonical config.")
    external_settings = _require_mapping(
        settings.get("external_replication"), context="external_replication"
    )
    external_settings = {
        **dict(external_settings),
        "resolved_seeds": dict(_require_mapping(settings.get("seeds"), context="canonical seeds")),
    }
    validate_external_replication_settings(settings)
    validate_external_replication_side_inputs(settings, project_root=PROJECT_ROOT)
    scope_contract, side_inputs = _validate_scope_and_side_inputs(
        config,
        supplied_scope_contract_hash=scope_contract_hash,
        records=expected_side_input_hashes,
    )
    acquisition_path = _side_input_path(
        expected_side_input_hashes["data_acquisition_contract"],
        key="data_acquisition_contract",
    )
    schema_path = _side_input_path(
        expected_side_input_hashes["external_hrdataset_v14_schema_mapping"],
        key="external_hrdataset_v14_schema_mapping",
    )
    model_grid_path = _side_input_path(
        expected_side_input_hashes["model_search_space"], key="model_search_space"
    )
    inx, raw_hr = _load_bound_datasets(
        config_path,
        expected_actual_input_receipts,
        acquisition_manifest_path=acquisition_path,
    )
    dataset_hashes = _validate_scientific_identity(
        settings,
        config_hash=config_hash,
        scope_contract_hash=scope_contract_hash,
        scope_contract=scope_contract,
        receipts=expected_actual_input_receipts,
        side_inputs=side_inputs,
        supplied_scientific_input_hash=scientific_input_hash,
    )
    dataset = load_external_dataset(
        DATASET_KEY,
        raw_frame=raw_hr.frame,
        schema_mapping_path=schema_path,
    )
    if dataset.task_type != PRIMARY_TASK or tuple(dataset.labels) != LABELS:
        raise HRDatasetStageError(
            f"Mapped HRDataset task/labels drifted: {dataset.task_type}/{dataset.labels}."
        )
    if dict(dataset.target_mapping) != dict(external_settings["target"]["mapping"]):
        raise HRDatasetStageError("Adapted target mapping differs from the canonical contract.")
    policy_frames, policy_roles, forbidden_by_policy, policy_rows = _feature_contract(
        dataset, external_settings
    )
    primary_policy = str(external_settings["feature_policy_contract"]["primary_policy"])
    benchmark_config = load_config(model_grid_path)
    result = evaluate_hrdataset_replication(
        policy_frames,
        policy_roles,
        forbidden_by_policy,
        dataset.canonical[dataset.target_column].astype(int),
        dataset.canonical["EmpNumber"],
        benchmark_config,
        primary_policy=primary_policy,
        run_id=run_id,
        config_hash=config_hash,
        scientific_input_hash=scientific_input_hash,
        dataset_sha256=str(raw_hr.receipt["actual_sha256"]),
        protocol=HRDatasetReplicationProtocol(
            outer_seed=int(settings["seeds"]["external_replication"]),
            inner_seed=int(settings["seeds"]["inner_cv"]),
            model_seed=int(settings["seeds"]["model"]),
            calibration_seed=int(settings["seeds"]["calibration"]),
            bootstrap_seed=int(settings["seeds"]["bootstrap"]),
        ),
    )
    if not result.canonical_eligible:
        raise HRDatasetStageError("A test-reduced external result may never be published.")
    protocol_metadata = dict(result.protocol_metadata)
    required_protocol_values = {
        "outer_splits": 10,
        "inner_splits": 5,
        "candidate_indices_evaluated": list(range(8)),
        "bootstrap_n_resamples": 5000,
        "canonical_eligible": True,
        "test_only_reduction": None,
        "calibration_method": "predeclared_cross_fitted_sigmoid",
        "outer_test_use": "evaluation_only",
    }
    protocol_drift = {
        field: {"expected": expected, "observed": protocol_metadata.get(field)}
        for field, expected in required_protocol_values.items()
        if protocol_metadata.get(field) != expected
    }
    if protocol_drift:
        raise HRDatasetStageError(
            "External computation returned a noncanonical protocol receipt: "
            + json.dumps(protocol_drift, sort_keys=True, ensure_ascii=True)
        )
    if (
        len(result.calibrator_model_relationships) != 10
        or result.calibrator_model_relationships["outer_fold"].duplicated().any()
    ):
        raise HRDatasetStageError(
            "External calibration must expose one exact calibrator/model relationship per outer fold."
        )
    identity, fold_models = _diagnostic_identity(
        result,
        policy_frames=policy_frames,
        primary_policy=primary_policy,
        run_id=run_id,
        config_hash=config_hash,
        scientific_input_hash=scientific_input_hash,
        dataset_sha256=str(raw_hr.receipt["actual_sha256"]),
        schema_mapping_sha256=side_inputs["external_hrdataset_v14_schema_mapping"]["sha256"],
    )
    shap, subgroup, proxy = _compute_diagnostics(
        result,
        dataset,
        policy_frames,
        external_settings,
        forbidden_by_policy,
        identity,
        fold_models,
    )
    common_identity = identity.as_dict()
    transport_rows, transport = _transport_evidence(
        inx,
        policy_frames[primary_policy],
        config,
        identity=common_identity,
    )
    if transport["n_common_safe_features"] != 3 or transport["status"] != "infeasible_too_few_common_safe_features":
        raise HRDatasetStageError(
            "The predeclared three-feature transport infeasibility finding changed; scientific review is required."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()):
        raise HRDatasetStageError(
            f"External output is non-empty and may not be overwritten: {output}."
        )
    with tempfile.TemporaryDirectory(prefix=".hrdataset-replication-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir(parents=True)
        write_shared_folds(result.folds, staging / "folds")
        model_receipts = _write_models(staging, result)
        table_outputs = {
            "feature_mapping": feature_mapping_rows(dataset).assign(**common_identity),
            "feature_policy_features": policy_rows.assign(**common_identity),
            "derived_feature_quality": _derived_feature_quality(
                dataset, identity=common_identity
            ),
            "target_support": _target_support(dataset, identity=common_identity),
            "candidate_fit_receipts": result.candidate_fit_receipts,
            "candidate_search_results": result.candidate_search_results,
            "selected_hyperparameters": result.selected_hyperparameters,
            "outer_model_receipts": model_receipts,
            "transformed_feature_lineage": result.transformed_feature_lineage,
            "raw_oof_predictions": result.raw_oof_predictions,
            "sigmoid_oof_predictions": result.calibrated_oof_predictions,
            "calibration_training_oof": result.calibration_training_oof,
            "calibration_fit_receipts": result.calibration_fit_receipts,
            "calibrator_parameters": result.calibrator_parameters,
            "calibrator_model_relationships": result.calibrator_model_relationships,
            "fold_metrics": result.fold_metrics,
            "fold_descriptive_summary": result.fold_descriptive_summary,
            "raw_metric_intervals": result.raw_metric_intervals,
            "policy_pairwise_differences": result.raw_policy_differences,
            "calibration_metric_intervals": result.calibration_metric_intervals,
            "calibration_paired_differences": result.calibration_differences,
            "calibration_reliability_bins": _calibration_reliability(result, identity),
        }
        for name, frame in table_outputs.items():
            _write_csv(staging / f"{name}.csv", frame)
        plan = result.bootstrap_resample_plan
        _write_csv(staging / "bootstrap_sample_order.csv", plan.sample_order)
        indices_path = staging / "bootstrap_resample_indices.npy.zlib"
        indices_path.write_bytes(plan.compressed_indices_bytes)
        if sha256_file(indices_path) != str(plan.receipt["compressed_indices_sha256"]):
            raise HRDatasetStageError("Persisted external bootstrap resample bytes changed.")
        _write_json(staging / "bootstrap_resample_plan.json", dict(plan.receipt))

        shap_dir = staging / "shap"
        _write_csv(shap_dir / "local_grouped_shap_values.csv", shap.local_values)
        _write_csv(shap_dir / "global_grouped_shap_importance.csv", shap.global_importance)
        for label in LABELS:
            _write_csv(
                shap_dir / f"class_{label}_grouped_shap_importance.csv",
                shap.class_importance[shap.class_importance["class_label"].astype(int) == label],
            )
        _write_csv(shap_dir / "class_grouped_shap_importance.csv", shap.class_importance)
        _write_csv(shap_dir / "fold_feature_rankings.csv", shap.fold_rankings)
        _write_csv(shap_dir / "shap_stability_pairwise.csv", shap.stability_pairwise)
        _write_csv(shap_dir / "shap_stability_summary.csv", shap.stability_summary)
        _write_csv(shap_dir / "representative_cases.csv", shap.representative_cases)
        _write_json(shap_dir / "shap_metadata.json", dict(shap.metadata))
        _write_local_reason_codes(staging, shap)

        subgroup_dir = staging / "subgroup_diagnostics"
        _write_csv(subgroup_dir / "group_metrics.csv", subgroup.group_metrics)
        _write_csv(subgroup_dir / "disparity_intervals.csv", subgroup.disparity_intervals)
        _write_json(subgroup_dir / "subgroup_metadata.json", dict(subgroup.metadata))

        proxy_dir = staging / "proxy_diagnostics"
        _write_csv(proxy_dir / "proxy_status.csv", proxy.status)
        _write_csv(proxy_dir / "proxy_feature_contracts.csv", proxy.feature_contracts)
        for filename, frame in (
            ("proxy_oof_predictions.csv", proxy.oof_predictions),
            ("proxy_fold_metrics.csv", proxy.fold_metrics),
            ("proxy_metric_intervals.csv", proxy.metric_intervals),
            ("proxy_paired_differences.csv", proxy.paired_differences),
        ):
            if frame.empty:
                _write_json(
                    proxy_dir / f"{Path(filename).stem}_not_estimated.json",
                    {
                        **common_identity,
                        "status": str(proxy.status.iloc[0]["analysis_status"]),
                        "reason": str(proxy.status.iloc[0]["reason"]),
                        "requested_output": filename,
                        "rows": 0,
                    },
                )
            else:
                _write_csv(proxy_dir / filename, frame)
        _write_json(proxy_dir / "proxy_metadata.json", dict(proxy.metadata))

        transport_dir = staging / "cross_dataset_transport"
        _write_csv(transport_dir / "feature_overlap.csv", transport_rows)
        _write_json(transport_dir / "transport_feasibility.json", transport)
        _write_interpretation(staging, result, transport, proxy, identity)
        _write_json(
            staging / "external_replication_metadata.json",
            {
                **common_identity,
                "status": "complete",
                "scope": str(external_settings["scope"]),
                "role": str(external_settings["role"]),
                "task_type": PRIMARY_TASK,
                "labels": list(LABELS),
                "primary_policy": primary_policy,
                "policy_order": list(POLICY_ORDER),
                "protocol": dict(result.protocol_metadata),
                "claim_boundary": str(external_settings["claim_boundary"]),
                "git_commit": git_commit,
                "source_tree_hash": source_tree_hash,
                "scope_contract_hash": scope_contract_hash,
                "dataset_hashes": dataset_hashes,
                "actual_input_receipts": {
                    key: dict(expected_actual_input_receipts[key])
                    for key in (INX_DATASET_KEY, DATASET_KEY)
                },
                "scientific_side_inputs": side_inputs,
                "paid_api_calls": 0,
                "network_calls": 0,
                "historical_artifacts_consumed": [],
                "worktree_clean_at_run_start": True,
            },
        )
        late = _late_revalidate(
            config_path,
            config_hash=config_hash,
            expected_receipts=expected_actual_input_receipts,
            expected_side_inputs=expected_side_input_hashes,
            acquisition_manifest_path=acquisition_path,
            git_commit=git_commit,
            expected_source_tree_hash=source_tree_hash,
            allowed_untracked_root=run_root,
        )
        _write_json(staging / "prepublication_input_validation.json", late)
        _validate_portability_and_scope(staging)
        _write_artifact_manifest(staging, identity)
        _validate_portability_and_scope(staging)
        _write_atomic_stage_contract(
            staging,
            output,
            run_id=str(run_id),
            config_hash=config_hash,
            scientific_input_hash=scientific_input_hash,
            scope_contract_hash=scope_contract_hash,
            git_commit=git_commit,
            source_tree_hash=source_tree_hash,
            dataset_hashes=dataset_hashes,
            actual_input_receipts={
                key: dict(expected_actual_input_receipts[key])
                for key in (INX_DATASET_KEY, DATASET_KEY)
            },
            side_input_hashes=dict(expected_side_input_hashes),
            started_at=stage_started_at,
            elapsed_seconds=time.perf_counter() - stage_started_perf,
        )
        _validate_portability_and_scope(staging)
        if output.exists():
            output.rmdir()
        os.replace(staging, output)

    return {
        "output": output,
        "metadata": output / "external_replication_metadata.json",
        "artifact_manifest": output / "artifact_manifest.json",
        "interpretation": output / "external_replication_interpretation.md",
        "files": _all_files(output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical HRDataset_v14 independent mapped-target replication evidence."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--scientific-input-hash", required=True)
    parser.add_argument(
        "--manifest-inputs",
        required=True,
        help="JSON containing actual_input_receipts, side_input_hashes and source identities.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    payload = json.loads(Path(arguments.manifest_inputs).read_text(encoding="utf-8"))
    run(
        arguments.config,
        output_dir=arguments.output_dir,
        run_id=arguments.run_id,
        config_hash=arguments.config_hash,
        scientific_input_hash=arguments.scientific_input_hash,
        expected_actual_input_receipts=payload["actual_input_receipts"],
        expected_side_input_hashes=payload["side_input_hashes"],
        git_commit=payload["git_commit"],
        source_tree_hash=payload["source_tree_hash"],
        scope_contract_hash=payload["scope_contract_hash"],
        expected_git_worktree_dirty=bool(payload["git_worktree_dirty"]),
    )


if __name__ == "__main__":  # pragma: no cover - exercised by integration command
    main()


__all__ = ["HRDatasetStageError", "run"]
