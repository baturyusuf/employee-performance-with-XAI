from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.data.load_data import _read_csv_with_best_effort
from src.utils.config import SETTINGS


EXTERNAL_DATA_ROOT = SETTINGS.project_root / "data" / "external"


@dataclass(frozen=True)
class ExternalDatasetConfig:
    dataset_name: str
    display_name: str
    recommended_role: str
    source_url: str
    canonical_source_note: str
    target: Dict[str, Any]
    rename_columns: Dict[str, str]
    id_columns: List[str]
    leakage_risk_columns: List[str]
    sensitive_audit_only_columns: List[str]
    proxy_risk_columns: List[str]
    feature_policy_variants: Dict[str, Dict[str, Any]]
    optional_attrition_target: Optional[Dict[str, Any]] = None
    date_columns: Optional[List[str]] = None
    derived_features: Optional[Dict[str, Dict[str, Any]]] = None

    @property
    def dataset_dir(self) -> Path:
        return EXTERNAL_DATA_ROOT / self.dataset_name

    @property
    def raw_path(self) -> Path:
        return self.dataset_dir / "raw.csv"


@dataclass
class ExternalDataset:
    config: ExternalDatasetConfig
    raw: pd.DataFrame
    canonical: pd.DataFrame
    target_column: str
    target_raw_column: str
    task_type: str
    target_mapping: Dict[str, Any]
    unmapped_target_values: List[str]

    @property
    def labels(self) -> List[int]:
        values = pd.to_numeric(self.canonical[self.target_column], errors="coerce").dropna().astype(int)
        return sorted(values.unique().tolist())


def available_dataset_names() -> List[str]:
    if not EXTERNAL_DATA_ROOT.exists():
        return []
    return sorted(
        path.name
        for path in EXTERNAL_DATA_ROOT.iterdir()
        if path.is_dir() and (path / "schema_mapping.json").exists()
    )


def load_external_config(dataset_name: str) -> ExternalDatasetConfig:
    path = EXTERNAL_DATA_ROOT / dataset_name / "schema_mapping.json"
    if not path.exists():
        raise FileNotFoundError(f"External schema mapping not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExternalDatasetConfig(
        dataset_name=str(payload["dataset_name"]),
        display_name=str(payload.get("display_name", payload["dataset_name"])),
        recommended_role=str(payload.get("recommended_role", "")),
        source_url=str(payload.get("source_url", "")),
        canonical_source_note=str(payload.get("canonical_source_note", "")),
        target=dict(payload["target"]),
        optional_attrition_target=payload.get("optional_attrition_target"),
        rename_columns={str(k): str(v) for k, v in payload.get("rename_columns", {}).items()},
        id_columns=[str(v) for v in payload.get("id_columns", [])],
        date_columns=[str(v) for v in payload.get("date_columns", [])],
        derived_features=payload.get("derived_features", {}),
        leakage_risk_columns=[str(v) for v in payload.get("leakage_risk_columns", [])],
        sensitive_audit_only_columns=[str(v) for v in payload.get("sensitive_audit_only_columns", [])],
        proxy_risk_columns=[str(v) for v in payload.get("proxy_risk_columns", [])],
        feature_policy_variants=dict(payload.get("feature_policy_variants", {})),
    )


def load_external_dataset(dataset_name: str, target_kind: str = "primary") -> ExternalDataset:
    config = load_external_config(dataset_name)
    if not config.raw_path.exists():
        raise FileNotFoundError(f"External raw CSV not found: {config.raw_path}")

    raw = _read_csv_with_best_effort(config.raw_path)
    raw = _strip_column_names(raw)
    target_spec = _target_spec_for(config, target_kind)
    raw_target = str(target_spec["raw_column"])
    canonical_target = str(target_spec["canonical_column"])

    if raw_target not in raw.columns:
        raise ValueError(
            f"Required target column '{raw_target}' missing for dataset '{dataset_name}'. "
            f"Available columns: {raw.columns.tolist()}"
        )

    canonical = raw.copy()
    rename_map = {src: dst for src, dst in config.rename_columns.items() if src in canonical.columns}
    canonical = canonical.rename(columns=rename_map)
    canonical = _strip_string_values(canonical)
    canonical["ExternalSampleId"] = np.arange(len(canonical), dtype=int)
    _add_derived_features(canonical, raw, config)

    # If the target was renamed by the general mapping, use the renamed source.
    source_after_rename = config.rename_columns.get(raw_target, raw_target)
    if source_after_rename not in canonical.columns:
        source_after_rename = raw_target

    mapped_target, unmapped_values = _map_target(
        canonical[source_after_rename],
        target_spec.get("mapping", {}),
        dataset_name=dataset_name,
        raw_column=raw_target,
    )
    canonical[canonical_target] = mapped_target
    if canonical[canonical_target].isna().any():
        missing_count = int(canonical[canonical_target].isna().sum())
        raise ValueError(
            f"Target mapping produced {missing_count} null values for dataset '{dataset_name}'. "
            f"Unmapped values: {unmapped_values}"
        )
    canonical[canonical_target] = canonical[canonical_target].astype(int)

    return ExternalDataset(
        config=config,
        raw=raw,
        canonical=canonical,
        target_column=canonical_target,
        target_raw_column=raw_target,
        task_type=str(target_spec.get("task_type", "unknown")),
        target_mapping={str(k): v for k, v in target_spec.get("mapping", {}).items()},
        unmapped_target_values=unmapped_values,
    )


def build_feature_columns(
    dataset: ExternalDataset,
    policy_name: str,
    *,
    exclude_sensitive: bool = True,
    exclude_leakage: bool = True,
) -> List[str]:
    config = dataset.config
    if policy_name not in config.feature_policy_variants:
        raise ValueError(
            f"Unknown feature policy '{policy_name}' for dataset '{config.dataset_name}'. "
            f"Allowed: {sorted(config.feature_policy_variants)}"
        )

    df = dataset.canonical
    policy = config.feature_policy_variants[policy_name]
    excluded = set(always_excluded_columns(dataset))
    if exclude_leakage:
        excluded.update(role_columns(dataset, "leakage"))
    if exclude_sensitive:
        excluded.update(role_columns(dataset, "sensitive"))
    excluded.update(_canonicalize_columns(config, policy.get("exclude_columns", [])))

    # Exclude non-model date/text/name fields unless explicitly mapped as safe derived features.
    excluded.update([col for col in df.columns if col.lower().endswith("date")])
    excluded.update([col for col in ["Employee_Name", "ManagerName"] if col in df.columns])

    columns = [
        col
        for col in df.columns
        if col not in excluded and col != dataset.target_column and _is_model_usable(df[col])
    ]
    if not columns:
        raise ValueError(f"No model features remain for {config.dataset_name}/{policy_name}.")
    return columns


def role_columns(dataset: ExternalDataset, role: str) -> List[str]:
    config = dataset.config
    if role == "id":
        values = list(config.id_columns) + ["ExternalSampleId", "EmpNumber"]
    elif role == "leakage":
        values = list(config.leakage_risk_columns) + [dataset.target_column, dataset.target_raw_column]
    elif role == "sensitive":
        values = list(config.sensitive_audit_only_columns)
    elif role == "proxy":
        values = list(config.proxy_risk_columns)
    else:
        raise ValueError(f"Unknown role: {role}")
    return sorted(set(col for col in _canonicalize_columns(config, values) if col in dataset.canonical.columns))


def always_excluded_columns(dataset: ExternalDataset) -> List[str]:
    columns = set(role_columns(dataset, "id"))
    columns.add(dataset.target_column)
    columns.add(dataset.target_raw_column)
    return sorted(col for col in columns if col in dataset.canonical.columns)


def audit_attribute_columns(dataset: ExternalDataset) -> List[str]:
    candidates = role_columns(dataset, "sensitive") + role_columns(dataset, "proxy")
    preferred = [
        "Gender",
        "RaceEthnicity",
        "HispanicLatino",
        "MaritalStatus",
        "EmpDepartment",
        "EmpJobRole",
        "SalaryBand",
    ]
    ordered = [col for col in preferred if col in candidates and col in dataset.canonical.columns]
    ordered.extend([col for col in candidates if col not in ordered and col in dataset.canonical.columns])
    return ordered


def feature_mapping_rows(dataset: ExternalDataset) -> pd.DataFrame:
    config = dataset.config
    rows: List[Dict[str, Any]] = []
    raw_columns = set(dataset.raw.columns)
    for raw_col in sorted(raw_columns):
        canonical = config.rename_columns.get(raw_col, raw_col)
        roles = []
        if raw_col == dataset.target_raw_column or canonical == dataset.target_column:
            roles.append("target")
        if raw_col in config.id_columns or canonical in role_columns(dataset, "id"):
            roles.append("id")
        if raw_col in config.leakage_risk_columns or canonical in role_columns(dataset, "leakage"):
            roles.append("leakage_risk")
        if raw_col in config.sensitive_audit_only_columns or canonical in role_columns(dataset, "sensitive"):
            roles.append("sensitive_audit_only")
        if raw_col in config.proxy_risk_columns or canonical in role_columns(dataset, "proxy"):
            roles.append("proxy_risk")
        rows.append(
            {
                "raw_column": raw_col,
                "canonical_column": canonical,
                "present": True,
                "role": ";".join(roles) if roles else "candidate_feature_or_unused",
            }
        )
    for raw_col, canonical in sorted(config.rename_columns.items()):
        if raw_col not in raw_columns:
            rows.append(
                {
                    "raw_column": raw_col,
                    "canonical_column": canonical,
                    "present": False,
                    "role": "mapping_not_observed",
                }
            )
    return pd.DataFrame(rows)


def target_distribution(dataset: ExternalDataset) -> pd.DataFrame:
    counts = dataset.canonical[dataset.target_column].value_counts(dropna=False).sort_index()
    total = len(dataset.canonical)
    return pd.DataFrame(
        {
            "target_column": dataset.target_column,
            "target_value": [str(v) for v in counts.index],
            "count": counts.astype(int).values,
            "proportion": [float(v / total) for v in counts.values],
        }
    )


def leakage_sensitive_proxy_rows(dataset: ExternalDataset) -> pd.DataFrame:
    rows = []
    for role in ["id", "leakage", "sensitive", "proxy"]:
        for col in role_columns(dataset, role):
            rows.append(
                {
                    "column": col,
                    "role": role,
                    "present": col in dataset.canonical.columns,
                    "used_as_model_feature_by_default": False if role in {"id", "leakage", "sensitive"} else "policy_dependent",
                }
            )
    return pd.DataFrame(rows).sort_values(["role", "column"])


def _target_spec_for(config: ExternalDatasetConfig, target_kind: str) -> Dict[str, Any]:
    if target_kind == "primary":
        return config.target
    if target_kind == "attrition" and config.optional_attrition_target is not None:
        return config.optional_attrition_target
    raise ValueError(f"Dataset '{config.dataset_name}' does not define target kind '{target_kind}'.")


def _strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).replace("\ufeff", "").strip() for col in out.columns]
    return out


def _strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or str(out[col].dtype).startswith("string"):
            out[col] = out[col].astype("string").str.strip()
    return out


def _add_derived_features(canonical: pd.DataFrame, raw: pd.DataFrame, config: ExternalDatasetConfig) -> None:
    derived = config.derived_features or {}
    for feature_name, spec in derived.items():
        source = spec.get("from")
        reference = spec.get("reference")
        if not source or source not in raw.columns:
            continue
        start = pd.to_datetime(raw[source], errors="coerce")
        if reference and reference in raw.columns:
            end = pd.to_datetime(raw[reference], errors="coerce")
            fallback = end.dropna().max()
            end = end.fillna(fallback)
        else:
            end = pd.Series(pd.Timestamp("today").normalize(), index=raw.index)
        canonical[feature_name] = ((end - start).dt.days / 365.25).round(3)


def _map_target(
    series: pd.Series,
    mapping: Dict[str, Any],
    *,
    dataset_name: str,
    raw_column: str,
) -> tuple[pd.Series, List[str]]:
    if mapping:
        as_str = series.astype("string").str.strip()
        mapped = as_str.map({str(k): v for k, v in mapping.items()})
        unmapped = sorted(as_str[mapped.isna()].dropna().unique().tolist())
        return pd.to_numeric(mapped, errors="coerce"), unmapped

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        as_str = series.astype("string").str.strip()
        if set(as_str.dropna().unique().tolist()).issubset({"No", "Yes"}):
            mapped = as_str.map({"No": 0, "Yes": 1})
            return pd.to_numeric(mapped, errors="coerce"), []
        invalid = sorted(as_str[numeric.isna()].dropna().unique().tolist())
        raise ValueError(
            f"Target column '{raw_column}' in dataset '{dataset_name}' is not numeric and has no mapping. "
            f"Invalid values: {invalid}"
        )
    return numeric, []


def _canonicalize_columns(config: ExternalDatasetConfig, columns: Iterable[str]) -> List[str]:
    out = []
    for col in columns:
        value = str(col)
        out.append(config.rename_columns.get(value, value))
    return out


def _is_model_usable(series: pd.Series) -> bool:
    if series.isna().all():
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    if pd.api.types.is_object_dtype(series) or str(series.dtype).startswith("string"):
        # Very high-cardinality free text is not useful for this governance layer.
        non_null = series.dropna().astype(str)
        if len(non_null) and non_null.nunique() / len(non_null) > 0.80:
            return False
    return True
