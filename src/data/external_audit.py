from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.data.external_adapters import (
    ExternalDataset,
    audit_attribute_columns,
    available_dataset_names,
    feature_mapping_rows,
    leakage_sensitive_proxy_rows,
    load_external_dataset,
    target_distribution,
)
from src.utils.config import SETTINGS
from src.utils.experiment_registry import append_registry_row, get_git_commit, utc_now_iso


EXTERNAL_REPORT_ROOT = SETTINGS.reports_dir / "external_validation"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def dataset_report_dir(dataset_name: str, target_kind: str = "primary") -> Path:
    suffix = "" if target_kind == "primary" else f"_{target_kind}"
    return EXTERNAL_REPORT_ROOT / f"{dataset_name}{suffix}"


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": col,
                "missing_count": int(df[col].isna().sum()),
                "missing_rate": float(df[col].isna().mean()),
                "non_null_count": int(df[col].notna().sum()),
                "n_unique": int(df[col].nunique(dropna=True)),
                "dtype": str(df[col].dtype),
            }
            for col in df.columns
        ]
    ).sort_values(["missing_count", "column"], ascending=[False, True])


def duplicate_id_report(dataset: ExternalDataset) -> pd.DataFrame:
    rows = []
    id_candidates = ["EmpNumber", "ExternalSampleId"]
    for col in id_candidates:
        if col in dataset.canonical.columns:
            rows.append(
                {
                    "id_column": col,
                    "duplicate_id_count": int(dataset.canonical[col].duplicated().sum()),
                    "unique_id_count": int(dataset.canonical[col].nunique(dropna=True)),
                    "row_count": int(len(dataset.canonical)),
                }
            )
    if not rows:
        rows.append(
            {
                "id_column": "",
                "duplicate_id_count": "",
                "unique_id_count": "",
                "row_count": int(len(dataset.canonical)),
            }
        )
    return pd.DataFrame(rows)


def write_dataset_audit_markdown(
    dataset: ExternalDataset,
    output_path: Path,
    mapping_df: pd.DataFrame,
    target_df: pd.DataFrame,
    lsp_df: pd.DataFrame,
) -> None:
    config = dataset.config
    leakage = lsp_df[lsp_df["role"] == "leakage"]["column"].tolist()
    sensitive = lsp_df[lsp_df["role"] == "sensitive"]["column"].tolist()
    proxy = lsp_df[lsp_df["role"] == "proxy"]["column"].tolist()
    class_rows = [
        f"- {row.target_value}: {row.count} ({row.proportion:.4f})"
        for row in target_df.itertuples(index=False)
    ]
    mapping_rows = [
        f"- `{row.raw_column}` -> `{row.canonical_column}` ({row.role})"
        for row in mapping_df.itertuples(index=False)
        if bool(row.present) and row.role != "candidate_feature_or_unused"
    ]

    target_mapping = dataset.target_mapping or {"identity_numeric_mapping": True}
    lines = [
        f"# Dataset Audit: {config.display_name}",
        "",
        f"Dataset name: `{config.dataset_name}`",
        f"Recommended role: {config.recommended_role}",
        f"Task type: `{dataset.task_type}`",
        f"Source URL: `{config.source_url}`",
        f"Source note: {config.canonical_source_note}",
        "",
        "## Shape",
        "",
        f"- Rows: {len(dataset.canonical)}",
        f"- Raw columns: {dataset.raw.shape[1]}",
        f"- Canonical columns: {dataset.canonical.shape[1]}",
        "",
        "## Target",
        "",
        f"- Raw target column: `{dataset.target_raw_column}`",
        f"- Canonical target column: `{dataset.target_column}`",
        f"- Final target mapping: `{json.dumps(target_mapping, sort_keys=True)}`",
        "",
        "## Target Distribution",
        "",
        *class_rows,
        "",
        "## Feature Mapping Highlights",
        "",
    ]
    lines.extend(mapping_rows if mapping_rows else ["No mapped feature highlights available."])
    lines.extend(
        [
            "",
            "## Leakage-Risk Columns",
            "",
            ", ".join(f"`{col}`" for col in leakage) if leakage else "None detected from mapping.",
            "",
            "## Sensitive / Audit-Only Columns",
            "",
            ", ".join(f"`{col}`" for col in sensitive) if sensitive else "None detected from mapping.",
            "",
            "## Proxy-Risk Columns",
            "",
            ", ".join(f"`{col}`" for col in proxy) if proxy else "None detected from mapping.",
            "",
            "## Available Audit Attributes",
            "",
            ", ".join(f"`{col}`" for col in audit_attribute_columns(dataset)) or "None.",
            "",
            "## Claim Boundaries",
            "",
            "- This audit is schema and target evidence, not proof of fairness or causal validity.",
            "- SHAP explanations in downstream reports must be attribution-only.",
            "- Counterfactuals in downstream reports must be framed as model scenarios, not employee prescriptions.",
            "- This project remains research-grade decision support only and must not make autonomous HR decisions.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset_audit(
    dataset_name: str,
    *,
    target_kind: str = "primary",
    write_registry: bool = True,
) -> Dict[str, Path]:
    dataset = load_external_dataset(dataset_name, target_kind=target_kind)
    output_dir = dataset_report_dir(dataset_name, target_kind)
    ensure_dir(output_dir)

    mapping_df = feature_mapping_rows(dataset)
    target_df = target_distribution(dataset)
    lsp_df = leakage_sensitive_proxy_rows(dataset)
    missing_df = missingness_report(dataset.canonical)
    duplicate_df = duplicate_id_report(dataset)

    outputs = {
        "dataset_audit": output_dir / "dataset_audit.md",
        "schema_mapping": output_dir / "schema_mapping.csv",
        "target_distribution": output_dir / "target_distribution.csv",
        "leakage_sensitive_proxy_audit": output_dir / "leakage_sensitive_proxy_audit.csv",
        "missingness": output_dir / "missingness_report.csv",
        "duplicate_ids": output_dir / "duplicate_id_report.csv",
        "metadata": output_dir / "metadata.json",
    }
    mapping_df.to_csv(outputs["schema_mapping"], index=False)
    target_df.to_csv(outputs["target_distribution"], index=False)
    lsp_df.to_csv(outputs["leakage_sensitive_proxy_audit"], index=False)
    missing_df.to_csv(outputs["missingness"], index=False)
    duplicate_df.to_csv(outputs["duplicate_ids"], index=False)
    write_dataset_audit_markdown(dataset, outputs["dataset_audit"], mapping_df, target_df, lsp_df)
    save_json(
        {
            "dataset_name": dataset_name,
            "target_kind": target_kind,
            "display_name": dataset.config.display_name,
            "recommended_role": dataset.config.recommended_role,
            "row_count": int(len(dataset.canonical)),
            "raw_column_count": int(dataset.raw.shape[1]),
            "canonical_column_count": int(dataset.canonical.shape[1]),
            "target_column": dataset.target_column,
            "target_raw_column": dataset.target_raw_column,
            "task_type": dataset.task_type,
            "labels": dataset.labels,
            "source_url": dataset.config.source_url,
            "outputs": {name: str(path) for name, path in outputs.items() if name != "metadata"},
        },
        outputs["metadata"],
    )

    if write_registry:
        append_registry_row(
            {
                "run_id": f"external_dataset_audit_{dataset_name}_{target_kind}_{utc_now_iso()}",
                "date_time": utc_now_iso(),
                "git_commit_if_available": get_git_commit(),
                "script": "python -m src.data.external_audit",
                "config": f"data/external/{dataset_name}/schema_mapping.json",
                "feature_set": "not_applicable",
                "model": "not_applicable",
                "seed": "not_applicable",
                "cv_strategy": "not_applicable",
                "primary_metrics": {
                    "rows": len(dataset.canonical),
                    "labels": dataset.labels,
                    "task_type": dataset.task_type,
                },
                "output_dir": _rel(output_dir),
                "notes": "External dataset schema, target, leakage, sensitive, proxy, missingness, and duplicate audit.",
                "decision_status": "candidate",
            }
        )

    return outputs


def run_all_audits(dataset_names: Iterable[str] | None = None) -> Dict[str, Dict[str, Path]]:
    names = list(dataset_names) if dataset_names is not None else available_dataset_names()
    results = {}
    for name in names:
        results[name] = run_dataset_audit(name)
    return results


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(SETTINGS.project_root))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate external dataset audit reports.")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset names or 'all'.")
    parser.add_argument("--target-kind", default="primary", choices=["primary", "attrition"])
    parser.add_argument("--no-registry", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected = available_dataset_names() if args.datasets == "all" else [p.strip() for p in args.datasets.split(",") if p.strip()]
    paths = {
        name: run_dataset_audit(name, target_kind=args.target_kind, write_registry=not args.no_registry)
        for name in selected
    }
    print(paths)
