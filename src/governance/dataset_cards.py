from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import pandas as pd

from src.governance.manuscript_contract import canonical_config_hash
from src.utils.config_loader import PROJECT_ROOT, load_config


DEFAULT_MANUSCRIPT_CONFIG = Path("configs/manuscript_final.yaml")
DEFAULT_PROVENANCE_CONFIG = Path("configs/dataset_provenance.yaml")
REQUIRED_CARD_FIELDS = (
    "dataset_id",
    "canonical_name",
    "raw_file_path",
    "retrieval_url",
    "retrieval_date",
    "sha256",
    "row_count",
    "column_count",
    "target_column",
    "target_mapping",
    "target_mapping_note",
    "role",
    "allowed_claim",
    "task_type",
    "known_source_mirror_status",
    "source_authenticity_verification_status",
    "licence",
    "licence_verification_status",
    "citation_source",
    "citation_verification_status",
    "unresolved_manual_verification_items",
    "run_id",
    "config_hash",
    "provenance_config_hash",
)
MANUAL_REVIEW_MARKER = "manual_review_required"


class DatasetCardValidationError(ValueError):
    """Raised when a required dataset-card field or provenance check fails."""


def _root_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key, payload)
    if not isinstance(value, Mapping):
        raise DatasetCardValidationError(f"{key} config root must be a mapping.")
    return value


def _resolve(path: str | Path, project_root: str | Path = PROJECT_ROOT) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(project_root) / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_mapping_hash(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Mapping):
        return len(value) == 0
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    return False


def _raw_value_key(value: Any) -> str:
    if pd.isna(value):
        return "__MISSING__"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _load_frame(path: Path, delimiter: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, sep=delimiter)
    except Exception as exc:
        raise DatasetCardValidationError(f"Cannot parse dataset {path}: {exc}") from exc
    if frame.shape[1] < 2:
        raise DatasetCardValidationError(
            f"Dataset {path} parsed as fewer than two columns with delimiter {delimiter!r}."
        )
    return frame


def _mapping_support(
    frame: pd.DataFrame,
    target_column: str,
    mapping: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], list[str]]:
    if target_column not in frame:
        return [], [f"raw target column is absent: {target_column}"]
    normalized_mapping = {str(key): value for key, value in mapping.items()}
    counts = frame[target_column].map(_raw_value_key).value_counts(dropna=False).sort_index()
    support: list[Dict[str, Any]] = []
    unmapped: list[str] = []
    for raw_value, count in counts.items():
        mapped = normalized_mapping.get(str(raw_value))
        if str(raw_value) not in normalized_mapping:
            unmapped.append(str(raw_value))
        support.append(
            {
                "raw_value": str(raw_value),
                "raw_count": int(count),
                "mapped_value": mapped,
            }
        )
    return support, unmapped


def build_dataset_cards(
    manuscript_config_path: str | Path = DEFAULT_MANUSCRIPT_CONFIG,
    provenance_config_path: str | Path = DEFAULT_PROVENANCE_CONFIG,
    *,
    run_id: str,
    config_hash: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
    dataset_keys: Iterable[str] | None = None,
) -> list[Dict[str, Any]]:
    manuscript_raw = load_config(manuscript_config_path)
    manuscript = _root_mapping(manuscript_raw, "manuscript_final")
    provenance_raw = load_config(provenance_config_path)
    provenance = _root_mapping(provenance_raw, "dataset_provenance")
    config_hash = config_hash or canonical_config_hash(manuscript_raw)
    provenance_config_hash = _canonical_mapping_hash(provenance_raw)
    datasets = manuscript.get("datasets", {})
    physical_sources = provenance.get("physical_sources", {})
    bindings = provenance.get("dataset_bindings", {})
    marker = str(provenance.get("manual_review_marker", MANUAL_REVIEW_MARKER))
    if not isinstance(datasets, Mapping) or not datasets:
        raise DatasetCardValidationError("Canonical manuscript config contains no datasets.")
    if set(datasets) != set(bindings):
        missing = sorted(set(datasets).difference(bindings))
        extra = sorted(set(bindings).difference(datasets))
        raise DatasetCardValidationError(
            f"Dataset provenance bindings must exactly cover canonical datasets; missing={missing}, extra={extra}."
        )
    selected = list(datasets) if dataset_keys is None else [str(value) for value in dataset_keys]
    if not selected or len(selected) != len(set(selected)):
        raise DatasetCardValidationError("dataset_keys must be non-empty and contain no duplicates.")
    unknown = sorted(set(selected).difference(datasets))
    if unknown:
        raise DatasetCardValidationError(
            f"dataset_keys contains datasets outside the canonical config: {unknown}."
        )

    cards: list[Dict[str, Any]] = []
    frame_cache: Dict[str, pd.DataFrame] = {}
    hash_cache: Dict[str, str] = {}
    for dataset_id in selected:
        dataset_definition = datasets[dataset_id]
        if not isinstance(dataset_definition, Mapping):
            raise DatasetCardValidationError(f"Canonical dataset {dataset_id} must be a mapping.")
        binding = bindings[dataset_id]
        if not isinstance(binding, Mapping):
            raise DatasetCardValidationError(f"Dataset binding {dataset_id} must be a mapping.")
        source_id = str(binding.get("physical_source", ""))
        source = physical_sources.get(source_id)
        if not isinstance(source, Mapping):
            raise DatasetCardValidationError(
                f"Dataset {dataset_id} references missing physical source {source_id!r}."
            )
        raw_file_path = str(source.get("raw_file_path", ""))
        raw_path = _resolve(raw_file_path, project_root)
        if not raw_path.is_file():
            raise DatasetCardValidationError(f"Raw dataset is missing: {raw_path}")
        if source_id not in frame_cache:
            frame_cache[source_id] = _load_frame(raw_path, str(source.get("csv_delimiter", ",")))
            hash_cache[source_id] = _sha256(raw_path)
        frame = frame_cache[source_id]
        target_column = str(binding.get("raw_target", dataset_definition.get("target", "")))
        target_mapping = binding.get("target_mapping", {})
        if not isinstance(target_mapping, Mapping):
            raise DatasetCardValidationError(f"Target mapping for {dataset_id} must be a mapping.")
        support, unmapped = _mapping_support(frame, target_column, target_mapping)
        canonical_name = str(source.get("canonical_name", dataset_id))
        if dataset_id.endswith("_attrition"):
            canonical_name += " (attrition related-task binding)"
        unresolved = [str(item) for item in source.get("unresolved_manual_verification_items", [])]
        card: Dict[str, Any] = {
            "dataset_id": str(dataset_id),
            "physical_source_id": source_id,
            "canonical_name": canonical_name,
            "raw_file_path": raw_file_path,
            "retrieval_url": source.get("retrieval_url", marker),
            "retrieval_date": source.get("retrieval_date", marker),
            "repository_first_recorded_date": source.get("repository_first_recorded_date", marker),
            "sha256": hash_cache[source_id],
            "row_count": int(frame.shape[0]),
            "column_count": int(frame.shape[1]),
            "target_column": target_column,
            "target_mapping": dict(target_mapping),
            "target_mapping_note": binding.get("target_mapping_note", ""),
            "target_mapping_support": support,
            "unmapped_observed_target_values": unmapped,
            "role": dataset_definition.get("role", ""),
            "allowed_claim": dataset_definition.get("allowed_claim", ""),
            "task_type": dataset_definition.get("task_type", ""),
            "known_source_mirror_status": source.get("known_source_mirror_status", marker),
            "source_authenticity_verification_status": source.get(
                "source_authenticity_verification_status", marker
            ),
            "licence": source.get("licence", marker),
            "licence_verification_status": source.get("licence_verification_status", marker),
            "citation_source": source.get("citation_source", marker),
            "citation_verification_status": source.get("citation_verification_status", marker),
            "unresolved_manual_verification_items": unresolved,
            "narrative_dataset_card": source.get("narrative_dataset_card", ""),
            "manual_review_marker": marker,
            "run_id": run_id,
            "config_hash": config_hash,
            "provenance_config_hash": provenance_config_hash,
        }
        cards.append(card)
    return cards


def validate_dataset_card_record(
    card: Mapping[str, Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> list[str]:
    errors: list[str] = []
    dataset_id = str(card.get("dataset_id", "<unknown>"))
    for field in REQUIRED_CARD_FIELDS:
        if field not in card or _is_blank(card.get(field)):
            errors.append(f"{dataset_id}: required field is blank: {field}")
    digest = str(card.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        errors.append(f"{dataset_id}: sha256 is not a lowercase SHA-256 digest")
    for field in ("row_count", "column_count"):
        value = card.get(field)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"{dataset_id}: {field} must be a positive integer")
    raw_path_value = card.get("raw_file_path")
    if isinstance(raw_path_value, str) and raw_path_value:
        raw_path = _resolve(raw_path_value, project_root)
        if not raw_path.is_file():
            errors.append(f"{dataset_id}: raw file is missing: {raw_path}")
        elif re.fullmatch(r"[0-9a-f]{64}", digest) and _sha256(raw_path) != digest:
            errors.append(f"{dataset_id}: raw file hash does not match dataset card")
    if card.get("unmapped_observed_target_values"):
        errors.append(
            f"{dataset_id}: observed target values are absent from the declared mapping: "
            f"{card.get('unmapped_observed_target_values')}"
        )
    support = card.get("target_mapping_support")
    if not isinstance(support, list) or not support:
        errors.append(f"{dataset_id}: target mapping support is empty")
    else:
        support_total = sum(int(row.get("raw_count", 0)) for row in support if isinstance(row, Mapping))
        if support_total != card.get("row_count"):
            errors.append(
                f"{dataset_id}: target support count {support_total} does not equal row count {card.get('row_count')}"
            )
    manual_fields = (
        "retrieval_url",
        "retrieval_date",
        "source_authenticity_verification_status",
        "licence",
        "licence_verification_status",
        "citation_source",
        "citation_verification_status",
    )
    marker = str(card.get("manual_review_marker", MANUAL_REVIEW_MARKER))
    if any(marker in str(card.get(field, "")) for field in manual_fields):
        unresolved = card.get("unresolved_manual_verification_items")
        if not isinstance(unresolved, list) or not unresolved:
            errors.append(
                f"{dataset_id}: manual-review provenance fields require unresolved_manual_verification_items"
            )
    run_id = card.get("run_id")
    config_hash = str(card.get("config_hash", ""))
    provenance_config_hash = str(card.get("provenance_config_hash", ""))
    if not isinstance(run_id, str) or not run_id:
        errors.append(f"{dataset_id}: run_id must be non-empty")
    if not re.fullmatch(r"[0-9a-f]{64}", config_hash):
        errors.append(f"{dataset_id}: config_hash is not a lowercase SHA-256 digest")
    if not re.fullmatch(r"[0-9a-f]{64}", provenance_config_hash):
        errors.append(f"{dataset_id}: provenance_config_hash is not a lowercase SHA-256 digest")
    return errors


def validate_dataset_cards(
    cards: Sequence[Mapping[str, Any]],
    *,
    expected_dataset_ids: Iterable[str] | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> None:
    errors: list[str] = []
    identifiers = [str(card.get("dataset_id", "")) for card in cards]
    if len(set(identifiers)) != len(identifiers):
        errors.append("dataset card identifiers are not unique")
    if expected_dataset_ids is not None:
        expected = set(str(value) for value in expected_dataset_ids)
        observed = set(identifiers)
        if expected != observed:
            errors.append(
                f"dataset cards do not exactly cover canonical datasets; missing={sorted(expected-observed)}, "
                f"extra={sorted(observed-expected)}"
            )
    identities = {
        (card.get("run_id"), card.get("config_hash"), card.get("provenance_config_hash"))
        for card in cards
    }
    if len(identities) != 1:
        errors.append("dataset cards do not share one run_id/config_hash identity")
    for card in cards:
        errors.extend(validate_dataset_card_record(card, project_root=project_root))
    if errors:
        raise DatasetCardValidationError("Invalid dataset cards:\n- " + "\n- ".join(errors))


def _card_markdown(card: Mapping[str, Any]) -> str:
    support_lines = [
        f"- `{row['raw_value']}`: n={row['raw_count']} -> `{row['mapped_value']}`"
        for row in card["target_mapping_support"]
    ]
    unresolved_lines = [
        f"- {item}" for item in card.get("unresolved_manual_verification_items", [])
    ]
    lines = [
        f"# Dataset Card: {card['canonical_name']}",
        "",
        f"Run ID: `{card['run_id']}`  ",
        f"Config hash: `{card['config_hash']}`",
        f"Provenance config hash: `{card['provenance_config_hash']}`",
        "",
        "## File identity",
        "",
        f"- Raw file: `{card['raw_file_path']}`",
        f"- SHA-256: `{card['sha256']}`",
        f"- Shape: {card['row_count']} rows x {card['column_count']} columns",
        "",
        "## Role and claim boundary",
        "",
        f"- Role: `{card['role']}`",
        f"- Task type: `{card['task_type']}`",
        f"- Allowed claim: {card['allowed_claim']}",
        "",
        "## Target mapping and observed support",
        "",
        f"Raw target: `{card['target_column']}`. {card['target_mapping_note']}",
        "",
        *support_lines,
        "",
        "## Source, mirror, and licence status",
        "",
        f"- Retrieval URL: `{card['retrieval_url']}`",
        f"- Retrieval date: `{card['retrieval_date']}`",
        f"- Known source/mirror status: `{card['known_source_mirror_status']}`",
        f"- Source-authenticity status: `{card['source_authenticity_verification_status']}`",
        f"- Licence: `{card['licence']}`",
        f"- Licence verification: `{card['licence_verification_status']}`",
        f"- Citation/source: {card['citation_source']}",
        f"- Citation verification: `{card['citation_verification_status']}`",
        "",
        "## Unresolved manual verification",
        "",
        *(unresolved_lines or ["No unresolved items recorded."]),
        "",
        "Automated hashing and schema checks do not authenticate the upstream source or determine legal reuse rights. `manual_review_required` fields must be resolved by the author or another authorised reviewer before making stronger provenance or licence claims.",
    ]
    return "\n".join(lines) + "\n"


def _summary_frame(cards: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for card in cards:
        rows.append(
            {
                "run_id": card["run_id"],
                "config_hash": card["config_hash"],
                "provenance_config_hash": card["provenance_config_hash"],
                "dataset_id": card["dataset_id"],
                "canonical_name": card["canonical_name"],
                "raw_file_path": card["raw_file_path"],
                "retrieval_url": card["retrieval_url"],
                "retrieval_date": card["retrieval_date"],
                "sha256": card["sha256"],
                "row_count": card["row_count"],
                "column_count": card["column_count"],
                "target_column": card["target_column"],
                "target_mapping": json.dumps(card["target_mapping"], sort_keys=True),
                "target_mapping_support": json.dumps(card["target_mapping_support"], sort_keys=True),
                "role": card["role"],
                "allowed_claim": card["allowed_claim"],
                "task_type": card["task_type"],
                "known_source_mirror_status": card["known_source_mirror_status"],
                "source_authenticity_verification_status": card[
                    "source_authenticity_verification_status"
                ],
                "licence": card["licence"],
                "licence_verification_status": card["licence_verification_status"],
                "citation_source": card["citation_source"],
                "citation_verification_status": card["citation_verification_status"],
                "unresolved_manual_verification_items": json.dumps(
                    card["unresolved_manual_verification_items"], ensure_ascii=False
                ),
            }
        )
    return pd.DataFrame(rows)


def run(
    manuscript_config_path: str | Path = DEFAULT_MANUSCRIPT_CONFIG,
    provenance_config_path: str | Path = DEFAULT_PROVENANCE_CONFIG,
    *,
    output_dir: str | Path,
    run_id: str,
    config_hash: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
    dataset_keys: Iterable[str] | None = None,
) -> Dict[str, Path]:
    root = Path(project_root).resolve()

    def repository_relative(path: str | Path, *, field: str) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise DatasetCardValidationError(
                f"{field} must remain inside the repository for portable metadata."
            ) from exc

    manuscript_raw = load_config(manuscript_config_path)
    manuscript = _root_mapping(manuscript_raw, "manuscript_final")
    config_hash = config_hash or canonical_config_hash(manuscript_raw)
    selected_dataset_keys = (
        None if dataset_keys is None else tuple(str(value) for value in dataset_keys)
    )
    cards = build_dataset_cards(
        manuscript_config_path,
        provenance_config_path,
        run_id=run_id,
        config_hash=config_hash,
        project_root=project_root,
        dataset_keys=selected_dataset_keys,
    )
    expected_dataset_ids = (
        manuscript.get("datasets", {}).keys()
        if selected_dataset_keys is None
        else selected_dataset_keys
    )
    validate_dataset_cards(
        cards,
        expected_dataset_ids=expected_dataset_ids,
        project_root=project_root,
    )
    output = Path(output_dir)
    cards_dir = output / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "dataset_cards_json": output / "dataset_cards.json",
        "dataset_cards_csv": output / "dataset_cards.csv",
        "validation_report": output / "dataset_card_validation_report.json",
        "metadata": output / "metadata.json",
    }
    outputs["dataset_cards_json"].write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "provenance_config_hash": cards[0]["provenance_config_hash"],
                "dataset_keys": [card["dataset_id"] for card in cards],
                "cards": cards,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _summary_frame(cards).to_csv(outputs["dataset_cards_csv"], index=False)
    markdown_paths: list[str] = []
    for card in cards:
        path = cards_dir / f"{card['dataset_id']}.md"
        path.write_text(_card_markdown(card), encoding="utf-8")
        markdown_paths.append(path.relative_to(output).as_posix())
    outputs["validation_report"].write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "provenance_config_hash": cards[0]["provenance_config_hash"],
                "dataset_keys": [card["dataset_id"] for card in cards],
                "status": "passed",
                "cards_validated": len(cards),
                "required_fields": list(REQUIRED_CARD_FIELDS),
                "manual_source_or_licence_authenticity_decisions_made": False,
                "manual_review_card_count": sum(
                    MANUAL_REVIEW_MARKER in json.dumps(card, sort_keys=True) for card in cards
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["metadata"].write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "provenance_config_hash": cards[0]["provenance_config_hash"],
                "dataset_keys": [card["dataset_id"] for card in cards],
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "manuscript_config": repository_relative(
                    manuscript_config_path,
                    field="manuscript_config_path",
                ),
                "provenance_config": repository_relative(
                    provenance_config_path,
                    field="provenance_config_path",
                ),
                "physical_source_count": len({card["physical_source_id"] for card in cards}),
                "logical_dataset_role_count": len(cards),
                "markdown_cards": markdown_paths,
                "claim_boundary": (
                    "Automated checks verify repository-local files, hashes, shape, mappings, and required fields; "
                    "they do not authenticate upstream sources or determine licence validity."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and validate canonical dataset cards.")
    parser.add_argument("--config", type=Path, default=DEFAULT_MANUSCRIPT_CONFIG)
    parser.add_argument("--provenance-config", type=Path, default=DEFAULT_PROVENANCE_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        run(
            arguments.config,
            arguments.provenance_config,
            output_dir=arguments.output_dir,
            run_id=arguments.run_id,
            config_hash=arguments.config_hash,
        )
    )
