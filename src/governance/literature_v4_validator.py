"""Validate and manifest the additive Round 2 core-method literature package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


DEFAULT_PACKAGE = Path("reports/research_log/major_revision_round2/LITERATURE_V4")
PAYLOAD_FILES = {
    "BIBTEX_CANDIDATES.bib",
    "CORE_METHOD_REFERENCES.csv",
    "NOVELTY_BOUNDARY.md",
    "README.md",
    "VERIFIED_CORE_METHOD_BIBLIOGRAPHY.md",
    "provenance_receipt.json",
}
EXPECTED_FILES = PAYLOAD_FILES | {"manifest.json"}
EXPECTED_COVERAGE = {
    "proportional_odds_and_cumulative_link",
    "cumulative_threshold_ordinal_classification",
    "ordinal_evaluation",
    "quadratic_weighted_kappa",
    "ranked_probability_score",
    "proper_probability_scoring",
    "random_forest",
    "xgboost",
    "lightgbm",
}
FIXED_NOVELTY = "The contribution is not the novelty of any single component, but their operational integration into a traceable evaluation contract."


class LiteratureV4ValidationError(RuntimeError):
    """Raised when the literature package violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LiteratureV4ValidationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_manifest(root: Path) -> dict[str, Any]:
    files = [
        {"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha256(root / name)}
        for name in sorted(PAYLOAD_FILES, key=str.lower)
    ]
    set_digest = hashlib.sha256(
        "".join(f"{row['path']}:{row['sha256']}:{row['bytes']}\n" for row in files).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "package": "round2_core_method_literature_v4",
        "file_count_excluding_manifest": len(files),
        "payload_set_sha256": set_digest,
        "files": files,
    }


def write_literature_v4_manifest(package: Path | str = DEFAULT_PACKAGE) -> Path:
    root = Path(package)
    present = {path.name for path in root.iterdir() if path.is_file()}
    _require(present in (PAYLOAD_FILES, EXPECTED_FILES), f"Literature payload inventory drifted: {sorted(present)}.")
    path = root / "manifest.json"
    path.write_text(json.dumps(_payload_manifest(root), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path


def validate_literature_v4(package: Path | str = DEFAULT_PACKAGE) -> dict[str, Any]:
    root = Path(package)
    files = {path.name for path in root.iterdir() if path.is_file()}
    _require(files == EXPECTED_FILES, f"Literature package is not closed-world: {sorted(files)}.")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_manifest = _payload_manifest(root)
    _require(manifest == expected_manifest, "Literature manifest content drifted.")

    references = pd.read_csv(root / "CORE_METHOD_REFERENCES.csv", keep_default_na=False)
    _require(len(references) == 9, "Expected exactly nine core-method references.")
    _require(references["reference_id"].is_unique and references["citation_key"].is_unique, "Reference identities are not unique.")
    _require(set(references["coverage"]) == EXPECTED_COVERAGE, "Core method coverage drifted.")
    _require(set(references["doi_status"]) == {"publisher_verified", "official_proceedings_no_doi"}, "DOI status vocabulary drifted.")
    lightgbm = references.loc[references["citation_key"] == "ke2017lightgbm"].iloc[0]
    _require(lightgbm["doi"] == "" and lightgbm["doi_status"] == "official_proceedings_no_doi", "LightGBM DOI boundary drifted.")
    _require(references["primary_url"].str.startswith("https://").all(), "A primary URL is not HTTPS.")

    bib = (root / "BIBTEX_CANDIDATES.bib").read_text(encoding="utf-8")
    bib_keys = re.findall(r"^@\w+\{([^,]+),", bib, flags=re.MULTILINE)
    _require(bib_keys == references["citation_key"].tolist(), "BibTeX keys/order differ from the reference table.")
    _require("doi =" not in bib[bib.index("@inproceedings{ke2017lightgbm"):], "A LightGBM DOI was invented.")

    novelty = (root / "NOVELTY_BOUNDARY.md").read_text(encoding="utf-8")
    _require(FIXED_NOVELTY in novelty, "Fixed novelty statement drifted.")
    lower = novelty.lower()
    _require('does not support “world first,” “first ever,”' in lower, "Prohibited novelty wording boundary is missing.")
    provenance = json.loads((root / "provenance_receipt.json").read_text(encoding="utf-8"))
    _require(provenance["paid_api_calls"] == 0, "Paid API receipt drifted.")
    _require(provenance["historical_phase4a_modified"] is False, "Phase 4A immutability receipt drifted.")
    _require(provenance["manuscript_bibliography_modified"] is False, "Bibliography gate receipt drifted.")
    return {
        "status": "passed",
        "reference_count": len(references),
        "coverage_count": len(EXPECTED_COVERAGE),
        "file_count": len(files),
        "payload_set_sha256": manifest["payload_set_sha256"],
        "paid_api_calls": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, nargs="?", default=DEFAULT_PACKAGE)
    parser.add_argument("--write-manifest", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.write_manifest:
        write_literature_v4_manifest(args.package)
    print(json.dumps(validate_literature_v4(args.package), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LiteratureV4ValidationError", "validate_literature_v4", "write_literature_v4_manifest"]
