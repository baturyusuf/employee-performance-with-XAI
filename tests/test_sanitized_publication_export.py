from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from src.governance import sanitized_publication_export as publication_export
from src.governance.manuscript_contract import sha256_file
from src.governance.sanitized_publication_export import (
    PublicationExportError,
    run,
    validate_receipt,
)
from src.utils.config_loader import PROJECT_ROOT, load_config


LOCAL_WORKBOOK = PROJECT_ROOT / "data/raw/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls"


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fixture_repository(tmp_path: Path, *, track_raw: bool = False) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    raw = root / "data/raw/local.csv"
    raw.parent.mkdir(parents=True)
    raw.write_text("employee_id,rating\nEMPLOYEE_SECRET_ROW,4\n", encoding="utf-8")
    (root / "README.md").write_text("# Sanitized fixture\n", encoding="utf-8")
    (root / ".gitignore").write_text("/data/raw/*\n", encoding="utf-8")
    contract = {
        "publication_export": {
            "schema_version": 1,
            "contract_status": "approved_d3_forward_tip_sanitation",
            "scope": "history_free_allowlisted_technical_export_without_local_hr_data",
            "archive_method": publication_export.ARCHIVE_METHOD,
            "receipt_path": "reports/receipt.json",
            "include_paths": ["README.md", "configs"],
            "required_tracked_paths": ["README.md"],
            "forbidden_tracked_globs": ["data/raw/*"],
            "allowed_placeholder_paths": [],
            "portable_content_prefixes": ["README.md", "configs/"],
            "maximum_archive_bytes": 1000000,
            "maximum_member_bytes": 100000,
            "local_preservation": [
                {
                    "path": "data/raw/local.csv",
                    "sha256": sha256_file(raw),
                    "size_bytes": raw.stat().st_size,
                    "category": "source_dataset",
                }
            ],
            "claim_boundary": "Fixture contract only.",
        }
    }
    _write_json(root / "configs/publication_export.yaml", contract)
    _git(root, "init")
    _git(root, "config", "user.name", "Codex Test")
    _git(root, "config", "user.email", "codex-test@example.invalid")
    _git(root, "add", ".gitignore", "README.md", "configs/publication_export.yaml")
    if track_raw:
        _git(root, "add", "-f", "data/raw/local.csv")
    _git(root, "commit", "-m", "fixture")
    return root


def test_history_free_export_preserves_local_data_without_exporting_values(tmp_path: Path) -> None:
    root = _fixture_repository(tmp_path)
    receipt = run(
        "configs/publication_export.yaml",
        output_path="reports/receipt.json",
        project_root=root,
    )
    receipt_path = root / "reports/receipt.json"
    serialized = receipt_path.read_text(encoding="utf-8")
    assert receipt["status"] == "passed"
    assert receipt["canonical_eligible"] is True
    assert receipt["local_preserved_file_count"] == 1
    assert receipt["local_files_preserved"] is True
    assert receipt["current_tip_raw_paths_absent"] is True
    assert receipt["archive_retained"] is False
    assert receipt["history_included"] is False
    assert receipt["git_metadata_present"] is False
    assert receipt["member_count"] == receipt["allowlisted_member_count"] == 2
    assert "EMPLOYEE_SECRET_ROW" not in serialized
    validate_receipt(receipt_path, project_root=root, rebuild_archive=True)


def test_export_fails_when_forbidden_raw_path_is_still_tracked(tmp_path: Path) -> None:
    root = _fixture_repository(tmp_path, track_raw=True)
    with pytest.raises(PublicationExportError, match="still tracks forbidden"):
        run(
            "configs/publication_export.yaml",
            output_path="reports/receipt.json",
            project_root=root,
        )
    assert not (root / "reports/receipt.json").exists()


def test_archive_rejects_path_traversal_member(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../raw.csv", "employee data")
    with pytest.raises(PublicationExportError, match="unsafe"):
        publication_export._archive_evidence(
            archive,
            expected_paths=("../raw.csv",),
            forbidden_patterns=(),
            allowed_placeholders=(),
            portable_content_prefixes=(),
            maximum_archive_bytes=10000,
            maximum_member_bytes=10000,
        )


def test_cli_defaults_are_portable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["sanitized-publication-export"])
    arguments = publication_export.parse_args()
    assert arguments.config == "configs/publication_export.yaml"
    assert arguments.output == (
        "reports/research_log/finalization_v2/12_publication_export_receipt.json"
    )
    assert "\\" not in arguments.config
    assert "\\" not in arguments.output


@pytest.mark.skipif(not LOCAL_WORKBOOK.is_file(), reason="requires preserved ignored local source files")
def test_repository_local_preservation_contract_matches_bytes_and_git_index() -> None:
    config = load_config(PROJECT_ROOT / "configs/publication_export.yaml")["publication_export"]
    entries = config["local_preservation"]
    assert len(entries) == 14
    for entry in entries:
        path = PROJECT_ROOT / entry["path"]
        assert path.is_file()
        assert path.stat().st_size == entry["size_bytes"]
        assert sha256_file(path) == entry["sha256"]
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", entry["path"]],
            cwd=PROJECT_ROOT,
            capture_output=True,
        )
        assert tracked.returncode != 0
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", entry["path"]],
            cwd=PROJECT_ROOT,
        )
        assert ignored.returncode == 0
    for path in config["required_tracked_paths"]:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", path],
            cwd=PROJECT_ROOT,
            capture_output=True,
        )
        assert tracked.returncode == 0
