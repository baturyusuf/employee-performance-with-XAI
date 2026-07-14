from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.governance import ci_repository_gate as gate


ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github/workflows/ci.yml"
RELEASE_WORKFLOW = ROOT / ".github/workflows/validate-release-candidate.yml"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_pull_request_ci_is_locked_offline_and_complete() -> None:
    text = _text(CI_WORKFLOW)
    required = (
        "pull_request:",
        "finalization/leakage-aware-v2",
        "permissions:\n  contents: read",
        "actions/checkout@v7",
        "persist-credentials: false",
        "actions/setup-python@v6",
        'python-version: "3.14"',
        "python -m pip install --requirement requirements-dev.txt",
        "python -m src.governance.dependency_contract --profile development --validate-only",
        "python -m pip check",
        "tests/test_core_build_makes_no_network_or_api_calls.py",
        "tests/test_ci_workflow_contract.py",
        "python -m pytest -q",
        'python -m unittest discover -s tests -p "test_*.py"',
        "python -m compileall -q src",
        "python -m src.governance.ci_repository_gate",
        "git diff --exit-code",
    )
    assert all(value in text for value in required)
    for key in (
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "COHERE_API_KEY",
        "GOOGLE_API_KEY",
        "MISTRAL_API_KEY",
        "OPENAI_API_KEY",
    ):
        assert f'{key}: ""' in text


def test_manual_release_workflow_is_local_read_only_and_nonpublishing() -> None:
    text = _text(RELEASE_WORKFLOW)
    required = (
        "workflow_dispatch:",
        "run_id:",
        "expected_commit:",
        "VALIDATE_ONLY",
        "runs-on: [self-hosted, Windows, X64, publication-release]",
        "actions/checkout@v7",
        "clean: false",
        "persist-credentials: false",
        "actions/setup-python@v6",
        'python-version: "3.14"',
        "python -m pytest -q",
        "python -m unittest discover",
        "python -m compileall -q src",
        "python -m src.governance.ci_repository_gate --release-run-id $env:RELEASE_RUN_ID",
        "Require every local real-data integration input",
        "data/raw/inx_employee_performance.csv",
        "data/external/hrdataset_v14/raw.csv",
        "data/external/ibm_hr_analytics/raw.csv",
        "data/external/employee_turnover/raw.csv",
    )
    assert all(value in text for value in required)
    lowered = text.casefold()
    for prohibited in (
        "actions/upload-artifact",
        "git push",
        "gh release",
        "zenodo",
        "--promote-run-id",
        "build_manuscript_evidence --run-id",
    ):
        assert prohibited not in lowered


def test_actual_tracked_inventory_and_readme_pass_static_gates() -> None:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    paths = sorted(value for value in completed.stdout.split("\0") if value)
    inventory = gate._validate_tracked_inventory(ROOT, paths)
    readme = gate._validate_readme_links(ROOT)
    issues = gate._validate_issue_register(ROOT)

    assert inventory["tracked_file_count"] >= 1900
    assert inventory["raw_data_paths"] == []
    assert inventory["environment_paths"] == []
    assert inventory["large_paths"] == []
    assert inventory["secret_paths"] == []
    assert inventory["machine_path_findings"] == []
    assert readme["local_link_count"] >= 20
    assert issues["row_count"] >= 32


def test_release_run_id_must_be_portable() -> None:
    with pytest.raises(gate.CIRepositoryGateError, match="not portable"):
        gate.main(["--project-root", str(ROOT), "--release-run-id", "../escape"])
