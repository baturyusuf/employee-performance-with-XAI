from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.governance import dependency_contract as dependencies


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_FILES = (
    "requirements.txt",
    "requirements-core.txt",
    "requirements-supplementary.txt",
    "requirements-legacy-optional.txt",
    "requirements-dev.txt",
    "environment.yml",
)


def _copy_contract(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "constraints").mkdir()
    shutil.copy2(ROOT / "configs" / "dependency_contract.yaml", tmp_path / "configs")
    shutil.copy2(ROOT / "constraints" / "py314-lock.txt", tmp_path / "constraints")
    for relative in DEPENDENCY_FILES:
        shutil.copy2(ROOT / relative, tmp_path / relative)


def _clean_core_environment(contract: dict[str, object]) -> dict[str, str]:
    groups = contract["groups"]
    lock = contract["lock"]
    direct = groups["core"]["packages"]
    installed = {name: lock[name]["version"] for name in direct}
    installed["pip"] = "25.2"
    return installed


def test_actual_dependency_contract_is_closed_and_core_excludes_optional_packages() -> None:
    contract = dependencies.validate_contract(ROOT)

    assert contract["lock_package_count"] == 96
    assert contract["declared_direct_package_count"] == 22
    core = set(contract["groups"]["core"]["packages"])
    supplementary = set(contract["groups"]["supplementary"]["packages"])
    assert supplementary == set()
    assert core.isdisjoint(contract["forbidden_core_packages"])
    assert "openai" not in core
    assert "openai-agents" not in core
    assert contract["groups"]["supplementary"]["includes"] == ["requirements-core.txt"]


def test_contract_rejects_forbidden_package_added_to_core(tmp_path: Path) -> None:
    _copy_contract(tmp_path)
    core = tmp_path / "requirements-core.txt"
    core.write_text(core.read_text(encoding="utf-8") + "\nopenai>=2.43,<2.44\n", encoding="utf-8")

    with pytest.raises(dependencies.DependencyContractError, match="direct package set"):
        dependencies.validate_contract(tmp_path)


@pytest.mark.parametrize(
    "replacement, message",
    [
        ("numpy>=2.4,<2.5", "not an exact pin"),
        ("--index-url https://example.invalid/simple", "Unsafe lock directive"),
    ],
)
def test_contract_rejects_nonexact_or_remote_lock_entries(
    tmp_path: Path, replacement: str, message: str
) -> None:
    _copy_contract(tmp_path)
    lock = tmp_path / "constraints" / "py314-lock.txt"
    text = lock.read_text(encoding="utf-8").replace("numpy==2.4.4", replacement)
    lock.write_text(text, encoding="utf-8")

    with pytest.raises(dependencies.DependencyContractError, match=message):
        dependencies.validate_contract(tmp_path)


def test_environment_validation_requires_exact_locked_direct_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = dependencies.validate_contract(ROOT)
    installed = _clean_core_environment(contract)
    monkeypatch.setattr(dependencies, "_installed_distributions", lambda: installed)
    monkeypatch.setattr(dependencies, "_pip_check", lambda: (0, "No broken requirements found."))

    result = dependencies.validate_environment(contract, profile="core")

    assert result["profile"] == "core"
    assert result["validated_package_count"] == len(installed) - 1
    assert set(result["direct_versions"]) == set(contract["groups"]["core"]["packages"])
    assert result["missing_package_count"] == 0
    assert result["unlocked_package_count"] == 0
    assert result["version_mismatch_count"] == 0


def test_environment_validation_rejects_version_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = dependencies.validate_contract(ROOT)
    installed = _clean_core_environment(contract)
    installed["numpy"] = "0.0.0"
    monkeypatch.setattr(dependencies, "_installed_distributions", lambda: installed)

    with pytest.raises(dependencies.DependencyContractError, match=r"mismatched=\['numpy'\]"):
        dependencies.validate_environment(contract, profile="core")


def test_development_environment_skips_windows_only_direct_package_on_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = dependencies.validate_contract(ROOT)
    groups = contract["groups"]
    lock = contract["lock"]
    direct = set(groups["core"]["packages"])
    direct.update(groups["legacy_optional"]["packages"])
    direct.update(groups["development"]["packages"])
    direct.remove("pywin32")
    installed = {name: lock[name]["version"] for name in direct}
    installed["pip"] = "25.2"
    monkeypatch.setattr(dependencies.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dependencies, "_installed_distributions", lambda: installed)
    monkeypatch.setattr(dependencies, "_pip_check", lambda: (0, "No broken requirements found."))

    result = dependencies.validate_environment(contract, profile="development")

    assert "pywin32" not in result["direct_versions"]
    assert set(result["direct_versions"]) == direct


def test_clean_git_receipt_is_atomic_compact_and_independently_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_contract(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "contract_source.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Dependency Contract Test",
            "-c",
            "user.email=dependency-contract@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    contract = dependencies.validate_contract(tmp_path)
    installed = _clean_core_environment(contract)
    monkeypatch.setattr(dependencies, "_installed_distributions", lambda: installed)
    monkeypatch.setattr(dependencies, "_pip_check", lambda: (0, "No broken requirements found."))

    receipt = dependencies.build_receipt(tmp_path)
    receipt_path = tmp_path / contract["receipt_path"]
    validated = dependencies.validate_receipt(contract["receipt_path"], tmp_path)

    assert validated == receipt
    assert receipt["status"] == "passed"
    assert receipt["canonical_eligible"] is True
    assert receipt["core_forbidden_package_count"] == 0
    assert receipt["paid_api_calls"] == 0
    assert receipt_path.is_file()
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert not list(receipt_path.parent.glob(f".{receipt_path.name}.*.tmp"))


def test_cli_defaults_use_portable_paths() -> None:
    parser = dependencies._build_parser()
    args = parser.parse_args(["--validate-only"])

    assert "\\" not in args.project_root
    assert args.config == "configs/dependency_contract.yaml"
