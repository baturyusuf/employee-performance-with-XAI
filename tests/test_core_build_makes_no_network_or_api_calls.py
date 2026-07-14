from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from src.experiments import build_manuscript_evidence as builder
from src.governance.offline_runtime import (
    OfflineRuntimeError,
    active_policy_receipt,
    enforce_offline_runtime,
    policy_receipt,
    validate_policy_receipt,
)


def test_public_build_wraps_the_complete_implementation_in_offline_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_build(*args, **kwargs):
        observed["policy"] = active_policy_receipt()
        observed["offline_marker"] = os.environ.get("HR_XAI_OFFLINE_RUNTIME")
        observed["openai_key_present"] = "OPENAI_API_KEY" in os.environ
        return {"run_dir": Path("reports/example")}

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-visible-to-core")
    monkeypatch.setattr(builder, "_build_impl", fake_build)

    result = builder.build(evidence_scope="core")

    assert result == {"run_dir": Path("reports/example")}
    assert observed["policy"] == policy_receipt()
    assert observed["offline_marker"] == "1"
    assert observed["openai_key_present"] is False
    assert os.environ["OPENAI_API_KEY"] == "must-not-be-visible-to-core"


def test_public_build_fails_on_dns_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    def attempt_network(*args, **kwargs):
        socket.getaddrinfo("example.com", 443)
        raise AssertionError("offline boundary did not run")

    monkeypatch.setattr(builder, "_build_impl", attempt_network)

    with pytest.raises(OfflineRuntimeError, match="socket.getaddrinfo"):
        builder.build(evidence_scope="core")


@pytest.mark.parametrize(
    "operation",
    [
        lambda: socket.create_connection(("127.0.0.1", 9)),
        lambda: socket.gethostbyname("example.com"),
        lambda: socket.socket().connect(("127.0.0.1", 9)),
        lambda: socket.socket().bind(("127.0.0.1", 0)),
        lambda: socket.socket().sendto(b"x", ("127.0.0.1", 9)),
    ],
)
def test_socket_dns_tcp_udp_and_listener_operations_are_denied(operation) -> None:
    with pytest.raises(OfflineRuntimeError, match="prohibited"):
        with enforce_offline_runtime():
            operation()


def test_caught_attempt_poisoning_prevents_success() -> None:
    with pytest.raises(OfflineRuntimeError, match="attempted prohibited"):
        with enforce_offline_runtime():
            try:
                socket.getaddrinfo("example.com", 443)
            except OfflineRuntimeError:
                pass


def test_shell_and_non_git_subprocesses_are_denied() -> None:
    with pytest.raises(OfflineRuntimeError, match="subprocess.shell"):
        with enforce_offline_runtime():
            subprocess.run("git --version", shell=True, check=False)

    with pytest.raises(OfflineRuntimeError, match="subprocess.executable"):
        with enforce_offline_runtime():
            subprocess.run([sys.executable, "-c", "print('not run')"], check=False)


def test_local_git_subprocess_is_the_only_admitted_child_command() -> None:
    with enforce_offline_runtime():
        completed = subprocess.run(
            ["git", "--version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    assert completed.stdout.startswith("git version ")


def test_remote_git_subcommand_is_denied_before_execution() -> None:
    with pytest.raises(OfflineRuntimeError, match="git_subcommand:fetch"):
        with enforce_offline_runtime():
            subprocess.run(["git", "fetch", "origin"], check=False)


def test_policy_receipt_is_exact_and_rejects_nonzero_attempts() -> None:
    expected = policy_receipt()

    assert validate_policy_receipt(expected) == expected
    changed = dict(expected)
    changed["attempted_network_operations"] = 1
    with pytest.raises(OfflineRuntimeError, match="differs"):
        validate_policy_receipt(changed)


def test_package_status_requires_active_boundary_and_records_zero_network(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "core"
    run_dir.mkdir()
    context = builder.StageContext(
        config_path=tmp_path / "config.json",
        config={},
        settings={},
        run_dir=run_dir,
        run_id="offline-package",
        config_hash="a" * 64,
        manifest={"scope_contract_hash": "b" * 64},
        evidence_scope="core",
        scope_contract={"stages": list(builder.CORE_STAGE_ORDER)},
    )
    for stage in builder.CORE_STAGE_ORDER:
        artifact = run_dir / stage / "result.txt"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(f"{stage}\n", encoding="utf-8")
        builder._write_stage_metadata(
            context,
            stage,
            [artifact],
            started_at="2026-07-14T00:00:00+00:00",
            elapsed_seconds=1.0,
        )

    with pytest.raises(OfflineRuntimeError, match="No offline"):
        builder._write_package_status(context)
    with enforce_offline_runtime():
        status_path = builder._write_package_status(context)

    status = builder._validate_package_status_contract(context)
    assert status_path.is_file()
    assert status["network_calls"] == 0
    assert status["paid_api_calls"] == 0
    assert status["runtime_network_policy"] == policy_receipt()
