"""Process-wide no-network and no-paid-API boundary for scientific builds."""

from __future__ import annotations

import contextvars
import os
import socket
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


POLICY_VERSION = 1
POLICY_MODE = "deny_all_network_and_paid_api_during_scientific_execution"
ALLOWED_SUBPROCESS_EXECUTABLES = ("git", "git.exe")
ALLOWED_GIT_SUBCOMMANDS = (
    "--version",
    "cat-file",
    "diff",
    "ls-files",
    "ls-tree",
    "rev-parse",
    "show",
    "status",
)
SENSITIVE_API_ENVIRONMENT_KEYS = (
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "COHERE_API_KEY",
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
    "OPENAI_API_KEY",
)
BLOCKED_SOCKET_OPERATIONS = (
    "socket.accept",
    "socket.bind",
    "socket.connect",
    "socket.connect_ex",
    "socket.listen",
    "socket.send",
    "socket.sendall",
    "socket.sendmsg",
    "socket.sendto",
    "socket.create_connection",
    "socket.create_server",
    "socket.getaddrinfo",
    "socket.gethostbyaddr",
    "socket.gethostbyname",
    "socket.gethostbyname_ex",
)


class OfflineRuntimeError(RuntimeError):
    """Raised on any prohibited network, API, shell, or subprocess attempt."""


@dataclass
class OfflineRuntimeState:
    """Mutable attempt ledger owned by one active process-wide boundary."""

    attempted_operations: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def block(self, operation: str) -> None:
        with self._lock:
            self.attempted_operations.append(operation)
        raise OfflineRuntimeError(
            f"Network/API operation is prohibited during scientific execution: {operation}."
        )

    def assert_clean(self) -> None:
        with self._lock:
            attempted = tuple(self.attempted_operations)
        if attempted:
            raise OfflineRuntimeError(
                "Scientific execution attempted prohibited network/API operations: "
                f"{list(attempted)}."
            )

    def receipt(self) -> dict[str, Any]:
        self.assert_clean()
        return policy_receipt()


_ACTIVE_STATE: contextvars.ContextVar[OfflineRuntimeState | None] = contextvars.ContextVar(
    "employee_performance_xai_offline_runtime_state", default=None
)


def policy_receipt() -> dict[str, Any]:
    """Return the only valid zero-attempt publication policy payload."""

    return {
        "policy_version": POLICY_VERSION,
        "mode": POLICY_MODE,
        "network_allowed": False,
        "paid_api_allowed": False,
        "attempted_network_operations": 0,
        "successful_network_operations": 0,
        "blocked_socket_operations": list(BLOCKED_SOCKET_OPERATIONS),
        "subprocess_policy": "shell_denied_and_only_git_executable_allowed",
        "allowed_subprocess_executables": list(ALLOWED_SUBPROCESS_EXECUTABLES),
        "allowed_git_subcommands": list(ALLOWED_GIT_SUBCOMMANDS),
        "api_credential_environment_cleared": list(SENSITIVE_API_ENVIRONMENT_KEYS),
    }


def validate_policy_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any drift from the exact zero-attempt runtime policy."""

    expected = policy_receipt()
    if dict(payload) != expected:
        raise OfflineRuntimeError("Runtime no-network/API policy receipt differs from the exact contract.")
    return expected


def active_policy_receipt() -> dict[str, Any]:
    """Return a receipt only while the process-wide boundary is active and clean."""

    state = _ACTIVE_STATE.get()
    if state is None:
        raise OfflineRuntimeError("No offline scientific runtime boundary is active.")
    return state.receipt()


def _command_executable(command: Any, explicit_executable: Any) -> str:
    raw = explicit_executable
    if raw is None:
        if isinstance(command, (list, tuple)) and command:
            raw = command[0]
        elif isinstance(command, (str, os.PathLike)):
            raw = command
    if not isinstance(raw, (str, os.PathLike)):
        return "<unknown>"
    return Path(os.fspath(raw)).name.casefold()


@contextmanager
def enforce_offline_runtime() -> Iterable[OfflineRuntimeState]:
    """Deny network/API activity process-wide for one scientific execution.

    Nested use shares the outer ledger. Socket and subprocess patches are global
    so worker threads cannot evade the boundary; the context variable is used
    only to expose the active receipt to publication code.
    """

    existing = _ACTIVE_STATE.get()
    if existing is not None:
        yield existing
        existing.assert_clean()
        return

    state = OfflineRuntimeState()
    token = _ACTIVE_STATE.set(state)
    originals: list[tuple[Any, str, Any]] = []

    def patch(owner: Any, attribute: str, replacement: Any) -> None:
        if hasattr(owner, attribute):
            originals.append((owner, attribute, getattr(owner, attribute)))
            setattr(owner, attribute, replacement)

    def blocked(operation: str):
        def deny(*_args: Any, **_kwargs: Any) -> Any:
            state.block(operation)

        return deny

    for attribute in ("accept", "bind", "connect", "connect_ex", "listen", "send", "sendall", "sendmsg", "sendto"):
        patch(socket.socket, attribute, blocked(f"socket.{attribute}"))
    for attribute in (
        "create_connection",
        "create_server",
        "getaddrinfo",
        "gethostbyaddr",
        "gethostbyname",
        "gethostbyname_ex",
    ):
        patch(socket, attribute, blocked(f"socket.{attribute}"))

    original_popen = subprocess.Popen

    def guarded_popen(*popenargs: Any, **kwargs: Any):
        command = kwargs.get("args", popenargs[0] if popenargs else None)
        if kwargs.get("shell"):
            state.block("subprocess.shell")
        executable = _command_executable(command, kwargs.get("executable"))
        if executable not in ALLOWED_SUBPROCESS_EXECUTABLES:
            state.block(f"subprocess.executable:{executable}")
        command_parts = list(command) if isinstance(command, (list, tuple)) else []
        git_subcommand = str(command_parts[1]).casefold() if len(command_parts) > 1 else "<missing>"
        if git_subcommand not in ALLOWED_GIT_SUBCOMMANDS:
            state.block(f"subprocess.git_subcommand:{git_subcommand}")
        return original_popen(*popenargs, **kwargs)

    patch(subprocess, "Popen", guarded_popen)

    missing = object()
    saved_environment: dict[str, object | str] = {}
    for key in SENSITIVE_API_ENVIRONMENT_KEYS:
        saved_environment[key] = os.environ.get(key, missing)
        os.environ.pop(key, None)
    marker_key = "HR_XAI_OFFLINE_RUNTIME"
    saved_environment[marker_key] = os.environ.get(marker_key, missing)
    os.environ[marker_key] = "1"

    try:
        yield state
        state.assert_clean()
    finally:
        for owner, attribute, original in reversed(originals):
            setattr(owner, attribute, original)
        for key, value in saved_environment.items():
            if value is missing:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        _ACTIVE_STATE.reset(token)
