from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core import atomic_publish


def _windows_permission_error(winerror: int) -> PermissionError:
    error = PermissionError("locked")
    error.winerror = winerror
    return error


def test_atomic_replace_retries_windows_sharing_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    (staging / "artifact.txt").write_text("complete", encoding="utf-8")
    real_replace = atomic_publish.os.replace
    calls = 0
    sleeps: list[float] = []

    def flaky_replace(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _windows_permission_error(32)
        real_replace(source, target)

    monkeypatch.setattr(atomic_publish.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic_publish.time, "sleep", sleeps.append)

    atomic_publish.atomic_replace_directory(
        staging, output, max_attempts=3, retry_delay_seconds=0.125
    )

    assert calls == 3
    assert sleeps == [0.125, 0.125]
    assert not staging.exists()
    assert (output / "artifact.txt").read_text(encoding="utf-8") == "complete"


def test_atomic_replace_never_overwrites_existing_output(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        atomic_publish.atomic_replace_directory(staging, output)


def test_atomic_replace_does_not_retry_unrelated_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    calls = 0

    def denied(*_args: object) -> None:
        nonlocal calls
        calls += 1
        raise _windows_permission_error(1)

    monkeypatch.setattr(atomic_publish.os, "replace", denied)

    with pytest.raises(PermissionError, match="locked"):
        atomic_publish.atomic_replace_directory(staging, tmp_path / "output")

    assert calls == 1


def test_cleanup_preserves_primary_exception_and_attaches_failure_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_error = _windows_permission_error(32)
    temporary = SimpleNamespace(cleanup=lambda: (_ for _ in ()).throw(cleanup_error))
    primary = RuntimeError("primary publication failure")
    monkeypatch.setattr(atomic_publish.time, "sleep", lambda _delay: None)

    atomic_publish.cleanup_temporary_directory(
        temporary,
        primary_error=primary,
        max_attempts=2,
        retry_delay_seconds=0.0,
    )

    assert primary.__notes__
    assert "cleanup also failed" in primary.__notes__[0]


def test_nonretryable_cleanup_error_also_cannot_mask_primary_exception() -> None:
    cleanup_error = OSError("unexpected cleanup failure")
    temporary = SimpleNamespace(cleanup=lambda: (_ for _ in ()).throw(cleanup_error))
    primary = RuntimeError("primary publication failure")

    atomic_publish.cleanup_temporary_directory(
        temporary,
        primary_error=primary,
        max_attempts=1,
        retry_delay_seconds=0.0,
    )

    assert primary.__notes__
    assert "unexpected cleanup failure" in primary.__notes__[0]
