"""Bounded atomic-directory publication for transient Windows file handles."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


DEFAULT_MAX_ATTEMPTS = 40
DEFAULT_RETRY_DELAY_SECONDS = 0.25
_RETRYABLE_WINDOWS_ERRORS = frozenset({5, 32, 33})


def _is_retryable_windows_permission_error(error: BaseException) -> bool:
    return isinstance(error, PermissionError) and getattr(error, "winerror", None) in (
        _RETRYABLE_WINDOWS_ERRORS
    )


def _validate_retry_contract(max_attempts: int, retry_delay_seconds: float) -> None:
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer.")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be non-negative.")


def atomic_replace_directory(
    staging: str | Path,
    output: str | Path,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> None:
    """Publish an absent directory with bounded Windows sharing-violation retries.

    The retry covers only Windows access-denied/sharing/lock violations. Every
    other error fails immediately. The destination must remain absent on each
    attempt, so this helper never weakens immutable-output or overwrite rules.
    """

    _validate_retry_contract(max_attempts, retry_delay_seconds)
    staging_path = Path(staging)
    output_path = Path(output)
    if not staging_path.is_dir() or staging_path.is_symlink():
        raise FileNotFoundError(f"Atomic publication staging directory is invalid: {staging_path}")
    if os.path.lexists(output_path):
        raise FileExistsError(f"Atomic publication destination already exists: {output_path}")

    for attempt in range(1, max_attempts + 1):
        try:
            os.replace(staging_path, output_path)
            return
        except PermissionError as error:
            if (
                not _is_retryable_windows_permission_error(error)
                or attempt == max_attempts
                or not staging_path.is_dir()
                or os.path.lexists(output_path)
            ):
                raise
            time.sleep(retry_delay_seconds)


def cleanup_temporary_directory(
    temporary: Any,
    *,
    primary_error: BaseException | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> None:
    """Clean a TemporaryDirectory without masking a primary publication error."""

    _validate_retry_contract(max_attempts, retry_delay_seconds)
    for attempt in range(1, max_attempts + 1):
        try:
            temporary.cleanup()
            return
        except Exception as cleanup_error:
            retryable = _is_retryable_windows_permission_error(cleanup_error)
            if retryable and attempt < max_attempts:
                time.sleep(retry_delay_seconds)
                continue
            if primary_error is None:
                raise
            primary_error.add_note(
                "Temporary-directory cleanup also failed after the primary error: "
                f"{cleanup_error!r}"
            )
            return
