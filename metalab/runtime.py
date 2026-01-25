"""
Runtime: Non-reproducible plumbing passed to operations.

The Runtime provides:
- Logger for operation output
- Scratch directory for temporary files
- Cancellation token for cooperative shutdown
- Resource hints (serializable only)
- Event callback for notifications
"""

from __future__ import annotations

import logging
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metalab.events import EventCallback


class CancellationToken:
    """
    Simple cancellation token using threading.Event.

    Operations should periodically check is_cancelled() and exit gracefully
    if True.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Signal cancellation."""
        self._event.set()

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """
        Wait for cancellation.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if cancelled, False if timeout elapsed.
        """
        return self._event.wait(timeout)

    def reset(self) -> None:
        """Reset the cancellation state (use with caution)."""
        self._event.clear()


@dataclass
class Runtime:
    """
    Non-reproducible runtime context for operations.

    This contains the "plumbing" that operations need but that should NOT
    affect reproducibility (unlike context, params, and seeds).

    Attributes:
        logger: A logger for operation output.
        scratch_dir: A directory for temporary files (cleaned up after run).
        cancel_token: For cooperative cancellation.
        resource_hints: Serializable hints (e.g., {"gpu": True, "memory_gb": 16}).
        event_callback: Optional callback for runtime events.
    """

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("metalab.operation"))
    scratch_dir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="metalab_")))
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    resource_hints: dict[str, Any] = field(default_factory=dict)
    event_callback: EventCallback | None = None

    def check_cancelled(self) -> None:
        """
        Check if cancellation was requested and raise if so.

        Raises:
            CancelledError: If cancellation was requested.
        """
        if self.cancel_token.is_cancelled():
            raise CancelledError("Operation was cancelled")


class CancelledError(Exception):
    """Raised when an operation is cancelled."""

    pass


def create_runtime(
    run_id: str,
    logger: logging.Logger | None = None,
    scratch_dir: Path | None = None,
    resource_hints: dict[str, Any] | None = None,
    event_callback: EventCallback | None = None,
) -> Runtime:
    """
    Create a Runtime for an operation.

    Args:
        run_id: The run ID (used for logger name and scratch dir).
        logger: Custom logger (default: creates one named after run_id).
        scratch_dir: Custom scratch directory (default: temp dir).
        resource_hints: Serializable resource hints.
        event_callback: Optional event callback.

    Returns:
        A configured Runtime instance.
    """
    if logger is None:
        logger = logging.getLogger(f"metalab.run.{run_id[:8]}")

    if scratch_dir is None:
        scratch_dir = Path(tempfile.mkdtemp(prefix=f"metalab_{run_id[:8]}_"))

    return Runtime(
        logger=logger,
        scratch_dir=scratch_dir,
        cancel_token=CancellationToken(),
        resource_hints=resource_hints or {},
        event_callback=event_callback,
    )
