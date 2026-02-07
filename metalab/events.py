"""
Events system: Stable event schema for hooks and progress tracking.

Ordering guarantees:
- Synchronous emission: Events are emitted inline (callback blocks the run)
- Best-effort delivery: If callback raises, exception is logged but run continues
- Per-run ordering: For a single run, events are ordered (run_started < artifact_saved < run_finished)
- Cross-run ordering: NOT guaranteed (concurrent runs may interleave)
- Crash behavior: Events before crash are delivered; no buffering/persistence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventKind(str, Enum):
    """Types of events emitted by the runner."""

    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"
    RUN_SKIPPED = "run_skipped"  # When resume=True and run already exists
    ARTIFACT_SAVED = "artifact_saved"
    PROGRESS = "progress"
    LOG = "log"


@dataclass(frozen=True)
class Event:
    """
    An event emitted during experiment execution.

    Attributes:
        kind: The type of event.
        run_id: The run this event relates to (None for global events).
        timestamp: When the event occurred.
        payload: Event-specific data.
    """

    kind: EventKind
    run_id: str | None
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def run_started(cls, run_id: str, **extra: Any) -> Event:
        """Create a run_started event."""
        return cls(
            kind=EventKind.RUN_STARTED,
            run_id=run_id,
            timestamp=datetime.now(),
            payload=extra,
        )

    @classmethod
    def run_finished(cls, run_id: str, **extra: Any) -> Event:
        """Create a run_finished event."""
        return cls(
            kind=EventKind.RUN_FINISHED,
            run_id=run_id,
            timestamp=datetime.now(),
            payload=extra,
        )

    @classmethod
    def run_failed(cls, run_id: str, error: str, **extra: Any) -> Event:
        """Create a run_failed event."""
        return cls(
            kind=EventKind.RUN_FAILED,
            run_id=run_id,
            timestamp=datetime.now(),
            payload={"error": error, **extra},
        )

    @classmethod
    def run_skipped(cls, run_id: str, reason: str = "already exists") -> Event:
        """Create a run_skipped event."""
        return cls(
            kind=EventKind.RUN_SKIPPED,
            run_id=run_id,
            timestamp=datetime.now(),
            payload={"reason": reason},
        )

    @classmethod
    def artifact_saved(
        cls, run_id: str, name: str, uri: str, **extra: Any
    ) -> Event:
        """Create an artifact_saved event."""
        return cls(
            kind=EventKind.ARTIFACT_SAVED,
            run_id=run_id,
            timestamp=datetime.now(),
            payload={"name": name, "uri": uri, **extra},
        )

    @classmethod
    def progress(
        cls,
        run_id: str | None,
        current: int,
        total: int,
        message: str = "",
        running: int = 0,
    ) -> Event:
        """Create a progress event."""
        return cls(
            kind=EventKind.PROGRESS,
            run_id=run_id,
            timestamp=datetime.now(),
            payload={
                "current": current,
                "total": total,
                "message": message,
                "running": running,
            },
        )

    @classmethod
    def log(cls, run_id: str | None, message: str, level: str = "info") -> Event:
        """Create a log event."""
        return cls(
            kind=EventKind.LOG,
            run_id=run_id,
            timestamp=datetime.now(),
            payload={"message": message, "level": level},
        )


# Type alias for event callbacks
EventCallback = Callable[[Event], None]


def emit_event(callback: EventCallback | None, event: Event) -> None:
    """
    Emit an event to a callback, with best-effort delivery.

    If the callback raises an exception, it is logged but the run continues.
    This ensures that a broken event handler doesn't crash the experiment.

    Args:
        callback: The event callback (may be None).
        event: The event to emit.
    """
    if callback is None:
        return

    try:
        callback(event)
    except Exception as e:
        logger.warning(f"Event callback failed for {event.kind}: {e}")
        # Do NOT re-raise; run continues


class EventEmitter:
    """
    Helper class for emitting events.

    Wraps a callback and provides convenience methods for common events.
    """

    def __init__(self, callback: EventCallback | None = None) -> None:
        """
        Initialize the emitter.

        Args:
            callback: The event callback (may be None for no-op).
        """
        self._callback = callback

    def emit(self, event: Event) -> None:
        """Emit an event."""
        emit_event(self._callback, event)

    def run_started(self, run_id: str, **extra: Any) -> None:
        """Emit a run_started event."""
        self.emit(Event.run_started(run_id, **extra))

    def run_finished(self, run_id: str, **extra: Any) -> None:
        """Emit a run_finished event."""
        self.emit(Event.run_finished(run_id, **extra))

    def run_failed(self, run_id: str, error: str, **extra: Any) -> None:
        """Emit a run_failed event."""
        self.emit(Event.run_failed(run_id, error, **extra))

    def run_skipped(self, run_id: str, reason: str = "already exists") -> None:
        """Emit a run_skipped event."""
        self.emit(Event.run_skipped(run_id, reason))

    def artifact_saved(
        self, run_id: str, name: str, uri: str, **extra: Any
    ) -> None:
        """Emit an artifact_saved event."""
        self.emit(Event.artifact_saved(run_id, name, uri, **extra))

    def progress(
        self,
        current: int,
        total: int,
        message: str = "",
        run_id: str | None = None,
    ) -> None:
        """Emit a progress event."""
        self.emit(Event.progress(run_id, current, total, message))

    def log(
        self,
        message: str,
        level: str = "info",
        run_id: str | None = None,
    ) -> None:
        """Emit a log event."""
        self.emit(Event.log(run_id, message, level))
