"""
RunHandle: Promise-like interface for tracking experiment execution.

Provides:
- RunStatus: Status summary for a batch of runs
- RunHandle: Protocol for tracking/awaiting experiment execution
- LocalRunHandle: Implementation for thread/process executors

Event-driven architecture:
- All handles support an optional on_event callback
- Events are emitted when run state changes (started, finished, failed, skipped)
- LocalRunHandle emits events synchronously from futures
- SlurmRunHandle emits events when polling detects state changes
- ProgressTrackers subscribe to these events for display
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from metalab.events import EventCallback
    from metalab.result import Results
    from metalab.store.base import Store
    from metalab.types import RunRecord, Status

logger = logging.getLogger(__name__)


@dataclass
class RunStatus:
    """
    Status summary for a batch of runs.

    Invariant: total == completed + failed + running + pending + skipped

    Attributes:
        total: Total number of runs in the batch.
        completed: Runs that executed and succeeded.
        running: Runs currently executing.
        pending: Runs waiting to start.
        failed: Runs that executed and failed.
        skipped: Runs skipped due to resume (already existed in store).
    """

    total: int
    completed: int
    running: int
    pending: int
    failed: int
    skipped: int = 0

    def __str__(self) -> str:
        return (
            f"RunStatus(total={self.total}, completed={self.completed}, "
            f"running={self.running}, pending={self.pending}, failed={self.failed}, "
            f"skipped={self.skipped})"
        )

    @property
    def done(self) -> int:
        """Total runs that are finished (completed + failed + skipped)."""
        return self.completed + self.failed + self.skipped

    @property
    def in_flight(self) -> int:
        """Total runs in progress (running + pending)."""
        return self.running + self.pending


class RunHandle(Protocol):
    """
    Promise-like handle for tracking experiment execution.

    All executors return a RunHandle from submit(). Users can:
    - Check status without blocking (.status, .is_complete)
    - Block until completion (.result())
    - Cancel pending/running jobs (.cancel())
    - Subscribe to events via on_event callback

    Event-driven progress tracking:
        Events are emitted when run state changes. For local executors,
        events are pushed synchronously. For SLURM, events are derived
        from store polling. ProgressTrackers subscribe to these events.
    """

    @property
    def job_id(self) -> str:
        """Unique identifier for this execution batch."""
        ...

    @property
    def status(self) -> RunStatus:
        """Current status of all runs (non-blocking)."""
        ...

    @property
    def is_complete(self) -> bool:
        """True if all runs have finished (success or failure)."""
        ...

    def result(self, timeout: float | None = None) -> Results:
        """
        Block until all runs complete and return Results.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            Results object containing all completed runs.

        Raises:
            TimeoutError: If timeout is reached before completion.
        """
        ...

    def cancel(self) -> None:
        """Cancel pending and running jobs."""
        ...

    def set_event_callback(self, callback: "EventCallback | None") -> None:
        """
        Set the event callback for progress tracking.

        Args:
            callback: Function to receive events, or None to disable.
        """
        ...


class LocalRunHandle:
    """
    RunHandle implementation for local executors (thread/process pools).

    Wraps a list of futures and provides the RunHandle interface.
    Emits events when run state changes for progress tracking.
    """

    def __init__(
        self,
        futures: list[tuple[str, Future[RunRecord]]],
        store: Store,
        run_ids: list[str],
        job_id: str | None = None,
        on_event: "EventCallback | None" = None,
        skipped_run_ids: list[str] | None = None,
    ) -> None:
        """
        Initialize the local run handle.

        Args:
            futures: List of (run_id, future) tuples.
            store: Store for persisting results.
            run_ids: All run IDs (including skipped).
            job_id: Optional job identifier. Generated if not provided.
            on_event: Optional callback for progress events.
            skipped_run_ids: Run IDs that were skipped due to resume.
        """
        self._futures = futures
        self._store = store
        self._run_ids = run_ids
        self._job_id = job_id or f"local-{uuid.uuid4().hex[:8]}"
        self._records: list[RunRecord] = []
        self._gathered = False
        self._on_event = on_event
        # Use list to preserve order for deterministic event emission
        self._skipped_run_ids: list[str] = list(skipped_run_ids or [])

        # Track which futures we've already emitted events for
        self._emitted_run_ids: set[str] = set()

        # Emit initial skip events
        self._emit_skip_events()

    def _emit_skip_events(self) -> None:
        """Emit skip events for runs that were skipped due to resume."""
        from metalab.events import Event, emit_event

        for run_id in self._skipped_run_ids:
            emit_event(
                self._on_event, Event.run_skipped(run_id, reason="already exists")
            )

    def _emit_completion_events(self) -> None:
        """Check futures and emit events for newly completed runs."""
        from metalab.events import Event, emit_event

        for run_id, future in self._futures:
            if run_id in self._emitted_run_ids:
                continue

            if future.done():
                try:
                    record = future.result(timeout=0)
                    if record.status.value == "success":
                        emit_event(
                            self._on_event,
                            Event.run_finished(
                                run_id,
                                duration_ms=record.duration_ms,
                                metrics=record.metrics,
                            ),
                        )
                    else:
                        error_msg = (
                            record.error.get("message", "") if record.error else ""
                        )
                        emit_event(
                            self._on_event,
                            Event.run_failed(run_id, error=error_msg),
                        )
                except Exception as e:
                    emit_event(
                        self._on_event,
                        Event.run_failed(run_id, error=str(e)),
                    )

                self._emitted_run_ids.add(run_id)

    def set_event_callback(self, callback: "EventCallback | None") -> None:
        """
        Set the event callback for progress tracking.

        Args:
            callback: Function to receive events, or None to disable.
        """
        self._on_event = callback

        # If setting a new callback, emit skip events that may have been missed
        if callback is not None:
            # Re-emit skip events (idempotent for progress trackers)
            self._emit_skip_events()

    @property
    def job_id(self) -> str:
        """Unique identifier for this execution batch."""
        return self._job_id

    @property
    def status(self) -> RunStatus:
        """Current status of all runs (non-blocking). Also emits pending events."""
        # Emit events for any newly completed futures
        self._emit_completion_events()

        completed = 0
        running = 0
        failed = 0

        for run_id, future in self._futures:
            if future.done():
                try:
                    record = future.result(timeout=0)
                    if record.status.value == "success":
                        completed += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
            else:
                running += 1

        # Skipped runs are separate from completed
        # Invariant: total = completed + failed + running + pending + skipped
        skipped = len(self._skipped_run_ids)
        pending = len(self._run_ids) - completed - running - failed - skipped

        return RunStatus(
            total=len(self._run_ids),
            completed=completed,
            running=running,
            pending=max(0, pending),
            failed=failed,
            skipped=skipped,
        )

    @property
    def is_complete(self) -> bool:
        """True if all runs have finished (success or failure)."""
        return all(f.done() for _, f in self._futures)

    def result(self, timeout: float | None = None) -> Results:
        """
        Block until all runs complete and return Results.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            Results object containing all completed runs.

        Raises:
            TimeoutError: If timeout is reached before completion.
        """
        from metalab.events import Event, emit_event
        from metalab.result import Results

        if self._gathered:
            return Results(store=self._store, records=self._records)

        records = []

        # Gather results from futures
        for run_id, future in self._futures:
            try:
                record = future.result(timeout=timeout)
                records.append(record)

                # Emit event BEFORE storage (so storage failures don't block progress)
                if run_id not in self._emitted_run_ids:
                    if record.status.value == "success":
                        emit_event(
                            self._on_event,
                            Event.run_finished(
                                run_id,
                                duration_ms=record.duration_ms,
                                metrics=record.metrics,
                            ),
                        )
                    else:
                        error_msg = (
                            record.error.get("message", "") if record.error else ""
                        )
                        emit_event(
                            self._on_event,
                            Event.run_failed(run_id, error=error_msg),
                        )
                    self._emitted_run_ids.add(run_id)

                # Persist record to store
                self._store.put_run_record(record)

            except FuturesTimeoutError:
                raise TimeoutError(f"Timeout waiting for run {run_id}")
            except Exception as e:
                # Log the exception for debugging, but don't crash the batch
                logger.warning(f"Error processing run {run_id}: {e}")

        # Load any skipped runs from store
        for run_id in self._run_ids:
            if not any(r.run_id == run_id for r in records):
                existing = self._store.get_run_record(run_id)
                if existing:
                    records.append(existing)

        self._records = records
        self._gathered = True

        return Results(store=self._store, records=records)

    def cancel(self) -> None:
        """Cancel pending and running jobs."""
        for _, future in self._futures:
            future.cancel()
