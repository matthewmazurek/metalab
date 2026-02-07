"""
Unit tests for SLURM progress tracking.

Tests the poll-to-event bridge in SlurmRunHandle.result() and the
running counter in progress trackers.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from metalab.events import Event, EventKind
from metalab.executor.slurm import SlurmRunHandle
from metalab.progress.base import SimpleProgressTracker
from metalab.store.file import FileStoreConfig


def _create_done_markers(store_path: Path, count: int) -> None:
    """Create .done marker files in the runs directory."""
    runs_dir = store_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        marker = runs_dir / f"run-{i:04d}.done"
        marker.write_text(json.dumps({"completed_at": "2025-01-01T00:00:00"}))


def _make_advancing_time(step: float = 20.0):
    """
    Create a time.time replacement that jumps forward by `step` seconds
    on each call, ensuring the settle loop completes quickly.
    """
    t = [0.0]

    def fake_time():
        t[0] += step
        return t[0]

    return fake_time


class TestSlurmRunHandleEventEmission:
    """Test that SlurmRunHandle.result() emits progress events."""

    def _make_handle(
        self,
        store: "FileStore",
        total_runs: int,
        skipped_count: int = 0,
    ) -> SlurmRunHandle:
        """Create a SlurmRunHandle with no real SLURM jobs."""
        return SlurmRunHandle(
            store=store,
            job_ids=[],  # No real SLURM jobs
            shards=[],
            total_runs=total_runs,
            chunk_size=1,
            skipped_count=skipped_count,
        )

    def test_emits_finished_events_for_done_markers(self, tmp_path: Path):
        """result() should emit RUN_FINISHED events matching .done marker count."""
        store = FileStoreConfig(root=str(tmp_path)).connect()
        _create_done_markers(tmp_path, 5)

        handle = self._make_handle(store, total_runs=5)
        handle._done_count_cache_ttl_seconds = 0  # Disable caching

        events: list[Event] = []
        handle.set_event_callback(lambda e: events.append(e))

        with (
            patch("metalab.executor.slurm.time.sleep"),
            patch("metalab.executor.slurm.time.time", _make_advancing_time()),
        ):
            handle.result()

        finished = [e for e in events if e.kind == EventKind.RUN_FINISHED]
        assert len(finished) == 5

    def test_emits_skipped_events(self, tmp_path: Path):
        """result() should emit RUN_SKIPPED events for skipped_count."""
        store = FileStoreConfig(root=str(tmp_path)).connect()
        _create_done_markers(tmp_path, 3)

        handle = self._make_handle(store, total_runs=5, skipped_count=2)
        handle._done_count_cache_ttl_seconds = 0

        events: list[Event] = []
        handle.set_event_callback(lambda e: events.append(e))

        with (
            patch("metalab.executor.slurm.time.sleep"),
            patch("metalab.executor.slurm.time.time", _make_advancing_time()),
        ):
            handle.result()

        skipped = [e for e in events if e.kind == EventKind.RUN_SKIPPED]
        finished = [e for e in events if e.kind == EventKind.RUN_FINISHED]
        assert len(skipped) == 2
        assert len(finished) == 3

    def test_emits_progress_events_with_running_count(self, tmp_path: Path):
        """result() should emit PROGRESS events containing running count."""
        store = FileStoreConfig(root=str(tmp_path)).connect()
        _create_done_markers(tmp_path, 5)

        handle = self._make_handle(store, total_runs=5)
        handle._done_count_cache_ttl_seconds = 0

        events: list[Event] = []
        handle.set_event_callback(lambda e: events.append(e))

        with (
            patch("metalab.executor.slurm.time.sleep"),
            patch("metalab.executor.slurm.time.time", _make_advancing_time()),
        ):
            handle.result()

        progress_events = [e for e in events if e.kind == EventKind.PROGRESS]
        assert len(progress_events) >= 1
        # All PROGRESS events should have a 'running' key in payload
        for pe in progress_events:
            assert "running" in pe.payload

    def test_no_duplicate_events_across_polls(self, tmp_path: Path):
        """Delta tracking should prevent duplicate finished events."""
        store = FileStoreConfig(root=str(tmp_path)).connect()
        _create_done_markers(tmp_path, 10)

        handle = self._make_handle(store, total_runs=10)
        handle._done_count_cache_ttl_seconds = 0

        events: list[Event] = []
        handle.set_event_callback(lambda e: events.append(e))

        with (
            patch("metalab.executor.slurm.time.sleep"),
            patch("metalab.executor.slurm.time.time", _make_advancing_time()),
        ):
            handle.result()

        finished = [e for e in events if e.kind == EventKind.RUN_FINISHED]
        # Should be exactly 10, not more (no duplicates from multiple poll loops)
        assert len(finished) == 10

    def test_no_events_when_no_callback(self, tmp_path: Path):
        """result() should not crash when no event callback is set."""
        store = FileStoreConfig(root=str(tmp_path)).connect()
        _create_done_markers(tmp_path, 3)

        handle = self._make_handle(store, total_runs=3)
        handle._done_count_cache_ttl_seconds = 0
        # Don't set any callback

        with (
            patch("metalab.executor.slurm.time.sleep"),
            patch("metalab.executor.slurm.time.time", _make_advancing_time()),
        ):
            result = handle.result()

        assert result is not None


class TestProgressEventRunningField:
    """Test that Event.progress() includes the running field."""

    def test_progress_event_default_running(self):
        """PROGRESS event defaults running to 0."""
        event = Event.progress(None, current=5, total=10)
        assert event.payload["running"] == 0

    def test_progress_event_custom_running(self):
        """PROGRESS event carries explicit running count."""
        event = Event.progress(None, current=5, total=10, running=3)
        assert event.payload["running"] == 3


class TestSimpleTrackerRunningCounter:
    """Test SimpleProgressTracker running counter from PROGRESS events."""

    def test_running_starts_at_zero(self):
        """Tracker should initialize running to 0."""
        tracker = SimpleProgressTracker(total=10)
        assert tracker.running == 0

    def test_running_updated_from_progress_event(self):
        """PROGRESS event should update running counter."""
        tracker = SimpleProgressTracker(total=10)

        event = Event.progress(None, current=3, total=10, running=5)
        tracker(event)

        assert tracker.running == 5

    def test_running_updates_on_each_progress_event(self):
        """Running counter should reflect latest PROGRESS event."""
        tracker = SimpleProgressTracker(total=10)

        tracker(Event.progress(None, current=1, total=10, running=3))
        assert tracker.running == 3

        tracker(Event.progress(None, current=5, total=10, running=1))
        assert tracker.running == 1

        tracker(Event.progress(None, current=10, total=10, running=0))
        assert tracker.running == 0

    def test_completed_and_running_work_together(self):
        """Finished events and progress events both update correctly."""
        tracker = SimpleProgressTracker(total=10)

        # Some runs finish
        tracker(Event.run_finished("run-1"))
        tracker(Event.run_finished("run-2"))
        assert tracker.completed == 2

        # PROGRESS event reports running count
        tracker(Event.progress(None, current=2, total=10, running=4))
        assert tracker.running == 4
        assert tracker.completed == 2

    def test_skipped_events_update_counter(self):
        """RUN_SKIPPED events should update skipped counter."""
        tracker = SimpleProgressTracker(total=10)

        tracker(Event.run_skipped("skip-1"))
        tracker(Event.run_skipped("skip-2"))
        tracker(Event.run_skipped("skip-3"))

        assert tracker.skipped == 3
