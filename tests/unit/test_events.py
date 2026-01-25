"""Tests for event system."""

from __future__ import annotations

from datetime import datetime

from metalab.events import Event, EventEmitter, EventKind, emit_event


class TestEvent:
    """Tests for Event dataclass."""

    def test_run_started_factory(self):
        """run_started() factory should create correct event."""
        event = Event.run_started("run123", extra_key="value")
        assert event.kind == EventKind.RUN_STARTED
        assert event.run_id == "run123"
        assert event.payload["extra_key"] == "value"
        assert isinstance(event.timestamp, datetime)

    def test_run_finished_factory(self):
        """run_finished() factory should create correct event."""
        event = Event.run_finished("run123", duration=1000)
        assert event.kind == EventKind.RUN_FINISHED
        assert event.run_id == "run123"
        assert event.payload["duration"] == 1000

    def test_run_failed_factory(self):
        """run_failed() factory should create correct event."""
        event = Event.run_failed("run123", error="Test error")
        assert event.kind == EventKind.RUN_FAILED
        assert event.payload["error"] == "Test error"

    def test_run_skipped_factory(self):
        """run_skipped() factory should create correct event."""
        event = Event.run_skipped("run123", reason="already exists")
        assert event.kind == EventKind.RUN_SKIPPED
        assert event.payload["reason"] == "already exists"

    def test_artifact_saved_factory(self):
        """artifact_saved() factory should create correct event."""
        event = Event.artifact_saved("run123", name="output", uri="/path")
        assert event.kind == EventKind.ARTIFACT_SAVED
        assert event.payload["name"] == "output"
        assert event.payload["uri"] == "/path"

    def test_progress_factory(self):
        """progress() factory should create correct event."""
        event = Event.progress("run123", current=5, total=10, message="Working...")
        assert event.kind == EventKind.PROGRESS
        assert event.payload["current"] == 5
        assert event.payload["total"] == 10

    def test_log_factory(self):
        """log() factory should create correct event."""
        event = Event.log("run123", message="Test message", level="warning")
        assert event.kind == EventKind.LOG
        assert event.payload["message"] == "Test message"
        assert event.payload["level"] == "warning"

    def test_event_is_frozen(self):
        """Events should be immutable."""
        event = Event.run_started("run123")
        # frozen=True means we can't modify attributes
        try:
            event.run_id = "other"  # type: ignore
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestEmitEvent:
    """Tests for emit_event()."""

    def test_emit_to_callback(self):
        """Events should be passed to callback."""
        received = []

        def callback(event):
            received.append(event)

        event = Event.run_started("run123")
        emit_event(callback, event)

        assert len(received) == 1
        assert received[0] is event

    def test_emit_to_none(self):
        """None callback should be a no-op."""
        event = Event.run_started("run123")
        emit_event(None, event)  # Should not raise

    def test_emit_exception_is_caught(self):
        """Callback exceptions should be caught."""

        def bad_callback(event):
            raise ValueError("Test error")

        event = Event.run_started("run123")
        # Should not raise, just log
        emit_event(bad_callback, event)


class TestEventEmitter:
    """Tests for EventEmitter helper."""

    def test_emitter_wraps_callback(self):
        """Emitter should pass events to callback."""
        received = []
        emitter = EventEmitter(lambda e: received.append(e))

        emitter.run_started("run123")
        emitter.run_finished("run123")

        assert len(received) == 2
        assert received[0].kind == EventKind.RUN_STARTED
        assert received[1].kind == EventKind.RUN_FINISHED

    def test_emitter_with_none_callback(self):
        """Emitter with None callback should be a no-op."""
        emitter = EventEmitter(None)
        emitter.run_started("run123")  # Should not raise

    def test_emitter_progress(self):
        """Emitter progress method should work."""
        received = []
        emitter = EventEmitter(lambda e: received.append(e))

        emitter.progress(current=5, total=10, message="Test")

        assert len(received) == 1
        assert received[0].kind == EventKind.PROGRESS
        assert received[0].payload["current"] == 5
