"""Unit tests for context manager support in executors and stores."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestThreadExecutorContextManager:
    """Tests for ThreadExecutor context manager support."""

    def test_context_manager_calls_shutdown(self):
        """Context manager should call shutdown on exit."""
        from metalab.executor.thread import ThreadExecutor

        with patch.object(ThreadExecutor, "shutdown") as mock_shutdown:
            executor = ThreadExecutor(max_workers=2)
            mock_shutdown.reset_mock()

            with executor:
                pass

            mock_shutdown.assert_called_once_with(wait=True)

    def test_context_manager_calls_shutdown_on_exception(self):
        """Context manager should call shutdown even on exception."""
        from metalab.executor.thread import ThreadExecutor

        with patch.object(ThreadExecutor, "shutdown") as mock_shutdown:
            executor = ThreadExecutor(max_workers=2)
            mock_shutdown.reset_mock()

            with pytest.raises(ValueError):
                with executor:
                    raise ValueError("test error")

            mock_shutdown.assert_called_once_with(wait=True)


class TestProcessExecutorContextManager:
    """Tests for ProcessExecutor context manager support."""

    def test_context_manager_calls_shutdown(self):
        """Context manager should call shutdown on exit."""
        from metalab.executor.process import ProcessExecutor

        with patch.object(ProcessExecutor, "shutdown") as mock_shutdown:
            executor = ProcessExecutor(max_workers=2)
            mock_shutdown.reset_mock()

            with executor:
                pass

            mock_shutdown.assert_called_once_with(wait=True)


class MockStore:
    """Mock store that doesn't implement SupportsLogPath."""

    def __init__(self):
        self.put_run_record = MagicMock()
        self.put_log = MagicMock()
        self.put_artifact = MagicMock()


class TestCaptureContextManager:
    """Tests for Capture context manager support."""

    def test_context_manager_calls_finalize(self, tmp_path: Path):
        """Context manager should call finalize on exit."""
        from metalab.capture import Capture

        mock_store = MockStore()

        with Capture(
            store=mock_store,
            run_id="test-run",
            artifact_dir=tmp_path / "artifacts",
        ) as capture:
            capture.metric("test", 42)

        # After context exit, capture should be finalized
        assert capture._finalized is True

    def test_context_manager_finalizes_on_exception(self, tmp_path: Path):
        """Context manager should finalize even on exception."""
        from metalab.capture import Capture

        mock_store = MockStore()

        capture = Capture(
            store=mock_store,
            run_id="test-run",
            artifact_dir=tmp_path / "artifacts",
        )

        with pytest.raises(ValueError):
            with capture:
                capture.metric("test", 42)
                raise ValueError("test error")

        # Should still be finalized
        assert capture._finalized is True

    def test_double_finalize_is_idempotent(self, tmp_path: Path):
        """Calling finalize multiple times should be safe."""
        from metalab.capture import Capture

        mock_store = MockStore()

        capture = Capture(
            store=mock_store,
            run_id="test-run",
            artifact_dir=tmp_path / "artifacts",
        )

        # First finalize
        result1 = capture.finalize()
        assert capture._finalized is True

        # Second finalize should return same result
        result2 = capture.finalize()
        assert result1 == result2


class TestProgressRunHandleContextManager:
    """Tests for ProgressRunHandle context manager support."""

    def test_context_manager_stops_tracker(self):
        """Context manager should stop the tracker on exit."""
        from metalab.runner import ProgressRunHandle

        mock_handle = MagicMock()
        mock_handle.job_id = "test-job"
        mock_handle.status = MagicMock()
        mock_handle.is_complete = False
        mock_handle.set_event_callback = MagicMock()

        mock_tracker = MagicMock()
        mock_tracker.__enter__ = MagicMock(return_value=mock_tracker)
        mock_tracker.__exit__ = MagicMock(return_value=None)

        progress_handle = ProgressRunHandle(mock_handle, mock_tracker)

        # Use context manager
        with progress_handle:
            pass

        # Tracker should have __exit__ called
        mock_tracker.__exit__.assert_called()


class TestPostgresStoreContextManager:
    """Tests for PostgresStore context manager support."""

    def test_context_manager_interface_exists(self):
        """PostgresStore should have context manager methods."""
        from metalab.store.postgres import PostgresStore

        # Verify the class has the methods
        assert hasattr(PostgresStore, "__enter__")
        assert hasattr(PostgresStore, "__exit__")
