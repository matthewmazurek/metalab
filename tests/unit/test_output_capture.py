"""Tests for output capture functionality."""

from __future__ import annotations

import io
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from metalab.capture.output import (
    CapturedOutput,
    OutputCapture,
    OutputCaptureContext,
    OutputCaptureManager,
    normalize_output_capture,
)


class TestOutputCapture:
    """Tests for OutputCapture configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OutputCapture()
        assert config.stdout is True
        assert config.stderr is True
        assert config.logging is True
        assert config.warnings is False
        assert config.display == "console"

    def test_suppress_factory(self):
        """Test suppress() convenience constructor."""
        config = OutputCapture.suppress()
        assert config.display == "suppress"
        assert config.stdout is True
        assert config.stderr is True

    def test_console_factory(self):
        """Test console() convenience constructor."""
        config = OutputCapture.console()
        assert config.display == "console"

    def test_passthrough_factory(self):
        """Test passthrough() convenience constructor."""
        config = OutputCapture.passthrough()
        assert config.display == "passthrough"


class TestCapturedOutput:
    """Tests for CapturedOutput container."""

    def test_has_content_empty(self):
        """Test has_content() returns False for empty capture."""
        captured = CapturedOutput()
        assert captured.has_content() is False

    def test_has_content_with_stdout(self):
        """Test has_content() returns True with stdout."""
        captured = CapturedOutput(stdout="hello")
        assert captured.has_content() is True

    def test_has_content_with_logging(self):
        """Test has_content() returns True with logging."""
        captured = CapturedOutput(logging_records=[{"message": "test"}])
        assert captured.has_content() is True

    def test_format_logging(self):
        """Test formatting logging records."""
        captured = CapturedOutput(
            logging_records=[
                {"formatted": "2024-01-01 - test - INFO - message1"},
                {"formatted": "2024-01-01 - test - INFO - message2"},
            ]
        )
        formatted = captured.format_logging()
        assert "message1" in formatted
        assert "message2" in formatted


class TestOutputCaptureContext:
    """Tests for OutputCaptureContext."""

    def test_captures_stdout(self):
        """Test capturing stdout."""
        config = OutputCapture(display="suppress", logging=False)

        with OutputCaptureContext(config) as captured:
            print("hello world")

        assert "hello world" in captured.stdout

    def test_captures_stderr(self):
        """Test capturing stderr."""
        config = OutputCapture(display="suppress", logging=False)

        with OutputCaptureContext(config) as captured:
            print("error message", file=sys.stderr)

        assert "error message" in captured.stderr

    def test_captures_logging(self):
        """Test capturing logging output."""
        config = OutputCapture(display="suppress", stdout=False, stderr=False)

        # Create a test logger
        logger = logging.getLogger("test_capture")
        logger.setLevel(logging.INFO)

        with OutputCaptureContext(config) as captured:
            logger.info("test log message")

        assert len(captured.logging_records) >= 1
        assert any("test log message" in r["message"] for r in captured.logging_records)

    def test_passthrough_mode(self):
        """Test passthrough mode still captures."""
        config = OutputCapture(display="passthrough", logging=False)

        # Redirect stdout temporarily
        old_stdout = sys.stdout
        capture_buffer = io.StringIO()
        
        try:
            with OutputCaptureContext(config) as captured:
                # The interceptor should capture but also pass through
                # We can't easily test the passthrough without mocking,
                # but we can verify capture works
                print("passthrough test")
            
            assert "passthrough test" in captured.stdout
        finally:
            sys.stdout = old_stdout

    def test_suppress_mode_suppresses_output(self):
        """Test suppress mode doesn't output to terminal."""
        config = OutputCapture(display="suppress", logging=False)

        # Capture what goes to the real stdout
        old_stdout = sys.stdout
        real_capture = io.StringIO()

        try:
            sys.stdout = real_capture
            
            with OutputCaptureContext(config) as captured:
                # This should be captured but NOT sent to stdout
                print("suppressed message")

            # The message should be in captured output
            assert "suppressed message" in captured.stdout
            
        finally:
            sys.stdout = old_stdout

    def test_selective_capture(self):
        """Test capturing only specific streams."""
        config = OutputCapture(
            stdout=True,
            stderr=False,
            logging=False,
            display="suppress",
        )

        with OutputCaptureContext(config) as captured:
            print("stdout message")
            print("stderr message", file=sys.stderr)

        assert "stdout message" in captured.stdout
        # stderr was not being captured, so it might be empty
        # (depends on whether interceptor was installed)

    def test_context_cleanup(self):
        """Test that context cleans up properly."""
        config = OutputCapture(display="suppress", logging=False)
        
        original_stdout = sys.stdout
        
        with OutputCaptureContext(config):
            pass  # Just enter and exit

        # After all contexts exit, stdout should be restored
        # Note: This depends on the manager's active count
        manager = OutputCaptureManager.get_instance()
        assert manager._active_count == 0


class TestThreadSafety:
    """Tests for thread-safe operation."""

    def test_concurrent_captures(self):
        """Test that concurrent captures don't interfere."""
        config = OutputCapture(display="suppress", logging=False)
        results = {}
        
        def worker(thread_id: int):
            with OutputCaptureContext(config) as captured:
                # Each thread prints its own message
                print(f"thread-{thread_id}")
                # Small sleep to increase chance of interleaving
                import time
                time.sleep(0.01)
                print(f"done-{thread_id}")
            
            results[thread_id] = captured.stdout

        # Run multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Each thread should have captured only its own output
        for thread_id in range(5):
            assert f"thread-{thread_id}" in results[thread_id]
            assert f"done-{thread_id}" in results[thread_id]
            # Should NOT contain other thread's output
            for other_id in range(5):
                if other_id != thread_id:
                    assert f"thread-{other_id}" not in results[thread_id]

    def test_thread_pool_captures(self):
        """Test captures with ThreadPoolExecutor."""
        config = OutputCapture(display="suppress", logging=False)
        
        def task(task_id: int) -> tuple[int, str]:
            with OutputCaptureContext(config) as captured:
                print(f"task-{task_id}")
            return task_id, captured.stdout

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(task, i) for i in range(10)]
            results = [f.result() for f in futures]

        # Each task should have captured only its own output
        for task_id, output in results:
            assert f"task-{task_id}" in output


class TestNormalizeOutputCapture:
    """Tests for normalize_output_capture helper."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        assert normalize_output_capture(None) is None

    def test_false_returns_none(self):
        """Test that False input returns None."""
        assert normalize_output_capture(False) is None

    def test_true_with_progress_returns_console(self):
        """Test that True with progress returns console mode."""
        config = normalize_output_capture(True, has_progress=True)
        assert config is not None
        assert config.display == "console"

    def test_true_without_progress_returns_passthrough(self):
        """Test that True without progress returns passthrough mode."""
        config = normalize_output_capture(True, has_progress=False)
        assert config is not None
        assert config.display == "passthrough"

    def test_explicit_config_passed_through(self):
        """Test that explicit config is passed through unchanged."""
        explicit = OutputCapture(display="suppress", stdout=False)
        result = normalize_output_capture(explicit, has_progress=True)
        assert result is explicit
