"""
Log capture utilities.

Provides context managers for capturing:
- stdout/stderr (LogCapture)
- Python logging output (LogHandler)
- warnings (WarningCapture)
"""

from __future__ import annotations

import io
import logging
import sys
import warnings
from contextlib import contextmanager
from typing import Generator, TextIO


class LogCapture:
    """
    Context manager for capturing stdout and stderr.

    Example:
        with LogCapture() as cap:
            print("hello")
            sys.stderr.write("error\\n")

        print(cap.stdout)  # "hello\\n"
        print(cap.stderr)  # "error\\n"
    """

    def __init__(self) -> None:
        self._stdout_buffer = io.StringIO()
        self._stderr_buffer = io.StringIO()
        self._old_stdout: TextIO | None = None
        self._old_stderr: TextIO | None = None

    def __enter__(self) -> LogCapture:
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = self._stdout_buffer
        sys.stderr = self._stderr_buffer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
        if self._old_stderr is not None:
            sys.stderr = self._old_stderr

    @property
    def stdout(self) -> str:
        """Get captured stdout content."""
        return self._stdout_buffer.getvalue()

    @property
    def stderr(self) -> str:
        """Get captured stderr content."""
        return self._stderr_buffer.getvalue()


class LogHandler(logging.Handler):
    """
    Logging handler that captures log records to a buffer.

    Example:
        handler = LogHandler()
        logger = logging.getLogger("mylogger")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info("test message")

        print(handler.logs)  # ["INFO:mylogger:test message"]
    """

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self._records: list[logging.LogRecord] = []
        self._buffer = io.StringIO()

        # Use a simple formatter
        self.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        """Store the log record."""
        self._records.append(record)
        msg = self.format(record)
        self._buffer.write(msg + "\n")

    @property
    def records(self) -> list[logging.LogRecord]:
        """Get all captured log records."""
        return list(self._records)

    @property
    def logs(self) -> list[str]:
        """Get formatted log messages."""
        return [self.format(r) for r in self._records]

    @property
    def text(self) -> str:
        """Get all logs as a single string."""
        return self._buffer.getvalue()

    def clear(self) -> None:
        """Clear captured logs."""
        self._records.clear()
        self._buffer = io.StringIO()


class WarningCapture:
    """
    Context manager for capturing warnings.

    Example:
        with WarningCapture() as cap:
            warnings.warn("test warning")

        print(cap.warnings)  # [("test warning", UserWarning, ...)]
    """

    def __init__(self) -> None:
        self._warnings: list[warnings.WarningMessage] = []
        self._catch_warnings = None

    def __enter__(self) -> WarningCapture:
        self._catch_warnings = warnings.catch_warnings(record=True)
        self._warnings = self._catch_warnings.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._catch_warnings is not None:
            self._catch_warnings.__exit__(exc_type, exc_val, exc_tb)

    @property
    def warnings(self) -> list[warnings.WarningMessage]:
        """Get captured warnings."""
        return list(self._warnings)

    @property
    def messages(self) -> list[str]:
        """Get warning messages as strings."""
        return [str(w.message) for w in self._warnings]


@contextmanager
def capture_all() -> Generator[tuple[LogCapture, LogHandler, WarningCapture], None, None]:
    """
    Context manager that captures stdout, stderr, logging, and warnings.

    Example:
        with capture_all() as (streams, logs, warns):
            print("hello")
            logging.warning("test")
            warnings.warn("oops")

        print(streams.stdout)
        print(logs.text)
        print(warns.messages)
    """
    stream_capture = LogCapture()
    log_handler = LogHandler()
    warning_capture = WarningCapture()

    # Add handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    try:
        with stream_capture, warning_capture:
            yield stream_capture, log_handler, warning_capture
    finally:
        root_logger.removeHandler(log_handler)
