"""
Output capture utilities for capturing stdout/stderr/logging from operations.

Provides thread-safe output capture that can:
- Route output through rich console (progress-bar safe)
- Suppress output entirely
- Pass output through to terminal unchanged

All captured output is stored to the run's log directory.
"""

from __future__ import annotations

import contextvars
import io
import logging
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, TextIO

if TYPE_CHECKING:
    from rich.console import Console


# Context variable for thread-local capture buffers
_capture_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "output_capture", default=None
)


@dataclass
class OutputCapture:
    """
    Configuration for capturing stdout/stderr/logging from operations.

    Controls what to capture and how to display it. All captured output
    is stored to the run's log directory regardless of display mode.

    Attributes:
        stdout: Capture stdout (print statements, etc.)
        stderr: Capture stderr
        logging: Capture Python logging output
        warnings: Capture Python warnings
        display: How to handle captured output for display:
            - "console": Route through rich console (progress-bar safe)
            - "suppress": Don't display, only capture
            - "passthrough": Don't intercept display (capture still works)

    Example:
        # Capture everything, display through console (default)
        OutputCapture()

        # Suppress all output (silent mode)
        OutputCapture.suppress()

        # Only capture logging, let stdout through normally
        OutputCapture(stdout=False, stderr=False, logging=True, display="passthrough")
    """

    stdout: bool = True
    stderr: bool = True
    logging: bool = True
    warnings: bool = False
    display: Literal["console", "suppress", "passthrough"] = "console"

    # Convenience constructors
    @classmethod
    def suppress(cls) -> OutputCapture:
        """Suppress all output, only capture to logs."""
        return cls(display="suppress")

    @classmethod
    def console(cls) -> OutputCapture:
        """Route all output through console (default)."""
        return cls(display="console")

    @classmethod
    def passthrough(cls) -> OutputCapture:
        """Don't intercept display, but still capture to logs."""
        return cls(display="passthrough")


@dataclass
class CapturedOutput:
    """Container for captured output from an operation."""

    stdout: str = ""
    stderr: str = ""
    logging_records: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def has_content(self) -> bool:
        """Check if any output was captured."""
        return bool(
            self.stdout or self.stderr or self.logging_records or self.warnings
        )

    def format_logging(self) -> str:
        """Format logging records as text."""
        lines = []
        for record in self.logging_records:
            lines.append(record.get("formatted", record.get("message", "")))
        return "\n".join(lines)


class OutputInterceptor:
    """
    Intercepts stdout/stderr and routes based on configuration.

    Thread-safe: uses context variables to maintain per-operation buffers.
    The interceptor is installed globally but only captures when a capture
    context is active for the current thread/task.
    """

    def __init__(
        self,
        original: TextIO,
        stream_name: str,
        console: Console | None = None,
        display: str = "console",
    ):
        self._original = original
        self._stream_name = stream_name
        self._console = console
        self._display = display
        self._lock = threading.Lock()
        # Track encoding attribute for compatibility
        self.encoding = getattr(original, "encoding", "utf-8")

    def write(self, s: str) -> int:
        # Get thread-local capture context
        ctx = _capture_context.get()

        # Always capture if context exists and this stream is being captured
        if ctx is not None:
            buffer = ctx.get(self._stream_name)
            if buffer is not None:
                buffer.write(s)

        # Handle display based on mode
        display_mode = ctx.get("display") if ctx else self._display

        if display_mode == "suppress":
            return len(s)
        elif display_mode == "console" and self._console is not None:
            # Route through rich console (progress-bar safe)
            if s.strip():  # Don't print empty strings
                with self._lock:
                    self._console.print(s, end="", markup=False, highlight=False)
            return len(s)
        else:
            # Passthrough to original
            return self._original.write(s)

    def flush(self) -> None:
        ctx = _capture_context.get()
        display_mode = ctx.get("display") if ctx else self._display

        if display_mode != "suppress":
            if display_mode == "console" and self._console:
                pass  # Console handles its own flushing
            else:
                self._original.flush()

    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return self._original.isatty()

    # Forward other TextIO methods
    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class CaptureLogHandler(logging.Handler):
    """
    Logging handler that captures records to a thread-local list.

    Only captures when a capture context is active.
    """

    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        ctx = _capture_context.get()
        if ctx is None:
            return

        target = ctx.get("logging_records")
        if target is None:
            return

        try:
            target.append(
                {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "formatted": self.format(record),
                }
            )
        except Exception:
            # Don't let logging failures break the operation
            pass


class OutputCaptureManager:
    """
    Manages global output interception.

    This class handles installing/uninstalling stream interceptors and
    the logging handler. It's designed to be used as a singleton.
    """

    _instance: OutputCaptureManager | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._active_count = 0
        self._old_stdout: TextIO | None = None
        self._old_stderr: TextIO | None = None
        self._stdout_interceptor: OutputInterceptor | None = None
        self._stderr_interceptor: OutputInterceptor | None = None
        self._log_handler: CaptureLogHandler | None = None
        self._console: Console | None = None

    @classmethod
    def get_instance(cls) -> OutputCaptureManager:
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def set_console(self, console: Console | None) -> None:
        """Set the rich console to use for output routing."""
        self._console = console
        # Update existing interceptors if they exist
        if self._stdout_interceptor is not None:
            self._stdout_interceptor._console = console
        if self._stderr_interceptor is not None:
            self._stderr_interceptor._console = console

    def activate(self, config: OutputCapture) -> None:
        """
        Activate output interception.

        Called when entering an OutputCaptureContext. Only installs
        interceptors on first activation.
        """
        with self._lock:
            self._active_count += 1

            if self._active_count == 1:
                # First activation - install interceptors
                self._install_interceptors(config)

    def deactivate(self) -> None:
        """
        Deactivate output interception.

        Called when exiting an OutputCaptureContext. Only removes
        interceptors when all contexts have exited.
        """
        with self._lock:
            self._active_count -= 1

            if self._active_count == 0:
                # Last deactivation - remove interceptors
                self._remove_interceptors()

    def _install_interceptors(self, config: OutputCapture) -> None:
        """Install stdout/stderr interceptors and logging handler."""
        # Install stdout interceptor
        if config.stdout:
            self._old_stdout = sys.stdout
            self._stdout_interceptor = OutputInterceptor(
                sys.stdout,
                "stdout",
                console=self._console,
                display=config.display,
            )
            sys.stdout = self._stdout_interceptor  # type: ignore

        # Install stderr interceptor
        if config.stderr:
            self._old_stderr = sys.stderr
            self._stderr_interceptor = OutputInterceptor(
                sys.stderr,
                "stderr",
                console=self._console,
                display=config.display,
            )
            sys.stderr = self._stderr_interceptor  # type: ignore

        # Install logging handler
        if config.logging:
            self._log_handler = CaptureLogHandler()
            logging.getLogger().addHandler(self._log_handler)

    def _remove_interceptors(self) -> None:
        """Remove stdout/stderr interceptors and logging handler."""
        # Restore stdout
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
            self._old_stdout = None
            self._stdout_interceptor = None

        # Restore stderr
        if self._old_stderr is not None:
            sys.stderr = self._old_stderr
            self._old_stderr = None
            self._stderr_interceptor = None

        # Remove logging handler
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None


class OutputCaptureContext:
    """
    Context manager for capturing output during operation execution.

    Thread-safe: each operation gets its own capture buffer via context vars.
    Multiple operations can run concurrently with independent capture.

    Example:
        config = OutputCapture(display="suppress")
        with OutputCaptureContext(config) as captured:
            print("Hello")
            logging.info("World")

        print(captured.stdout)  # "Hello\n"
        print(captured.format_logging())  # "... - INFO - World"
    """

    def __init__(
        self,
        config: OutputCapture,
        console: Console | None = None,
    ):
        self._config = config
        self._console = console
        self._captured = CapturedOutput()
        self._token: contextvars.Token | None = None
        self._manager = OutputCaptureManager.get_instance()

    def __enter__(self) -> CapturedOutput:
        # Set console on manager
        if self._console is not None:
            self._manager.set_console(self._console)

        # Set up thread-local capture buffers
        ctx: dict[str, Any] = {
            "display": self._config.display,
            "stdout": io.StringIO() if self._config.stdout else None,
            "stderr": io.StringIO() if self._config.stderr else None,
            "logging_records": [] if self._config.logging else None,
            "warnings": [] if self._config.warnings else None,
        }
        self._token = _capture_context.set(ctx)

        # Activate global interception
        self._manager.activate(self._config)

        return self._captured

    def __exit__(self, *args: Any) -> None:
        # Collect captured output from context
        ctx = _capture_context.get()
        if ctx:
            if ctx.get("stdout"):
                self._captured.stdout = ctx["stdout"].getvalue()
            if ctx.get("stderr"):
                self._captured.stderr = ctx["stderr"].getvalue()
            if ctx.get("logging_records"):
                self._captured.logging_records = ctx["logging_records"]
            if ctx.get("warnings"):
                self._captured.warnings = ctx["warnings"]

        # Reset context
        if self._token is not None:
            _capture_context.reset(self._token)

        # Deactivate global interception
        self._manager.deactivate()


def normalize_output_capture(
    capture_output: bool | OutputCapture | None,
    has_progress: bool = False,
) -> OutputCapture | None:
    """
    Normalize output capture configuration.

    Args:
        capture_output: User-provided configuration.
        has_progress: Whether progress tracking is enabled.

    Returns:
        Normalized OutputCapture or None if disabled.
    """
    if capture_output is None or capture_output is False:
        return None

    if capture_output is True:
        # Sensible defaults based on progress
        if has_progress:
            return OutputCapture.console()
        else:
            return OutputCapture.passthrough()

    return capture_output
