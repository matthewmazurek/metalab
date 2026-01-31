"""
Capture: Interface for emitting metrics, artifacts, and logs.

The Capture object is passed to Operation.run() and provides methods
for recording experimental outputs without returning large objects.

Logging:
    Logs are streamed in real-time to the store using Python's logging module.
    This enables:
    - Real-time log visibility (tail -f works)
    - Crash resilience (logs written immediately)
    - Integration with third-party library logging via subscribe_logger()
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from metalab.capture.registry import SerializerRegistry
from metalab.types import ArtifactDescriptor

if TYPE_CHECKING:
    from metalab.store.base import Store


class Capture:
    """
    Interface for capturing metrics, artifacts, and logs during a run.

    The Capture object is created by the executor and passed to the operation.
    Logs are streamed in real-time to the store.

    Example:
        @metalab.operation
        def my_operation(context, params, seeds, capture, runtime):
            capture.log("Starting operation")

            # Subscribe to third-party library logs
            capture.subscribe_logger("dynamo")
            capture.subscribe_logger("sklearn", level=logging.WARNING)

            # Capture scalar metrics
            capture.metric("accuracy", 0.95)
            capture.log_metrics({"loss": 0.05, "epoch": 10})

            # Capture artifacts (serialized automatically)
            capture.artifact("predictions", predictions_array, kind="numpy")

            # Capture a file you generated
            capture.file("plot", "/tmp/plot.png", kind="image")

            capture.log("Operation completed")
            # No return needed - success is implicit
    """

    # Default log format
    LOG_FORMAT = "%(asctime)s [%(levelname)-7s] [%(name)s] %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        store: Store,
        run_id: str,
        registry: SerializerRegistry | None = None,
        allow_pickle: bool = False,
        artifact_dir: Path | None = None,
        worker_id: str | None = None,
    ) -> None:
        """
        Initialize the capture interface.

        Args:
            store: The store to persist artifacts to.
            run_id: The ID of the current run.
            registry: Serializer registry (created if None).
            allow_pickle: Whether to allow pickle serialization.
            artifact_dir: Directory for temporary artifact files.
            worker_id: Worker identifier for log messages (e.g., "thread:2", "process:3").
        """
        self._store = store
        self._run_id = run_id
        self._registry = registry or SerializerRegistry(allow_pickle=allow_pickle)
        self._allow_pickle = allow_pickle
        self._artifact_dir = artifact_dir or Path("/tmp/metalab_artifacts") / run_id

        # Buffered data (metrics and artifacts only - logs stream directly)
        self._metrics: dict[str, Any] = {}
        self._stepped_metrics: list[dict[str, Any]] = []
        self._artifacts: list[ArtifactDescriptor] = []
        self._finalized = False

        # Logging setup
        self._worker_id = worker_id
        self._log_handler: logging.Handler | None = None
        self._log_path: Path | None = None
        self._subscribed_loggers: list[tuple[logging.Logger, logging.Handler]] = []

        # Set up streaming logger
        self._setup_streaming_logger()

    def _setup_streaming_logger(self) -> None:
        """Set up the streaming file logger."""
        # Try to get a direct log path from the store (FileStore has this)
        if hasattr(self._store, "get_log_path"):
            self._log_path = self._store.get_log_path(self._run_id, "run")
        elif hasattr(self._store, "root"):
            # Fallback: construct path manually for FileStore-like stores
            self._log_path = Path(self._store.root) / "logs" / f"{self._run_id}_run.log"
        else:
            # Remote store - use artifact_dir as scratch space, upload at finalize
            self._log_path = self._artifact_dir / "run.log"

        # Ensure parent directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file handler for streaming
        self._log_handler = logging.FileHandler(self._log_path, mode="a", encoding="utf-8")
        self._log_handler.setLevel(logging.DEBUG)

        # Format includes logger name for distinguishing sources
        formatter = logging.Formatter(self.LOG_FORMAT, datefmt=self.LOG_DATE_FORMAT)
        self._log_handler.setFormatter(formatter)

        # Create the run's logger
        logger_name = f"metalab.run.{self._run_id[:8]}"
        if self._worker_id:
            logger_name = f"{logger_name}.{self._worker_id}"

        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(self._log_handler)
        self._logger.propagate = False  # Don't propagate to root

    def subscribe_logger(
        self,
        name: str,
        level: int = logging.DEBUG,
    ) -> None:
        """
        Subscribe to a named logger to capture its output.

        This attaches the capture's file handler to the specified logger,
        capturing all its log messages to the run's log file. Works even
        if the logger has propagate=False (like dynamo-release).

        Args:
            name: The logger name (e.g., "dynamo", "sklearn", "tensorflow").
            level: Minimum log level to capture (default: DEBUG).

        Example:
            # Capture all dynamo logs
            capture.subscribe_logger("dynamo")

            # Capture sklearn warnings and above
            capture.subscribe_logger("sklearn", level=logging.WARNING)

            # Now any logging from these libraries is captured
            import dynamo as dyn
            dyn.tl.dynamics(adata)  # Logs captured automatically
        """
        if self._log_handler is None:
            return

        logger = logging.getLogger(name)

        # Add our handler to their logger
        logger.addHandler(self._log_handler)

        # Ensure the logger level allows our desired messages through
        if logger.level == logging.NOTSET or logger.level > level:
            logger.setLevel(level)

        # Track for cleanup
        self._subscribed_loggers.append((logger, self._log_handler))

        # Log the subscription
        self._logger.debug(f"Subscribed to logger: {name} (level={logging.getLevelName(level)})")

    def unsubscribe_logger(self, name: str) -> None:
        """
        Unsubscribe from a named logger.

        Args:
            name: The logger name to unsubscribe from.
        """
        if self._log_handler is None:
            return

        logger = logging.getLogger(name)

        # Remove our handler if present
        if self._log_handler in logger.handlers:
            logger.removeHandler(self._log_handler)

        # Remove from tracking list
        self._subscribed_loggers = [
            (lg, h) for lg, h in self._subscribed_loggers if lg.name != name
        ]

    def log(
        self,
        message: str,
        level: str = "info",
    ) -> None:
        """
        Log a message for this run.

        Messages are streamed immediately to the run's log file, enabling
        real-time visibility (e.g., via tail -f).

        Args:
            message: The log message.
            level: Log level - "debug", "info", "warning", "error" (default: "info").

        Example:
            capture.log("Starting optimization")
            capture.log(f"Iteration {i}: loss={loss:.4f}")
            capture.log("Convergence failed", level="warning")
        """
        log_method = getattr(self._logger, level.lower(), self._logger.info)
        log_method(message)

    @property
    def logger(self) -> logging.Logger:
        """
        Get the Python logger for this run.

        Use this for direct access to Python's logging API.

        Example:
            capture.logger.info("Using standard logging API")
            capture.logger.exception("Caught error", exc_info=True)
        """
        return self._logger

    def _flush_logs(self) -> None:
        """Flush any buffered log content."""
        if self._log_handler:
            self._log_handler.flush()

    @property
    def metrics(self) -> dict[str, Any]:
        """Get the captured metrics."""
        return dict(self._metrics)

    @property
    def artifacts(self) -> list[ArtifactDescriptor]:
        """Get the captured artifact descriptors."""
        return list(self._artifacts)

    def metric(
        self, name: str, value: float | int | str | bool, step: int | None = None
    ) -> None:
        """
        Capture a scalar metric.

        Args:
            name: The metric name.
            value: The metric value (must be a scalar).
            step: Optional step number (for time-series metrics).
        """
        if step is None:
            self._metrics[name] = value
        else:
            self._stepped_metrics.append(
                {
                    "name": name,
                    "value": value,
                    "step": step,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def log_metrics(self, values: dict[str, Any], step: int | None = None) -> None:
        """
        Capture multiple metrics at once.

        Args:
            values: Dict of metric names to values.
            step: Optional step number (for time-series metrics).
        """
        for name, value in values.items():
            self.metric(name, value, step=step)

    def artifact(
        self,
        name: str,
        obj: Any,
        *,
        kind: str | None = None,
        format: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactDescriptor:
        """
        Capture an artifact by serializing an object.

        The serializer is selected automatically based on the object type,
        unless kind is explicitly specified.

        Args:
            name: The artifact name.
            obj: The object to serialize.
            kind: Explicit serializer kind (e.g., "json", "numpy", "pickle").
            format: Explicit format (usually inferred from serializer).
            metadata: Additional metadata to attach.

        Returns:
            The ArtifactDescriptor for the saved artifact.
        """
        # Find appropriate serializer
        serializer = self._registry.find(obj, kind=kind)

        # Serialize to temp location
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        base_path = self._artifact_dir / name
        result = serializer.dump(obj, base_path)

        # Compute content hash
        artifact_path = Path(result["path"])
        content_hash = self._compute_hash(artifact_path)

        # Create descriptor (include run_id in metadata for storage)
        artifact_id = str(uuid.uuid4())
        full_metadata = {"_run_id": self._run_id, **(metadata or {})}
        descriptor = ArtifactDescriptor(
            artifact_id=artifact_id,
            name=name,
            kind=kind or serializer.kind,
            format=format or result.get("format", serializer.format),
            uri=result["path"],
            content_hash=content_hash,
            size_bytes=result.get("size_bytes"),
            metadata=full_metadata,
        )

        # Store the artifact
        self._store.put_artifact(artifact_path, descriptor)
        self._artifacts.append(descriptor)

        return descriptor

    def file(
        self,
        name: str,
        path: str | Path,
        *,
        kind: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactDescriptor:
        """
        Capture a file that was generated by the operation.

        Use this for files you've already written (e.g., plots, reports).

        Args:
            name: The artifact name.
            path: Path to the file.
            kind: The kind of artifact.
            metadata: Additional metadata.

        Returns:
            The ArtifactDescriptor for the saved artifact.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact file not found: {path}")

        # Compute content hash
        content_hash = self._compute_hash(path)

        # Infer format from extension
        format_ext = path.suffix.lstrip(".") or "binary"

        # Create descriptor (include run_id in metadata for storage)
        artifact_id = str(uuid.uuid4())
        full_metadata = {"_run_id": self._run_id, **(metadata or {})}
        descriptor = ArtifactDescriptor(
            artifact_id=artifact_id,
            name=name,
            kind=kind or "file",
            format=format_ext,
            uri=str(path),
            content_hash=content_hash,
            size_bytes=path.stat().st_size,
            metadata=full_metadata,
        )

        # Store the artifact
        self._store.put_artifact(path, descriptor)
        self._artifacts.append(descriptor)

        return descriptor

    def figure(
        self,
        name: str,
        fig: Any,
        *,
        format: str = "png",
        dpi: int = 150,
        bbox_inches: str = "tight",
        metadata: dict[str, Any] | None = None,
        close: bool = True,
    ) -> ArtifactDescriptor:
        """
        Capture a matplotlib figure as an image artifact.

        This is a convenience method that handles saving the figure to a
        temporary file and capturing it, eliminating boilerplate.

        Args:
            name: The artifact name.
            fig: A matplotlib Figure object.
            format: Image format (default: "png"). Options: png, pdf, svg, jpg.
            dpi: Resolution in dots per inch (default: 150).
            bbox_inches: Bounding box (default: "tight").
            metadata: Additional metadata to attach.
            close: Whether to close the figure after saving (default: True).

        Returns:
            The ArtifactDescriptor for the saved artifact.

        Example:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 9])
            ax.set_title("My Plot")

            capture.figure("my_plot", fig)  # Saves and closes figure
        """
        # Ensure artifact directory exists
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

        # Generate path for the figure
        fig_path = self._artifact_dir / f"{name}.{format}"

        # Save the figure
        fig.savefig(fig_path, dpi=dpi, bbox_inches=bbox_inches, format=format)

        # Optionally close the figure
        if close:
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except ImportError:
                pass  # matplotlib not available, skip closing

        # Capture as file
        return self.file(
            name=name,
            path=fig_path,
            kind="image",
            metadata=metadata,
        )

    def flush(self) -> None:
        """Flush any buffered log content to disk."""
        self._flush_logs()

    def finalize(self) -> dict[str, Any]:
        """
        Finalize capture and return collected data.

        This is called by the executor in a finally block to ensure
        partial results are captured even on failure. Cleans up logging
        handlers and uploads logs for remote stores.

        Returns:
            Dict containing metrics, artifacts, and logs.
        """
        if self._finalized:
            return self._get_summary()

        self._finalized = True

        # Clean up subscribed loggers
        for logger, handler in self._subscribed_loggers:
            try:
                logger.removeHandler(handler)
            except Exception:
                pass  # Best effort cleanup

        self._subscribed_loggers.clear()

        # Close and flush the main log handler
        if self._log_handler:
            try:
                self._log_handler.flush()
                self._log_handler.close()
            except Exception:
                pass  # Best effort cleanup

            # Remove handler from our logger
            if self._logger and self._log_handler in self._logger.handlers:
                self._logger.removeHandler(self._log_handler)

        # For remote stores, upload the log file
        # (FileStore writes directly, no upload needed)
        if self._log_path and self._log_path.exists():
            if not hasattr(self._store, "get_log_path") and not hasattr(self._store, "root"):
                # Remote store - upload via put_log
                try:
                    log_content = self._log_path.read_text(encoding="utf-8")
                    self._store.put_log(self._run_id, "run", log_content)
                except Exception:
                    pass  # Best effort

        return self._get_summary()

    def _get_summary(self) -> dict[str, Any]:
        """Get a summary of captured data."""
        return {
            "metrics": self._metrics,
            "stepped_metrics": self._stepped_metrics,
            "artifacts": self._artifacts,
        }

    @staticmethod
    def _compute_hash(path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
