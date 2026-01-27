"""
Capture: Interface for emitting metrics, artifacts, and logs.

The Capture object is passed to Operation.run() and provides methods
for recording experimental outputs without returning large objects.
"""

from __future__ import annotations

import hashlib
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
    It buffers data and persists it to the store.

    Example:
        @metalab.operation
        def my_operation(context, params, seeds, capture, runtime):
            capture.log("Starting operation")

            # Capture scalar metrics
            capture.metric("accuracy", 0.95)
            capture.metrics({"loss": 0.05, "epoch": 10})

            # Capture artifacts (serialized automatically)
            capture.artifact("predictions", predictions_array, kind="numpy")

            # Capture a file you generated
            capture.file("plot", "/tmp/plot.png", kind="image")

            capture.log("Operation completed")
            # No return needed - success is implicit
    """

    def __init__(
        self,
        store: Store,
        run_id: str,
        registry: SerializerRegistry | None = None,
        allow_pickle: bool = False,
        artifact_dir: Path | None = None,
        worker_id: str | None = None,
        immediate_logging: bool = False,
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
            immediate_logging: If True, log() calls write immediately to store.
        """
        self._store = store
        self._run_id = run_id
        self._registry = registry or SerializerRegistry(allow_pickle=allow_pickle)
        self._allow_pickle = allow_pickle
        self._artifact_dir = artifact_dir or Path("/tmp/metalab_artifacts") / run_id

        # Buffered data
        self._metrics: dict[str, Any] = {}
        self._stepped_metrics: list[dict[str, Any]] = []
        self._artifacts: list[ArtifactDescriptor] = []
        self._finalized = False

        # Logging
        self._logs: list[str] = []
        self._worker_id = worker_id
        self._immediate_logging = immediate_logging
        self._log_label: str | None = None

    def set_immediate_logging(self, enabled: bool) -> None:
        """
        Set whether logs are written immediately or buffered.

        For long-running operations, enable immediate logging so progress
        is visible even if the operation fails partway through.

        Args:
            enabled: If True, all subsequent log() calls write immediately.
                     If False, logs are buffered until finalize().

        Example:
            capture.set_immediate_logging(True)
            capture.log("Progress checkpoint")  # Written immediately
        """
        self._immediate_logging = enabled

    def set_log_label(self, label: str) -> None:
        """
        Set the label used for log filenames.

        This is typically called by the executor to set a human-readable
        label for the log file (e.g., based on parameters).

        Args:
            label: Human-readable label for log filenames.
        """
        self._log_label = label

    def log(
        self,
        message: str,
        level: str = "info",
        immediate: bool | None = None,
    ) -> None:
        """
        Log a message for this run.

        Messages are stored with timestamp, level, and worker ID, then saved
        to the run's log file. By default, logs are buffered and written when
        the operation completes. Use immediate=True or set_immediate_logging(True)
        for long-running operations.

        Args:
            message: The log message.
            level: Log level - "debug", "info", "warning", "error" (default: "info").
            immediate: Override immediate write behavior for this call.
                       None = use capture-level default (set via set_immediate_logging).

        Example:
            capture.log("Starting optimization")
            capture.log(f"Iteration {i}: loss={loss:.4f}")
            capture.log("Convergence failed", level="warning")

            # For long-running operations:
            capture.set_immediate_logging(True)
            capture.log("Progress checkpoint")  # Written immediately

            # Or override per-call:
            capture.log("Critical checkpoint", immediate=True)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        worker_part = f" [{self._worker_id}]" if self._worker_id else ""
        formatted = f"{timestamp} [{level.upper():7}]{worker_part} {message}"
        self._logs.append(formatted)

        # Determine if we should write immediately
        write_now = immediate if immediate is not None else self._immediate_logging
        if write_now:
            self._flush_logs()

    def _flush_logs(self) -> None:
        """Write buffered logs to store."""
        if self._logs:
            log_content = "\n".join(self._logs)
            self._store.put_log(self._run_id, "run", log_content, label=self._log_label)

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

    def metrics(self, values: dict[str, Any], step: int | None = None) -> None:
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
        """Persist any buffered data to the store."""
        self._flush_logs()

    def finalize(self) -> dict[str, Any]:
        """
        Finalize capture and return collected data.

        This is called by the executor in a finally block to ensure
        partial results are captured even on failure.

        Returns:
            Dict containing metrics, artifacts, and logs.
        """
        if self._finalized:
            return self._get_summary()

        self._finalized = True
        self._flush_logs()
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
