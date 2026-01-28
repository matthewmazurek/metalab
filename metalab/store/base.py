"""
Store protocol: Backend-agnostic persistence interface.

The Store abstraction covers:
- Run records (append/query)
- Artifacts (write/read)
- Logs (optional)

Backend implementations can be:
- Filesystem (FileStore)
- S3/GCS object store
- MLflow/W&B adapter
- Database + blob store
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from metalab.types import ArtifactDescriptor, Metric, RunRecord


class Store(Protocol):
    """
    Protocol for storage backends.

    A Store persists RunRecords and artifacts. Implementations must
    provide atomic writes and handle concurrent access safely.
    """

    # Run record operations

    def put_run_record(self, record: RunRecord) -> None:
        """
        Persist a run record.

        Args:
            record: The RunRecord to store.
        """
        ...

    def get_run_record(self, run_id: str) -> RunRecord | None:
        """
        Retrieve a run record by ID.

        Args:
            run_id: The run identifier.

        Returns:
            The RunRecord, or None if not found.
        """
        ...

    def list_run_records(self, experiment_id: str | None = None) -> list[RunRecord]:
        """
        List run records, optionally filtered by experiment.

        Args:
            experiment_id: Optional filter by experiment ID.

        Returns:
            List of matching RunRecords.
        """
        ...

    def run_exists(self, run_id: str) -> bool:
        """
        Check if a run record exists.

        Args:
            run_id: The run identifier.

        Returns:
            True if the run exists.
        """
        ...

    # Artifact operations

    def put_artifact(
        self,
        data: bytes | Path,
        descriptor: ArtifactDescriptor,
    ) -> ArtifactDescriptor:
        """
        Store an artifact.

        Args:
            data: The artifact data (bytes or path to file).
            descriptor: Metadata about the artifact.

        Returns:
            Updated descriptor with final URI.
        """
        ...

    def get_artifact(self, uri: str) -> bytes:
        """
        Retrieve artifact data.

        Args:
            uri: The artifact URI.

        Returns:
            The artifact data as bytes.
        """
        ...

    def list_artifacts(self, run_id: str) -> list[ArtifactDescriptor]:
        """
        List artifacts for a run.

        Args:
            run_id: The run identifier.

        Returns:
            List of artifact descriptors.
        """
        ...

    # Derived metrics operations

    def put_derived(self, run_id: str, derived: dict[str, Metric]) -> None:
        """
        Persist derived metrics for a run.

        Args:
            run_id: The run identifier.
            derived: Dict of derived metric values.
        """
        ...

    def get_derived(self, run_id: str) -> dict[str, Metric] | None:
        """
        Retrieve derived metrics for a run.

        Args:
            run_id: The run identifier.

        Returns:
            Dict of derived metrics, or None if not found.
        """
        ...

    def derived_exists(self, run_id: str) -> bool:
        """
        Check if derived metrics exist for a run.

        Args:
            run_id: The run identifier.

        Returns:
            True if derived metrics exist.
        """
        ...

    # Optional log operations

    def put_log(
        self,
        run_id: str,
        name: str,
        content: str,
        label: str | None = None,
    ) -> None:
        """
        Store a log file for a run.

        Args:
            run_id: The run identifier.
            name: The log name (e.g., "stdout", "stderr", "logging").
            content: The log content.
            label: Optional human-readable label for the log filename.
                   If provided, filename becomes: {label}_{run_id[:8]}_{name}.log
                   If not provided, filename is: {run_id}_{name}.log
        """
        ...

    def get_log(self, run_id: str, name: str) -> str | None:
        """
        Retrieve a log file.

        Args:
            run_id: The run identifier.
            name: The log name.

        Returns:
            The log content, or None if not found.
        """
        ...
