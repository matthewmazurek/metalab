"""
Store protocol: Backend-agnostic persistence interface.

The Store abstraction covers:
- Run records (append/query)
- Artifacts (write/read)
- Logs (optional)
- Experiment manifests

The supported implementation is the filesystem run store. The protocol keeps
core orchestration decoupled from concrete file IO without implying database,
service, or object-store backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from metalab.types import ArtifactDescriptor, RunRecord


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

    def get_run_statuses(self, run_ids: list[str]) -> dict[str, Any]:
        """Bulk lookup run statuses by run id."""
        ...

    def successful_run_ids(self, experiment_id: str) -> set[str]:
        """Return successful run ids for an experiment."""
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

    # Experiment manifest operations

    def get_experiment_manifest(self, experiment_id: str) -> dict[str, Any] | None:
        """
        Retrieve the experiment manifest by ID.

        Returns the most recent manifest for the given experiment_id.
        The manifest contains experiment-level metadata including:
        - name, version, description, tags
        - metadata dict (user-defined)
        - operation info, params, seeds config

        Args:
            experiment_id: The experiment identifier (name:version).

        Returns:
            The experiment manifest dict, or None if not found.
        """
        ...
