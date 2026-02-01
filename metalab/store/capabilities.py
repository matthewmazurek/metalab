"""
Store capability protocols: Optional interfaces for store features.

These protocols allow store backends to expose optional capabilities
that orchestration code can depend on via isinstance() checks,
replacing scattered hasattr() checks.

Pattern:
    if isinstance(store, SupportsWorkingDirectory):
        path = store.get_working_directory()
        # use path...
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


@runtime_checkable
class SupportsWorkingDirectory(Protocol):
    """
    Store capability: provides a local filesystem working directory.

    Used by:
    - SLURM executor: for staging array spec files
    - Runner: for persisting context manifests
    - Capture: for streaming log paths

    Only FileStore and similar filesystem-backed stores implement this.
    """

    def get_working_directory(self) -> Path:
        """
        Return the root directory of this store.

        Returns:
            Path to the store's filesystem root.
        """
        ...


@runtime_checkable
class SupportsExperimentManifests(Protocol):
    """
    Store capability: persists versioned experiment manifests.

    Used by:
    - Runner: to store experiment configuration at submission time

    Both FileStore (filesystem) and PostgresStore (database) implement this.
    """

    def put_experiment_manifest(
        self,
        experiment_id: str,
        manifest: dict[str, Any],
        timestamp: str | None = None,
    ) -> None:
        """
        Store an experiment manifest.

        Args:
            experiment_id: The experiment identifier.
            manifest: The manifest data to persist.
            timestamp: Optional timestamp string (defaults to current time).
        """
        ...


@runtime_checkable
class SupportsArtifactOpen(Protocol):
    """
    Store capability: opens artifacts for reading.

    Used by:
    - Result.artifact(): to load artifacts without URI-scheme special-casing

    Stores implement this to handle their specific URI schemes:
    - FileStore: file:// and plain paths
    - PostgresStore: pgblob:// inline blobs

    The returned file-like object should be a context manager supporting
    binary read mode.
    """

    def open_artifact(self, uri: str) -> BinaryIO:
        """
        Open an artifact for reading.

        Args:
            uri: The artifact URI (may be store-specific like pgblob://).

        Returns:
            A file-like object in binary read mode. The caller is
            responsible for closing it (use as context manager).

        Raises:
            FileNotFoundError: If the artifact doesn't exist.
        """
        ...


@runtime_checkable
class SupportsLogPath(Protocol):
    """
    Store capability: provides filesystem paths for log streaming.

    Used by:
    - Capture: to configure file handlers for streaming logs

    Only filesystem-backed stores implement this (FileStore).
    PostgresStore stores logs directly to the database instead.
    """

    def get_log_path(self, run_id: str, name: str) -> Path:
        """
        Get the filesystem path for a log file.

        The parent directory is created if needed.

        Args:
            run_id: The run identifier.
            name: The log name (e.g., "run", "stdout", "stderr").

        Returns:
            Path to the log file location.
        """
        ...


@runtime_checkable
class SupportsStructuredResults(Protocol):
    """
    Store capability: stores structured result data inline (not as artifacts).

    Used by:
    - Capture: to store intermediate data for derived metrics
    - Derived metric computation: to retrieve stored data

    PostgresStore implements this (stores in results table).
    FileStore does not (results stored in capture summary as fallback).
    """

    def put_result(
        self,
        run_id: str,
        name: str,
        data: Any,
        dtype: str | None = None,
        shape: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Store a structured result.

        Args:
            run_id: The run identifier.
            name: The result name.
            data: The data to store (JSON-serializable).
            dtype: Optional data type hint.
            shape: Optional shape hint (for arrays).
            metadata: Optional additional metadata.
        """
        ...

    def get_result(
        self,
        run_id: str,
        name: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve a structured result.

        Args:
            run_id: The run identifier.
            name: The result name.

        Returns:
            The result data dict or None if not found.
        """
        ...

    def list_results(self, run_id: str) -> list[str]:
        """
        List available result names for a run.

        Args:
            run_id: The run identifier.

        Returns:
            List of result names.
        """
        ...


@runtime_checkable
class SupportsLogListing(Protocol):
    """
    Store capability: lists and retrieves log files for a run.

    Used by:
    - Result.logs(): to enumerate available logs
    - Store transfer: to sync logs between stores

    Both FileStore and PostgresStore implement this.
    """

    def list_logs(self, run_id: str) -> list[str]:
        """
        List available log names for a run.

        Args:
            run_id: The run identifier.

        Returns:
            List of log names (e.g., ["run", "stdout", "stderr"]).
        """
        ...

    def get_log(self, run_id: str, name: str) -> str | None:
        """
        Retrieve log content.

        Args:
            run_id: The run identifier.
            name: The log name.

        Returns:
            The log content as a string, or None if not found.
        """
        ...
