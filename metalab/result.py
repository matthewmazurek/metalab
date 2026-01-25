"""
ResultHandle: Query interface for experiment results.

Provides:
- Access to RunRecords
- Tabular view of results
- Artifact loading
- Filtering capabilities
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from metalab.store.base import Store

from metalab.types import ArtifactDescriptor, RunRecord, Status


class ResultHandle:
    """
    Handle for querying and accessing experiment results.

    Wraps a Store and provides convenient access to:
    - Run records as a table
    - Artifact loading
    - Filtering by status, tags, or parameters

    Example:
        result = metalab.run(experiment, store="./runs")

        # Get tabular view
        df = result.table(as_dataframe=True)

        # Load an artifact
        points = result.load(run_id, "points")

        # Filter results
        successful = result.filter(status="success")
    """

    def __init__(
        self,
        store: Store,
        records: list[RunRecord],
    ) -> None:
        """
        Initialize the result handle.

        Args:
            store: The store containing artifacts.
            records: List of RunRecords from the experiment.
        """
        self._store = store
        self._records = records

    @property
    def records(self) -> list[RunRecord]:
        """Get all run records."""
        return list(self._records)

    def table(self, as_dataframe: bool = False) -> list[dict[str, Any]] | Any:
        """
        Get results as a table.

        Args:
            as_dataframe: If True, return a pandas DataFrame (requires pandas).

        Returns:
            List of dicts by default, or DataFrame if as_dataframe=True.

        Raises:
            ImportError: If as_dataframe=True but pandas is not installed.
        """
        rows = []
        for record in self._records:
            row = {
                "run_id": record.run_id,
                "experiment_id": record.experiment_id,
                "status": record.status.value,
                "duration_ms": record.duration_ms,
                "started_at": record.started_at.isoformat(),
                "finished_at": record.finished_at.isoformat(),
                "context_fingerprint": record.context_fingerprint,
                "params_fingerprint": record.params_fingerprint,
                "seed_fingerprint": record.seed_fingerprint,
                # Flatten metrics
                **record.metrics,
                # Include tags as comma-separated string
                "tags": ",".join(record.tags) if record.tags else "",
            }
            rows.append(row)

        if not as_dataframe:
            return rows

        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError as e:
            raise ImportError(
                "pandas is required for as_dataframe=True. "
                "Install it with: pip install metalab[pandas]"
            ) from e

    def load(self, run_id: str, artifact_name: str) -> Any:
        """
        Load an artifact from a run.

        Args:
            run_id: The run identifier.
            artifact_name: The name of the artifact.

        Returns:
            The deserialized artifact.

        Raises:
            FileNotFoundError: If the artifact doesn't exist.
        """
        # Get artifact descriptor
        artifacts = self._store.list_artifacts(run_id)
        descriptor = None
        for art in artifacts:
            if art.name == artifact_name:
                descriptor = art
                break

        if descriptor is None:
            raise FileNotFoundError(
                f"Artifact '{artifact_name}' not found in run {run_id}"
            )

        # Load and deserialize based on format
        from pathlib import Path

        from metalab.capture.registry import SerializerRegistry

        registry = SerializerRegistry()
        data_path = Path(descriptor.uri)

        # Find appropriate serializer by kind
        serializer = registry.get(descriptor.kind)
        if serializer is None:
            # Fall back to reading raw bytes
            return self._store.get_artifact(descriptor.uri)

        return serializer.load(data_path)

    def filter(
        self,
        status: str | Status | None = None,
        tags: list[str] | None = None,
        **params: Any,
    ) -> ResultHandle:
        """
        Filter results by criteria.

        Args:
            status: Filter by status ("success", "failed", "cancelled").
            tags: Filter by tags (all must be present).
            **params: Filter by metric values.

        Returns:
            A new ResultHandle with filtered records.
        """
        filtered = self._records

        # Filter by status
        if status is not None:
            if isinstance(status, str):
                status = Status(status)
            filtered = [r for r in filtered if r.status == status]

        # Filter by tags
        if tags is not None:
            tag_set = set(tags)
            filtered = [r for r in filtered if tag_set.issubset(set(r.tags))]

        # Filter by metrics/params
        for key, value in params.items():
            filtered = [r for r in filtered if r.metrics.get(key) == value]

        return ResultHandle(store=self._store, records=filtered)

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self._records)

    def __iter__(self) -> Iterator[RunRecord]:
        """Iterate over records."""
        return iter(self._records)

    def __getitem__(self, index: int) -> RunRecord:
        """Get a record by index."""
        return self._records[index]

    @property
    def successful(self) -> ResultHandle:
        """Get only successful runs."""
        return self.filter(status=Status.SUCCESS)

    @property
    def failed(self) -> ResultHandle:
        """Get only failed runs."""
        return self.filter(status=Status.FAILED)

    def summary(self) -> dict[str, Any]:
        """
        Get a summary of the results.

        Returns:
            Dict with counts and basic statistics.
        """
        total = len(self._records)
        by_status = {}
        for record in self._records:
            status = record.status.value
            by_status[status] = by_status.get(status, 0) + 1

        durations = [r.duration_ms for r in self._records]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "total_runs": total,
            "by_status": by_status,
            "avg_duration_ms": avg_duration,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
        }
