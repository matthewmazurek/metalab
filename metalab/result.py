"""
Results: Query interface for experiment results.

Provides:
- Run class for single run access (metrics, artifacts)
- Results class for collections of runs
- Tabular view of results
- Artifact loading
- Filtering capabilities
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, overload

if TYPE_CHECKING:
    from metalab.store.base import Store

from metalab.types import ArtifactDescriptor, RunRecord, Status


class Run:
    """
    A single experiment run with access to its metrics and artifacts.

    The Run object wraps a RunRecord and provides convenient access to:
    - Run metadata (run_id, status, timestamps)
    - Metrics captured during the run
    - Artifacts stored for the run

    Example:
        result = metalab.run(experiment)
        run = result[0]  # Get first run

        # Access metrics
        print(run.metrics)
        print(run.status)

        # Load artifacts
        summary = run.artifact("summary")
        for desc in run.artifacts():
            print(f"  {desc.name}: {desc.kind}")
    """

    def __init__(self, record: RunRecord, store: Store) -> None:
        """
        Initialize the Run wrapper.

        Args:
            record: The underlying RunRecord.
            store: The store containing artifacts.
        """
        self._record = record
        self._store = store

    # Delegate properties to record
    @property
    def run_id(self) -> str:
        """The unique run identifier."""
        return self._record.run_id

    @property
    def experiment_id(self) -> str:
        """The experiment identifier."""
        return self._record.experiment_id

    @property
    def status(self) -> Status:
        """The run status (success, failed, cancelled)."""
        return self._record.status

    @property
    def metrics(self) -> dict[str, Any]:
        """Metrics captured during the run."""
        return dict(self._record.metrics)

    @property
    def tags(self) -> list[str]:
        """Tags associated with the run."""
        return list(self._record.tags)

    @property
    def duration_ms(self) -> int:
        """Run duration in milliseconds."""
        return self._record.duration_ms

    @property
    def started_at(self) -> datetime:
        """When the run started."""
        return self._record.started_at

    @property
    def finished_at(self) -> datetime:
        """When the run finished."""
        return self._record.finished_at

    @property
    def error(self) -> dict[str, Any] | None:
        """Error information if the run failed."""
        return self._record.error

    @property
    def context_fingerprint(self) -> str:
        """Fingerprint of the context used."""
        return self._record.context_fingerprint

    @property
    def params_fingerprint(self) -> str:
        """Fingerprint of the parameters used."""
        return self._record.params_fingerprint

    @property
    def seed_fingerprint(self) -> str:
        """Fingerprint of the seeds used."""
        return self._record.seed_fingerprint

    @property
    def record(self) -> RunRecord:
        """Access the underlying RunRecord."""
        return self._record

    def artifact(self, name: str) -> Any:
        """
        Load an artifact by name.

        Args:
            name: The artifact name.

        Returns:
            The deserialized artifact.

        Raises:
            FileNotFoundError: If the artifact doesn't exist.
        """
        # Get artifact descriptor
        artifacts = self._store.list_artifacts(self.run_id)
        descriptor = None
        for art in artifacts:
            if art.name == name:
                descriptor = art
                break

        if descriptor is None:
            raise FileNotFoundError(
                f"Artifact '{name}' not found in run {self.run_id}"
            )

        # Load and deserialize based on format
        from metalab.capture.registry import SerializerRegistry

        registry = SerializerRegistry()
        data_path = Path(descriptor.uri)

        # Find appropriate serializer by kind
        serializer = registry.get(descriptor.kind)
        if serializer is None:
            # Fall back to reading raw bytes
            return self._store.get_artifact(descriptor.uri)

        return serializer.load(data_path)

    def artifacts(self) -> list[ArtifactDescriptor]:
        """
        List available artifacts for this run.

        Returns:
            List of artifact descriptors.
        """
        return self._store.list_artifacts(self.run_id)

    def __repr__(self) -> str:
        return f"Run({self.run_id[:8]}..., status={self.status.value})"


class Results:
    """
    Collection of experiment runs with querying and access capabilities.

    Results wraps a Store and provides convenient access to:
    - Individual Run objects via indexing
    - Tabular view of results
    - Filtering by status, tags, or parameters

    Example:
        result = metalab.run(experiment)  # stores in ./runs/{name} by default

        # Access individual runs
        run = result[0]
        print(run.metrics)
        artifact = run.artifact("summary")

        # Get tabular view
        df = result.table(as_dataframe=True)

        # Filter results
        successful = result.successful
        filtered = result.filter(gene="KLF1")

        # Export
        result.to_csv("./output/results.csv")

        # Display summary
        result.display()
    """

    def __init__(
        self,
        store: Store,
        records: list[RunRecord],
    ) -> None:
        """
        Initialize the Results collection.

        Args:
            store: The store containing artifacts.
            records: List of RunRecords from the experiment.
        """
        self._store = store
        self._records = records

    @property
    def runs(self) -> list[Run]:
        """Get all runs as Run objects."""
        return [Run(record, self._store) for record in self._records]

    @property
    def records(self) -> list[RunRecord]:
        """Get all run records (raw dataclass form)."""
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

    def to_csv(
        self,
        path: str | Path,
        *,
        include_fingerprints: bool = False,
        timestamp: bool = False,
    ) -> Path:
        """
        Export results to a CSV file.

        Args:
            path: Output path. If a directory, generates a timestamped filename.
            include_fingerprints: Include fingerprint columns (default: False).
            timestamp: Add timestamp to filename if path is a file (default: False).

        Returns:
            Path to the written CSV file.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_csv(). "
                "Install it with: pip install metalab[pandas]"
            ) from e

        path = Path(path)

        # Generate filename if directory
        if path.is_dir() or not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = path / f"results_{ts}.csv"
        elif timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = path.with_stem(f"{path.stem}_{ts}")

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build DataFrame from rows
        rows = self.table(as_dataframe=False)
        df = pd.DataFrame(rows)

        # Optionally drop fingerprint columns
        if not include_fingerprints:
            fingerprint_cols = [
                c for c in df.columns if c.endswith("_fingerprint")
            ]
            df = df.drop(columns=fingerprint_cols, errors="ignore")

        df.to_csv(path, index=False)
        return path

    def load(self, run_id: str, artifact_name: str) -> Any:
        """
        Load an artifact from a run by run_id.

        Note: Prefer using run.artifact(name) for cleaner access:
            result[0].artifact("summary")

        Args:
            run_id: The run identifier.
            artifact_name: The name of the artifact.

        Returns:
            The deserialized artifact.

        Raises:
            FileNotFoundError: If the artifact doesn't exist.
        """
        # Find the run and delegate to Run.artifact()
        for record in self._records:
            if record.run_id == run_id:
                return Run(record, self._store).artifact(artifact_name)

        raise FileNotFoundError(f"Run '{run_id}' not found in results")

    def filter(
        self,
        status: str | Status | None = None,
        tags: list[str] | None = None,
        **params: Any,
    ) -> Results:
        """
        Filter results by criteria.

        Args:
            status: Filter by status ("success", "failed", "cancelled").
            tags: Filter by tags (all must be present).
            **params: Filter by metric values.

        Returns:
            A new Results with filtered runs.

        Example:
            # Filter by status
            successful = result.filter(status="success")

            # Filter by metric values
            filtered = result.filter(gene="KLF1", perturbation_value=100)

            # Chain filters
            runs = result.filter(status="success").filter(gene="KLF1")
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

        return Results(store=self._store, records=filtered)

    def __len__(self) -> int:
        """Return the number of runs."""
        return len(self._records)

    def __iter__(self) -> Iterator[Run]:
        """Iterate over runs."""
        for record in self._records:
            yield Run(record, self._store)

    @overload
    def __getitem__(self, index: int) -> Run: ...
    @overload
    def __getitem__(self, index: slice) -> Results: ...

    def __getitem__(self, index: int | slice) -> Run | Results:
        """
        Get a run by index or a slice of results.

        Args:
            index: Integer index or slice.

        Returns:
            Run for integer index, Results for slice.
        """
        if isinstance(index, slice):
            return Results(store=self._store, records=self._records[index])
        return Run(self._records[index], self._store)

    @property
    def successful(self) -> Results:
        """Get only successful runs."""
        return self.filter(status=Status.SUCCESS)

    @property
    def failed(self) -> Results:
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

    def display(
        self,
        *,
        group_by: list[str] | None = None,
        show_summary: bool = True,
    ) -> None:
        """
        Display results summary to console.

        Uses rich if available, falls back to plain text.

        Args:
            group_by: Optional metric keys to group results by.
            show_summary: Show overall summary statistics.

        Example:
            result.display()
            result.display(group_by=["gene", "perturbation_value"])
        """
        summary = self.summary()

        # Try to use rich for nicer output
        try:
            from rich.console import Console  # type: ignore[import-not-found]
            from rich.table import Table  # type: ignore[import-not-found]

            console = Console()

            # Rich output
            if show_summary:
                console.print("\n[bold]Results Summary[/bold]")
                console.print(f"  Total runs: {summary['total_runs']}")
                for status, count in summary["by_status"].items():
                    color = "green" if status == "success" else "red" if status == "failed" else "yellow"
                    console.print(f"  [{color}]{status}[/{color}]: {count}")
                console.print(f"  Avg duration: {summary['avg_duration_ms']:.1f}ms")

            if group_by and self._records:
                console.print(f"\n[bold]By {', '.join(group_by)}:[/bold]")
                groups: dict[tuple[Any, ...], list[RunRecord]] = {}
                for record in self._records:
                    key = tuple(record.metrics.get(k) for k in group_by)
                    groups.setdefault(key, []).append(record)

                table = Table()
                for col in group_by:
                    table.add_column(col)
                table.add_column("Success", justify="right")
                table.add_column("Failed", justify="right")
                table.add_column("Total", justify="right")

                for key, recs in sorted(groups.items()):
                    success = sum(1 for r in recs if r.status == Status.SUCCESS)
                    failed = sum(1 for r in recs if r.status == Status.FAILED)
                    row = [str(v) for v in key] + [str(success), str(failed), str(len(recs))]
                    table.add_row(*row)

                console.print(table)

        except ImportError:
            # Plain text output (fallback)
            if show_summary:
                print("\nResults Summary")
                print(f"  Total runs: {summary['total_runs']}")
                for status, count in summary["by_status"].items():
                    print(f"  {status}: {count}")
                print(f"  Avg duration: {summary['avg_duration_ms']:.1f}ms")

            if group_by and self._records:
                print(f"\nBy {', '.join(group_by)}:")
                groups_plain: dict[tuple[Any, ...], list[RunRecord]] = {}
                for record in self._records:
                    key = tuple(record.metrics.get(k) for k in group_by)
                    groups_plain.setdefault(key, []).append(record)

                for key, recs in sorted(groups_plain.items()):
                    success = sum(1 for r in recs if r.status == Status.SUCCESS)
                    failed = sum(1 for r in recs if r.status == Status.FAILED)
                    key_str = ", ".join(f"{k}={v}" for k, v in zip(group_by, key))
                    print(f"  {key_str}: {success}/{len(recs)} success")

    @classmethod
    def from_store(
        cls,
        store: Store,
        experiment_id: str | None = None,
    ) -> Results:
        """
        Load results from a store.

        Args:
            store: The store to load from.
            experiment_id: Optional filter by experiment ID.

        Returns:
            Results containing the loaded runs.

        Example:
            from metalab.store import FileStore

            store = FileStore("./runs/my_experiment")
            results = Results.from_store(store)
        """
        records = store.list_run_records(experiment_id=experiment_id)
        return cls(store=store, records=records)

    def __repr__(self) -> str:
        summary = self.summary()
        status_parts = [f"{k}={v}" for k, v in summary["by_status"].items()]
        return f"Results({summary['total_runs']} runs: {', '.join(status_parts)})"


# Backward compatibility alias
ResultHandle = Results
