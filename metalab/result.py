"""
Results: Query interface for experiment results.

Provides:

- Run class for single run access (metrics, artifacts)
- Results class for collections of runs
- ExperimentInfo for experiment-level metadata
- Tabular view of results
- Artifact loading
- Filtering capabilities
"""

from __future__ import annotations

import inspect
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, overload

if TYPE_CHECKING:
    from metalab.store.base import Store

from metalab.store.capabilities import SupportsArtifactOpen, SupportsStructuredResults
from metalab.types import ArtifactDescriptor, RunRecord, Status


@dataclass
class ExperimentInfo:
    """
    Experiment-level information accessible from a Run.

    This provides access to experiment metadata without needing
    to load the full experiment manifest repeatedly.

    Attributes:
        experiment_id: The experiment identifier (name:version).
        name: The experiment name.
        version: The experiment version.
        description: Human-readable description.
        metadata: User-defined metadata dict.
        tags: List of tags for categorization.
    """

    experiment_id: str
    name: str = ""
    version: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> ExperimentInfo:
        """
        Create ExperimentInfo from an experiment manifest dict.

        Args:
            manifest: The experiment manifest dictionary.

        Returns:
            ExperimentInfo populated from the manifest.
        """
        return cls(
            experiment_id=manifest.get("experiment_id", ""),
            name=manifest.get("name", ""),
            version=manifest.get("version", ""),
            description=manifest.get("description", ""),
            metadata=manifest.get("metadata", {}),
            tags=manifest.get("tags", []),
        )

    @classmethod
    def from_experiment_id(cls, experiment_id: str) -> ExperimentInfo:
        """
        Create minimal ExperimentInfo from just an experiment_id.

        Used as a fallback when the manifest is not available.

        Args:
            experiment_id: The experiment identifier (name:version).

        Returns:
            ExperimentInfo with minimal data extracted from the ID.
        """
        if ":" in experiment_id:
            name, version = experiment_id.split(":", 1)
        else:
            name, version = experiment_id, ""

        return cls(
            experiment_id=experiment_id,
            name=name,
            version=version,
        )


# Type aliases for artifact reducers
ArtifactReducer = Callable[[Any], dict[str, Any]]
ContextAwareReducer = Callable[[Any, "Run"], dict[str, Any]]


class Run:
    """
    A single experiment run with access to its metrics and artifacts.

    The Run object wraps a RunRecord and provides convenient access to:
    - Run metadata (run_id, status, timestamps)
    - Metrics captured during the run
    - Artifacts stored for the run
    - Experiment-level metadata via the `experiment` property

    Example:
    ```python
    result = metalab.run(experiment)
    run = result[0]  # Get first run

    # Access metrics
    print(run.metrics)
    print(run.status)

    # Access experiment metadata
    print(run.experiment.metadata)

    # Load artifacts
    summary = run.artifact("summary")
    for desc in run.artifacts():
        print(f"  {desc.name}: {desc.kind}")
    ```
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
        self._experiment_info: ExperimentInfo | None = None

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
    def params(self) -> dict[str, Any]:
        """Resolved parameters for this run."""
        return dict(self._record.params_resolved)

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

    @property
    def experiment(self) -> ExperimentInfo:
        """
        Experiment-level information including metadata.

        Lazily loads from the experiment manifest on first access.
        If the manifest is not found, returns minimal info extracted
        from the experiment_id.

        Returns:
            ExperimentInfo with experiment metadata.

        Example:
        ```python
        # Access user-defined metadata
        group_labels = run.experiment.metadata.get("group_labels")
        markov_iter = run.experiment.metadata.get("markov_iter", 3)
        ```
        """
        if self._experiment_info is None:
            manifest = self._store.get_experiment_manifest(self.experiment_id)
            if manifest:
                self._experiment_info = ExperimentInfo.from_manifest(manifest)
            else:
                # Fallback with minimal info if manifest not found
                self._experiment_info = ExperimentInfo.from_experiment_id(
                    self.experiment_id
                )
        return self._experiment_info

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
            raise FileNotFoundError(f"Artifact '{name}' not found in run {self.run_id}")

        # Load and deserialize based on format
        from metalab.capture.registry import SerializerRegistry

        registry = SerializerRegistry()

        # Find appropriate serializer by kind
        serializer = registry.get(descriptor.kind)
        if serializer is None:
            # Fall back to reading raw bytes
            return self._store.get_artifact(descriptor.uri)

        uri = descriptor.uri

        # Check if URI is a simple filesystem path (no scheme or file://)
        # These can be loaded directly by the serializer
        is_filesystem_path = "://" not in uri or uri.startswith("file://")
        if is_filesystem_path:
            # Strip file:// prefix if present
            path_str = uri.replace("file://", "") if uri.startswith("file://") else uri
            return serializer.load(Path(path_str))

        # Non-filesystem URI (e.g., pgblob://) - use store capability to open
        if isinstance(self._store, SupportsArtifactOpen):
            with self._store.open_artifact(uri) as f:
                ext = f".{descriptor.format}" if descriptor.format else ""
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(f.read())
                    tmp_path = Path(tmp.name)
                try:
                    return serializer.load(tmp_path)
                finally:
                    tmp_path.unlink(missing_ok=True)

        # Fall back to get_artifact + temp file for stores without open_artifact
        data = self._store.get_artifact(uri)
        ext = f".{descriptor.format}" if descriptor.format else ""
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            return serializer.load(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def artifacts(self) -> list[ArtifactDescriptor]:
        """
        List available artifacts for this run.

        Returns:
            List of artifact descriptors.
        """
        return self._store.list_artifacts(self.run_id)

    def data(self, name: str) -> Any:
        """
        Load structured result data by name.

        Structured data is stored via capture.data() as packed outputs.

        Args:
            name: The data name.

        Returns:
            The data object. Arrays are returned as numpy arrays if
            shape/dtype metadata is available.

        Raises:
            KeyError: If the data doesn't exist.

        Example:
        ```python
        matrix = run.data("transition_matrix")
        ```
        """
        if not isinstance(self._store, SupportsStructuredResults):
            raise NotImplementedError(
                f"Store {type(self._store).__name__} does not support structured results."
            )

        result = self._store.get_result(self.run_id, name)
        if result is None:
            raise KeyError(f"Result '{name}' not found in run {self.run_id}")

        data = result["data"]

        # Reconstruct numpy array if metadata available
        if result.get("shape") and result.get("dtype"):
            try:
                import numpy as np

                return np.array(data, dtype=result["dtype"]).reshape(result["shape"])
            except ImportError:
                pass  # numpy not available, return raw data

        return data

    def list_data(self) -> list[str]:
        """
        List available structured data names for this run.

        Returns:
            List of data names.
        """
        if not isinstance(self._store, SupportsStructuredResults):
            return []
        return self._store.list_results(self.run_id)

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
    ```python
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
    ```
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
    def store(self) -> "Store":
        """The store containing the run data."""
        return self._store

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

    def to_dataframe(
        self,
        *,
        include_params: bool = True,
        include_metrics: bool = True,
        include_record: bool = True,
        artifact_reducers: (
            dict[str, ArtifactReducer | ContextAwareReducer] | None
        ) = None,
    ) -> Any:
        """
        Export results to a pandas DataFrame with optional artifact reduction.

        This method provides flexible DataFrame export with:
        - Resolved parameters (prefixed with 'param_')
        - Captured metrics
        - Record metadata (run_id, status, duration, etc.)
        - Optional on-the-fly artifact reducers

        Args:
            include_params: Include params_resolved columns (prefixed with 'param_').
            include_metrics: Include metrics columns.
            include_record: Include record fields (run_id, status, duration, etc.).
            artifact_reducers: Dict mapping artifact name to reducer function.

        Returns:
            pandas DataFrame with the requested columns.

        Raises:
            ImportError: If pandas is not installed.

        Example (artifact reducer):
            def reduce_history(arr):
                return {"final": arr[:, -1].mean(), "best": arr.min()}

            df = results.to_dataframe(
                artifact_reducers={"history": reduce_history}
            )
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install metalab[pandas]"
            ) from e

        rows = []

        for run in self.runs:
            row: dict[str, Any] = {}

            # Include record fields
            if include_record:
                row["run_id"] = run.run_id
                row["experiment_id"] = run.experiment_id
                row["status"] = run.status.value
                row["duration_ms"] = run.duration_ms
                row["started_at"] = run.started_at.isoformat()
                row["finished_at"] = run.finished_at.isoformat()

            # Include params (prefixed to avoid collisions)
            if include_params:
                for key, value in run.params.items():
                    row[f"param_{key}"] = value

            # Include metrics
            if include_metrics:
                for key, value in run.metrics.items():
                    row[key] = value

            # Apply artifact reducers
            if artifact_reducers:
                for artifact_name, reducer in artifact_reducers.items():
                    try:
                        artifact = run.artifact(artifact_name)

                        # Detect if reducer needs run context (2 args vs 1 arg)
                        sig = inspect.signature(reducer)
                        num_params = len(
                            [
                                p
                                for p in sig.parameters.values()
                                if p.default is inspect.Parameter.empty
                            ]
                        )

                        if num_params >= 2:
                            # Context-aware reducer
                            reduced = reducer(artifact, run)
                        else:
                            # Simple reducer
                            reduced = reducer(artifact)

                        row.update(reduced)
                    except FileNotFoundError:
                        # Artifact missing - leave columns as NaN
                        pass
                    except Exception as e:
                        # Log warning but continue
                        import warnings

                        warnings.warn(
                            f"Reducer for '{artifact_name}' failed on run "
                            f"{run.run_id[:8]}: {e}"
                        )

            rows.append(row)

        return pd.DataFrame(rows)

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
            fingerprint_cols = [c for c in df.columns if c.endswith("_fingerprint")]
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
                    color = (
                        "green"
                        if status == "success"
                        else "red" if status == "failed" else "yellow"
                    )
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
                    row = [str(v) for v in key] + [
                        str(success),
                        str(failed),
                        str(len(recs)),
                    ]
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


class IndexedResults:
    """DuckDB-backed Results facade that keeps run records lazy."""

    def __init__(
        self,
        *,
        store: Store,
        db_path: Path,
        experiment_id: str | None = None,
        status: Status | None = None,
        field_filters: list[tuple[str, str, Any]] | None = None,
    ) -> None:
        self._store = store
        self._db_path = db_path
        self._experiment_id = experiment_id
        self._status = status
        self._field_filters = field_filters or []

    @classmethod
    def from_store(
        cls,
        store: Store,
        experiment_id: str | None = None,
        *,
        refresh: bool = False,
        rebuild: bool = True,
    ) -> IndexedResults:
        """Open or build the DuckDB sidecar index for a file store."""
        from metalab.index import _duckdb, index_is_current, rebuild_index
        from metalab.store.layout import FileStoreLayout

        root = getattr(store, "root", None)
        if root is None:
            raise RuntimeError("Indexed results require a filesystem store")

        root_path = Path(root)
        db_path = FileStoreLayout(root_path).duckdb_path()
        current = index_is_current(root_path)
        if refresh or (rebuild and not current):
            db_path = rebuild_index(root_path, force=True)
        elif not current:
            raise RuntimeError(
                "DuckDB sidecar index is missing or stale. Run `metalab index rebuild "
                "PATH`, call `metalab.load_results(PATH, refresh_index=True)`, or pass "
                "`indexed=False` to eagerly scan canonical run records."
            )
        else:
            _duckdb()
        return cls(store=store, db_path=db_path, experiment_id=experiment_id)

    @property
    def store(self) -> Store:
        """The store containing the run data and sidecars."""
        return self._store

    @property
    def records(self) -> list[RunRecord]:
        """Materialize all run records. Prefer iteration or indexed summaries at scale."""
        return list(self._iter_records())

    @property
    def runs(self) -> list[Run]:
        """Materialize all runs. Prefer iteration at scale."""
        return [Run(record, self._store) for record in self._iter_records()]

    def _connect(self) -> Any:
        from metalab.index import _duckdb

        return _duckdb().connect(str(self._db_path), read_only=True)

    def _where_sql(self) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if self._experiment_id is not None:
            clauses.append("r.experiment_id = ?")
            params.append(self._experiment_id)
        if self._status is not None:
            clauses.append("r.status = ?")
            params.append(self._status.value)
        for index, (namespace, name, value) in enumerate(self._field_filters):
            alias = f"f{index}"
            clauses.append(
                f"EXISTS (SELECT 1 FROM fields {alias} "
                f"WHERE {alias}.run_id = r.run_id "
                f"AND {alias}.namespace = ? "
                f"AND {alias}.field_name = ? "
                f"AND {alias}.field_value = ?)"
            )
            params.extend([namespace, name, str(value)])
        if not clauses:
            return "", params
        return "WHERE " + " AND ".join(clauses), params

    def _iter_records(self) -> Iterator[RunRecord]:
        where_sql, params = self._where_sql()
        conn = self._connect()
        try:
            cursor = conn.execute(
                f"SELECT run_id FROM runs r {where_sql} ORDER BY started_at",
                params,
            )
            while rows := cursor.fetchmany(1000):
                for (run_id,) in rows:
                    record = self._store.get_run_record(run_id)
                    if record is not None:
                        yield record
        finally:
            conn.close()

    def _count(self) -> int:
        where_sql, params = self._where_sql()
        conn = self._connect()
        try:
            return int(
                conn.execute(f"SELECT COUNT(*) FROM runs r {where_sql}", params).fetchone()[0]
            )
        finally:
            conn.close()

    def __len__(self) -> int:
        return self._count()

    def __iter__(self) -> Iterator[Run]:
        for record in self._iter_records():
            yield Run(record, self._store)

    @overload
    def __getitem__(self, index: int) -> Run: ...
    @overload
    def __getitem__(self, index: slice) -> Results: ...

    def __getitem__(self, index: int | slice) -> Run | Results:
        if isinstance(index, slice):
            start = index.start or 0
            stop = index.stop if index.stop is not None else self._count()
            step = index.step or 1
            return Results(
                store=self._store,
                records=[
                    record
                    for row_index, record in enumerate(self._iter_records())
                    if start <= row_index < stop and (row_index - start) % step == 0
                ],
            )
        if index < 0:
            index = self._count() + index
        where_sql, params = self._where_sql()
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT run_id FROM runs r {where_sql} ORDER BY started_at LIMIT 1 OFFSET ?",
                [*params, index],
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise IndexError(index)
        record = self._store.get_run_record(row[0])
        if record is None:
            raise IndexError(index)
        return Run(record, self._store)

    @property
    def successful(self) -> IndexedResults:
        return self.filter(status=Status.SUCCESS)

    @property
    def failed(self) -> IndexedResults:
        return self.filter(status=Status.FAILED)

    def filter(
        self,
        status: str | Status | None = None,
        tags: list[str] | None = None,
        **params: Any,
    ) -> IndexedResults | Results:
        if tags is not None:
            # Tags are not in the current DuckDB sidecar schema; materialize for this uncommon path.
            return Results(store=self._store, records=self.records).filter(
                status=status,
                tags=tags,
                **params,
            )  # type: ignore[return-value]
        next_status = self._status
        if status is not None:
            next_status = Status(status) if isinstance(status, str) else status
        filters = list(self._field_filters)
        for key, value in params.items():
            namespace = "params" if key.startswith("param_") else "metrics"
            field_name = key.removeprefix("param_")
            filters.append((namespace, field_name, value))
        return IndexedResults(
            store=self._store,
            db_path=self._db_path,
            experiment_id=self._experiment_id,
            status=next_status,
            field_filters=filters,
        )

    def summary(self) -> dict[str, Any]:
        where_sql, params = self._where_sql()
        conn = self._connect()
        try:
            status_rows = conn.execute(
                f"SELECT status, COUNT(*) FROM runs r {where_sql} GROUP BY status",
                params,
            ).fetchall()
            duration_row = conn.execute(
                f"""
                SELECT AVG(duration_ms), MIN(duration_ms), MAX(duration_ms)
                FROM runs r {where_sql}
                """,
                params,
            ).fetchone()
        finally:
            conn.close()
        total = sum(count for _, count in status_rows)
        avg_duration, min_duration, max_duration = duration_row
        return {
            "total_runs": total,
            "by_status": {status: count for status, count in status_rows},
            "avg_duration_ms": avg_duration or 0,
            "min_duration_ms": min_duration or 0,
            "max_duration_ms": max_duration or 0,
        }

    @staticmethod
    def _decode_json(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return json.loads(value)
        return {}

    def table(self, as_dataframe: bool = False) -> list[dict[str, Any]] | Any:
        where_sql, params = self._where_sql()
        conn = self._connect()
        try:
            rows = conn.execute(
                f"""
                SELECT run_id, experiment_id, status, duration_ms, started_at, finished_at,
                       params_json, metrics_json
                FROM runs r {where_sql}
                ORDER BY started_at
                """,
                params,
            ).fetchall()
        finally:
            conn.close()
        table_rows = []
        for row in rows:
            params_json = self._decode_json(row[6])
            metrics_json = self._decode_json(row[7])
            table_rows.append(
                {
                    "run_id": row[0],
                    "experiment_id": row[1],
                    "status": row[2],
                    "duration_ms": row[3],
                    "started_at": str(row[4]),
                    "finished_at": str(row[5]),
                    **{f"param_{key}": value for key, value in params_json.items()},
                    **metrics_json,
                }
            )
        if not as_dataframe:
            return table_rows
        try:
            import pandas as pd

            return pd.DataFrame(table_rows)
        except ImportError as e:
            raise ImportError(
                "pandas is required for as_dataframe=True. "
                "Install it with: pip install metalab[pandas]"
            ) from e

    def to_dataframe(
        self,
        *,
        include_params: bool = True,
        include_metrics: bool = True,
        include_record: bool = True,
        artifact_reducers: (
            dict[str, ArtifactReducer | ContextAwareReducer] | None
        ) = None,
    ) -> Any:
        if artifact_reducers:
            return Results(store=self._store, records=self.records).to_dataframe(
                include_params=include_params,
                include_metrics=include_metrics,
                include_record=include_record,
                artifact_reducers=artifact_reducers,
            )

        df = self.table(as_dataframe=True)
        if not include_params:
            df = df.drop(columns=[c for c in df.columns if c.startswith("param_")])
        if not include_metrics:
            metric_cols = [
                c
                for c in df.columns
                if c
                not in {
                    "run_id",
                    "experiment_id",
                    "status",
                    "duration_ms",
                    "started_at",
                    "finished_at",
                }
                and not c.startswith("param_")
            ]
            df = df.drop(columns=metric_cols)
        if not include_record:
            df = df.drop(
                columns=[
                    "run_id",
                    "experiment_id",
                    "status",
                    "duration_ms",
                    "started_at",
                    "finished_at",
                ],
                errors="ignore",
            )
        return df

    def to_csv(
        self,
        path: str | Path,
        *,
        include_fingerprints: bool = False,
        timestamp: bool = False,
    ) -> Path:
        # Keep the public behavior aligned with Results.to_csv by exporting the
        # flattened table. For very large exports, prefer the CLI parquet export.
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_csv(). "
                "Install it with: pip install metalab[pandas]"
            ) from e

        path = Path(path)
        if path.is_dir() or not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = path / f"results_{ts}.csv"
        elif timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = path.with_stem(f"{path.stem}_{ts}")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.table(as_dataframe=False)).to_csv(path, index=False)
        return path

    def load(self, run_id: str, artifact_name: str) -> Any:
        record = self._store.get_run_record(run_id)
        if record is None:
            raise FileNotFoundError(f"Run '{run_id}' not found in results")
        return Run(record, self._store).artifact(artifact_name)

    def display(
        self,
        *,
        group_by: list[str] | None = None,
        show_summary: bool = True,
    ) -> None:
        Results(store=self._store, records=self.records).display(
            group_by=group_by,
            show_summary=show_summary,
        )

    def __repr__(self) -> str:
        summary = self.summary()
        status_parts = [f"{k}={v}" for k, v in summary["by_status"].items()]
        return f"IndexedResults({summary['total_runs']} runs: {', '.join(status_parts)})"
