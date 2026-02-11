"""
PostgresStore: FileStore with PostgreSQL query index.

Architecture:

- `FileStore`: Source of truth (logs, artifacts, run records)
- `PostgresIndex`: Query acceleration layer (indexed lookups)

All data is written to FileStore first (permanent), then indexed in Postgres.
If Postgres is lost, call rebuild_index() to restore from files.

This is the recommended way to use Postgres with metalab—it provides
fast queries while keeping files as the reliable, portable archive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar

if TYPE_CHECKING:
    from typing import BinaryIO

    from metalab.store.locator import LocatorInfo

from metalab.store.config import StoreConfig
from metalab.store.file import FileStore, FileStoreConfig

from metalab.store.postgres_index import PostgresIndex
from metalab.types import ArtifactDescriptor, Metric, RunRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class PostgresStoreConfig(StoreConfig):
    """
    Configuration for PostgresStore.

    Example:
    ```python
    config = PostgresStoreConfig(
        connection_string="postgresql://localhost/metalab",
        file_root="/data/experiments",
    )
    scoped = config.scoped("my_exp:1.0")
    store = scoped.connect()

    # Or let runner handle scoping:
    metalab.run(exp, store=config)  # auto-scopes to experiment
    ```
    """

    scheme: ClassVar[str] = "postgresql"
    connection_string: str
    file_root: str
    experiment_id: str | None = None
    schema: str = "public"
    auto_migrate: bool = True
    connect_timeout: float = 10.0
    pool_min_size: int = 1
    pool_max_size: int = 2

    def __post_init__(self) -> None:
        # Normalize file_root to absolute path
        resolved = str(Path(self.file_root).resolve())
        if self.file_root != resolved:
            object.__setattr__(self, "file_root", resolved)

    def connect(self) -> "PostgresStore":
        """Create a PostgresStore from this config.

        If the connection_string lacks credentials, attempts to discover them
        from service.json at {file_root}/services/postgres/service.json.
        """
        config = self._resolve_credentials()
        return PostgresStore(config)

    def _resolve_credentials(self) -> "PostgresStoreConfig":
        """Resolve credentials from service.json if needed.

        If the connection_string already has a password, returns self unchanged.
        Otherwise, looks for service.json and uses its connection_string.
        """
        from dataclasses import replace
        from urllib.parse import urlparse

        parsed = urlparse(self.connection_string)

        # If password is already present, use as-is
        if parsed.password:
            return self

        # Try to load credentials from service.json
        service_json_path = (
            Path(self.file_root) / "services" / "postgres" / "service.json"
        )

        if not service_json_path.exists():
            logger.debug(
                f"No service.json found at {service_json_path}, using connection as-is"
            )
            return self

        try:
            from metalab.services.postgres import PostgresService

            service = PostgresService.load(service_json_path)
            logger.debug("Resolved Postgres credentials from service.json")
            return replace(self, connection_string=service.connection_string)
        except Exception as e:
            logger.warning(
                f"Failed to load Postgres credentials from service.json: {e}"
            )
            return self

    @classmethod
    def from_locator(cls, info: "LocatorInfo", **kwargs: Any) -> "PostgresStoreConfig":
        """Parse postgresql:// locator into config."""
        file_root = kwargs.pop("file_root", None) or info.params.get("file_root")
        if not file_root:
            raise ValueError("PostgresStoreConfig requires file_root")
        experiment_id = kwargs.pop("experiment_id", None) or info.params.get(
            "experiment_id"
        )
        schema = info.params.get("schema", "public")
        # Strip query params from connection string for clean storage
        conn_str = info.raw.split("?")[0] if "?" in info.raw else info.raw
        return cls(
            connection_string=conn_str,
            file_root=file_root,
            experiment_id=experiment_id,
            schema=schema,
        )

    def list_experiments(self) -> list[str]:
        """
        List all experiment IDs in this collection.

        Discovers experiments by scanning subdirectories of file_root for
        _meta.json files that contain experiment_id. Only works on unscoped configs.

        Returns:
            List of experiment IDs found in this collection.

        Raises:
            ValueError: If called on a scoped config.
        """
        if self.experiment_id is not None:
            raise ValueError("Cannot list experiments on a scoped config")

        root = Path(self.file_root)
        if not root.exists():
            return []

        experiments = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            meta_path = child / "_meta.json"
            if meta_path.exists():
                try:
                    import json

                    with open(meta_path) as f:
                        meta = json.load(f)
                    if "experiment_id" in meta:
                        experiments.append(meta["experiment_id"])
                except (json.JSONDecodeError, OSError):
                    pass
        return sorted(experiments)

    def for_experiment(self, experiment_id: str) -> "PostgresStoreConfig":
        """
        Get a scoped config for a specific experiment.

        Alias for scoped() with a clearer name for loading/browsing context.

        Args:
            experiment_id: The experiment ID to scope to.

        Returns:
            A new PostgresStoreConfig scoped to the experiment.
        """
        return self.scoped(experiment_id)



class PostgresStore:
    """
    FileStore with PostgreSQL query index.

    Writes to both FileStore (source of truth) and PostgresIndex (fast queries).
    File operations (logs, artifacts) delegate directly to FileStore.
    Record operations write to files first, then index in Postgres.

    If Postgres is unavailable or lost, rebuild_index() restores it from files.

    Create via PostgresStoreConfig:
        config = PostgresStoreConfig(
            connection_string="postgresql://localhost/metalab",
            file_root="/data/experiments",
        )
        store = config.connect()
    """

    def __init__(self, config: PostgresStoreConfig) -> None:
        """
        Initialize from config.

        Use PostgresStoreConfig(...).connect() to create instances.

        Args:
            config: The PostgresStoreConfig for this store.
        """
        self._config = config

        # Create FileStore (source of truth) via its config
        # Pass experiment_id through so FileStore computes the effective root
        # and writes experiment_id into _meta.json for collection discovery.
        file_config = FileStoreConfig(
            root=config.file_root,
            experiment_id=config.experiment_id,
        )
        self._files = file_config.connect()

        # Create PostgresIndex (query acceleration)
        self._index = PostgresIndex(
            config.connection_string,
            schema=config.schema,
            auto_migrate=config.auto_migrate,
            pool_min_size=config.pool_min_size,
            pool_max_size=config.pool_max_size,
            connect_timeout=config.connect_timeout,
        )

        logger.info(
            f"PostgresStore initialized: index at {self._sanitize_connection_string(config.connection_string)}, "
            f"files at {self._files.root}"
        )

    @property
    def config(self) -> PostgresStoreConfig:
        """The configuration for this store."""
        return self._config

    @property
    def is_scoped(self) -> bool:
        """True if this store is scoped to a specific experiment."""
        return self._config.is_scoped

    def scoped(self, experiment_id: str) -> "PostgresStore":
        """
        Return a store scoped to the given experiment.

        Args:
            experiment_id: The experiment identifier.

        Returns:
            A new PostgresStore scoped to the experiment.
        """
        return self._config.scoped(experiment_id).connect()

    def __repr__(self) -> str:
        """Return a string representation of the store."""
        safe_conn = self._sanitize_connection_string(self._config.connection_string)
        return f"PostgresStore(index={safe_conn!r}, files={self._config.file_root!r})"

    @staticmethod
    def _sanitize_connection_string(conn: str) -> str:
        """Hide password in connection string for safe display."""
        from urllib.parse import urlparse, urlunparse

        try:
            parsed = urlparse(conn)
            if parsed.password:
                # Replace password with asterisks
                netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
                return urlunparse(parsed._replace(netloc=netloc))
        except Exception:
            pass
        return conn

    @property
    def file_store(self) -> FileStore:
        """Access the underlying FileStore."""
        return self._files

    @property
    def index(self) -> PostgresIndex:
        """Access the underlying PostgresIndex."""
        return self._index

    # =========================================================================
    # Capability: SupportsWorkingDirectory
    # =========================================================================

    def get_working_directory(self) -> Path:
        """Return the filesystem directory for shared run coordination."""
        return self._files.get_working_directory()

    # =========================================================================
    # Run record operations (write to files, then index)
    # =========================================================================

    def put_run_record(self, record: RunRecord) -> None:
        """
        Persist a run record.

        Writes to FileStore first (source of truth), then indexes in Postgres.
        """
        # Write to files (source of truth)
        self._files.put_run_record(record)

        # Index in Postgres
        try:
            self._index.index_record(record)
        except Exception as e:
            logger.warning(f"Failed to index record in Postgres: {e}")

    def get_run_record(self, run_id: str) -> RunRecord | None:
        """
        Retrieve a run record by ID.

        Tries Postgres index first (fast), falls back to files.
        """
        try:
            result = self._index.get_record(run_id)
            if result is not None:
                return result
        except Exception:
            pass

        return self._files.get_run_record(run_id)

    def list_run_records(self, experiment_id: str | None = None) -> list[RunRecord]:
        """
        List run records, optionally filtered by experiment.

        Uses Postgres index for fast queries, falls back to files.
        """
        try:
            return self._index.list_records(experiment_id)
        except Exception:
            return self._files.list_run_records(experiment_id)

    def run_exists(self, run_id: str) -> bool:
        """Check if a run record exists."""
        try:
            if self._index.record_exists(run_id):
                return True
        except Exception:
            pass

        return self._files.run_exists(run_id)

    # =========================================================================
    # Artifact operations (delegate to FileStore)
    # =========================================================================

    def put_artifact(
        self,
        data: bytes | Path,
        descriptor: ArtifactDescriptor,
    ) -> ArtifactDescriptor:
        """Store an artifact (delegated to FileStore)."""
        return self._files.put_artifact(data, descriptor)

    def get_artifact(self, uri: str) -> bytes:
        """Retrieve artifact data."""
        return self._files.get_artifact(uri)

    def open_artifact(self, uri: str) -> "BinaryIO":
        """Open an artifact for reading."""
        return self._files.open_artifact(uri)

    def list_artifacts(self, run_id: str) -> list[ArtifactDescriptor]:
        """List artifacts for a run."""
        return self._files.list_artifacts(run_id)

    # =========================================================================
    # Derived metrics operations (write to files, then index)
    # =========================================================================

    def put_derived(self, run_id: str, derived: dict[str, Metric]) -> None:
        """Persist derived metrics for a run."""
        # Write to files (source of truth)
        self._files.put_derived(run_id, derived)

        # Index in Postgres
        try:
            self._index.index_derived(run_id, derived)
        except Exception as e:
            logger.warning(f"Failed to index derived in Postgres: {e}")

    def get_derived(self, run_id: str) -> dict[str, Metric] | None:
        """Retrieve derived metrics for a run."""
        try:
            result = self._index.get_derived(run_id)
            if result is not None:
                return result
        except Exception:
            pass

        return self._files.get_derived(run_id)

    def derived_exists(self, run_id: str) -> bool:
        """Check if derived metrics exist for a run."""
        try:
            if self._index.derived_exists(run_id):
                return True
        except Exception:
            pass

        return self._files.derived_exists(run_id)

    # =========================================================================
    # Structured results operations (delegate to FileStore)
    # =========================================================================

    def put_result(
        self,
        run_id: str,
        name: str,
        data: Any,
        dtype: str | None = None,
        shape: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store structured result data for a run."""
        self._files.put_result(run_id, name, data, dtype, shape, metadata)

    def get_result(self, run_id: str, name: str) -> dict[str, Any] | None:
        """Retrieve structured result data."""
        return self._files.get_result(run_id, name)

    def list_results(self, run_id: str) -> list[str]:
        """List result names for a run."""
        return self._files.list_results(run_id)

    # =========================================================================
    # Log operations (delegate to FileStore)
    # =========================================================================

    def put_log(self, run_id: str, name: str, content: str) -> None:
        """Store a log file for a run."""
        self._files.put_log(run_id, name, content)

    def get_log_path(self, run_id: str, name: str) -> Path:
        """Get the path where a log file should be written."""
        return self._files.get_log_path(run_id, name)

    def get_log(self, run_id: str, name: str) -> str | None:
        """Retrieve a log file."""
        return self._files.get_log(run_id, name)

    def list_logs(self, run_id: str) -> list[str]:
        """List available log names for a run."""
        return self._files.list_logs(run_id)

    # =========================================================================
    # Experiment manifest operations (write to files, then index)
    # =========================================================================

    def put_experiment_manifest(
        self,
        experiment_id: str,
        manifest: dict[str, Any],
        timestamp: str | None = None,
    ) -> None:
        """Store an experiment manifest."""
        # Write to files (source of truth)
        self._files.put_experiment_manifest(experiment_id, manifest, timestamp)

        # Index in Postgres
        try:
            self._index.index_manifest(experiment_id, manifest, timestamp)
        except Exception as e:
            logger.warning(f"Failed to index manifest in Postgres: {e}")

    def get_experiment_manifest(self, experiment_id: str) -> dict[str, Any] | None:
        """Retrieve the most recent experiment manifest."""
        try:
            result = self._index.get_manifest(experiment_id)
            if result is not None:
                return result
        except Exception:
            pass

        return self._files.get_experiment_manifest(experiment_id)

    # =========================================================================
    # Index management
    # =========================================================================

    def rebuild_index(
        self,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> int:
        """
        Rebuild the Postgres index from FileStore.

        Uses batch operations for efficiency — a single transaction for
        all records instead of per-record commits.

        When the store is unscoped (no experiment_id), discovers all experiment
        subdirectories via _meta.json and indexes records from each one.

        Call this if:
        - Postgres was wiped/reset
        - Connecting to a fresh database
        - Index got out of sync

        Args:
            progress: Optional callback ``(phase, completed, total)`` invoked
                as work proceeds. *phase* is a short label (e.g.
                ``"Scanning experiments"``, ``"Indexing derived"``).

        Returns:
            Number of records indexed.
        """
        logger.info("Rebuilding Postgres index from files...")

        def _tick(phase: str, completed: int, total: int) -> None:
            if progress is not None:
                progress(phase, completed, total)

        # ------------------------------------------------------------------
        # Phase 1: discover and load run records from files
        # ------------------------------------------------------------------
        all_records: list[RunRecord] = []
        record_stores: list[tuple[RunRecord, FileStore]] = []

        if self._config.experiment_id:
            _tick("Scanning experiments", 0, 1)
            records = self._files.list_run_records()
            all_records.extend(records)
            record_stores.extend((r, self._files) for r in records)
            _tick("Scanning experiments", 1, 1)
        else:
            base_config = FileStoreConfig(root=self._config.file_root)
            experiment_ids = base_config.list_experiments()
            n_exps = len(experiment_ids)
            logger.info(
                f"Discovered {n_exps} experiments in {self._config.file_root}"
            )

            for i, exp_id in enumerate(experiment_ids):
                _tick("Scanning experiments", i, n_exps)
                scoped_store = base_config.for_experiment(exp_id).connect()
                records = scoped_store.list_run_records()
                all_records.extend(records)
                record_stores.extend((r, scoped_store) for r in records)
            _tick("Scanning experiments", n_exps, n_exps)

        # ------------------------------------------------------------------
        # Phase 2: clear and write index
        # ------------------------------------------------------------------
        _tick("Writing index", 0, 3)
        self._index.clear()

        self._index.batch_index_records(all_records)
        _tick("Writing index", 1, 3)

        # Collect and batch index derived metrics
        derived_pairs = []
        n_records = len(record_stores)
        for i, (record, store) in enumerate(record_stores):
            if i % 50 == 0:
                _tick("Collecting derived metrics", i, n_records)
            derived = store.get_derived(record.run_id)
            if derived:
                derived_pairs.append((record.run_id, derived))
        _tick("Collecting derived metrics", n_records, n_records)

        self._index.batch_index_derived(derived_pairs)
        _tick("Writing index", 2, 3)

        self._index.update_field_catalog(all_records)
        _tick("Writing index", 3, 3)

        logger.info(f"Indexed {len(all_records)} records from files")
        return len(all_records)

    @classmethod
    def from_filestore(
        cls,
        connection_string: str,
        filestore: FileStore,
        *,
        schema: str = "public",
        rebuild: bool = True,
    ) -> "PostgresStore":
        """
        Create PostgresStore from an existing FileStore.

        Args:
            connection_string: PostgreSQL connection URL.
            filestore: Existing FileStore to wrap.
            schema: Database schema name.
            rebuild: If True, rebuild index from files.

        Returns:
            A PostgresStore wrapping the FileStore.
        """
        config = PostgresStoreConfig(
            connection_string=connection_string,
            file_root=str(filestore.root),
            schema=schema,
        )
        store = config.connect()
        if rebuild:
            store.rebuild_index()
        return store

    def to_filestore(self, destination: Path | None = None) -> FileStore:
        """
        Export to a standalone FileStore.

        Since FileStore is the source of truth, this essentially returns
        the internal FileStore (or copies to a new location).

        Args:
            destination: If provided, copies to this location.
                        If None, returns the internal FileStore.

        Returns:
            A FileStore with the exported data.
        """
        if destination is None:
            return self._files

        import shutil

        shutil.copytree(self._files.root, destination)
        return FileStoreConfig(root=str(destination)).connect()

    # =========================================================================
    # Utility methods
    # =========================================================================

    def delete_run(self, run_id: str) -> None:
        """Delete a run and all associated data."""
        # Delete from files (source of truth)
        self._files.delete_run(run_id)

        # Delete from index
        try:
            self._index.delete_record(run_id)
            self._index.delete_derived(run_id)
        except Exception as e:
            logger.warning(f"Failed to delete from Postgres index: {e}")

    def close(self) -> None:
        """Close the connection pool."""
        self._index.close()

    def __enter__(self) -> "PostgresStore":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager."""
        self.close()
