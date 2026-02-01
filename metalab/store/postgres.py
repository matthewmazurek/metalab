"""
PostgresStore: PostgreSQL-based storage backend with FileStore composition.

PostgresStore wraps FileStore for file operations (logs, artifacts) while
PostgreSQL handles indexed run records, structured results, and Atlas features.

Architecture:
    - FileStore: Logs (streaming), artifacts, working directory
    - PostgreSQL: Run records (indexed), results (structured data), derived metrics,
                  field catalog, experiment manifests

Schema:
    runs: Primary run record storage with JSONB for full record
    results: Structured result data (arrays, matrices, dicts) for derived metrics
    derived: Per-run derived metrics (JSONB)
    logs: Log content (legacy, for backward compatibility)
    artifacts: Artifact metadata (legacy, for backward compatibility)
    artifact_blobs: Optional inline blob storage (legacy)
    experiment_manifests: Versioned experiment manifests
    field_catalog: Pre-computed field index for Atlas
    meta: Schema version and migrations

Connection:
    postgresql://user:password@host:port/database?schema=public&file_root=/path

Concurrency:
    Uses upsert (INSERT ON CONFLICT) for idempotent writes.
    Connection pooling via psycopg_pool.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

if TYPE_CHECKING:
    from typing import BinaryIO

from metalab.schema import dump_run_record, load_run_record


def _coerce_naive_local_to_utc(dt: datetime | None) -> datetime | None:
    """
    Coerce a datetime to tz-aware UTC for TIMESTAMPTZ columns.

    metalab historically uses naive local datetimes. If we pass those directly into
    TIMESTAMPTZ columns, Postgres interprets them in the session timezone (often UTC),
    shifting them by the local offset (e.g. ~7 hours).
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        assert local_tz is not None
        dt = dt.replace(tzinfo=local_tz)
    return dt.astimezone(timezone.utc)
from metalab.types import ArtifactDescriptor, Metric, Provenance, RunRecord, Status

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 2  # Bumped for results table

# Default artifact blob size threshold (legacy, not used with FileStore composition)
DEFAULT_BLOB_THRESHOLD = 1024 * 1024  # 1MB


def _parse_connection_string(conn_str: str) -> dict[str, Any]:
    """Parse PostgreSQL connection string.

    Extracts standard PostgreSQL connection parameters and custom metalab
    parameters (schema, experiments_root). Returns both the parsed config
    and a clean connection string suitable for psycopg.
    """
    parsed = urlparse(conn_str)

    # Parse query parameters
    params = {}
    if parsed.query:
        for key, values in parse_qs(parsed.query).items():
            params[key] = values[0] if values else ""

    # Custom metalab params (not passed to psycopg)
    custom_params = {"schema", "experiments_root"}

    # Build clean query string for psycopg (only standard postgres params)
    clean_params = {k: v for k, v in params.items() if k not in custom_params}
    clean_query = urlencode(clean_params) if clean_params else ""

    # Rebuild connection string without custom params
    clean_parsed = parsed._replace(query=clean_query)
    clean_conn_str = urlunparse(clean_parsed)

    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username,
        "password": parsed.password,
        "dbname": parsed.path.lstrip("/") if parsed.path else "metalab",
        "schema": params.get("schema", "public"),
        "experiments_root": params.get("experiments_root"),
        "clean_connection_string": clean_conn_str,
    }


class PostgresStore:
    """
    PostgreSQL-based storage backend with FileStore composition.

    PostgresStore wraps FileStore for file operations while PostgreSQL handles
    indexed data and Atlas features.

    Responsibilities:
        FileStore (delegated):
            - Logs (streaming via SupportsLogPath)
            - Artifacts (file-backed)
            - Working directory (for SLURM coordination)
            - Run records (backup/fallback)

        PostgreSQL (primary):
            - Run records (indexed for queries)
            - Results (structured data for derived metrics)
            - Derived metrics
            - Field catalog (Atlas)
            - Experiment manifests
    """

    def __init__(
        self,
        connection_string: str,
        *,
        experiments_root: str | Path | None = None,
        experiment_id: str | None = None,
        connect_timeout: float = 10.0,
        write_to_both: bool = False,
        auto_migrate: bool = True,
    ) -> None:
        """
        Initialize PostgresStore with FileStore composition.

        Args:
            connection_string: PostgreSQL connection URL.
            experiments_root: Root directory containing experiment subdirectories.
                             Each experiment creates {experiments_root}/{safe_id}/.
                             Required. Can also be set via URL query param.
            experiment_id: If provided, FileStore is created at {experiments_root}/{safe_id}/
                          where safe_id is experiment_id with ':' replaced by '_'.
            connect_timeout: Connection timeout in seconds.
            write_to_both: If True, write run records to both Postgres AND FileStore.
                          Default False (Postgres primary, FileStore fallback only).
            auto_migrate: Automatically create schema if needed.
        """
        try:
            import psycopg
            from psycopg_pool import ConnectionPool
        except ImportError:
            raise ImportError(
                "PostgresStore requires psycopg. "
                "Install with: pip install metalab[postgres]"
            )

        self._connection_string = connection_string
        self._connect_timeout = connect_timeout
        self._write_to_both = write_to_both

        # Parse config (extracts custom params and builds clean connection string)
        config = _parse_connection_string(connection_string)
        self._schema = config["schema"]

        # Determine experiments_root
        resolved_experiments_root = experiments_root or config.get("experiments_root")

        if resolved_experiments_root is None:
            raise ValueError(
                "PostgresStore requires 'experiments_root' parameter for FileStore composition. "
                "Set via constructor argument or URL query param, e.g.: "
                "postgresql://host/db?experiments_root=/shared/experiments"
            )

        self._experiments_root = Path(resolved_experiments_root)

        # If experiment_id provided, nest FileStore under sanitized experiment_id
        filestore_root = self._experiments_root
        if experiment_id:
            safe_id = experiment_id.replace(":", "_")  # my_exp:1.0 -> my_exp_1.0
            filestore_root = self._experiments_root / safe_id

        # Create FileStore for file operations
        from metalab.store.file import FileStore

        self._file_store = FileStore(filestore_root)

        # Track Postgres availability for fallback
        self._pg_available = True

        # Initialize connection pool with clean connection string
        # (psycopg doesn't accept custom params like 'schema' or 'experiments_root')
        self._pool = ConnectionPool(
            config["clean_connection_string"],
            min_size=1,
            max_size=10,
            timeout=connect_timeout,
        )

        # Auto-migrate schema
        if auto_migrate:
            self._ensure_schema()

        logger.info(
            f"PostgresStore initialized: {config['host']}:{config['port']}/{config['dbname']} "
            f"with FileStore at {self._experiments_root}"
        )

    @property
    def locator(self) -> str:
        """The store locator URI."""
        return self._connection_string

    @property
    def file_store(self) -> "FileStore":
        """Access the underlying FileStore."""
        from metalab.store.file import FileStore

        return self._file_store

    # =========================================================================
    # Delegated to FileStore: SupportsWorkingDirectory
    # =========================================================================

    def get_working_directory(self) -> Path:
        """
        Return the filesystem directory used for shared run coordination.

        Delegated to FileStore.
        """
        return self._file_store.get_working_directory()

    # =========================================================================
    # Delegated to FileStore: SupportsLogPath
    # =========================================================================

    def get_log_path(self, run_id: str, name: str) -> Path:
        """
        Get the path where a log file should be written.

        Delegated to FileStore for streaming log support.
        """
        return self._file_store.get_log_path(run_id, name)

    def put_log(
        self,
        run_id: str,
        name: str,
        content: str,
        label: str | None = None,
    ) -> None:
        """
        Store a log file for a run.

        Delegated to FileStore.
        """
        self._file_store.put_log(run_id, name, content)

    def get_log(self, run_id: str, name: str) -> str | None:
        """
        Retrieve a log file.

        Reads from FileStore. Falls back to legacy database storage.
        """
        # Primary: FileStore
        result = self._file_store.get_log(run_id, name)
        if result is not None:
            return result

        # Fallback: Legacy database storage
        if self._pg_available:
            try:
                with self._conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"""
                            SELECT content FROM {self._table('logs')}
                            WHERE run_id = %s AND name = %s
                        """,
                            [run_id, name],
                        )
                        row = cur.fetchone()
                        if row:
                            return row[0]
            except Exception:
                pass

        return None

    def list_logs(self, run_id: str) -> list[str]:
        """
        List available log names for a run.

        Delegated to FileStore.
        """
        return self._file_store.list_logs(run_id)

    # =========================================================================
    # Delegated to FileStore: Artifacts
    # =========================================================================

    def put_artifact(
        self,
        data: bytes | Path,
        descriptor: ArtifactDescriptor,
    ) -> ArtifactDescriptor:
        """
        Store an artifact.

        Delegated to FileStore for file-backed storage.
        """
        return self._file_store.put_artifact(data, descriptor)

    def get_artifact(self, uri: str) -> bytes:
        """
        Retrieve artifact data by URI.

        Handles both FileStore paths and legacy pgblob:// URIs.
        """
        if uri.startswith("pgblob://"):
            # Legacy inline blob - read from database
            artifact_id = uri.replace("pgblob://", "")
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT content FROM {self._table('artifact_blobs')}
                        WHERE artifact_id = %s
                    """,
                        [artifact_id],
                    )
                    row = cur.fetchone()
                    if row is None:
                        raise FileNotFoundError(f"Artifact blob not found: {artifact_id}")
                    return bytes(row[0])
        else:
            # FileStore path
            return self._file_store.get_artifact(uri)

    def open_artifact(self, uri: str) -> "BinaryIO":
        """
        Open an artifact for reading.

        Implements SupportsArtifactOpen capability.
        """
        import io

        if uri.startswith("pgblob://"):
            # Legacy inline blob - wrap in BytesIO
            content = self.get_artifact(uri)
            return io.BytesIO(content)
        else:
            # FileStore path
            return self._file_store.open_artifact(uri)

    def list_artifacts(self, run_id: str) -> list[ArtifactDescriptor]:
        """
        List artifacts for a run.

        Reads from FileStore manifest.
        """
        return self._file_store.list_artifacts(run_id)

    # =========================================================================
    # PostgreSQL connection management
    # =========================================================================

    @contextmanager
    def _conn(self) -> Generator[Any, None, None]:
        """Get a connection from the pool."""
        with self._pool.connection() as conn:
            yield conn

    def _table(self, name: str) -> str:
        """Get fully qualified table name."""
        return f"{self._schema}.{name}"

    def _check_pg_available(self) -> bool:
        """Check if PostgreSQL is available."""
        if not self._pg_available:
            return False
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            self._pg_available = False
            logger.warning("PostgreSQL unavailable, using FileStore fallback")
            return False

    # =========================================================================
    # Schema management
    # =========================================================================

    def _ensure_schema(self) -> None:
        """Create schema and tables if they don't exist."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                # Create schema
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")

                # Create runs table
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table('runs')} (
                        run_id TEXT PRIMARY KEY,
                        experiment_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        context_fingerprint TEXT NOT NULL,
                        params_fingerprint TEXT NOT NULL,
                        seed_fingerprint TEXT NOT NULL,
                        started_at TIMESTAMPTZ NOT NULL,
                        finished_at TIMESTAMPTZ,
                        duration_ms INTEGER,
                        record_json JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """
                )

                # Create indexes for runs
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_runs_experiment_id 
                    ON {self._table('runs')} (experiment_id)
                """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_runs_status 
                    ON {self._table('runs')} (status)
                """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_runs_started_at 
                    ON {self._table('runs')} (started_at DESC)
                """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_runs_experiment_started 
                    ON {self._table('runs')} (experiment_id, started_at DESC)
                """
                )

                # Create results table (NEW: for structured data)
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table('results')} (
                        id SERIAL PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        data JSONB NOT NULL,
                        dtype TEXT,
                        shape INTEGER[],
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(run_id, name)
                    )
                """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_results_run_id 
                    ON {self._table('results')} (run_id)
                """
                )

                # Create derived metrics table
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table('derived')} (
                        run_id TEXT PRIMARY KEY,
                        derived_json JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """
                )

                # Create logs table (legacy, for backward compatibility)
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table('logs')} (
                        id SERIAL PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE (run_id, name)
                    )
                """
                )

                # Create artifacts table (legacy metadata, for backward compatibility)
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table('artifacts')} (
                        artifact_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        format TEXT NOT NULL,
                        uri TEXT NOT NULL,
                        content_hash TEXT,
                        size_bytes INTEGER,
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_artifacts_run_id 
                    ON {self._table('artifacts')} (run_id)
                """
                )

                # Create artifact_blobs table (legacy inline storage)
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table('artifact_blobs')} (
                        artifact_id TEXT PRIMARY KEY,
                        content BYTEA NOT NULL,
                        content_type TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """
                )

                # Create experiment_manifests table
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table('experiment_manifests')} (
                        id SERIAL PRIMARY KEY,
                        experiment_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        manifest_json JSONB NOT NULL,
                        submitted_at TIMESTAMPTZ NOT NULL,
                        total_runs INTEGER,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE (experiment_id, timestamp)
                    )
                """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_manifests_experiment_id 
                    ON {self._table('experiment_manifests')} (experiment_id, submitted_at DESC)
                """
                )

                # Create field_catalog table (for Atlas field index)
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table('field_catalog')} (
                        namespace TEXT NOT NULL,
                        field_name TEXT NOT NULL,
                        field_type TEXT,
                        count INTEGER DEFAULT 0,
                        values TEXT[],
                        min_value DOUBLE PRECISION,
                        max_value DOUBLE PRECISION,
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (namespace, field_name)
                    )
                """
                )

                # Create meta table
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table('meta')} (
                        key TEXT PRIMARY KEY,
                        value JSONB NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """
                )

                # Set schema version
                cur.execute(
                    f"""
                    INSERT INTO {self._table('meta')} (key, value)
                    VALUES ('schema_version', %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """,
                    [json.dumps(SCHEMA_VERSION)],
                )

                conn.commit()
                logger.debug(f"Schema ensured for {self._schema}")

    # =========================================================================
    # Run record operations (Postgres primary, FileStore fallback)
    # =========================================================================

    def put_run_record(self, record: RunRecord) -> None:
        """
        Persist a run record.

        Primary: PostgreSQL (indexed)
        Fallback: FileStore (if Postgres fails)
        Dual-write: Both (if write_to_both=True)
        """
        wrote_pg = False

        if self._pg_available or self._check_pg_available():
            try:
                self._put_run_record_pg(record)
                wrote_pg = True
            except Exception as e:
                logger.warning(f"PostgreSQL write failed, using FileStore: {e}")
                self._pg_available = False

        # Write to FileStore if: dual-write enabled OR Postgres failed
        if self._write_to_both or not wrote_pg:
            self._file_store.put_run_record(record)

    def _put_run_record_pg(self, record: RunRecord) -> None:
        """Write run record to PostgreSQL."""
        data = dump_run_record(record)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._table('runs')} (
                        run_id, experiment_id, status,
                        context_fingerprint, params_fingerprint, seed_fingerprint,
                        started_at, finished_at, duration_ms, record_json
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (run_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        finished_at = EXCLUDED.finished_at,
                        duration_ms = EXCLUDED.duration_ms,
                        record_json = EXCLUDED.record_json,
                        updated_at = NOW()
                """,
                    [
                        record.run_id,
                        record.experiment_id,
                        record.status.value,
                        record.context_fingerprint,
                        record.params_fingerprint,
                        record.seed_fingerprint,
                        _coerce_naive_local_to_utc(record.started_at),
                        _coerce_naive_local_to_utc(record.finished_at),
                        record.duration_ms,
                        json.dumps(data),
                    ],
                )
                conn.commit()

    def get_run_record(self, run_id: str) -> RunRecord | None:
        """
        Retrieve a run record by ID.

        Tries PostgreSQL first, then FileStore fallback.
        """
        if self._pg_available or self._check_pg_available():
            try:
                record = self._get_run_record_pg(run_id)
                if record is not None:
                    return record
            except Exception:
                self._pg_available = False

        return self._file_store.get_run_record(run_id)

    def _get_run_record_pg(self, run_id: str) -> RunRecord | None:
        """Read run record from PostgreSQL."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT record_json FROM {self._table('runs')}
                    WHERE run_id = %s
                """,
                    [run_id],
                )
                row = cur.fetchone()
                if row is None:
                    return None
                data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                return load_run_record(data)

    def list_run_records(self, experiment_id: str | None = None) -> list[RunRecord]:
        """
        List run records, optionally filtered by experiment.

        Tries PostgreSQL first, then FileStore fallback.
        """
        if self._pg_available or self._check_pg_available():
            try:
                return self._list_run_records_pg(experiment_id)
            except Exception:
                self._pg_available = False

        return self._file_store.list_run_records(experiment_id)

    def _list_run_records_pg(self, experiment_id: str | None = None) -> list[RunRecord]:
        """List run records from PostgreSQL."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                if experiment_id:
                    cur.execute(
                        f"""
                        SELECT record_json FROM {self._table('runs')}
                        WHERE experiment_id = %s
                        ORDER BY started_at DESC
                    """,
                        [experiment_id],
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT record_json FROM {self._table('runs')}
                        ORDER BY started_at DESC
                    """
                    )

                records = []
                for (record_json,) in cur.fetchall():
                    data = (
                        record_json
                        if isinstance(record_json, dict)
                        else json.loads(record_json)
                    )
                    records.append(load_run_record(data))
                return records

    def run_exists(self, run_id: str) -> bool:
        """
        Check if a run record exists.

        Checks both PostgreSQL and FileStore.
        """
        if self._pg_available or self._check_pg_available():
            try:
                if self._run_exists_pg(run_id):
                    return True
            except Exception:
                self._pg_available = False

        return self._file_store.run_exists(run_id)

    def _run_exists_pg(self, run_id: str) -> bool:
        """Check if run exists in PostgreSQL."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT 1 FROM {self._table('runs')}
                    WHERE run_id = %s
                """,
                    [run_id],
                )
                return cur.fetchone() is not None

    # =========================================================================
    # Results operations (structured data for derived metrics)
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
        """
        Store structured result data for a run.

        Args:
            run_id: The run identifier.
            name: The result name.
            data: The data (must be JSON-serializable, e.g., list, dict).
            dtype: Optional data type hint (e.g., "float64").
            shape: Optional shape for arrays.
            metadata: Optional metadata.
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._table('results')} (
                        run_id, name, data, dtype, shape, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, name) DO UPDATE SET
                        data = EXCLUDED.data,
                        dtype = EXCLUDED.dtype,
                        shape = EXCLUDED.shape,
                        metadata = EXCLUDED.metadata
                """,
                    [
                        run_id,
                        name,
                        json.dumps(data),
                        dtype,
                        shape,
                        json.dumps(metadata or {}),
                    ],
                )
                conn.commit()

    def get_result(self, run_id: str, name: str) -> dict[str, Any] | None:
        """
        Retrieve structured result data.

        Args:
            run_id: The run identifier.
            name: The result name.

        Returns:
            Dict with keys: data, dtype, shape, metadata. Or None if not found.
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT data, dtype, shape, metadata
                    FROM {self._table('results')}
                    WHERE run_id = %s AND name = %s
                """,
                    [run_id, name],
                )
                row = cur.fetchone()
                if row is None:
                    return None

                return {
                    "data": row[0] if isinstance(row[0], (dict, list)) else json.loads(row[0]),
                    "dtype": row[1],
                    "shape": list(row[2]) if row[2] else None,
                    "metadata": row[3] if isinstance(row[3], dict) else json.loads(row[3] or "{}"),
                }

    def list_results(self, run_id: str) -> list[str]:
        """
        List result names for a run.

        Args:
            run_id: The run identifier.

        Returns:
            List of result names.
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT name FROM {self._table('results')}
                    WHERE run_id = %s
                """,
                    [run_id],
                )
                return [row[0] for row in cur.fetchall()]

    # =========================================================================
    # Derived metrics operations
    # =========================================================================

    def put_derived(self, run_id: str, derived: dict[str, Metric]) -> None:
        """Persist derived metrics for a run."""
        wrote_pg = False

        if self._pg_available or self._check_pg_available():
            try:
                with self._conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"""
                            INSERT INTO {self._table('derived')} (run_id, derived_json)
                            VALUES (%s, %s)
                            ON CONFLICT (run_id) DO UPDATE SET
                                derived_json = EXCLUDED.derived_json,
                                updated_at = NOW()
                        """,
                            [run_id, json.dumps(derived)],
                        )
                        conn.commit()
                wrote_pg = True
            except Exception as e:
                logger.warning(f"PostgreSQL derived write failed: {e}")
                self._pg_available = False

        # Fallback to FileStore
        if self._write_to_both or not wrote_pg:
            self._file_store.put_derived(run_id, derived)

    def get_derived(self, run_id: str) -> dict[str, Metric] | None:
        """Retrieve derived metrics for a run."""
        if self._pg_available or self._check_pg_available():
            try:
                with self._conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"""
                            SELECT derived_json FROM {self._table('derived')}
                            WHERE run_id = %s
                        """,
                            [run_id],
                        )
                        row = cur.fetchone()
                        if row is not None:
                            return row[0] if isinstance(row[0], dict) else json.loads(row[0])
            except Exception:
                self._pg_available = False

        return self._file_store.get_derived(run_id)

    def derived_exists(self, run_id: str) -> bool:
        """Check if derived metrics exist for a run."""
        if self._pg_available or self._check_pg_available():
            try:
                with self._conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"""
                            SELECT 1 FROM {self._table('derived')}
                            WHERE run_id = %s
                        """,
                            [run_id],
                        )
                        if cur.fetchone() is not None:
                            return True
            except Exception:
                self._pg_available = False

        return self._file_store.derived_exists(run_id)

    # =========================================================================
    # Experiment manifest operations
    # =========================================================================

    def put_experiment_manifest(
        self,
        experiment_id: str,
        manifest: dict[str, Any],
        timestamp: str | None = None,
    ) -> None:
        """Store an experiment manifest."""
        submitted_at = manifest.get("submitted_at")
        if isinstance(submitted_at, str):
            submitted_at = datetime.fromisoformat(submitted_at)
        elif submitted_at is None:
            submitted_at = datetime.now()

        total_runs = manifest.get("total_runs", 0)

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        wrote_pg = False

        if self._pg_available or self._check_pg_available():
            try:
                with self._conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"""
                            INSERT INTO {self._table('experiment_manifests')} (
                                experiment_id, timestamp, manifest_json, submitted_at, total_runs
                            ) VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (experiment_id, timestamp) DO UPDATE SET
                                manifest_json = EXCLUDED.manifest_json,
                                total_runs = EXCLUDED.total_runs
                        """,
                            [
                                experiment_id,
                                timestamp,
                                json.dumps(manifest),
                                submitted_at,
                                total_runs,
                            ],
                        )
                        conn.commit()
                wrote_pg = True
            except Exception as e:
                logger.warning(f"PostgreSQL manifest write failed: {e}")
                self._pg_available = False

        # Fallback to FileStore
        if self._write_to_both or not wrote_pg:
            self._file_store.put_experiment_manifest(experiment_id, manifest, timestamp)

    def get_experiment_manifest(self, experiment_id: str) -> dict[str, Any] | None:
        """Retrieve the most recent experiment manifest by ID."""
        if self._pg_available or self._check_pg_available():
            try:
                with self._conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"""
                            SELECT manifest_json FROM {self._table('experiment_manifests')}
                            WHERE experiment_id = %s
                            ORDER BY submitted_at DESC
                            LIMIT 1
                        """,
                            [experiment_id],
                        )
                        row = cur.fetchone()
                        if row is not None:
                            return row[0] if isinstance(row[0], dict) else json.loads(row[0])
            except Exception:
                self._pg_available = False

        return self._file_store.get_experiment_manifest(experiment_id)

    # =========================================================================
    # Field catalog operations (for Atlas)
    # =========================================================================

    def update_field_catalog(self, run_id: str | None = None) -> None:
        """
        Update the field catalog for Atlas field index.

        If run_id is provided, incrementally updates from that run.
        Otherwise, rebuilds from all runs (slow).
        """
        if not self._pg_available and not self._check_pg_available():
            logger.warning("Field catalog update skipped: PostgreSQL unavailable")
            return

        with self._conn() as conn:
            with conn.cursor() as cur:
                # Get runs to process
                if run_id:
                    cur.execute(
                        f"""
                        SELECT record_json FROM {self._table('runs')}
                        WHERE run_id = %s
                    """,
                        [run_id],
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT record_json FROM {self._table('runs')}
                    """
                    )

                # Collect field stats
                stats: dict[tuple[str, str], dict] = {}

                for (record_json,) in cur.fetchall():
                    data = (
                        record_json
                        if isinstance(record_json, dict)
                        else json.loads(record_json)
                    )

                    # Process params
                    for key, value in data.get("params_resolved", {}).items():
                        self._update_field_stats(stats, "params", key, value)

                    # Process metrics
                    for key, value in data.get("metrics", {}).items():
                        self._update_field_stats(stats, "metrics", key, value)

                # Upsert to field_catalog
                for (namespace, field_name), field_stats in stats.items():
                    values = list(field_stats.get("values", set()))[:100]  # Limit values
                    cur.execute(
                        f"""
                        INSERT INTO {self._table('field_catalog')} (
                            namespace, field_name, field_type, count, values, min_value, max_value
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (namespace, field_name) DO UPDATE SET
                            field_type = EXCLUDED.field_type,
                            count = {self._table('field_catalog')}.count + EXCLUDED.count,
                            values = EXCLUDED.values,
                            min_value = LEAST({self._table('field_catalog')}.min_value, EXCLUDED.min_value),
                            max_value = GREATEST({self._table('field_catalog')}.max_value, EXCLUDED.max_value),
                            updated_at = NOW()
                    """,
                        [
                            namespace,
                            field_name,
                            field_stats.get("type", "unknown"),
                            field_stats.get("count", 0),
                            values if values else None,
                            field_stats.get("min"),
                            field_stats.get("max"),
                        ],
                    )

                conn.commit()

    def _update_field_stats(
        self,
        stats: dict[tuple[str, str], dict],
        namespace: str,
        key: str,
        value: Any,
    ) -> None:
        """Update field statistics."""
        stat_key = (namespace, key)

        if stat_key not in stats:
            stats[stat_key] = {
                "type": self._infer_type(value),
                "count": 0,
                "values": set(),
                "min": None,
                "max": None,
            }

        s = stats[stat_key]
        s["count"] += 1

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if s["min"] is None or value < s["min"]:
                s["min"] = value
            if s["max"] is None or value > s["max"]:
                s["max"] = value
        elif isinstance(value, (str, bool)):
            if len(s["values"]) < 100:
                s["values"].add(str(value))

    def _infer_type(self, value: Any) -> str:
        """Infer field type from value."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, (int, float)):
            return "numeric"
        elif isinstance(value, str):
            return "string"
        return "unknown"

    # =========================================================================
    # Utility methods
    # =========================================================================

    def delete_run(self, run_id: str) -> None:
        """Delete a run and all associated data."""
        # Delete from PostgreSQL
        if self._pg_available or self._check_pg_available():
            try:
                with self._conn() as conn:
                    with conn.cursor() as cur:
                        # Delete results
                        cur.execute(
                            f"""
                            DELETE FROM {self._table('results')}
                            WHERE run_id = %s
                        """,
                            [run_id],
                        )

                        # Delete blobs (legacy)
                        cur.execute(
                            f"""
                            DELETE FROM {self._table('artifact_blobs')}
                            WHERE artifact_id IN (
                                SELECT artifact_id FROM {self._table('artifacts')}
                                WHERE run_id = %s
                            )
                        """,
                            [run_id],
                        )

                        # Delete artifacts (legacy)
                        cur.execute(
                            f"""
                            DELETE FROM {self._table('artifacts')}
                            WHERE run_id = %s
                        """,
                            [run_id],
                        )

                        # Delete logs (legacy)
                        cur.execute(
                            f"""
                            DELETE FROM {self._table('logs')}
                            WHERE run_id = %s
                        """,
                            [run_id],
                        )

                        # Delete derived
                        cur.execute(
                            f"""
                            DELETE FROM {self._table('derived')}
                            WHERE run_id = %s
                        """,
                            [run_id],
                        )

                        # Delete run record
                        cur.execute(
                            f"""
                            DELETE FROM {self._table('runs')}
                            WHERE run_id = %s
                        """,
                            [run_id],
                        )

                        conn.commit()
            except Exception as e:
                logger.warning(f"PostgreSQL delete failed: {e}")

        # Also delete from FileStore
        self._file_store.delete_run(run_id)

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            self._pool.close()

    def __enter__(self) -> "PostgresStore":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager, ensuring close is called."""
        self.close()
