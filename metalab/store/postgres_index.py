"""
PostgresIndex: Query index backed by PostgreSQL.

This is NOT a complete store—it only provides indexed queries.
The actual data lives in FileStore. PostgresIndex accelerates:
- Run record lookups and filtering
- Derived metrics queries
- Experiment manifest lookups
- Field catalog for Atlas

If Postgres is lost, rebuild the index from FileStore.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from metalab.types import RunRecord

from metalab.schema import dump_run_record, load_run_record

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 2

# Query parameters that are metalab-specific and should be stripped
# before passing the connection string to psycopg
METALAB_PARAMS = {"file_root", "schema"}


def _strip_metalab_params(connection_string: str) -> str:
    """
    Strip metalab-specific query parameters from a PostgreSQL connection string.

    psycopg rejects unknown query parameters, so we need to remove metalab's
    custom params (like file_root) before passing to the connection pool.

    Args:
        connection_string: PostgreSQL URI, possibly with metalab params.

    Returns:
        PostgreSQL URI with only psycopg-compatible parameters.
    """
    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

    parsed = urlparse(connection_string)

    if not parsed.query:
        return connection_string

    # Parse and filter params
    params = parse_qs(parsed.query, keep_blank_values=True)
    filtered = {k: v for k, v in params.items() if k not in METALAB_PARAMS}

    # Rebuild URI
    new_query = urlencode(filtered, doseq=True) if filtered else ""
    return urlunparse(parsed._replace(query=new_query))


def _coerce_naive_local_to_utc(dt: datetime | None) -> datetime | None:
    """
    Coerce a datetime to tz-aware UTC for TIMESTAMPTZ columns.

    metalab historically uses naive local datetimes. If we pass those directly
    into TIMESTAMPTZ columns, Postgres interprets them in the session timezone.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        assert local_tz is not None
        dt = dt.replace(tzinfo=local_tz)
    return dt.astimezone(timezone.utc)


class PostgresIndex:
    """
    Query index backed by PostgreSQL.

    Not a complete store—only provides fast lookups and filtering.
    The real data lives in FileStore.
    """

    def __init__(
        self,
        connection_string: str,
        *,
        schema: str = "public",
        auto_migrate: bool = True,
        pool_min_size: int = 1,
        pool_max_size: int = 10,
        connect_timeout: float = 10.0,
    ) -> None:
        """
        Initialize PostgresIndex.

        Args:
            connection_string: PostgreSQL connection URL.
            schema: Database schema name.
            auto_migrate: Automatically create schema if needed.
            pool_min_size: Minimum connection pool size.
            pool_max_size: Maximum connection pool size.
            connect_timeout: Connection timeout in seconds.
        """
        try:
            from psycopg_pool import ConnectionPool
        except ImportError:
            raise ImportError(
                "PostgresIndex requires psycopg. "
                "Install with: pip install metalab[postgres]"
            )

        self._connection_string = connection_string
        self._schema = schema

        # Strip metalab-specific params before passing to psycopg
        # (psycopg rejects unknown query parameters like file_root)
        clean_connection_string = _strip_metalab_params(connection_string)

        # Initialize connection pool
        self._pool = ConnectionPool(
            clean_connection_string,
            min_size=pool_min_size,
            max_size=pool_max_size,
            timeout=connect_timeout,
        )

        if auto_migrate:
            self._ensure_schema()

    def __repr__(self) -> str:
        """Return a string representation of the index."""
        safe_conn = self._sanitize_connection_string(self._connection_string)
        return f"PostgresIndex(connection={safe_conn!r}, schema={self._schema!r})"

    @staticmethod
    def _sanitize_connection_string(conn: str) -> str:
        """Hide password in connection string for safe display."""
        from urllib.parse import urlparse, urlunparse

        try:
            parsed = urlparse(conn)
            if parsed.password:
                netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
                return urlunparse(parsed._replace(netloc=netloc))
        except Exception:
            pass
        return conn

    @contextmanager
    def _conn(self) -> Generator[Any, None, None]:
        """Get a connection from the pool."""
        with self._pool.connection() as conn:
            yield conn

    def _table(self, name: str) -> str:
        """Get fully qualified table name."""
        return f"{self._schema}.{name}"

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

                # Create field_catalog table (for Atlas)
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
    # Run record indexing
    # =========================================================================

    def index_record(self, record: "RunRecord") -> None:
        """
        Add or update a run record in the index.

        Args:
            record: The RunRecord to index.
        """
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

    def get_record(self, run_id: str) -> "RunRecord | None":
        """
        Retrieve a run record from the index.

        Args:
            run_id: The run identifier.

        Returns:
            The RunRecord, or None if not found.
        """
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

    def list_records(self, experiment_id: str | None = None) -> list["RunRecord"]:
        """
        List run records from the index.

        Args:
            experiment_id: Optional filter by experiment.

        Returns:
            List of matching RunRecords.
        """
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

    def record_exists(self, run_id: str) -> bool:
        """Check if a run record exists in the index."""
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

    def delete_record(self, run_id: str) -> None:
        """Remove a run record from the index."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {self._table('runs')}
                    WHERE run_id = %s
                """,
                    [run_id],
                )
                conn.commit()

    # =========================================================================
    # Derived metrics indexing
    # =========================================================================

    def index_derived(self, run_id: str, derived: dict[str, Any]) -> None:
        """Index derived metrics for a run."""
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

    def get_derived(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve derived metrics from the index."""
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
                if row is None:
                    return None
                return row[0] if isinstance(row[0], dict) else json.loads(row[0])

    def derived_exists(self, run_id: str) -> bool:
        """Check if derived metrics exist in the index."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT 1 FROM {self._table('derived')}
                    WHERE run_id = %s
                """,
                    [run_id],
                )
                return cur.fetchone() is not None

    def delete_derived(self, run_id: str) -> None:
        """Remove derived metrics from the index."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {self._table('derived')}
                    WHERE run_id = %s
                """,
                    [run_id],
                )
                conn.commit()

    # =========================================================================
    # Experiment manifest indexing
    # =========================================================================

    def index_manifest(
        self,
        experiment_id: str,
        manifest: dict[str, Any],
        timestamp: str | None = None,
    ) -> None:
        """Index an experiment manifest."""
        submitted_at = manifest.get("submitted_at")
        if isinstance(submitted_at, str):
            submitted_at = datetime.fromisoformat(submitted_at)
        elif submitted_at is None:
            submitted_at = datetime.now()

        total_runs = manifest.get("total_runs", 0)

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    def get_manifest(self, experiment_id: str) -> dict[str, Any] | None:
        """Retrieve the most recent experiment manifest from the index."""
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
                if row is None:
                    return None
                return row[0] if isinstance(row[0], dict) else json.loads(row[0])

    # =========================================================================
    # Field catalog (for Atlas)
    # =========================================================================

    def update_field_catalog(self, records: list["RunRecord"] | None = None) -> None:
        """
        Update the field catalog from run records.

        Args:
            records: Records to process. If None, processes all indexed records.
        """
        if records is None:
            records = self.list_records()

        # Collect field stats
        stats: dict[tuple[str, str], dict] = {}

        for record in records:
            data = dump_run_record(record)

            # Process params
            for key, value in data.get("params_resolved", {}).items():
                self._update_field_stats(stats, "params", key, value)

            # Process metrics
            for key, value in data.get("metrics", {}).items():
                self._update_field_stats(stats, "metrics", key, value)

        # Upsert to field_catalog
        with self._conn() as conn:
            with conn.cursor() as cur:
                for (namespace, field_name), field_stats in stats.items():
                    values = list(field_stats.get("values", set()))[:100]
                    cur.execute(
                        f"""
                        INSERT INTO {self._table('field_catalog')} (
                            namespace, field_name, field_type, count, values, min_value, max_value
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (namespace, field_name) DO UPDATE SET
                            field_type = EXCLUDED.field_type,
                            count = EXCLUDED.count,
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
    # Index management
    # =========================================================================

    def clear(self) -> None:
        """Clear all indexed data (for rebuild)."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE {self._table('runs')} CASCADE")
                cur.execute(f"TRUNCATE {self._table('derived')} CASCADE")
                cur.execute(f"TRUNCATE {self._table('experiment_manifests')} CASCADE")
                cur.execute(f"TRUNCATE {self._table('field_catalog')} CASCADE")
                conn.commit()
        logger.info("Index cleared")

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            self._pool.close()

    def __enter__(self) -> "PostgresIndex":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager."""
        self.close()
