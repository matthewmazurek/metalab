"""
PostgresStore: PostgreSQL-based storage backend.

Provides efficient storage and querying for run records, derived metrics,
logs, artifacts metadata, and experiment manifests.

Schema:
    runs: Primary run record storage with JSONB for full record
    derived: Per-run derived metrics (JSONB)
    logs: Log content (TEXT) 
    artifacts: Artifact metadata with URI to file storage
    artifact_blobs: Optional inline blob storage for small artifacts
    experiment_manifests: Versioned experiment manifests
    field_catalog: Pre-computed field index for Atlas
    meta: Schema version and migrations

Connection:
    postgresql://user:password@host:port/database?schema=public&artifact_root=/path

Concurrency:
    Uses upsert (INSERT ON CONFLICT) for idempotent writes.
    Connection pooling via psycopg_pool.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator
from urllib.parse import parse_qs, urlparse

from metalab.schema import dump_run_record, load_run_record
from metalab.types import ArtifactDescriptor, Metric, Provenance, RunRecord, Status

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 1

# Default artifact blob size threshold (store in DB if smaller)
DEFAULT_BLOB_THRESHOLD = 1024 * 1024  # 1MB


def _parse_connection_string(conn_str: str) -> dict[str, Any]:
    """Parse PostgreSQL connection string."""
    parsed = urlparse(conn_str)
    
    params = {}
    if parsed.query:
        for key, values in parse_qs(parsed.query).items():
            params[key] = values[0] if values else ""
    
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username,
        "password": parsed.password,
        "dbname": parsed.path.lstrip("/") if parsed.path else "metalab",
        "schema": params.get("schema", "public"),
        "artifact_root": params.get("artifact_root"),
    }


class PostgresStore:
    """
    PostgreSQL-based storage backend implementing the Store protocol.
    
    Provides:
    - Efficient run record storage with JSONB indexing
    - Idempotent upserts for concurrent writes
    - Connection pooling
    - Optional inline blob storage for small artifacts
    """
    
    def __init__(
        self,
        connection_string: str,
        *,
        connect_timeout: float = 10.0,
        artifact_root: str | None = None,
        blob_threshold: int = DEFAULT_BLOB_THRESHOLD,
        auto_migrate: bool = True,
    ) -> None:
        """
        Initialize PostgresStore.
        
        Args:
            connection_string: PostgreSQL connection URL.
            connect_timeout: Connection timeout in seconds.
            artifact_root: Path for file-backed artifacts. If None, extracted from URL.
            blob_threshold: Max size for inline blob storage (bytes).
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
        self._blob_threshold = blob_threshold
        
        # Parse config
        config = _parse_connection_string(connection_string)
        self._schema = config["schema"]
        self._artifact_root = Path(artifact_root) if artifact_root else (
            Path(config["artifact_root"]) if config.get("artifact_root") else None
        )
        
        # Initialize connection pool
        self._pool = ConnectionPool(
            connection_string,
            min_size=1,
            max_size=10,
            timeout=connect_timeout,
        )
        
        # Auto-migrate schema
        if auto_migrate:
            self._ensure_schema()
        
        logger.info(
            f"PostgresStore initialized: {config['host']}:{config['port']}/{config['dbname']}"
        )
    
    @property
    def locator(self) -> str:
        """The store locator URI."""
        return self._connection_string
    
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
                cur.execute(f"""
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
                """)
                
                # Create indexes for runs
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_runs_experiment_id 
                    ON {self._table('runs')} (experiment_id)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_runs_status 
                    ON {self._table('runs')} (status)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_runs_started_at 
                    ON {self._table('runs')} (started_at DESC)
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_runs_experiment_started 
                    ON {self._table('runs')} (experiment_id, started_at DESC)
                """)
                
                # Create derived metrics table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table('derived')} (
                        run_id TEXT PRIMARY KEY REFERENCES {self._table('runs')}(run_id),
                        derived_json JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                
                # Create logs table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table('logs')} (
                        id SERIAL PRIMARY KEY,
                        run_id TEXT NOT NULL REFERENCES {self._table('runs')}(run_id),
                        name TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE (run_id, name)
                    )
                """)
                
                # Create artifacts table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table('artifacts')} (
                        artifact_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL REFERENCES {self._table('runs')}(run_id),
                        name TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        format TEXT NOT NULL,
                        uri TEXT NOT NULL,
                        content_hash TEXT,
                        size_bytes INTEGER,
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_artifacts_run_id 
                    ON {self._table('artifacts')} (run_id)
                """)
                
                # Create artifact_blobs table (optional inline storage)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table('artifact_blobs')} (
                        artifact_id TEXT PRIMARY KEY REFERENCES {self._table('artifacts')}(artifact_id),
                        content BYTEA NOT NULL,
                        content_type TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                
                # Create experiment_manifests table
                cur.execute(f"""
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
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_manifests_experiment_id 
                    ON {self._table('experiment_manifests')} (experiment_id, submitted_at DESC)
                """)
                
                # Create field_catalog table (for Atlas field index)
                cur.execute(f"""
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
                """)
                
                # Create meta table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table('meta')} (
                        key TEXT PRIMARY KEY,
                        value JSONB NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                
                # Set schema version
                cur.execute(f"""
                    INSERT INTO {self._table('meta')} (key, value)
                    VALUES ('schema_version', %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """, [json.dumps(SCHEMA_VERSION)])
                
                conn.commit()
                logger.debug(f"Schema ensured for {self._schema}")
    
    # =========================================================================
    # Run record operations
    # =========================================================================
    
    def put_run_record(self, record: RunRecord) -> None:
        """
        Persist a run record with upsert semantics.
        
        Updates existing record if run_id exists (for status transitions).
        """
        data = dump_run_record(record)
        
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
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
                """, [
                    record.run_id,
                    record.experiment_id,
                    record.status.value,
                    record.context_fingerprint,
                    record.params_fingerprint,
                    record.seed_fingerprint,
                    record.started_at,
                    record.finished_at,
                    record.duration_ms,
                    json.dumps(data),
                ])
                conn.commit()
    
    def get_run_record(self, run_id: str) -> RunRecord | None:
        """Retrieve a run record by ID."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT record_json FROM {self._table('runs')}
                    WHERE run_id = %s
                """, [run_id])
                row = cur.fetchone()
                if row is None:
                    return None
                data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                return load_run_record(data)
    
    def list_run_records(self, experiment_id: str | None = None) -> list[RunRecord]:
        """List run records, optionally filtered by experiment."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                if experiment_id:
                    cur.execute(f"""
                        SELECT record_json FROM {self._table('runs')}
                        WHERE experiment_id = %s
                        ORDER BY started_at DESC
                    """, [experiment_id])
                else:
                    cur.execute(f"""
                        SELECT record_json FROM {self._table('runs')}
                        ORDER BY started_at DESC
                    """)
                
                records = []
                for (record_json,) in cur.fetchall():
                    data = record_json if isinstance(record_json, dict) else json.loads(record_json)
                    records.append(load_run_record(data))
                return records
    
    def run_exists(self, run_id: str) -> bool:
        """Check if a run record exists."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT 1 FROM {self._table('runs')}
                    WHERE run_id = %s
                """, [run_id])
                return cur.fetchone() is not None
    
    # =========================================================================
    # Artifact operations
    # =========================================================================
    
    def put_artifact(
        self,
        data: bytes | Path,
        descriptor: ArtifactDescriptor,
    ) -> ArtifactDescriptor:
        """
        Store an artifact.
        
        Small artifacts (< blob_threshold) are stored inline in Postgres.
        Large artifacts are stored in the artifact root filesystem.
        """
        # Get run_id from descriptor metadata
        run_id = descriptor.metadata.get("_run_id")
        if not run_id:
            run_id = descriptor.artifact_id[:16]
        
        # Get data bytes
        if isinstance(data, Path):
            content = data.read_bytes()
            ext = data.suffix
        else:
            content = data
            ext = f".{descriptor.format}" if descriptor.format else ""
        
        size_bytes = len(content)
        
        # Determine storage location
        if size_bytes <= self._blob_threshold:
            # Store inline
            uri = f"pgblob://{descriptor.artifact_id}"
            inline = True
        else:
            # Store on filesystem
            if self._artifact_root is None:
                raise ValueError("artifact_root required for large artifacts")
            
            artifact_dir = self._artifact_root / "artifacts" / run_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            
            dest_name = f"{descriptor.name}{ext}"
            dest_path = artifact_dir / dest_name
            dest_path.write_bytes(content)
            uri = str(dest_path)
            inline = False
        
        # Clean metadata (remove internal keys)
        clean_metadata = {
            k: v for k, v in descriptor.metadata.items() if not k.startswith("_")
        }
        
        with self._conn() as conn:
            with conn.cursor() as cur:
                # Insert artifact metadata
                cur.execute(f"""
                    INSERT INTO {self._table('artifacts')} (
                        artifact_id, run_id, name, kind, format, uri,
                        content_hash, size_bytes, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (artifact_id) DO UPDATE SET
                        uri = EXCLUDED.uri,
                        size_bytes = EXCLUDED.size_bytes
                """, [
                    descriptor.artifact_id,
                    run_id,
                    descriptor.name,
                    descriptor.kind,
                    descriptor.format,
                    uri,
                    descriptor.content_hash,
                    size_bytes,
                    json.dumps(clean_metadata),
                ])
                
                # Store inline blob if applicable
                if inline:
                    cur.execute(f"""
                        INSERT INTO {self._table('artifact_blobs')} (
                            artifact_id, content, content_type
                        ) VALUES (%s, %s, %s)
                        ON CONFLICT (artifact_id) DO UPDATE SET
                            content = EXCLUDED.content
                    """, [
                        descriptor.artifact_id,
                        content,
                        descriptor.format,
                    ])
                
                conn.commit()
        
        return ArtifactDescriptor(
            artifact_id=descriptor.artifact_id,
            name=descriptor.name,
            kind=descriptor.kind,
            format=descriptor.format,
            uri=uri,
            content_hash=descriptor.content_hash,
            size_bytes=size_bytes,
            metadata=clean_metadata,
        )
    
    def get_artifact(self, uri: str) -> bytes:
        """Retrieve artifact data by URI."""
        if uri.startswith("pgblob://"):
            # Inline blob
            artifact_id = uri.replace("pgblob://", "")
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT content FROM {self._table('artifact_blobs')}
                        WHERE artifact_id = %s
                    """, [artifact_id])
                    row = cur.fetchone()
                    if row is None:
                        raise FileNotFoundError(f"Artifact blob not found: {artifact_id}")
                    return bytes(row[0])
        else:
            # Filesystem
            path = Path(uri)
            if not path.exists():
                raise FileNotFoundError(f"Artifact not found: {uri}")
            return path.read_bytes()
    
    def list_artifacts(self, run_id: str) -> list[ArtifactDescriptor]:
        """List artifacts for a run."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT artifact_id, name, kind, format, uri,
                           content_hash, size_bytes, metadata
                    FROM {self._table('artifacts')}
                    WHERE run_id = %s
                """, [run_id])
                
                artifacts = []
                for row in cur.fetchall():
                    metadata = row[7] if isinstance(row[7], dict) else json.loads(row[7] or "{}")
                    artifacts.append(ArtifactDescriptor(
                        artifact_id=row[0],
                        name=row[1],
                        kind=row[2],
                        format=row[3],
                        uri=row[4],
                        content_hash=row[5],
                        size_bytes=row[6],
                        metadata=metadata,
                    ))
                return artifacts
    
    # =========================================================================
    # Derived metrics operations
    # =========================================================================
    
    def put_derived(self, run_id: str, derived: dict[str, Metric]) -> None:
        """Persist derived metrics for a run."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self._table('derived')} (run_id, derived_json)
                    VALUES (%s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        derived_json = EXCLUDED.derived_json,
                        updated_at = NOW()
                """, [run_id, json.dumps(derived)])
                conn.commit()
    
    def get_derived(self, run_id: str) -> dict[str, Metric] | None:
        """Retrieve derived metrics for a run."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT derived_json FROM {self._table('derived')}
                    WHERE run_id = %s
                """, [run_id])
                row = cur.fetchone()
                if row is None:
                    return None
                return row[0] if isinstance(row[0], dict) else json.loads(row[0])
    
    def derived_exists(self, run_id: str) -> bool:
        """Check if derived metrics exist for a run."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT 1 FROM {self._table('derived')}
                    WHERE run_id = %s
                """, [run_id])
                return cur.fetchone() is not None
    
    # =========================================================================
    # Log operations
    # =========================================================================
    
    def put_log(
        self,
        run_id: str,
        name: str,
        content: str,
        label: str | None = None,
    ) -> None:
        """Store a log file for a run."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self._table('logs')} (run_id, name, content)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (run_id, name) DO UPDATE SET
                        content = EXCLUDED.content
                """, [run_id, name, content])
                conn.commit()
    
    def get_log(self, run_id: str, name: str) -> str | None:
        """Retrieve a log file."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT content FROM {self._table('logs')}
                    WHERE run_id = %s AND name = %s
                """, [run_id, name])
                row = cur.fetchone()
                return row[0] if row else None
    
    def list_logs(self, run_id: str) -> list[str]:
        """List available log names for a run."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT DISTINCT name FROM {self._table('logs')}
                    WHERE run_id = %s
                """, [run_id])
                return [row[0] for row in cur.fetchall()]
    
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
        
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self._table('experiment_manifests')} (
                        experiment_id, timestamp, manifest_json, submitted_at, total_runs
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (experiment_id, timestamp) DO UPDATE SET
                        manifest_json = EXCLUDED.manifest_json,
                        total_runs = EXCLUDED.total_runs
                """, [
                    experiment_id,
                    timestamp,
                    json.dumps(manifest),
                    submitted_at,
                    total_runs,
                ])
                conn.commit()
    
    def get_experiment_manifest(self, experiment_id: str) -> dict[str, Any] | None:
        """Retrieve the most recent experiment manifest by ID."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT manifest_json FROM {self._table('experiment_manifests')}
                    WHERE experiment_id = %s
                    ORDER BY submitted_at DESC
                    LIMIT 1
                """, [experiment_id])
                row = cur.fetchone()
                if row is None:
                    return None
                return row[0] if isinstance(row[0], dict) else json.loads(row[0])
    
    # =========================================================================
    # Field catalog operations (for Atlas)
    # =========================================================================
    
    def update_field_catalog(self, run_id: str | None = None) -> None:
        """
        Update the field catalog for Atlas field index.
        
        If run_id is provided, incrementally updates from that run.
        Otherwise, rebuilds from all runs (slow).
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                # Get runs to process
                if run_id:
                    cur.execute(f"""
                        SELECT record_json FROM {self._table('runs')}
                        WHERE run_id = %s
                    """, [run_id])
                else:
                    cur.execute(f"""
                        SELECT record_json FROM {self._table('runs')}
                    """)
                
                # Collect field stats
                stats: dict[tuple[str, str], dict] = {}
                
                for (record_json,) in cur.fetchall():
                    data = record_json if isinstance(record_json, dict) else json.loads(record_json)
                    
                    # Process params
                    for key, value in data.get("params_resolved", {}).items():
                        self._update_field_stats(stats, "params", key, value)
                    
                    # Process metrics
                    for key, value in data.get("metrics", {}).items():
                        self._update_field_stats(stats, "metrics", key, value)
                
                # Upsert to field_catalog
                for (namespace, field_name), field_stats in stats.items():
                    values = list(field_stats.get("values", set()))[:100]  # Limit values
                    cur.execute(f"""
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
                    """, [
                        namespace,
                        field_name,
                        field_stats.get("type", "unknown"),
                        field_stats.get("count", 0),
                        values if values else None,
                        field_stats.get("min"),
                        field_stats.get("max"),
                    ])
                
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
        with self._conn() as conn:
            with conn.cursor() as cur:
                # Delete blobs first (foreign key)
                cur.execute(f"""
                    DELETE FROM {self._table('artifact_blobs')}
                    WHERE artifact_id IN (
                        SELECT artifact_id FROM {self._table('artifacts')}
                        WHERE run_id = %s
                    )
                """, [run_id])
                
                # Delete artifacts
                cur.execute(f"""
                    DELETE FROM {self._table('artifacts')}
                    WHERE run_id = %s
                """, [run_id])
                
                # Delete logs
                cur.execute(f"""
                    DELETE FROM {self._table('logs')}
                    WHERE run_id = %s
                """, [run_id])
                
                # Delete derived
                cur.execute(f"""
                    DELETE FROM {self._table('derived')}
                    WHERE run_id = %s
                """, [run_id])
                
                # Delete run record
                cur.execute(f"""
                    DELETE FROM {self._table('runs')}
                    WHERE run_id = %s
                """, [run_id])
                
                conn.commit()
    
    def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            self._pool.close()
