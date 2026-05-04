"""DuckDB sidecar indexing, summary, and export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from metalab.schema import load_run_record
from metalab.store.events import iter_events
from metalab.store.layout import FileStoreLayout

DUCKDB_HINT = "Install DuckDB support with: uv sync --extra hpc"


def _duckdb():
    try:
        import duckdb
    except ImportError as e:
        raise RuntimeError(DUCKDB_HINT) from e
    return duckdb


def _iter_run_files(root: Path) -> list[Path]:
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return []
    return sorted(runs_dir.glob("*/*.json"))


def rebuild_index(store_root: str | Path, *, force: bool = False) -> Path:
    """Rebuild the DuckDB sidecar index from canonical run records and events."""
    duckdb = _duckdb()
    root = Path(store_root)
    layout = FileStoreLayout(root)
    db_path = layout.duckdb_path()
    if db_path.exists() and force:
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_path))
    conn.execute("DROP TABLE IF EXISTS runs")
    conn.execute("DROP TABLE IF EXISTS fields")
    conn.execute("DROP TABLE IF EXISTS events")
    conn.execute(
        """
        CREATE TABLE runs (
            run_id VARCHAR PRIMARY KEY,
            experiment_id VARCHAR,
            status VARCHAR,
            started_at TIMESTAMP,
            finished_at TIMESTAMP,
            duration_ms BIGINT,
            error_type VARCHAR,
            error_message VARCHAR,
            params_json JSON,
            metrics_json JSON,
            record_path VARCHAR
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE fields (
            run_id VARCHAR,
            namespace VARCHAR,
            field_name VARCHAR,
            field_value VARCHAR,
            numeric_value DOUBLE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE events (
            event_id VARCHAR,
            kind VARCHAR,
            run_id VARCHAR,
            experiment_id VARCHAR,
            job_id VARCHAR,
            worker_id VARCHAR,
            timestamp TIMESTAMP,
            payload_json JSON
        )
        """
    )

    for path in _iter_run_files(root):
        data = json.loads(path.read_text(encoding="utf-8"))
        record = load_run_record(data)
        error = record.error or {}
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                record.run_id,
                record.experiment_id,
                record.status.value,
                record.started_at,
                record.finished_at,
                record.duration_ms,
                error.get("type"),
                error.get("message"),
                json.dumps(record.params_resolved),
                json.dumps(record.metrics),
                str(path),
            ],
        )
        for namespace, values in (
            ("params", record.params_resolved),
            ("metrics", record.metrics),
        ):
            for key, value in values.items():
                numeric = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
                conn.execute(
                    "INSERT INTO fields VALUES (?, ?, ?, ?, ?)",
                    [record.run_id, namespace, key, str(value), numeric],
                )

    for event in iter_events(root):
        conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                event.event_id,
                event.kind,
                event.run_id,
                event.experiment_id,
                event.job_id,
                event.worker_id,
                event.timestamp,
                json.dumps(event.payload),
            ],
        )

    conn.close()
    return db_path


def summary(
    store_root: str | Path,
    *,
    group_by: str | None = None,
    metric: str | None = None,
) -> list[dict[str, Any]]:
    """Return a compact summary from the DuckDB sidecar index."""
    duckdb = _duckdb()
    db_path = FileStoreLayout(Path(store_root)).duckdb_path()
    if not db_path.exists():
        rebuild_index(store_root)
    conn = duckdb.connect(str(db_path), read_only=True)
    if group_by and metric:
        ns, _, name = group_by.partition(".")
        m_ns, _, m_name = metric.partition(".")
        rows = conn.execute(
            """
            SELECT g.field_value AS group_value, COUNT(*) AS n, AVG(m.numeric_value) AS mean_metric
            FROM fields g
            JOIN fields m USING (run_id)
            JOIN runs r USING (run_id)
            WHERE g.namespace = ? AND g.field_name = ?
              AND m.namespace = ? AND m.field_name = ?
              AND r.status = 'success'
            GROUP BY g.field_value
            ORDER BY group_value
            """,
            [ns, name, m_ns, m_name],
        ).fetchall()
        columns = ["group_value", "n", "mean_metric"]
    else:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM runs GROUP BY status ORDER BY status"
        ).fetchall()
        columns = ["status", "n"]
    conn.close()
    return [dict(zip(columns, row)) for row in rows]


def export(
    store_root: str | Path,
    *,
    fmt: str,
    out: str | Path,
) -> Path:
    """Export indexed runs to CSV, Parquet, or JSONL."""
    duckdb = _duckdb()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    db_path = FileStoreLayout(Path(store_root)).duckdb_path()
    if not db_path.exists():
        rebuild_index(store_root)
    conn = duckdb.connect(str(db_path), read_only=True)
    query = "SELECT * FROM runs ORDER BY started_at"
    if fmt == "csv":
        conn.execute(f"COPY ({query}) TO ? (HEADER, DELIMITER ',')", [str(out_path)])
    elif fmt == "parquet":
        conn.execute(f"COPY ({query}) TO ? (FORMAT PARQUET)", [str(out_path)])
    elif fmt == "jsonl":
        cursor = conn.execute(query)
        cols = [d[0] for d in cursor.description]
        with out_path.open("w", encoding="utf-8", newline="") as f:
            while rows := cursor.fetchmany(1000):
                for row in rows:
                    f.write(json.dumps(dict(zip(cols, row)), default=str) + "\n")
    else:
        raise ValueError("format must be one of: csv, parquet, jsonl")
    conn.close()
    return out_path
