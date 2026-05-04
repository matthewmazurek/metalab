"""DuckDB sidecar indexing, summary, and export helpers."""

from __future__ import annotations

import csv
import json
import shutil
import tarfile
import tempfile
import warnings
from pathlib import Path
from typing import Any

from metalab.store.events import iter_event_files, iter_events
from metalab.store.layout import FileStoreLayout
from metalab.store.records import iter_ndjson_rows, iter_run_records
from metalab.types import Status

DUCKDB_HINT = "Install DuckDB support with: uv sync --extra hpc"
DATASET_HINT = (
    "Install dataset export support with: uv sync --extra dataset "
    "(or install anndata, pandas, numpy, and zarr)"
)
ANNDATA_ZARR_V2_WARNING = (
    "Writing zarr v2 data will no longer be the default in the next minor release."
)


def _duckdb():
    try:
        import duckdb
    except ImportError as e:
        raise RuntimeError(DUCKDB_HINT) from e
    return duckdb


def _anndata_stack():
    try:
        import anndata as ad
        import numpy as np
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(DATASET_HINT) from e
    return ad, np, pd


def _decode_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return {}


def _flat_rows_from_index(store_root: str | Path, *, successful_only: bool = False) -> list[dict[str, Any]]:
    duckdb = _duckdb()
    db_path = FileStoreLayout(Path(store_root)).duckdb_path()
    if not index_is_current(store_root):
        rebuild_index(store_root)
    conn = duckdb.connect(str(db_path), read_only=True)
    where = "WHERE status = 'success'" if successful_only else ""
    try:
        rows = conn.execute(
            f"""
            SELECT run_id, experiment_id, status, duration_ms, started_at, finished_at,
                   error_type, error_message, record_path, params_json, metrics_json
            FROM runs
            {where}
            ORDER BY started_at
            """
        ).fetchall()
    finally:
        conn.close()

    table_rows: list[dict[str, Any]] = []
    for row in rows:
        params_json = _decode_json(row[9])
        metrics_json = _decode_json(row[10])
        table_rows.append(
            {
                "run_id": row[0],
                "experiment_id": row[1],
                "status": row[2],
                "duration_ms": row[3],
                "started_at": str(row[4]),
                "finished_at": str(row[5]),
                "error_type": row[6],
                "error_message": row[7],
                "record_path": row[8],
                **{f"params.{key}": value for key, value in params_json.items()},
                **{f"metrics.{key}": value for key, value in metrics_json.items()},
            }
        )
    return table_rows


def _infer_table_format(out_path: Path, fmt: str | None = None) -> str:
    if fmt:
        return fmt
    suffix = out_path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    raise ValueError("table export format must be csv, parquet, or jsonl")


def _write_table_rows(rows: list[dict[str, Any]], out_path: Path, *, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    if fmt == "csv":
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    elif fmt == "jsonl":
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")
    elif fmt == "parquet":
        duckdb = _duckdb()
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            tmp_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")
        try:
            conn = duckdb.connect()
            conn.execute(
                "COPY (SELECT * FROM read_json_auto(?)) TO ? (FORMAT PARQUET)",
                [str(tmp_path), str(out_path)],
            )
            conn.close()
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        raise ValueError("table export format must be csv, parquet, or jsonl")


def _event_offsets(root: Path) -> dict[str, int]:
    """Return event file byte offsets without reading event contents."""
    offsets: dict[str, int] = {}
    for path in iter_event_files(root):
        offsets[str(path.relative_to(root))] = path.stat().st_size
    return offsets


def _shard_offsets(root: Path) -> dict[str, int]:
    """Return shard/index byte sizes without reading full contents."""
    layout = FileStoreLayout(root)
    offsets: dict[str, int] = {}
    for base in [
        layout.record_shards_dir_path(),
        layout.outputs_dir_path(),
        layout.index_dir_path(),
    ]:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.ndjson")) + sorted(base.rglob("*.idx")):
            offsets[str(path.relative_to(root))] = path.stat().st_size
    return offsets


def _source_manifest_mtime(root: Path) -> float | None:
    manifest_path = FileStoreLayout(root).root_manifest_path()
    if not manifest_path.exists():
        return None
    return manifest_path.stat().st_mtime


def _write_index_meta(root: Path, *, run_count: int) -> None:
    """Write cheap freshness metadata for the DuckDB sidecar."""
    layout = FileStoreLayout(root)
    meta = {
        "schema_version": 1,
        "indexed_at": layout.duckdb_path().stat().st_mtime,
        "indexed_run_count": run_count,
        "source_manifest_mtime": _source_manifest_mtime(root),
        "source_event_offsets": _event_offsets(root),
        "source_shard_offsets": _shard_offsets(root),
    }
    tmp = layout.duckdb_meta_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    tmp.rename(layout.duckdb_meta_path())


def index_is_current(store_root: str | Path) -> bool:
    """Return true when the sidecar index matches cheap source freshness signals."""
    root = Path(store_root)
    layout = FileStoreLayout(root)
    if not layout.duckdb_path().exists() or not layout.duckdb_meta_path().exists():
        return False
    try:
        meta = json.loads(layout.duckdb_meta_path().read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        meta.get("schema_version") == 1
        and meta.get("source_manifest_mtime") == _source_manifest_mtime(root)
        and meta.get("source_event_offsets") == _event_offsets(root)
        and meta.get("source_shard_offsets") == _shard_offsets(root)
    )


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

    run_count = 0
    for record, record_ref in iter_run_records(root):
        run_count += 1
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
                record_ref,
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
    _write_index_meta(root, run_count=run_count)
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
    if not index_is_current(store_root):
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


def export_table(
    store_root: str | Path,
    *,
    out: str | Path,
    fmt: str | None = None,
    successful_only: bool = False,
) -> Path:
    """Export indexed runs as a flat table with run, params, and metrics columns."""
    out_path = Path(out)
    table_format = _infer_table_format(out_path, fmt)
    rows = _flat_rows_from_index(store_root, successful_only=successful_only)
    _write_table_rows(rows, out_path, fmt=table_format)
    return out_path


def export_snapshot(store_root: str | Path, *, out: str | Path) -> Path:
    """Export a point-in-time DuckDB snapshot of the sidecar index."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not index_is_current(store_root):
        rebuild_index(store_root)
    db_path = FileStoreLayout(Path(store_root)).duckdb_path()
    if out_path.resolve() == db_path.resolve():
        raise ValueError("snapshot output must not be the live sidecar index path")
    shutil.copy2(db_path, out_path)
    return out_path


def export_archive(store_root: str | Path, *, out: str | Path) -> Path:
    """Export the run store as a tar archive."""
    root = Path(store_root)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_resolved = out_path.resolve()
    with tarfile.open(out_path, "w") as archive:
        for path in sorted(root.rglob("*")):
            if path.resolve() == out_resolved:
                continue
            archive.add(path, arcname=path.relative_to(root))
    return out_path


def _latest_result_rows(layout: FileStoreLayout, run_id: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in iter_ndjson_rows(layout.output_shard_path("results", run_id)):
        if row.get("run_id") != run_id:
            continue
        name = row.get("name")
        if not isinstance(name, str):
            continue
        prev = latest.get(name)
        if prev is None or int(row.get("sequence", 0)) >= int(prev.get("sequence", 0)):
            latest[name] = row
    return latest


def _experiment_uns(root: Path, records: list[Any]) -> dict[str, Any]:
    """Return self-describing experiment metadata for AnnData/Zarr exports."""
    layout = FileStoreLayout(root)
    root_manifest = {}
    if layout.root_manifest_path().exists():
        root_manifest = json.loads(layout.root_manifest_path().read_text(encoding="utf-8"))

    experiment_id = root_manifest.get("experiment_id")
    if experiment_id is None and records:
        experiment_id = records[0].experiment_id

    latest_submission: dict[str, Any] = {}
    for row in iter_ndjson_rows(layout.submissions_path()):
        manifest = row.get("manifest")
        if row.get("experiment_id") == experiment_id and isinstance(manifest, dict):
            latest_submission = manifest

    return {
        "experiment_id": experiment_id,
        "name": latest_submission.get("name"),
        "version": latest_submission.get("version"),
        "description": latest_submission.get("description"),
        "tags": latest_submission.get("tags", []),
        "metadata": latest_submission.get("metadata", {}),
    }


def export_dataset(store_root: str | Path, *, out: str | Path) -> Path:
    """Export successful observation-oriented runs as AnnData/Zarr."""
    ad, np, pd = _anndata_stack()
    root = Path(store_root)
    layout = FileStoreLayout(root)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = [record for record, _ in iter_run_records(root) if record.status == Status.SUCCESS]
    obs_rows = []
    result_values: dict[str, list[Any]] = {}
    result_shapes: dict[str, list[Any]] = {}
    for record in records:
        obs_rows.append(
            {
                "run_id": record.run_id,
                "experiment_id": record.experiment_id,
                "duration_ms": record.duration_ms,
                "started_at": record.started_at.isoformat(),
                "finished_at": record.finished_at.isoformat(),
                "context_fingerprint": record.context_fingerprint,
                "params_fingerprint": record.params_fingerprint,
                "seed_fingerprint": record.seed_fingerprint,
                **{f"params.{key}": value for key, value in record.params_resolved.items()},
                **{f"metrics.{key}": value for key, value in record.metrics.items()},
            }
        )
        for name, row in _latest_result_rows(layout, record.run_id).items():
            result = row.get("result")
            if not isinstance(result, dict):
                continue
            result_values.setdefault(name, []).append(result.get("data"))
            result_shapes.setdefault(name, []).append(result.get("shape"))

    obs = pd.DataFrame(obs_rows)
    if "run_id" in obs:
        obs = obs.set_index("run_id", drop=False)
    adata = ad.AnnData(obs=obs)
    adata.uns["metalab"] = {
        "schema_version": 1,
        "source_store": str(root),
        "source_is_run_store": True,
        "successful_runs": len(records),
        "experiment": _experiment_uns(root, records),
        "export": {
            "target": "dataset",
            "format": "anndata-zarr",
            "zarr_format": 2,
        },
        "capture": {
            "obsm": {},
            "skipped": {},
        },
    }

    skipped_results: dict[str, str] = {}
    for name, values in result_values.items():
        if len(values) != len(records):
            skipped_results[name] = "not present for every successful run"
            continue
        try:
            stacked = np.stack([np.asarray(value) for value in values])
        except Exception:
            skipped_results[name] = "values are not stackable arrays"
            continue
        if stacked.ndim < 2:
            skipped_results[name] = "stacked result is not observation-aligned"
            continue
        adata.obsm[name] = stacked.reshape((len(records), -1))
        original_shape = result_shapes.get(name, [None])[0] or list(stacked.shape[1:])
        adata.uns["metalab"]["capture"]["obsm"][name] = {
            "stored_in": "obsm",
            "original_shape": original_shape,
            "stacked_shape": list(stacked.shape),
        }
    if skipped_results:
        adata.uns["metalab"]["capture"]["skipped"] = skipped_results

    from anndata._settings import settings as anndata_settings

    previous_zarr_format = anndata_settings.zarr_write_format
    try:
        anndata_settings.zarr_write_format = 2
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=f"^{ANNDATA_ZARR_V2_WARNING}",
                category=UserWarning,
                module="anndata._io.zarr",
            )
            adata.write_zarr(out_path)
    finally:
        anndata_settings.zarr_write_format = previous_zarr_format
    return out_path


def export(
    store_root: str | Path,
    *,
    fmt: str,
    out: str | Path,
) -> Path:
    """Backward-compatible table export to CSV, Parquet, or JSONL."""
    return export_table(store_root, fmt=fmt, out=out)


def export_target(
    store_root: str | Path,
    *,
    target: str,
    out: str | Path,
    fmt: str | None = None,
) -> Path:
    """Export a run store to one of MetaLab's typed targets."""
    if target == "table":
        return export_table(store_root, out=out, fmt=fmt)
    if target == "snapshot":
        return export_snapshot(store_root, out=out)
    if target == "dataset":
        return export_dataset(store_root, out=out)
    if target == "archive":
        return export_archive(store_root, out=out)
    raise ValueError("export target must be one of: table, snapshot, dataset, archive")
