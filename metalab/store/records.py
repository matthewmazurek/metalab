"""Shared readers for canonical v4 run-record shards."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Generator

from metalab.schema import load_run_record
from metalab.store.layout import FileStoreLayout
from metalab.types import RunRecord

logger = logging.getLogger(__name__)


def iter_ndjson_rows(path: Path) -> Generator[dict[str, Any], None, None]:
    """Yield JSON object rows from an NDJSON file, skipping malformed rows."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception as e:
                logger.warning(f"Failed to load row {path}:{line_no}: {e}")
                continue
            if isinstance(data, dict):
                yield data


def latest_run_index_entries(layout: FileStoreLayout) -> dict[str, dict[str, Any]]:
    """Return latest run index entry per run id by sequence."""
    latest: dict[str, dict[str, Any]] = {}
    paths = [layout.shard_map_path()]
    if not paths[0].exists():
        paths = sorted(layout.record_shards_dir_path().glob("*.idx"))
    for path in paths:
        for row in iter_ndjson_rows(path):
            run_id = row.get("run_id")
            if not isinstance(run_id, str):
                continue
            prev = latest.get(run_id)
            if prev is None or int(row.get("sequence", 0)) >= int(
                prev.get("sequence", 0)
            ):
                latest[run_id] = row
    return latest


def latest_run_index_entry(
    layout: FileStoreLayout,
    run_id: str,
) -> dict[str, Any] | None:
    """Return latest run index entry for one run id."""
    latest: dict[str, Any] | None = None
    for row in iter_ndjson_rows(layout.record_shard_index_path(run_id)):
        if row.get("run_id") != run_id:
            continue
        if latest is None or int(row.get("sequence", 0)) >= int(
            latest.get("sequence", 0)
        ):
            latest = row
    return latest


def read_run_at_index(
    layout: FileStoreLayout,
    row: dict[str, Any],
) -> tuple[RunRecord, str] | None:
    """Read a run record using a byte index row and return it with a record ref."""
    shard_id = row.get("shard_id")
    if not isinstance(shard_id, str):
        return None
    path = layout.record_shards_dir_path() / f"{shard_id}.ndjson"
    try:
        with path.open("rb") as handle:
            handle.seek(int(row["offset"]))
            payload = handle.read(int(row["length"]))
        record = load_run_record(json.loads(payload.decode("utf-8")))
    except Exception as e:
        logger.warning(f"Failed to load run record from {path}: {e}")
        return None
    ref = f"{path}:{row['offset']}:{row['length']}"
    return record, ref


def iter_run_records(
    root: Path,
) -> Generator[tuple[RunRecord, str], None, None]:
    """Stream latest canonical run records from v4 record shards."""
    layout = FileStoreLayout(root)
    for row in latest_run_index_entries(layout).values():
        record_ref = read_run_at_index(layout, row)
        if record_ref is not None:
            yield record_ref
