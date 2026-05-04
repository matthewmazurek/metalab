"""Lightweight structured progress observer for event shards."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from metalab.store.events import iter_event_files

BASIC_FIELDS = [
    "kind",
    "run_id",
    "worker_id",
    "duration_ms",
]
DEFAULT_FIELDS = BASIC_FIELDS
DEFAULT_BY_RUN_FIELDS = BASIC_FIELDS
SMOKE_FIELDS = [
    *BASIC_FIELDS,
    "params.x",
    "params.scale",
    "metrics.score",
]
FIELD_ALIASES = {
    "time": "timestamp",
    "event": "kind",
    "run": "run_id",
    "worker": "worker_id",
    "dur": "duration_ms",
    "duration": "duration_ms",
    "error": "error_message",
    "score": "metrics.score",
    "x": "params.x",
    "scale": "params.scale",
}
FIELD_PRESETS = {
    "default": BASIC_FIELDS,
    "basic": BASIC_FIELDS,
    "timing": BASIC_FIELDS,
    "smoke": SMOKE_FIELDS,
    "errors": [
        *BASIC_FIELDS,
        "error_type",
        "error_message",
    ],
}

FIELD_LABELS = {
    "timestamp": "time",
    "kind": "event",
    "run_id": "run",
    "worker_id": "worker",
    "duration_ms": "dur",
}


def parse_fields(fields: str | None) -> list[str]:
    """Parse a comma-separated field list."""
    if not fields:
        return DEFAULT_FIELDS
    if fields in FIELD_PRESETS:
        return list(FIELD_PRESETS[fields])
    return [
        FIELD_ALIASES.get(field.strip(), field.strip())
        for field in fields.split(",")
        if field.strip()
    ]


def append_fields(base: list[str], extra: list[str]) -> list[str]:
    """Append fields without duplicating columns."""
    result = list(base)
    seen = set(result)
    for field in extra:
        if field not in seen:
            result.append(field)
            seen.add(field)
    return result


def _get_path(data: dict[str, Any], field: str) -> Any:
    """Read dotted fields from event or payload data."""
    if field in data:
        return data[field]
    current: Any = data
    for part in field.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def project_event(event: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Project an event to user-requested fields."""
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    merged = {**payload, **event}
    return {field: _get_path(merged, field) for field in fields}


def merge_run_row(existing: dict[str, Any] | None, event: dict[str, Any]) -> dict[str, Any]:
    """Merge one event into the latest per-run observer row."""
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    row = dict(existing or {})
    timestamp = str(event.get("timestamp", ""))
    previous_timestamp = str(row.get("timestamp", ""))
    if previous_timestamp and timestamp < previous_timestamp:
        return row
    row.update(
        {
            "timestamp": timestamp,
            "kind": event.get("kind"),
            "run_id": event.get("run_id"),
            "worker_id": event.get("worker_id"),
        }
    )
    if "duration_ms" in payload:
        row["duration_ms"] = payload["duration_ms"]
    for namespace in ("params", "metrics"):
        values = payload.get(namespace)
        if isinstance(values, dict):
            for key, value in values.items():
                row[f"{namespace}.{key}"] = value
    if "error_type" in payload:
        row["error_type"] = payload["error_type"]
    if "error_message" in payload:
        row["error_message"] = payload["error_message"]
    return row


def format_row(row: dict[str, Any]) -> str:
    """Format one projected event as tab-separated key=value pairs."""
    parts = []
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            value_text = json.dumps(value, sort_keys=True)
        else:
            value_text = "" if value is None else str(value)
        parts.append(f"{key}={value_text}")
    return "\t".join(parts)


def format_value(value: Any) -> str:
    """Format one value for human display."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def shorten_value(field: str, value: Any) -> str:
    """Format compact values for pretty observer tables."""
    if value is None:
        return ""
    if field == "timestamp" and isinstance(value, str):
        try:
            return datetime.fromisoformat(value).strftime("%H:%M:%S")
        except ValueError:
            return value
    if field == "run_id" and isinstance(value, str):
        return value[:12]
    if field == "kind" and value == "skipped":
        return "skip"
    if field == "duration_ms" and isinstance(value, int | float):
        return f"{value / 1000:.1f}s"
    return format_value(value)


def field_label(field: str) -> str:
    """Return a compact display label for a field name."""
    if field in FIELD_LABELS:
        return FIELD_LABELS[field]
    if field.startswith("params."):
        return field.removeprefix("params.")
    if field.startswith("metrics."):
        return field.removeprefix("metrics.")
    return field


def read_new_events(root: str | Path, offsets: dict[str, int]) -> Iterator[dict[str, Any]]:
    """Yield new JSON events and update byte offsets in-place."""
    root_path = Path(root)
    for path in iter_event_files(root_path):
        key = str(path.relative_to(root_path))
        offset = offsets.get(key, 0)
        try:
            with path.open("rb") as f:
                f.seek(offset)
                for raw in f:
                    try:
                        yield json.loads(raw.decode("utf-8"))
                    except Exception:
                        continue
                offsets[key] = f.tell()
        except FileNotFoundError:
            continue


def observe_events(
    root: str | Path,
    *,
    fields: list[str] | None = None,
    interval: float = 2.0,
    once: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield projected events from a store, optionally polling forever."""
    selected = fields or DEFAULT_FIELDS
    offsets: dict[str, int] = {}
    while True:
        yield from (
            project_event(event, selected) for event in read_new_events(root, offsets)
        )
        if once:
            return
        time.sleep(interval)
