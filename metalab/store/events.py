"""Persistent filesystem events for HPC progress and recovery."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from metalab.store.layout import FileStoreLayout

EVENT_KINDS = {
    "planned_batch",
    "started",
    "finished",
    "failed",
    "skipped",
    "heartbeat",
}


@dataclass(frozen=True)
class PersistentEvent:
    """A JSON-serializable event written by workers."""

    kind: str
    run_id: str | None
    experiment_id: str
    job_id: str
    worker_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"Unknown persistent event kind: {self.kind}")
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "job_id": self.job_id,
            "worker_id": self.worker_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersistentEvent":
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            event_id=data["event_id"],
            kind=data["kind"],
            run_id=data.get("run_id"),
            experiment_id=data.get("experiment_id", ""),
            job_id=data.get("job_id", ""),
            worker_id=data.get("worker_id", ""),
            timestamp=timestamp or datetime.now(),
            payload=data.get("payload", {}),
        )


class FileEventSink:
    """Append-only event writer scoped to one job and worker."""

    def __init__(self, path: Path, *, experiment_id: str, job_id: str, worker_id: str):
        self.path = path
        self.experiment_id = experiment_id
        self.job_id = job_id
        self.worker_id = worker_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        kind: str,
        *,
        run_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> PersistentEvent:
        event = PersistentEvent(
            kind=kind,
            run_id=run_id,
            experiment_id=self.experiment_id,
            job_id=self.job_id,
            worker_id=self.worker_id,
            payload=payload or {},
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        return event


def iter_event_files(root: Path) -> list[Path]:
    """Return all event shard files in deterministic order."""
    events_root = FileStoreLayout(root).events_dir_path()
    if not events_root.exists():
        return []
    return sorted(events_root.glob("*/*.ndjson"))


def tail_event_rows(root: Path, offsets: dict[str, int]) -> Iterator[dict[str, Any]]:
    """
    Yield new event rows from shard files and update byte offsets in-place.

    This is the shared low-level primitive behind live observation and compact
    status aggregation.  Callers own the interpretation of event rows.
    """
    for path in iter_event_files(root):
        key = str(path.relative_to(root))
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


def iter_events(root: Path) -> list[PersistentEvent]:
    """Load persistent events from a run store."""
    events: list[PersistentEvent] = []
    for path in iter_event_files(root):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(PersistentEvent.from_dict(json.loads(line)))
                except Exception:
                    continue
    return events
