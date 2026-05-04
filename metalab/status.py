"""Status aggregation from filesystem manifests, events, and heartbeats."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from metalab.store.events import iter_event_files


@dataclass
class StoreStatus:
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    running: int = 0
    pending: int = 0
    stale_workers: int = 0
    workers: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "skipped": self.skipped,
            "running": self.running,
            "pending": self.pending,
            "stale_workers": self.stale_workers,
            "workers": self.workers,
        }


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_status(
    store_root: str | Path,
    *,
    use_cache: bool = True,
    stale_after: timedelta = timedelta(minutes=5),
) -> StoreStatus:
    """Compute run-store status without scanning canonical run records."""
    root = Path(store_root)
    manifest = _load_json(root / "manifest.json") or {}
    total = int(manifest.get("expected_run_count") or manifest.get("total_runs") or 0)

    cache_path = root / "index" / "status-cache.json"
    cache = _load_json(cache_path) if use_cache else None
    offsets: dict[str, int] = {}
    per_run: dict[str, dict[str, str]] = {}
    if cache:
        offsets = {k: int(v) for k, v in cache.get("offsets", {}).items()}
        for run_id, state in dict(cache.get("per_run", {})).items():
            if isinstance(state, str):
                per_run[run_id] = {"kind": state, "timestamp": ""}
            elif isinstance(state, dict):
                per_run[run_id] = {
                    "kind": str(state.get("kind", "")),
                    "timestamp": str(state.get("timestamp", "")),
                }

    new_offsets = dict(offsets)
    for path in iter_event_files(root):
        key = str(path.relative_to(root))
        offset = offsets.get(key, 0)
        try:
            with path.open("rb") as f:
                f.seek(offset)
                for raw in f:
                    try:
                        event = json.loads(raw.decode("utf-8"))
                    except Exception:
                        continue
                    run_id = event.get("run_id")
                    kind = event.get("kind")
                    if run_id and kind in {
                        "started",
                        "finished",
                        "failed",
                        "skipped",
                    }:
                        timestamp = str(event.get("timestamp", ""))
                        previous = per_run.get(run_id)
                        if previous is None or timestamp >= previous.get("timestamp", ""):
                            per_run[run_id] = {"kind": kind, "timestamp": timestamp}
                new_offsets[key] = f.tell()
        except FileNotFoundError:
            continue

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "layout_version": 2,
                    "updated_at": datetime.now().isoformat(),
                    "offsets": new_offsets,
                    "per_run": per_run,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    success = sum(1 for state in per_run.values() if state.get("kind") == "finished")
    failed = sum(1 for state in per_run.values() if state.get("kind") == "failed")
    skipped = sum(1 for state in per_run.values() if state.get("kind") == "skipped")
    running = sum(1 for state in per_run.values() if state.get("kind") == "started")
    done = success + failed + skipped + running
    pending = max(0, total - done)
    has_active_work = running > 0 or pending > 0

    workers = []
    stale = 0
    now = datetime.now()
    hb_root = root / "heartbeats"
    for path in sorted(hb_root.glob("*/*.json")) if hb_root.exists() else []:
        data = _load_json(path)
        if not data:
            continue
        updated_at = datetime.fromisoformat(data["updated_at"])
        is_stale = has_active_work and now - updated_at > stale_after
        stale += int(is_stale)
        workers.append({**data, "stale": is_stale})

    return StoreStatus(
        total=total,
        success=success,
        failed=failed,
        skipped=skipped,
        running=running,
        pending=pending,
        stale_workers=stale,
        workers=workers,
    )
