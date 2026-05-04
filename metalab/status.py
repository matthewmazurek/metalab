"""Status aggregation from filesystem manifests, events, and heartbeats."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from metalab.store.events import iter_event_files

KIND_TO_CODE = {
    "started": "r",
    "finished": "s",
    "failed": "f",
    "skipped": "k",
}
CODE_TO_KIND = {code: kind for kind, code in KIND_TO_CODE.items()}


class RunStoreNotFoundError(ValueError):
    """Raised when a path does not look like a metalab run store."""


@dataclass
class StoreStatus:
    total: int = 0
    success: int = 0
    failed: int = 0
    running: int = 0
    pending: int = 0
    stale_workers: int = 0
    workers: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
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


def validate_run_store(store_root: str | Path) -> Path:
    """Return a run-store path or raise a clear user-facing error."""
    root = Path(store_root)
    manifest_path = root / "manifest.json"
    if not root.exists():
        raise RunStoreNotFoundError(
            f"No metalab run store found at {root}. The path does not exist. "
            "Run `metalab run ... --store PATH` first or pass the correct store path."
        )
    if not root.is_dir():
        raise RunStoreNotFoundError(
            f"No metalab run store found at {root}. Expected a directory containing "
            "manifest.json."
        )
    if not manifest_path.exists():
        raise RunStoreNotFoundError(
            f"No metalab run store found at {root}. Expected manifest.json. "
            "Run `metalab run ... --store PATH` first or pass the correct store path."
        )
    manifest = _load_json(manifest_path)
    if not manifest:
        raise RunStoreNotFoundError(
            f"Malformed metalab run store at {root}. Could not read manifest.json."
        )
    return root


def _decode_cached_state(state: Any) -> dict[str, str] | None:
    """Decode legacy or compact cached per-run state."""
    if isinstance(state, str):
        return {"kind": CODE_TO_KIND.get(state, state), "timestamp": ""}
    if isinstance(state, list) and state:
        return {
            "kind": CODE_TO_KIND.get(str(state[0]), str(state[0])),
            "timestamp": str(state[1]) if len(state) > 1 else "",
        }
    if isinstance(state, dict):
        return {
            "kind": CODE_TO_KIND.get(str(state.get("kind", "")), str(state.get("kind", ""))),
            "timestamp": str(state.get("timestamp", "")),
        }
    return None


def _encode_cached_state(state: dict[str, str]) -> list[str]:
    """Encode cached per-run state compactly."""
    return [KIND_TO_CODE.get(state.get("kind", ""), state.get("kind", "")), state.get("timestamp", "")]


def _merge_run_state(
    previous: dict[str, str] | None,
    *,
    kind: str,
    timestamp: str,
) -> dict[str, str]:
    """Merge an event kind into per-run state.

    A skipped event means "already successful, not executed in this submission".
    It must not replace a known successful canonical state from an earlier
    finished event, or status would report completed runs as no longer success.
    """
    if previous is None:
        return {
            "kind": "finished" if kind == "skipped" else kind,
            "timestamp": timestamp,
        }
    if timestamp < previous.get("timestamp", ""):
        return previous
    if kind == "skipped" and previous.get("kind") == "finished":
        return previous
    return {"kind": "finished" if kind == "skipped" else kind, "timestamp": timestamp}


def read_status(
    store_root: str | Path,
    *,
    use_cache: bool = True,
    stale_after: timedelta = timedelta(minutes=5),
) -> StoreStatus:
    """Compute run-store status without scanning canonical run records."""
    root = validate_run_store(store_root)
    manifest = _load_json(root / "manifest.json") or {}
    total = int(manifest.get("expected_run_count") or manifest.get("total_runs") or 0)

    cache_path = root / "index" / "status-cache.json"
    cache = _load_json(cache_path) if use_cache else None
    offsets: dict[str, int] = {}
    per_run: dict[str, dict[str, str]] = {}
    if cache:
        offsets = {k: int(v) for k, v in cache.get("offsets", {}).items()}
        for run_id, state in dict(cache.get("per_run", {})).items():
            decoded = _decode_cached_state(state)
            if decoded is not None:
                per_run[run_id] = decoded

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
                        per_run[run_id] = _merge_run_state(
                            per_run.get(run_id),
                            kind=kind,
                            timestamp=timestamp,
                        )
                new_offsets[key] = f.tell()
        except FileNotFoundError:
            continue

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "layout_version": 3,
                    "format": "compact-v1",
                    "updated_at": datetime.now().isoformat(),
                    "offsets": new_offsets,
                    "per_run": {
                        run_id: _encode_cached_state(state)
                        for run_id, state in per_run.items()
                    },
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

    success = sum(1 for state in per_run.values() if state.get("kind") == "finished")
    failed = sum(1 for state in per_run.values() if state.get("kind") == "failed")
    running = sum(1 for state in per_run.values() if state.get("kind") == "started")
    done = success + failed + running
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
        running=running,
        pending=pending,
        stale_workers=stale,
        workers=workers,
    )
