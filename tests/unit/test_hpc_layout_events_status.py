from __future__ import annotations

import json
from datetime import datetime, timedelta

from metalab.observe import (
    append_fields,
    field_label,
    format_row,
    merge_run_row,
    parse_fields,
    project_event,
    read_new_events,
    shorten_value,
)
from metalab.status import read_status
from metalab.store.events import PersistentEvent
from metalab.store.file import FileStoreConfig
from metalab.store.layout import FileStoreLayout


def test_v2_layout_uses_sharded_paths(tmp_path):
    layout = FileStoreLayout(tmp_path)
    run_id = "abcdef123456"

    assert layout.run_path(run_id) == tmp_path / "runs" / "ab" / f"{run_id}.json"
    assert layout.log_path(run_id, "run") == tmp_path / "logs" / "ab" / f"{run_id}_run.log"
    assert layout.artifact_dir(run_id) == tmp_path / "artifacts" / "ab" / run_id
    assert layout.duckdb_path() == tmp_path / "index" / "metalab.duckdb"


def test_persistent_event_roundtrip():
    event = PersistentEvent(
        kind="planned_batch",
        run_id=None,
        experiment_id="exp:1",
        job_id="job1",
        worker_id="worker1",
        payload={"count": 100},
    )

    restored = PersistentEvent.from_dict(event.to_dict())

    assert restored.kind == "planned_batch"
    assert restored.run_id is None
    assert restored.payload == {"count": 100}


def test_status_reads_events_and_heartbeats_incrementally(tmp_path):
    store = FileStoreConfig(root=str(tmp_path)).connect()
    store.write_root_manifest(
        {
            "experiment_id": "exp:1",
            "expected_run_count": 3,
            "expected_run_ids": ["r1", "r2", "r3"],
            "executor_type": "local",
            "job_id": "job1",
            "created_at": datetime.now().isoformat(),
        }
    )
    sink = store.event_sink("job1", "worker1", "exp:1")
    sink.emit("started", run_id="r1")
    sink.emit("finished", run_id="r1")
    sink.emit("failed", run_id="r2")
    store.put_heartbeat(
        job_id="job1",
        worker_id="worker1",
        experiment_id="exp:1",
        state="idle",
    )

    status = read_status(tmp_path)

    assert status.total == 3
    assert status.success == 1
    assert status.failed == 1
    assert status.pending == 1
    assert status.stale_workers == 0

    hb_path = FileStoreLayout(tmp_path).heartbeat_path("job1", "worker1")
    hb = json.loads(hb_path.read_text())
    hb["updated_at"] = (datetime.now() - timedelta(minutes=10)).isoformat()
    hb_path.write_text(json.dumps(hb))

    status = read_status(tmp_path)
    assert status.stale_workers == 1

    sink.emit("finished", run_id="r3")

    status = read_status(tmp_path)
    assert status.pending == 0
    assert status.stale_workers == 0


def test_observer_projects_fields_and_uses_offsets(tmp_path):
    store = FileStoreConfig(root=str(tmp_path)).connect()
    sink = store.event_sink("job1", "worker1", "exp:1")
    sink.emit(
        "finished",
        run_id="r1",
        payload={"params": {"x": 2}, "metrics": {"score": 4.5}},
    )

    offsets: dict[str, int] = {}
    events = list(read_new_events(tmp_path, offsets))
    assert len(events) == 1
    assert list(read_new_events(tmp_path, offsets)) == []

    row = project_event(events[0], ["kind", "run_id", "params.x", "metrics.score"])

    assert row == {
        "kind": "finished",
        "run_id": "r1",
        "params.x": 2,
        "metrics.score": 4.5,
    }
    assert "params.x=2" in format_row(row)
    assert shorten_value("run_id", "abcdef1234567890") == "abcdef123456"
    assert shorten_value("duration_ms", 12345) == "12.3s"
    assert field_label("metrics.score") == "score"


def test_observer_merges_events_into_latest_run_row():
    started = {
        "timestamp": "2026-01-01T00:00:01",
        "kind": "started",
        "run_id": "r1",
        "worker_id": "thread:1",
        "payload": {"params": {"x": 1}},
    }
    finished = {
        "timestamp": "2026-01-01T00:00:30",
        "kind": "finished",
        "run_id": "r1",
        "worker_id": "thread:1",
        "payload": {"duration_ms": 30000, "metrics": {"score": 2.5}},
    }

    row = merge_run_row(None, started)
    row = merge_run_row(row, finished)

    assert row["kind"] == "finished"
    assert row["params.x"] == 1
    assert row["metrics.score"] == 2.5
    assert row["duration_ms"] == 30000


def test_observer_merge_ignores_older_state_transitions():
    finished = {
        "timestamp": "2026-01-01T00:00:30",
        "kind": "finished",
        "run_id": "r1",
        "worker_id": "thread:1",
        "payload": {
            "duration_ms": 30000,
            "params": {"x": 1},
            "metrics": {"score": 2.5},
        },
    }
    stale_started = {
        "timestamp": "2026-01-01T00:00:01",
        "kind": "started",
        "run_id": "r1",
        "worker_id": "thread:2",
        "payload": {"params": {"x": 1}},
    }

    row = merge_run_row(None, finished)
    row = merge_run_row(row, stale_started)

    assert row["kind"] == "finished"
    assert row["worker_id"] == "thread:1"
    assert row["metrics.score"] == 2.5


def test_observer_field_aliases_and_presets():
    assert parse_fields("event,run,worker,score,dur") == [
        "kind",
        "run_id",
        "worker_id",
        "metrics.score",
        "duration_ms",
    ]
    assert parse_fields("basic") == ["kind", "run_id", "worker_id", "duration_ms"]
    assert parse_fields("default") == ["kind", "run_id", "worker_id", "duration_ms"]
    assert append_fields(parse_fields("default"), ["params.x", "metrics.score"]) == [
        "kind",
        "run_id",
        "worker_id",
        "duration_ms",
        "params.x",
        "metrics.score",
    ]
