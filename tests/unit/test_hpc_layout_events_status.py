from __future__ import annotations

import json
from datetime import datetime, timedelta

from metalab.types import ArtifactDescriptor, RunRecord, Status
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
from metalab.status import RunStoreNotFoundError, read_status
from metalab.store.events import PersistentEvent
from metalab.store.file import FileStoreConfig
from metalab.store.layout import FileStoreLayout


def test_v3_layout_uses_hash_sharded_paths(tmp_path):
    layout = FileStoreLayout(tmp_path)
    run_id = "abcdef123456"
    shard_id = layout.shard_id(run_id)

    assert layout.run_path(run_id) == tmp_path / "runs" / "shards" / f"{shard_id}.ndjson"
    assert layout.run_shard_index_path(run_id) == tmp_path / "runs" / "shards" / f"{shard_id}.idx"
    assert layout.log_path(run_id, "run") == tmp_path / "metadata" / "logs" / f"{shard_id}.ndjson"
    assert layout.scratch_log_path(run_id, "run") == tmp_path / ".scratch" / "logs" / shard_id / f"{run_id}_run.log"
    assert layout.artifact_dir(run_id) == tmp_path / "artifacts" / "ab" / run_id
    assert layout.duckdb_path() == tmp_path / "index" / "metalab.duckdb"
    assert layout.planned_runs_path("ab") == tmp_path / "index" / "planned-runs" / "ab.ndjson"


def test_v3_hash_shard_assignment_is_stable_and_bounded(tmp_path):
    layout = FileStoreLayout(tmp_path)
    run_ids = [f"run-{idx:06d}" for idx in range(300_000)]
    shard_ids = {layout.shard_id(run_id) for run_id in run_ids}

    assert layout.shard_count == 64
    assert len(shard_ids) == 64
    assert all(0 <= int(shard_id) < 64 for shard_id in shard_ids)


def test_file_store_packed_metadata_latest_wins(tmp_path):
    store = FileStoreConfig(root=str(tmp_path)).connect()
    run_id = "aaaaaaaaaaaaaaaa"
    first = RunRecord.running(
        run_id=run_id,
        experiment_id="exp:1",
        context_fingerprint="ctx",
        params_fingerprint="params",
        seed_fingerprint="seed",
    )
    final = RunRecord.success(
        run_id=run_id,
        experiment_id="exp:1",
        context_fingerprint="ctx",
        params_fingerprint="params",
        seed_fingerprint="seed",
        metrics={"score": 2},
    )

    store.put_run_record(first)
    store.put_run_record(final)
    store.put_result(run_id, "table", {"old": True})
    store.put_result(run_id, "table", {"old": False})
    store.put_log(run_id, "run", "first")
    store.put_log(run_id, "run", "second")
    store.put_artifact(
        b"payload",
        ArtifactDescriptor(
            artifact_id="artifact-1",
            name="payload",
            kind="blob",
            format="bin",
            uri="",
            metadata={"_run_id": run_id},
        ),
    )

    assert store.get_run_record(run_id).status == Status.SUCCESS
    assert store.get_result(run_id, "table")["data"] == {"old": False}
    assert store.get_log(run_id, "run") == "second"
    assert [artifact.name for artifact in store.list_artifacts(run_id)] == ["payload"]
    assert not (tmp_path / "derived").exists()


def test_success_cache_is_freshness_checked(tmp_path):
    store = FileStoreConfig(root=str(tmp_path)).connect()
    success = RunRecord.success(
        run_id="aaaaaaaaaaaaaaaa",
        experiment_id="exp:1",
        context_fingerprint="ctx",
        params_fingerprint="params",
        seed_fingerprint="seed",
    )
    failed = RunRecord.failed(
        run_id="bbbbbbbbbbbbbbbb",
        experiment_id="exp:1",
        context_fingerprint="ctx",
        params_fingerprint="params",
        seed_fingerprint="seed",
        error_type="Error",
        error_message="boom",
    )
    store.put_run_record(success)
    store.put_run_record(failed)

    statuses = store.get_run_statuses([success.run_id, failed.run_id])

    assert statuses == {success.run_id: Status.SUCCESS}
    cache_path = FileStoreLayout(tmp_path).success_cache_path()
    cache = json.loads(cache_path.read_text())
    assert cache["success_run_ids"] == [success.run_id]

    cache["success_run_ids"] = []
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    assert store.get_run_statuses([success.run_id]) == {}

    store.put_run_record(
        RunRecord.success(
            run_id="cccccccccccccccc",
            experiment_id="exp:1",
            context_fingerprint="ctx",
            params_fingerprint="params",
            seed_fingerprint="seed",
        )
    )

    statuses = store.get_run_statuses([success.run_id, "cccccccccccccccc"])

    assert statuses == {
        success.run_id: Status.SUCCESS,
        "cccccccccccccccc": Status.SUCCESS,
    }


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


def test_status_requires_manifest_with_clear_error(tmp_path):
    try:
        read_status(tmp_path)
    except RunStoreNotFoundError as e:
        assert "Expected manifest.json" in str(e)
        assert "metalab run" in str(e)
    else:
        raise AssertionError("read_status should reject non-store directories")


def test_status_keeps_success_after_resume_skip_event(tmp_path):
    store = FileStoreConfig(root=str(tmp_path)).connect()
    store.write_root_manifest(
        {
            "experiment_id": "exp:1",
            "expected_run_count": 1,
            "executor_type": "local",
            "job_id": "job1",
            "created_at": datetime.now().isoformat(),
        }
    )
    worker_sink = store.event_sink("job1", "worker1", "exp:1")
    worker_sink.emit("started", run_id="r1")
    worker_sink.emit("finished", run_id="r1")

    assert read_status(tmp_path).success == 1

    runner_sink = store.event_sink("job2", "runner", "exp:1")
    runner_sink.emit("skipped", run_id="r1", payload={"reason": "already success"})

    status = read_status(tmp_path)
    assert status.success == 1
    assert status.pending == 0


def test_status_handles_large_event_stream_with_compact_cache(tmp_path):
    total = 100_000
    store = FileStoreConfig(root=str(tmp_path)).connect()
    store.write_root_manifest(
        {
            "experiment_id": "exp:1",
            "expected_run_count": total,
            "executor_type": "local",
            "job_id": "job1",
            "created_at": datetime.now().isoformat(),
        }
    )
    event_path = FileStoreLayout(tmp_path).event_log_path("job1", "worker1")
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("w", encoding="utf-8") as f:
        for idx in range(total):
            f.write(
                json.dumps(
                    {
                        "kind": "finished",
                        "run_id": f"run-{idx:06d}",
                        "timestamp": "2026-01-01T00:00:00",
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

    status = read_status(tmp_path)

    assert status.total == total
    assert status.success == total
    assert status.pending == 0
    cache = json.loads((tmp_path / "index" / "status-cache.json").read_text())
    assert cache["format"] == "compact-v1"
    assert cache["per_run"]["run-000000"][0] == "s"


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
    assert shorten_value("kind", "skipped") == "skip"
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
