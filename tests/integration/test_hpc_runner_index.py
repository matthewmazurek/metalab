from __future__ import annotations

import json

import metalab
from metalab import ProcessExecutor, RunRecord, Status, ThreadExecutor
from metalab.index import export, index_is_current, rebuild_index, summary
from metalab.store.file import FileStoreConfig


@metalab.operation
def _op(params, seeds, capture):
    capture.metric("score", params["x"] + seeds.replicate_index)


def _experiment():
    return metalab.Experiment(
        name="hpc",
        version="1",
        context={},
        operation=_op,
        params=metalab.grid(x=[1, 2]),
        seeds=metalab.seeds(base=1, replicates=2),
    )


def test_local_run_writes_v3_store_and_resumes(tmp_path):
    exp = _experiment()

    handle = metalab.run(
        exp,
        store=str(tmp_path),
        executor=ThreadExecutor(max_workers=2),
        verbose=False,
    )
    results = handle.result()

    assert len(results) == 4
    assert (tmp_path / "manifest.json").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["layout_version"] == 3
    assert manifest["metadata_layout"] == "hash-mod-sharded-ndjson"
    assert manifest["shard_count"] == 64
    assert manifest["expected_run_count"] == 4
    assert "expected_run_ids" not in manifest
    assert manifest["expected_run_ids_path"] == "index/planned-runs/{prefix}.ndjson"
    assert len(list((tmp_path / "index" / "planned-runs").glob("*.ndjson"))) >= 1
    assert len(list((tmp_path / "runs").glob("*/*.json"))) == 0
    assert len(list((tmp_path / "runs" / "shards").glob("*.ndjson"))) >= 1
    assert len(list((tmp_path / "runs" / "shards").glob("*.idx"))) >= 1
    assert len(list((tmp_path / "metadata" / "logs").glob("*.ndjson"))) >= 1
    assert len(list((tmp_path / "logs").glob("*/*.log"))) == 0
    assert len(list((tmp_path / "events").glob("*/*.ndjson"))) >= 1
    assert len(list((tmp_path / "heartbeats").glob("*/*.json"))) >= 1
    events = [
        json.loads(line)
        for path in (tmp_path / "events").glob("*/*.ndjson")
        for line in path.read_text().splitlines()
    ]
    assert sum(1 for event in events if event["kind"] == "planned_batch") == 1
    assert not any(event["kind"] == "planned" for event in events)
    heartbeat_files = list((tmp_path / "heartbeats").glob("*/*.json"))
    assert 1 <= len(heartbeat_files) <= 2

    second = metalab.run(exp, store=str(tmp_path), verbose=False)
    assert second.status.skipped == 4


def test_load_results_uses_indexed_facade_without_eager_records(tmp_path):
    exp = _experiment()
    metalab.run(exp, store=str(tmp_path), verbose=False).result()

    results = metalab.load_results(str(tmp_path), indexed=True)

    assert isinstance(results, metalab.IndexedResults)
    assert index_is_current(tmp_path)
    assert len(results) == 4
    assert len(results.successful) == 4
    assert results.summary()["by_status"] == {"success": 4}
    assert results[0].status == Status.SUCCESS

    rows = results.table()
    assert len(rows) == 4
    assert "param_x" in rows[0]
    assert "score" in rows[0]

    eager = metalab.load_results(str(tmp_path), indexed=False)
    assert isinstance(eager, metalab.Results)


def test_index_freshness_uses_event_offsets_not_run_tree_mtime(tmp_path):
    exp = _experiment()
    metalab.run(exp, store=str(tmp_path), verbose=False).result()
    rebuild_index(tmp_path, force=True)
    assert index_is_current(tmp_path)

    run_shard = next((tmp_path / "runs" / "shards").glob("*.ndjson"))
    run_shard.touch()
    assert index_is_current(tmp_path)

    event_path = next((tmp_path / "events").glob("*/*.ndjson"))
    with event_path.open("a", encoding="utf-8") as f:
        f.write("\n")
    assert not index_is_current(tmp_path)


def test_resume_reruns_running_records(tmp_path):
    exp = _experiment()
    metalab.run(exp, store=str(tmp_path), verbose=False).result()

    store = FileStoreConfig(root=str(tmp_path)).connect()
    stale_record = store.list_run_records()[0]
    store.put_run_record(
        RunRecord.running(
            run_id=stale_record.run_id,
            experiment_id=stale_record.experiment_id,
            context_fingerprint=stale_record.context_fingerprint,
            params_fingerprint=stale_record.params_fingerprint,
            seed_fingerprint=stale_record.seed_fingerprint,
            params_resolved=stale_record.params_resolved,
            provenance=stale_record.provenance,
        )
    )

    rerun = metalab.run(exp, store=str(tmp_path), verbose=False)
    assert rerun.status.skipped == 3

    results = rerun.result()
    assert len(results.successful) == 4
    assert store.get_run_record(stale_record.run_id).status == Status.SUCCESS


def test_resume_reruns_malformed_records(tmp_path):
    exp = _experiment()
    metalab.run(exp, store=str(tmp_path), verbose=False).result()

    store = FileStoreConfig(root=str(tmp_path)).connect()
    malformed_record = store.list_run_records()[0]
    shard_path = store.layout.run_shard_path(malformed_record.run_id)
    offset = shard_path.stat().st_size
    bad = b"{bad json\n"
    with shard_path.open("ab") as handle:
        handle.write(bad)
    existing_sequences = [
        json.loads(line)["sequence"]
        for line in store.layout.run_shard_index_path(malformed_record.run_id)
        .read_text()
        .splitlines()
        if line.strip()
    ]
    index_row = {
        "run_id": malformed_record.run_id,
        "shard_id": store.layout.shard_id(malformed_record.run_id),
        "offset": offset,
        "length": len(bad),
        "status": "success",
        "sequence": max(existing_sequences) + 1,
    }
    with store.layout.run_shard_index_path(malformed_record.run_id).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(index_row, sort_keys=True) + "\n")
    with store.layout.shard_map_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(index_row, sort_keys=True) + "\n")

    rerun = metalab.run(exp, store=str(tmp_path), verbose=False)
    assert rerun.status.skipped == 3

    results = rerun.result()
    assert len(results.successful) == 4
    assert store.get_run_record(malformed_record.run_id).status == Status.SUCCESS


def test_process_executor_writes_events_and_stable_worker_heartbeats(tmp_path):
    exp = _experiment()

    with ProcessExecutor(max_workers=2) as executor:
        results = metalab.run(
            exp,
            store=str(tmp_path),
            executor=executor,
            verbose=False,
        ).result()

    assert len(results.successful) == 4

    events = [
        json.loads(line)
        for path in (tmp_path / "events").glob("*/*.ndjson")
        for line in path.read_text().splitlines()
    ]
    transition_events = [event for event in events if event.get("run_id")]
    assert {event["kind"] for event in transition_events} >= {"started", "finished"}

    heartbeat_files = list((tmp_path / "heartbeats").glob("*/*.json"))
    assert 1 <= len(heartbeat_files) <= 2
    assert all(path.name.startswith("process_") for path in heartbeat_files)


def test_runner_uses_plan_based_executor_without_concrete_type_check(tmp_path):
    class FakePlanExecutor:
        def __init__(self) -> None:
            self.plan = None

        def submit_experiment(self, plan):
            self.plan = plan
            return object()

    exp = _experiment()
    metalab.run(exp, store=str(tmp_path), verbose=False).result()

    executor = FakePlanExecutor()
    metalab.run(exp, store=str(tmp_path), executor=executor, verbose=False)

    assert executor.plan is not None
    assert executor.plan.skipped_count == 4
    assert executor.plan.total_runs == 4
    assert executor.plan.pending_entries == []
    assert len(executor.plan.all_run_ids) == 4
    assert executor.plan.context_fingerprint
    assert executor.plan.store.get_working_directory() == tmp_path

    events = [
        json.loads(line)
        for path in (tmp_path / "events").glob("*/*.ndjson")
        for line in path.read_text().splitlines()
    ]
    plan_job_id = json.loads((tmp_path / "manifest.json").read_text())["job_id"]
    plan_events = [event for event in events if event["job_id"] == plan_job_id]

    assert sum(1 for event in plan_events if event["kind"] == "planned_batch") == 1
    assert sum(1 for event in plan_events if event["kind"] == "skipped") == 4


def test_duckdb_rebuild_summary_and_export(tmp_path):
    metalab.run(_experiment(), store=str(tmp_path), verbose=False).result()

    db_path = rebuild_index(tmp_path, force=True)
    assert db_path.exists()

    rows = summary(tmp_path)
    by_status = {row["status"]: row["n"] for row in rows}
    assert by_status["success"] == 4

    csv_path = export(tmp_path, fmt="csv", out=tmp_path / "runs.csv")
    assert csv_path.exists()
    assert "run_id" in csv_path.read_text()


def test_non_file_locator_is_rejected():
    from metalab.store.locator import parse_to_config

    try:
        parse_to_config("postgresql://localhost/db")
    except ValueError as e:
        assert "filesystem stores only" in str(e)
    else:
        raise AssertionError("postgres locator should fail")
