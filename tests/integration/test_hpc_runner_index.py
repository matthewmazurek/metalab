from __future__ import annotations

import json
import tarfile

import metalab
from metalab import ProcessExecutor, RunRecord, Status, ThreadExecutor
from metalab.index import export, export_target, index_is_current, rebuild_index, summary
from metalab.store.file import FileStore, FileStoreConfig
from metalab.store.layout import FileStoreLayout


@metalab.operation
def _op(params, seeds, capture):
    score = params["x"] + seeds.replicate_index
    capture.metric("score", score)
    capture.data("curve", [params["x"], score])


def _experiment(context=None):
    return metalab.Experiment(
        name="hpc",
        version="1",
        context=context or {},
        operation=_op,
        params=metalab.grid(x=[1, 2]),
        seeds=metalab.seeds(base=1, replicates=2),
    )


def test_local_run_writes_v4_store_and_resumes(tmp_path):
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
    layout = FileStoreLayout(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        ".metalab",
        "manifest.json",
        "outputs",
        "records",
    ]
    assert manifest["layout_version"] == 4
    assert manifest["outputs_layout"] == "hash-mod-sharded-ndjson"
    assert manifest["shard_count"] == 64
    assert manifest["expected_run_count"] == 4
    assert "expected_run_ids" not in manifest
    assert manifest["expected_run_ids_path"] == ".metalab/index/planned-runs/{prefix}.ndjson"
    assert len(list(layout.planned_runs_dir_path().glob("*.ndjson"))) >= 1
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "events").exists()
    assert not (tmp_path / "heartbeats").exists()
    assert not (tmp_path / "index").exists()
    assert len(list((tmp_path / "records").glob("*.ndjson"))) >= 1
    assert len(list((tmp_path / "records").glob("*.idx"))) >= 1
    assert len(list((tmp_path / "outputs" / "logs").glob("*.ndjson"))) >= 1
    assert len(list((tmp_path / "outputs" / "artifacts" / "metadata").glob("*.ndjson"))) == 0
    assert (tmp_path / "outputs" / "artifacts" / "files").exists()
    assert not (tmp_path / "metadata").exists()
    assert not (tmp_path / "artifacts").exists()
    assert len(list((tmp_path / "logs").glob("*/*.log"))) == 0
    assert len(list(layout.events_dir_path().glob("*/*.ndjson"))) >= 1
    assert len(list(layout.heartbeats_dir_path().glob("*/*.json"))) >= 1
    submissions_path = layout.submissions_path()
    assert submissions_path.exists()
    assert not (tmp_path / ".metalab" / "experiments").exists()
    assert not (tmp_path / ".metalab" / "contexts").exists()
    submissions = [json.loads(line) for line in submissions_path.read_text().splitlines()]
    assert len(submissions) == 1
    assert submissions[0]["manifest"]["submission_id"] == manifest["job_id"]
    events = [
        json.loads(line)
        for path in layout.events_dir_path().glob("*/*.ndjson")
        for line in path.read_text().splitlines()
    ]
    assert sum(1 for event in events if event["kind"] == "planned_batch") == 1
    assert not any(event["kind"] == "planned" for event in events)
    heartbeat_files = list(layout.heartbeats_dir_path().glob("*/*.json"))
    assert 1 <= len(heartbeat_files) <= 2

    second = metalab.run(exp, store=str(tmp_path), verbose=False)
    assert second.status.skipped == 4
    assert len(submissions_path.read_text().splitlines()) == 2


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


def test_load_results_auto_requires_current_sidecar(tmp_path):
    exp = _experiment()
    metalab.run(exp, store=str(tmp_path), verbose=False).result()

    try:
        metalab.load_results(str(tmp_path))
    except RuntimeError as e:
        assert "sidecar index is missing or stale" in str(e)
        assert "metalab index rebuild" in str(e)
    else:
        raise AssertionError("auto indexed load should not implicitly rebuild")

    rebuild_index(tmp_path, force=True)
    results = metalab.load_results(str(tmp_path))
    assert isinstance(results, metalab.IndexedResults)
    assert len(results) == 4

    layout = FileStoreLayout(tmp_path)
    event_path = next(layout.events_dir_path().glob("*/*.ndjson"))
    with event_path.open("a", encoding="utf-8") as f:
        f.write("\n")

    try:
        metalab.load_results(str(tmp_path))
    except RuntimeError as e:
        assert "sidecar index is missing or stale" in str(e)
    else:
        raise AssertionError("auto indexed load should reject stale sidecars")

    eager = metalab.load_results(str(tmp_path), indexed=False)
    assert isinstance(eager, metalab.Results)


def test_reconnect_constructs_handle_without_loading_records(tmp_path, monkeypatch):
    manifest = {
        "layout_version": 4,
        "outputs_layout": "hash-mod-sharded-ndjson",
        "shard_hash": "sha256",
        "shard_count": 64,
        "record_schema_version": 1,
        "experiment_id": "hpc:1",
        "executor_type": "slurm",
        "submission_mode": "array_indexed",
        "job_ids": ["12345"],
        "shards": [
            {
                "start_idx": 0,
                "end_idx": 0,
                "array_range": "0",
                "job_id": "12345",
            }
        ],
        "total_runs": 1,
        "total_chunks": 1,
        "chunk_size": 1,
        "skipped_count": 0,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def fail_list_records(self, experiment_id=None):
        raise AssertionError("reconnect should not materialize run records")

    monkeypatch.setattr(FileStore, "list_run_records", fail_list_records)

    handle = metalab.reconnect(str(tmp_path), verbose=False)

    assert handle.can_reconnect
    assert handle.job_id == "12345"


def test_slurm_result_can_delegate_to_indexed_results(tmp_path, monkeypatch):
    exp = _experiment()
    metalab.run(exp, store=str(tmp_path), verbose=False).result()
    store = FileStoreConfig(root=str(tmp_path)).connect()
    handle = metalab.SlurmRunHandle(
        store=store,
        job_ids=[],
        shards=[],
        total_runs=4,
        chunk_size=1,
    )

    monkeypatch.setattr(
        metalab.SlurmRunHandle,
        "_await_completion",
        lambda self, timeout=None: None,
    )

    results = handle.result(indexed=True)

    assert isinstance(results, metalab.IndexedResults)
    assert index_is_current(tmp_path)
    assert len(results) == 4


def test_resolved_context_is_stored_in_submission_log(tmp_path):
    context_file = tmp_path / "input.txt"
    context_file.write_text("hello", encoding="utf-8")
    exp = _experiment(context={"input": metalab.FilePath(str(context_file))})

    metalab.run(exp, store=str(tmp_path), verbose=False).result()

    layout = FileStoreLayout(tmp_path)
    submissions = [
        json.loads(line)
        for line in layout.submissions_path().read_text(encoding="utf-8").splitlines()
    ]

    assert len(submissions) == 1
    manifest = submissions[0]["manifest"]
    assert "resolved_context" in manifest
    assert "context.input" in manifest["resolved_context"]["resolved_fields"]
    assert not (tmp_path / ".metalab" / "contexts").exists()


def test_index_freshness_uses_event_offsets_not_run_tree_mtime(tmp_path):
    exp = _experiment()
    metalab.run(exp, store=str(tmp_path), verbose=False).result()
    rebuild_index(tmp_path, force=True)
    assert index_is_current(tmp_path)

    layout = FileStoreLayout(tmp_path)
    record_shard = next((tmp_path / "records").glob("*.ndjson"))
    record_shard.touch()
    assert index_is_current(tmp_path)

    event_path = next(layout.events_dir_path().glob("*/*.ndjson"))
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
    shard_path = store.layout.record_shard_path(malformed_record.run_id)
    offset = shard_path.stat().st_size
    bad = b"{bad json\n"
    with shard_path.open("ab") as handle:
        handle.write(bad)
    existing_sequences = [
        json.loads(line)["sequence"]
        for line in store.layout.record_shard_index_path(malformed_record.run_id)
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
    with store.layout.record_shard_index_path(malformed_record.run_id).open("a", encoding="utf-8") as handle:
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
    layout = FileStoreLayout(tmp_path)

    events = [
        json.loads(line)
        for path in layout.events_dir_path().glob("*/*.ndjson")
        for line in path.read_text().splitlines()
    ]
    transition_events = [event for event in events if event.get("run_id")]
    assert {event["kind"] for event in transition_events} >= {"started", "finished"}

    heartbeat_files = list(layout.heartbeats_dir_path().glob("*/*.json"))
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

    layout = FileStoreLayout(tmp_path)
    events = [
        json.loads(line)
        for path in layout.events_dir_path().glob("*/*.ndjson")
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
    assert "params.x" in csv_path.read_text()


def test_typed_export_targets_table_snapshot_and_archive(tmp_path):
    metalab.run(_experiment(), store=str(tmp_path), verbose=False).result()

    table_path = export_target(tmp_path, target="table", out=tmp_path / "runs.jsonl")
    assert table_path.exists()
    table_row = json.loads(table_path.read_text().splitlines()[0])
    assert "params.x" in table_row
    assert "metrics.score" in table_row

    snapshot_path = export_target(tmp_path, target="snapshot", out=tmp_path / "runs.duckdb")
    assert snapshot_path.exists()

    archive_path = export_target(tmp_path, target="archive", out=tmp_path / "runs.tar")
    assert archive_path.exists()
    with tarfile.open(archive_path) as archive:
        names = set(archive.getnames())
    assert "manifest.json" in names
    assert "runs.tar" not in names


def test_typed_export_dataset_writes_anndata_zarr(tmp_path):
    import anndata as ad

    metalab.run(_experiment(), store=str(tmp_path), verbose=False).result()

    dataset_path = export_target(tmp_path, target="dataset", out=tmp_path / "runs.zarr")

    adata = ad.read_zarr(dataset_path)
    assert adata.n_obs == 4
    assert "params.x" in adata.obs
    assert "metrics.score" in adata.obs
    assert "curve" in adata.obsm
    assert adata.uns["metalab"]["source_is_run_store"] is True
    experiment = adata.uns["metalab"]["experiment"]
    assert experiment["experiment_id"] == "hpc:1"
    assert experiment["name"] == "hpc"
    assert experiment["version"] == "1"
    assert experiment["description"] is None
    assert list(experiment["tags"]) == []
    assert experiment["metadata"] == {}
    capture = adata.uns["metalab"]["capture"]
    assert "curve" in capture["obsm"]
    assert capture["obsm"]["curve"]["stored_in"] == "obsm"
    assert list(capture["obsm"]["curve"]["original_shape"]) == [2]
    assert list(capture["obsm"]["curve"]["stacked_shape"]) == [4, 2]
    assert capture["skipped"] == {}


def test_non_file_locator_is_rejected():
    from metalab.store.locator import parse_to_config

    try:
        parse_to_config("postgresql://localhost/db")
    except ValueError as e:
        assert "filesystem stores only" in str(e)
    else:
        raise AssertionError("postgres locator should fail")
