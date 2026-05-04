from __future__ import annotations

import json

import metalab
from metalab import ThreadExecutor
from metalab.index import export, rebuild_index, summary


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


def test_local_run_writes_v2_store_and_resumes(tmp_path):
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
    assert len(list((tmp_path / "runs").glob("*/*.json"))) == 4
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
