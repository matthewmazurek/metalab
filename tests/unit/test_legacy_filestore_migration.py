from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from metalab.status import read_status
from metalab.store.file import FileStoreConfig


def _load_migration_module():
    path = Path(__file__).parents[2] / "scripts" / "migrate_legacy_filestore.py"
    spec = importlib.util.spec_from_file_location("migrate_legacy_filestore", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_legacy_record(root: Path, run_id: str, *, done: bool) -> None:
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{run_id}.json").write_text(
        json.dumps(
            {
                "_schema_version": "0.1",
                "run_id": run_id,
                "experiment_id": "legacy_exp:1",
                "status": "success",
                "context_fingerprint": "ctx",
                "params_fingerprint": f"params-{run_id}",
                "seed_fingerprint": "seed",
                "started_at": "2026-01-01T00:00:00",
                "finished_at": "2026-01-01T00:00:01",
                "duration_ms": 1000,
                "metrics": {"score": 1.5},
                "params_resolved": {"x": 1},
                "provenance": {
                    "code_hash": "abc",
                    "python_version": "3.11",
                    "metalab_version": "0.1",
                    "executor_id": "slurm",
                    "host": "hpc",
                    "extra": {"kept": True},
                },
            }
        ),
        encoding="utf-8",
    )
    if done:
        (runs / f"{run_id}.done").write_text("{}", encoding="utf-8")


def test_legacy_migration_only_promotes_done_successes(tmp_path):
    module = _load_migration_module()
    legacy = tmp_path / "legacy"
    output = tmp_path / "v2"
    _write_legacy_record(legacy, "aaaaaaaaaaaaaaaa", done=True)
    _write_legacy_record(legacy, "bbbbbbbbbbbbbbbb", done=False)

    counts = module.migrate(legacy, output)

    assert counts["planned"] == 2
    assert counts["migrated"] == 1
    assert counts["left_pending"] == 1

    migrated = FileStoreConfig(root=str(output)).connect().get_run_record(
        "aaaaaaaaaaaaaaaa"
    )
    assert migrated is not None
    assert migrated.status.value == "success"
    assert migrated.provenance.extra == {"kept": True}

    assert (
        FileStoreConfig(root=str(output)).connect().get_run_record("bbbbbbbbbbbbbbbb")
        is None
    )

    status = read_status(output)
    assert status.total == 2
    assert status.success == 1
    assert status.pending == 1
