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


def _write_legacy_submission(root: Path, *, total_runs: int = 3) -> None:
    experiments = root / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    (experiments / "legacy_exp_1_20260101_000000.json").write_text(
        json.dumps(
            {
                "experiment_id": "legacy_exp:1",
                "submitted_at": "2026-01-01T00:00:00",
                "total_runs": total_runs,
                "operation": {"code_hash": "abc"},
                "params": {"type": "GridSource", "total_cases": total_runs},
                "seeds": {"type": "SeedPlan", "base": 1, "replicates": 1},
            }
        ),
        encoding="utf-8",
    )


def test_legacy_migration_only_promotes_done_successes(tmp_path):
    module = _load_migration_module()
    legacy = tmp_path / "legacy"
    output = tmp_path / "v4"
    _write_legacy_record(legacy, "aaaaaaaaaaaaaaaa", done=True)
    _write_legacy_record(legacy, "bbbbbbbbbbbbbbbb", done=False)
    _write_legacy_submission(legacy, total_runs=3)

    counts = module.migrate(legacy, output)

    assert counts["planned"] == 3
    assert counts["observed_records"] == 2
    assert counts["migrated"] == 1
    assert counts["left_pending"] == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["layout_version"] == 4
    assert manifest["expected_run_count"] == 3
    assert manifest["observed_legacy_record_count"] == 2
    meta = json.loads((output / ".metalab" / "meta.json").read_text())
    assert meta["experiment_id"] == "legacy_exp:1"
    assert (output / "records").exists()
    assert (output / "outputs").exists()
    assert (output / "outputs" / "artifacts" / "metadata").exists()
    assert not (output / "runs").exists()
    assert not (output / "metadata").exists()
    assert not (output / "artifacts").exists()
    assert (output / ".metalab" / "events" / "legacy-migration" / "legacy.ndjson").exists()
    submissions = [
        json.loads(line)
        for line in (output / ".metalab" / "submissions.ndjson").read_text().splitlines()
    ]
    assert submissions[0]["manifest"]["migration_source"] == str(legacy)

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
    assert status.total == 3
    assert status.success == 1
    assert status.pending == 2


def test_legacy_migration_dry_run_uses_fast_count_path(tmp_path):
    module = _load_migration_module()
    legacy = tmp_path / "legacy"
    output = tmp_path / "v4"
    _write_legacy_record(legacy, "aaaaaaaaaaaaaaaa", done=True)
    _write_legacy_record(legacy, "bbbbbbbbbbbbbbbb", done=False)
    _write_legacy_submission(legacy, total_runs=3)

    def fail_if_normalized(*args, **kwargs):
        raise AssertionError("dry-run should not normalize full run records")

    module._run_record = fail_if_normalized

    counts = module.migrate(legacy, output, dry_run=True)

    assert counts["planned"] == 3
    assert counts["observed_records"] == 2
    assert counts["migrated"] == 1
    assert counts["left_pending"] == 1
    assert not output.exists()
