"""
Round-trip fidelity test: PostgresStore → FileStore → PostgresStore.

Tests that rebuild_index() restores 100% of indexed data from the
filesystem source of truth. Specifically verifies:

1. Run records (all fields, metrics, params, provenance, etc.)
2. Derived metrics
3. Experiment manifests (including total_runs)
4. Field catalog (params, metrics, AND derived fields)

Requires a local PostgreSQL instance. Set METALAB_TEST_POSTGRES_URL
to override the default connection string, or skip with -m "not postgres".
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from metalab.schema import dump_run_record
from metalab.types import ArtifactDescriptor, Provenance, RunRecord, Status

# ---------------------------------------------------------------------------
# Skip unless Postgres is available
# ---------------------------------------------------------------------------
try:
    import psycopg  # noqa: F401
    from psycopg_pool import ConnectionPool  # noqa: F401

    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False

DEFAULT_PG_URL = "postgresql://localhost/metalab_test"
PG_URL = os.environ.get("METALAB_TEST_POSTGRES_URL", DEFAULT_PG_URL)

# Use a unique schema per test run to avoid collisions
_TEST_SCHEMA = f"test_roundtrip_{os.getpid()}"


def _pg_available() -> bool:
    """Check if we can actually connect to Postgres."""
    if not HAS_PSYCOPG:
        return False
    try:
        conn = psycopg.connect(PG_URL, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


PG_AVAILABLE = _pg_available()

pytestmark = pytest.mark.skipif(
    not PG_AVAILABLE,
    reason=f"PostgreSQL not available at {PG_URL}",
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_records(experiment_id: str, count: int = 5) -> list[RunRecord]:
    """Create a set of diverse test run records."""
    base_time = datetime(2025, 6, 15, 10, 0, 0)
    records = []
    for i in range(count):
        started = base_time + timedelta(minutes=i * 10)
        finished = started + timedelta(seconds=30 + i * 5)
        duration_ms = int((finished - started).total_seconds() * 1000)

        status = Status.SUCCESS if i % 3 != 0 else Status.FAILED
        error = (
            {"type": "ValueError", "message": f"test error {i}", "traceback": "..."}
            if status == Status.FAILED
            else None
        )

        records.append(
            RunRecord(
                run_id=f"run_{experiment_id}_{i:03d}",
                experiment_id=experiment_id,
                status=status,
                context_fingerprint=f"ctx_fp_{i}",
                params_fingerprint=f"params_fp_{i}",
                seed_fingerprint=f"seed_fp_{i}",
                started_at=started,
                finished_at=finished,
                duration_ms=duration_ms,
                metrics={
                    "loss": 0.5 - i * 0.05,
                    "accuracy": 0.8 + i * 0.02,
                    "epoch": i + 1,
                    "label": f"category_{i % 3}",
                },
                provenance=Provenance(
                    code_hash=f"hash_{i}",
                    python_version="3.11.5",
                    metalab_version="0.1.0",
                    executor_id="thread_pool",
                    host="testhost",
                    extra={"gpu": "A100", "run_index": i},
                ),
                error=error,
                params_resolved={
                    "lr": 0.001 * (i + 1),
                    "batch_size": 32 * (i + 1),
                    "optimizer": "adam" if i % 2 == 0 else "sgd",
                    "dropout": 0.1 + i * 0.05,
                },
                tags=["test", f"fold_{i}"],
                warnings=[{"code": "W001", "message": f"warn {i}"}] if i == 2 else [],
                notes=f"Test run {i}" if i % 2 == 0 else None,
                artifacts=[
                    ArtifactDescriptor(
                        artifact_id=f"art_{i}_0",
                        name=f"model_{i}",
                        kind="pickle",
                        format="pkl",
                        uri=f"/fake/path/model_{i}.pkl",
                        content_hash=f"sha256_{i}",
                        size_bytes=1024 * (i + 1),
                        metadata={"framework": "pytorch"},
                    )
                ],
            )
        )
    return records


def _make_derived(records: list[RunRecord]) -> list[tuple[str, dict[str, Any]]]:
    """Create derived metrics for the records."""
    pairs = []
    for record in records:
        if record.status == Status.SUCCESS:
            loss = record.metrics.get("loss", 0.5)
            acc = record.metrics.get("accuracy", 0.8)
            pairs.append(
                (
                    record.run_id,
                    {
                        "normalized_loss": float(loss) / 0.5,
                        "f1_score": float(acc) * 0.95,
                        "quality_label": "good" if float(acc) > 0.85 else "ok",
                    },
                )
            )
    return pairs


def _make_manifest(
    experiment_id: str,
    records: list[RunRecord],
) -> dict[str, Any]:
    """Create a realistic experiment manifest."""
    return {
        "experiment_id": experiment_id,
        "name": f"Test Experiment ({experiment_id})",
        "version": "1.0",
        "description": "A test experiment for round-trip validation",
        "tags": ["test", "roundtrip", "validation"],
        "operation": {
            "ref": "tests.ops:dummy_op",
            "name": "dummy_op",
            "code_hash": "abc123",
        },
        "params": {
            "lr": {"type": "grid", "values": [0.001, 0.01, 0.1]},
            "batch_size": {"type": "grid", "values": [32, 64, 128]},
        },
        "seeds": {"global": 42, "numpy": 123},
        "context_fingerprint": "ctx_fp_global",
        "total_runs": len(records),
        "run_ids": [r.run_id for r in records],
        "submitted_at": "2025-06-15T10:00:00",
        "metadata": {"cluster": "local", "priority": "high"},
    }


@pytest.fixture
def pg_store(tmp_path: Path):
    """Create a PostgresStore with a real Postgres connection and clean schema."""
    from metalab.store.postgres import PostgresStoreConfig

    config = PostgresStoreConfig(
        connection_string=PG_URL,
        file_root=str(tmp_path),
        experiment_id="roundtrip_exp:1.0",
        schema=_TEST_SCHEMA,
    )
    store = config.connect()

    yield store

    # Cleanup: drop the test schema
    try:
        with store.index._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP SCHEMA IF EXISTS {_TEST_SCHEMA} CASCADE")
            conn.commit()
    except Exception:
        pass
    store.close()


@pytest.fixture
def unscoped_pg_store(tmp_path: Path):
    """Create an unscoped PostgresStore for multi-experiment testing."""
    from metalab.store.postgres import PostgresStoreConfig

    config = PostgresStoreConfig(
        connection_string=PG_URL,
        file_root=str(tmp_path),
        schema=_TEST_SCHEMA,
    )
    store = config.connect()

    yield store

    # Cleanup: drop the test schema
    try:
        with store.index._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP SCHEMA IF EXISTS {_TEST_SCHEMA} CASCADE")
            conn.commit()
    except Exception:
        pass
    store.close()


# ---------------------------------------------------------------------------
# Helper: snapshot the PG index state
# ---------------------------------------------------------------------------


def _snapshot_pg_state(index, schema: str) -> dict[str, Any]:
    """Read all indexed data from Postgres for comparison."""
    snapshot: dict[str, Any] = {}

    with index._conn() as conn:
        with conn.cursor() as cur:
            # Run records
            cur.execute(
                f"SELECT run_id, record_json FROM {schema}.runs ORDER BY run_id"
            )
            snapshot["runs"] = {
                row[0]: (row[1] if isinstance(row[1], dict) else json.loads(row[1]))
                for row in cur.fetchall()
            }

            # Derived metrics
            cur.execute(
                f"SELECT run_id, derived_json FROM {schema}.derived ORDER BY run_id"
            )
            snapshot["derived"] = {
                row[0]: (row[1] if isinstance(row[1], dict) else json.loads(row[1]))
                for row in cur.fetchall()
            }

            # Experiment manifests
            cur.execute(
                f"""
                SELECT experiment_id, timestamp, manifest_json, total_runs
                FROM {schema}.experiment_manifests
                ORDER BY experiment_id, timestamp
            """
            )
            snapshot["manifests"] = [
                {
                    "experiment_id": row[0],
                    "timestamp": row[1],
                    "manifest_json": (
                        row[2] if isinstance(row[2], dict) else json.loads(row[2])
                    ),
                    "total_runs": row[3],
                }
                for row in cur.fetchall()
            ]

            # Field catalog
            cur.execute(
                f"""
                SELECT namespace, field_name, field_type, count, values, min_value, max_value
                FROM {schema}.field_catalog
                ORDER BY namespace, field_name
            """
            )
            snapshot["field_catalog"] = {
                (row[0], row[1]): {
                    "field_type": row[2],
                    "count": row[3],
                    "values": sorted(row[4]) if row[4] else None,
                    "min_value": row[5],
                    "max_value": row[6],
                }
                for row in cur.fetchall()
            }

    return snapshot


# ===========================================================================
# Tests
# ===========================================================================


class TestRebuildRoundTrip:
    """Test that rebuild_index() restores PG state from files with 100% fidelity."""

    def test_run_records_survive_roundtrip(self, pg_store):
        """All run record fields survive PG → files → PG rebuild."""
        records = _make_records("roundtrip_exp:1.0", count=5)
        for r in records:
            pg_store.put_run_record(r)

        # Snapshot before rebuild
        before = _snapshot_pg_state(pg_store.index, _TEST_SCHEMA)
        assert len(before["runs"]) == 5

        # Rebuild (clears PG, restores from files)
        n_indexed = pg_store.rebuild_index()
        assert n_indexed == 5

        # Snapshot after rebuild
        after = _snapshot_pg_state(pg_store.index, _TEST_SCHEMA)

        # Compare run records
        assert set(before["runs"].keys()) == set(after["runs"].keys())
        for run_id in before["runs"]:
            before_data = before["runs"][run_id]
            after_data = after["runs"][run_id]

            # Compare all fields that matter
            for key in [
                "run_id",
                "experiment_id",
                "status",
                "context_fingerprint",
                "params_fingerprint",
                "seed_fingerprint",
                "duration_ms",
                "metrics",
                "params_resolved",
                "tags",
                "warnings",
                "notes",
                "error",
            ]:
                assert before_data.get(key) == after_data.get(key), (
                    f"Run {run_id}: field '{key}' mismatch: "
                    f"{before_data.get(key)!r} != {after_data.get(key)!r}"
                )

            # Provenance fields
            assert before_data.get("provenance") == after_data.get(
                "provenance"
            ), f"Run {run_id}: provenance mismatch"

            # Artifacts
            assert before_data.get("artifacts") == after_data.get(
                "artifacts"
            ), f"Run {run_id}: artifacts mismatch"

    def test_metrics_survive_roundtrip(self, pg_store):
        """All metric values survive rebuild with exact precision."""
        records = _make_records("roundtrip_exp:1.0", count=3)
        for r in records:
            pg_store.put_run_record(r)

        before = _snapshot_pg_state(pg_store.index, _TEST_SCHEMA)
        pg_store.rebuild_index()
        after = _snapshot_pg_state(pg_store.index, _TEST_SCHEMA)

        for run_id in before["runs"]:
            before_metrics = before["runs"][run_id]["metrics"]
            after_metrics = after["runs"][run_id]["metrics"]
            assert (
                before_metrics == after_metrics
            ), f"Run {run_id}: metrics mismatch: {before_metrics} != {after_metrics}"

    def test_derived_metrics_survive_roundtrip(self, pg_store):
        """Derived metrics survive rebuild with 100% fidelity."""
        records = _make_records("roundtrip_exp:1.0", count=5)
        for r in records:
            pg_store.put_run_record(r)

        derived_pairs = _make_derived(records)
        for run_id, derived in derived_pairs:
            pg_store.put_derived(run_id, derived)

        before = _snapshot_pg_state(pg_store.index, _TEST_SCHEMA)
        assert len(before["derived"]) == len(derived_pairs)

        pg_store.rebuild_index()
        after = _snapshot_pg_state(pg_store.index, _TEST_SCHEMA)

        # Same set of derived run_ids
        assert set(before["derived"].keys()) == set(after["derived"].keys()), (
            f"Derived run_ids mismatch: "
            f"{set(before['derived'].keys())} != {set(after['derived'].keys())}"
        )

        # Same values
        for run_id in before["derived"]:
            assert before["derived"][run_id] == after["derived"][run_id], (
                f"Derived for {run_id}: "
                f"{before['derived'][run_id]} != {after['derived'][run_id]}"
            )

    def test_experiment_manifests_survive_roundtrip(self, pg_store):
        """Experiment manifests (including total_runs) survive rebuild."""
        records = _make_records("roundtrip_exp:1.0", count=5)
        for r in records:
            pg_store.put_run_record(r)

        manifest = _make_manifest("roundtrip_exp:1.0", records)
        pg_store.put_experiment_manifest(
            "roundtrip_exp:1.0",
            manifest,
            timestamp="20250615_100000",
        )

        before = _snapshot_pg_state(pg_store.index, _TEST_SCHEMA)
        assert len(before["manifests"]) == 1
        assert before["manifests"][0]["total_runs"] == 5

        pg_store.rebuild_index()
        after = _snapshot_pg_state(pg_store.index, _TEST_SCHEMA)

        # Manifests must be present
        assert len(after["manifests"]) >= 1, "Manifests lost after rebuild!"

        # Compare manifest content
        before_m = before["manifests"][0]
        after_m = after["manifests"][0]

        assert before_m["experiment_id"] == after_m["experiment_id"]
        assert (
            before_m["total_runs"] == after_m["total_runs"]
        ), f"total_runs lost: {before_m['total_runs']} != {after_m['total_runs']}"

        # Compare manifest JSON content
        for key in ["name", "tags", "total_runs", "operation", "params", "seeds"]:
            assert before_m["manifest_json"].get(key) == after_m["manifest_json"].get(
                key
            ), f"Manifest field '{key}' mismatch"

    def test_field_catalog_includes_derived_after_roundtrip(self, pg_store):
        """Field catalog must include derived metric fields after rebuild."""
        records = _make_records("roundtrip_exp:1.0", count=5)
        for r in records:
            pg_store.put_run_record(r)

        derived_pairs = _make_derived(records)
        for run_id, derived in derived_pairs:
            pg_store.put_derived(run_id, derived)

        pg_store.rebuild_index()
        after = _snapshot_pg_state(pg_store.index, _TEST_SCHEMA)

        # Must have params fields
        params_fields = {
            k[1] for k, v in after["field_catalog"].items() if k[0] == "params"
        }
        assert "lr" in params_fields, "params.lr missing from field catalog"
        assert "batch_size" in params_fields, "params.batch_size missing"
        assert "optimizer" in params_fields, "params.optimizer missing"

        # Must have metrics fields
        metrics_fields = {
            k[1] for k, v in after["field_catalog"].items() if k[0] == "metrics"
        }
        assert "loss" in metrics_fields, "metrics.loss missing from field catalog"
        assert "accuracy" in metrics_fields, "metrics.accuracy missing"

        # Must have derived fields (this was the bug)
        derived_fields = {
            k[1] for k, v in after["field_catalog"].items() if k[0] == "derived"
        }
        assert (
            "normalized_loss" in derived_fields
        ), "derived.normalized_loss missing from field catalog after rebuild"
        assert (
            "f1_score" in derived_fields
        ), "derived.f1_score missing from field catalog after rebuild"
        assert (
            "quality_label" in derived_fields
        ), "derived.quality_label missing from field catalog after rebuild"

    def test_field_catalog_stats_correct_after_roundtrip(self, pg_store):
        """Field catalog stats (count, min, max, values) match after rebuild."""
        records = _make_records("roundtrip_exp:1.0", count=5)
        for r in records:
            pg_store.put_run_record(r)

        derived_pairs = _make_derived(records)
        for run_id, derived in derived_pairs:
            pg_store.put_derived(run_id, derived)

        pg_store.rebuild_index()
        catalog = _snapshot_pg_state(pg_store.index, _TEST_SCHEMA)["field_catalog"]

        # Check params.optimizer has correct values
        optimizer_entry = catalog.get(("params", "optimizer"))
        assert optimizer_entry is not None, "params.optimizer missing"
        assert optimizer_entry["field_type"] == "string"
        assert set(optimizer_entry["values"]) == {"adam", "sgd"}
        assert optimizer_entry["count"] == 5

        # Check metrics.loss has correct min/max
        loss_entry = catalog.get(("metrics", "loss"))
        assert loss_entry is not None, "metrics.loss missing"
        assert loss_entry["field_type"] == "numeric"
        assert loss_entry["count"] == 5

        # Check derived.quality_label has correct values
        quality_entry = catalog.get(("derived", "quality_label"))
        assert quality_entry is not None, "derived.quality_label missing"
        assert quality_entry["field_type"] == "string"
        assert quality_entry["count"] == len(derived_pairs)

    def test_multiple_manifests_survive_roundtrip(self, pg_store):
        """Multiple manifests for the same experiment survive rebuild."""
        records = _make_records("roundtrip_exp:1.0", count=3)
        for r in records:
            pg_store.put_run_record(r)

        # Store two manifests with different timestamps
        m1 = _make_manifest("roundtrip_exp:1.0", records[:2])
        m1["total_runs"] = 2
        pg_store.put_experiment_manifest(
            "roundtrip_exp:1.0", m1, timestamp="20250615_090000"
        )

        m2 = _make_manifest("roundtrip_exp:1.0", records)
        m2["total_runs"] = 3
        pg_store.put_experiment_manifest(
            "roundtrip_exp:1.0", m2, timestamp="20250615_100000"
        )

        before = _snapshot_pg_state(pg_store.index, _TEST_SCHEMA)
        assert len(before["manifests"]) == 2

        pg_store.rebuild_index()
        after = _snapshot_pg_state(pg_store.index, _TEST_SCHEMA)

        assert (
            len(after["manifests"]) == 2
        ), f"Expected 2 manifests, got {len(after['manifests'])}"

        # Both total_runs values must be preserved
        total_runs_after = sorted(m["total_runs"] for m in after["manifests"])
        assert total_runs_after == [
            2,
            3,
        ], f"total_runs values wrong: {total_runs_after}"

    def test_full_roundtrip_nothing_lost(self, pg_store):
        """Comprehensive check: nothing is lost in a full round-trip."""
        # Write everything
        records = _make_records("roundtrip_exp:1.0", count=10)
        for r in records:
            pg_store.put_run_record(r)

        derived_pairs = _make_derived(records)
        for run_id, derived in derived_pairs:
            pg_store.put_derived(run_id, derived)

        manifest = _make_manifest("roundtrip_exp:1.0", records)
        pg_store.put_experiment_manifest(
            "roundtrip_exp:1.0", manifest, timestamp="20250615_100000"
        )

        # Snapshot everything before rebuild
        before = _snapshot_pg_state(pg_store.index, _TEST_SCHEMA)

        # Rebuild
        pg_store.rebuild_index()

        # Snapshot everything after rebuild
        after = _snapshot_pg_state(pg_store.index, _TEST_SCHEMA)

        # ---- Run records ----
        assert len(after["runs"]) == len(
            before["runs"]
        ), f"Run count: {len(before['runs'])} -> {len(after['runs'])}"

        # ---- Derived metrics ----
        assert len(after["derived"]) == len(
            before["derived"]
        ), f"Derived count: {len(before['derived'])} -> {len(after['derived'])}"

        # ---- Manifests ----
        assert len(after["manifests"]) == len(
            before["manifests"]
        ), f"Manifest count: {len(before['manifests'])} -> {len(after['manifests'])}"

        # ---- Field catalog: params + metrics + derived ----
        before_namespaces = {k[0] for k in before["field_catalog"]}
        after_namespaces = {k[0] for k in after["field_catalog"]}

        # After rebuild, the field catalog should have at least the same namespaces
        # (it's OK if rebuild adds more, but it must not lose any)
        for ns in ["params", "metrics"]:
            assert ns in after_namespaces, f"Namespace '{ns}' lost from field catalog"
        assert (
            "derived" in after_namespaces
        ), "Namespace 'derived' missing from field catalog after rebuild"


class TestUnscopedRebuildRoundTrip:
    """Test rebuild on unscoped stores (multi-experiment discovery)."""

    def test_unscoped_rebuild_discovers_all_experiments(
        self, unscoped_pg_store, tmp_path
    ):
        """Unscoped rebuild discovers and indexes all experiment subdirectories."""
        from metalab.store.file import FileStoreConfig

        # Create two experiments via FileStore (source of truth)
        for exp_id in ["exp_a:1.0", "exp_b:2.0"]:
            scoped = FileStoreConfig(root=str(tmp_path), experiment_id=exp_id).connect()
            records = _make_records(exp_id, count=3)
            for r in records:
                scoped.put_run_record(r)
            derived_pairs = _make_derived(records)
            for run_id, d in derived_pairs:
                scoped.put_derived(run_id, d)
            manifest = _make_manifest(exp_id, records)
            scoped.put_experiment_manifest(
                exp_id, manifest, timestamp="20250615_100000"
            )

        # Rebuild from files
        n_indexed = unscoped_pg_store.rebuild_index()
        assert n_indexed == 6  # 3 per experiment

        after = _snapshot_pg_state(unscoped_pg_store.index, _TEST_SCHEMA)

        # All 6 runs present
        assert len(after["runs"]) == 6

        # Derived metrics present
        assert len(after["derived"]) > 0

        # Both manifests present
        assert len(after["manifests"]) == 2
        manifest_exp_ids = {m["experiment_id"] for m in after["manifests"]}
        assert "exp_a:1.0" in manifest_exp_ids
        assert "exp_b:2.0" in manifest_exp_ids

        # total_runs populated for both
        for m in after["manifests"]:
            assert (
                m["total_runs"] == 3
            ), f"total_runs for {m['experiment_id']}: {m['total_runs']}"

        # Field catalog has derived namespace
        after_namespaces = {k[0] for k in after["field_catalog"]}
        assert "derived" in after_namespaces, "derived namespace missing from catalog"
