"""Integration tests for derived metrics feature."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from metalab.result import Results, Run
from metalab.store.file import FileStore
from metalab.types import Metric, RunRecord, Status


@pytest.fixture
def store(tmp_path: Path) -> FileStore:
    """Create a FileStore in a temporary directory."""
    return FileStore(tmp_path)


@pytest.fixture
def sample_record() -> RunRecord:
    """Create a sample run record."""
    return RunRecord(
        run_id="test_run_abc",
        experiment_id="test_exp:1.0",
        status=Status.SUCCESS,
        context_fingerprint="ctx_fp",
        params_fingerprint="params_fp",
        seed_fingerprint="seed_fp",
        started_at=datetime.now(),
        finished_at=datetime.now(),
        duration_ms=1000,
        metrics={"accuracy": 0.95, "loss": 0.05},
        params_resolved={"lr": 0.01, "batch_size": 32},
    )


class TestFileStoreDerived:
    """Tests for FileStore derived metrics operations."""

    def test_layout_includes_derived_dir(self, tmp_path: Path):
        """Store should create derived directory."""
        FileStore(tmp_path)
        assert (tmp_path / "derived").is_dir()

    def test_put_and_get_derived(self, store: FileStore):
        """Put and get derived metrics."""
        derived = {"final_loss": 0.01, "converged": True}
        store.put_derived("run123", derived)

        result = store.get_derived("run123")
        assert result == derived

    def test_get_nonexistent_derived(self, store: FileStore):
        """Get nonexistent derived returns None."""
        assert store.get_derived("nonexistent") is None

    def test_derived_exists(self, store: FileStore):
        """derived_exists should work correctly."""
        assert not store.derived_exists("run123")

        store.put_derived("run123", {"x": 1})
        assert store.derived_exists("run123")

    def test_derived_stored_in_correct_path(self, store: FileStore):
        """Derived metrics stored in /derived/{run_id}.json."""
        store.put_derived("run456", {"metric": 42})

        path = store.root / "derived" / "run456.json"
        assert path.exists()

    def test_delete_run_removes_derived(self, store: FileStore):
        """delete_run should remove derived metrics."""
        store.put_run_record(RunRecord.success(run_id="run789"))
        store.put_derived("run789", {"x": 1})

        assert store.derived_exists("run789")
        store.delete_run("run789")
        assert not store.derived_exists("run789")


class TestRunDerivedProperty:
    """Tests for Run.derived property."""

    def test_derived_returns_empty_dict_when_none(
        self, store: FileStore, sample_record: RunRecord
    ):
        """derived property returns {} when no derived metrics exist."""
        store.put_run_record(sample_record)
        run = Run(sample_record, store)
        assert run.derived == {}

    def test_derived_returns_stored_metrics(
        self, store: FileStore, sample_record: RunRecord
    ):
        """derived property returns stored metrics."""
        store.put_run_record(sample_record)
        store.put_derived(sample_record.run_id, {"final_loss": 0.001})

        run = Run(sample_record, store)
        assert run.derived == {"final_loss": 0.001}


class TestResultsComputeDerived:
    """Tests for Results.compute_derived() method."""

    def test_compute_derived_stores_metrics(
        self, store: FileStore, sample_record: RunRecord
    ):
        """compute_derived stores metrics in store."""
        store.put_run_record(sample_record)
        results = Results(store, [sample_record])

        def simple_metric(run: Run) -> dict[str, Metric]:
            return {"computed": run.metrics["accuracy"] * 100}

        results.compute_derived([simple_metric])

        # Check stored
        derived = store.get_derived(sample_record.run_id)
        assert derived == {"computed": 95.0}

    def test_compute_derived_skips_existing(
        self, store: FileStore, sample_record: RunRecord
    ):
        """compute_derived skips runs with existing derived metrics."""
        store.put_run_record(sample_record)
        store.put_derived(sample_record.run_id, {"original": 1})
        results = Results(store, [sample_record])

        def new_metric(run: Run) -> dict[str, Metric]:
            return {"new": 2}

        results.compute_derived([new_metric])

        # Original should be preserved
        derived = store.get_derived(sample_record.run_id)
        assert derived == {"original": 1}

    def test_compute_derived_with_overwrite(
        self, store: FileStore, sample_record: RunRecord
    ):
        """compute_derived with overwrite=True replaces existing."""
        store.put_run_record(sample_record)
        store.put_derived(sample_record.run_id, {"original": 1})
        results = Results(store, [sample_record])

        def new_metric(run: Run) -> dict[str, Metric]:
            return {"new": 2}

        results.compute_derived([new_metric], overwrite=True)

        # Should be replaced
        derived = store.get_derived(sample_record.run_id)
        assert derived == {"new": 2}

    def test_compute_derived_multiple_runs(self, store: FileStore):
        """compute_derived works with multiple runs."""
        records = []
        for i in range(3):
            record = RunRecord.success(
                run_id=f"run_{i}",
                metrics={"value": i},
            )
            store.put_run_record(record)
            records.append(record)

        results = Results(store, records)

        def double_value(run: Run) -> dict[str, Metric]:
            return {"doubled": run.metrics["value"] * 2}

        results.compute_derived([double_value])

        # Check all runs have derived metrics
        for i in range(3):
            derived = store.get_derived(f"run_{i}")
            assert derived == {"doubled": i * 2}


class TestToDataframeWithDerived:
    """Tests for to_dataframe with derived metrics."""

    def test_include_derived_adds_columns(
        self, store: FileStore, sample_record: RunRecord
    ):
        """include_derived=True adds derived columns."""
        store.put_run_record(sample_record)
        store.put_derived(
            sample_record.run_id, {"final_loss": 0.001, "converged": True}
        )

        results = Results(store, [sample_record])
        df = results.to_dataframe(include_derived=True)

        assert "derived_final_loss" in df.columns
        assert "derived_converged" in df.columns
        assert df["derived_final_loss"].iloc[0] == 0.001
        assert df["derived_converged"].iloc[0] == True  # noqa: E712 (JSON bool)

    def test_derived_metrics_on_the_fly(
        self, store: FileStore, sample_record: RunRecord
    ):
        """derived_metrics parameter computes on-the-fly."""
        store.put_run_record(sample_record)
        results = Results(store, [sample_record])

        def compute_accuracy_pct(run: Run) -> dict[str, Metric]:
            return {"accuracy_pct": run.metrics["accuracy"] * 100}

        df = results.to_dataframe(derived_metrics=[compute_accuracy_pct])

        assert "accuracy_pct" in df.columns
        assert df["accuracy_pct"].iloc[0] == 95.0

    def test_on_the_fly_not_persisted(self, store: FileStore, sample_record: RunRecord):
        """On-the-fly derived_metrics are NOT persisted to store."""
        store.put_run_record(sample_record)
        results = Results(store, [sample_record])

        def compute_something(run: Run) -> dict[str, Metric]:
            return {"computed": 42}

        results.to_dataframe(derived_metrics=[compute_something])

        # Should NOT be stored
        assert store.get_derived(sample_record.run_id) is None


class TestDerivedMetricsDoNotAffectFingerprint:
    """Tests ensuring derived metrics don't affect run fingerprints."""

    def test_payload_derived_refs_not_in_fingerprint(self):
        """derived_metric_refs should not be included in fingerprint computation."""
        from metalab.executor.payload import RunPayload
        from metalab.seeds.bundle import SeedBundle

        payload1 = RunPayload(
            run_id="run1",
            experiment_id="exp:1.0",
            context_spec={},
            params_resolved={"lr": 0.01},
            seed_bundle=SeedBundle(root_seed=42),
            store_locator="/tmp/store",
            fingerprints={"context": "a", "params": "b", "seed": "c"},
            derived_metric_refs=None,
        )

        payload2 = RunPayload(
            run_id="run1",
            experiment_id="exp:1.0",
            context_spec={},
            params_resolved={"lr": 0.01},
            seed_bundle=SeedBundle(root_seed=42),
            store_locator="/tmp/store",
            fingerprints={"context": "a", "params": "b", "seed": "c"},
            derived_metric_refs=["some.module:metric_func"],
        )

        # The fingerprints dict should be identical regardless of derived_metric_refs
        assert payload1.fingerprints == payload2.fingerprints

        # Serialization should handle None correctly
        dict1 = payload1.to_dict()
        dict2 = payload2.to_dict()

        assert "derived_metric_refs" not in dict1
        assert dict2["derived_metric_refs"] == ["some.module:metric_func"]
