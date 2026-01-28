"""Integration tests for the runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import metalab
from metalab.types import Status


@dataclass(frozen=True)
class SimpleContext:
    """Simple context for testing."""

    name: str = "test"


@metalab.operation
def simple_operation(
    context: Any,
    params: dict[str, Any],
    seeds: metalab.SeedBundle,
    runtime: metalab.Runtime,
    capture: metalab.Capture,
) -> None:
    """Simple operation that captures a metric."""
    capture.metric("result", params["x"] * 2)
    # No return needed - success is implicit


@metalab.operation
def failing_operation(
    context: Any,
    params: dict[str, Any],
    seeds: metalab.SeedBundle,
    runtime: metalab.Runtime,
    capture: metalab.Capture,
) -> metalab.RunRecord:
    """Operation that fails."""
    raise ValueError("Intentional failure")


@metalab.operation
def operation_with_artifact(
    context: Any,
    params: dict[str, Any],
    seeds: metalab.SeedBundle,
    runtime: metalab.Runtime,
    capture: metalab.Capture,
) -> None:
    """Operation that captures an artifact."""
    import numpy as np

    # Create a simple array based on params
    arr = np.array([[params["x"] * i for i in range(5)] for _ in range(3)])
    capture.artifact("data", arr, kind="numpy")
    capture.metric("sum", float(arr.sum()))
    capture.metric("threshold", params.get("threshold", 0.5))


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    """Get a temporary store path."""
    return tmp_path / "runs"


class TestBasicExecution:
    """Tests for basic experiment execution."""

    def test_single_run(self, store_path: Path):
        """Run a single parameter configuration."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[5]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        assert len(result) == 1
        assert result[0].status == Status.SUCCESS

    def test_multiple_params(self, store_path: Path):
        """Run with multiple parameter values."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1, 2, 3]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        assert len(result) == 3
        assert all(r.status == Status.SUCCESS for r in result)

    def test_multiple_replicates(self, store_path: Path):
        """Run with multiple replicates."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[5]),
            seeds=metalab.seeds(base=42, replicates=3),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        assert len(result) == 3

    def test_params_times_replicates(self, store_path: Path):
        """Total runs = params * replicates."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1, 2]),
            seeds=metalab.seeds(base=42, replicates=3),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        # 2 param values * 3 replicates = 6 runs
        assert len(result) == 6


class TestResume:
    """Tests for resume functionality."""

    def test_resume_skips_existing(self, store_path: Path):
        """Resume should skip already successful runs."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1, 2, 3]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        # First run
        handle1 = metalab.run(exp, store=str(store_path), resume=True)
        result1 = handle1.result()
        assert len(result1) == 3

        # Second run should skip all (but still return 3 records from store)
        handle2 = metalab.run(exp, store=str(store_path), resume=True)
        result2 = handle2.result()

        # Should still return 3 records (loaded from store)
        assert len(result2) == 3

    def test_resume_false_reruns(self, store_path: Path):
        """resume=False should rerun everything."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        # Run twice without resume
        handle1 = metalab.run(exp, store=str(store_path), resume=False)
        result1 = handle1.result()
        handle2 = metalab.run(exp, store=str(store_path), resume=False)
        result2 = handle2.result()

        # Both should execute
        assert len(result1) == 1
        assert len(result2) == 1


class TestFailures:
    """Tests for failure handling."""

    def test_failed_run_captured(self, store_path: Path):
        """Failed runs should be captured with error info."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=failing_operation,
            params=metalab.grid(x=[1]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        assert len(result) == 1
        assert result[0].status == Status.FAILED
        assert result[0].error is not None
        assert "Intentional failure" in result[0].error["message"]


class TestResultHandle:
    """Tests for Results functionality (accessed via RunHandle.result())."""

    def test_table_returns_list(self, store_path: Path):
        """table() should return list of dicts by default."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1, 2]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()
        table = result.table()

        assert isinstance(table, list)
        assert len(table) == 2
        assert "run_id" in table[0]
        assert "status" in table[0]

    def test_filter_by_status(self, store_path: Path):
        """filter() should work by status."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1, 2, 3]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()
        successful = result.filter(status="success")

        assert len(successful) == 3

    def test_summary(self, store_path: Path):
        """summary() should return statistics."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1, 2]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()
        summary = result.summary()

        assert summary["total_runs"] == 2
        assert "success" in summary["by_status"]

    def test_run_params_property(self, store_path: Path):
        """Run.params should expose resolved parameters."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[10, 20]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        # Check that params are accessible
        run = result[0]
        assert "x" in run.params
        assert run.params["x"] in [10, 20]

    def test_to_dataframe_basic(self, store_path: Path):
        """to_dataframe() should return a DataFrame with params and metrics."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1, 2, 3]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        import pandas as pd

        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        # Check param column is prefixed
        assert "param_x" in df.columns
        # Check metric is included
        assert "result" in df.columns
        # Check record fields are included by default
        assert "run_id" in df.columns
        assert "status" in df.columns

    def test_to_dataframe_exclude_options(self, store_path: Path):
        """to_dataframe() should respect include_* flags."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        # Exclude everything
        df = result.to_dataframe(
            include_params=False, include_metrics=False, include_record=False
        )

        assert len(df) == 1
        assert "param_x" not in df.columns
        assert "result" not in df.columns
        assert "run_id" not in df.columns

    def test_to_dataframe_with_simple_reducer(self, store_path: Path):
        """to_dataframe() should apply simple artifact reducers."""
        pytest.importorskip("numpy")

        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=operation_with_artifact,
            params=metalab.grid(x=[2, 3]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        def simple_reducer(arr):
            return {"arr_mean": float(arr.mean()), "arr_max": float(arr.max())}

        df = result.to_dataframe(artifact_reducers={"data": simple_reducer})

        assert "arr_mean" in df.columns
        assert "arr_max" in df.columns
        assert len(df) == 2

    def test_to_dataframe_with_context_aware_reducer(self, store_path: Path):
        """to_dataframe() should pass run context to 2-arg reducers."""
        pytest.importorskip("numpy")

        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=operation_with_artifact,
            params=metalab.grid(x=[2], threshold=[0.5, 1.0]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        def context_reducer(arr, run):
            # Use threshold from run params
            threshold = run.params.get("threshold", 0.5)
            return {"above_threshold": float((arr > threshold).mean())}

        df = result.to_dataframe(artifact_reducers={"data": context_reducer})

        assert "above_threshold" in df.columns
        assert len(df) == 2


class TestRunHandle:
    """Tests for RunHandle functionality."""

    def test_status_property(self, store_path: Path):
        """status should return RunStatus with counts."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1, 2]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))

        # After completion, status should show all completed
        result = handle.result()
        status = handle.status

        assert status.total == 2
        assert status.completed == 2
        assert status.failed == 0

    def test_is_complete_property(self, store_path: Path):
        """is_complete should return True after all runs finish."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        handle.result()  # Wait for completion

        assert handle.is_complete

    def test_job_id_property(self, store_path: Path):
        """job_id should return a string identifier."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))

        assert isinstance(handle.job_id, str)
        assert len(handle.job_id) > 0


@metalab.operation
def logging_operation(
    context: Any,
    params: dict[str, Any],
    seeds: metalab.SeedBundle,
    runtime: metalab.Runtime,
    capture: metalab.Capture,
) -> None:
    """Operation that uses capture.log() for logging."""
    capture.log(f"Starting with x={params['x']}")
    capture.log(f"Debug info for x={params['x']}", level="debug")

    # Do some work
    result = params["x"] * 2
    capture.metric("result", result)

    capture.log(f"Computed result={result}")
    if params["x"] > 5:
        capture.log("x is large", level="warning")


class TestCaptureLog:
    """Tests for capture.log() integration."""

    def test_capture_log_basic(self, store_path: Path):
        """Test that capture.log() stores logs."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=logging_operation,
            params=metalab.grid(x=[5]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        assert len(result) == 1
        assert result[0].status == Status.SUCCESS

        # Check that logs were stored
        from metalab.store.file import FileStore

        store = FileStore(str(store_path))
        run_id = result[0].run_id

        run_log = store.get_log(run_id, "run")
        assert run_log is not None
        assert "Starting with x=5" in run_log
        assert "Debug info for x=5" in run_log
        assert "Computed result=10" in run_log
        # Check timestamp format is present
        assert "[INFO   ]" in run_log
        assert "[DEBUG  ]" in run_log

    def test_capture_log_with_warning(self, store_path: Path):
        """Test that capture.log() handles warning levels."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=logging_operation,
            params=metalab.grid(x=[10]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        assert len(result) == 1
        assert result[0].status == Status.SUCCESS

        from metalab.store.file import FileStore

        store = FileStore(str(store_path))
        run_id = result[0].run_id

        run_log = store.get_log(run_id, "run")
        assert run_log is not None
        assert "x is large" in run_log
        assert "[WARNING]" in run_log

    def test_capture_log_multiple_runs(self, store_path: Path):
        """Test that capture.log() works correctly for multiple runs."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=logging_operation,
            params=metalab.grid(x=[1, 2, 3]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        assert len(result) == 3

        from metalab.store.file import FileStore

        store = FileStore(str(store_path))

        for run in result:
            run_log = store.get_log(run.run_id, "run")
            assert run_log is not None
            # Each should contain its own log messages
            x_val = run.metrics.get("result", 0) // 2  # result = x * 2
            assert f"Starting with x={x_val}" in run_log

    def test_capture_log_worker_id(self, store_path: Path):
        """Test that capture.log() includes worker ID."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=logging_operation,
            params=metalab.grid(x=[5]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        from metalab.store.file import FileStore

        store = FileStore(str(store_path))
        run_id = result[0].run_id

        run_log = store.get_log(run_id, "run")
        assert run_log is not None
        # ThreadExecutor uses "thread:N" format in the logger name
        assert "thread:" in run_log

    def test_capture_subscribe_logger(self, store_path: Path):
        """Test that capture.subscribe_logger() captures third-party logs."""
        import logging

        # Create a "third-party" logger that doesn't propagate
        third_party_logger = logging.getLogger("my_library")
        third_party_logger.setLevel(logging.DEBUG)
        third_party_logger.propagate = False  # Like dynamo-release

        @metalab.operation
        def op_with_subscribe(context, params, seeds, capture, runtime):
            # Subscribe to the third-party logger
            capture.subscribe_logger("my_library")

            capture.log("Operation started")

            # Log from the third-party logger
            third_party_logger.info("Third-party info message")
            third_party_logger.warning("Third-party warning")

            capture.log("Operation completed")

        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=op_with_subscribe,
            params=metalab.grid(x=[1]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        from metalab.store.file import FileStore

        store = FileStore(str(store_path))
        run_id = result[0].run_id

        run_log = store.get_log(run_id, "run")
        assert run_log is not None

        # Check that our operation logs are captured
        assert "Operation started" in run_log
        assert "Operation completed" in run_log

        # Check that third-party logger output is captured
        assert "Third-party info message" in run_log
        assert "Third-party warning" in run_log
        assert "[my_library]" in run_log  # Logger name should appear

    def test_capture_logger_property(self, store_path: Path):
        """Test that capture.logger provides access to Python's logging API."""

        @metalab.operation
        def op_with_logger_property(context, params, seeds, capture, runtime):
            # Use the logger property directly
            capture.logger.info("Using logger.info")
            capture.logger.debug("Using logger.debug")
            capture.logger.warning("Using logger.warning")

            # Can also use capture.log() interchangeably
            capture.log("Using capture.log")

        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=op_with_logger_property,
            params=metalab.grid(x=[1]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        from metalab.store.file import FileStore

        store = FileStore(str(store_path))
        run_id = result[0].run_id

        run_log = store.get_log(run_id, "run")
        assert run_log is not None

        # All logging methods should work
        assert "Using logger.info" in run_log
        assert "Using logger.debug" in run_log
        assert "Using logger.warning" in run_log
        assert "Using capture.log" in run_log


class TestExperimentManifest:
    """Tests for experiment manifest creation."""

    def test_manifest_created_on_run(self, store_path: Path):
        """Experiment manifest should be created when run() is called."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            description="Test experiment",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1, 2, 3]),
            seeds=metalab.seeds(base=42, replicates=2),
            tags=["test", "integration"],
        )

        handle = metalab.run(exp, store=str(store_path))
        handle.result()

        # Check that experiments directory exists
        experiments_dir = store_path / "experiments"
        assert experiments_dir.exists()

        # Check that a manifest file was created
        manifest_files = list(experiments_dir.glob("test_1.0_*.json"))
        assert len(manifest_files) == 1

        # Load and verify content
        manifest = json.loads(manifest_files[0].read_text())
        assert manifest["experiment_id"] == "test:1.0"
        assert manifest["name"] == "test"
        assert manifest["version"] == "1.0"
        assert manifest["description"] == "Test experiment"
        assert manifest["tags"] == ["test", "integration"]
        assert manifest["total_runs"] == 6  # 3 params * 2 replicates

    def test_manifest_contains_param_source(self, store_path: Path):
        """Manifest should contain serialized param source."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1, 2, 3], y=[10, 20]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        handle.result()

        experiments_dir = store_path / "experiments"
        manifest_files = list(experiments_dir.glob("test_1.0_*.json"))
        manifest = json.loads(manifest_files[0].read_text())

        # Check params section
        assert manifest["params"]["type"] == "GridSource"
        assert manifest["params"]["spec"] == {"x": [1, 2, 3], "y": [10, 20]}
        assert manifest["params"]["total_cases"] == 6

    def test_manifest_contains_seed_plan(self, store_path: Path):
        """Manifest should contain serialized seed plan."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1]),
            seeds=metalab.seeds(base=123, replicates=5),
        )

        handle = metalab.run(exp, store=str(store_path))
        handle.result()

        experiments_dir = store_path / "experiments"
        manifest_files = list(experiments_dir.glob("test_1.0_*.json"))
        manifest = json.loads(manifest_files[0].read_text())

        # Check seeds section
        assert manifest["seeds"]["type"] == "SeedPlan"
        assert manifest["seeds"]["base"] == 123
        assert manifest["seeds"]["replicates"] == 5

    def test_manifest_contains_operation_info(self, store_path: Path):
        """Manifest should contain operation reference and code hash."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        handle.result()

        experiments_dir = store_path / "experiments"
        manifest_files = list(experiments_dir.glob("test_1.0_*.json"))
        manifest = json.loads(manifest_files[0].read_text())

        # Check operation section
        assert "operation" in manifest
        assert manifest["operation"]["name"] == "simple_operation"
        assert "ref" in manifest["operation"]
        assert "code_hash" in manifest["operation"]

    def test_manifest_has_timestamp(self, store_path: Path):
        """Manifest should include submission timestamp."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        handle.result()

        experiments_dir = store_path / "experiments"
        manifest_files = list(experiments_dir.glob("test_1.0_*.json"))
        manifest = json.loads(manifest_files[0].read_text())

        # Check timestamp
        assert "submitted_at" in manifest
        # Should be ISO format
        from datetime import datetime

        datetime.fromisoformat(manifest["submitted_at"])

    def test_multiple_runs_create_multiple_manifests(self, store_path: Path):
        """Multiple runs should create multiple timestamped manifests."""
        import time

        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        # First run
        handle1 = metalab.run(exp, store=str(store_path))
        handle1.result()

        # Wait a second to ensure different timestamp
        time.sleep(1.1)

        # Second run (with resume=False to force re-execution)
        handle2 = metalab.run(exp, store=str(store_path), resume=False)
        handle2.result()

        experiments_dir = store_path / "experiments"
        manifest_files = list(experiments_dir.glob("test_1.0_*.json"))
        assert len(manifest_files) == 2

    def test_manifest_contains_run_ids(self, store_path: Path):
        """Manifest should contain all expected run_ids."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1, 2, 3]),
            seeds=metalab.seeds(base=42, replicates=2),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        # Get the run IDs from the results
        result_run_ids = set(run.run_id for run in result)

        # Check the manifest
        experiments_dir = store_path / "experiments"
        manifest_files = list(experiments_dir.glob("test_1.0_*.json"))
        manifest = json.loads(manifest_files[0].read_text())

        # Manifest should have run_ids field
        assert "run_ids" in manifest
        assert manifest["run_ids"] is not None

        # All result run IDs should be in the manifest
        manifest_run_ids = set(manifest["run_ids"])
        assert result_run_ids == manifest_run_ids

        # Count should match total_runs
        assert len(manifest["run_ids"]) == manifest["total_runs"]


class TestRunningStatusRecords:
    """Tests for RUNNING status record functionality."""

    def test_running_record_written_before_completion(self, store_path: Path):
        """A running record should be written before the operation completes."""
        import threading
        import time

        # Track when we see the running record
        seen_running = threading.Event()
        seen_run_id = [None]

        @metalab.operation
        def slow_operation(
            context: Any,
            params: dict[str, Any],
            seeds: metalab.SeedBundle,
            runtime: metalab.Runtime,
            capture: metalab.Capture,
        ) -> None:
            """Operation that takes some time."""
            # Give the test time to check for RUNNING record
            time.sleep(0.5)
            capture.metric("result", params["x"] * 2)

        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=slow_operation,
            params=metalab.grid(x=[1]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        from metalab.store.file import FileStore

        # Use a monitor to check for running records
        def monitor_for_running():
            store = FileStore(str(store_path))
            for _ in range(20):  # Try for up to 2 seconds
                records = store.list_run_records()
                for record in records:
                    if record.status == Status.RUNNING:
                        seen_run_id[0] = record.run_id
                        seen_running.set()
                        return
                time.sleep(0.1)

        monitor_thread = threading.Thread(target=monitor_for_running)
        monitor_thread.start()

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        monitor_thread.join()

        # We should have seen a running record
        assert seen_running.is_set(), "Should have seen a RUNNING status record"
        # The run should have eventually succeeded
        assert len(result) == 1
        assert result[0].status == Status.SUCCESS
        # The run ID should match
        assert result[0].run_id == seen_run_id[0]

    def test_final_record_overwrites_running(self, store_path: Path):
        """The final SUCCESS/FAILED record should overwrite the RUNNING record."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        handle = metalab.run(exp, store=str(store_path))
        result = handle.result()

        # Check the final record in the store
        from metalab.store.file import FileStore

        store = FileStore(str(store_path))
        records = store.list_run_records()

        # Should only have one record per run_id
        assert len(records) == 1
        # Final status should be SUCCESS, not RUNNING
        assert records[0].status == Status.SUCCESS
