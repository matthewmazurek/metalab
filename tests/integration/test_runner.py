"""Integration tests for the runner."""

from __future__ import annotations

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
        # ThreadExecutor uses "thread:N" format
        assert "[thread:" in run_log
