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


@metalab.operation(name="simple_op", version="1.0")
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


@metalab.operation(name="failing_op", version="1.0")
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

        result = metalab.run(exp, store=str(store_path))

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

        result = metalab.run(exp, store=str(store_path))

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

        result = metalab.run(exp, store=str(store_path))

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

        result = metalab.run(exp, store=str(store_path))

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
        result1 = metalab.run(exp, store=str(store_path), resume=True)
        assert len(result1) == 3

        # Second run should skip all
        events = []
        result2 = metalab.run(
            exp,
            store=str(store_path),
            resume=True,
            on_event=lambda e: events.append(e),
        )

        # Should still return 3 records (loaded from store)
        assert len(result2) == 3

        # Check that runs were skipped
        skipped = [e for e in events if e.kind.value == "run_skipped"]
        assert len(skipped) == 3

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
        result1 = metalab.run(exp, store=str(store_path), resume=False)
        result2 = metalab.run(exp, store=str(store_path), resume=False)

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

        result = metalab.run(exp, store=str(store_path))

        assert len(result) == 1
        assert result[0].status == Status.FAILED
        assert result[0].error is not None
        assert "Intentional failure" in result[0].error["message"]


class TestResultHandle:
    """Tests for ResultHandle functionality."""

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

        result = metalab.run(exp, store=str(store_path))
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

        result = metalab.run(exp, store=str(store_path))
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

        result = metalab.run(exp, store=str(store_path))
        summary = result.summary()

        assert summary["total_runs"] == 2
        assert "success" in summary["by_status"]


class TestEvents:
    """Tests for event emission."""

    def test_events_emitted(self, store_path: Path):
        """Events should be emitted during execution."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=simple_operation,
            params=metalab.grid(x=[1]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        events = []
        metalab.run(
            exp,
            store=str(store_path),
            on_event=lambda e: events.append(e),
        )

        # Should have at least started and finished events
        kinds = [e.kind.value for e in events]
        assert "run_started" in kinds
        assert "run_finished" in kinds


@metalab.operation(name="noisy_op", version="1.0")
def noisy_operation(
    context: Any,
    params: dict[str, Any],
    seeds: metalab.SeedBundle,
    runtime: metalab.Runtime,
    capture: metalab.Capture,
) -> None:
    """Operation that produces stdout/stderr/logging output."""
    import logging
    import sys

    print(f"Processing x={params['x']}")
    print(f"Error output for x={params['x']}", file=sys.stderr)

    logger = logging.getLogger("test_noisy")
    logger.setLevel(logging.INFO)
    logger.info(f"Log message for x={params['x']}")

    capture.metric("result", params["x"] * 2)


class TestOutputCapture:
    """Tests for output capture integration."""

    def test_capture_output_suppress(self, store_path: Path):
        """Test that output capture with suppress mode stores logs."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=noisy_operation,
            params=metalab.grid(x=[5]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        result = metalab.run(
            exp,
            store=str(store_path),
            capture_output=metalab.OutputCapture.suppress(),
        )

        assert len(result) == 1
        assert result[0].status == Status.SUCCESS

        # Check that logs were stored
        from metalab.store.file import FileStore

        store = FileStore(str(store_path))
        run_id = result[0].run_id

        stdout_log = store.get_log(run_id, "stdout")
        assert stdout_log is not None
        assert "Processing x=5" in stdout_log

        stderr_log = store.get_log(run_id, "stderr")
        assert stderr_log is not None
        assert "Error output for x=5" in stderr_log

        logging_log = store.get_log(run_id, "logging")
        assert logging_log is not None
        assert "Log message for x=5" in logging_log

    def test_capture_output_passthrough(self, store_path: Path):
        """Test that output capture with passthrough mode still stores logs."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=noisy_operation,
            params=metalab.grid(x=[10]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        result = metalab.run(
            exp,
            store=str(store_path),
            capture_output=metalab.OutputCapture.passthrough(),
        )

        assert len(result) == 1
        assert result[0].status == Status.SUCCESS

        # Check that logs were stored
        from metalab.store.file import FileStore

        store = FileStore(str(store_path))
        run_id = result[0].run_id

        stdout_log = store.get_log(run_id, "stdout")
        assert stdout_log is not None
        assert "Processing x=10" in stdout_log

    def test_capture_output_true_defaults(self, store_path: Path):
        """Test that capture_output=True uses sensible defaults."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=noisy_operation,
            params=metalab.grid(x=[7]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        # Without progress, should use passthrough
        result = metalab.run(
            exp,
            store=str(store_path),
            capture_output=True,
            progress=False,
        )

        assert len(result) == 1
        assert result[0].status == Status.SUCCESS

        # Logs should still be captured
        from metalab.store.file import FileStore

        store = FileStore(str(store_path))
        run_id = result[0].run_id

        stdout_log = store.get_log(run_id, "stdout")
        assert stdout_log is not None
        assert "Processing x=7" in stdout_log

    def test_capture_output_multiple_runs(self, store_path: Path):
        """Test that output capture works correctly for multiple runs."""
        exp = metalab.Experiment(
            name="test",
            version="1.0",
            context=SimpleContext(),
            operation=noisy_operation,
            params=metalab.grid(x=[1, 2, 3]),
            seeds=metalab.seeds(base=42, replicates=1),
        )

        result = metalab.run(
            exp,
            store=str(store_path),
            capture_output=metalab.OutputCapture.suppress(),
        )

        assert len(result) == 3

        # Each run should have its own captured output
        from metalab.store.file import FileStore

        store = FileStore(str(store_path))

        for run in result:
            stdout_log = store.get_log(run.run_id, "stdout")
            assert stdout_log is not None
            # Each should contain only its own x value
            x_val = run.metrics.get("result", 0) // 2  # result = x * 2
            assert f"Processing x={x_val}" in stdout_log
