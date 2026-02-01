"""Unit tests for executor core module."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from metalab.executor.core import execute_payload
from metalab.types import RunRecord, Status


@dataclass(frozen=True)
class MockContext:
    """Mock context for testing."""
    name: str = "test"


class MockOperation:
    """Mock operation for testing."""

    ref = "test_module:test_op"
    name = "test_op"
    code_hash = "abc123"

    def run(self, context, params, seeds, runtime, capture):
        capture.metric("test_metric", 42)
        return RunRecord.success()


class FailingOperation:
    """Operation that raises an exception."""

    ref = "test_module:failing_op"
    name = "failing_op"
    code_hash = "def456"

    def run(self, context, params, seeds, runtime, capture):
        raise ValueError("Intentional failure")


class MockSeedBundle:
    """Mock seed bundle for testing."""

    def fingerprint(self):
        return "seed123"


class MockStore:
    """Mock store that doesn't implement SupportsLogPath."""

    def __init__(self, tmp_path: Path):
        self._tmp_path = tmp_path
        self.put_run_record = MagicMock()
        self.put_log = MagicMock()
        self.put_artifact = MagicMock()

    # Not implementing get_log_path means Capture will use artifact_dir for logs


class TestRootLoggerPreservation:
    """Tests that root logger handlers are preserved during execution."""

    def test_root_logger_handlers_preserved_after_execution(self, tmp_path: Path):
        """Root logger handlers should not be removed by execute_payload."""
        # Set up a custom root logger handler
        root_logger = logging.getLogger()
        original_handler_count = len(root_logger.handlers)

        # Add a test handler
        test_handler = logging.StreamHandler()
        test_handler.setFormatter(logging.Formatter("TEST: %(message)s"))
        root_logger.addHandler(test_handler)

        # Create mock store
        mock_store = MockStore(tmp_path)

        # Execute with third-party log capture enabled
        with patch("metalab.runtime.create_runtime") as mock_runtime:
            mock_runtime.return_value = MagicMock(
                scratch_dir=tmp_path,
            )

            try:
                execute_payload(
                    run_id="test-run-001",
                    experiment_id="test:1.0",
                    context_spec=MockContext(),
                    params_resolved={"x": 1},
                    seed_bundle=MockSeedBundle(),
                    fingerprints={"context": "ctx", "params": "prm", "seed": "sd"},
                    metadata={},
                    operation=MockOperation(),
                    store=mock_store,
                    worker_id="test:1",
                    capture_third_party_logs=True,
                )
            except Exception:
                pass  # Ignore errors, we're testing handler cleanup

        # Verify our handler is still there
        assert test_handler in root_logger.handlers
        # Clean up
        root_logger.removeHandler(test_handler)

    def test_root_logger_level_not_changed(self, tmp_path: Path):
        """Root logger level should not be changed by execute_payload."""
        root_logger = logging.getLogger()
        original_level = root_logger.level

        # Set a specific level
        root_logger.setLevel(logging.WARNING)

        mock_store = MockStore(tmp_path)

        with patch("metalab.runtime.create_runtime") as mock_runtime:
            mock_runtime.return_value = MagicMock(
                scratch_dir=tmp_path,
            )

            try:
                execute_payload(
                    run_id="test-run-002",
                    experiment_id="test:1.0",
                    context_spec=MockContext(),
                    params_resolved={"x": 1},
                    seed_bundle=MockSeedBundle(),
                    fingerprints={"context": "ctx", "params": "prm", "seed": "sd"},
                    metadata={},
                    operation=MockOperation(),
                    store=mock_store,
                    worker_id="test:1",
                    capture_third_party_logs=True,
                )
            except Exception:
                pass

        # Level should be unchanged
        assert root_logger.level == logging.WARNING
        # Restore
        root_logger.setLevel(original_level)


class TestDurablePersistence:
    """Tests that records are persisted directly in execute_payload."""

    def test_success_record_persisted(self, tmp_path: Path):
        """Successful run record should be persisted via store.put_run_record."""
        mock_store = MockStore(tmp_path)

        with patch("metalab.runtime.create_runtime") as mock_runtime:
            mock_runtime.return_value = MagicMock(
                scratch_dir=tmp_path,
            )

            result = execute_payload(
                run_id="test-run-003",
                experiment_id="test:1.0",
                context_spec=MockContext(),
                params_resolved={"x": 1},
                seed_bundle=MockSeedBundle(),
                fingerprints={"context": "ctx", "params": "prm", "seed": "sd"},
                metadata={},
                operation=MockOperation(),
                store=mock_store,
                worker_id="test:1",
            )

        # Should have called put_run_record at least twice:
        # 1. For RUNNING record
        # 2. For final SUCCESS record
        assert mock_store.put_run_record.call_count >= 2

        # Final call should be with SUCCESS status
        final_call = mock_store.put_run_record.call_args_list[-1]
        final_record = final_call[0][0]
        assert final_record.status == Status.SUCCESS

    def test_failed_record_persisted(self, tmp_path: Path):
        """Failed run record should be persisted via store.put_run_record."""
        mock_store = MockStore(tmp_path)

        with patch("metalab.runtime.create_runtime") as mock_runtime:
            mock_runtime.return_value = MagicMock(
                scratch_dir=tmp_path,
            )

            result = execute_payload(
                run_id="test-run-004",
                experiment_id="test:1.0",
                context_spec=MockContext(),
                params_resolved={"x": 1},
                seed_bundle=MockSeedBundle(),
                fingerprints={"context": "ctx", "params": "prm", "seed": "sd"},
                metadata={},
                operation=FailingOperation(),
                store=mock_store,
                worker_id="test:1",
            )

        # Should have called put_run_record at least twice:
        # 1. For RUNNING record
        # 2. For final FAILED record
        assert mock_store.put_run_record.call_count >= 2

        # Final call should be with FAILED status
        final_call = mock_store.put_run_record.call_args_list[-1]
        final_record = final_call[0][0]
        assert final_record.status == Status.FAILED
        assert final_record.error["type"] == "ValueError"

    def test_persistence_failure_does_not_crash(self, tmp_path: Path):
        """Persistence failures should be logged but not crash execution."""
        mock_store = MockStore(tmp_path)
        # Make the final put_run_record fail
        call_count = [0]

        def failing_put(record):
            call_count[0] += 1
            if call_count[0] > 1:  # Fail on non-RUNNING records
                raise RuntimeError("DB unavailable")

        mock_store.put_run_record = MagicMock(side_effect=failing_put)

        with patch("metalab.runtime.create_runtime") as mock_runtime:
            mock_runtime.return_value = MagicMock(
                scratch_dir=tmp_path,
            )

            # Should not raise despite persistence failure
            result = execute_payload(
                run_id="test-run-005",
                experiment_id="test:1.0",
                context_spec=MockContext(),
                params_resolved={"x": 1},
                seed_bundle=MockSeedBundle(),
                fingerprints={"context": "ctx", "params": "prm", "seed": "sd"},
                metadata={},
                operation=MockOperation(),
                store=mock_store,
                worker_id="test:1",
            )

        # Should still return a result
        assert result is not None
        assert result.status == Status.SUCCESS
