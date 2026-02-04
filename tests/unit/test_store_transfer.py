"""
Tests for store transfer functionality.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from metalab.store import FileStore, FileStoreConfig, export_store
from metalab.types import Provenance, RunRecord, Status


def _make_test_record(run_id: str, experiment_id: str = "test:v1") -> RunRecord:
    """Create a test run record."""
    return RunRecord(
        run_id=run_id,
        experiment_id=experiment_id,
        status=Status.SUCCESS,
        context_fingerprint="ctx_fp",
        params_fingerprint="params_fp",
        seed_fingerprint="seed_fp",
        started_at=datetime.now(),
        finished_at=datetime.now(),
        duration_ms=1000,
        metrics={"loss": 0.5, "accuracy": 0.9},
        provenance=Provenance(host="test"),
        params_resolved={"lr": 0.01, "batch_size": 32},
    )


class TestExportStore:
    """Tests for export_store function."""

    def test_export_empty_store(self):
        """Export from empty store."""
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as dst_dir:
                src = FileStoreConfig(root=src_dir).connect()
                dst = FileStoreConfig(root=dst_dir).connect()

                counts = export_store(src, dst)

                assert counts["runs"] == 0
                assert counts["derived"] == 0
                assert counts["logs"] == 0

    def test_export_single_run(self):
        """Export a single run record."""
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as dst_dir:
                src = FileStoreConfig(root=src_dir).connect()
                dst = FileStoreConfig(root=dst_dir).connect()

                # Create source record
                record = _make_test_record("run_001")
                src.put_run_record(record)

                # Export
                counts = export_store(src, dst)

                assert counts["runs"] == 1

                # Verify in destination
                exported = dst.get_run_record("run_001")
                assert exported is not None
                assert exported.run_id == "run_001"
                assert exported.metrics == {"loss": 0.5, "accuracy": 0.9}

    def test_export_with_derived(self):
        """Export run records with derived metrics."""
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as dst_dir:
                src = FileStoreConfig(root=src_dir).connect()
                dst = FileStoreConfig(root=dst_dir).connect()

                # Create source data
                record = _make_test_record("run_002")
                src.put_run_record(record)
                src.put_derived("run_002", {"final_loss": 0.1})

                # Export
                counts = export_store(src, dst, include_derived=True)

                assert counts["runs"] == 1
                assert counts["derived"] == 1

                # Verify derived in destination
                derived = dst.get_derived("run_002")
                assert derived == {"final_loss": 0.1}

    def test_export_with_logs(self):
        """Export run records with logs."""
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as dst_dir:
                src = FileStoreConfig(root=src_dir).connect()
                dst = FileStoreConfig(root=dst_dir).connect()

                # Create source data
                record = _make_test_record("run_003")
                src.put_run_record(record)
                src.put_log("run_003", "run", "Test log content")

                # Export
                counts = export_store(src, dst, include_logs=True)

                assert counts["runs"] == 1
                assert counts["logs"] == 1

                # Verify log in destination
                log = dst.get_log("run_003", "run")
                assert log == "Test log content"

    def test_export_filter_by_experiment(self):
        """Export only specific experiment."""
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as dst_dir:
                src = FileStoreConfig(root=src_dir).connect()
                dst = FileStoreConfig(root=dst_dir).connect()

                # Create source data for two experiments
                src.put_run_record(_make_test_record("run_a1", "exp_a:v1"))
                src.put_run_record(_make_test_record("run_a2", "exp_a:v1"))
                src.put_run_record(_make_test_record("run_b1", "exp_b:v1"))

                # Export only exp_a
                counts = export_store(src, dst, experiment_id="exp_a:v1")

                assert counts["runs"] == 2

                # Verify only exp_a records exported
                assert dst.run_exists("run_a1")
                assert dst.run_exists("run_a2")
                assert not dst.run_exists("run_b1")

    def test_export_skip_existing(self):
        """Export skips existing records by default."""
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as dst_dir:
                src = FileStoreConfig(root=src_dir).connect()
                dst = FileStoreConfig(root=dst_dir).connect()

                # Create source and destination records
                record1 = _make_test_record("run_001")
                record1_modified = _make_test_record("run_001")
                record1_modified.metrics["loss"] = 0.99  # Different value

                src.put_run_record(record1)
                dst.put_run_record(record1_modified)

                # Export (should skip existing)
                counts = export_store(src, dst, overwrite=False)

                assert counts["runs"] == 0

                # Destination should have original value
                exported = dst.get_run_record("run_001")
                assert exported.metrics["loss"] == 0.99

    def test_export_overwrite(self):
        """Export overwrites existing records when specified."""
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as dst_dir:
                src = FileStoreConfig(root=src_dir).connect()
                dst = FileStoreConfig(root=dst_dir).connect()

                # Create source and destination records
                record1 = _make_test_record("run_001")
                record1.metrics["loss"] = 0.5

                record1_old = _make_test_record("run_001")
                record1_old.metrics["loss"] = 0.99

                src.put_run_record(record1)
                dst.put_run_record(record1_old)

                # Export with overwrite
                counts = export_store(src, dst, overwrite=True)

                assert counts["runs"] == 1

                # Destination should have new value
                exported = dst.get_run_record("run_001")
                assert exported.metrics["loss"] == 0.5
