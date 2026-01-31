"""
Tests for store transfer functionality.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from metalab.store import FileStore, export_store, FallbackStore
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
                src = FileStore(src_dir)
                dst = FileStore(dst_dir)
                
                counts = export_store(src, dst)
                
                assert counts["runs"] == 0
                assert counts["derived"] == 0
                assert counts["logs"] == 0

    def test_export_single_run(self):
        """Export a single run record."""
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as dst_dir:
                src = FileStore(src_dir)
                dst = FileStore(dst_dir)
                
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
                src = FileStore(src_dir)
                dst = FileStore(dst_dir)
                
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
                src = FileStore(src_dir)
                dst = FileStore(dst_dir)
                
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
                src = FileStore(src_dir)
                dst = FileStore(dst_dir)
                
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
                src = FileStore(src_dir)
                dst = FileStore(dst_dir)
                
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
                src = FileStore(src_dir)
                dst = FileStore(dst_dir)
                
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


class TestFallbackStore:
    """Tests for FallbackStore wrapper."""

    def test_fallback_uses_primary_when_available(self):
        """FallbackStore uses primary when available."""
        with tempfile.TemporaryDirectory() as primary_dir:
            with tempfile.TemporaryDirectory() as fallback_dir:
                primary = FileStore(primary_dir)
                fallback = FileStore(fallback_dir)
                store = FallbackStore(primary, fallback)
                
                # Write record
                record = _make_test_record("run_001")
                store.put_run_record(record)
                
                # Should be in primary
                assert primary.run_exists("run_001")
                # Should not be in fallback (write_to_both=False)
                assert not fallback.run_exists("run_001")

    def test_fallback_writes_to_both(self):
        """FallbackStore writes to both when configured."""
        with tempfile.TemporaryDirectory() as primary_dir:
            with tempfile.TemporaryDirectory() as fallback_dir:
                primary = FileStore(primary_dir)
                fallback = FileStore(fallback_dir)
                store = FallbackStore(primary, fallback, write_to_both=True)
                
                # Write record
                record = _make_test_record("run_001")
                store.put_run_record(record)
                
                # Should be in both
                assert primary.run_exists("run_001")
                assert fallback.run_exists("run_001")

    def test_fallback_reads_from_primary_first(self):
        """FallbackStore reads from primary first."""
        with tempfile.TemporaryDirectory() as primary_dir:
            with tempfile.TemporaryDirectory() as fallback_dir:
                primary = FileStore(primary_dir)
                fallback = FileStore(fallback_dir)
                store = FallbackStore(primary, fallback)
                
                # Write different values to each
                record_primary = _make_test_record("run_001")
                record_primary.metrics["source"] = "primary"
                
                record_fallback = _make_test_record("run_001")
                record_fallback.metrics["source"] = "fallback"
                
                primary.put_run_record(record_primary)
                fallback.put_run_record(record_fallback)
                
                # Read should return primary
                result = store.get_run_record("run_001")
                assert result.metrics["source"] == "primary"

    def test_fallback_reads_from_fallback_when_not_in_primary(self):
        """FallbackStore reads from fallback when not in primary."""
        with tempfile.TemporaryDirectory() as primary_dir:
            with tempfile.TemporaryDirectory() as fallback_dir:
                primary = FileStore(primary_dir)
                fallback = FileStore(fallback_dir)
                store = FallbackStore(primary, fallback)
                
                # Write only to fallback
                record = _make_test_record("run_001")
                fallback.put_run_record(record)
                
                # Read should return fallback
                result = store.get_run_record("run_001")
                assert result is not None
                assert result.run_id == "run_001"

    def test_fallback_derived_operations(self):
        """FallbackStore handles derived metrics."""
        with tempfile.TemporaryDirectory() as primary_dir:
            with tempfile.TemporaryDirectory() as fallback_dir:
                primary = FileStore(primary_dir)
                fallback = FileStore(fallback_dir)
                store = FallbackStore(primary, fallback, write_to_both=True)
                
                # Create run first
                record = _make_test_record("run_001")
                store.put_run_record(record)
                
                # Write derived
                store.put_derived("run_001", {"final": 0.1})
                
                # Should be in both
                assert primary.derived_exists("run_001")
                assert fallback.derived_exists("run_001")
                
                # Read should work
                result = store.get_derived("run_001")
                assert result == {"final": 0.1}

    def test_fallback_log_operations(self):
        """FallbackStore handles logs."""
        with tempfile.TemporaryDirectory() as primary_dir:
            with tempfile.TemporaryDirectory() as fallback_dir:
                primary = FileStore(primary_dir)
                fallback = FileStore(fallback_dir)
                store = FallbackStore(primary, fallback)
                
                # Create run first
                record = _make_test_record("run_001")
                primary.put_run_record(record)
                
                # Write log
                store.put_log("run_001", "run", "test log")
                
                # Read should work
                result = store.get_log("run_001", "run")
                assert result == "test log"
