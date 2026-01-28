"""Tests for schema loading and migration."""

from __future__ import annotations

from datetime import datetime

from metalab.schema import (
    SCHEMA_VERSION,
    dump_artifact_descriptor,
    dump_run_record,
    load_artifact_descriptor,
    load_run_record,
)
from metalab.types import ArtifactDescriptor, Provenance, RunRecord, Status


class TestSchemaVersion:
    """Tests for schema versioning."""

    def test_schema_version_exists(self):
        """Schema version should be defined."""
        assert SCHEMA_VERSION == "0.1"


class TestLoadRunRecord:
    """Tests for load_run_record()."""

    def test_load_complete_record(self):
        """Load a complete record."""
        data = {
            "run_id": "abc123",
            "experiment_id": "test:1.0",
            "status": "success",
            "context_fingerprint": "ctx",
            "params_fingerprint": "params",
            "seed_fingerprint": "seed",
            "started_at": "2024-01-01T10:00:00",
            "finished_at": "2024-01-01T10:01:00",
            "duration_ms": 60000,
            "metrics": {"accuracy": 0.95},
            "provenance": {"code_hash": "abc"},
            "tags": ["test"],
        }
        record = load_run_record(data)
        assert record.run_id == "abc123"
        assert record.status == Status.SUCCESS
        assert record.metrics["accuracy"] == 0.95

    def test_load_minimal_record(self):
        """Load with missing optional fields."""
        data = {
            "run_id": "abc123",
            "status": "failed",
        }
        record = load_run_record(data)
        assert record.run_id == "abc123"
        assert record.status == Status.FAILED
        assert record.metrics == {}
        assert record.tags == []

    def test_load_with_missing_timestamps(self):
        """Missing timestamps should default to now."""
        data = {"run_id": "test", "status": "success"}
        record = load_run_record(data)
        assert record.started_at is not None
        assert record.finished_at is not None

    def test_load_with_artifacts(self):
        """Artifacts should be loaded."""
        data = {
            "run_id": "test",
            "status": "success",
            "artifacts": [
                {
                    "artifact_id": "art1",
                    "name": "output",
                    "kind": "json",
                    "format": "json",
                    "uri": "/path/to/file",
                }
            ],
        }
        record = load_run_record(data)
        assert len(record.artifacts) == 1
        assert record.artifacts[0].name == "output"


class TestLoadArtifactDescriptor:
    """Tests for load_artifact_descriptor()."""

    def test_load_complete(self):
        """Load complete descriptor."""
        data = {
            "artifact_id": "abc",
            "name": "output",
            "kind": "numpy",
            "format": "npz",
            "uri": "/path/to/file.npz",
            "content_hash": "hash123",
            "size_bytes": 1024,
            "metadata": {"shape": [100, 100]},
        }
        desc = load_artifact_descriptor(data)
        assert desc.artifact_id == "abc"
        assert desc.kind == "numpy"
        assert desc.size_bytes == 1024

    def test_load_minimal(self):
        """Load with defaults."""
        data = {"artifact_id": "abc", "name": "test"}
        desc = load_artifact_descriptor(data)
        assert desc.kind == "blob"
        assert desc.format == "binary"
        assert desc.metadata == {}


class TestDumpRunRecord:
    """Tests for dump_run_record()."""

    def test_dump_and_load_roundtrip(self):
        """Dump and load should preserve data."""
        record = RunRecord(
            run_id="test123",
            experiment_id="exp:1.0",
            status=Status.SUCCESS,
            context_fingerprint="ctx",
            params_fingerprint="params",
            seed_fingerprint="seed",
            started_at=datetime(2024, 1, 1, 10, 0, 0),
            finished_at=datetime(2024, 1, 1, 10, 1, 0),
            duration_ms=60000,
            metrics={"accuracy": 0.95, "loss": 0.05},
            provenance=Provenance(code_hash="abc123"),
            params_resolved={"learning_rate": 0.01, "batch_size": 32},
            tags=["test", "example"],
        )

        data = dump_run_record(record)
        restored = load_run_record(data)

        assert restored.run_id == record.run_id
        assert restored.status == record.status
        assert restored.metrics == record.metrics
        assert restored.params_resolved == record.params_resolved
        assert restored.tags == record.tags

    def test_dump_includes_schema_version(self):
        """Dumped data should include schema version."""
        record = RunRecord.success(run_id="test")
        data = dump_run_record(record)
        assert "_schema_version" in data
        assert data["_schema_version"] == SCHEMA_VERSION


class TestDumpArtifactDescriptor:
    """Tests for dump_artifact_descriptor()."""

    def test_dump_and_load_roundtrip(self):
        """Dump and load should preserve data."""
        desc = ArtifactDescriptor(
            artifact_id="art123",
            name="predictions",
            kind="numpy",
            format="npz",
            uri="/path/to/file.npz",
            content_hash="abc",
            size_bytes=2048,
            metadata={"shape": [100, 10]},
        )

        data = dump_artifact_descriptor(desc)
        restored = load_artifact_descriptor(data)

        assert restored.artifact_id == desc.artifact_id
        assert restored.name == desc.name
        assert restored.metadata == desc.metadata


class TestRunningStatus:
    """Tests for RUNNING status support."""

    def test_running_status_exists(self):
        """RUNNING should be a valid status value."""
        assert Status.RUNNING.value == "running"

    def test_running_factory_method(self):
        """RunRecord.running() should create a running record."""
        record = RunRecord.running(
            run_id="test123",
            experiment_id="exp:1.0",
            context_fingerprint="ctx",
            params_fingerprint="params",
            seed_fingerprint="seed",
        )
        assert record.status == Status.RUNNING
        assert record.run_id == "test123"
        assert record.duration_ms == 0  # Placeholder

    def test_running_record_roundtrip(self):
        """RUNNING status should survive serialization round-trip."""
        record = RunRecord.running(
            run_id="test123",
            experiment_id="exp:1.0",
            context_fingerprint="ctx",
            params_fingerprint="params",
            seed_fingerprint="seed",
            params_resolved={"x": 10},
        )

        data = dump_run_record(record)
        restored = load_run_record(data)

        assert restored.status == Status.RUNNING
        assert restored.run_id == record.run_id
        assert restored.params_resolved == record.params_resolved

    def test_load_running_status_from_string(self):
        """Load should handle 'running' status string."""
        data = {
            "run_id": "test",
            "status": "running",
            "experiment_id": "exp:1.0",
        }
        record = load_run_record(data)
        assert record.status == Status.RUNNING
