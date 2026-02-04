"""Integration tests for FileStore."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from metalab.schema import SCHEMA_VERSION
from metalab.store.file import FileStore, FileStoreConfig
from metalab.types import ArtifactDescriptor, RunRecord, Status


@pytest.fixture
def store(tmp_path: Path) -> FileStore:
    """Create a FileStore in a temporary directory."""
    return FileStoreConfig(root=str(tmp_path)).connect()


class TestFileStoreLayout:
    """Tests for FileStore directory layout."""

    def test_creates_layout(self, tmp_path: Path):
        """Store should create expected directories."""
        FileStoreConfig(root=str(tmp_path)).connect()

        assert (tmp_path / "runs").is_dir()
        assert (tmp_path / "artifacts").is_dir()
        assert (tmp_path / "logs").is_dir()
        assert (tmp_path / ".locks").is_dir()
        assert (tmp_path / "_meta.json").is_file()

    def test_meta_file_content(self, tmp_path: Path):
        """Meta file should contain schema version."""
        FileStoreConfig(root=str(tmp_path)).connect()

        meta = json.loads((tmp_path / "_meta.json").read_text())
        assert meta["schema_version"] == SCHEMA_VERSION


class TestRunRecords:
    """Tests for run record operations."""

    def test_put_and_get(self, store: FileStore):
        """Put and get run record."""
        record = RunRecord(
            run_id="test123",
            experiment_id="exp:1.0",
            status=Status.SUCCESS,
            context_fingerprint="ctx",
            params_fingerprint="params",
            seed_fingerprint="seed",
            started_at=datetime.now(),
            finished_at=datetime.now(),
            duration_ms=1000,
            metrics={"accuracy": 0.95},
        )

        store.put_run_record(record)
        retrieved = store.get_run_record("test123")

        assert retrieved is not None
        assert retrieved.run_id == "test123"
        assert retrieved.status == Status.SUCCESS
        assert retrieved.metrics["accuracy"] == 0.95

    def test_get_nonexistent(self, store: FileStore):
        """Get nonexistent record returns None."""
        assert store.get_run_record("nonexistent") is None

    def test_run_exists(self, store: FileStore):
        """run_exists should work correctly."""
        record = RunRecord.success(run_id="test123")
        assert not store.run_exists("test123")

        store.put_run_record(record)
        assert store.run_exists("test123")

    def test_list_run_records(self, store: FileStore):
        """List all run records."""
        for i in range(3):
            store.put_run_record(RunRecord.success(run_id=f"run{i}"))

        records = store.list_run_records()
        assert len(records) == 3

    def test_list_filtered_by_experiment(self, store: FileStore):
        """List records filtered by experiment."""
        store.put_run_record(RunRecord.success(run_id="r1", experiment_id="exp1:1.0"))
        store.put_run_record(RunRecord.success(run_id="r2", experiment_id="exp1:1.0"))
        store.put_run_record(RunRecord.success(run_id="r3", experiment_id="exp2:1.0"))

        exp1_records = store.list_run_records(experiment_id="exp1:1.0")
        assert len(exp1_records) == 2


class TestArtifacts:
    """Tests for artifact operations."""

    def test_put_artifact_from_path(self, store: FileStore, tmp_path: Path):
        """Store artifact from file path."""
        # Create a test file
        test_file = tmp_path / "test_artifact" / "data.json"
        test_file.parent.mkdir(parents=True)
        test_file.write_text('{"test": "data"}')

        descriptor = ArtifactDescriptor(
            artifact_id="art123",
            name="data",
            kind="json",
            format="json",
            uri=str(test_file),
        )

        result = store.put_artifact(test_file, descriptor)
        assert result.uri.endswith(".json")
        assert Path(result.uri).exists()

    def test_list_artifacts(self, store: FileStore, tmp_path: Path):
        """List artifacts for a run."""
        # Create test file
        test_file = tmp_path / "test_run" / "data.json"
        test_file.parent.mkdir(parents=True)
        test_file.write_text('{"test": "data"}')

        descriptor = ArtifactDescriptor(
            artifact_id="art123",
            name="data",
            kind="json",
            format="json",
            uri=str(test_file),
            metadata={
                "_run_id": "test_run"
            },  # Required for proper artifact association
        )

        store.put_artifact(test_file, descriptor)
        artifacts = store.list_artifacts("test_run")
        assert len(artifacts) == 1
        assert artifacts[0].name == "data"


class TestLogs:
    """Tests for log operations."""

    def test_put_and_get_log(self, store: FileStore):
        """Store and retrieve log."""
        store.put_log("run123", "stdout", "Hello, World!")
        content = store.get_log("run123", "stdout")
        assert content == "Hello, World!"

    def test_get_nonexistent_log(self, store: FileStore):
        """Get nonexistent log returns None."""
        assert store.get_log("run123", "nonexistent") is None

    def test_multiple_logs(self, store: FileStore):
        """Store multiple logs for same run."""
        store.put_log("run123", "stdout", "Standard output")
        store.put_log("run123", "stderr", "Standard error")

        assert store.get_log("run123", "stdout") == "Standard output"
        assert store.get_log("run123", "stderr") == "Standard error"

    def test_log_filename_format(self, store: FileStore):
        """Log files use {run_id}_{name}.log format."""
        store.put_log("abc123def456", "run", "Output")
        content = store.get_log("abc123def456", "run")
        assert content == "Output"

        # Verify flat file structure: {run_id}_{name}.log
        log_path = store.root / "logs" / "abc123def456_run.log"
        assert log_path.exists()


class TestAtomicity:
    """Tests for atomic operations."""

    def test_atomic_write_creates_valid_json(self, store: FileStore):
        """Atomic writes should create valid JSON."""
        record = RunRecord.success(run_id="test123", metrics={"x": 1})
        store.put_run_record(record)

        # Read raw file and verify it's valid JSON
        path = store.root / "runs" / "test123.json"
        data = json.loads(path.read_text())
        assert data["run_id"] == "test123"

    def test_delete_run(self, store: FileStore, tmp_path: Path):
        """Delete should remove run and related files."""
        # Create run with artifacts and logs
        store.put_run_record(RunRecord.success(run_id="test123"))
        store.put_log("test123", "stdout", "log content")

        # Verify exists
        assert store.run_exists("test123")

        # Delete
        store.delete_run("test123")

        # Verify gone
        assert not store.run_exists("test123")
        assert store.get_log("test123", "stdout") is None
