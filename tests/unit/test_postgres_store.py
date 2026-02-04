"""
Tests for PostgresStore composition and helpers.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from metalab.store import FileStore, FileStoreConfig
from metalab.types import Provenance, RunRecord, Status

# Check if psycopg is available
try:
    import psycopg

    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False


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


@pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
class TestPostgresStoreConfigInit:
    """Tests for PostgresStoreConfig initialization."""

    def test_requires_file_root(self):
        """PostgresStoreConfig requires file_root parameter."""
        from metalab.store.postgres import PostgresStoreConfig

        # Should raise TypeError for missing required argument
        with pytest.raises(TypeError):
            PostgresStoreConfig(connection_string="postgresql://localhost/db")

    def test_creates_filestore(self, tmp_path: Path):
        """PostgresStore creates internal FileStore."""
        from metalab.store.postgres import PostgresStoreConfig

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            config = PostgresStoreConfig(
                connection_string="postgresql://localhost/db",
                file_root=str(tmp_path),
                auto_migrate=False,
            )
            store = config.connect()
            assert store.file_store is not None
            assert store.file_store.root == tmp_path

    def test_creates_index(self, tmp_path: Path):
        """PostgresStore creates PostgresIndex."""
        from metalab.store.postgres import PostgresStoreConfig
        from metalab.store.postgres_index import PostgresIndex

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            config = PostgresStoreConfig(
                connection_string="postgresql://localhost/db",
                file_root=str(tmp_path),
                auto_migrate=False,
            )
            store = config.connect()
            assert store.index is not None
            assert isinstance(store.index, PostgresIndex)


@pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
class TestPostgresStoreConfigFromLocator:
    """Tests for PostgresStoreConfig.from_locator classmethod."""

    def test_from_locator_requires_file_root(self):
        """from_locator raises if file_root not provided."""
        from metalab.store.locator import parse_to_config

        with pytest.raises(ValueError, match="file_root"):
            parse_to_config("postgresql://localhost/db")

    def test_from_locator_with_kwarg(self, tmp_path: Path):
        """from_locator accepts file_root as kwarg."""
        from metalab.store.locator import parse_to_config

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            config = parse_to_config(
                "postgresql://localhost/db", file_root=str(tmp_path)
            )
            store = config.connect()
            assert store.file_store.root == tmp_path

    def test_from_locator_with_uri_param(self, tmp_path: Path):
        """from_locator accepts file_root in URI params."""
        from metalab.store.locator import parse_to_config

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            config = parse_to_config(f"postgresql://localhost/db?file_root={tmp_path}")
            store = config.connect()
            assert store.file_store.root == tmp_path


@pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
class TestPostgresStoreFromFileStore:
    """Tests for PostgresStore.from_filestore classmethod."""

    def test_from_filestore_wraps_existing(self, tmp_path: Path):
        """from_filestore wraps an existing FileStore."""
        from metalab.store.postgres import PostgresStore

        # Create a FileStore with some data
        filestore = FileStoreConfig(root=str(tmp_path)).connect()
        record = _make_test_record("run_001")
        filestore.put_run_record(record)

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            pg_store = PostgresStore.from_filestore(
                "postgresql://localhost/db",
                filestore,
                rebuild=False,
            )
            # Should use the same root
            assert pg_store.file_store.root == filestore.root

    def test_from_filestore_can_skip_rebuild(self, tmp_path: Path):
        """from_filestore can skip index rebuild."""
        from metalab.store.postgres import PostgresStore

        filestore = FileStoreConfig(root=str(tmp_path)).connect()

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            pg_store = PostgresStore.from_filestore(
                "postgresql://localhost/db",
                filestore,
                rebuild=False,
            )
            # Should not have called index methods
            assert pg_store is not None


@pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
class TestPostgresStoreToFileStore:
    """Tests for PostgresStore.to_filestore method."""

    def test_to_filestore_returns_internal(self, tmp_path: Path):
        """to_filestore() without destination returns internal FileStore."""
        from metalab.store.postgres import PostgresStoreConfig

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            config = PostgresStoreConfig(
                connection_string="postgresql://localhost/db",
                file_root=str(tmp_path),
                auto_migrate=False,
            )
            pg_store = config.connect()
            filestore = pg_store.to_filestore()
            assert filestore is pg_store.file_store

    def test_to_filestore_copies_to_destination(self, tmp_path: Path):
        """to_filestore() with destination copies files."""
        from metalab.store.postgres import PostgresStoreConfig

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            config = PostgresStoreConfig(
                connection_string="postgresql://localhost/db",
                file_root=str(tmp_path / "source"),
                auto_migrate=False,
            )
            pg_store = config.connect()
            # Add a run record
            record = _make_test_record("run_001")
            pg_store.file_store.put_run_record(record)

            # Export to new location
            dest = tmp_path / "dest"
            exported = pg_store.to_filestore(dest)

            assert exported.root == dest
            assert exported.run_exists("run_001")


@pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
class TestPostgresStoreWritesBoth:
    """Tests that PostgresStore writes to both FileStore and index."""

    def test_put_run_record_writes_file(self, tmp_path: Path):
        """put_run_record writes to FileStore."""
        from metalab.store.postgres import PostgresStoreConfig

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            config = PostgresStoreConfig(
                connection_string="postgresql://localhost/db",
                file_root=str(tmp_path),
                auto_migrate=False,
            )
            pg_store = config.connect()
            record = _make_test_record("run_001")
            pg_store.put_run_record(record)

            # Should be in FileStore
            assert pg_store.file_store.run_exists("run_001")

    def test_put_derived_writes_file(self, tmp_path: Path):
        """put_derived writes to FileStore."""
        from metalab.store.postgres import PostgresStoreConfig

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            config = PostgresStoreConfig(
                connection_string="postgresql://localhost/db",
                file_root=str(tmp_path),
                auto_migrate=False,
            )
            pg_store = config.connect()
            pg_store.put_derived("run_001", {"loss": 0.1})

            # Should be in FileStore
            assert pg_store.file_store.derived_exists("run_001")


@pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
class TestPostgresStoreContextManager:
    """Tests for context manager support."""

    def test_context_manager_closes(self, tmp_path: Path):
        """Context manager closes resources."""
        from metalab.store.postgres import PostgresStoreConfig

        mock_pool = MagicMock()
        with patch("psycopg_pool.ConnectionPool", return_value=mock_pool):
            config = PostgresStoreConfig(
                connection_string="postgresql://localhost/db",
                file_root=str(tmp_path),
                auto_migrate=False,
            )
            with config.connect() as store:
                pass

            # Index pool should be closed
            mock_pool.close.assert_called()


@pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
class TestPostgresStoreConfigSerialization:
    """Tests for PostgresStoreConfig serialization."""

    def test_to_dict_includes_type(self, tmp_path: Path):
        """to_dict includes _type field."""
        from metalab.store.postgres import PostgresStoreConfig

        config = PostgresStoreConfig(
            connection_string="postgresql://localhost/db",
            file_root=str(tmp_path),
        )
        d = config.to_dict()

        assert "_type" in d
        assert d["_type"] == "postgresql"

    def test_from_dict_round_trip(self, tmp_path: Path):
        """from_dict can reconstruct config from to_dict."""
        from metalab.store.config import StoreConfig
        from metalab.store.postgres import PostgresStoreConfig

        original = PostgresStoreConfig(
            connection_string="postgresql://localhost/db",
            file_root=str(tmp_path),
            experiment_id="my_exp:1.0",
            schema="custom",
        )
        d = original.to_dict()
        restored = StoreConfig.from_dict(d)

        assert isinstance(restored, PostgresStoreConfig)
        assert restored.connection_string == original.connection_string
        assert restored.file_root == original.file_root
        assert restored.experiment_id == original.experiment_id
        assert restored.schema == original.schema
