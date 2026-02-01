"""
Tests for store capability protocols.

These tests verify that:
1. FileStore implements expected capabilities
2. PostgresStore implements expected capabilities (when psycopg available)
3. isinstance() checks work with capability protocols
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from metalab.store import FileStore
from metalab.store.capabilities import (
    SupportsArtifactOpen,
    SupportsExperimentManifests,
    SupportsLogPath,
    SupportsWorkingDirectory,
)

# Check if psycopg is available for PostgresStore tests
try:
    import psycopg
    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False


class TestFileStoreCapabilities:
    """Tests for FileStore capability implementations."""

    def test_supports_working_directory(self, tmp_path: Path) -> None:
        """FileStore should implement SupportsWorkingDirectory."""
        store = FileStore(tmp_path)
        assert isinstance(store, SupportsWorkingDirectory)
        assert store.get_working_directory() == tmp_path

    def test_supports_experiment_manifests(self, tmp_path: Path) -> None:
        """FileStore should implement SupportsExperimentManifests."""
        store = FileStore(tmp_path)
        assert isinstance(store, SupportsExperimentManifests)

        # Test writing a manifest
        manifest = {"name": "test_experiment", "total_runs": 10}
        store.put_experiment_manifest("test:1", manifest, timestamp="20240101_120000")

        # Verify manifest was written
        manifest_path = tmp_path / "experiments" / "test_1_20240101_120000.json"
        assert manifest_path.exists()

        # Test reading it back
        loaded = store.get_experiment_manifest("test:1")
        assert loaded is not None
        assert loaded["name"] == "test_experiment"

    def test_supports_log_path(self, tmp_path: Path) -> None:
        """FileStore should implement SupportsLogPath."""
        store = FileStore(tmp_path)
        assert isinstance(store, SupportsLogPath)

        run_id = "test_run_123"
        log_path = store.get_log_path(run_id, "run")

        # Should return a path in the logs directory
        assert log_path.parent == tmp_path / "logs"
        assert run_id in str(log_path)

    def test_supports_artifact_open(self, tmp_path: Path) -> None:
        """FileStore should implement SupportsArtifactOpen."""
        store = FileStore(tmp_path)
        assert isinstance(store, SupportsArtifactOpen)

        # Create a test file
        test_content = b"test artifact content"
        artifact_path = tmp_path / "test_artifact.txt"
        artifact_path.write_bytes(test_content)

        # Open via capability method
        with store.open_artifact(str(artifact_path)) as f:
            content = f.read()
        assert content == test_content

    def test_open_artifact_not_found(self, tmp_path: Path) -> None:
        """open_artifact should raise FileNotFoundError for missing files."""
        store = FileStore(tmp_path)

        with pytest.raises(FileNotFoundError):
            store.open_artifact(str(tmp_path / "nonexistent.txt"))


class TestCapabilityProtocolIsInstance:
    """Tests that isinstance() works with capability protocols."""

    def test_file_store_all_capabilities(self, tmp_path: Path) -> None:
        """FileStore should pass isinstance for all capabilities it supports."""
        store = FileStore(tmp_path)

        # All capabilities that FileStore should implement
        assert isinstance(store, SupportsWorkingDirectory)
        assert isinstance(store, SupportsExperimentManifests)
        assert isinstance(store, SupportsLogPath)
        assert isinstance(store, SupportsArtifactOpen)

    def test_non_store_object_not_capability(self) -> None:
        """Non-store objects should not pass capability isinstance checks."""
        regular_dict = {"key": "value"}
        assert not isinstance(regular_dict, SupportsWorkingDirectory)
        assert not isinstance(regular_dict, SupportsExperimentManifests)
        assert not isinstance(regular_dict, SupportsLogPath)
        assert not isinstance(regular_dict, SupportsArtifactOpen)


class TestWorkingDirectoryCapability:
    """Tests specifically for SupportsWorkingDirectory capability."""

    def test_get_working_directory_returns_path(self, tmp_path: Path) -> None:
        """get_working_directory should return a Path object."""
        store = FileStore(tmp_path)
        result = store.get_working_directory()
        assert isinstance(result, Path)

    def test_working_directory_matches_root(self, tmp_path: Path) -> None:
        """Working directory should match the store's root."""
        store = FileStore(tmp_path)
        assert store.get_working_directory() == store.root

    def test_working_directory_is_absolute(self, tmp_path: Path) -> None:
        """Working directory should be an absolute path."""
        store = FileStore(tmp_path)
        wd = store.get_working_directory()
        assert wd.is_absolute()


@pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
class TestPostgresStoreComposition:
    """Tests for PostgresStore composition with FileStore."""

    def test_postgres_store_requires_experiments_root(self, tmp_path: Path) -> None:
        """PostgresStore should require experiments_root parameter."""
        from metalab.store.postgres import PostgresStore

        # Should raise ValueError without experiments_root
        with pytest.raises(ValueError, match="experiments_root"):
            # Use a fake connection string - will fail on connection but
            # should fail on validation first
            PostgresStore("postgresql://fake@localhost/db")

    def test_postgres_store_delegates_to_file_store(self, tmp_path: Path) -> None:
        """PostgresStore should have an internal FileStore for delegation."""
        from metalab.store.postgres import PostgresStore
        from unittest.mock import patch, MagicMock

        # Mock the connection pool at the psycopg_pool module level
        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()

            store = PostgresStore(
                "postgresql://fake@localhost/db",
                experiments_root=tmp_path,
                auto_migrate=False,  # Skip schema creation
            )

            # Should have a FileStore instance
            assert store._file_store is not None
            assert isinstance(store._file_store, FileStore)
            assert store._file_store.root == tmp_path

    def test_postgres_store_implements_log_path(self, tmp_path: Path) -> None:
        """PostgresStore should implement SupportsLogPath via FileStore."""
        from metalab.store.postgres import PostgresStore
        from unittest.mock import patch, MagicMock

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()

            store = PostgresStore(
                "postgresql://fake@localhost/db",
                experiments_root=tmp_path,
                auto_migrate=False,
            )

            # Should implement SupportsLogPath
            assert isinstance(store, SupportsLogPath)

            # Should delegate to FileStore
            log_path = store.get_log_path("run123", "test")
            assert log_path == tmp_path / "logs" / "run123_test.log"

    def test_postgres_store_implements_working_directory(self, tmp_path: Path) -> None:
        """PostgresStore should implement SupportsWorkingDirectory via FileStore."""
        from metalab.store.postgres import PostgresStore
        from unittest.mock import patch, MagicMock

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()

            store = PostgresStore(
                "postgresql://fake@localhost/db",
                experiments_root=tmp_path,
                auto_migrate=False,
            )

            # Should implement SupportsWorkingDirectory
            assert isinstance(store, SupportsWorkingDirectory)

            # Should return experiments_root
            wd = store.get_working_directory()
            assert wd == tmp_path

    def test_postgres_store_experiment_id_creates_subdirectory(
        self, tmp_path: Path
    ) -> None:
        """PostgresStore with experiment_id should nest FileStore under sanitized id."""
        from metalab.store.postgres import PostgresStore
        from unittest.mock import patch, MagicMock

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()

            store = PostgresStore(
                "postgresql://fake@localhost/db",
                experiments_root=tmp_path,
                experiment_id="my_exp:1.0",
                auto_migrate=False,
            )

            # FileStore should be created at {experiments_root}/{safe_exp_id}/
            # my_exp:1.0 -> my_exp_1.0
            expected_path = tmp_path / "my_exp_1.0"
            assert store._file_store.root == expected_path

    def test_postgres_store_experiment_id_sanitizes_colon(
        self, tmp_path: Path
    ) -> None:
        """PostgresStore should replace colons with underscores in experiment_id."""
        from metalab.store.postgres import PostgresStore
        from unittest.mock import patch, MagicMock

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()

            # Test various experiment_id formats
            test_cases = [
                ("exp:1.0", "exp_1.0"),
                ("my_exp:2.0.1", "my_exp_2.0.1"),
                ("simple", "simple"),  # No colon should be unchanged
            ]

            for exp_id, expected_dir in test_cases:
                store = PostgresStore(
                    "postgresql://fake@localhost/db",
                    experiments_root=tmp_path,
                    experiment_id=exp_id,
                    auto_migrate=False,
                )
                expected_path = tmp_path / expected_dir
                assert store._file_store.root == expected_path, (
                    f"Failed for {exp_id}: expected {expected_path}, "
                    f"got {store._file_store.root}"
                )
