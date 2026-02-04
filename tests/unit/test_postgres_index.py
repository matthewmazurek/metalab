"""
Tests for PostgresIndex query acceleration layer.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
class TestPostgresIndexInit:
    """Tests for PostgresIndex initialization."""

    def test_requires_psycopg(self):
        """PostgresIndex imports psycopg on init."""
        from metalab.store.postgres_index import PostgresIndex

        # Should not raise ImportError (psycopg is available in test env)
        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            index = PostgresIndex(
                "postgresql://localhost/db",
                auto_migrate=False,
            )
            assert index is not None

    def test_default_schema(self):
        """Default schema is 'public'."""
        from metalab.store.postgres_index import PostgresIndex

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            index = PostgresIndex(
                "postgresql://localhost/db",
                auto_migrate=False,
            )
            assert index._schema == "public"

    def test_custom_schema(self):
        """Can specify custom schema."""
        from metalab.store.postgres_index import PostgresIndex

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            index = PostgresIndex(
                "postgresql://localhost/db",
                schema="custom_schema",
                auto_migrate=False,
            )
            assert index._schema == "custom_schema"


@pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
class TestPostgresIndexTableNames:
    """Tests for table name generation."""

    def test_table_names_include_schema(self):
        """Table names are schema-qualified."""
        from metalab.store.postgres_index import PostgresIndex

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            index = PostgresIndex(
                "postgresql://localhost/db",
                schema="myschema",
                auto_migrate=False,
            )
            assert index._table("runs") == "myschema.runs"
            assert index._table("derived") == "myschema.derived"


@pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
class TestPostgresIndexContextManager:
    """Tests for context manager support."""

    def test_context_manager_closes_pool(self):
        """Context manager closes connection pool on exit."""
        from metalab.store.postgres_index import PostgresIndex

        mock_pool = MagicMock()
        with patch("psycopg_pool.ConnectionPool", return_value=mock_pool):
            with PostgresIndex(
                "postgresql://localhost/db",
                auto_migrate=False,
            ) as index:
                pass

            # Pool should be closed after context exit
            mock_pool.close.assert_called_once()


class TestStripMetalabParams:
    """Tests for _strip_metalab_params helper function."""

    def test_strips_file_root(self):
        """Strips file_root param from connection string."""
        from metalab.store.postgres_index import _strip_metalab_params

        uri = "postgresql://localhost/db?file_root=/path/to/files"
        result = _strip_metalab_params(uri)
        assert result == "postgresql://localhost/db"

    def test_strips_schema_param(self):
        """Strips schema param from connection string."""
        from metalab.store.postgres_index import _strip_metalab_params

        uri = "postgresql://localhost/db?schema=custom"
        result = _strip_metalab_params(uri)
        assert result == "postgresql://localhost/db"

    def test_strips_multiple_metalab_params(self):
        """Strips multiple metalab-specific params."""
        from metalab.store.postgres_index import _strip_metalab_params

        uri = "postgresql://localhost/db?file_root=/path&schema=custom"
        result = _strip_metalab_params(uri)
        assert result == "postgresql://localhost/db"

    def test_preserves_postgres_params(self):
        """Preserves params that psycopg understands."""
        from metalab.store.postgres_index import _strip_metalab_params

        uri = "postgresql://localhost/db?sslmode=require&connect_timeout=10"
        result = _strip_metalab_params(uri)
        assert "sslmode=require" in result
        assert "connect_timeout=10" in result

    def test_preserves_postgres_params_when_stripping(self):
        """Preserves postgres params while stripping metalab params."""
        from metalab.store.postgres_index import _strip_metalab_params

        uri = "postgresql://localhost/db?sslmode=require&file_root=/path&connect_timeout=10"
        result = _strip_metalab_params(uri)
        assert "sslmode=require" in result
        assert "connect_timeout=10" in result
        assert "file_root" not in result

    def test_no_params_unchanged(self):
        """URI without params is unchanged."""
        from metalab.store.postgres_index import _strip_metalab_params

        uri = "postgresql://localhost/db"
        result = _strip_metalab_params(uri)
        assert result == uri

    def test_preserves_credentials(self):
        """Preserves username and password in URI."""
        from metalab.store.postgres_index import _strip_metalab_params

        uri = "postgresql://user:pass@localhost:5432/db?file_root=/path"
        result = _strip_metalab_params(uri)
        assert "user:pass@localhost:5432" in result
        assert "file_root" not in result


@pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
class TestPostgresIndexStripsParams:
    """Tests that PostgresIndex strips metalab params before connecting."""

    def test_connection_pool_receives_clean_uri(self):
        """ConnectionPool receives URI without metalab-specific params."""
        from metalab.store.postgres_index import PostgresIndex

        with patch("psycopg_pool.ConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()
            PostgresIndex(
                "postgresql://localhost/db?file_root=/path&sslmode=require",
                auto_migrate=False,
            )

            # Check what URI was passed to ConnectionPool
            call_args = mock_pool.call_args
            connection_string = call_args[0][0]

            assert "file_root" not in connection_string
            assert "sslmode=require" in connection_string
