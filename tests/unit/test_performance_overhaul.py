"""
Tests for the Atlas performance overhaul (metalab-core side).

Covers:
- Batch indexing (batch_index_records, batch_index_derived)
- Incremental field catalog updates (_update_catalog_for_record)
- Schema version bump to v3
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from metalab.types import Provenance, RunRecord, Status

# Check if psycopg is available
try:
    import psycopg

    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False


def _make_test_record(
    run_id: str,
    experiment_id: str = "test:v1",
    params: dict | None = None,
    metrics: dict | None = None,
) -> RunRecord:
    """Create a test run record."""
    return RunRecord(
        run_id=run_id,
        experiment_id=experiment_id,
        status=Status.SUCCESS,
        context_fingerprint="ctx_fp",
        params_fingerprint="params_fp",
        seed_fingerprint="seed_fp",
        started_at=datetime.now(tz=timezone.utc),
        finished_at=datetime.now(tz=timezone.utc),
        duration_ms=1000,
        metrics=metrics or {"loss": 0.5, "accuracy": 0.9},
        provenance=Provenance(host="test"),
        params_resolved=params or {"lr": 0.01, "batch_size": 32},
    )


# =========================================================================
# Phase 3: Batch indexing tests
# =========================================================================


@pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
class TestBatchIndexRecords:
    """Tests for batch_index_records and batch_index_derived."""

    def test_batch_index_records_uses_copy(self):
        """batch_index_records uses COPY + temp table in a single transaction."""
        from metalab.store.postgres_index import PostgresIndex

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_copy_ctx = MagicMock()
        mock_cur.copy.return_value.__enter__ = MagicMock(return_value=mock_copy_ctx)
        mock_cur.copy.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("psycopg_pool.ConnectionPool", return_value=mock_pool):
            index = PostgresIndex(
                "postgresql://localhost/db",
                auto_migrate=False,
            )
            index._schema_ensured = True  # Skip deferred migration

            records = [
                _make_test_record("run_001"),
                _make_test_record("run_002"),
                _make_test_record("run_003"),
            ]

            index.batch_index_records(records)

            # Should use COPY for bulk loading
            mock_cur.copy.assert_called_once()

            # Should write one row per record
            assert mock_copy_ctx.write_row.call_count == 3

            # Should commit once
            mock_conn.commit.assert_called_once()

    def test_batch_index_records_empty_is_noop(self):
        """batch_index_records with empty list does nothing."""
        from metalab.store.postgres_index import PostgresIndex

        mock_pool = MagicMock()
        with patch("psycopg_pool.ConnectionPool", return_value=mock_pool):
            index = PostgresIndex(
                "postgresql://localhost/db",
                auto_migrate=False,
            )

            index.batch_index_records([])

            # Should not attempt to get a connection
            mock_pool.connection.assert_not_called()

    def test_batch_index_derived_uses_copy(self):
        """batch_index_derived uses COPY + temp table in a single transaction."""
        from metalab.store.postgres_index import PostgresIndex

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_copy_ctx = MagicMock()
        mock_cur.copy.return_value.__enter__ = MagicMock(return_value=mock_copy_ctx)
        mock_cur.copy.return_value.__exit__ = MagicMock(return_value=False)
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("psycopg_pool.ConnectionPool", return_value=mock_pool):
            index = PostgresIndex(
                "postgresql://localhost/db",
                auto_migrate=False,
            )
            index._schema_ensured = True

            pairs = [
                ("run_001", {"derived_loss": 0.1}),
                ("run_002", {"derived_loss": 0.2}),
            ]

            index.batch_index_derived(pairs)

            # Should use COPY for bulk loading
            mock_cur.copy.assert_called_once()

            # Should write one row per pair
            assert mock_copy_ctx.write_row.call_count == 2

            mock_conn.commit.assert_called_once()

    def test_batch_index_derived_empty_is_noop(self):
        """batch_index_derived with empty list does nothing."""
        from metalab.store.postgres_index import PostgresIndex

        mock_pool = MagicMock()
        with patch("psycopg_pool.ConnectionPool", return_value=mock_pool):
            index = PostgresIndex(
                "postgresql://localhost/db",
                auto_migrate=False,
            )

            index.batch_index_derived([])

            mock_pool.connection.assert_not_called()


# =========================================================================
# Phase 1C: Incremental field catalog tests
# =========================================================================


@pytest.mark.skipif(not HAS_PSYCOPG, reason="psycopg not installed")
class TestIncrementalFieldCatalog:
    """Tests for _update_catalog_for_record."""

    def test_index_record_updates_catalog(self):
        """index_record calls _update_catalog_for_record."""
        from metalab.store.postgres_index import PostgresIndex

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("psycopg_pool.ConnectionPool", return_value=mock_pool):
            index = PostgresIndex(
                "postgresql://localhost/db",
                auto_migrate=False,
            )
            index._schema_ensured = True

            record = _make_test_record(
                "run_001",
                params={"lr": 0.01, "optimizer": "adam"},
                metrics={"loss": 0.5},
            )

            index.index_record(record)

            # Should have executed the INSERT for the run record (1 call)
            # plus upserts for field_catalog:
            #   2 params (lr, optimizer) + 1 metric (loss) = 3 catalog upserts
            # Total: at least 4 execute calls
            assert mock_cur.execute.call_count >= 4

    def test_update_catalog_handles_numeric_values(self):
        """_update_catalog_for_record sets min/max for numeric fields."""
        from metalab.store.postgres_index import PostgresIndex

        mock_pool = MagicMock()
        mock_cur = MagicMock()

        with patch("psycopg_pool.ConnectionPool", return_value=mock_pool):
            index = PostgresIndex(
                "postgresql://localhost/db",
                auto_migrate=False,
            )

            data = {
                "params_resolved": {"lr": 0.01},
                "metrics": {"loss": 0.5},
            }

            index._update_catalog_for_record(mock_cur, data)

            # Should have 2 execute calls (1 param + 1 metric)
            assert mock_cur.execute.call_count == 2

            # Check that numeric params pass min/max values
            first_call = mock_cur.execute.call_args_list[0]
            params = first_call[0][1]  # The SQL params
            assert params[0] == "params"  # namespace
            assert params[1] == "lr"  # field_name
            assert params[2] == "numeric"  # field_type
            assert params[4] == 0.01  # min_value
            assert params[5] == 0.01  # max_value

    def test_update_catalog_handles_string_values(self):
        """_update_catalog_for_record sets values array for string fields."""
        from metalab.store.postgres_index import PostgresIndex

        mock_pool = MagicMock()
        mock_cur = MagicMock()

        with patch("psycopg_pool.ConnectionPool", return_value=mock_pool):
            index = PostgresIndex(
                "postgresql://localhost/db",
                auto_migrate=False,
            )

            data = {
                "params_resolved": {"optimizer": "adam"},
                "metrics": {},
            }

            index._update_catalog_for_record(mock_cur, data)

            assert mock_cur.execute.call_count == 1
            params = mock_cur.execute.call_args_list[0][0][1]
            assert params[0] == "params"
            assert params[1] == "optimizer"
            assert params[2] == "string"
            assert params[3] == ["adam"]  # values array
            assert params[4] is None  # min_value (not numeric)
            assert params[5] is None  # max_value (not numeric)


# =========================================================================
# Phase 2: Schema version tests
# =========================================================================


class TestSchemaVersion:
    """Tests for schema version bump."""

    def test_schema_version_is_3(self):
        """SCHEMA_VERSION should be 3 after the overhaul."""
        from metalab.store.postgres_index import SCHEMA_VERSION

        assert SCHEMA_VERSION == 3
