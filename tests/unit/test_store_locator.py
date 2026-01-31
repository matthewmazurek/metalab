"""
Tests for store locator and factory.
"""

import tempfile
from pathlib import Path

import pytest

from metalab.store import (
    FileStore,
    StoreFactory,
    create_store,
    parse_locator,
    to_locator,
)
from metalab.store.locator import LocatorInfo


class TestParseLocator:
    """Tests for parse_locator function."""

    def test_parse_absolute_path(self):
        """Parse absolute filesystem path."""
        info = parse_locator("/tmp/store")
        assert info.scheme == "file"
        # Note: macOS resolves /tmp to /private/tmp
        assert "tmp/store" in info.path

    def test_parse_relative_path(self):
        """Parse relative filesystem path."""
        info = parse_locator("./store")
        assert info.scheme == "file"
        assert "store" in info.path

    def test_parse_file_url(self):
        """Parse file:// URL."""
        info = parse_locator("file:///tmp/store")
        assert info.scheme == "file"
        assert "/tmp/store" in info.path

    def test_parse_postgres_url(self):
        """Parse postgresql:// URL."""
        info = parse_locator("postgresql://user@localhost:5432/db")
        assert info.scheme == "postgresql"
        assert info.host == "localhost"
        assert info.port == 5432
        assert info.user == "user"
        assert info.path == "/db"

    def test_parse_postgres_with_password(self):
        """Parse postgresql:// URL with password."""
        info = parse_locator("postgresql://user:pass@localhost:5432/db")
        assert info.user == "user"
        assert info.password == "pass"

    def test_parse_postgres_with_params(self):
        """Parse postgresql:// URL with query params."""
        info = parse_locator("postgresql://localhost/db?schema=myschema&artifact_root=/path")
        assert info.params.get("schema") == "myschema"
        assert info.params.get("artifact_root") == "/path"

    def test_parse_auto_url(self):
        """Parse auto:// URL with fallback."""
        info = parse_locator("auto://?primary=postgresql://localhost/db&fallback=file:///tmp")
        assert info.scheme == "auto"
        assert info.params.get("primary") == "postgresql://localhost/db"
        assert info.params.get("fallback") == "file:///tmp"


class TestStoreFactory:
    """Tests for StoreFactory."""

    def test_supports_file(self):
        """File scheme is supported."""
        assert StoreFactory.supports("file")

    def test_supports_postgresql(self):
        """PostgreSQL scheme is supported (when psycopg available)."""
        # This may be True or False depending on environment
        # Just test it doesn't error
        StoreFactory.supports("postgresql")

    def test_create_file_store_from_path(self):
        """Create FileStore from path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StoreFactory.from_locator(tmpdir)
            assert isinstance(store, FileStore)
            # Compare resolved paths (macOS may use /private symlinks)
            assert store.root.resolve() == Path(tmpdir).resolve()

    def test_create_file_store_from_url(self):
        """Create FileStore from file:// URL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StoreFactory.from_locator(f"file://{tmpdir}")
            assert isinstance(store, FileStore)

    def test_create_store_convenience(self):
        """Test create_store convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_store(tmpdir)
            assert isinstance(store, FileStore)

    def test_unknown_scheme_raises(self):
        """Unknown scheme raises ValueError."""
        with pytest.raises(ValueError, match="Unknown store scheme"):
            StoreFactory.from_locator("unknown://localhost")


class TestToLocator:
    """Tests for to_locator function."""

    def test_file_store_locator(self):
        """FileStore produces file:// locator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStore(tmpdir)
            locator = to_locator(store)
            assert locator.startswith("file://")
            assert tmpdir in locator

    def test_round_trip(self):
        """Store -> locator -> store round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = FileStore(tmpdir)
            locator = to_locator(store1)
            store2 = create_store(locator)
            assert isinstance(store2, FileStore)
            # Compare resolved paths (macOS may use /private symlinks)
            assert store2.root.resolve() == store1.root.resolve()


class TestFallback:
    """Tests for fallback store creation."""

    def test_fallback_on_failure(self):
        """Falls back to secondary store when primary fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try to create PostgresStore (which will fail without psycopg/server)
            # with fallback to FileStore
            try:
                store = StoreFactory.from_locator(
                    "postgresql://nonexistent:5432/db",
                    fallback=tmpdir,
                    connect_timeout=0.1,  # Short timeout
                )
                # If we get here, either psycopg is installed and connection failed
                # (which should trigger fallback), or postgres scheme is not supported
                # In either case, we should get a FileStore from fallback
                assert isinstance(store, FileStore)
            except ValueError as e:
                # PostgresStore not available (psycopg not installed)
                # This is also a valid outcome
                assert "PostgreSQL store not yet implemented" in str(e) or "Unknown store scheme" in str(e)
