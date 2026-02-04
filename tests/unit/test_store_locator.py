"""
Tests for store locator and config parsing.
"""

import tempfile
from pathlib import Path

import pytest

from metalab.store import (
    ConfigRegistry,
    FileStore,
    FileStoreConfig,
    create_store,
    parse_locator,
    parse_to_config,
    safe_experiment_id,
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
        info = parse_locator(
            "postgresql://localhost/db?schema=myschema&file_root=/shared/experiments"
        )
        assert info.params.get("schema") == "myschema"
        assert info.params.get("file_root") == "/shared/experiments"


class TestParseToConfig:
    """Tests for parse_to_config function."""

    def test_parse_path_to_config(self):
        """Plain path creates FileStoreConfig."""
        config = parse_to_config("./experiments")
        assert isinstance(config, FileStoreConfig)

    def test_parse_file_url_to_config(self):
        """file:// URL creates FileStoreConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = parse_to_config(f"file://{tmpdir}")
            assert isinstance(config, FileStoreConfig)

    def test_parse_with_experiment_id(self):
        """experiment_id kwarg is passed through."""
        config = parse_to_config("./experiments", experiment_id="my_exp:1.0")
        assert config.experiment_id == "my_exp:1.0"

    def test_unknown_scheme_raises(self):
        """Unknown scheme raises ValueError."""
        with pytest.raises(ValueError, match="Unknown store scheme"):
            parse_to_config("unknown://localhost")


class TestConfigRegistry:
    """Tests for ConfigRegistry."""

    def test_file_scheme_registered(self):
        """FileStoreConfig is registered for 'file' scheme."""
        config_class = ConfigRegistry.get("file")
        assert config_class is FileStoreConfig

    def test_postgresql_scheme_registered(self):
        """PostgresStoreConfig is registered for 'postgresql' scheme (when available)."""
        # This may return None if psycopg not installed
        config_class = ConfigRegistry.get("postgresql")
        # Just ensure no error - may be None or PostgresStoreConfig

    def test_schemes_includes_file(self):
        """schemes() includes 'file'."""
        assert "file" in ConfigRegistry.schemes()


class TestCreateStore:
    """Tests for create_store convenience function."""

    def test_create_file_store_from_path(self):
        """Create FileStore from path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_store(tmpdir)
            assert isinstance(store, FileStore)
            # Compare resolved paths (macOS may use /private symlinks)
            assert store.root.resolve() == Path(tmpdir).resolve()

    def test_create_file_store_from_url(self):
        """Create FileStore from file:// URL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_store(f"file://{tmpdir}")
            assert isinstance(store, FileStore)

    def test_unknown_scheme_raises(self):
        """Unknown scheme raises ValueError."""
        with pytest.raises(ValueError, match="Unknown store scheme"):
            create_store("unknown://localhost")

    def test_create_store_with_experiment_id(self):
        """create_store passes experiment_id to config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_store(tmpdir, experiment_id="test:2.0")
            assert isinstance(store, FileStore)
            expected = Path(tmpdir) / "test_2.0"
            assert store.root.resolve() == expected.resolve()


class TestPostgresConfigCreation:
    """Tests for PostgresStoreConfig creation (requires psycopg)."""

    def test_postgres_locator_without_file_root(self):
        """Postgres locator without file_root has empty params."""
        config_class = ConfigRegistry.get("postgresql")
        if config_class is None:
            pytest.skip("psycopg not installed")

        info = parse_locator("postgresql://localhost/db")
        assert info.params.get("file_root") is None

    def test_postgres_config_requires_file_root(self):
        """PostgresStoreConfig requires file_root."""
        config_class = ConfigRegistry.get("postgresql")
        if config_class is None:
            pytest.skip("psycopg not installed")

        with pytest.raises(ValueError, match="file_root"):
            parse_to_config("postgresql://localhost/db")

    def test_postgres_with_file_root_in_uri(self):
        """PostgresStoreConfig accepts file_root in URI."""
        config_class = ConfigRegistry.get("postgresql")
        if config_class is None:
            pytest.skip("psycopg not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = parse_to_config(f"postgresql://localhost/db?file_root={tmpdir}")
            # Compare resolved paths (macOS resolves /var -> /private/var)
            assert Path(config.file_root).resolve() == Path(tmpdir).resolve()

    def test_postgres_with_file_root_kwarg(self):
        """PostgresStoreConfig accepts file_root as kwarg."""
        config_class = ConfigRegistry.get("postgresql")
        if config_class is None:
            pytest.skip("psycopg not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = parse_to_config(
                "postgresql://localhost/db",
                file_root=tmpdir,
            )
            # Compare resolved paths (macOS resolves /var -> /private/var)
            assert Path(config.file_root).resolve() == Path(tmpdir).resolve()


class TestSafeExperimentId:
    """Tests for safe_experiment_id helper."""

    def test_sanitizes_colon(self):
        """Colons are replaced with underscores."""
        assert safe_experiment_id("my_exp:1.0") == "my_exp_1.0"

    def test_multiple_colons(self):
        """Multiple colons are all replaced."""
        assert safe_experiment_id("a:b:c") == "a_b_c"

    def test_no_colon(self):
        """IDs without colons pass through unchanged."""
        assert safe_experiment_id("my_experiment") == "my_experiment"

    def test_empty_string(self):
        """Empty string is handled."""
        assert safe_experiment_id("") == ""


class TestExperimentScopedStorage:
    """Tests for experiment-scoped directory nesting."""

    def test_filestore_without_experiment_id(self):
        """FileStore without experiment_id uses root directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStoreConfig(root=tmpdir).connect()
            # Root should be the exact directory we passed
            assert store.root.resolve() == Path(tmpdir).resolve()

    def test_filestore_with_experiment_id(self):
        """FileStore with experiment_id creates nested directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStoreConfig(root=tmpdir, experiment_id="my_exp:1.0").connect()
            # Root should be under {tmpdir}/my_exp_1.0/
            expected = Path(tmpdir) / "my_exp_1.0"
            assert store.root.resolve() == expected.resolve()
            # Directory should be created
            assert expected.exists()

    def test_scoped_via_method(self):
        """FileStoreConfig.scoped() creates nested directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = FileStoreConfig(root=tmpdir)
            scoped_config = config.scoped("exp:3.0")
            store = scoped_config.connect()
            expected = Path(tmpdir) / "exp_3.0"
            assert store.root.resolve() == expected.resolve()

    def test_config_round_trip_with_experiment_id(self):
        """Config with experiment_id round-trips through to_dict/from_dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config with experiment_id
            config1 = FileStoreConfig(root=tmpdir, experiment_id="exp:3.0")

            # Serialize and deserialize
            d = config1.to_dict()
            from metalab.store.config import StoreConfig

            config2 = StoreConfig.from_dict(d)

            assert config2.root == config1.root
            assert config2.experiment_id == config1.experiment_id

    def test_multiple_experiments_separate_directories(self):
        """Different experiments get their own directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = FileStoreConfig(root=tmpdir, experiment_id="exp:1.0").connect()
            store2 = FileStoreConfig(root=tmpdir, experiment_id="exp:2.0").connect()

            # Each should have its own directory
            assert store1.root != store2.root
            assert store1.root.name == "exp_1.0"
            assert store2.root.name == "exp_2.0"

            # Both should exist
            assert store1.root.exists()
            assert store2.root.exists()
