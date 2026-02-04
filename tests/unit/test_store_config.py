"""
Tests for StoreConfig dataclass architecture.

Tests:
- FileStoreConfig creation and path normalization
- scoped() returns new config with experiment_id
- scoped() is idempotent (same experiment_id returns self)
- scoped() raises on re-scoping to different experiment
- to_dict() / from_dict() round-trip
- connect() creates working stores
- is_scoped property
- ConfigRegistry auto-registration
"""

from pathlib import Path

import pytest

from metalab.store.config import ConfigRegistry, StoreConfig
from metalab.store.file import FileStore, FileStoreConfig
from metalab.store.locator import LocatorInfo, parse_to_config


class TestFileStoreConfig:
    """Tests for FileStoreConfig."""

    def test_create_with_relative_path(self, tmp_path: Path) -> None:
        """Config normalizes relative paths to absolute."""
        config = FileStoreConfig(root="./experiments")
        assert Path(config.root).is_absolute()

    def test_create_with_absolute_path(self, tmp_path: Path) -> None:
        """Config preserves absolute paths."""
        abs_path = str(tmp_path / "experiments")
        config = FileStoreConfig(root=abs_path)
        assert config.root == abs_path

    def test_scheme_is_file(self) -> None:
        """FileStoreConfig has scheme 'file'."""
        assert FileStoreConfig.scheme == "file"

    def test_is_scoped_false_by_default(self) -> None:
        """Config is unscoped by default."""
        config = FileStoreConfig(root="./experiments")
        assert not config.is_scoped
        assert config.experiment_id is None

    def test_is_scoped_true_when_set(self) -> None:
        """Config is scoped when experiment_id is set."""
        config = FileStoreConfig(root="./experiments", experiment_id="my_exp:1.0")
        assert config.is_scoped
        assert config.experiment_id == "my_exp:1.0"


class TestFileStoreConfigScoped:
    """Tests for FileStoreConfig.scoped()."""

    def test_scoped_returns_new_config(self) -> None:
        """scoped() returns a new config with experiment_id."""
        config = FileStoreConfig(root="./experiments")
        scoped = config.scoped("my_exp:1.0")

        assert scoped is not config
        assert scoped.experiment_id == "my_exp:1.0"
        assert config.experiment_id is None  # Original unchanged

    def test_scoped_preserves_root(self) -> None:
        """scoped() preserves the root path."""
        config = FileStoreConfig(root="./experiments")
        scoped = config.scoped("my_exp:1.0")

        assert scoped.root == config.root

    def test_scoped_idempotent_same_experiment(self) -> None:
        """scoped() with same experiment_id returns self."""
        config = FileStoreConfig(root="./experiments", experiment_id="my_exp:1.0")
        scoped = config.scoped("my_exp:1.0")

        assert scoped is config

    def test_scoped_raises_on_different_experiment(self) -> None:
        """scoped() raises ValueError when already scoped to different experiment."""
        config = FileStoreConfig(root="./experiments", experiment_id="my_exp:1.0")

        with pytest.raises(ValueError, match="already scoped"):
            config.scoped("other_exp:2.0")


class TestFileStoreConfigSerialization:
    """Tests for FileStoreConfig serialization."""

    def test_to_dict_includes_type(self) -> None:
        """to_dict() includes _type field."""
        config = FileStoreConfig(root="./experiments")
        d = config.to_dict()

        assert "_type" in d
        assert d["_type"] == "file"

    def test_to_dict_includes_all_fields(self) -> None:
        """to_dict() includes all config fields."""
        config = FileStoreConfig(root="./experiments", experiment_id="my_exp:1.0")
        d = config.to_dict()

        assert "root" in d
        assert "experiment_id" in d
        assert d["experiment_id"] == "my_exp:1.0"

    def test_from_dict_round_trip(self) -> None:
        """from_dict() can reconstruct config from to_dict()."""
        original = FileStoreConfig(root="./experiments", experiment_id="my_exp:1.0")
        d = original.to_dict()
        restored = StoreConfig.from_dict(d)

        assert isinstance(restored, FileStoreConfig)
        assert restored.root == original.root
        assert restored.experiment_id == original.experiment_id

    def test_from_dict_raises_on_missing_type(self) -> None:
        """from_dict() raises ValueError when _type is missing."""
        with pytest.raises(ValueError, match="Missing _type"):
            StoreConfig.from_dict({"root": "./experiments"})

    def test_from_dict_raises_on_unknown_type(self) -> None:
        """from_dict() raises ValueError for unknown _type."""
        with pytest.raises(ValueError, match="Unknown config type"):
            StoreConfig.from_dict({"_type": "unknown", "root": "./experiments"})


class TestFileStoreConfigConnect:
    """Tests for FileStoreConfig.connect()."""

    def test_connect_creates_filestore(self, tmp_path: Path) -> None:
        """connect() creates a FileStore instance."""
        config = FileStoreConfig(root=str(tmp_path / "store"))
        store = config.connect()

        assert isinstance(store, FileStore)

    def test_connect_unscoped_store_root(self, tmp_path: Path) -> None:
        """Unscoped config creates store at root."""
        config = FileStoreConfig(root=str(tmp_path / "store"))
        store = config.connect()

        assert store.root == Path(tmp_path / "store")

    def test_connect_scoped_store_root(self, tmp_path: Path) -> None:
        """Scoped config creates store at root/safe_experiment_id."""
        config = FileStoreConfig(
            root=str(tmp_path / "store"), experiment_id="my_exp:1.0"
        )
        store = config.connect()

        assert store.root == Path(tmp_path / "store" / "my_exp_1.0")

    def test_store_has_config_reference(self, tmp_path: Path) -> None:
        """Connected store has reference to its config."""
        config = FileStoreConfig(root=str(tmp_path / "store"))
        store = config.connect()

        assert store.config == config


class TestFileStoreScoped:
    """Tests for FileStore.scoped()."""

    def test_store_scoped_creates_new_store(self, tmp_path: Path) -> None:
        """FileStore.scoped() creates a new scoped store."""
        config = FileStoreConfig(root=str(tmp_path / "store"))
        store = config.connect()

        scoped_store = store.scoped("my_exp:1.0")

        assert scoped_store is not store
        assert scoped_store.is_scoped
        assert scoped_store.config.experiment_id == "my_exp:1.0"

    def test_store_is_scoped_property(self, tmp_path: Path) -> None:
        """FileStore.is_scoped delegates to config."""
        unscoped = FileStoreConfig(root=str(tmp_path / "store")).connect()
        scoped = FileStoreConfig(
            root=str(tmp_path / "store"), experiment_id="my_exp:1.0"
        ).connect()

        assert not unscoped.is_scoped
        assert scoped.is_scoped


class TestConfigRegistry:
    """Tests for ConfigRegistry."""

    def test_file_scheme_registered(self) -> None:
        """FileStoreConfig is auto-registered for 'file' scheme."""
        config_class = ConfigRegistry.get("file")
        assert config_class is FileStoreConfig

    def test_unknown_scheme_returns_none(self) -> None:
        """Unknown scheme returns None."""
        config_class = ConfigRegistry.get("unknown_scheme")
        assert config_class is None

    def test_schemes_lists_registered(self) -> None:
        """schemes() lists all registered schemes."""
        schemes = ConfigRegistry.schemes()
        assert "file" in schemes


class TestParseToConfig:
    """Tests for parse_to_config()."""

    def test_parse_plain_path(self) -> None:
        """Plain path creates FileStoreConfig."""
        config = parse_to_config("./experiments")

        assert isinstance(config, FileStoreConfig)

    def test_parse_file_uri(self, tmp_path: Path) -> None:
        """file:// URI creates FileStoreConfig."""
        config = parse_to_config(f"file://{tmp_path}/experiments")

        assert isinstance(config, FileStoreConfig)
        assert config.root == str(tmp_path / "experiments")

    def test_parse_with_experiment_id_kwarg(self) -> None:
        """experiment_id kwarg is passed through."""
        config = parse_to_config("./experiments", experiment_id="my_exp:1.0")

        assert config.experiment_id == "my_exp:1.0"

    def test_parse_unknown_scheme_raises(self) -> None:
        """Unknown scheme raises ValueError."""
        with pytest.raises(ValueError, match="Unknown store scheme"):
            parse_to_config("unknown://foo/bar")


class TestFileStoreConfigFromLocator:
    """Tests for FileStoreConfig.from_locator()."""

    def test_from_locator_extracts_path(self) -> None:
        """from_locator() extracts path from LocatorInfo."""
        info = LocatorInfo(
            scheme="file", path="/path/to/store", raw="file:///path/to/store"
        )
        config = FileStoreConfig.from_locator(info)

        assert config.root == "/path/to/store"

    def test_from_locator_extracts_experiment_id(self) -> None:
        """from_locator() extracts experiment_id from params."""
        info = LocatorInfo(
            scheme="file",
            path="/path/to/store",
            params={"experiment_id": "my_exp:1.0"},
            raw="file:///path/to/store?experiment_id=my_exp:1.0",
        )
        config = FileStoreConfig.from_locator(info)

        assert config.experiment_id == "my_exp:1.0"

    def test_from_locator_kwarg_overrides_params(self) -> None:
        """experiment_id kwarg overrides params."""
        info = LocatorInfo(
            scheme="file",
            path="/path/to/store",
            params={"experiment_id": "from_params"},
            raw="file:///path/to/store",
        )
        config = FileStoreConfig.from_locator(info, experiment_id="from_kwarg")

        assert config.experiment_id == "from_kwarg"


class TestCollectionAPI:
    """Tests for collection browsing API (list_experiments, for_experiment)."""

    def test_list_experiments_empty(self, tmp_path: Path) -> None:
        """list_experiments() returns empty list for empty collection."""
        config = FileStoreConfig(root=str(tmp_path))
        assert config.list_experiments() == []

    def test_list_experiments_discovers_experiments(self, tmp_path: Path) -> None:
        """list_experiments() discovers experiment subdirectories."""
        # Create some experiment stores
        config = FileStoreConfig(root=str(tmp_path))
        config.scoped("exp_a:1.0").connect()
        config.scoped("exp_b:2.0").connect()

        experiments = config.list_experiments()

        assert sorted(experiments) == ["exp_a:1.0", "exp_b:2.0"]

    def test_list_experiments_ignores_non_experiment_dirs(self, tmp_path: Path) -> None:
        """list_experiments() ignores directories without _meta.json."""
        config = FileStoreConfig(root=str(tmp_path))
        config.scoped("real_exp:1.0").connect()

        # Create a non-experiment directory
        (tmp_path / "not_an_experiment").mkdir()

        experiments = config.list_experiments()

        assert experiments == ["real_exp:1.0"]

    def test_list_experiments_raises_on_scoped(self, tmp_path: Path) -> None:
        """list_experiments() raises if config is already scoped."""
        config = FileStoreConfig(root=str(tmp_path), experiment_id="some_exp:1.0")

        with pytest.raises(ValueError, match="Cannot list experiments on a scoped"):
            config.list_experiments()

    def test_list_experiments_nonexistent_root(self) -> None:
        """list_experiments() returns empty list for nonexistent root."""
        config = FileStoreConfig(root="/nonexistent/path/to/nowhere")
        assert config.list_experiments() == []

    def test_for_experiment_returns_scoped_config(self, tmp_path: Path) -> None:
        """for_experiment() returns a scoped config."""
        config = FileStoreConfig(root=str(tmp_path))
        scoped = config.for_experiment("my_exp:1.0")

        assert scoped.experiment_id == "my_exp:1.0"
        assert scoped.root == config.root

    def test_for_experiment_same_as_scoped(self, tmp_path: Path) -> None:
        """for_experiment() is equivalent to scoped()."""
        config = FileStoreConfig(root=str(tmp_path))

        via_for = config.for_experiment("my_exp:1.0")
        via_scoped = config.scoped("my_exp:1.0")

        assert via_for == via_scoped
