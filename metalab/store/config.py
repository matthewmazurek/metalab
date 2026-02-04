"""
StoreConfig: Base configuration classes for store backends.

This module provides:

- StoreConfig: Abstract base class for store configurations
- ConfigRegistry: Registry mapping scheme names to config classes

StoreConfig separates configuration (pure data, serializable) from store instances
(connections, file handles). This enables:

- Pre-configuring stores before experiments are created
- Automatic experiment scoping at runtime
- Clean serialization via dataclasses

Each StoreConfig subclass auto-registers itself via __init_subclass__.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from metalab.store.base import Store
    from metalab.store.locator import LocatorInfo


class ConfigRegistry:
    """
    Registry mapping scheme names to config classes.

    Configs self-register via StoreConfig.__init_subclass__.
    """

    _configs: dict[str, type[StoreConfig]] = {}

    @classmethod
    def register(cls, scheme: str, config_class: type[StoreConfig]) -> None:
        """Register a config class for a scheme."""
        cls._configs[scheme] = config_class

    @classmethod
    def get(cls, scheme: str) -> type[StoreConfig] | None:
        """Get the config class for a scheme."""
        return cls._configs.get(scheme)

    @classmethod
    def schemes(cls) -> list[str]:
        """List all registered schemes."""
        return list(cls._configs.keys())


@dataclass(frozen=True, kw_only=True)
class StoreConfig(ABC):
    """
    Abstract base class for store configurations.

    StoreConfig is a pure data object that:

    - Serializes trivially to/from dict/JSON via dataclasses
    - Can produce a Store instance via connect()
    - Can produce a scoped version via scoped()

    Subclasses must define:

    - scheme: ClassVar[str] - the URI scheme (e.g., "file", "postgresql")
    - connect() -> Store - create a connected store instance
    - from_locator(info, **kwargs) -> StoreConfig - parse a locator URI

    Example:
    ```python
    config = FileStoreConfig(root="./experiments")
    scoped = config.scoped("my_exp:1.0")
    store = scoped.connect()
    ```
    """

    scheme: ClassVar[str]
    experiment_id: str | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register subclasses when they're defined."""
        super().__init_subclass__(**kwargs)
        # Only register if scheme is defined and non-empty
        if hasattr(cls, "scheme") and isinstance(cls.scheme, str) and cls.scheme:
            ConfigRegistry.register(cls.scheme, cls)

    @abstractmethod
    def connect(self) -> Store:
        """
        Create a connected Store instance from this config.

        Returns:
            A Store instance ready for use.
        """
        ...

    @classmethod
    @abstractmethod
    def from_locator(cls, info: LocatorInfo, **kwargs: Any) -> StoreConfig:
        """
        Create config from parsed locator info.

        Each subclass handles its own URI parsing.

        Args:
            info: Parsed locator information.
            **kwargs: Additional arguments (can override URI params).

        Returns:
            A StoreConfig instance.
        """
        ...

    def scoped(self, experiment_id: str) -> StoreConfig:
        """
        Return a config scoped to the given experiment.

        Uses dataclasses.replace() to create a new immutable config.

        Args:
            experiment_id: The experiment identifier (e.g., "my_exp:1.0").

        Returns:
            A new StoreConfig with the experiment_id set.

        Raises:
            ValueError: If already scoped to a different experiment.
        """
        if self.experiment_id == experiment_id:
            return self
        if self.experiment_id is not None:
            raise ValueError(
                f"Config already scoped to {self.experiment_id}, "
                f"cannot re-scope to {experiment_id}"
            )
        from dataclasses import replace

        return replace(self, experiment_id=experiment_id)

    @property
    def is_scoped(self) -> bool:
        """True if this config is scoped to a specific experiment."""
        return self.experiment_id is not None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize config to a dictionary.

        Includes a _type field for polymorphic deserialization.

        Returns:
            Dictionary representation of the config.
        """
        d = asdict(self)
        d["_type"] = self.scheme
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoreConfig:
        """
        Deserialize a config from a dictionary.

        Uses the _type field to dispatch to the correct subclass.

        Args:
            data: Dictionary with config data and _type field.

        Returns:
            A StoreConfig instance of the appropriate subclass.

        Raises:
            ValueError: If _type is missing or unknown.
        """
        data = data.copy()
        type_tag = data.pop("_type", None)
        if type_tag is None:
            raise ValueError("Missing _type field in config dict")
        config_class = ConfigRegistry.get(type_tag)
        if config_class is None:
            raise ValueError(f"Unknown config type: {type_tag}")
        return config_class(**data)
