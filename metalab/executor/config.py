"""
ExecutorConfig: Base configuration classes for executor backends.

Mirrors the StoreConfig/ConfigRegistry pattern from metalab/store/config.py.
Each executor backend registers its own config class via __init_subclass__.

Usage:
    # Create executor from config dict (e.g., parsed from YAML)
    executor = executor_from_config("slurm", {"partition": "gpu", "time": "2:00:00"})

    # Or create config object first
    config = ExecutorConfigRegistry.get("slurm").from_dict({...})
    executor = config.create()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from metalab.executor.base import Executor

logger = logging.getLogger(__name__)


class ExecutorConfigRegistry:
    """
    Registry mapping executor type names to config classes.

    Configs self-register via ExecutorConfig.__init_subclass__.
    """

    _configs: dict[str, type[ExecutorConfig]] = {}

    @classmethod
    def register(cls, name: str, config_class: type[ExecutorConfig]) -> None:
        """Register a config class for an executor type."""
        cls._configs[name] = config_class

    @classmethod
    def get(cls, name: str) -> type[ExecutorConfig] | None:
        """Get the config class for an executor type."""
        return cls._configs.get(name)

    @classmethod
    def types(cls) -> list[str]:
        """List all registered executor types."""
        return list(cls._configs.keys())


@dataclass
class ExecutorConfig(ABC):
    """
    Abstract base class for executor configurations.

    Subclasses must define:

    - executor_type: ClassVar[str] -- the type name (e.g., "local", "slurm")
    - create() -> Executor | None -- create an executor instance
    - from_dict(d) -> ExecutorConfig -- parse from config dict
    """

    executor_type: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register subclasses when they're defined."""
        super().__init_subclass__(**kwargs)
        if (
            hasattr(cls, "executor_type")
            and isinstance(cls.executor_type, str)
            and cls.executor_type
        ):
            ExecutorConfigRegistry.register(cls.executor_type, cls)

    @abstractmethod
    def create(self) -> Executor | None:
        """
        Create an executor instance from this config.

        Returns:
            An Executor instance, or None for single-threaded local execution.
        """
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecutorConfig:
        """
        Parse from a config dict (e.g., YAML section).

        Args:
            d: Configuration dictionary.

        Returns:
            An ExecutorConfig instance.
        """
        ...


def executor_from_config(
    executor_type: str,
    config: dict[str, Any] | None = None,
) -> Any:  # Returns Executor | None
    """
    Create an executor from a type name and config dict.

    This is the main entry point for config-driven executor creation.
    Each executor backend registers its own config class, so this function
    has zero coupling to specific executor implementations.

    Args:
        executor_type: Registered executor type (e.g., "local", "slurm").
        config: Configuration dict (e.g., parsed YAML section).

    Returns:
        An Executor instance, or None for single-threaded local execution.

    Raises:
        ValueError: If executor_type is not registered.
    """
    config_class = ExecutorConfigRegistry.get(executor_type)
    if config_class is None:
        raise ValueError(
            f"Unknown executor type: {executor_type!r}. "
            f"Available: {ExecutorConfigRegistry.types()}"
        )
    return config_class.from_dict(config or {}).create()


def resolve_executor(
    platform: str,
    overrides: dict[str, Any] | None = None,
) -> Any:  # Returns Executor | None
    """
    Create an executor from ``.metalab.toml`` defaults + per-experiment overrides.

    Loads the project config (if present), resolves the named environment
    profile matching *platform*, merges its settings with *overrides*,
    and creates the executor via :class:`ExecutorConfigRegistry`.

    Falls back to *overrides*-only if no ``.metalab.toml`` is found or the
    platform name does not match a configured environment.

    Args:
        platform: Executor type name (e.g., ``"local"``, ``"slurm"``).
            Also used to look up the matching environment profile in
            ``.metalab.toml``.
        overrides: Per-experiment executor overrides (e.g., from a YAML
            config).  These take precedence over ``.metalab.toml`` defaults.

    Returns:
        An Executor instance, or None for single-threaded local execution.

    Raises:
        ValueError: If *platform* is not a registered executor type.
    """
    defaults: dict[str, Any] = {}
    try:
        from metalab.config import ProjectConfig

        project_config = ProjectConfig.load()
        resolved = project_config.resolve(platform)
        defaults = dict(resolved.env_config)
        logger.info(
            "Loaded executor defaults from .metalab.toml [%s] (%d settings)",
            platform, len(defaults),
        )
    except FileNotFoundError:
        logger.debug("No .metalab.toml found; using overrides only")
    except (ValueError, ImportError):
        logger.debug("Could not resolve environment %r; using overrides only", platform)

    merged = {**defaults, **(overrides or {})}
    return executor_from_config(platform, merged)
