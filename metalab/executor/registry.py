"""
HandleRegistry: Registry for reconnectable executor handles.

This module provides:
- HandleRegistry: Registry mapping executor_type strings to handle classes

Handle classes are discovered via :class:`ExecutorConfigRegistry`:
each :class:`ExecutorConfig` subclass can override :meth:`handle_class`
to expose a reconnectable handle.  This folds handle discovery into the
same ``metalab.executors`` entry-point mechanism used for executor configs.

Example:
    handle_class = HandleRegistry.get("slurm")
    handle = handle_class.from_store(store)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metalab.executor.handle import RunHandle

logger = logging.getLogger(__name__)


class HandleRegistry:
    """
    Registry mapping executor_type strings to reconnectable handle classes.

    Delegates to :class:`ExecutorConfigRegistry` entry points: each
    executor config exposes an optional ``handle_class()`` classmethod.
    Results are cached after first lookup.
    """

    _cache: dict[str, type[RunHandle] | None] = {}
    _loaded: bool = False

    @classmethod
    def _ensure_loaded(cls) -> None:
        """Populate cache from all registered executor configs (once)."""
        if cls._loaded:
            return
        cls._loaded = True
        from metalab.executor.config import ExecutorConfigRegistry

        for name in ExecutorConfigRegistry.types():
            config_class = ExecutorConfigRegistry.get(name)
            if config_class is not None:
                try:
                    hc = config_class.handle_class()
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Failed to load handle_class for executor %r", name, exc_info=True
                    )
                    hc = None
                if hc is not None:
                    cls._cache[name] = hc

    @classmethod
    def get(cls, executor_type: str) -> type[RunHandle] | None:
        """
        Get the handle class for an executor type.

        Args:
            executor_type: The executor type string (e.g., "slurm").

        Returns:
            The handle class, or None if not registered.
        """
        cls._ensure_loaded()
        return cls._cache.get(executor_type)

    @classmethod
    def types(cls) -> list[str]:
        """
        List all executor types that have reconnectable handles.

        Returns:
            List of registered executor type strings.
        """
        cls._ensure_loaded()
        return list(cls._cache.keys())
