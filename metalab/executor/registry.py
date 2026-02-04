"""
HandleRegistry: Registry for reconnectable executor handles.

This module provides:
- HandleRegistry: Registry mapping executor_type strings to handle classes

The registry enables executor-agnostic reconnection by allowing each handle
type to self-register. When reconnecting, the manifest's executor_type is
used to look up the appropriate handle class.

Example:
    # Handle classes register themselves
    class SlurmRunHandle:
        executor_type: ClassVar[str] = "slurm"
        # Auto-registers via __init_subclass__

    # Reconnect dispatches to the right handle
    handle_class = HandleRegistry.get("slurm")
    handle = handle_class.from_store(store)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metalab.executor.handle import RunHandle
    from metalab.store.base import Store


class HandleRegistry:
    """
    Registry mapping executor_type strings to reconnectable handle classes.

    Handle classes self-register via __init_subclass__ when they define
    an executor_type class variable.
    """

    _handles: dict[str, type] = {}

    @classmethod
    def register(cls, executor_type: str, handle_class: type) -> None:
        """
        Register a handle class for an executor type.

        Args:
            executor_type: The executor type string (e.g., "slurm").
            handle_class: The handle class that supports from_store().
        """
        cls._handles[executor_type] = handle_class

    @classmethod
    def get(cls, executor_type: str) -> type | None:
        """
        Get the handle class for an executor type.

        Args:
            executor_type: The executor type string.

        Returns:
            The handle class, or None if not registered.
        """
        return cls._handles.get(executor_type)

    @classmethod
    def types(cls) -> list[str]:
        """
        List all registered executor types.

        Returns:
            List of registered executor type strings.
        """
        return list(cls._handles.keys())
