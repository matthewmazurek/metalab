"""
ContextProvider: Simple pass-through for context specs.

The provider is kept for API compatibility but now simply returns the spec as-is.
The spec IS the context that operations receive.
"""

from __future__ import annotations

from typing import Protocol

from metalab.context.spec import ContextSpec, FrozenContext


class ContextProvider(Protocol):
    """
    Protocol for context providers.

    A ContextProvider returns the context for a given spec.
    In the simplified model, this just returns the spec itself.
    """

    def get(self, spec: ContextSpec) -> FrozenContext:
        """
        Get the context for the given spec.

        Args:
            spec: The context specification.

        Returns:
            The context (which is the spec itself).
        """
        ...


class DefaultContextProvider:
    """
    Default ContextProvider that returns the spec as-is.

    This is a simple pass-through - the spec IS the context that
    operations receive. Operations are responsible for loading any
    heavy data they need using paths from the spec.
    """

    def __init__(self, maxsize: int = 1) -> None:
        """
        Initialize the context provider.

        Args:
            maxsize: Ignored (kept for API compatibility).
        """
        pass

    def get(self, spec: ContextSpec) -> FrozenContext:
        """
        Return the spec as the context.

        Args:
            spec: The context specification.

        Returns:
            The spec itself (operations receive this directly).
        """
        return spec

    def clear(self) -> None:
        """No-op (kept for API compatibility)."""
        pass

    def close(self) -> None:
        """No-op (kept for API compatibility)."""
        pass

    @property
    def cache_size(self) -> int:
        """Always returns 0 (no caching needed)."""
        return 0
