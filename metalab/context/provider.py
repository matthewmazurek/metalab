"""
ContextProvider: Build and cache contexts by fingerprint.

The ContextProvider manages the build-once-reuse semantics for contexts:
- ThreadExecutor: All runs share one built context instance (if read-only)
- ProcessExecutor: Each process gets its own cache
- ARC/HPC: Each job gets its own cache

Thread-safe implementation using internal locking.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Protocol

from metalab._ids import fingerprint_context
from metalab.context.builder import ContextBuilder, DefaultContextBuilder
from metalab.context.spec import ContextSpec, FrozenContext


class ContextProvider(Protocol):
    """
    Protocol for context providers.

    A ContextProvider builds and caches FrozenContext instances,
    keyed by the fingerprint of the ContextSpec.
    """

    def get(self, spec: ContextSpec) -> FrozenContext:
        """
        Build or return cached context for the given spec.

        Args:
            spec: The context specification.

        Returns:
            The FrozenContext, either freshly built or from cache.
        """
        ...


class DefaultContextProvider:
    """
    Default ContextProvider with LRU caching.

    Features:
    - Thread-safe cache access
    - LRU eviction when cache exceeds maxsize
    - Keyed by context fingerprint for deduplication
    """

    def __init__(
        self,
        builder: ContextBuilder | None = None,
        maxsize: int = 1,
    ) -> None:
        """
        Initialize the context provider.

        Args:
            builder: The ContextBuilder to use. Defaults to DefaultContextBuilder.
            maxsize: Maximum number of contexts to cache. Default 1.
        """
        self._builder = builder or DefaultContextBuilder()
        self._cache: OrderedDict[str, FrozenContext] = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def get(self, spec: ContextSpec) -> FrozenContext:
        """
        Build or return cached context for the given spec.

        Thread-safe with LRU eviction.

        Args:
            spec: The context specification.

        Returns:
            The FrozenContext instance.
        """
        fp = fingerprint_context(spec)

        with self._lock:
            if fp in self._cache:
                # LRU: mark as recently used
                self._cache.move_to_end(fp)
                return self._cache[fp]

            # Build new context (outside lock would be better for long builds,
            # but simpler to keep inside for correctness)
            context = self._builder.build(spec)
            self._cache[fp] = context

            # Evict LRU if over capacity
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)  # Remove oldest

            return context

    def clear(self) -> None:
        """Evict all cached contexts."""
        with self._lock:
            self._cache.clear()

    def close(self) -> None:
        """
        Optional cleanup hook.

        Override in subclasses if contexts need explicit cleanup.
        """
        self.clear()

    @property
    def cache_size(self) -> int:
        """Return the current number of cached contexts."""
        with self._lock:
            return len(self._cache)
