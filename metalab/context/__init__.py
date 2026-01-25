"""
Context module: Serializable manifests that reconstruct on workers.

This module provides the two-tier context model:
- ContextSpec (serializable): Crosses process boundaries
- FrozenContext (in-memory): Cached within worker processes
"""

from metalab.context.builder import ContextBuilder, DefaultContextBuilder
from metalab.context.provider import ContextProvider, DefaultContextProvider
from metalab.context.spec import ContextSpec, FrozenContext

__all__ = [
    "ContextSpec",
    "FrozenContext",
    "ContextBuilder",
    "DefaultContextBuilder",
    "ContextProvider",
    "DefaultContextProvider",
]
