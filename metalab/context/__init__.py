"""
Context module: Lightweight, serializable context specs.

The context system provides:
- ContextSpec: A serializable manifest (paths, config, checksums)
- context_spec: Decorator to create frozen dataclasses with fingerprinting

Operations receive the spec directly and load any heavy data themselves.
"""

from metalab.context.provider import ContextProvider, DefaultContextProvider
from metalab.context.spec import ContextSpec, FrozenContext, context_spec

__all__ = [
    "ContextSpec",
    "FrozenContext",
    "context_spec",
    "ContextProvider",
    "DefaultContextProvider",
]
