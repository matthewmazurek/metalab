"""
ContextSpec protocol and FrozenContext type alias.

The context system provides lightweight, serializable configuration that travels
across executor boundaries. Operations receive the spec directly and are responsible
for loading any heavy data they need.

Invariants:
1. ContextSpec MUST be serializable (JSON-compatible or reconstructable from manifest)
2. ContextSpec is passed directly to operations (no separate "builder" step)
3. Operations load their own data using paths/references from the spec
4. Context fingerprint is computed from spec fields only

Design rationale:
- Specs are lightweight manifests (paths, config, checksums)
- Operations load data themselves (avoids mutation/serialization issues)
- Each run gets fresh data if needed (no shared mutable state)
- Preprocessing is explicit (run before experiment, not hidden in framework)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Protocol, TypeVar, overload, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class ContextSpec(Protocol):
    """
    Protocol for context specifications.

    A ContextSpec is a serializable manifest that describes the shared context
    for an experiment. It should contain:
    - Dataset/resource identifiers (paths, URIs, IDs, checksums)
    - Configuration fragments
    - Versions of upstream processing steps (optional but recommended)

    The spec itself should be small and serializable. Operations load data
    themselves using paths/references from the spec.

    Implementations can be:
    - A frozen dataclass (recommended, use @context_spec decorator)
    - A plain dict
    - Any object that can be canonically serialized

    Example:
        @metalab.context_spec
        class DataContext:
            dataset: metalab.FilePath  # Hash computed lazily at run() time
            vocab_size: int = 10000

        @metalab.operation
        def my_op(context, params, capture):
            # Load data using paths from context
            data = pd.read_csv(str(context.dataset))
            ...

        # Preprocessing can happen after spec creation
        spec = DataContext(dataset=metalab.FilePath("./cache/data.csv"))
        preprocess(spec)  # File created here
        metalab.run(...)  # Hash computed here
    """

    # No required methods - any serializable object can be a ContextSpec
    # The protocol exists for type hints and documentation
    pass


# Type alias for context (the spec IS the context now)
FrozenContext = Any


def _compute_fingerprint(obj: Any) -> str:
    """
    Compute a stable fingerprint for a dataclass instance.

    Recursively serializes all fields to JSON and hashes the result.
    """

    def to_serializable(val: Any) -> Any:
        """Convert a value to a JSON-serializable form."""
        if val is None or isinstance(val, (bool, int, float, str)):
            return val
        if isinstance(val, (list, tuple)):
            return [to_serializable(v) for v in val]
        if isinstance(val, dict):
            return {k: to_serializable(v) for k, v in sorted(val.items())}
        if dataclasses.is_dataclass(val) and not isinstance(val, type):
            return {
                f.name: to_serializable(getattr(val, f.name))
                for f in dataclasses.fields(val)
                if not f.name.startswith("_") and f.name != "fingerprint"
            }
        # Fall back to string representation
        return str(val)

    data = to_serializable(obj)
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


@overload
def context_spec(cls: type[T]) -> type[T]: ...
@overload
def context_spec(
    *,
    frozen: bool = True,
) -> Any: ...


def context_spec(
    cls: type[T] | None = None,
    *,
    frozen: bool = True,
) -> type[T] | Any:
    """
    Decorator that creates a frozen dataclass with automatic fingerprinting.

    This decorator:
    1. Applies @dataclass(frozen=True) to the class (unless frozen=False)
    2. Adds a `fingerprint` property that computes a stable hash of all fields

    The fingerprint is computed lazily on first access and cached.

    Args:
        cls: The class to decorate (when used without parentheses).
        frozen: Whether to make the dataclass frozen (default: True).

    Returns:
        The decorated class.

    Example:
        @metalab.context_spec
        class MyContextSpec:
            name: str
            version: str = "1.0"
            dataset_path: str = ""

        spec = MyContextSpec(name="test", dataset_path="/data/train.csv")
        print(spec.fingerprint)  # Auto-computed hash
    """

    def decorator(cls: type[T]) -> type[T]:
        # Apply dataclass decorator if not already a dataclass
        if not dataclasses.is_dataclass(cls):
            cls = dataclasses.dataclass(frozen=frozen)(cls)
        elif frozen and not cls.__dataclass_fields__:  # type: ignore
            # Re-apply with frozen if needed
            cls = dataclasses.dataclass(frozen=frozen)(cls)

        # Cache for fingerprint
        _fingerprint_cache: dict[int, str] = {}

        @property
        def fingerprint(self) -> str:
            """Auto-computed fingerprint based on all fields."""
            obj_id = id(self)
            if obj_id not in _fingerprint_cache:
                _fingerprint_cache[obj_id] = _compute_fingerprint(self)
            return _fingerprint_cache[obj_id]

        # Add fingerprint property
        cls.fingerprint = fingerprint  # type: ignore

        return cls

    # Handle both @context_spec and @context_spec() syntax
    if cls is not None:
        return decorator(cls)
    return decorator
