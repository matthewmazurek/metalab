"""
ContextSpec protocol and FrozenContext type alias.

Invariants (codified here):
1. ContextSpec MUST be serializable (JSON-compatible or reconstructable from manifest)
2. ContextBuilder.build(spec) MUST be deterministic given the same environment/resources
3. FrozenContext MUST be treated as read-only by Operations (no mutation)
4. Runner MAY cache FrozenContext by context_fingerprint within a worker process
5. Context builders SHOULD avoid lazy mutation after build (thread safety)

What "Shared Context" Means:
- ThreadExecutor: Same FrozenContext instance reused across runs (in-memory cache)
- ProcessExecutor: Each process has its own cache (useful for batched runs)
- ARC/HPC: Each job has its own cache; cross-node sharing via external storage only

IMPORTANT: "Shared across ARC workers" means shared via external materialization
(dataset paths, cached artifacts on shared storage)—NOT in-memory sharing.
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

    A ContextSpec is a serializable manifest that describes how to construct
    the actual context (FrozenContext) on any worker. It should contain:
    - Dataset/resource identifiers (paths, URIs, IDs, checksums)
    - Configuration fragments
    - Versions of upstream processing steps (optional but recommended)

    The spec itself should be small and serializable. Heavy data loading
    happens in the ContextBuilder.

    Implementations can be:
    - A frozen dataclass
    - A plain dict
    - Any object that can be canonically serialized

    Example:
        @dataclass(frozen=True)
        class MyContextSpec:
            dataset_path: str
            dataset_checksum: str
            config: dict
    """

    # No required methods - any serializable object can be a ContextSpec
    # The protocol exists for type hints and documentation
    pass


# Type alias for built context
# User decides actual immutability; we just promise to treat it as read-only
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

        # Store original __init__ if it exists
        original_init = cls.__init__

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
