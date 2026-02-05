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
import importlib
import json
import logging
from typing import (
    Any,
    Protocol,
    TypeVar,
    dataclass_transform,
    overload,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

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
    ```python
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
    ```
    """

    # No required methods - any serializable object can be a ContextSpec
    # The protocol exists for type hints and documentation
    pass


# Type alias for context (the spec IS the context now)
FrozenContext = Any


def serialize_context_spec(obj: Any) -> Any:
    """
    Serialize a context spec to a JSON-compatible structure with type preservation.

    Handles dataclasses (FilePath, DirPath, context_spec decorated classes),
    dicts, lists, and primitives. Type information is preserved via __type__ keys
    to enable reconstruction via deserialize_context_spec().

    Args:
        obj: The context spec object to serialize.

    Returns:
        A JSON-serializable structure.

    Example:
    ```python
    @context_spec
    class MyContext:
        path: FilePath
        value: int

    ctx = MyContext(path=FilePath("/data"), value=42)
    data = serialize_context_spec(ctx)
    # {"__type__": "mymodule.MyContext", "path": {...}, "value": 42}
    ```
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [serialize_context_spec(item) for item in obj]
    if isinstance(obj, dict):
        return {k: serialize_context_spec(v) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        type_name = f"{type(obj).__module__}.{type(obj).__qualname__}"
        return {
            "__type__": type_name,
            **{
                f.name: serialize_context_spec(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
            },
        }
    # Fall back to string representation
    return str(obj)


def deserialize_context_spec(obj: Any) -> Any:
    """
    Deserialize a context spec from JSON structure with type reconstruction.

    Reconstructs dataclasses (FilePath, DirPath, context_spec decorated classes)
    from their serialized form using __type__ metadata.

    Args:
        obj: The JSON-loaded structure.

    Returns:
        The reconstructed context spec object.

    Example:
    ```python
    data = {"__type__": "mymodule.MyContext", "path": {...}, "value": 42}
    ctx = deserialize_context_spec(data)
    # Returns MyContext instance
    ```
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, list):
        return [deserialize_context_spec(item) for item in obj]
    if isinstance(obj, dict):
        if "__type__" in obj:
            # Reconstruct the dataclass
            type_path = obj["__type__"]
            module_path, class_name = type_path.rsplit(".", 1)
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                # Get field values, excluding __type__
                field_values = {
                    k: deserialize_context_spec(v)
                    for k, v in obj.items()
                    if k != "__type__"
                }
                return cls(**field_values)
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not reconstruct type {type_path}: {e}")
                # Return as dict if reconstruction fails
                return {
                    k: deserialize_context_spec(v)
                    for k, v in obj.items()
                    if k != "__type__"
                }
        else:
            return {k: deserialize_context_spec(v) for k, v in obj.items()}
    return obj


def _compute_fingerprint(obj: Any) -> str:
    """
    Compute a stable fingerprint for a context spec.

    Uses serialize_context_spec() for consistency, then hashes the JSON.
    """
    data = serialize_context_spec(obj)
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


@overload
def context_spec(cls: type[T]) -> type[T]: ...
@overload
def context_spec(
    *,
    frozen: bool = True,
) -> Any: ...


@dataclass_transform(frozen_default=True, field_specifiers=(dataclasses.Field,))
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
    ```python
    @metalab.context_spec
    class MyContextSpec:
        name: str
        version: str = "1.0"
        dataset_path: str = ""

    spec = MyContextSpec(name="test", dataset_path="/data/train.csv")
    print(spec.fingerprint)  # Auto-computed hash
    ```
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
