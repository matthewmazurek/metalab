"""
Low-level canonicalization primitives (internal).

This module provides deterministic JSON encoding and fingerprinting
for reproducible identity computation.

Key design decisions:
- Floats use repr() for full precision (NOT configurable)
- NaN/Inf raise CanonicalizeError (not silently encoded)
- Sets become sorted lists
- Dataclasses become dicts
- Keys are always sorted
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any


class CanonicalizeError(Exception):
    """Raised when an object cannot be canonicalized."""

    pass


def _encode_value(obj: Any) -> Any:
    """
    Recursively encode a value for canonical JSON serialization.

    Raises:
        CanonicalizeError: If the value cannot be canonicalized (e.g., NaN, Inf)
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            raise CanonicalizeError("NaN not allowed in canonical params")
        if math.isinf(obj):
            raise CanonicalizeError("Inf not allowed in canonical params")
        # Use repr() for full precision to ensure run_id stability
        return repr(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        # Encode bytes as hex string
        return {"__bytes__": obj.hex()}
    if isinstance(obj, (list, tuple)):
        return [_encode_value(item) for item in obj]
    if isinstance(obj, set):
        # Sets become sorted lists for determinism
        return sorted(_encode_value(item) for item in obj)
    if isinstance(obj, frozenset):
        return sorted(_encode_value(item) for item in obj)
    if isinstance(obj, dict):
        return {str(k): _encode_value(v) for k, v in sorted(obj.items())}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _encode_value(dataclasses.asdict(obj))
    if hasattr(obj, "__dict__"):
        # Generic object: use its __dict__
        return _encode_value(vars(obj))

    raise CanonicalizeError(f"Cannot canonicalize type: {type(obj).__name__}")


def canonical(obj: Any) -> str:
    """
    Convert an object to a canonical JSON string.

    The output is deterministic: the same input always produces the same string,
    regardless of dict ordering or float representation quirks.

    Args:
        obj: The object to canonicalize. Must be JSON-compatible or a
             dataclass/object with __dict__.

    Returns:
        A canonical JSON string with sorted keys and consistent formatting.

    Raises:
        CanonicalizeError: If the object contains NaN, Inf, or non-serializable types.

    Example:
        >>> canonical({"b": 1, "a": 2})
        '{"a": 2, "b": 1}'
        >>> canonical({"x": 1.0000000000000001})
        '{"x": "1.0000000000000001"}'
    """
    encoded = _encode_value(obj)
    return json.dumps(encoded, sort_keys=True, separators=(",", ":"))


def fingerprint(obj: Any) -> str:
    """
    Compute a stable fingerprint (hash) of an object.

    Uses SHA-256 of the canonical representation, truncated to 16 hex characters.

    Args:
        obj: The object to fingerprint.

    Returns:
        A 16-character hex string.

    Raises:
        CanonicalizeError: If the object cannot be canonicalized.

    Example:
        >>> fingerprint({"a": 1, "b": 2})
        'a1b2c3d4e5f67890'  # (example, actual value differs)
    """
    canonical_str = canonical(obj)
    hash_bytes = hashlib.sha256(canonical_str.encode("utf-8")).digest()
    return hash_bytes.hex()[:16]
