"""
NumPy serializer (optional).

Handles numpy arrays and dicts of arrays using .npz format.
Only available when numpy is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore


class NumpySerializer:
    """
    Serializer for NumPy arrays and dicts of arrays.

    Uses .npz format for efficient storage of numerical data.
    """

    def __init__(self) -> None:
        if not HAS_NUMPY:
            raise ImportError(
                "numpy is required for NumpySerializer. "
                "Install it with: pip install metalab[numpy]"
            )

    @property
    def kind(self) -> str:
        return "numpy"

    @property
    def format(self) -> str:
        return "npz"

    def can_handle(self, obj: Any) -> bool:
        """Check if the object is a numpy array or dict of arrays."""
        if not HAS_NUMPY:
            return False

        # Single array
        if isinstance(obj, np.ndarray):
            return True

        # Dict of arrays (common pattern)
        if isinstance(obj, dict):
            return all(isinstance(v, np.ndarray) for v in obj.values())

        return False

    def dump(self, obj: Any, path: Path) -> dict[str, Any]:
        """
        Serialize numpy arrays to .npz format.

        Args:
            obj: A numpy array or dict of arrays.
            path: The base path (extension will be added).

        Returns:
            Metadata about the serialization.
        """
        actual_path = path.with_suffix(".npz")
        actual_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(obj, np.ndarray):
            # Single array: save with key 'arr'
            np.savez_compressed(actual_path, arr=obj)
            shape_info = {"arr": list(obj.shape)}
        else:
            # Dict of arrays
            np.savez_compressed(actual_path, **obj)
            shape_info = {k: list(v.shape) for k, v in obj.items()}

        return {
            "path": str(actual_path),
            "format": self.format,
            "size_bytes": actual_path.stat().st_size,
            "shapes": shape_info,
        }

    def load(self, path: Path) -> Any:
        """
        Deserialize numpy arrays from .npz format.

        Args:
            path: The path to read from.

        Returns:
            A dict of arrays, or a single array if only 'arr' key exists.
        """
        with np.load(path) as data:
            # If single array was saved, return just the array
            if list(data.keys()) == ["arr"]:
                return data["arr"]
            # Otherwise return dict of arrays
            return {k: data[k] for k in data.keys()}
