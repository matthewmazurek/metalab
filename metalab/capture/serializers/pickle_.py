"""
Pickle serializer - OPT-IN ONLY.

WARNING: Pickle has significant drawbacks:
- Non-portable across Python versions
- Insecure if loading from untrusted sources
- Brittle for long-term artifact compatibility
- Hides accidental huge-object persistence

This serializer is only used when:
- kind="pickle" is explicitly specified, OR
- Capture(allow_pickle=True) is set

A warning is emitted when pickle is used.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PickleSerializer:
    """
    Serializer using Python's pickle module.

    USE WITH CAUTION - see module docstring for warnings.
    """

    @property
    def kind(self) -> str:
        return "pickle"

    @property
    def format(self) -> str:
        return "pkl"

    def can_handle(self, obj: Any) -> bool:
        """
        Check if the object can be pickled.

        Note: Almost anything can be pickled, so this is a last-resort serializer.
        """
        try:
            pickle.dumps(obj)
            return True
        except (pickle.PicklingError, TypeError, AttributeError):
            return False

    def dump(self, obj: Any, path: Path) -> dict[str, Any]:
        """
        Serialize an object using pickle.

        Args:
            obj: The object to serialize.
            path: The base path (extension will be added).

        Returns:
            Metadata about the serialization.
        """
        logger.warning(
            f"Using pickle to serialize {type(obj).__name__}. "
            "Consider implementing a custom serializer for better portability."
        )

        actual_path = path.with_suffix(".pkl")
        actual_path.parent.mkdir(parents=True, exist_ok=True)

        with actual_path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {
            "path": str(actual_path),
            "format": self.format,
            "size_bytes": actual_path.stat().st_size,
            "_warning": "Pickled artifacts may not be portable across Python versions",
        }

    def load(self, path: Path) -> Any:
        """
        Deserialize an object from pickle.

        WARNING: Only load pickle files from trusted sources.

        Args:
            path: The path to read from.

        Returns:
            The deserialized object.
        """
        with path.open("rb") as f:
            return pickle.load(f)
