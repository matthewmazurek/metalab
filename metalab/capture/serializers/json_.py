"""
JSON serializer (stdlib).

The default serializer for metalab. Always available, always registered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JSONSerializer:
    """
    Serializer for JSON-compatible objects.

    Handles: dict, list, str, int, float, bool, None
    """

    @property
    def kind(self) -> str:
        return "json"

    @property
    def format(self) -> str:
        return "json"

    def can_handle(self, obj: Any) -> bool:
        """Check if the object is JSON-serializable."""
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False

    def dump(self, obj: Any, path: Path) -> dict[str, Any]:
        """
        Serialize an object to JSON.

        Args:
            obj: The object to serialize.
            path: The base path (extension will be added).

        Returns:
            Metadata about the serialization.
        """
        actual_path = path.with_suffix(".json")
        actual_path.parent.mkdir(parents=True, exist_ok=True)

        content = json.dumps(obj, indent=2, sort_keys=True)
        actual_path.write_text(content, encoding="utf-8")

        return {
            "path": str(actual_path),
            "format": self.format,
            "size_bytes": actual_path.stat().st_size,
        }

    def load(self, path: Path) -> Any:
        """
        Deserialize an object from JSON.

        Args:
            path: The path to read from.

        Returns:
            The deserialized object.
        """
        content = path.read_text(encoding="utf-8")
        return json.loads(content)
