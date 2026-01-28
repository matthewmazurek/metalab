"""
ManifestSerializable protocol for experiment manifest serialization.

This protocol allows objects to define how they should be serialized
when writing experiment manifests. Built-in param sources and seed plans
implement this protocol; custom implementations can opt-in.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ManifestSerializable(Protocol):
    """
    Protocol for objects that can be serialized to experiment manifests.

    Implement this to enable proper JSON serialization of custom param sources,
    seed plans, or other experiment components.

    Example:
        class MyCustomSource:
            def __init__(self, values: list[int]):
                self._values = values

            def to_manifest_dict(self) -> dict[str, Any]:
                return {
                    "type": "MyCustomSource",
                    "values": self._values,
                    "total_cases": len(self._values),
                }
    """

    def to_manifest_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serializable dict representation.

        Should include:
        - "type": class name for identification
        - All configuration needed to understand/recreate the object

        Returns:
            A dictionary that can be serialized to JSON.
        """
        ...
