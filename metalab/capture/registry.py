"""
SerializerRegistry: Pluggable serialization system.

Design decisions:
- JSON is the default serializer (always registered)
- Pickle is OPT-IN ONLY (requires explicit kind="pickle" or allow_pickle=True)
- NumPy serializer is available when numpy is installed
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Serializer(Protocol):
    """
    Protocol for artifact serializers.

    A serializer handles a specific type of object, converting it to/from
    a file on disk.
    """

    @property
    def kind(self) -> str:
        """The kind of artifacts this serializer handles."""
        ...

    @property
    def format(self) -> str:
        """The file format this serializer produces (e.g., 'json', 'npz')."""
        ...

    def can_handle(self, obj: Any) -> bool:
        """
        Check if this serializer can handle the given object.

        Args:
            obj: The object to check.

        Returns:
            True if this serializer can serialize the object.
        """
        ...

    def dump(self, obj: Any, path: Path) -> dict[str, Any]:
        """
        Serialize an object to a file.

        Args:
            obj: The object to serialize.
            path: The path to write to (without extension).

        Returns:
            A dict with metadata about the serialization:
            - 'path': The actual path written (with extension)
            - 'format': The format used
            - 'size_bytes': Optional size in bytes
            - Additional serializer-specific metadata
        """
        ...

    def load(self, path: Path) -> Any:
        """
        Deserialize an object from a file.

        Args:
            path: The path to read from.

        Returns:
            The deserialized object.
        """
        ...


class SerializerRegistry:
    """
    Registry of serializers for different artifact types.

    By default, only the JSON serializer is registered.
    Pickle must be explicitly enabled.
    """

    def __init__(self, allow_pickle: bool = False) -> None:
        """
        Initialize the registry.

        Args:
            allow_pickle: If True, allow pickle serializer as fallback.
        """
        self._serializers: dict[str, Serializer] = {}
        self._fallback_order: list[str] = []
        self._allow_pickle = allow_pickle

        # Register default serializers
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register the default serializers."""
        from metalab.capture.serializers.json_ import JSONSerializer

        self.register(JSONSerializer())

        # Try to register numpy serializer if available
        try:
            from metalab.capture.serializers.numpy_ import NumpySerializer

            self.register(NumpySerializer())
        except ImportError:
            pass  # numpy not installed

        # Pickle is only registered if allow_pickle=True
        if self._allow_pickle:
            from metalab.capture.serializers.pickle_ import PickleSerializer

            self.register(PickleSerializer())

    def register(self, serializer: Serializer) -> None:
        """
        Register a serializer.

        Args:
            serializer: The serializer to register.
        """
        self._serializers[serializer.kind] = serializer
        if serializer.kind not in self._fallback_order:
            self._fallback_order.append(serializer.kind)

    def get(self, kind: str) -> Serializer | None:
        """
        Get a serializer by kind.

        Args:
            kind: The serializer kind.

        Returns:
            The serializer, or None if not found.
        """
        return self._serializers.get(kind)

    def find(self, obj: Any, kind: str | None = None) -> Serializer:
        """
        Find a serializer for an object.

        Args:
            obj: The object to serialize.
            kind: Optional explicit kind to use.

        Returns:
            A suitable serializer.

        Raises:
            ValueError: If no suitable serializer is found.
        """
        # If explicit kind requested
        if kind is not None:
            # Special handling for pickle
            if kind == "pickle":
                if not self._allow_pickle:
                    raise ValueError(
                        "Pickle serializer requires allow_pickle=True. "
                        "This is disabled by default for safety."
                    )
                from metalab.capture.serializers.pickle_ import PickleSerializer

                if "pickle" not in self._serializers:
                    self.register(PickleSerializer())
                return self._serializers["pickle"]

            serializer = self._serializers.get(kind)
            if serializer is None:
                raise ValueError(f"No serializer registered for kind: {kind}")
            return serializer

        # Auto-detect based on can_handle
        for kind_name in self._fallback_order:
            serializer = self._serializers[kind_name]
            if serializer.can_handle(obj):
                return serializer

        # If allow_pickle, use it as last resort
        if self._allow_pickle:
            logger.warning(
                f"Using pickle serializer for {type(obj).__name__}. "
                "Consider implementing a custom serializer for better portability."
            )
            from metalab.capture.serializers.pickle_ import PickleSerializer

            if "pickle" not in self._serializers:
                self.register(PickleSerializer())
            return self._serializers["pickle"]

        raise ValueError(
            f"No serializer can handle object of type {type(obj).__name__}. "
            "Consider using kind='pickle' with allow_pickle=True, or "
            "implement a custom serializer."
        )

    @property
    def kinds(self) -> list[str]:
        """List of registered serializer kinds."""
        return list(self._serializers.keys())
