"""
Capture module: Flexible artifact/metric emission with pluggable serializers.

Provides:
- Capture: Interface for emitting metrics, artifacts, and logs
- SerializerRegistry: Pluggable serialization system
- LogCapture: Capture stdout/stderr/logging output
"""

from metalab.capture.capture import Capture
from metalab.capture.logs import LogCapture, LogHandler, WarningCapture
from metalab.capture.registry import Serializer, SerializerRegistry

__all__ = [
    "Capture",
    "Serializer",
    "SerializerRegistry",
    "LogCapture",
    "LogHandler",
    "WarningCapture",
]
