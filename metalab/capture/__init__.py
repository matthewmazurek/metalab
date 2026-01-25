"""
Capture module: Flexible artifact/metric emission with pluggable serializers.

Provides:
- Capture: Interface for emitting metrics, artifacts, and logs
- SerializerRegistry: Pluggable serialization system
- LogCapture: Capture stdout/stderr/logging output (legacy)
- OutputCapture: Thread-safe output capture for operations
"""

from metalab.capture.capture import Capture
from metalab.capture.logs import LogCapture, LogHandler, WarningCapture
from metalab.capture.output import (
    CapturedOutput,
    OutputCapture,
    OutputCaptureContext,
    OutputCaptureManager,
    normalize_output_capture,
)
from metalab.capture.registry import Serializer, SerializerRegistry

__all__ = [
    "Capture",
    "Serializer",
    "SerializerRegistry",
    # Legacy capture utilities
    "LogCapture",
    "LogHandler",
    "WarningCapture",
    # New output capture system
    "OutputCapture",
    "OutputCaptureContext",
    "OutputCaptureManager",
    "CapturedOutput",
    "normalize_output_capture",
]
