"""Unit tests for context spec JSON serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from metalab._ids import DirPath, FilePath
from metalab.context.spec import deserialize_context_spec, serialize_context_spec


@dataclass(frozen=True)
class SimpleContext:
    """Simple context for testing."""

    name: str
    value: int


@dataclass(frozen=True)
class NestedContext:
    """Context with nested dataclasses."""

    name: str
    file: FilePath
    dir: DirPath


@dataclass(frozen=True)
class ComplexContext:
    """Context with various data types."""

    name: str
    values: list[int]
    config: dict[str, Any]
    enabled: bool
    threshold: float


class TestContextSpecSerialization:
    """Tests for context spec serialization."""

    def test_serialize_simple_dict(self):
        """Simple dicts should serialize cleanly."""
        ctx = {"name": "test", "value": 42}
        result = serialize_context_spec(ctx)

        assert result == {"name": "test", "value": 42}
        # Should be JSON serializable
        json.dumps(result)

    def test_serialize_simple_dataclass(self):
        """Simple dataclasses should include __type__ info."""
        ctx = SimpleContext(name="test", value=42)
        result = serialize_context_spec(ctx)

        assert "__type__" in result
        assert result["name"] == "test"
        assert result["value"] == 42
        # Should be JSON serializable
        json.dumps(result)

    def test_serialize_filepath(self):
        """FilePath should serialize with type info."""
        fp = FilePath("/path/to/file.txt")
        result = serialize_context_spec(fp)

        assert "__type__" in result
        assert "FilePath" in result["__type__"]
        assert result["path"] == "/path/to/file.txt"
        assert result["hash_path"] is False

    def test_serialize_dirpath(self):
        """DirPath should serialize with type info."""
        dp = DirPath("/path/to/dir", pattern="*.csv")
        result = serialize_context_spec(dp)

        assert "__type__" in result
        assert "DirPath" in result["__type__"]
        assert result["path"] == "/path/to/dir"
        assert result["pattern"] == "*.csv"

    def test_serialize_nested_context(self):
        """Nested dataclasses should serialize recursively."""
        ctx = NestedContext(
            name="test",
            file=FilePath("/data/input.csv"),
            dir=DirPath("/data/outputs", pattern="*.json"),
        )
        result = serialize_context_spec(ctx)

        assert "__type__" in result
        assert result["name"] == "test"
        assert "__type__" in result["file"]
        assert "FilePath" in result["file"]["__type__"]
        assert "__type__" in result["dir"]
        assert "DirPath" in result["dir"]["__type__"]

    def test_serialize_complex_types(self):
        """Lists, dicts, and primitives should serialize."""
        ctx = ComplexContext(
            name="test",
            values=[1, 2, 3],
            config={"lr": 0.01, "nested": {"a": 1}},
            enabled=True,
            threshold=0.5,
        )
        result = serialize_context_spec(ctx)

        assert result["values"] == [1, 2, 3]
        assert result["config"]["lr"] == 0.01
        assert result["config"]["nested"]["a"] == 1
        assert result["enabled"] is True
        assert result["threshold"] == 0.5


class TestContextSpecDeserialization:
    """Tests for context spec deserialization."""

    def test_deserialize_simple_dict(self):
        """Simple dicts should pass through unchanged."""
        data = {"name": "test", "value": 42}
        result = deserialize_context_spec(data)

        assert result == {"name": "test", "value": 42}

    def test_deserialize_filepath(self):
        """FilePath should be reconstructed from type info."""
        data = {
            "__type__": "metalab._ids.FilePath",
            "path": "/path/to/file.txt",
            "hash_path": False,
        }
        result = deserialize_context_spec(data)

        assert isinstance(result, FilePath)
        assert result.path == "/path/to/file.txt"
        assert result.hash_path is False

    def test_deserialize_dirpath(self):
        """DirPath should be reconstructed from type info."""
        data = {
            "__type__": "metalab._ids.DirPath",
            "path": "/path/to/dir",
            "pattern": "*.csv",
            "hash_path": False,
        }
        result = deserialize_context_spec(data)

        assert isinstance(result, DirPath)
        assert result.path == "/path/to/dir"
        assert result.pattern == "*.csv"

    def test_deserialize_nested_lists(self):
        """Lists should be deserialized recursively."""
        data = [
            {"__type__": "metalab._ids.FilePath", "path": "/a.txt", "hash_path": False},
            {"__type__": "metalab._ids.FilePath", "path": "/b.txt", "hash_path": True},
        ]
        result = deserialize_context_spec(data)

        assert len(result) == 2
        assert isinstance(result[0], FilePath)
        assert isinstance(result[1], FilePath)
        assert result[0].path == "/a.txt"
        assert result[1].hash_path is True

    def test_round_trip_simple_context(self):
        """Serialize and deserialize should preserve data."""
        ctx = SimpleContext(name="test", value=42)
        serialized = serialize_context_spec(ctx)
        deserialized = deserialize_context_spec(serialized)

        assert isinstance(deserialized, SimpleContext)
        assert deserialized.name == "test"
        assert deserialized.value == 42

    def test_round_trip_nested_context(self):
        """Round trip should work for nested contexts."""
        ctx = NestedContext(
            name="test",
            file=FilePath("/data/input.csv"),
            dir=DirPath("/data/outputs", pattern="*.json"),
        )
        serialized = serialize_context_spec(ctx)
        deserialized = deserialize_context_spec(serialized)

        assert isinstance(deserialized, NestedContext)
        assert deserialized.name == "test"
        assert isinstance(deserialized.file, FilePath)
        assert deserialized.file.path == "/data/input.csv"
        assert isinstance(deserialized.dir, DirPath)
        assert deserialized.dir.pattern == "*.json"

    def test_json_round_trip(self):
        """Should survive JSON serialization."""
        ctx = NestedContext(
            name="test",
            file=FilePath("/data/input.csv"),
            dir=DirPath("/data/outputs"),
        )
        serialized = serialize_context_spec(ctx)
        json_str = json.dumps(serialized)
        json_data = json.loads(json_str)
        deserialized = deserialize_context_spec(json_data)

        assert isinstance(deserialized, NestedContext)
        assert isinstance(deserialized.file, FilePath)
        assert isinstance(deserialized.dir, DirPath)
