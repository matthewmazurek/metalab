"""Unit tests for HandleRegistry and reconnect() function."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar
from unittest.mock import MagicMock, patch

import pytest

from metalab.executor.handle import RunHandle, RunStatus
from metalab.executor.registry import HandleRegistry

if TYPE_CHECKING:
    from metalab.events import EventCallback
    from metalab.store.base import Store


class TestHandleRegistry:
    """Tests for HandleRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving a handle class."""

        # Create a mock handle class
        class MockHandle:
            executor_type: ClassVar[str] = "mock"

            @classmethod
            def from_store(cls, store, on_event=None):
                return cls()

        # Register it
        HandleRegistry.register("mock", MockHandle)

        # Retrieve it
        retrieved = HandleRegistry.get("mock")
        assert retrieved is MockHandle

    def test_get_unknown_returns_none(self):
        """Test that getting an unknown type returns None."""
        result = HandleRegistry.get("nonexistent_executor_type_xyz")
        assert result is None

    def test_types_lists_registered(self):
        """Test that types() returns registered executor types."""
        # slurm should be registered via the import of SlurmRunHandle
        from metalab.executor.slurm import SlurmRunHandle  # noqa: F401

        types = HandleRegistry.types()
        assert "slurm" in types

    def test_slurm_handle_registered(self):
        """Test that SlurmRunHandle is registered."""
        from metalab.executor.slurm import SlurmRunHandle

        handle_class = HandleRegistry.get("slurm")
        assert handle_class is SlurmRunHandle


class TestReconnectLocalExecutorError:
    """Tests for reconnect() rejecting local executors."""

    def _create_manifest(self, tmp_path: Path, executor_type: str) -> Path:
        """Helper to create a manifest file."""
        manifest_path = tmp_path / "manifest.json"
        manifest = {
            "experiment_id": "test_exp:1.0",
            "executor_type": executor_type,
            "total_runs": 10,
        }
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    def test_rejects_local_executor(self, tmp_path):
        """Test that reconnect() raises error for local executor type."""
        from metalab.runner import reconnect
        from metalab.store.file import FileStoreConfig

        # Create manifest with local executor type
        self._create_manifest(tmp_path, "local")

        # Attempt to reconnect
        with pytest.raises(ValueError) as exc_info:
            reconnect(str(tmp_path))

        assert "local" in str(exc_info.value)
        assert "load_results()" in str(exc_info.value)

    def test_rejects_thread_executor(self, tmp_path):
        """Test that reconnect() raises error for thread executor type."""
        from metalab.runner import reconnect

        # Create manifest with thread executor type
        self._create_manifest(tmp_path, "thread")

        # Attempt to reconnect
        with pytest.raises(ValueError) as exc_info:
            reconnect(str(tmp_path))

        assert "thread" in str(exc_info.value)
        assert "load_results()" in str(exc_info.value)

    def test_rejects_process_executor(self, tmp_path):
        """Test that reconnect() raises error for process executor type."""
        from metalab.runner import reconnect

        # Create manifest with process executor type
        self._create_manifest(tmp_path, "process")

        # Attempt to reconnect
        with pytest.raises(ValueError) as exc_info:
            reconnect(str(tmp_path))

        assert "process" in str(exc_info.value)
        assert "load_results()" in str(exc_info.value)


class TestReconnectUnknownExecutor:
    """Tests for reconnect() handling unknown executor types."""

    def test_rejects_unknown_executor(self, tmp_path):
        """Test that reconnect() raises error for unknown executor type."""
        from metalab.runner import reconnect

        # Create manifest with unknown executor type
        manifest_path = tmp_path / "manifest.json"
        manifest = {
            "experiment_id": "test_exp:1.0",
            "executor_type": "unknown_executor_xyz",
            "total_runs": 10,
        }
        manifest_path.write_text(json.dumps(manifest))

        # Attempt to reconnect
        with pytest.raises(ValueError) as exc_info:
            reconnect(str(tmp_path))

        assert "unknown_executor_xyz" in str(exc_info.value)
        assert "No reconnectable handle registered" in str(exc_info.value)


class TestReconnectMissingManifest:
    """Tests for reconnect() handling missing manifest."""

    def test_raises_on_missing_manifest(self, tmp_path):
        """Test that reconnect() raises FileNotFoundError when no manifest."""
        from metalab.runner import reconnect

        # Empty directory - no manifest
        with pytest.raises(FileNotFoundError) as exc_info:
            reconnect(str(tmp_path))

        assert "manifest" in str(exc_info.value).lower()


class TestReconnectStoreConfig:
    """Tests for reconnect() accepting StoreConfig."""

    def test_accepts_store_config(self, tmp_path):
        """Test that reconnect() accepts StoreConfig instead of string."""
        from metalab.runner import reconnect
        from metalab.store.file import FileStoreConfig

        # Create manifest with local executor (will fail, but tests config parsing)
        manifest_path = tmp_path / "manifest.json"
        manifest = {
            "experiment_id": "test_exp:1.0",
            "executor_type": "local",
        }
        manifest_path.write_text(json.dumps(manifest))

        # Use StoreConfig directly
        config = FileStoreConfig(root=str(tmp_path))

        # Should raise ValueError for local executor (not TypeError for config)
        with pytest.raises(ValueError) as exc_info:
            reconnect(config)

        assert "local" in str(exc_info.value)


class TestReconnectLocatorParsing:
    """Tests for reconnect() parsing locator URIs."""

    def test_parses_file_uri(self, tmp_path):
        """Test that reconnect() parses file:// URIs."""
        from metalab.runner import reconnect

        # Create manifest
        manifest_path = tmp_path / "manifest.json"
        manifest = {
            "experiment_id": "test_exp:1.0",
            "executor_type": "local",
        }
        manifest_path.write_text(json.dumps(manifest))

        # Use file:// URI
        uri = f"file://{tmp_path}"

        # Should parse URI and fail on local executor (not URI parsing)
        with pytest.raises(ValueError) as exc_info:
            reconnect(uri)

        assert "local" in str(exc_info.value)
