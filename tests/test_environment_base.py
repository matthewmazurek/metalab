"""Tests for metalab.environment base, connector, and bundle modules."""

from __future__ import annotations

import pytest
from metalab.environment.base import ServiceHandle, ServiceSpec
from metalab.environment.connector import ConnectionTarget, TunnelHandle
from metalab.environment.bundle import ServiceBundle


# ---------------------------------------------------------------------------
# ServiceSpec
# ---------------------------------------------------------------------------


class TestServiceSpec:
    def test_defaults(self):
        spec = ServiceSpec(name="postgres")
        assert spec.name == "postgres"
        assert spec.config == {}
        assert spec.resources == {}

    def test_with_config(self):
        spec = ServiceSpec(
            name="postgres",
            config={"database": "metalab"},
            resources={"memory": "4G"},
        )
        assert spec.config["database"] == "metalab"
        assert spec.resources["memory"] == "4G"


# ---------------------------------------------------------------------------
# ServiceHandle serialization
# ---------------------------------------------------------------------------


class TestServiceHandle:
    def test_round_trip(self):
        handle = ServiceHandle(
            name="postgres",
            host="fc48",
            port=5432,
            credentials={"user": "test", "password": "secret"},
            process_id="12345",
        )
        d = handle.to_dict()
        restored = ServiceHandle.from_dict(d)
        assert restored.name == "postgres"
        assert restored.host == "fc48"
        assert restored.port == 5432
        assert restored.credentials["password"] == "secret"
        assert restored.process_id == "12345"

    def test_defaults(self):
        handle = ServiceHandle(name="redis", host="localhost", port=6379)
        assert handle.status == "running"
        assert handle.credentials == {}
        assert handle.process_id is None
        assert handle.metadata == {}

    def test_from_dict_defaults(self):
        d = {"name": "redis", "host": "localhost", "port": 6379}
        handle = ServiceHandle.from_dict(d)
        assert handle.status == "running"
        assert handle.credentials == {}
        assert handle.process_id is None
        assert handle.metadata == {}

    def test_to_dict_contains_all_fields(self):
        handle = ServiceHandle(
            name="pg", host="h", port=1234,
            status="stopped", credentials={"k": "v"},
            process_id="99", metadata={"m": "val"},
        )
        d = handle.to_dict()
        assert set(d.keys()) == {
            "name", "host", "port", "status",
            "credentials", "process_id", "metadata",
        }


# ---------------------------------------------------------------------------
# ConnectionTarget / TunnelHandle
# ---------------------------------------------------------------------------


class TestConnectionTarget:
    def test_basic(self):
        target = ConnectionTarget(
            remote_host="fc48",
            remote_port=8000,
            local_port=8000,
            gateway="gateway.example.com",
            user="testuser",
        )
        assert target.gateway == "gateway.example.com"
        assert target.user == "testuser"
        assert target.remote_host == "fc48"

    def test_defaults(self):
        target = ConnectionTarget(
            remote_host="node01", remote_port=5432, local_port=5432,
        )
        assert target.gateway is None
        assert target.user is None
        assert target.ssh_key is None


class TestTunnelHandle:
    def test_local_url(self):
        handle = TunnelHandle(
            local_host="127.0.0.1",
            local_port=8000,
            remote_host="fc48",
            remote_port=8000,
        )
        assert handle.local_url == "http://127.0.0.1:8000"

    def test_pid_default(self):
        handle = TunnelHandle(
            local_host="127.0.0.1", local_port=9000,
            remote_host="node", remote_port=9000,
        )
        assert handle.pid is None


# ---------------------------------------------------------------------------
# ServiceBundle serialization
# ---------------------------------------------------------------------------


class TestServiceBundle:
    def _make_bundle(self) -> ServiceBundle:
        bundle = ServiceBundle(environment="slurm", profile="test")
        bundle.add(
            "postgres",
            ServiceHandle(
                name="postgres",
                host="node01",
                port=5432,
                credentials={"user": "test", "password": "pw"},
                process_id="111",
            ),
        )
        bundle.store_locator = "postgresql://test:pw@node01:5432/metalab"
        return bundle

    def test_round_trip(self, tmp_path):
        bundle = self._make_bundle()
        path = tmp_path / "services" / "bundle.json"
        bundle.save(path)

        loaded = ServiceBundle.load(path)
        assert loaded.environment == "slurm"
        assert loaded.profile == "test"
        assert loaded.store_locator == bundle.store_locator
        assert "postgres" in loaded.services
        assert loaded.services["postgres"].host == "node01"

    def test_save_creates_parent_dirs(self, tmp_path):
        bundle = ServiceBundle(environment="local", profile="local")
        path = tmp_path / "deep" / "nested" / "bundle.json"
        bundle.save(path)
        assert path.exists()

    def test_file_permissions(self, tmp_path):
        bundle = ServiceBundle(environment="local", profile="local")
        path = tmp_path / "bundle.json"
        bundle.save(path)
        import stat
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600

    def test_remove(self, tmp_path):
        path = tmp_path / "bundle.json"
        bundle = ServiceBundle(environment="local", profile="local")
        bundle.save(path)
        assert path.exists()
        bundle.remove(path)
        assert not path.exists()

    def test_remove_nonexistent(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        bundle = ServiceBundle(environment="local", profile="local")
        # Should not raise
        bundle.remove(path)

    def test_add_and_get(self):
        bundle = ServiceBundle(environment="local", profile="local")
        handle = ServiceHandle(name="pg", host="localhost", port=5432)
        bundle.add("pg", handle)
        assert bundle.get("pg") is handle
        assert bundle.get("nonexistent") is None

    def test_to_dict(self):
        bundle = self._make_bundle()
        d = bundle.to_dict()
        assert d["environment"] == "slurm"
        assert d["profile"] == "test"
        assert "postgres" in d["services"]
        assert d["store_locator"] is not None
        assert "created_at" in d

    def test_find_nearest_via_config(self, tmp_path, monkeypatch):
        """find_nearest discovers bundle via .metalab.toml project/env."""
        # Set BUNDLE_HOME to a temp location
        bundle_home = tmp_path / "bundle_home"
        monkeypatch.setattr(ServiceBundle, "BUNDLE_HOME", bundle_home)

        # Create a .metalab.toml in the project root
        config = tmp_path / "project" / ".metalab.toml"
        config.parent.mkdir()
        config.write_text(
            '[project]\nname = "myproject"\n\n'
            '[environments.local]\ntype = "local"\n'
        )

        # Save bundle at the canonical path
        bundle = ServiceBundle(environment="local", profile="local")
        bundle_path = bundle_home / "myproject" / "local" / "bundle.json"
        bundle.save(bundle_path)

        # Search from a subdirectory of the project
        subdir = tmp_path / "project" / "a" / "b"
        subdir.mkdir(parents=True)
        found = ServiceBundle.find_nearest(subdir)
        assert found is not None
        assert found.environment == "local"

    def test_find_nearest_scan_fallback(self, tmp_path, monkeypatch):
        """find_nearest falls back to scanning BUNDLE_HOME."""
        bundle_home = tmp_path / "bundle_home"
        monkeypatch.setattr(ServiceBundle, "BUNDLE_HOME", bundle_home)

        # Save bundle without any .metalab.toml nearby
        bundle = ServiceBundle(environment="slurm", profile="hpc")
        bundle_path = bundle_home / "someproject" / "slurm" / "bundle.json"
        bundle.save(bundle_path)

        # Search from an unrelated directory (no .metalab.toml)
        search_dir = tmp_path / "random"
        search_dir.mkdir()
        found = ServiceBundle.find_nearest(search_dir)
        assert found is not None
        assert found.environment == "slurm"

    def test_find_nearest_not_found(self, tmp_path, monkeypatch):
        """find_nearest returns None when no bundle exists."""
        bundle_home = tmp_path / "empty_home"
        monkeypatch.setattr(ServiceBundle, "BUNDLE_HOME", bundle_home)
        found = ServiceBundle.find_nearest(tmp_path)
        assert found is None
