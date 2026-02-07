"""Tests for metalab.environment.orchestrator module."""

from __future__ import annotations

import pytest
from metalab.config import ProjectInfo, ResolvedConfig
from metalab.environment.bundle import ServiceBundle
from metalab.environment.orchestrator import ServiceOrchestrator, ServiceStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def resolved_config():
    return ResolvedConfig(
        project=ProjectInfo(name="test"),
        env_name="local",
        env_type="local",
        env_config={},
        services={"postgres": {"database": "metalab"}, "atlas": {"port": 8000}},
        file_root="/tmp/test-root",
    )


@pytest.fixture
def resolved_config_no_services():
    return ResolvedConfig(
        project=ProjectInfo(name="test"),
        env_name="local",
        env_type="local",
        env_config={},
        services={},
        file_root="/tmp/test-root",
    )


# ---------------------------------------------------------------------------
# Bundle path
# ---------------------------------------------------------------------------


class TestBundlePath:
    def test_always_in_home(self, resolved_config):
        """Bundle path is always under ~/.metalab/services/{project}/{env}/."""
        orch = ServiceOrchestrator(resolved_config)
        path = str(orch._bundle_path)
        assert ".metalab/services/test/local/bundle.json" in path
        # Should NOT be under file_root
        assert "/tmp/test-root" not in path

    def test_includes_project_and_env(self):
        config = ResolvedConfig(
            project=ProjectInfo(name="myproject"),
            env_name="dev",
            env_type="local",
            env_config={},
            services={},
            file_root=None,
        )
        orch = ServiceOrchestrator(config)
        path = str(orch._bundle_path)
        assert "myproject" in path
        assert "dev" in path
        assert path.endswith("bundle.json")

    def test_default_project_name(self):
        """Falls back to 'default' when project name is empty."""
        config = ResolvedConfig(
            project=ProjectInfo(name=""),
            env_name="local",
            env_type="local",
            env_config={},
            services={},
            file_root=None,
        )
        orch = ServiceOrchestrator(config)
        assert "default/local" in str(orch._bundle_path)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_no_bundle(self, resolved_config, tmp_path):
        orch = ServiceOrchestrator(resolved_config)
        orch._bundle_path = tmp_path / "nonexistent" / "bundle.json"
        status = orch.status()
        assert not status.bundle_found
        assert status.services == {}

    def test_no_bundle_not_healthy(self, resolved_config, tmp_path):
        orch = ServiceOrchestrator(resolved_config)
        orch._bundle_path = tmp_path / "nonexistent" / "bundle.json"
        status = orch.status()
        assert not status.is_healthy()


# ---------------------------------------------------------------------------
# Down
# ---------------------------------------------------------------------------


class TestDown:
    def test_no_bundle(self, resolved_config, tmp_path):
        orch = ServiceOrchestrator(resolved_config)
        orch._bundle_path = tmp_path / "nonexistent" / "bundle.json"
        # Should not raise
        orch.down()


# ---------------------------------------------------------------------------
# ServiceStatus
# ---------------------------------------------------------------------------


class TestServiceStatus:
    def test_is_healthy_all_available(self):
        status = ServiceStatus(
            bundle_found=True,
            services={
                "postgres": {"available": True, "host": "localhost", "port": 5432, "status": "running"},
                "atlas": {"available": True, "host": "localhost", "port": 8000, "status": "running"},
            },
        )
        assert status.is_healthy()

    def test_unhealthy_when_unavailable(self):
        status = ServiceStatus(
            bundle_found=True,
            services={
                "postgres": {"available": False, "host": "localhost", "port": 5432, "status": "unreachable"},
            },
        )
        assert not status.is_healthy()

    def test_unhealthy_when_no_bundle(self):
        status = ServiceStatus(bundle_found=False, services={})
        assert not status.is_healthy()

    def test_healthy_with_no_services(self):
        # Bundle found but no services tracked — vacuously healthy
        status = ServiceStatus(bundle_found=True, services={})
        assert status.is_healthy()

    def test_mixed_availability(self):
        status = ServiceStatus(
            bundle_found=True,
            services={
                "postgres": {"available": True, "status": "running"},
                "atlas": {"available": False, "status": "unreachable"},
            },
        )
        assert not status.is_healthy()
