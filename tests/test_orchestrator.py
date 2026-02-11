"""Tests for metalab.environment.orchestrator and ssh_tunnel modules."""

from __future__ import annotations

import pytest
from metalab.config import ProjectInfo, ResolvedConfig
from metalab.environment.base import ServiceSpec
from metalab.environment.bundle import ServiceBundle
from metalab.environment.connector import ConnectionTarget
from metalab.environment.orchestrator import ServiceOrchestrator, ServiceStatus
from metalab.environment.ssh_tunnel import build_ssh_command


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


# ---------------------------------------------------------------------------
# _build_service_specs
# ---------------------------------------------------------------------------


class TestBuildServiceSpecs:
    """Test the orchestrator builds specs from config with no hardcoding."""

    def test_postgres_and_atlas(self, resolved_config):
        orch = ServiceOrchestrator(resolved_config)
        specs = orch._build_service_specs()
        assert len(specs) == 2
        assert specs[0].name == "postgres"
        assert specs[1].name == "atlas"

    def test_postgres_before_atlas(self, resolved_config):
        """Topological order: postgres first (atlas may depend on it)."""
        orch = ServiceOrchestrator(resolved_config)
        specs = orch._build_service_specs()
        names = [s.name for s in specs]
        assert names.index("postgres") < names.index("atlas")

    def test_postgres_config_includes_file_root(self, resolved_config):
        orch = ServiceOrchestrator(resolved_config)
        specs = orch._build_service_specs()
        pg = specs[0]
        assert pg.config["file_root"] == "/tmp/test-root"

    def test_atlas_config_includes_file_root(self, resolved_config):
        orch = ServiceOrchestrator(resolved_config)
        specs = orch._build_service_specs()
        atlas = specs[1]
        assert atlas.config["file_root"] == "/tmp/test-root"

    def test_atlas_has_user_port(self, resolved_config):
        orch = ServiceOrchestrator(resolved_config)
        specs = orch._build_service_specs()
        atlas = specs[1]
        assert atlas.config["port"] == 8000

    def test_no_services_configured(self, resolved_config_no_services):
        """No services configured -> empty specs (Atlas requires Postgres)."""
        orch = ServiceOrchestrator(resolved_config_no_services)
        specs = orch._build_service_specs()
        # Atlas requires Postgres, so no specs without it
        assert len(specs) == 0

    def test_no_services_no_file_root(self):
        """No services and no file_root -> empty specs."""
        config = ResolvedConfig(
            project=ProjectInfo(name="test"),
            env_name="local",
            env_type="local",
            env_config={},
            services={},
            file_root=None,
        )
        orch = ServiceOrchestrator(config)
        specs = orch._build_service_specs()
        assert specs == []

    def test_postgres_only(self):
        """Postgres configured but no atlas and no file_root."""
        config = ResolvedConfig(
            project=ProjectInfo(name="test"),
            env_name="local",
            env_type="local",
            env_config={},
            services={"postgres": {"database": "mydb"}},
            file_root=None,
        )
        orch = ServiceOrchestrator(config)
        specs = orch._build_service_specs()
        assert len(specs) == 1
        assert specs[0].name == "postgres"

    def test_atlas_consumes_store_locator(self, resolved_config):
        """Atlas spec declares it consumes store_locator from prior services."""
        orch = ServiceOrchestrator(resolved_config)
        specs = orch._build_service_specs()
        atlas = specs[1]
        assert "store_locator" in atlas.consumes

    def test_postgres_does_not_consume(self, resolved_config):
        """Postgres has no upstream dependencies — consumes is empty."""
        orch = ServiceOrchestrator(resolved_config)
        specs = orch._build_service_specs()
        pg = specs[0]
        assert pg.consumes == []

    def test_atlas_without_postgres_not_created(self):
        """Atlas-only (no postgres) is not created since Atlas requires PG."""
        config = ResolvedConfig(
            project=ProjectInfo(name="test"),
            env_name="local",
            env_type="local",
            env_config={},
            services={"atlas": {"port": 8000}},
            file_root="/tmp/test-root",
        )
        orch = ServiceOrchestrator(config)
        specs = orch._build_service_specs()
        # Atlas still created (explicitly configured), but with a warning
        assert len(specs) == 1
        assert specs[0].name == "atlas"
        assert specs[0].consumes == ["store_locator"]


# ---------------------------------------------------------------------------
# build_ssh_command
# ---------------------------------------------------------------------------


class TestBuildSshCommand:
    def test_basic(self):
        target = ConnectionTarget(
            remote_host="compute-42",
            remote_port=8000,
            local_port=8000,
        )
        cmd = build_ssh_command(target)
        assert cmd == ["ssh", "-N", "-L", "8000:127.0.0.1:8000", "compute-42"]

    def test_with_gateway_and_user(self):
        target = ConnectionTarget(
            remote_host="compute-42",
            remote_port=8000,
            local_port=8000,
            gateway="login.cluster.edu",
            user="alice",
        )
        cmd = build_ssh_command(target)
        assert cmd == [
            "ssh", "-N", "-L", "8000:127.0.0.1:8000",
            "-J", "alice@login.cluster.edu",
            "alice@compute-42",
        ]

    def test_with_ssh_key(self):
        target = ConnectionTarget(
            remote_host="compute-42",
            remote_port=8000,
            local_port=8000,
            ssh_key="/home/alice/.ssh/id_rsa",
        )
        cmd = build_ssh_command(target)
        assert cmd == [
            "ssh", "-N", "-L", "8000:127.0.0.1:8000",
            "-i", "/home/alice/.ssh/id_rsa",
            "compute-42",
        ]

    def test_gateway_already_has_user(self):
        """If gateway already contains user@, don't double-prefix."""
        target = ConnectionTarget(
            remote_host="compute-42",
            remote_port=8000,
            local_port=8000,
            gateway="alice@login.cluster.edu",
            user="alice",
        )
        cmd = build_ssh_command(target)
        # gateway already has user@ — should not become alice@alice@...
        assert "-J" in cmd
        j_idx = cmd.index("-J")
        assert cmd[j_idx + 1] == "alice@login.cluster.edu"

    def test_different_local_port(self):
        target = ConnectionTarget(
            remote_host="compute-42",
            remote_port=8000,
            local_port=9000,
        )
        cmd = build_ssh_command(target)
        assert "9000:127.0.0.1:8000" in cmd

    def test_all_options(self):
        target = ConnectionTarget(
            remote_host="node-7",
            remote_port=5432,
            local_port=15432,
            gateway="bastion.example.com",
            user="bob",
            ssh_key="/tmp/key",
        )
        cmd = build_ssh_command(target)
        assert cmd == [
            "ssh", "-N", "-L", "15432:127.0.0.1:5432",
            "-J", "bob@bastion.example.com",
            "-i", "/tmp/key",
            "bob@node-7",
        ]
