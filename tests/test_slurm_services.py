"""Tests for service plugins and SLURM fragment composition."""

from __future__ import annotations

import pytest
from metalab.environment.base import ReadinessCheck, ServiceHandle, ServiceSpec
from metalab.environment.slurm import SlurmEnvironment, SlurmFragment


# ---------------------------------------------------------------------------
# Postgres plugin — plan_slurm
# ---------------------------------------------------------------------------


class TestPostgresSlurmProvider:
    """Test PostgresPlugin.plan_slurm returns a valid SlurmFragment."""

    @pytest.fixture
    def pg_spec(self):
        return ServiceSpec(
            name="postgres",
            config={
                "file_root": "/shared/experiments",
                "database": "metalab",
                "auth_method": "scram-sha-256",
                "password": "test-password-123",
                "port": 5432,
                "user": "researcher",
            },
        )

    @pytest.fixture
    def env_config(self):
        return {
            "file_root": "/shared/experiments",
            "user": "researcher",
            "services": {"partition": "cpu2019", "time": "7-00:00:00", "memory": "10G"},
        }

    def test_returns_slurm_fragment(self, pg_spec, env_config, tmp_path):
        from metalab.services.postgres import PostgresPlugin

        pg_spec.config["file_root"] = str(tmp_path)
        env_config["file_root"] = str(tmp_path)

        plugin = PostgresPlugin()
        frag = plugin.plan(pg_spec, "slurm", env_config)

        assert isinstance(frag, SlurmFragment)
        assert frag.name == "postgres"
        assert frag.cpus == 2

    def test_setup_bash_has_pg_commands(self, pg_spec, env_config, tmp_path):
        from metalab.services.postgres import PostgresPlugin

        pg_spec.config["file_root"] = str(tmp_path)
        env_config["file_root"] = str(tmp_path)

        plugin = PostgresPlugin()
        frag = plugin.plan(pg_spec, "slurm", env_config)

        assert "pg_ctl" in frag.setup_bash
        assert "initdb" in frag.setup_bash
        assert "pg_isready" in frag.setup_bash
        assert "createdb" in frag.setup_bash
        assert "service.json" in frag.setup_bash

    def test_cleanup_bash_stops_postgres(self, pg_spec, env_config, tmp_path):
        from metalab.services.postgres import PostgresPlugin

        pg_spec.config["file_root"] = str(tmp_path)
        env_config["file_root"] = str(tmp_path)

        plugin = PostgresPlugin()
        frag = plugin.plan(pg_spec, "slurm", env_config)

        assert "pg_ctl" in frag.cleanup_bash
        assert "stop" in frag.cleanup_bash
        assert "-m fast" in frag.cleanup_bash

    def test_readiness_has_port_and_file(self, pg_spec, env_config, tmp_path):
        from metalab.services.postgres import PostgresPlugin

        pg_spec.config["file_root"] = str(tmp_path)
        env_config["file_root"] = str(tmp_path)

        plugin = PostgresPlugin()
        frag = plugin.plan(pg_spec, "slurm", env_config)

        assert frag.readiness.port == 5432
        assert frag.readiness.file is not None
        assert frag.readiness.file.name == "service.json"

    def test_build_handle_sets_store_locator(self, pg_spec, env_config, tmp_path):
        from metalab.services.postgres import PostgresPlugin

        pg_spec.config["file_root"] = str(tmp_path)
        env_config["file_root"] = str(tmp_path)

        plugin = PostgresPlugin()
        frag = plugin.plan(pg_spec, "slurm", env_config)
        handle = frag.build_handle("12345", "node01")

        assert handle.name == "postgres"
        assert handle.host == "node01"
        assert handle.port == 5432
        assert handle.process_id == "12345"
        assert "store_locator" in handle.metadata
        assert "postgresql://" in handle.metadata["store_locator"]
        assert "file_root=" in handle.metadata["store_locator"]


# ---------------------------------------------------------------------------
# Atlas plugin — plan_slurm
# ---------------------------------------------------------------------------


class TestAtlasSlurmProvider:
    """Test AtlasPlugin.plan_slurm returns a valid SlurmFragment."""

    @pytest.fixture
    def atlas_spec(self):
        return ServiceSpec(
            name="atlas",
            config={
                "port": 8000,
                "store": "postgresql://user:pw@host:5432/metalab",
                "file_root": "/shared/experiments",
            },
        )

    def test_returns_slurm_fragment(self, atlas_spec):
        from metalab.services.atlas import AtlasPlugin

        plugin = AtlasPlugin()
        frag = plugin.plan(atlas_spec, "slurm", {"file_root": "/shared"})

        assert isinstance(frag, SlurmFragment)
        assert frag.name == "atlas"
        assert frag.cpus == 1

    def test_setup_bash_has_uvicorn(self, atlas_spec):
        from metalab.services.atlas import AtlasPlugin

        plugin = AtlasPlugin()
        frag = plugin.plan(atlas_spec, "slurm", {})

        assert "uvicorn atlas.main:app" in frag.setup_bash
        assert "ATLAS_STORE_PATH" in frag.setup_bash
        assert "ATLAS_FILE_ROOT" in frag.setup_bash
        assert "--port 8000" in frag.setup_bash

    def test_cleanup_bash_kills_atlas(self, atlas_spec):
        """Atlas cleanup should kill the backgrounded uvicorn process."""
        from metalab.services.atlas import AtlasPlugin

        plugin = AtlasPlugin()
        frag = plugin.plan(atlas_spec, "slurm", {})
        assert "ATLAS_PID" in frag.cleanup_bash
        assert "kill" in frag.cleanup_bash

    def test_readiness_has_port(self, atlas_spec):
        from metalab.services.atlas import AtlasPlugin

        plugin = AtlasPlugin()
        frag = plugin.plan(atlas_spec, "slurm", {})
        assert frag.readiness.port == 8000

    def test_build_handle_sets_tunnel_target_with_gateway(self, atlas_spec):
        from metalab.services.atlas import AtlasPlugin

        plugin = AtlasPlugin()
        frag = plugin.plan(atlas_spec, "slurm", {"gateway": "login.cluster.edu"})
        handle = frag.build_handle("12345", "node01")

        assert handle.name == "atlas"
        assert "tunnel_target" in handle.metadata
        tt = handle.metadata["tunnel_target"]
        assert tt["host"] == "node01"
        assert tt["remote_port"] == 8000
        assert tt["local_port"] == 8000

    def test_build_handle_no_tunnel_without_gateway(self, atlas_spec):
        from metalab.services.atlas import AtlasPlugin

        plugin = AtlasPlugin()
        frag = plugin.plan(atlas_spec, "slurm", {})
        handle = frag.build_handle("12345", "node01")

        assert "tunnel_target" not in handle.metadata


# ---------------------------------------------------------------------------
# Unsupported platform raises NotImplementedError
# ---------------------------------------------------------------------------


class TestUnsupportedPlatform:
    """Plugins raise NotImplementedError for unknown platforms."""

    def test_postgres_unknown_platform(self):
        from metalab.services.postgres import PostgresPlugin

        plugin = PostgresPlugin()
        spec = ServiceSpec(name="postgres", config={})
        with pytest.raises(NotImplementedError, match="k8s"):
            plugin.plan(spec, "k8s", {})

    def test_atlas_unknown_platform(self):
        from metalab.services.atlas import AtlasPlugin

        plugin = AtlasPlugin()
        spec = ServiceSpec(name="atlas", config={})
        with pytest.raises(NotImplementedError, match="k8s"):
            plugin.plan(spec, "k8s", {})


# ---------------------------------------------------------------------------
# SlurmEnvironment._compose
# ---------------------------------------------------------------------------


class TestCompose:
    """Test the generic fragment composition into a single sbatch script."""

    @pytest.fixture
    def env(self):
        return SlurmEnvironment(
            file_root="/shared/experiments",
            user="researcher",
            services={"partition": "cpu2019", "time": "7-00:00:00", "memory": "10G"},
        )

    @pytest.fixture
    def pg_fragment(self):
        return SlurmFragment(
            name="postgres",
            setup_bash='echo "starting postgres"\npg_ctl -D "$PGDATA" start',
            cleanup_bash='pg_ctl -D "$PGDATA" stop -m fast 2>/dev/null || true',
            readiness=ReadinessCheck(port=5432),
            cpus=2,
            build_handle=lambda jid, host: ServiceHandle(
                name="postgres", host=host, port=5432, process_id=jid,
            ),
        )

    @pytest.fixture
    def atlas_fragment(self):
        return SlurmFragment(
            name="atlas",
            setup_bash='echo "starting atlas"\npython -m uvicorn atlas.main:app --host 0.0.0.0 --port 8000',
            cleanup_bash="",
            readiness=ReadinessCheck(port=8000),
            cpus=1,
            build_handle=lambda jid, host: ServiceHandle(
                name="atlas", host=host, port=8000, process_id=jid,
            ),
        )

    def test_job_name(self, env, pg_fragment, atlas_fragment, tmp_path):
        svc_dir = tmp_path / "services"
        svc_dir.mkdir()
        script = env._compose([pg_fragment, atlas_fragment], svc_dir)
        assert "#SBATCH --job-name=metalab-services" in script

    def test_cpus_summed(self, env, pg_fragment, atlas_fragment, tmp_path):
        svc_dir = tmp_path / "services"
        svc_dir.mkdir()
        script = env._compose([pg_fragment, atlas_fragment], svc_dir)
        assert "--cpus-per-task=3" in script

    def test_partition_from_config(self, tmp_path):
        env = SlurmEnvironment(
            file_root=str(tmp_path),
            services={"partition": "gpu", "time": "3-00:00:00", "memory": "16G"},
        )
        frag = SlurmFragment(
            name="test",
            setup_bash="echo test",
            cleanup_bash="",
            readiness=ReadinessCheck(),
            cpus=1,
        )
        svc_dir = tmp_path / "services"
        svc_dir.mkdir()
        script = env._compose([frag], svc_dir)
        assert "--partition=gpu" in script
        assert "--time=3-00:00:00" in script
        assert "--mem=16G" in script

    def test_cleanup_in_reverse_order(self, env, tmp_path):
        svc_dir = tmp_path / "services"
        svc_dir.mkdir()

        frag_a = SlurmFragment(
            name="svc_a",
            setup_bash="echo a",
            cleanup_bash="cleanup_a",
            readiness=ReadinessCheck(),
            cpus=1,
        )
        frag_b = SlurmFragment(
            name="svc_b",
            setup_bash="echo b",
            cleanup_bash="cleanup_b",
            readiness=ReadinessCheck(),
            cpus=1,
        )

        script = env._compose([frag_a, frag_b], svc_dir)

        # In the cleanup section, svc_b should come before svc_a (reverse order)
        cleanup_b_pos = script.index("cleanup_b")
        cleanup_a_pos = script.index("cleanup_a")
        assert cleanup_b_pos < cleanup_a_pos

    def test_setup_in_order(self, env, pg_fragment, atlas_fragment, tmp_path):
        svc_dir = tmp_path / "services"
        svc_dir.mkdir()
        script = env._compose([pg_fragment, atlas_fragment], svc_dir)

        pg_pos = script.index("starting postgres")
        atlas_pos = script.index("starting atlas")
        assert pg_pos < atlas_pos

    def test_trap_handler(self, env, pg_fragment, atlas_fragment, tmp_path):
        svc_dir = tmp_path / "services"
        svc_dir.mkdir()
        script = env._compose([pg_fragment, atlas_fragment], svc_dir)
        assert "trap cleanup EXIT SIGTERM" in script

    def test_single_fragment(self, env, pg_fragment, tmp_path):
        """Single fragment should still produce valid script."""
        svc_dir = tmp_path / "services"
        svc_dir.mkdir()
        script = env._compose([pg_fragment], svc_dir)
        assert "#SBATCH --job-name=metalab-services" in script
        assert "--cpus-per-task=2" in script


# ---------------------------------------------------------------------------
# Stop deduplication
# ---------------------------------------------------------------------------


class TestStopDeduplication:
    """Test that scancel is only called once for shared job IDs."""

    def test_stop_service_deduplicates(self):
        env = SlurmEnvironment(file_root="/tmp/test")

        pg_handle = ServiceHandle(
            name="postgres", host="node01", port=5432, process_id="12345"
        )
        atlas_handle = ServiceHandle(
            name="atlas", host="node01", port=8000, process_id="12345"
        )

        # Simulate first cancel
        env._cancelled_jobs.add("12345")

        # Second call should skip because job is already in the set
        env.stop_service(atlas_handle)
        assert atlas_handle.status == "stopped"
        assert "12345" in env._cancelled_jobs

    def test_different_job_ids_both_cancelled(self):
        env = SlurmEnvironment(file_root="/tmp/test")

        env._cancelled_jobs.add("111")
        env._cancelled_jobs.add("222")

        assert "111" in env._cancelled_jobs
        assert "222" in env._cancelled_jobs

    def test_stop_with_no_process_id(self):
        env = SlurmEnvironment(file_root="/tmp/test")

        handle = ServiceHandle(
            name="postgres", host="node01", port=5432, process_id=None
        )
        env.stop_service(handle)
        assert handle.status == "stopped"
        assert len(env._cancelled_jobs) == 0

    def test_cancelled_jobs_reset_between_instances(self):
        env1 = SlurmEnvironment(file_root="/tmp/test")
        env2 = SlurmEnvironment(file_root="/tmp/test")

        env1._cancelled_jobs.add("999")
        assert "999" not in env2._cancelled_jobs


# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------


class TestPluginRegistry:
    """Test the ServicePluginRegistry (entry-point based)."""

    def test_missing_plugin_returns_none(self):
        from metalab.services.registry import get_plugin

        assert get_plugin("nonexistent") is None

    def test_postgres_discovered(self):
        from metalab.services.registry import get_plugin

        plugin = get_plugin("postgres")
        assert plugin is not None
        assert plugin.name == "postgres"

    def test_atlas_discovered(self):
        from metalab.services.registry import get_plugin

        plugin = get_plugin("atlas")
        assert plugin is not None
        assert plugin.name == "atlas"

    def test_registered_plugins_includes_known(self):
        from metalab.services.registry import registered_plugins

        names = registered_plugins()
        assert "postgres" in names
        assert "atlas" in names

    def test_plugin_plan_dispatches(self):
        """Verify plugin.plan dispatches to the correct plan_{env} method."""
        from metalab.services.registry import get_plugin

        plugin = get_plugin("postgres")
        assert plugin is not None
        assert hasattr(plugin, "plan_slurm")
        assert hasattr(plugin, "plan_local")

    def test_plugin_discover_returns_none_for_unknown_env(self):
        """Discover with unknown env_type returns None (not error)."""
        from pathlib import Path

        from metalab.services.registry import get_plugin

        plugin = get_plugin("postgres")
        assert plugin is not None
        result = plugin.discover(Path("/tmp"), "k8s", {})
        assert result is None
