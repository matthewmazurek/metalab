"""
SlurmEnvironment: SLURM-based service management for HPC clusters.

Manages services by submitting SLURM batch jobs. Services run on compute
nodes and are discoverable via bundle.json on the shared filesystem.
"""

from __future__ import annotations

import logging
import os
import secrets
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from metalab.environment import EnvironmentRegistry
from metalab.environment.base import ServiceHandle, ServiceSpec

logger = logging.getLogger(__name__)


class SlurmEnvironment:
    """SLURM-based service environment for HPC clusters."""

    env_type = "slurm"

    def __init__(self, **config: Any) -> None:
        self.config = config
        self.gateway = config.get("gateway")
        self.user = config.get("user", os.environ.get("USER", ""))
        # Service node resources from [environments.slurm.services]
        svc_res = config.get("services", {})
        self.svc_partition = svc_res.get("partition", "default")
        self.svc_time = svc_res.get("time", "24:00:00")
        self.svc_memory = svc_res.get("memory", "4G")

    def start_service(self, spec: ServiceSpec) -> ServiceHandle:
        if spec.name == "postgres":
            return self._start_postgres(spec)
        elif spec.name == "atlas":
            return self._start_atlas(spec)
        else:
            raise ValueError(f"Unknown service: {spec.name!r}")

    def _start_postgres(self, spec: ServiceSpec) -> ServiceHandle:
        """Start PostgreSQL via SLURM job.

        Delegates to metalab.services.postgres.start_postgres_slurm().
        """
        from metalab.services.postgres import (
            PostgresServiceConfig,
            start_postgres_slurm,
        )

        # Determine store root from spec or config
        file_root = spec.config.get("file_root") or self.config.get("file_root")
        if not file_root:
            raise ValueError("file_root is required for SLURM PostgreSQL service")
        store_root = Path(file_root)

        auth_method = spec.config.get("auth_method", "scram-sha-256")
        password = spec.config.get("password")
        if auth_method == "scram-sha-256" and not password:
            password = secrets.token_urlsafe(16)

        pg_config = PostgresServiceConfig(
            port=spec.config.get("port", 5432),
            database=spec.config.get("database", "metalab"),
            user=spec.config.get("user", self.user),
            password=password,
            auth_method=auth_method,
            listen_addresses="*",  # Network access from other nodes
        )

        service = start_postgres_slurm(
            pg_config,
            store_root=store_root,
            slurm_partition=self.svc_partition,
            slurm_time=self.svc_time,
            slurm_memory=self.svc_memory,
        )

        return ServiceHandle(
            name="postgres",
            host=service.host,
            port=service.port,
            credentials={
                "user": service.user,
                "password": service.password,
                "database": service.database,
            },
            process_id=service.slurm_job_id,
            metadata={
                "connection_string": service.connection_string,
                "pgdata": service.pgdata,
            },
        )

    def _start_atlas(self, spec: ServiceSpec) -> ServiceHandle:
        """Start Atlas dashboard via SLURM job.

        Submits a lightweight SLURM job.  If ``pg_host`` is present in
        ``spec.config``, co-locates Atlas on the same node via ``--nodelist``.
        """
        port = spec.config.get("port", 8000)
        store = spec.config.get("store", "")
        file_root = spec.config.get("file_root") or self.config.get("file_root", "")
        nodelist = spec.config.get("pg_host")  # co-locate with PG

        script = f"""#!/bin/bash
#SBATCH --job-name=metalab-atlas
#SBATCH --partition={self.svc_partition}
#SBATCH --time={self.svc_time}
#SBATCH --mem={self.svc_memory}
#SBATCH --cpus-per-task=1
"""
        if nodelist:
            script += f"#SBATCH --nodelist={nodelist}\n"

        script += f"""
export ATLAS_STORE_PATH="{store}"
export ATLAS_FILE_ROOT="{file_root}"

echo "Starting Atlas on $(hostname):${port}"
python -m uvicorn atlas.main:app --host 0.0.0.0 --port {port}
"""
        # Write and submit
        import tempfile

        script_dir = (
            Path(file_root) / "services" / "atlas"
            if file_root
            else Path(tempfile.gettempdir())
        )
        script_dir.mkdir(parents=True, exist_ok=True)
        script_path = script_dir / "atlas_slurm.sh"
        script_path.write_text(script)

        result = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")

        job_id = result.stdout.strip().split(";")[0]
        logger.info(f"Submitted Atlas SLURM job {job_id}")

        # Wait for job to start and get hostname
        hostname = self._wait_for_job_host(job_id, timeout=120.0)

        # Wait for Atlas to be ready on that host
        self._wait_for_port(hostname, port, timeout=60.0)

        return ServiceHandle(
            name="atlas",
            host=hostname,
            port=port,
            process_id=job_id,
        )

    def stop_service(self, handle: ServiceHandle) -> None:
        """Stop a SLURM service by cancelling its job."""
        if handle.process_id:
            try:
                subprocess.run(
                    ["scancel", handle.process_id],
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                )
                logger.info(f"Cancelled SLURM job {handle.process_id} ({handle.name})")
            except Exception as e:
                logger.warning(f"Failed to cancel job {handle.process_id}: {e}")

        handle.status = "stopped"

    def discover(self, store_root: Path, service_name: str) -> ServiceHandle | None:
        """Discover a running service via service files on shared filesystem."""
        if service_name == "postgres":
            return self._discover_postgres(store_root)
        return None

    def _discover_postgres(self, store_root: Path) -> ServiceHandle | None:
        from metalab.services.postgres import get_service_info

        try:
            service = get_service_info(store_root=store_root)
            if service and self._check_port(service.host, service.port):
                return ServiceHandle(
                    name="postgres",
                    host=service.host,
                    port=service.port,
                    credentials={
                        "user": service.user,
                        "password": service.password,
                        "database": service.database,
                    },
                    process_id=service.slurm_job_id,
                    metadata={"connection_string": service.connection_string},
                )
        except Exception:
            pass
        return None

    def is_available(self, handle: ServiceHandle) -> bool:
        return self._check_port(handle.host, handle.port)

    @staticmethod
    def _check_port(host: str, port: int, timeout: float = 2.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (OSError, ConnectionRefusedError):
            return False

    @staticmethod
    def _wait_for_port(host: str, port: int, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return
            except (OSError, ConnectionRefusedError):
                time.sleep(2.0)
        logger.warning(f"Timeout waiting for {host}:{port}")

    @staticmethod
    def _wait_for_job_host(job_id: str, timeout: float = 120.0) -> str:
        """Wait for a SLURM job to start and return its hostname."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%N"],
                capture_output=True,
                text=True,
                timeout=10.0,
            )
            hostname = result.stdout.strip()
            if hostname and hostname != "(None)":
                return hostname
            time.sleep(5.0)
        raise RuntimeError(f"SLURM job {job_id} did not start within {timeout}s")


# Auto-register
EnvironmentRegistry.register("slurm", SlurmEnvironment)
