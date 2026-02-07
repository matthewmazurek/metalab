"""
LocalEnvironment: Subprocess-based service management for local development.

Manages services via subprocess.Popen (or delegates to existing metalab
service managers). Services run on localhost with no tunneling required.
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from metalab.environment import EnvironmentRegistry
from metalab.environment.base import ServiceHandle, ServiceSpec

logger = logging.getLogger(__name__)


class LocalEnvironment:
    """Local subprocess-based service environment."""

    env_type = "local"

    def __init__(self, **config: Any) -> None:
        self.config = config

    def start_service(self, spec: ServiceSpec) -> ServiceHandle:
        """Start a service locally.

        For 'postgres': delegates to metalab.services.postgres.start_postgres_local()
        For 'atlas': spawns uvicorn serving the Atlas app
        """
        if spec.name == "postgres":
            return self._start_postgres(spec)
        elif spec.name == "atlas":
            return self._start_atlas(spec)
        else:
            raise ValueError(f"Unknown service: {spec.name!r}")

    def _start_postgres(self, spec: ServiceSpec) -> ServiceHandle:
        """Start local PostgreSQL using existing metalab service."""
        from metalab.services.postgres import (
            PostgresServiceConfig,
            start_postgres_local,
        )

        pg_config = PostgresServiceConfig(
            port=spec.config.get("port", 5432),
            database=spec.config.get("database", "metalab"),
            user=spec.config.get("user", os.environ.get("USER", "postgres")),
            password=spec.config.get("password"),
            auth_method=spec.config.get("auth_method", "trust"),
            listen_addresses=spec.config.get("listen_addresses", "localhost"),
        )

        service = start_postgres_local(pg_config)

        return ServiceHandle(
            name="postgres",
            host=service.host,
            port=service.port,
            credentials={
                "user": service.user,
                "password": service.password,
                "database": service.database,
            },
            process_id=str(service.pid) if service.pid else None,
            metadata={"connection_string": service.connection_string},
        )

    def _start_atlas(self, spec: ServiceSpec) -> ServiceHandle:
        """Start Atlas dashboard as a subprocess."""
        port = spec.config.get("port", 8000)
        store = spec.config.get("store", "")
        file_root = spec.config.get("file_root")
        host = spec.config.get("host", "127.0.0.1")

        cmd = [
            "python", "-m", "uvicorn", "atlas.main:app",
            "--host", host, "--port", str(port),
        ]

        env = os.environ.copy()
        env["ATLAS_STORE_PATH"] = store
        if file_root:
            env["ATLAS_FILE_ROOT"] = str(file_root)

        log_path = self._service_log_path("atlas", file_root)
        log_file = open(log_path, "a")  # noqa: SIM115
        logger.info(f"Atlas logs: {log_path}")

        proc = subprocess.Popen(
            cmd, env=env,
            stdout=log_file,
            stderr=log_file,
        )

        # Wait for atlas to be ready
        self._wait_for_port(host, port, timeout=30.0)

        return ServiceHandle(
            name="atlas",
            host=host,
            port=port,
            process_id=str(proc.pid),
            metadata={"log_file": str(log_path)},
        )

    def stop_service(self, handle: ServiceHandle) -> None:
        """Stop a service by sending SIGTERM then SIGKILL."""
        if handle.name == "postgres":
            from metalab.services.postgres import stop_postgres
            # Try to use existing stop logic
            try:
                stop_postgres(service_id="default")
                return
            except Exception:
                pass

        if handle.process_id:
            pid = int(handle.process_id)
            try:
                os.kill(pid, signal.SIGTERM)
                # Wait briefly for graceful shutdown
                for _ in range(10):
                    try:
                        os.kill(pid, 0)  # Check if still running
                        time.sleep(0.5)
                    except OSError:
                        return  # Process exited
                # Force kill
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass  # Process already gone

    def discover(self, store_root: Path, service_name: str) -> ServiceHandle | None:
        """Discover a running local service."""
        if service_name == "postgres":
            return self._discover_postgres()
        return None

    def _discover_postgres(self) -> ServiceHandle | None:
        """Try to find a running local postgres via service.json."""
        from metalab.services.postgres import get_service_info
        try:
            service = get_service_info()
            if service:
                return ServiceHandle(
                    name="postgres",
                    host=service.host,
                    port=service.port,
                    credentials={
                        "user": service.user,
                        "password": service.password,
                        "database": service.database,
                    },
                    process_id=str(service.pid) if service.pid else None,
                    metadata={"connection_string": service.connection_string},
                )
        except Exception:
            pass
        return None

    def is_available(self, handle: ServiceHandle) -> bool:
        """Check if a service is reachable via TCP."""
        return self._check_port(handle.host, handle.port)

    @staticmethod
    def _service_log_path(service_name: str, file_root: str | None) -> Path:
        """
        Determine the log file path for a service.

        If *file_root* is set the log goes alongside the data
        (``{file_root}/services/{service}.log``).  Otherwise it falls back to
        ``~/.metalab/services/{service}.log``.
        """
        if file_root:
            log_dir = Path(file_root) / "services"
        else:
            log_dir = Path.home() / ".metalab" / "services"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"{service_name}.log"

    @staticmethod
    def _check_port(host: str, port: int, timeout: float = 2.0) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (OSError, ConnectionRefusedError):
            return False

    @staticmethod
    def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return
            except (OSError, ConnectionRefusedError):
                time.sleep(1.0)
        logger.warning(f"Timeout waiting for {host}:{port}")


# Auto-register
EnvironmentRegistry.register("local", LocalEnvironment)
