"""
PostgresPlugin: Class-based service plugin for PostgreSQL.

Consolidates SLURM/local providers and discovery into a single class
with shared configuration resolution, handle construction, and bash
template rendering.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metalab.services.base import ServicePlugin
from metalab.services.postgres._bash import PgBashParams, render_cleanup_bash, render_setup_bash
from metalab.services.postgres.config import (
    DEFAULT_DATABASE,
    DEFAULT_PORT,
    PostgresServiceConfig,
    build_store_locator,
)
from metalab.services.postgres.lifecycle import (
    get_service_info,
    start_postgres_local,
    stop_postgres,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ResolvedPgConfig:
    """Internal container for shared config values."""

    user: str
    password: str | None
    port: int
    database: str
    auth_method: str
    file_root: str
    data_dir: Path
    service_dir: Path
    service_file: Path


class PostgresPlugin(ServicePlugin):
    """Service plugin for PostgreSQL."""

    name = "postgres"

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_config(
        spec: Any,
        env_config: dict[str, Any],
    ) -> _ResolvedPgConfig:
        """Extract and normalise config from *spec* and *env_config*."""
        user = spec.config.get(
            "user", env_config.get("user", os.environ.get("USER", "postgres"))
        )
        auth_method = spec.config.get("auth_method", "scram-sha-256")
        password = spec.config.get("password")
        if auth_method == "scram-sha-256" and not password:
            password = secrets.token_urlsafe(16)

        port = spec.config.get("port", DEFAULT_PORT)
        database = spec.config.get("database", DEFAULT_DATABASE)

        file_root = spec.config.get("file_root") or env_config.get("file_root", "")
        store_root = Path(file_root) if file_root else Path("/tmp")

        service_dir = store_root / "services" / "postgres"
        service_dir.mkdir(parents=True, exist_ok=True)
        service_file = service_dir / "service.json"
        data_dir = spec.config.get("data_dir") or (service_dir / "data")
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        return _ResolvedPgConfig(
            user=user,
            password=password,
            port=port,
            database=database,
            auth_method=auth_method,
            file_root=file_root,
            data_dir=data_dir,
            service_dir=service_dir,
            service_file=service_file,
        )

    @staticmethod
    def _make_handle(
        *,
        hostname: str,
        port: int,
        user: str,
        password: str | None,
        database: str,
        conn_string: str,
        file_root: str,
        process_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Build a :class:`ServiceHandle` with standard postgres metadata."""
        from metalab.environment.base import ServiceHandle

        metadata: dict[str, Any] = {
            "connection_string": conn_string,
            "store_locator": build_store_locator(conn_string, file_root=file_root),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return ServiceHandle(
            name="postgres",
            host=hostname,
            port=port,
            credentials={
                "user": user,
                "password": password,
                "database": database,
            },
            process_id=process_id,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # plan_slurm
    # ------------------------------------------------------------------

    def plan_slurm(
        self,
        spec: Any,
        env_config: dict[str, Any],
    ) -> Any:
        """Return a :class:`SlurmFragment` for PostgreSQL."""
        from metalab.environment.base import ReadinessCheck
        from metalab.environment.slurm import SlurmFragment

        cfg = self._resolve_config(spec, env_config)

        params = PgBashParams(
            user=cfg.user,
            password=cfg.password,
            port=cfg.port,
            database=cfg.database,
            auth_method=cfg.auth_method,
            data_dir=cfg.data_dir,
            service_dir=cfg.service_dir,
            service_file=cfg.service_file,
        )

        setup_bash = render_setup_bash(params)
        cleanup_bash = render_cleanup_bash()

        # Capture for closure
        _cfg = cfg

        def _build_handle(job_id: str, hostname: str) -> Any:
            # Prefer live values from service.json if available
            creds_user = _cfg.user
            creds_password = _cfg.password
            creds_database = _cfg.database
            conn_string = ""
            pgdata = str(_cfg.data_dir)

            if _cfg.service_file.exists():
                try:
                    with open(_cfg.service_file) as f:
                        info = json.load(f)
                    creds_user = info.get("user", _cfg.user)
                    creds_password = info.get("password", _cfg.password)
                    creds_database = info.get("database", _cfg.database)
                    conn_string = info.get("connection_string", "")
                    pgdata = info.get("pgdata", pgdata)
                except Exception:
                    pass

            if not conn_string:
                auth_prefix = params.auth_prefix
                conn_string = (
                    f"postgresql://{auth_prefix}@{hostname}:{_cfg.port}/{_cfg.database}"
                )

            return PostgresPlugin._make_handle(
                hostname=hostname,
                port=_cfg.port,
                user=creds_user,
                password=creds_password,
                database=creds_database,
                conn_string=conn_string,
                file_root=_cfg.file_root,
                process_id=job_id,
                extra_metadata={
                    "pgdata": pgdata,
                    "log_file": str(_cfg.service_dir / "postgres.log"),
                },
            )

        return SlurmFragment(
            name="postgres",
            setup_bash=setup_bash,
            cleanup_bash=cleanup_bash,
            readiness=ReadinessCheck(port=cfg.port, file=cfg.service_file),
            cpus=2,
            build_handle=_build_handle,
        )

    # ------------------------------------------------------------------
    # plan_local
    # ------------------------------------------------------------------

    def plan_local(
        self,
        spec: Any,
        env_config: dict[str, Any],
    ) -> Any:
        """Return a :class:`LocalFragment` for PostgreSQL."""
        from metalab.environment.base import ReadinessCheck
        from metalab.environment.local import LocalFragment

        user = spec.config.get("user", os.environ.get("USER", "postgres"))
        pg_config = PostgresServiceConfig(
            port=spec.config.get("port", DEFAULT_PORT),
            database=spec.config.get("database", DEFAULT_DATABASE),
            user=user,
            password=spec.config.get("password"),
            auth_method=spec.config.get("auth_method", "trust"),
            listen_addresses=spec.config.get("listen_addresses", "localhost"),
        )

        file_root = spec.config.get("file_root") or env_config.get("file_root", "")
        service = start_postgres_local(pg_config)

        _service = service
        _file_root = file_root

        def _build_handle(pid: str, hostname: str) -> Any:
            conn_string = _service.connection_string
            return PostgresPlugin._make_handle(
                hostname=_service.host,
                port=_service.port,
                user=_service.user,
                password=_service.password,
                database=_service.database,
                conn_string=conn_string,
                file_root=_file_root,
                process_id=str(_service.pid) if _service.pid else None,
            )

        return LocalFragment(
            name="postgres",
            command=[],
            readiness=ReadinessCheck(port=service.port),
            stop_fn=lambda: stop_postgres(service_id="default"),
            build_handle=_build_handle,
        )

    # ------------------------------------------------------------------
    # discover_slurm
    # ------------------------------------------------------------------

    def discover_slurm(
        self,
        store_root: Path,
        env_config: dict[str, Any],
    ) -> Any:
        """Discover a running PostgreSQL service via shared filesystem."""
        try:
            service = get_service_info(store_root=store_root)
            if service:
                try:
                    with socket.create_connection(
                        (service.host, service.port), timeout=2.0
                    ):
                        pass
                except (OSError, ConnectionRefusedError):
                    return None

                return self._make_handle(
                    hostname=service.host,
                    port=service.port,
                    user=service.user,
                    password=service.password,
                    database=service.database,
                    conn_string=service.connection_string,
                    file_root=str(store_root),
                    process_id=service.slurm_job_id,
                )
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # discover_local
    # ------------------------------------------------------------------

    def discover_local(
        self,
        store_root: Path,
        env_config: dict[str, Any],
    ) -> Any:
        """Discover a running local PostgreSQL service."""
        try:
            service = get_service_info()
            if service:
                return self._make_handle(
                    hostname=service.host,
                    port=service.port,
                    user=service.user,
                    password=service.password,
                    database=service.database,
                    conn_string=service.connection_string,
                    file_root=str(store_root),
                    process_id=str(service.pid) if service.pid else None,
                )
        except Exception:
            pass
        return None
