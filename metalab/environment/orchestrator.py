"""
ServiceOrchestrator: Composes environments, connectors, and bundles.

Provides the high-level operations that CLI commands delegate to:

- **up**: Provision services and optionally open tunnels
- **down**: Stop all services and clean up
- **status**: Check health of running services
- **tunnel**: Open a managed tunnel to running services

The orchestrator is config-driven: it reads :class:`ResolvedConfig` to
determine which services to provision.  Only starts postgres if
``[services.postgres]`` is configured.  Only starts atlas if
``[services.atlas]`` is configured (or a store exists).

Typical usage::

    from metalab.config import ProjectConfig
    from metalab.environment.orchestrator import ServiceOrchestrator

    config = ProjectConfig.load().resolve("slurm")
    orch = ServiceOrchestrator(config)
    bundle = orch.up(tunnel=True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metalab.config import ResolvedConfig
from metalab.environment import (
    EnvironmentRegistry,
    ServiceBundle,
    ServiceHandle,
    ServiceSpec,
    ConnectionTarget,
    TunnelHandle,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------


@dataclass
class ServiceStatus:
    """
    Status report for running services.

    Attributes:
        bundle_found: Whether a persisted service bundle was found on disk.
        services: Map of service name to status details including host, port,
            availability, and process ID.
    """

    bundle_found: bool
    services: dict[str, dict[str, Any]]  # name -> {status, host, port, ...}

    def is_healthy(self) -> bool:
        """Return ``True`` if all tracked services are available."""
        return self.bundle_found and all(
            s.get("available", False) for s in self.services.values()
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ServiceOrchestrator:
    """
    Orchestrates service lifecycle for an environment.

    Config-driven: reads the :class:`ResolvedConfig` to determine which
    services to provision.

    * Only starts **postgres** if ``[services.postgres]`` is configured.
    * Starts **atlas** if ``[services.atlas]`` is configured, or whenever a
      store locator or ``file_root`` exists (atlas needs a backing store to
      serve).

    Args:
        config: A fully-resolved project configuration for the target
            environment.
    """

    def __init__(self, config: ResolvedConfig) -> None:
        self.config = config
        self._bundle_path = self._get_bundle_path()

    # ------------------------------------------------------------------
    # Bundle path
    # ------------------------------------------------------------------

    def _get_bundle_path(self) -> Path:
        """
        Determine where to persist the service bundle.

        Bundles are runtime artifacts (PIDs, credentials, connection strings)
        and always live under ``~/.metalab/services/<project>/<env>/bundle.json``.
        This makes discovery trivial from any working directory — only the
        project name and environment are needed.

        Returns:
            Path to ``bundle.json``.
        """
        project = self.config.project.name or "default"
        return (
            Path.home()
            / ".metalab"
            / "services"
            / project
            / self.config.env_name
            / "bundle.json"
        )

    # ------------------------------------------------------------------
    # Public lifecycle operations
    # ------------------------------------------------------------------

    def up(self, *, tunnel: bool = False) -> ServiceBundle:
        """
        Provision services per config, optionally open a tunnel.

        Logic:

        1. Check for an existing bundle — if all services are alive, reuse it.
        2. Create a :class:`ServiceEnvironment` from the registry.
        3. Start **postgres** if configured.
        4. Start **atlas** (pointed at postgres or ``file_root``).
        5. Save the bundle.
        6. Optionally open a tunnel.

        Args:
            tunnel: If ``True``, open a managed tunnel after provisioning.

        Returns:
            The :class:`ServiceBundle` with handles for all running services.
        """
        # Check for existing bundle
        existing = self._load_existing_bundle()
        if existing:
            logger.info("Found existing service bundle, reusing")
            return existing

        # Import environments to trigger registration
        self._ensure_environments_registered()

        env = EnvironmentRegistry.create(self.config.env_type, self.config.env_config)
        bundle = ServiceBundle(
            environment=self.config.env_type,
            profile=self.config.env_name,
        )

        # Start postgres if configured
        if self.config.has_service("postgres"):
            pg_config = self.config.get_service("postgres")
            pg_config["file_root"] = self.config.file_root

            spec = ServiceSpec(
                name="postgres",
                config=pg_config,
                resources={
                    "partition": self.config.env_config.get("partition"),
                    "time": self.config.env_config.get("time"),
                    "memory": self.config.env_config.get("memory"),
                },
            )

            pg_handle = env.start_service(spec)
            bundle.add("postgres", pg_handle)

            # Build store locator from postgres handle
            store_locator = self._build_store_locator(pg_handle)
            bundle.store_locator = store_locator
            logger.info(f"PostgreSQL started: {pg_handle.host}:{pg_handle.port}")

        # Start atlas if configured (or always if we have a store)
        if (
            self.config.has_service("atlas")
            or bundle.store_locator
            or self.config.file_root
        ):
            atlas_config: dict[str, Any] = {}
            if self.config.has_service("atlas"):
                atlas_config = dict(self.config.get_service("atlas"))

            # Point atlas at the store
            atlas_config["store"] = (
                bundle.store_locator or self.config.file_root or ""
            )
            atlas_config["file_root"] = self.config.file_root

            # Co-locate with postgres if on SLURM
            pg_handle = bundle.get("postgres")
            if pg_handle and self.config.env_type == "slurm":
                atlas_config["nodelist"] = pg_handle.host

            spec = ServiceSpec(
                name="atlas",
                config=atlas_config,
                resources={
                    "partition": self.config.env_config.get("partition"),
                    "time": self.config.env_config.get("time"),
                    "memory": "1G",
                },
            )

            atlas_handle = env.start_service(spec)
            bundle.add("atlas", atlas_handle)

            # Set up tunnel target (only for remote environments)
            if self.config.env_type != "local":
                bundle.tunnel_targets.append(
                    {
                        "host": atlas_handle.host,
                        "remote_port": atlas_handle.port,
                        "local_port": atlas_config.get("port", 8000),
                    }
                )
            logger.info(f"Atlas started: {atlas_handle.host}:{atlas_handle.port}")

        bundle.save(self._bundle_path)
        logger.info(f"Bundle saved: {self._bundle_path}")

        if tunnel:
            self.tunnel()

        return bundle

    def down(self) -> None:
        """
        Stop all services and clean up the bundle file.

        Services are stopped in reverse order (atlas first, then postgres) so
        that dependents are torn down before the services they rely on.  If
        stopping a particular service fails, a warning is logged and teardown
        continues for the remaining services.
        """
        if not self._bundle_path.exists():
            logger.info("No service bundle found, nothing to stop")
            return

        bundle = ServiceBundle.load(self._bundle_path)

        self._ensure_environments_registered()
        env = EnvironmentRegistry.create(
            bundle.environment, self.config.env_config
        )

        # Stop in reverse order (atlas first, then postgres)
        for name in reversed(list(bundle.services)):
            handle = bundle.services[name]
            try:
                env.stop_service(handle)
                logger.info(f"Stopped {name}")
            except Exception as e:
                logger.warning(f"Failed to stop {name}: {e}")

        bundle.remove(self._bundle_path)
        logger.info("Bundle removed")

    def status(self) -> ServiceStatus:
        """
        Check health of all running services.

        Loads the persisted bundle and queries each service's availability
        through the environment backend.

        Returns:
            A :class:`ServiceStatus` with per-service health information.
        """
        if not self._bundle_path.exists():
            return ServiceStatus(bundle_found=False, services={})

        bundle = ServiceBundle.load(self._bundle_path)

        self._ensure_environments_registered()
        env = EnvironmentRegistry.create(
            bundle.environment, self.config.env_config
        )

        statuses: dict[str, dict[str, Any]] = {}
        for name, handle in bundle.services.items():
            available = env.is_available(handle)
            statuses[name] = {
                "host": handle.host,
                "port": handle.port,
                "status": "running" if available else "unreachable",
                "available": available,
                "process_id": handle.process_id,
            }

        return ServiceStatus(bundle_found=True, services=statuses)

    def logs(self, service_name: str | None = None, tail: int = 0) -> dict[str, str]:
        """
        Read log files for running services.

        Args:
            service_name: If given, return logs for this service only.
                Otherwise return logs for all services that have a log file.
            tail: If > 0, return only the last *tail* lines of each log.

        Returns:
            A mapping of service name to log content.  Services without a
            persisted log file are omitted.
        """
        if not self._bundle_path.exists():
            return {}

        bundle = ServiceBundle.load(self._bundle_path)

        results: dict[str, str] = {}
        for name, handle in bundle.services.items():
            if service_name and name != service_name:
                continue
            log_file = handle.metadata.get("log_file")
            if not log_file:
                continue
            path = Path(log_file)
            if not path.exists():
                results[name] = "(log file not found)"
                continue
            text = path.read_text()
            if tail > 0:
                lines = text.splitlines()
                text = "\n".join(lines[-tail:])
            results[name] = text

        return results

    def tunnel(self) -> TunnelHandle | None:
        """
        Open a managed tunnel to running services.

        Uses the first tunnel target from the persisted bundle.  For local
        environments a :class:`DirectConnector` is used (essentially a no-op);
        for remote environments an SSH tunnel is established.

        Returns:
            A :class:`TunnelHandle` for the active tunnel, or ``None`` if
            there are no tunnel targets (e.g., services are local).

        Raises:
            RuntimeError: If no service bundle exists on disk.
        """
        if not self._bundle_path.exists():
            raise RuntimeError(
                "No service bundle found. Run 'metalab services up' first."
            )

        bundle = ServiceBundle.load(self._bundle_path)

        if not bundle.tunnel_targets:
            logger.info("No tunnel targets in bundle (services may be local)")
            return None

        # Get the connector for this environment type
        connector = self._get_connector()

        target_info = bundle.tunnel_targets[0]
        target = ConnectionTarget(
            remote_host=target_info["host"],
            remote_port=target_info["remote_port"],
            local_port=target_info.get("local_port", target_info["remote_port"]),
            gateway=self.config.env_config.get("gateway"),
            user=self.config.env_config.get("user"),
            ssh_key=self.config.env_config.get("ssh_key"),
        )

        handle = connector.connect(target)
        logger.info(f"Tunnel established: {handle.local_url}")
        return handle

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_existing_bundle(self) -> ServiceBundle | None:
        """
        Load and validate an existing bundle.

        If a bundle file exists and all of its services are still reachable,
        it is returned for reuse.  Otherwise ``None`` is returned so that
        :meth:`up` can provision fresh services.

        Returns:
            The existing :class:`ServiceBundle` if still healthy, else ``None``.
        """
        if not self._bundle_path.exists():
            return None

        try:
            bundle = ServiceBundle.load(self._bundle_path)
        except Exception:
            return None

        # Check if services are still alive
        self._ensure_environments_registered()
        try:
            env = EnvironmentRegistry.create(
                bundle.environment, self.config.env_config
            )
            all_alive = all(
                env.is_available(h) for h in bundle.services.values()
            )
            if all_alive:
                return bundle
        except Exception:
            pass

        return None

    def _build_store_locator(self, pg_handle: ServiceHandle) -> str:
        """
        Build a PostgreSQL store locator URI from a service handle.

        The URI follows the standard ``postgresql://`` scheme.  If
        ``file_root`` is set in the config, it is appended as a query
        parameter so downstream code can locate file-backed artifacts.

        Args:
            pg_handle: Running postgres service handle with credentials.

        Returns:
            A ``postgresql://`` connection URI.
        """
        creds = pg_handle.credentials
        user = creds.get("user", "")
        password = creds.get("password", "")
        database = creds.get("database", "metalab")

        auth = f"{user}:{password}" if password else user
        locator = f"postgresql://{auth}@{pg_handle.host}:{pg_handle.port}/{database}"

        if self.config.file_root:
            from urllib.parse import urlencode

            locator += "?" + urlencode({"file_root": self.config.file_root})

        return locator

    def _get_connector(self):
        """
        Get the appropriate connector for this environment type.

        Local environments use :class:`DirectConnector` (a passthrough);
        all others use :class:`SSHTunnelConnector` for SSH ``-L`` tunneling.

        Returns:
            A :class:`Connector` implementation.
        """
        if self.config.env_type == "local":
            from metalab.environment.direct import DirectConnector

            return DirectConnector()
        else:
            from metalab.environment.ssh_tunnel import SSHTunnelConnector

            return SSHTunnelConnector()

    @staticmethod
    def _ensure_environments_registered() -> None:
        """
        Import environment modules to trigger auto-registration.

        Environment implementations register themselves with
        :class:`EnvironmentRegistry` at import time.  This method ensures
        those imports have occurred before the registry is queried.
        """
        try:
            import metalab.environment.local  # noqa: F401
        except ImportError:
            pass
        try:
            import metalab.environment.slurm  # noqa: F401
        except ImportError:
            pass
