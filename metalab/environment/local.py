"""
LocalEnvironment: Subprocess-based service management for local development.

Services provide :class:`LocalFragment` objects describing their command,
environment, and readiness check.  The environment spawns subprocesses
sequentially, waits for readiness, and returns handles.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from metalab.environment import EnvironmentRegistry
from metalab.environment.base import ReadinessCheck, ServiceHandle, ServiceSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fragment type
# ---------------------------------------------------------------------------


@dataclass
class LocalFragment:
    """A service's contribution to a local subprocess launch.

    Attributes:
        name: Service identifier.
        command: Command + args to run as a subprocess.
        env: Extra environment variables to set.
        readiness: How to check the service is ready.
        log_name: Base name for the log file (e.g. ``"atlas"``).
        stop_fn: Optional custom stop function (e.g. ``stop_postgres``).
            If ``None``, default SIGTERM/SIGKILL is used.
        build_handle: Called with ``(pid, hostname)`` after readiness.
    """

    name: str
    command: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    readiness: ReadinessCheck = field(default_factory=ReadinessCheck)
    log_name: str | None = None
    stop_fn: Callable[[], None] | None = None
    build_handle: Callable[[str, str], ServiceHandle] = field(
        default=lambda pid, host: ServiceHandle(  # type: ignore[arg-type]
            name="unknown", host=host, port=0, process_id=pid
        )
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class LocalEnvironment:
    """Local subprocess-based service environment."""

    env_type = "local"

    def __init__(self, **config: Any) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Service lifecycle
    # ------------------------------------------------------------------

    def start_service(self, spec: ServiceSpec) -> ServiceHandle:
        """Start a single service (convenience wrapper)."""
        return self.start_service_group([spec])[0]

    def start_service_group(
        self, specs: list[ServiceSpec]
    ) -> list[ServiceHandle]:
        """Start services sequentially as local subprocesses.

        Resolves a :class:`LocalFragment` for each spec via the
        provider registry, spawns each subprocess, waits for readiness,
        and returns handles.

        Handle metadata from earlier services is propagated into later
        specs via :attr:`ServiceSpec.consumes`: if a spec declares
        consumed keys, matching metadata values from already-started
        services are injected into that spec's ``config`` before planning.

        Args:
            specs: Ordered list of service specifications.

        Returns:
            List of :class:`ServiceHandle` in the same order.
        """
        from metalab.services.registry import get_plugin

        handles: list[ServiceHandle] = []
        resolved_metadata: dict[str, Any] = {}

        for spec in specs:
            # Inject consumed metadata from earlier services
            for key in spec.consumes:
                if key in resolved_metadata:
                    spec.config[key] = resolved_metadata[key]

            plugin = get_plugin(spec.name)
            if plugin is None:
                raise ValueError(
                    f"No service plugin registered for {spec.name!r}"
                )
            fragment: LocalFragment = plugin.plan(spec, self.env_type, self.config)
            handle = self._start_fragment(fragment)
            handles.append(handle)

            # Accumulate metadata for downstream services
            resolved_metadata.update(handle.metadata)

        return handles

    def _start_fragment(self, fragment: LocalFragment) -> ServiceHandle:
        """Spawn a subprocess for a single fragment and wait for readiness."""
        # If no command, the provider handled startup itself (e.g. postgres)
        # and build_handle was already called with the right info.
        if not fragment.command:
            # Provider pre-started the service; just build the handle.
            return fragment.build_handle("", "localhost")

        # Merge environment
        env = os.environ.copy()
        env.update(fragment.env)

        # Log file
        file_root = self.config.get("file_root")
        log_path = self._service_log_path(
            fragment.log_name or fragment.name, file_root
        )
        log_file = open(log_path, "a")  # noqa: SIM115
        logger.info(f"{fragment.name} logs: {log_path}")

        proc = subprocess.Popen(
            fragment.command,
            env=env,
            stdout=log_file,
            stderr=log_file,
        )

        # Wait for readiness
        host = "127.0.0.1"
        if fragment.readiness.port:
            self._wait_for_port(host, fragment.readiness.port, timeout=30.0)

        handle = fragment.build_handle(str(proc.pid), host)

        # Persist stop_fn and log_file in metadata for later use
        if fragment.stop_fn:
            handle.metadata["_stop_fn"] = fragment.stop_fn
        if log_path:
            handle.metadata["log_file"] = str(log_path)

        return handle

    def stop_service(self, handle: ServiceHandle) -> None:
        """Stop a service.

        Uses the provider's ``stop_fn`` if available (stored in handle
        metadata), otherwise sends SIGTERM then SIGKILL.
        """
        stop_fn = handle.metadata.get("_stop_fn")
        if callable(stop_fn):
            try:
                stop_fn()
                return
            except Exception:
                pass

        if handle.process_id:
            pid = int(handle.process_id)
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(10):
                    try:
                        os.kill(pid, 0)
                        time.sleep(0.5)
                    except OSError:
                        return
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    def discover(self, store_root: Path, service_name: str) -> ServiceHandle | None:
        """Discover a running local service via the plugin registry."""
        from metalab.services.registry import get_plugin

        plugin = get_plugin(service_name)
        if plugin is None:
            return None
        return plugin.discover(store_root, self.env_type, self.config)

    def is_available(self, handle: ServiceHandle) -> bool:
        """Check if a service is reachable via TCP."""
        return self._check_port(handle.host, handle.port)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _service_log_path(service_name: str, file_root: str | None) -> Path:
        """Determine the log file path for a service."""
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
