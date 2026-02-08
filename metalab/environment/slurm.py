"""
SlurmEnvironment: SLURM-based service management for HPC clusters.

Manages services by submitting SLURM batch jobs. Services run on compute
nodes and are discoverable via bundle.json on the shared filesystem.

Services provide :class:`SlurmFragment` objects that describe their bash
setup/cleanup scripts.  The environment composes fragments into a single
sbatch script, submits it, and waits for all readiness checks.
"""

from __future__ import annotations

import json
import logging
import os
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
class SlurmFragment:
    """A service's contribution to a combined SLURM sbatch script.

    Each service provider returns one of these.  The environment
    composes them into a single script: setup sections in order,
    cleanup sections in reverse, CPUs summed.

    Attributes:
        name: Service identifier.
        setup_bash: Bash fragment for init + start.  Must background
            itself (or be the last fragment, which runs in the
            foreground as the job keep-alive).
        cleanup_bash: Bash fragment for the trap handler, run on
            EXIT/SIGTERM.
        readiness: How the Python side waits for this service.
        cpus: CPU cores this service needs.
        build_handle: Called with ``(job_id, hostname)`` after all
            readiness checks pass.  Returns the ServiceHandle.
    """

    name: str
    setup_bash: str
    cleanup_bash: str
    readiness: ReadinessCheck
    cpus: int = 1
    build_handle: Callable[[str, str], ServiceHandle] = field(
        default=lambda jid, host: ServiceHandle(  # type: ignore[arg-type]
            name="unknown", host=host, port=0, process_id=jid
        )
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


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
        # Track cancelled job IDs within a teardown cycle
        self._cancelled_jobs: set[str] = set()

    # ------------------------------------------------------------------
    # Service lifecycle
    # ------------------------------------------------------------------

    def start_service(self, spec: ServiceSpec) -> ServiceHandle:
        """Start a single service (convenience wrapper)."""
        return self.start_service_group([spec])[0]

    def start_service_group(
        self, specs: list[ServiceSpec]
    ) -> list[ServiceHandle]:
        """Start services as a single SLURM job.

        Resolves a :class:`SlurmFragment` for each spec via the
        provider registry, composes them into one sbatch script,
        submits it, and waits for all readiness checks.

        Args:
            specs: Ordered list of service specifications.

        Returns:
            List of :class:`ServiceHandle` in the same order.
        """
        from metalab.services.registry import get_provider

        # Resolve fragments
        fragments: list[SlurmFragment] = []
        for spec in specs:
            provider = get_provider(spec.name, self.env_type)
            if provider is None:
                raise ValueError(
                    f"No SLURM provider registered for service {spec.name!r}"
                )
            fragment = provider(spec, self.config)
            fragments.append(fragment)

        # Compose script
        file_root = self.config.get("file_root", "")
        svc_dir = Path(file_root) / "services" if file_root else Path("/tmp")
        svc_dir.mkdir(parents=True, exist_ok=True)

        script = self._compose(fragments, svc_dir)

        # Write and submit
        script_path = svc_dir / "start_services.sh"
        script_path.write_text(script)
        os.chmod(script_path, 0o700)

        result = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")

        job_id = result.stdout.strip().split(";")[0]
        logger.info(f"Submitted services SLURM job {job_id}")

        # Wait for all readiness checks
        hostname = self._wait_for_all_ready(fragments, job_id)

        # Build handles
        return [f.build_handle(job_id, hostname) for f in fragments]

    def _compose(
        self, fragments: list[SlurmFragment], svc_dir: Path
    ) -> str:
        """Compose fragments into a single sbatch script."""
        total_cpus = sum(f.cpus for f in fragments)
        names = " + ".join(f.name for f in fragments)

        # Cleanup in reverse order
        cleanup_lines = "\n".join(
            f"    # {f.name}\n    {f.cleanup_bash}"
            for f in reversed(fragments)
            if f.cleanup_bash.strip()
        )

        # Setup sections in order
        setup_sections = "\n\n".join(
            f"# --- {f.name} ---\n{f.setup_bash}" for f in fragments
        )

        return f"""#!/bin/bash
#SBATCH --job-name=metalab-services
#SBATCH --partition={self.svc_partition}
#SBATCH --time={self.svc_time}
#SBATCH --mem={self.svc_memory}
#SBATCH --cpus-per-task={total_cpus}
#SBATCH --output={svc_dir}/slurm-%j.out
#SBATCH --error={svc_dir}/slurm-%j.err

# ===================================================================
# metalab services: {names}
# ===================================================================

HOSTNAME=$(hostname)
echo "Starting metalab services on $HOSTNAME (job $SLURM_JOB_ID)"

# -------------------------------------------------------------------
# Cleanup trap
# -------------------------------------------------------------------
cleanup() {{
    echo "Shutting down services..."
{cleanup_lines}
    echo "Services stopped."
}}
trap cleanup EXIT SIGTERM

# -------------------------------------------------------------------
# Services
# -------------------------------------------------------------------

{setup_sections}
"""

    def stop_service(self, handle: ServiceHandle) -> None:
        """Stop a SLURM service by cancelling its job.

        Deduplicates cancellations: if the job ID was already cancelled
        (e.g. because multiple service handles share a single job),
        the ``scancel`` is skipped.
        """
        if handle.process_id:
            if handle.process_id not in self._cancelled_jobs:
                try:
                    subprocess.run(
                        ["scancel", handle.process_id],
                        capture_output=True,
                        text=True,
                        timeout=10.0,
                    )
                    self._cancelled_jobs.add(handle.process_id)
                    logger.info(
                        f"Cancelled SLURM job {handle.process_id} ({handle.name})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to cancel job {handle.process_id}: {e}")
            else:
                logger.debug(
                    f"SLURM job {handle.process_id} already cancelled, "
                    f"skipping for {handle.name}"
                )

        handle.status = "stopped"

    def discover(self, store_root: Path, service_name: str) -> ServiceHandle | None:
        """Discover a running service via the provider registry."""
        from metalab.services.registry import get_discover

        discover_fn = get_discover(service_name, self.env_type)
        if discover_fn is None:
            return None
        return discover_fn(store_root, self.config)

    def is_available(self, handle: ServiceHandle) -> bool:
        return self._check_port(handle.host, handle.port)

    # ------------------------------------------------------------------
    # Waiting helpers
    # ------------------------------------------------------------------

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

    def _wait_for_all_ready(
        self,
        fragments: list[SlurmFragment],
        job_id: str,
        timeout: float = 120.0,
    ) -> str:
        """Wait for all fragments' readiness checks and return hostname.

        Handles two kinds of readiness:
        - ``file``: polls for a file on shared filesystem, then reads
          the hostname from a JSON ``host`` field.
        - ``port``: waits for a TCP port (requires hostname first).

        The hostname is discovered from the first file-based readiness
        check, or from ``squeue`` if all checks are port-based.
        """
        hostname: str | None = None
        deadline = time.monotonic() + timeout

        # Phase 1: resolve hostname (from file readiness or squeue)
        file_fragments = [
            f for f in fragments if f.readiness.file is not None
        ]
        port_fragments = [
            f for f in fragments if f.readiness.port is not None and f.readiness.file is None
        ]

        if file_fragments:
            # Wait for the first file-based fragment (typically postgres)
            frag = file_fragments[0]
            service_file = frag.readiness.file
            assert service_file is not None
            while time.monotonic() < deadline:
                if service_file.exists():
                    try:
                        with open(service_file) as fh:
                            info = json.load(fh)
                        host = info.get("host", "")
                        port = info.get("port")
                        if host:
                            # Verify reachable if port is known
                            if port:
                                try:
                                    with socket.create_connection(
                                        (host, port), timeout=2.0
                                    ):
                                        hostname = host
                                        break
                                except (OSError, ConnectionRefusedError):
                                    pass
                            else:
                                hostname = host
                                break
                    except (json.JSONDecodeError, OSError):
                        pass
                time.sleep(5.0)
                self._check_job_alive(job_id)

            if not hostname:
                raise RuntimeError(
                    f"Service {file_fragments[0].name} did not become ready "
                    f"within {timeout}s (job {job_id})"
                )
        else:
            # No file-based readiness; get hostname from squeue
            hostname = self._wait_for_job_host(job_id, timeout=timeout)

        # Phase 2: wait for remaining port-based readiness checks
        for frag in port_fragments:
            assert frag.readiness.port is not None
            remaining = max(0.0, deadline - time.monotonic())
            self._wait_for_port(hostname, frag.readiness.port, timeout=remaining)

        # Also wait for file fragments' ports if they have one
        for frag in file_fragments[1:]:
            if frag.readiness.port:
                remaining = max(0.0, deadline - time.monotonic())
                self._wait_for_port(hostname, frag.readiness.port, timeout=remaining)

        return hostname

    @staticmethod
    def _check_job_alive(job_id: str) -> None:
        """Raise if the SLURM job has exited."""
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h"],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(
                f"SLURM job {job_id} exited before services were ready"
            )


# Auto-register
EnvironmentRegistry.register("slurm", SlurmEnvironment)
