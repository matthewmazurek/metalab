"""
Environment protocols: Platform-agnostic service management interface.

The ServiceEnvironment abstraction supports:
- Local (subprocess-based services)
- SLURM/HPC (sbatch-submitted services)
- Future: Kubernetes, cloud providers

ServiceEnvironment implementations manage the lifecycle of infrastructure
services (databases, monitoring, etc.) that experiments depend on. Each
implementation handles platform-specific details (process management, job
submission, pod creation) behind a uniform protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class ServiceSpec:
    """
    Specification for a service to start.

    ServiceSpec is a platform-agnostic description of *what* to run.
    The ServiceEnvironment decides *how* to run it.

    Attributes:
        name: Service type identifier (e.g., "postgres", "atlas").
        config: Service-specific configuration (ports, passwords, etc.).
        resources: Resource requirements (memory, cpus, gpus, etc.).
    """

    name: str
    config: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceHandle:
    """
    Runtime information about a running service.

    ServiceHandle is returned by ServiceEnvironment.start_service() and
    contains everything needed to connect to and manage the service.

    Attributes:
        name: Service type identifier (matches ServiceSpec.name).
        host: Hostname or IP address where the service is reachable.
        port: Port number the service is listening on.
        status: Current status ("running", "stopped", "failed").
        credentials: Authentication info (user, password, database, etc.).
        process_id: Platform-specific process identifier
            (PID, SLURM job ID, Kubernetes pod name, etc.).
        metadata: Additional platform-specific information.
    """

    name: str
    host: str
    port: int
    status: str = "running"
    credentials: dict[str, Any] = field(default_factory=dict)
    process_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "status": self.status,
            "credentials": self.credentials,
            "process_id": self.process_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ServiceHandle:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            host=data["host"],
            port=data["port"],
            status=data.get("status", "running"),
            credentials=data.get("credentials", {}),
            process_id=data.get("process_id"),
            metadata=data.get("metadata", {}),
        )


@runtime_checkable
class ServiceEnvironment(Protocol):
    """
    Protocol for service management backends.

    Each implementation handles a specific platform:
    - LocalEnvironment: subprocess-based, Docker/native binaries
    - SlurmEnvironment: sbatch job submission on HPC clusters
    - Future: KubernetesEnvironment, CloudRunEnvironment, etc.

    Implementations should register themselves with EnvironmentRegistry
    at import time so they can be instantiated by name.
    """

    env_type: str  # "local", "slurm", etc.

    def start_service(self, spec: ServiceSpec) -> ServiceHandle:
        """
        Start a service according to the given specification.

        Args:
            spec: What service to start and how to configure it.

        Returns:
            A ServiceHandle with connection info for the running service.

        Raises:
            RuntimeError: If the service cannot be started.
        """
        ...

    def stop_service(self, handle: ServiceHandle) -> None:
        """
        Stop a running service.

        Args:
            handle: The service handle returned by start_service().
        """
        ...

    def discover(self, store_root: Path, service_name: str) -> ServiceHandle | None:
        """
        Discover an already-running service by inspecting known locations.

        This enables reconnection after process restarts without
        re-provisioning infrastructure.

        Args:
            store_root: Root directory of the store (for service discovery files).
            service_name: The service type to discover (e.g., "postgres").

        Returns:
            A ServiceHandle if found and reachable, None otherwise.
        """
        ...

    def is_available(self, handle: ServiceHandle) -> bool:
        """
        Check whether a service is still reachable.

        Args:
            handle: The service handle to check.

        Returns:
            True if the service is running and reachable.
        """
        ...
