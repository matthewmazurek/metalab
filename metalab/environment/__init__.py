"""
Environment module: Pluggable service management layer.

Provides:
- ServiceEnvironment: Protocol for service backends (local, SLURM, k8s, …)
- Connector: Protocol for tunneling/connectivity
- ServiceBundle: Service state tracking and persistence
- EnvironmentRegistry: Auto-registration for environment plugins

Usage::

    from metalab.environment import EnvironmentRegistry, ServiceSpec

    # Create an environment by registered name
    env = EnvironmentRegistry.create("slurm", {"partition": "gpu"})

    # Start a service
    handle = env.start_service(ServiceSpec(name="postgres"))

    # Track in a bundle
    bundle = ServiceBundle(environment="slurm", profile="slurm")
    bundle.add("postgres", handle)
    bundle.save(store_root / "services" / "bundle.json")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metalab.environment.base import ServiceEnvironment


class EnvironmentRegistry:
    """
    Registry mapping environment type names to implementation classes.

    Mirrors the HandleRegistry pattern from ``metalab/executor/registry.py``.
    Environment implementations register themselves at import time::

        EnvironmentRegistry.register("slurm", SlurmEnvironment)

    Then callers can instantiate by name::

        env = EnvironmentRegistry.create("slurm", {"partition": "gpu"})
    """

    _envs: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, env_class: type) -> None:
        """
        Register an environment class for a type name.

        Args:
            name: The environment type string (e.g., "local", "slurm").
            env_class: The environment class to register.
        """
        cls._envs[name] = env_class

    @classmethod
    def get(cls, name: str) -> type | None:
        """
        Get the environment class for a type name.

        Args:
            name: The environment type string.

        Returns:
            The environment class, or None if not registered.
        """
        return cls._envs.get(name)

    @classmethod
    def create(cls, name: str, config: dict[str, Any]) -> ServiceEnvironment:
        """
        Create an environment instance by type name.

        Args:
            name: The registered environment type (e.g., "slurm").
            config: Configuration dict passed to the environment constructor.

        Returns:
            A ServiceEnvironment instance.

        Raises:
            ValueError: If the environment type is not registered.
        """
        env_class = cls._envs.get(name)
        if env_class is None:
            registered = ", ".join(cls._envs.keys()) or "(none)"
            raise ValueError(
                f"Unknown environment type: {name!r}. "
                f"Registered types: {registered}"
            )
        return env_class(**config)

    @classmethod
    def types(cls) -> list[str]:
        """
        List all registered environment types.

        Returns:
            List of registered environment type strings.
        """
        return list(cls._envs.keys())


# Re-export key types
from metalab.environment.base import (
    ReadinessCheck,  # noqa: E402
    ServiceEnvironment,  # noqa: E402
    ServiceHandle,
    ServiceSpec,
)
from metalab.environment.bundle import ServiceBundle  # noqa: E402
from metalab.environment.connector import (
    ConnectionTarget,  # noqa: E402
    Connector,
    TunnelHandle,
)

__all__ = [
    "EnvironmentRegistry",
    "ReadinessCheck",
    "ServiceEnvironment",
    "ServiceSpec",
    "ServiceHandle",
    "Connector",
    "ConnectionTarget",
    "TunnelHandle",
    "ServiceBundle",
]
