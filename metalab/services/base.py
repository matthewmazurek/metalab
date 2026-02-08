"""
ServicePlugin: Base class for service plugins.

Each service (postgres, atlas, ...) provides a single plugin class that
dispatches to platform-specific methods (``plan_slurm``, ``plan_local``, etc.).

Environments call ``plugin.plan(spec, env_type, env_config)`` and
``plugin.discover(store_root, env_type, env_config)`` without needing to
know which service they are talking to.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metalab.environment.base import ServiceHandle, ServiceSpec


class ServicePlugin:
    """Base class for service plugins.

    Subclasses implement ``plan_{env_type}`` and optionally
    ``discover_{env_type}`` methods for each supported platform.

    Example::

        class PostgresPlugin(ServicePlugin):
            name = "postgres"

            def plan_slurm(self, spec, env_config):
                ...

            def plan_local(self, spec, env_config):
                ...

            def discover_slurm(self, store_root, env_config):
                ...
    """

    name: str = ""

    def plan(
        self,
        spec: ServiceSpec,
        env_type: str,
        env_config: dict[str, Any],
    ) -> Any:
        """Dispatch to ``plan_{env_type}`` and return a Fragment.

        Raises:
            NotImplementedError: If no ``plan_{env_type}`` method exists.
        """
        method = getattr(self, f"plan_{env_type}", None)
        if method is None:
            raise NotImplementedError(
                f"Service {self.name!r} has no provider for platform {env_type!r}"
            )
        return method(spec, env_config)

    def discover(
        self,
        store_root: Path,
        env_type: str,
        env_config: dict[str, Any],
    ) -> ServiceHandle | None:
        """Dispatch to ``discover_{env_type}`` or return ``None``.

        Returns ``None`` if no discover method exists for the given
        platform (i.e. the service does not support discovery there).
        """
        method = getattr(self, f"discover_{env_type}", None)
        if method is None:
            return None
        return method(store_root, env_config)
