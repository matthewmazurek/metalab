"""
ServiceProviderRegistry: Maps (service_name, env_type) to provider functions.

Service providers and discover functions are discovered via entry points:

- ``metalab.services``  -- provider callables  (entry name: ``{service}_{env}``)
- ``metalab.service_discover`` -- discover callables (same naming convention)

Environments resolve providers at runtime::

    from metalab.services.registry import get_provider

    provider = get_provider("postgres", "slurm")
    fragment = provider(spec, env_config)
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any, Callable

logger = logging.getLogger(__name__)

# (service_name, env_type) -> provider_fn(spec, env_config) -> Fragment
_providers: dict[tuple[str, str], Callable[..., Any]] = {}

# (service_name, env_type) -> discover_fn(store_root, env_config) -> ServiceHandle | None
_discover_providers: dict[tuple[str, str], Callable[..., Any]] = {}

_providers_loaded = False
_discover_loaded = False


def _parse_key(name: str) -> tuple[str, str]:
    """Parse an entry-point name like ``postgres_slurm`` into ``("postgres", "slurm")``."""
    service, _, env = name.rpartition("_")
    return (service, env)


def _ensure_providers_loaded() -> None:
    """Load all ``metalab.services`` entry points (once)."""
    global _providers_loaded
    if _providers_loaded:
        return
    _providers_loaded = True
    for ep in entry_points(group="metalab.services"):
        try:
            _providers[_parse_key(ep.name)] = ep.load()
        except Exception:  # noqa: BLE001
            logger.debug("Failed to load service entry point %r", ep.name, exc_info=True)


def _ensure_discover_loaded() -> None:
    """Load all ``metalab.service_discover`` entry points (once)."""
    global _discover_loaded
    if _discover_loaded:
        return
    _discover_loaded = True
    for ep in entry_points(group="metalab.service_discover"):
        try:
            _discover_providers[_parse_key(ep.name)] = ep.load()
        except Exception:  # noqa: BLE001
            logger.debug("Failed to load discover entry point %r", ep.name, exc_info=True)


def get_provider(
    service_name: str, env_type: str
) -> Callable[..., Any] | None:
    """Look up a service provider.

    Args:
        service_name: Service identifier (e.g. ``"postgres"``).
        env_type: Environment type (e.g. ``"slurm"``, ``"local"``).

    Returns:
        The provider callable, or ``None`` if not registered.
    """
    _ensure_providers_loaded()
    return _providers.get((service_name, env_type))


def get_discover(
    service_name: str, env_type: str
) -> Callable[..., Any] | None:
    """Look up a discover function for a service + environment.

    Args:
        service_name: Service identifier.
        env_type: Environment type.

    Returns:
        The discover callable, or ``None`` if not registered.
    """
    _ensure_discover_loaded()
    return _discover_providers.get((service_name, env_type))


def registered_providers() -> list[tuple[str, str]]:
    """List all registered (service_name, env_type) pairs."""
    _ensure_providers_loaded()
    return list(_providers.keys())
