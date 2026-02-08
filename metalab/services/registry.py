"""
ServiceProviderRegistry: Maps (service_name, env_type) to provider functions.

Each service module registers provider functions at import time::

    from metalab.services.registry import register_provider

    def slurm_provider(spec, env_config):
        ...
        return SlurmFragment(...)

    register_provider("postgres", "slurm", slurm_provider)

Environments resolve providers at runtime::

    from metalab.services.registry import get_provider

    provider = get_provider("postgres", "slurm")
    fragment = provider(spec, env_config)

Services can also register discover functions so that environments can
reconnect to already-running services without hardcoding service names::

    from metalab.services.registry import register_discover

    def discover_postgres(store_root, env_config):
        ...
        return ServiceHandle(...) or None

    register_discover("postgres", "slurm", discover_postgres)
"""

from __future__ import annotations

from typing import Any, Callable

# (service_name, env_type) -> provider_fn(spec, env_config) -> Fragment
_providers: dict[tuple[str, str], Callable[..., Any]] = {}

# (service_name, env_type) -> discover_fn(store_root, env_config) -> ServiceHandle | None
_discover_providers: dict[tuple[str, str], Callable[..., Any]] = {}

_loaded = False


def register_provider(
    service_name: str, env_type: str, provider: Callable[..., Any]
) -> None:
    """Register a service provider for a given environment type.

    Args:
        service_name: Service identifier (e.g. ``"postgres"``).
        env_type: Environment type (e.g. ``"slurm"``, ``"local"``).
        provider: Callable that takes ``(ServiceSpec, env_config_dict)``
            and returns a platform-specific fragment.
    """
    _providers[(service_name, env_type)] = provider


def register_discover(
    service_name: str, env_type: str, discover_fn: Callable[..., Any]
) -> None:
    """Register a discover function for a service + environment.

    The discover function is called with ``(store_root, env_config)``
    and returns a :class:`ServiceHandle` if found, else ``None``.

    Args:
        service_name: Service identifier (e.g. ``"postgres"``).
        env_type: Environment type (e.g. ``"slurm"``, ``"local"``).
        discover_fn: Callable that takes ``(store_root, env_config)``
            and returns ``ServiceHandle | None``.
    """
    _discover_providers[(service_name, env_type)] = discover_fn


def get_provider(
    service_name: str, env_type: str
) -> Callable[..., Any] | None:
    """Look up a provider, auto-loading service modules if needed.

    Args:
        service_name: Service identifier.
        env_type: Environment type.

    Returns:
        The provider callable, or ``None`` if not registered.
    """
    ensure_providers_loaded()
    return _providers.get((service_name, env_type))


def get_discover(
    service_name: str, env_type: str
) -> Callable[..., Any] | None:
    """Look up a discover function, auto-loading service modules if needed.

    Args:
        service_name: Service identifier.
        env_type: Environment type.

    Returns:
        The discover callable, or ``None`` if not registered.
    """
    ensure_providers_loaded()
    return _discover_providers.get((service_name, env_type))


def ensure_providers_loaded() -> None:
    """Import service modules to trigger auto-registration.

    Called lazily on first :func:`get_provider` call.  Safe to call
    multiple times.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True

    # Import known service modules so they register their providers.
    # New services just need an import here.
    try:
        import metalab.services.postgres  # noqa: F401
    except ImportError:
        pass
    try:
        import metalab.services.atlas  # noqa: F401
    except ImportError:
        pass


def registered_providers() -> list[tuple[str, str]]:
    """List all registered (service_name, env_type) pairs."""
    ensure_providers_loaded()
    return list(_providers.keys())
