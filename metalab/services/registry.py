"""
ServicePluginRegistry: Maps service_name to ServicePlugin instances.

Plugins are discovered via entry points::

    [project.entry-points."metalab.service_plugins"]
    postgres = "metalab.services.postgres:PostgresPlugin"
    atlas = "metalab.services.atlas:AtlasPlugin"

Environments resolve plugins at runtime::

    from metalab.services.registry import get_plugin

    plugin = get_plugin("postgres")
    fragment = plugin.plan(spec, "slurm", env_config)
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metalab.services.base import ServicePlugin

logger = logging.getLogger(__name__)

# service_name -> ServicePlugin instance
_plugins: dict[str, ServicePlugin] = {}
_loaded = False


def _ensure_loaded() -> None:
    """Load all ``metalab.service_plugins`` entry points (once)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for ep in entry_points(group="metalab.service_plugins"):
        try:
            cls = ep.load()
            _plugins[ep.name] = cls()
        except Exception:  # noqa: BLE001
            logger.debug(
                "Failed to load service plugin entry point %r",
                ep.name,
                exc_info=True,
            )


def get_plugin(service_name: str) -> ServicePlugin | None:
    """Look up a service plugin by name.

    Args:
        service_name: Service identifier (e.g. ``"postgres"``).

    Returns:
        The :class:`ServicePlugin` instance, or ``None`` if not registered.
    """
    _ensure_loaded()
    return _plugins.get(service_name)


def registered_plugins() -> list[str]:
    """List all registered service plugin names."""
    _ensure_loaded()
    return list(_plugins.keys())
