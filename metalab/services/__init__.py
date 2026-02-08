"""
Services module: Manage external services for metalab.

Provides service management for:
- PostgreSQL database for Postgres-first storage
- Atlas dashboard
- Service plugin registry for backend-agnostic composition
"""

from metalab.services.postgres import (
    PostgresService,
    PostgresServiceConfig,
    build_store_locator,
    get_service_info,
    resolve_password,
    start_postgres_local,
    stop_postgres,
)
from metalab.services.registry import (
    get_plugin,
    registered_plugins,
)

__all__ = [
    "PostgresService",
    "PostgresServiceConfig",
    "build_store_locator",
    "start_postgres_local",
    "get_service_info",
    "resolve_password",
    "stop_postgres",
    "get_plugin",
    "registered_plugins",
]
