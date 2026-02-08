"""
Services module: Manage external services for metalab.

Provides service management for:
- PostgreSQL database for Postgres-first storage
- Atlas dashboard
- Service provider registry for backend-agnostic composition
"""

from metalab.services.postgres import (
    PostgresService,
    PostgresServiceConfig,
    build_store_locator,
    get_service_info,
    start_postgres_local,
    start_postgres_slurm,
    stop_postgres,
)
from metalab.services.registry import (
    ensure_providers_loaded,
    get_discover,
    get_provider,
    register_discover,
    register_provider,
    registered_providers,
)

# Import atlas to trigger auto-registration
import metalab.services.atlas  # noqa: F401

__all__ = [
    "PostgresService",
    "PostgresServiceConfig",
    "build_store_locator",
    "start_postgres_local",
    "start_postgres_slurm",
    "get_service_info",
    "stop_postgres",
    "register_provider",
    "register_discover",
    "get_provider",
    "get_discover",
    "ensure_providers_loaded",
    "registered_providers",
]
