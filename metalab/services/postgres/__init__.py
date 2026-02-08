"""
PostgreSQL service plugin and lifecycle utilities.

Re-exports the public API so that existing ``from metalab.services.postgres import ...``
import statements continue to work.
"""

from metalab.services.postgres.config import (
    DEFAULT_DATABASE,
    DEFAULT_LOCAL_SERVICE_DIR,
    DEFAULT_PORT,
    PostgresService,
    PostgresServiceConfig,
    build_store_locator,
)
from metalab.services.postgres.lifecycle import (
    get_service_info,
    start_postgres_local,
    start_postgres_slurm,
    stop_postgres,
)
from metalab.services.postgres.plugin import PostgresPlugin

__all__ = [
    "DEFAULT_DATABASE",
    "DEFAULT_LOCAL_SERVICE_DIR",
    "DEFAULT_PORT",
    "PostgresPlugin",
    "PostgresService",
    "PostgresServiceConfig",
    "build_store_locator",
    "get_service_info",
    "start_postgres_local",
    "start_postgres_slurm",
    "stop_postgres",
]
