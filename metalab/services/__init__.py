"""
Services module: Manage external services for metalab.

Provides service management for:
- PostgreSQL database for Postgres-first storage
"""

from metalab.services.postgres import (
    PostgresService,
    PostgresServiceConfig,
    start_postgres_local,
    start_postgres_slurm,
    get_service_info,
    stop_postgres,
)

__all__ = [
    "PostgresService",
    "PostgresServiceConfig",
    "start_postgres_local",
    "start_postgres_slurm",
    "get_service_info",
    "stop_postgres",
]
