"""
Store module: Backend-agnostic persistence for records and artifacts.

Provides:

- Store: Protocol for storage backends
- StoreConfig: Abstract base for store configurations (serializable)
- ConfigRegistry: Registry mapping schemes to config classes
- FileStore: Filesystem-based storage (source of truth)
- FileStoreConfig: Configuration for FileStore
- PostgresStore: FileStore + Postgres query index
- PostgresStoreConfig: Configuration for PostgresStore
- PostgresIndex: Query index backed by PostgreSQL
- FileStoreLayout: Filesystem layout configuration
- create_store: Convenience function for creating stores from locators
- parse_to_config: Parse locator URIs to StoreConfig
- parse_locator: Low-level URI parsing
- export_store: Transfer data between stores

Capability protocols (for optional store features):

- SupportsWorkingDirectory: stores with local filesystem roots
- SupportsExperimentManifests: stores that persist experiment manifests
- SupportsArtifactOpen: stores that can open artifacts for reading
- SupportsLogPath: stores that provide log file paths for streaming
- SupportsStructuredResults: stores that support inline structured data
- SupportsLogListing: stores that can list and retrieve logs
"""

from metalab.store.base import Store
from metalab.store.capabilities import (
    SupportsArtifactOpen,
    SupportsExperimentManifests,
    SupportsLogListing,
    SupportsLogPath,
    SupportsStructuredResults,
    SupportsWorkingDirectory,
)
from metalab.store.config import ConfigRegistry, StoreConfig
from metalab.store.file import FileStore, FileStoreConfig
from metalab.store.layout import FileStoreLayout, safe_experiment_id
from metalab.store.locator import (
    DEFAULT_STORE_ROOT,
    LocatorInfo,
    create_store,
    parse_locator,
    parse_to_config,
)
from metalab.store.transfer import export_store

__all__ = [
    # Base protocol
    "Store",
    # Config classes
    "StoreConfig",
    "ConfigRegistry",
    "FileStoreConfig",
    "PostgresStoreConfig",
    "PostgresStore",
    # Capability protocols
    "SupportsWorkingDirectory",
    "SupportsExperimentManifests",
    "SupportsArtifactOpen",
    "SupportsLogPath",
    "SupportsStructuredResults",
    "SupportsLogListing",
    # Implementations
    "FileStore",
    "FileStoreLayout",
    # Locator utilities
    "DEFAULT_STORE_ROOT",
    "LocatorInfo",
    "create_store",
    "parse_locator",
    "parse_to_config",
    # Transfer utilities
    "export_store",
    # Utilities
    "safe_experiment_id",
]


# Optional imports (require psycopg)
def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "PostgresStore":
        from metalab.store.postgres import PostgresStore

        return PostgresStore
    if name == "PostgresStoreConfig":
        from metalab.store.postgres import PostgresStoreConfig

        return PostgresStoreConfig
    if name == "PostgresIndex":
        from metalab.store.postgres_index import PostgresIndex

        return PostgresIndex
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
