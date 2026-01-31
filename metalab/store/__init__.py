"""
Store module: Backend-agnostic persistence for records and artifacts.

Provides:
- Store: Protocol for storage backends
- FileStore: Filesystem-based storage with defined layout
- StoreFactory: Factory for creating stores from locator URIs
- create_store: Convenience function for creating stores
- parse_locator: Parse store locator URIs
- to_locator: Convert store instances to locator strings
- export_store, FallbackStore: Transfer and fallback utilities
"""

from metalab.store.base import Store
from metalab.store.file import FileStore
from metalab.store.locator import (
    LocatorInfo,
    StoreFactory,
    create_store,
    parse_locator,
    to_locator,
)
from metalab.store.transfer import (
    FallbackStore,
    export_store,
    export_to_filestore,
    import_from_filestore,
)

__all__ = [
    "Store",
    "FileStore",
    "LocatorInfo",
    "StoreFactory",
    "create_store",
    "parse_locator",
    "to_locator",
    "export_store",
    "export_to_filestore",
    "import_from_filestore",
    "FallbackStore",
]
