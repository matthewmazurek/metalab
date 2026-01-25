"""
Store module: Backend-agnostic persistence for records and artifacts.

Provides:
- Store: Protocol for storage backends
- FileStore: Filesystem-based storage with defined layout
"""

from metalab.store.base import Store
from metalab.store.file import FileStore

__all__ = [
    "Store",
    "FileStore",
]
