# Storage

metalab supports pluggable storage backends with filesystem as the default and PostgreSQL for query acceleration on large-scale experiments.

## Quick Start

```python
import metalab

# Simple path - creates FileStore at that location
handle = metalab.run(exp, store="./experiments")

# Results are stored under:
handle.store.get_working_directory()  # ./experiments/<safe_experiment_id>/
```

## Store Configuration

Stores are configured via `StoreConfig` dataclasses, which are serializable and can be passed around before connecting:

```python
from metalab.store import FileStoreConfig

# Create a config (doesn't connect yet)
config = FileStoreConfig(root="./experiments")

# Scope to an experiment (creates subdirectory)
scoped = config.scoped("my_exp:1.0")

# Connect to get a usable store
store = scoped.connect()
```

### Experiment Scoping

When you call `metalab.run()`, the store is automatically scoped to the experiment:

```python
# These are equivalent:
metalab.run(exp, store="./experiments")
metalab.run(exp, store=FileStoreConfig(root="./experiments"))

# Data ends up in: ./experiments/my_exp_1.0/
```

## FileStore (default)

FileStore is the source of truth for all data. It stores everything on the local filesystem in a well-organized layout.

```python
from metalab.store import FileStoreConfig

# Path string - this is the collection root
metalab.run(exp, store="./experiments")
# Data stored in: ./experiments/<safe_experiment_id>/

# file:// URI 
metalab.run(exp, store="file:///shared/experiments")

# Explicit config (equivalent)
config = FileStoreConfig(root="./experiments")
metalab.run(exp, store=config)
```

The path you provide is the **collection root**—a directory that can hold multiple experiments. The runner automatically scopes storage to your experiment's ID, creating a subdirectory like `my_exp_1.0/`.

### Layout

```
{store_root}/
├── runs/{run_id}.json           # Run records (versioned JSON)
├── derived/{run_id}.json        # Derived metrics
├── artifacts/{run_id}/          # Artifact files + manifest
├── logs/{run_id}_{name}.log     # Log files
├── results/{run_id}/{name}.json # Structured results
├── experiments/{exp_id}_{ts}.json # Experiment manifests
└── _meta.json                   # Store metadata
```

## PostgresStore (optional)

PostgresStore wraps a FileStore with a PostgreSQL query index for fast lookups and filtering. Files remain the source of truth—Postgres accelerates queries.

```python
from metalab.store import PostgresStoreConfig

# PostgresStore requires file_root for logs/artifacts
config = PostgresStoreConfig(
    connection_string="postgresql://user@localhost/metalab",
    file_root="/shared/experiments",
)
metalab.run(exp, store=config)

# Or via URI with file_root parameter
metalab.run(
    exp,
    store="postgresql://user@localhost/metalab?file_root=/shared/experiments",
)
```

Install support:

```bash
uv add metalab[postgres]
```

### Architecture

`PostgresStore = FileStore (source of truth) + PostgresIndex (query acceleration)`

| Component | Role | Responsibilities |
| --- | --- | --- |
| **FileStore** | Source of truth | <ul><li>Run records</li><li>Artifacts</li><li>Logs</li><li>Structured data</li></ul> |
| **PostgresIndex** | Query acceleration | <ul><li>Fast record lookups</li><li>Experiment filtering</li><li>Field catalog (Atlas)</li><li>Derived metrics index</li></ul> |

**Key principle**: All data writes go to FileStore first (permanent), then indexed in Postgres (ephemeral). If Postgres is lost, call `rebuild_index()` to restore from files.

### Index Rebuild

If your Postgres database is wiped or out of sync:

```python
from metalab.store import PostgresStoreConfig

config = PostgresStoreConfig(
    connection_string="postgresql://localhost/db",
    file_root="/path/to/files",
)
store = config.connect()
store.rebuild_index()  # Re-indexes all records from FileStore
```

### Working with Existing FileStores

```python
from metalab.store import FileStoreConfig, PostgresStore

# Wrap an existing FileStore with Postgres indexing
filestore = FileStoreConfig(root="/path/to/existing/store").connect()
pg_store = PostgresStore.from_filestore(
    "postgresql://localhost/db",
    filestore,
    rebuild=True,  # Index existing records
)

# Export back to standalone FileStore
exported = pg_store.to_filestore("/path/to/export")
```

## Browsing Collections

A **collection** is an unscoped config pointing to a root directory that may contain multiple experiments. You can discover and browse experiments programmatically:

```python
from metalab import load_results
from metalab.store import FileStoreConfig

# Create an unscoped config (collection root)
collection = FileStoreConfig(root="./experiments")

# List all experiments in the collection
experiments = collection.list_experiments()
# ['my_exp:1.0', 'my_exp:2.0', 'other_exp:1.0']

# Get config for a specific experiment
config = collection.for_experiment("my_exp:1.0")
results = load_results(config)
```

The `list_experiments()` method scans subdirectories for `_meta.json` files that contain experiment IDs. Only works on unscoped configs.

## Loading Results

Load results from a store for analysis:

```python
from metalab import load_results
from metalab.store import FileStoreConfig

# From path string
results = load_results("./experiments", experiment_id="my_exp:1.0")

# From config (using collection API)
collection = FileStoreConfig(root="./experiments")
results = load_results(collection.for_experiment("my_exp:1.0"))

# Convert to DataFrame
df = results.to_dataframe()
```

## Store Transfer

Copy data between stores:

```python
from metalab.store import export_store

# Export specific experiment
export_store(
    source="postgresql://localhost/db?file_root=/data",
    destination="./backup",
    experiment_id="my_exp:1.0",
)
```

Or via CLI:

```bash
metalab store export --from ./runs/my_exp --to ./backup
```

## Creating Stores Programmatically

Use `create_store()` for URI-based store creation, or configs for explicit control:

```python
from metalab.store import create_store, FileStoreConfig, parse_to_config

# URI-based (convenience)
store = create_store("./runs/my_exp")
store = create_store("file:///absolute/path")
store = create_store("postgresql://localhost/db", file_root="/path/to/files")

# Config-based (recommended for programmatic use)
config = FileStoreConfig(root="./runs/my_exp")
store = config.connect()

# Parse URI to config (useful for inspection/modification)
config = parse_to_config("postgresql://localhost/db?file_root=/data")
config = config.scoped("my_exp:1.0")  # Add experiment scoping
store = config.connect()
```

## Config Serialization

StoreConfig objects are serializable for use across process boundaries (e.g., distributed execution):

```python
from metalab.store import FileStoreConfig, StoreConfig

config = FileStoreConfig(root="./experiments", experiment_id="my_exp:1.0")

# Serialize to dict
d = config.to_dict()
# {'root': '/abs/path/to/experiments', 'experiment_id': 'my_exp:1.0', '_type': 'file'}

# Deserialize back
restored = StoreConfig.from_dict(d)
assert restored == config
```

## Custom Stores

Create custom store implementations by subclassing `StoreConfig`:

```python
from dataclasses import dataclass
from typing import ClassVar, Any

from metalab.store import StoreConfig
from metalab.store.locator import LocatorInfo

@dataclass(frozen=True, kw_only=True)
class MyCustomStoreConfig(StoreConfig):
    # scheme is used for URI parsing and auto-registration
    scheme: ClassVar[str] = "myscheme"
    
    # Your config fields
    endpoint: str
    bucket: str
    experiment_id: str | None = None
    
    def connect(self) -> "MyCustomStore":
        return MyCustomStore(self)
    
    @classmethod
    def from_locator(cls, info: LocatorInfo, **kwargs: Any) -> "MyCustomStoreConfig":
        # Parse from URI like "myscheme://endpoint/bucket"
        return cls(
            endpoint=info.host,
            bucket=info.path.lstrip("/"),
            experiment_id=kwargs.get("experiment_id"),
        )

class MyCustomStore:
    def __init__(self, config: MyCustomStoreConfig):
        self.config = config
        # Connect to your backend...
    
    # Implement Store protocol methods...

# Auto-registered! Now usable via create_store
store = create_store("myscheme://my-endpoint/my-bucket")
```

The `__init_subclass__` mechanism in `StoreConfig` automatically registers your config class with the `ConfigRegistry` when the module is imported.
