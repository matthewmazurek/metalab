# Storage

Stores persist run records and artifacts. metalab supports file-based storage for local development and PostgreSQL for production deployments.

## FileStore

::: metalab.FileStore

::: metalab.FileStoreConfig

## PostgresStore

For query-accelerated storage. Requires the `postgres` extra (`uv add metalab[postgres]`).

::: metalab.store.postgres.PostgresStoreConfig
