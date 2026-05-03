# AGENTS.md

## Purpose
metalab is a general experiment runner: (ContextSpec, Params, SeedBundle) -> RunRecord + Artifacts.
Backends (execution + storage) are pluggable. Domain logic stays in user Operations.

## Non-goals
- No domain-specific assumptions (ML/bio/etc.)
- No opinionated storage policy: users decide what to capture
- No hard dependency on distributed frameworks

## Core invariants (do not break)
- ContextSpec is a lightweight, serializable manifest (paths, config, checksums).
- Operations receive the context spec directly and load data themselves.
- Params inputs are immutable; resolve/derive into new objects.
- All randomness must be controlled via SeedBundle.
- run_id is stable and derived from experiment + context + params + seed fingerprints.
- Artifacts are emitted via capture; do not return large objects from Operation.run.
- Executor boundary payloads must be serializable or reconstructable via manifests.
- Core orchestration must remain backend-agnostic.

## Dev environment (uv)
- Python: 3.11+
- Install: `uv sync`
- Test: `uv run pytest`

## Git workflow
- Branches: feat/*, fix/*, chore/*
- Commits: imperative, scoped when helpful
- PRs must include tests for new behavior and doc updates for API changes
- **Pushing/pulling**: Always use `required_permissions: ["all"]` for git push,
  pull, and fetch. The `gh` credential helper stores tokens in the macOS Keyring,
  which requires full system access (not just network).
- **Submodules**: `atlas/` is a git submodule. Commit and push inside the
  submodule first, then commit and push the parent repo.

## Contracts
- Operation.run(context, params, seeds, runtime, capture) -> RunRecord
- ParamSource.iter() -> ParamCase(params, case_id, tags?)
- Executor.submit(payload) -> RunHandle
- Store.put_run_record(record); Store.put_artifact(...) -> ArtifactDescriptor
- Capture.metric(s)/artifact/file/log

RunRecord required fields: run_id, experiment_id, status, context_fingerprint,
params_fingerprint, seed_fingerprint, timestamps, metrics, provenance, error?

## Testing (pytest)
- Unit: canonicalization, hashing, param generation
- Contract: seed discipline, context immutability, payload serialization round-trip
- Integration: end-to-end run with ThreadExecutor + FileStore, including resume/dedupe

No network. Avoid flaky timing assumptions. Use tmp_path.

## PostgreSQL & Atlas at scale (300k+ runs)

metalab experiments can produce hundreds of thousands of runs. The store and
Atlas layers have been designed for this scale. Follow these rules to keep it
that way.

### Architecture: FileStore + PostgresIndex

- **FileStore is the source of truth.** All data writes go to files first,
  then get indexed in Postgres. If Postgres is lost, `rebuild_index()` restores
  it from files.
- **PostgresIndex is the query acceleration layer.** It provides fast indexed
  lookups but is expendable and rebuildable.
- Never bypass this write order: files first, index second.

### Schema & migrations (idempotent DDL)

- There is no migration framework. All schema DDL in `_run_schema_ddl()` uses
  `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` so it is
  idempotent and safe to re-run.
- Bump `SCHEMA_VERSION` in `postgres_index.py` when adding tables, columns,
  or indexes. Document each version inline.
- New indexes must use `IF NOT EXISTS` and handle failure gracefully
  (like the `pg_trgm` block does).
- Never use destructive DDL (`DROP`, `ALTER ... DROP COLUMN`) in auto-migrate.

### Query discipline (every query must be bounded)

- **Every SELECT must have a LIMIT** or be provably bounded (e.g., primary key
  lookup, `COUNT(*)`, `GROUP BY` on an indexed low-cardinality column).
- Use server-side pagination (`LIMIT`/`OFFSET`) for list endpoints; enforce
  `MAX_PAGE_SIZE` (currently 1000) as a hard cap.
- Never load all runs into Python memory for processing. If aggregation is
  needed, push it to SQL (`GROUP BY`, window functions) or use streaming
  cursors.
- Prefer column projection over `SELECT *` or full `record_json` when the
  caller only needs summary fields. This avoids TOAST decompression of large
  JSONB blobs.
- New JSONB filter paths must be backed by a GIN index or use the existing
  `record_json` GIN index (containment `@>` queries).

### Bulk operations (COPY over executemany)

- Use `COPY`-to-temp-table + `INSERT...ON CONFLICT` for batch inserts
  (see `batch_index_records`). This is ~100x faster than `executemany` at
  scale.
- Keep the rebuild pipeline batch-oriented: scan → TRUNCATE → COPY → COPY →
  catalog rebuild. Never do per-record commits during bulk operations.

### Atlas backend API rules

- Every new endpoint that returns a list of runs must accept `limit` and
  `offset` parameters and enforce `MAX_PAGE_SIZE`.
- Export/download endpoints must use streaming (`StreamingResponse` with a
  server-side cursor) rather than loading all matching rows into memory.
- Aggregate endpoints must use SQL pushdown when Postgres is the backend;
  the in-memory fallback is only for non-Postgres stores and must still
  respect `MAX_PAGE_SIZE`.
- Use TTL caching (currently 60s) for frequently polled metadata endpoints
  (experiments list, field index, timeline, status counts).
- New search paths should use SQL-native queries (targeted `ILIKE`, trigram
  indexes, `field_catalog`) rather than Python-loop fallback.

### Atlas frontend rules

- Fetch only one page of data at a time (currently `PAGE_SIZE = 25` for
  run tables). Never request unbounded lists from the API.
- Use filter-based "select all" (store the `FilterSpec`, not an array of
  IDs) when operating on bulk selections.
- Use `placeholderData` / `keepPreviousData` from React Query to avoid
  loading flickers during re-fetches.
- Plot data must be capped with `max_points` (default 10k, max 50k) and
  use server-side random sampling (`TABLESAMPLE BERNOULLI`).
- Lazy-load heavy views (`React.lazy` + `Suspense`) and vary
  `refetchInterval` based on run state (fast when running, slow when idle).

### Connection pool tuning

- `PostgresStoreConfig` defaults: `pool_min_size=1`, `pool_max_size=2`.
  Atlas adapter uses `pool_max_size=5`. Tune based on expected concurrency.
- Always set `connect_timeout` to prevent hung connections from blocking
  the pool.

### Testing expectations

- Unit tests for postgres use mocked `ConnectionPool` (no real DB needed).
- Integration tests requiring a real Postgres use the
  `METALAB_TEST_POSTGRES_URL` env var and create unique per-test schemas.
- Rebuild round-trip tests must verify 100% fidelity: records, metrics
  precision, derived metrics, manifests, and field catalog.

## Adding plugins
- New ParamSource: metalab/params/, add unit tests
- New Executor: metalab/executor/, add integration test
- New Serializer: metalab/capture/serializers/, add round-trip test
- New Store: metalab/store/, add integration test (artifacts + run records)
