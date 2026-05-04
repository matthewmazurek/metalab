# AGENTS.md

## Purpose
metalab is a filesystem-native experiment runner for HPC workflows:

`(ContextSpec, Params, SeedBundle) -> RunRecord + filesystem artifacts`.

The runner coordinates through shared files only. DuckDB is an optional embedded
sidecar for indexing/export after runs have written canonical records.

## Non-goals
- No domain-specific assumptions.
- No database, web dashboard, service orchestration, or tunnel management.
- No migration compatibility for pre-v2 run-store layouts.
- No required distributed framework.

## Core invariants
- ContextSpec is a lightweight, serializable manifest.
- Operations receive the context spec directly and load data themselves.
- Params inputs are immutable; resolve/derive into new objects.
- All randomness must be controlled via SeedBundle.
- run_id is stable and derived from experiment + context + params + seed + code fingerprints.
- Run records are the canonical source of truth.
- Persistent events, heartbeats, status caches, and DuckDB indexes are rebuildable accelerators.
- Workers never write to DuckDB and never require network/database access.
- Core orchestration must remain backend-agnostic except for the built-in filesystem store.

## Dev environment
- Python: 3.11+
- Install: `uv sync`
- Test: `uv run pytest`

## Git workflow
- Branches: feat/*, fix/*, chore/*
- Commits: imperative, scoped when helpful
- PRs must include tests for new behavior and docs for API changes

## Run-store layout
New stores use the v2 layout only:

```text
manifest.json
runs/{prefix}/{run_id}.json
events/{job_id}/{worker_id}.ndjson
heartbeats/{job_id}/{worker_id}.json
logs/{prefix}/{run_id}.log
artifacts/{prefix}/{run_id}/...
index/status-cache.json
index/metalab.duckdb
```

## Query/export discipline
- `metalab status` reads one-shot run-store status; `metalab observe` provides the live dashboard.
- Resume skips only canonical successful run records.
- `metalab index rebuild` creates the DuckDB sidecar from canonical records and events.
- `metalab summary` and `metalab export` use DuckDB and should print the hpc extra hint if missing.

## Adding plugins
- New ParamSource: `metalab/params/`, add unit tests.
- New Executor: `metalab/executor/`, add integration tests.
- New Serializer: `metalab/capture/serializers/`, add round-trip tests.
- New Store behavior: keep filesystem as the source of truth and add integration tests.
