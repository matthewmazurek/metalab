# AGENTS.md

## Purpose
metalab is a filesystem-native experiment runner for HPC workflows:

`(ContextSpec, Params, SeedBundle) -> RunRecord + filesystem artifacts`.

The runner coordinates through shared files only. DuckDB is an optional embedded
sidecar for indexing/export after runs have written canonical records.

## Non-goals
- No domain-specific assumptions.
- No database, web dashboard, service orchestration, or tunnel management.
- No read compatibility for pre-v3 run-store layouts.
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

## Scale target
- Design for 300k+ runs on shared HPC filesystems.
- Avoid full run-record scans in interactive paths (`status`, `observe`, `load_results`).
- Use manifest metadata, event byte offsets, and sidecar index metadata as cheap freshness signals.
- Canonical JSON scans are acceptable for explicit rebuild/export steps, not routine status or API opens.
- Keep APIs batch-oriented and streaming where possible; do not materialize all runs unless the user asks for records/dataframes.

## Dev environment
- Python: 3.11+
- Install: `uv sync`
- Test: `uv run pytest`

## Git workflow
- Branches: feat/*, fix/*, chore/*
- Commits: imperative, scoped when helpful
- PRs must include tests for new behavior and docs for API changes

## Run-store layout
New stores use the v3 hash-sharded metadata layout only:

```text
manifest.json
runs/manifest.json
runs/shards/{shard_id}.ndjson
runs/shards/{shard_id}.idx
metadata/manifest.json
metadata/results/{shard_id}.ndjson
metadata/artifacts/{shard_id}.ndjson
metadata/logs/{shard_id}.ndjson
events/{job_id}/{worker_id}.ndjson
heartbeats/{job_id}/{worker_id}.json
artifacts/{prefix}/{run_id}/...
index/status-cache.json
index/shard-map.ndjson
index/metalab.duckdb
```

## Query/export discipline
- `metalab status` reads one-shot run-store status; `metalab observe` provides the live dashboard.
- Resume skips only canonical successful run records.
- `metalab index rebuild` creates the DuckDB sidecar from canonical records and events.
- `metalab.load_results(PATH)` should use the DuckDB sidecar under the hood when available and stay lazy for summaries/filtering/iteration.
- `metalab summary` and `metalab export` use DuckDB and should print the hpc extra hint if missing.

## Adding plugins
- New ParamSource: `metalab/params/`, add unit tests.
- New Executor: `metalab/executor/`, add integration tests.
- New Serializer: `metalab/capture/serializers/`, add round-trip tests.
- New Store behavior: keep filesystem as the source of truth and add integration tests.
