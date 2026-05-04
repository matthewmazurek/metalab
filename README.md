# metalab

`metalab` is a filesystem-native experiment runner for HPC workflows.

The core contract is:

```text
(ContextSpec, Params, SeedBundle) -> RunRecord + files
```

Workers coordinate only through a shared filesystem. There is no database
service, dashboard service, tunnel, or remote daemon.

## Install

```bash
uv add git+https://github.com/matthewmazurek/metalab.git

# Optional extras
uv add "metalab[numpy] @ git+https://github.com/matthewmazurek/metalab.git"
uv add "metalab[pandas] @ git+https://github.com/matthewmazurek/metalab.git"
uv add "metalab[rich] @ git+https://github.com/matthewmazurek/metalab.git"
uv add "metalab[hpc] @ git+https://github.com/matthewmazurek/metalab.git"
```

## Minimal Experiment

```python
import metalab


@metalab.operation
def train(params, seeds, capture):
    rng = seeds.rng()
    capture.metric("score", rng.random() * params["scale"])


experiment = metalab.Experiment(
    name="mvp",
    version="0.1",
    context={},
    operation=train,
    params=metalab.grid(scale=[0.1, 1.0, 10.0]),
    seeds=metalab.seeds(base=123, replicates=2),
)
```

Run it:

```bash
metalab run mvp.py --store ./runs --executor local --workers 4
metalab status ./runs
metalab index rebuild ./runs
metalab export ./runs --format csv --out results.csv
```

## Run Store Layout

New stores use a clean v3 hash-sharded metadata layout:

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

Run-record shards are canonical. Artifact payload files remain regular
filesystem artifacts. Event logs, heartbeats, status cache, shard maps, and
DuckDB indexes are accelerators that can be regenerated.

`metalab.load_results(PATH)` is the public Python entry point for completed
results. When DuckDB is installed, it opens/rebuilds the sidecar index under the
hood and keeps run records lazy for summaries, filters, and tables. Pass
`indexed=False` to force direct eager loading from canonical run shards.

## CLI

```bash
metalab run experiment.py --store /scratch/me/runs --executor slurm
metalab status /scratch/me/runs
metalab observe /scratch/me/runs
metalab index rebuild /scratch/me/runs
metalab summary /scratch/me/runs --group-by params.lr --metric metrics.score
metalab export /scratch/me/runs --format parquet --out results.parquet
```

Local runs execute in the CLI process and return when work is complete. SLURM
runs submit the array job, print the job id and store path, then exit; use
`metalab observe /scratch/me/runs` to follow progress.

Runs are resume-first: completed successful run records are not re-executed,
while missing, failed, stale, or malformed records are eligible to run again.
`skipped` events are displayed as `skip` in `metalab observe` and describe a
particular submission; they do not change the durable experiment status of an
already successful run.

Custom executors implement the plan-based contract described in
[`docs/executors.md`](docs/executors.md).

## Development

Python 3.11+ required.

```bash
uv sync
uv run pytest
```
