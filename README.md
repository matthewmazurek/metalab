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

New stores use a clean v2 layout:

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

Run JSON files are canonical. Event logs, heartbeats, status cache, and DuckDB
indexes are accelerators that can be regenerated.

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
