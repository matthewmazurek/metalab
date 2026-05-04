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

New stores use a clean v4 hash-sharded output layout:

```text
manifest.json
records/manifest.json
records/{shard_id}.ndjson
records/{shard_id}.idx
outputs/manifest.json
outputs/results/{shard_id}.ndjson
outputs/logs/{shard_id}.ndjson
outputs/artifacts/metadata/{shard_id}.ndjson
outputs/artifacts/files/{prefix}/{run_id}/...
.metalab/events/{job_id}/{worker_id}.ndjson
.metalab/heartbeats/{job_id}/{worker_id}.json
.metalab/index/status-cache.json
.metalab/index/shard-map.ndjson
.metalab/index/metalab.duckdb
.metalab/submissions.ndjson
```

Run-record shards are canonical. Artifact payload files remain regular
filesystem artifacts. Event logs, heartbeats, status cache, shard maps, and
DuckDB indexes are accelerators that can be regenerated.

`metalab.load_results(PATH)` is the public Python entry point for completed
results. By default, it opens the DuckDB sidecar only when the index is already
current, keeping summaries, filters, and iteration lazy without surprising
full-store rebuilds. Run `metalab index rebuild PATH`, pass `indexed=True`, or
pass `refresh_index=True` to build the sidecar. Pass `indexed=False` to force
direct eager loading from canonical run shards.

## CLI

```bash
metalab run experiment.py --store /scratch/me/runs --executor slurm
metalab status /scratch/me/runs
metalab observe /scratch/me/runs
metalab index rebuild /scratch/me/runs
metalab summary /scratch/me/runs --group-by params.lr --metric metrics.score
metalab export /scratch/me/runs --format parquet --out results.parquet
```

Projects can also use a config-aware target:

```bash
metalab run my_project.experiment:build --config configs/smoke.yaml
```

The config file may be JSON, TOML, or YAML. YAML requires the `config` extra.
MetaLab reserves the top-level `metalab` key for execution options and passes
the remaining application config to the target function:

```yaml
experiment_name: smoke

data:
  input_file: data/example.h5ad

metalab:
  store: experiments/{experiment_name}
  executor: local
  resume: true
  workers: 4
```

Only strings inside the `metalab` section support simple `{top_level_key}`
interpolation. CLI flags such as `--store`, `--executor`, `--workers`, and
`--no-resume` override config values.

Local runs execute in the CLI process and return when work is complete. SLURM
runs submit the array job, print the job id and store path, then exit; use
`metalab observe /scratch/me/runs` to follow progress.

In Python, local runs commonly use `metalab.run(...).result()` because the
handle owns in-process futures.  For SLURM-scale runs, prefer `observe` or
`metalab.reconnect(PATH).status` for monitoring and `metalab.load_results(PATH)`
for completed result queries.  `SlurmRunHandle.result(indexed="auto")` is
available when a script intentionally wants to block and then use the same
indexed result-loading policy.

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
