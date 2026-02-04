# Execution

metalab supports local parallel execution and SLURM clusters with stable, index-addressed tasking.

## Local Parallelism

```python
from metalab import ProcessExecutor

handle = metalab.run(exp, executor=ProcessExecutor(max_workers=4))
results = handle.result()
```

## SLURM

```python
handle = metalab.run(
    exp,
    store="/scratch/runs/my_exp",
    executor=metalab.SlurmExecutor(
        metalab.SlurmConfig(
            partition="gpu",
            time="2:00:00",
            cpus=4,
            memory="16G",
            gpus=1,
            max_concurrent=100,
        )
    ),
    progress=True,
)
results = handle.result()
```

!!! note
    - Uses index-addressed job arrays for scalable sweeps
    - Parameter sources support O(1) index access
    - Works efficiently for very large experiments

## Resume and Deduplication

Run IDs are derived from experiment, context, params, and seeds. Completed runs are skipped on resume.

```python
results = metalab.run(exp, resume=True).result()
```

## Reconnect

```python
handle = metalab.reconnect("/scratch/runs/my_exp", progress=True)
results = handle.result()
```
