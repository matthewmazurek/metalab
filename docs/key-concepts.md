# Key Concepts

metalab is built around a small set of invariants that make experiments reproducible and backend-agnostic.

## Context Specs

Context specs are lightweight, serializable manifests. Operations receive the spec directly and load data themselves.

```python
@metalab.context_spec
class DataContext:
    dataset: metalab.FilePath
    vocab_size: int = 10000
```

Key points:
- Use `metalab.FilePath` and `metalab.DirPath` for lazy hashing
- The context is fingerprinted and contributes to `run_id`
- No large data objects are passed across executor boundaries

## Parameter Sources

Use `grid`, `random`, or `manual` to define parameter cases.

```python
params = metalab.grid(lr=[1e-3, 1e-2], batch=[32, 64])
params = metalab.random(space={"dropout": metalab.uniform(0.1, 0.5)}, n_trials=20, seed=123)
params = metalab.manual([{"lr": 0.01}, {"lr": 0.001}])
```

## Seed Discipline

All randomness is controlled by `SeedBundle`.

```python
seeds = metalab.seeds(base=42, replicates=5)

@metalab.operation
def op(seeds, capture):
    rng = seeds.numpy()
    model_seed = seeds.derive("model")
    capture.metric("sample", rng.random())
```

## Capture System

Capture scalars, structured data, artifacts, and logs from your operation.

```python
capture.metric("accuracy", 0.95)
capture.data("transition_matrix", matrix)
capture.artifact("model_weights", weights, kind="numpy")
capture.log("Training completed")
```

Structured data is optimized for derived metrics when using Postgres-backed stores.

## Derived Metrics

Derived metrics compute post-hoc results from stored runs. They do not affect fingerprints.

```python
def final_score(run):
    return {"final_score": float(run.metrics["score"])}
```
