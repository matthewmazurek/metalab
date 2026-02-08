# metalab

**A lightweight, backend-agnostic framework for reproducible experiments.**

```
(ContextSpec, Params, SeedBundle) → RunRecord + Artifacts
```

Define your operation once. Sweep parameters, control randomness, capture results—with built-in resume, deduplication, and parallel execution.

---

## Key Features

<div class="grid cards" markdown>

- :material-file-tree: **Context Specs**  
  Lightweight manifests with lazy hashing for files and directories

- :material-grid: **Parameter Sources**  
  Grid, random, or manual parameter sweeps

- :material-dice-multiple: **Seed Discipline**  
  `SeedBundle` with replicates and derived RNGs for full reproducibility

- :material-download: **Capture System**  
  Metrics, structured data, artifacts, and logs

- :material-skip-next: **Resume + Dedupe**  
  Stable run IDs automatically skip completed work

- :material-server-network: **Parallel + SLURM**  
  Local thread/process executors and cluster job arrays

- :material-database: **Pluggable Stores**  
  Filesystem by default, PostgreSQL for query acceleration

</div>

---

## Quick Example

```python
import metalab

@metalab.operation
def train_model(params, seeds, capture):
    # Initialize model with reproducible random weights
    rng = seeds.numpy()
    model = init_model(rng)
    
    # Train with swept hyperparameters
    for epoch in range(params["epochs"]):
        loss = train_epoch(model, lr=params["learning_rate"])
        capture.metric("loss", loss, step=epoch)
    
    # Capture final results
    capture.metric("final_loss", loss)
    capture.artifact("weights", model.weights, kind="numpy")

exp = metalab.Experiment(
    name="hyperparameter_sweep",
    version="1.0",
    context={},
    operation=train_model,
    params=metalab.grid(
        learning_rate=[1e-4, 1e-3, 1e-2],
        epochs=[50, 100],
    ),
    seeds=metalab.seeds(base=42, replicates=3),
)

results = metalab.run(exp, store="./experiments").result()
print(results.to_dataframe())
```

This runs **18 experiments** (3 learning rates × 2 epoch counts × 3 replicates) with:

- **Deterministic run IDs** — same inputs always produce the same run ID
- **Automatic deduplication** — re-running skips completed work
- **Full reproducibility** — seeds ensure identical results across machines

---

## Installation

```bash
uv add git+https://github.com/matthewmazurek/metalab.git

# Optional extras
uv add "metalab[numpy] @ git+https://github.com/matthewmazurek/metalab.git"    # Array serialization
uv add "metalab[pandas] @ git+https://github.com/matthewmazurek/metalab.git"   # DataFrame export
uv add "metalab[rich] @ git+https://github.com/matthewmazurek/metalab.git"     # Rich progress bars
uv add "metalab[postgres] @ git+https://github.com/matthewmazurek/metalab.git" # PostgreSQL store
uv add "metalab[atlas] @ git+https://github.com/matthewmazurek/metalab.git"   # Atlas web UI
uv add "metalab[full] @ git+https://github.com/matthewmazurek/metalab.git"    # All of the above
```

---

## Next Steps

- [Key Concepts](key-concepts.md) — Context specs, params, seeds, and capture
- [Architecture](architecture.md) — How the internals fit together
- [Execution](execution.md) — Local and SLURM executors
- [Storage](storage.md) — FileStore and PostgreSQL backends
- [Remote Analysis](remote-analysis-workflow.md) — Working with remote experiment data
