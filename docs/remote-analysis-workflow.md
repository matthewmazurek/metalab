# Remote Analysis Workflow

A guide for analyzing experiment runs that were performed remotely.

## Overview

1. **Rsync** the data from the remote machine
2. **Load** the data using metalab
3. **Write reducers** to extract scalar metrics from artifacts
4. **Export** to a DataFrame
5. **Analyze** with pandas, matplotlib, etc.

---

## 1. Rsync the Data

The `FileStore` layout is just files, so rsync works perfectly:

```bash
# Sync a single experiment
rsync -avz remote_host:~/runs/my_experiment ./local_runs/

# Sync an entire runs directory
rsync -avz remote_host:~/runs/ ./local_runs/

# Dry run first to see what will be transferred
rsync -avzn remote_host:~/runs/my_experiment ./local_runs/
```

### FileStore Directory Layout

```
{store_root}/
├── runs/
│   └── {run_id}.json           # RunRecord
├── derived/
│   └── {run_id}.json           # Derived metrics (computed post-hoc)
├── artifacts/
│   └── {run_id}/
│       ├── {name}.{ext}        # Artifact files
│       └── _manifest.json      # ArtifactDescriptor list
├── logs/
│   └── {run_id}_{name}.log     # Log files
├── experiments/
│   └── {exp_id}_{timestamp}.json
└── _meta.json                  # Store metadata
```

---

## 2. Load Data with metalab

```python
from metalab import load_results

# Load all runs from the synced directory
results = load_results("./local_runs/my_experiment")

# Quick sanity check
print(f"Loaded {len(results)} runs")
print(f"Successful: {len(results.successful)}")
print(f"Failed: {len(results.failed)}")

# Peek at one run
run = results[0]
print(f"Run ID: {run.run_id[:8]}")
print(f"Params: {run.params}")
print(f"Metrics: {run.metrics}")
print(f"Artifacts: {[a.name for a in run.artifacts()]}")
```

### Filtering Runs

```python
# Filter by status
successful_runs = results.successful
failed_runs = results.failed

# Iterate over runs
for run in results:
    if run.status.value == "SUCCESS":
        print(run.run_id, run.metrics)
```

---

## 3. Write Reducing Functions for Data and Artifacts

Reducers extract scalar metrics from data and artifacts. They take a `Run` and return `dict[str, Metric]`:

```python
from metalab.types import Metric

def final_loss(run) -> dict[str, Metric]:
    """Extract final loss from training history artifact."""
    history = run.artifact("loss_history")  # Deserializes automatically
    return {
        "final_loss": float(history[-1]),
        "min_loss": float(min(history)),
    }

def convergence_epoch(run) -> dict[str, Metric]:
    """Find epoch where loss dropped below threshold."""
    history = run.artifact("loss_history")
    threshold = 0.01
    for i, loss in enumerate(history):
        if loss < threshold:
            return {"convergence_epoch": i}
    return {"convergence_epoch": len(history)}  # Never converged

def best_result_stats(run) -> dict[str, Metric]:
    """Extract statistics from a results array artifact."""
    results = run.artifact("results")  # e.g., numpy array
    return {
        "best_value": float(results.max()),
        "mean_value": float(results.mean()),
        "std_value": float(results.std()),
    }
```

### Compute and Persist Derived Metrics

```python
# Compute and save to disk (stored in derived/{run_id}.json)
results.compute_derived([final_loss, convergence_epoch, best_result_stats])
```

This persists the derived metrics, so subsequent loads don't need to re-process artifacts.

---

## 4. Export to DataFrame

```python
import pandas as pd

# Export with all data
df = results.to_dataframe(
    include_params=True,       # Columns: param_{key}
    include_metrics=True,      # Columns: {key} (as captured)
    include_derived=True,      # Columns: derived_{key} (persisted)
    progress=True,
)

# Compute derived metrics on-the-fly (without persisting)
df = results.to_dataframe(
    derived_metrics=[final_loss],  # Compute these on-the-fly
)

# Only successful runs
df = results.successful.to_dataframe()

# Save to CSV
results.to_csv("./analysis/results.csv")
```

### Column Naming Convention

| Source | Column Prefix | Example |
|--------|---------------|---------|
| Record fields | (none) | `run_id`, `status`, `duration_ms` |
| Parameters | `param_` | `param_learning_rate`, `param_batch_size` |
| Metrics | (none) | `accuracy`, `loss` |
| Derived metrics | `derived_` | `derived_final_loss`, `derived_best_acc` |

---

## 5. Further Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Group by a parameter and aggregate
summary = df.groupby("param_learning_rate").agg({
    "derived_final_loss": ["mean", "std"],
    "derived_convergence_epoch": "mean",
})
print(summary)

# Quick scatter plot
df.plot.scatter(x="param_learning_rate", y="derived_final_loss", logx=True)
plt.xlabel("Learning Rate")
plt.ylabel("Final Loss")
plt.savefig("lr_vs_loss.png")

# Filter and analyze best runs
best_runs = df[df["derived_final_loss"] < 0.01]
print(best_runs[["run_id", "param_learning_rate", "param_batch_size"]])

# Pivot table
pivot = df.pivot_table(
    values="derived_final_loss",
    index="param_learning_rate",
    columns="param_optimizer",
    aggfunc="mean"
)
```

---

## Complete Example Script

```python
#!/usr/bin/env python
"""analyze_remote_runs.py - Post-hoc analysis of synced experiment data."""

from metalab import load_results
from metalab.types import Metric
import pandas as pd

# --- Load ---
results = load_results("./local_runs/my_experiment")
print(f"Loaded {len(results)} runs ({len(results.successful)} successful)")

# --- Define reducers ---
def final_metrics(run) -> dict[str, Metric]:
    history = run.artifact("metrics_history")
    return {
        "final_loss": float(history["loss"][-1]),
        "final_acc": float(history["accuracy"][-1]),
        "best_acc": float(max(history["accuracy"])),
    }

# --- Compute and persist derived metrics ---
results.compute_derived([final_metrics])

# --- Export ---
df = results.successful.to_dataframe(include_derived=True)
df.to_csv("./analysis/experiment_results.csv", index=False)

# --- Analyze ---
print("\nBest runs by accuracy:")
print(df.nlargest(5, "derived_best_acc")[["run_id", "param_model", "derived_best_acc"]])

print("\nMean accuracy by model type:")
print(df.groupby("param_model")["derived_best_acc"].agg(["mean", "std", "count"]))
```

---

## Tips

- **Persist derived metrics**: Use `compute_derived()` once, then subsequent loads are fast
- **Handle missing artifacts**: Wrap artifact access in try/except if some runs may lack certain artifacts
- **Use progress bars**: Pass `progress=True` to `to_dataframe()` for long operations
- **Incremental sync**: Use `rsync -avz --update` to only transfer new/changed files
