# Curve Fitting Example

This example demonstrates **stepped metrics**, **artifacts**, **logging**, and **explicit parallelism** by fitting an exponential decay curve to noisy data using gradient descent.

## Features Demonstrated

| Feature | Description |
|---------|-------------|
| `capture.metric(step=)` | Stepped metrics for training curves |
| `capture.artifact()` | Save numpy arrays and JSON artifacts |
| `capture.log()` | Operation logging |
| `ThreadExecutor` | Explicit thread-based parallelism |
| `metalab observe` | Live run monitoring |

## Running the Example

```bash
uv run python examples/curve_fit/run.py
```

## Code Highlights

### Stepped Metrics

Record values at each training iteration for time-series visualization:

```python
for i in range(n_iterations):
    loss = compute_loss(x_data, y_data, theta)
    capture.metric("loss", float(loss), step=i)  # Training curve
```

### Artifacts

Save numpy arrays and other data for later analysis:

```python
capture.artifact("fitted_params", theta)           # Numpy array [a, b, c]
capture.artifact("loss_history", np.array(history)) # 1D array for visualization
```

### Logging

Record operation progress and messages:

```python
capture.log(f"Starting optimization with lr={lr}")
capture.log(f"Converged at iteration {i} with loss={loss:.2e}")
```

### Explicit Parallelism

Use ThreadExecutor with configurable workers:

```python
executor = ThreadExecutor(max_workers=4)
handle = metalab.run(exp, executor=executor)
```

## CLI Analysis

With the sidecar index, try:

```bash
metalab index rebuild ./runs
metalab summary ./runs --group-by params.learning_rate --metric metrics.final_loss
metalab export ./runs --format csv --out curve_fit.csv
```

## Parameter Grid

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `learning_rate` | [0.01, 0.05, 0.1] | X-axis sweep |
| `n_iterations` | [5000, 10000] | Grouping dimension |
| replicates | 3 | Aggregate summaries |

**Total runs:** 18 (6 param combos × 3 seeds)
**Runtime:** ~5-10 seconds
