"""
Curve Fitting Example: Training loops, stepped metrics, and artifacts.

Usage:
    uv run python examples/curve_fit/run.py

This example demonstrates:
- Stepped metrics: capture.metric("loss", value, step=i) for training curves
- Artifacts: capture.artifact() for numpy arrays and JSON
- Logging: capture.log() for operation messages
- ThreadExecutor: explicit parallelism with max_workers
- Progress tracking: progress=True for live progress display

Domain: Fit exponential decay y = a * exp(-b * x) + c to noisy data via gradient descent.
"""

import numpy as np

import metalab
from metalab import ThreadExecutor


def generate_data(rng, n_points: int = 50, noise: float = 0.1):
    """Generate noisy exponential decay data."""
    # True parameters: a=2.0, b=0.5, c=0.5
    x = np.linspace(0, 5, n_points)
    y_true = 2.0 * np.exp(-0.5 * x) + 0.5
    y = y_true + rng.normal(0, noise, n_points)
    return x, y


def exponential_decay(x, params):
    """Compute y = a * exp(-b * x) + c."""
    a, b, c = params
    return a * np.exp(-b * x) + c


def compute_loss(x, y, params):
    """Mean squared error loss."""
    y_pred = exponential_decay(x, params)
    return np.mean((y - y_pred) ** 2)


def compute_gradients(x, y, params):
    """Compute gradients of MSE loss w.r.t. parameters [a, b, c]."""
    a, b, c = params
    y_pred = exponential_decay(x, params)
    residual = y_pred - y  # (n,)
    n = len(x)

    # Gradients: d(loss)/d(param) = 2/n * sum(residual * d(y_pred)/d(param))
    exp_term = np.exp(-b * x)
    grad_a = 2 / n * np.sum(residual * exp_term)
    grad_b = 2 / n * np.sum(residual * (-a * x * exp_term))
    grad_c = 2 / n * np.sum(residual)

    return np.array([grad_a, grad_b, grad_c])


# ============================================================================
# NEW FEATURE: @metalab.operation with logging, stepped metrics, and artifacts
# ============================================================================
@metalab.operation
def fit_curve(params, seeds, capture):
    """Fit exponential decay curve using gradient descent."""
    lr = params["learning_rate"]
    n_iters = params["n_iterations"]

    # Get reproducible RNG from seeds
    rng = seeds.numpy()

    # Generate noisy training data
    x_data, y_data = generate_data(rng, n_points=50, noise=0.1)

    # Initialize parameters randomly: [a, b, c]
    theta = rng.uniform(0.5, 3.0, size=3)

    # NEW FEATURE: capture.log() for operation logging
    capture.log(f"Starting optimization with lr={lr}, {n_iters} iterations")
    capture.log(f"Initial params: a={theta[0]:.3f}, b={theta[1]:.3f}, c={theta[2]:.3f}")

    # Track loss history for artifact
    loss_history = []
    convergence_threshold = 1e-6
    converged_at = None

    # Gradient descent loop
    for i in range(n_iters):
        loss = compute_loss(x_data, y_data, theta)
        loss_history.append(loss)

        # NEW FEATURE: capture.metric() with step= for time-series metrics
        # This creates a training curve viewable in atlas
        capture.metric("loss", float(loss), step=i)

        # Check for convergence
        if loss < convergence_threshold and converged_at is None:
            converged_at = i
            capture.log(f"Converged at iteration {i} with loss={loss:.2e}")

        # Gradient descent update
        grads = compute_gradients(x_data, y_data, theta)
        theta = theta - lr * grads

    # Final metrics (scalars)
    final_loss = compute_loss(x_data, y_data, theta)
    capture.metric("final_loss", float(final_loss))
    capture.metric("iterations_to_converge", converged_at if converged_at else n_iters)

    # NEW FEATURE: capture.artifact() for numpy arrays
    # These are saved and can be loaded later with run.artifact("name")
    capture.artifact("fitted_params", theta)  # numpy array [a, b, c]
    capture.artifact("loss_history", np.array(loss_history))  # 1D array for atlas viz

    # Log final results
    capture.log(f"Final params: a={theta[0]:.3f}, b={theta[1]:.3f}, c={theta[2]:.3f}")
    capture.log(f"Final loss: {final_loss:.6f}")


# Define the experiment
exp = metalab.Experiment(
    name="curve_fit",
    version="0.1",
    context={},
    operation=fit_curve,
    # Grid search over learning rates and iteration counts
    params=metalab.grid(
        learning_rate=[0.01, 0.05, 0.1],  # X-axis sweep in atlas
        n_iterations=[200, 500],  # Grouping dimension
    ),
    # 3 replicates for error bars in atlas
    seeds=metalab.seeds(base=42, replicates=3),
)

if __name__ == "__main__":
    print("Running curve_fit example...")
    print(f"Total runs: {len(list(exp.params)) * 3}")  # 6 param combos × 3 seeds

    # NEW FEATURE: ThreadExecutor with explicit max_workers
    # For CPU-bound work, ProcessExecutor is better; ThreadExecutor works for I/O or simple tasks
    executor = ThreadExecutor(max_workers=4)

    # NEW FEATURE: progress=True for live progress display
    handle = metalab.run(exp, executor=executor, progress=True)

    # Block until complete
    results = handle.result()

    # Display results table
    print("\n" + "=" * 60)
    print("Results Table")
    print("=" * 60)
    print(results.table(as_dataframe=True))

    # Show how to access artifacts
    print("\n" + "=" * 60)
    print("Accessing Artifacts")
    print("=" * 60)
    best_run = min(
        results.successful, key=lambda r: r.metrics.get("final_loss", float("inf"))
    )
    print(f"Best run: {best_run.run_id[:8]}...")
    print(f"  final_loss: {best_run.metrics['final_loss']:.6f}")

    # Load the fitted parameters artifact
    fitted_params = best_run.artifact("fitted_params")
    print(
        f"  fitted_params: a={fitted_params[0]:.3f}, b={fitted_params[1]:.3f}, c={fitted_params[2]:.3f}"
    )
    print(f"  (true params:  a=2.000, b=0.500, c=0.500)")

    # Show loss history artifact shape
    loss_history = best_run.artifact("loss_history")
    print(f"  loss_history: {len(loss_history)} values")

    print("\n" + "=" * 60)
    print("Atlas Visualization Tips")
    print("=" * 60)
    print("In metalab-atlas, try these visualizations:")
    print(
        "  - Plot metrics.final_loss vs params.learning_rate, group by params.n_iterations"
    )
    print("  - View loss_history artifact as a line chart (1D array)")
    print("  - Compare runs with different learning rates side-by-side")
