"""
Hyperparameter Search Example: Random search, resume, and result analysis.

Usage:
    uv run python examples/hypersearch/run.py

This example demonstrates:
- Random parameter sampling: metalab.random() with loguniform, uniform distributions
- Resume capability: resume=True to continue interrupted experiments
- Result filtering: results.filter() to select runs by criteria
- Result display: results.display(group_by=) for grouped summaries
- Multiple experiments: Comparing grid vs random search in one script

Domain: Optimize the Booth function f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
        Optimal solution: x=1, y=3, f=0
"""

import numpy as np

import metalab


def booth_function(x, y):
    """Booth function - a classic optimization test function.

    Global minimum: f(1, 3) = 0
    Search domain: typically [-10, 10] for both x and y
    """
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def gradient_booth(x, y):
    """Gradient of Booth function."""
    df_dx = 2 * (x + 2 * y - 7) + 4 * (2 * x + y - 5)
    df_dy = 4 * (x + 2 * y - 7) + 2 * (2 * x + y - 5)
    return np.array([df_dx, df_dy])


@metalab.operation
def optimize_booth(params, seeds, capture):
    """Optimize Booth function using gradient descent with momentum."""
    lr = params["lr"]
    momentum = params["momentum"]
    n_iterations = 10_000

    # Initialize from random starting point
    rng = seeds.numpy()
    x = rng.uniform(-10, 10)
    y = rng.uniform(-10, 10)

    capture.log(
        f"Starting from ({x:.2f}, {y:.2f}) with lr={lr:.4f}, momentum={momentum:.2f}"
    )

    # Momentum buffer
    v = np.array([0.0, 0.0])

    # Track best solution found
    best_f = float("inf")
    best_x, best_y = x, y

    # Optimization loop
    for i in range(n_iterations):
        f = booth_function(x, y)

        # Track stepped metrics for visualization
        capture.metric("value", float(f), step=i)

        # Update best if improved
        if f < best_f:
            best_f = f
            best_x, best_y = x, y

        # Gradient descent with momentum
        grad = gradient_booth(x, y)
        v = momentum * v - lr * grad
        x, y = x + v[0], y + v[1]

    # Final metrics
    capture.metric("final_value", float(best_f))
    capture.metric("best_x", float(best_x))
    capture.metric("best_y", float(best_y))
    capture.metric("iterations", n_iterations)

    # Distance from optimal solution (1, 3)
    distance_to_opt = np.sqrt((best_x - 1) ** 2 + (best_y - 3) ** 2)
    capture.metric("distance_to_optimal", float(distance_to_opt))

    capture.log(f"Best found: ({best_x:.4f}, {best_y:.4f}) with f={best_f:.6f}")
    capture.log(f"Distance to optimal (1, 3): {distance_to_opt:.4f}")


# ============================================================================
# EXPERIMENT 1: Grid Search
# Traditional Cartesian product of parameter values
# ============================================================================
grid_exp = metalab.Experiment(
    name="hypersearch_grid",
    version="0.1",
    context={},
    operation=optimize_booth,
    # Grid search: 3 x 3 = 9 combinations
    params=metalab.grid(
        lr=[0.001, 0.01, 0.1],
        momentum=[0.0, 0.5, 0.9],
    ),
    seeds=metalab.seeds(base=42, replicates=3),
)


# ============================================================================
# EXPERIMENT 2: Random Search
# NEW FEATURE: metalab.random() with distribution functions
# ============================================================================
random_exp = metalab.Experiment(
    name="hypersearch_random",
    version="0.1",
    context={},
    operation=optimize_booth,
    # Random search: sample from continuous distributions
    # loguniform is great for learning rates (spans orders of magnitude)
    # uniform is good for bounded continuous parameters
    params=metalab.random(
        space={
            "lr": metalab.loguniform(1e-4, 1e-1),  # Log-uniform: 0.0001 to 0.1
            "momentum": metalab.uniform(0.0, 0.99),  # Uniform: 0 to 0.99
        },
        n_trials=27,  # Match grid search for fair comparison
        seed=123,  # Reproducible random sampling
    ),
    seeds=metalab.seeds(base=42, replicates=3),
)


def run_experiment(exp, name: str, resume: bool = False):
    """Run an experiment and return results."""
    print(f"\n{'='*60}")
    print(f"Running {name}")
    print(f"{'='*60}")

    # NEW FEATURE: resume=True skips already-completed runs
    # Useful for continuing after interruption or adding more trials
    handle = metalab.run(exp, resume=resume, progress=True)
    return handle.result()


def analyze_results(results, name: str):
    """Analyze and display results."""
    print(f"\n{'='*60}")
    print(f"Analysis: {name}")
    print(f"{'='*60}")

    # NEW FEATURE: results.filter() to select runs
    successful = results.successful
    print(f"Successful runs: {len(successful)} / {len(results)}")

    if not successful:
        print("No successful runs to analyze")
        return

    # Find best run
    best = min(successful, key=lambda r: r.metrics.get("final_value", float("inf")))
    print(f"\nBest run: {best.run_id[:12]}...")
    print(f"  final_value: {best.metrics['final_value']:.6f}")
    print(f"  best_x: {best.metrics['best_x']:.4f} (optimal: 1.0)")
    print(f"  best_y: {best.metrics['best_y']:.4f} (optimal: 3.0)")
    print(f"  distance_to_optimal: {best.metrics['distance_to_optimal']:.4f}")

    # Summary statistics
    final_values = [r.metrics["final_value"] for r in successful]
    print(f"\nSummary statistics:")
    print(f"  Mean final_value: {np.mean(final_values):.6f}")
    print(f"  Min final_value:  {np.min(final_values):.6f}")
    print(f"  Max final_value:  {np.max(final_values):.6f}")


if __name__ == "__main__":
    print("Hyperparameter Search Example")
    print("Comparing Grid Search vs Random Search on Booth function")
    print("Optimal solution: x=1, y=3, f(1,3)=0")

    # Run both experiments
    grid_results = run_experiment(
        grid_exp, "Grid Search (9 param combos x 3 seeds = 27 runs)"
    )
    random_results = run_experiment(
        random_exp, "Random Search (27 trials × 3 seeds = 81 runs)"
    )

    # Analyze results
    analyze_results(grid_results, "Grid Search")
    analyze_results(random_results, "Random Search")

    # ========================================================================
    # NEW FEATURE: results.display() with grouping
    # ========================================================================
    print("\n" + "=" * 60)
    print("Grouped Display (Grid Search by learning rate)")
    print("=" * 60)
    # Note: display() prints a formatted table grouped by the specified parameters
    # This shows aggregate statistics per group
    try:
        grid_results.display(group_by=["lr"])
    except Exception as e:
        print(f"  (display not available: {e})")

    # ========================================================================
    # NEW FEATURE: results.table(as_dataframe=True) for pandas export
    # ========================================================================
    print("\n" + "=" * 60)
    print("DataFrame Export")
    print("=" * 60)
    try:
        df = grid_results.table(as_dataframe=True)
        print(f"Grid results as DataFrame: {len(df)} rows x {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)[:8]}...")  # First 8 columns
    except ImportError:
        print("  (pandas not installed - install with: uv add pandas)")
    except Exception as e:
        print(f"  (DataFrame export failed: {e})")

    # ========================================================================
    # Demonstrate resume capability
    # ========================================================================
    print("\n" + "=" * 60)
    print("Resume Demonstration")
    print("=" * 60)
    print("Running grid experiment again with resume=True...")
    print("(Should skip all runs since they're already complete)")

    # This should be instant - all runs are skipped
    handle = metalab.run(grid_exp, resume=True, progress=True)
    resumed_results = handle.result()
    print(
        f"Result: {len(resumed_results.successful)} runs (all skipped, loaded from cache)"
    )

    # ========================================================================
    # Atlas Visualization Tips
    # ========================================================================
    print("\n" + "=" * 60)
    print("Atlas Visualization Tips")
    print("=" * 60)
    print("In metalab-atlas, try these visualizations:")
    print("  - Scatter: metrics.final_value vs params.lr (compare search strategies)")
    print("  - Compare: grid vs random search efficiency")
    print("  - Filter: by status='success' to exclude failures")
    print("  - Aggregate: mean final_value with error bars across replicates")
    print("  - Side-by-side: compare best runs from each search strategy")
