"""
Minimal Working Example: Monte Carlo π estimation.

Usage:
    uv run python examples/pi_mc/run.py

This demonstrates the core metalab workflow:
- Define an operation with @metalab.operation
- Create an experiment with parameter sweep and replicates
- Run and display results
"""

import metalab


# The @operation decorator marks this function as your experiment logic.
# The operation name defaults to the function name (can override with name="custom").
# metalab injects only the arguments you request: params, seeds, capture
# (you can also request 'context' and 'runtime' if needed).
@metalab.operation
def estimate_pi(params, seeds, capture):
    # params: dict of parameter values for this run (from your param sweep)
    n = params["n_samples"]

    # seeds: use this for ALL randomness to ensure reproducibility
    # Never use np.random directly—always derive RNGs from seeds
    rng = seeds.numpy()  # Returns a numpy.random.Generator

    # Your experiment logic
    x, y = rng.random(n), rng.random(n)
    pi_est = 4.0 * (x**2 + y**2 <= 1).mean()

    # capture: record metrics (scalars) and artifacts (arrays, files, etc.)
    # These are automatically saved and appear in your results table
    capture.metric("pi_estimate", pi_est)


# Define the experiment configuration
exp = metalab.Experiment(
    name="pi_mc",
    version="0.1",
    # context: shared data/config across all runs (empty here, but could be
    # file paths, dataset configs, etc.)
    context={},
    operation=estimate_pi,
    # grid(): Cartesian product of parameters → 3 parameter combinations
    params=metalab.grid(n_samples=[100_000, 1_000_000, 5_000_000]),
    # seeds(): base seed + replicates → 3 independent runs per param config
    # Total runs = 3 params × 3 replicates = 9 runs
    seeds=metalab.seeds(base=42, replicates=10),
)

if __name__ == "__main__":
    # run() returns immediately with a handle; runs execute in background
    handle = metalab.run(exp)  # Results stored in ./runs/pi_mc by default

    # result() blocks until all runs complete
    results = handle.result()

    # View results as a table (params + metrics for each run)
    print(results.table(as_dataframe=True))

    # Other useful result methods:
    # results.table(as_dataframe=True)  # Get a pandas DataFrame
    # results.successful                # Filter to successful runs only
    # results.filter(n_samples=1000)    # Filter by parameter values
    # results[0].metrics                # Access individual run metrics
    # results[0].artifact("name")       # Load a saved artifact
