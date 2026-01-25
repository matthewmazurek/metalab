"""
metalab: A general experiment runner framework.

An Experiment is a reproducible mapping from (FrozenContext, Params, Seeds)
to (RunRecord + Artifacts), executed over a ParamSource using an Executor,
persisted by a Store.

Everything else is plug-ins.

Example:
    import metalab

    @metalab.operation(name="pi_mc")
    def estimate_pi(context, params, seeds, runtime, capture):
        n = params["n_samples"]
        rng = seeds.numpy()
        x, y = rng.random(n), rng.random(n)
        pi_est = 4.0 * (x**2 + y**2 <= 1).mean()
        capture.metric("pi_estimate", pi_est)

    exp = metalab.Experiment(
        name="pi_mc",
        version="0.1",
        context={},
        operation=estimate_pi,
        params=metalab.grid(n_samples=[1000, 10000, 100000]),
        seeds=metalab.seeds(base=42, replicates=3),
    )

    result = metalab.run(exp)  # stores in ./runs/pi_mc by default
    print(result.table())
"""

__version__ = "0.1.0"

# Capture / Runtime
from metalab.capture import Capture

# Context
from metalab.context import (
    ContextBuilder,
    ContextProvider,
    ContextSpec,
    DefaultContextBuilder,
    DefaultContextProvider,
    FrozenContext,
    context_spec,
)

# Events
from metalab.events import Event, EventCallback, EventKind

# Executor (for advanced usage)
from metalab.executor import Executor, ProcessExecutor, RunPayload, ThreadExecutor

# Experiment
from metalab.experiment import Experiment

# Operation decorator
from metalab.operation import OperationWrapper, operation

# Param sources
from metalab.params import (
    GridSource,
    ManualSource,
    ParamCase,
    ParamResolver,
    ParamSource,
    RandomSource,
    ResolvedSource,
    choice,
    grid,
    loguniform,
    loguniform_int,
    manual,
    randint,
    random,
    uniform,
    with_resolver,
)

# Progress (optional rich support)
from metalab.progress import (
    MetricDisplay,
    Progress,
    ProgressTracker,
    SimpleProgressTracker,
    create_progress_tracker,
)
from metalab.progress.display import display_results

# Result
from metalab.result import Results, ResultHandle, Run

# High-level run facade
from metalab.runner import load_results, run
from metalab.runtime import CancellationToken, CancelledError, Runtime

# Seeds
from metalab.seeds import SeedBundle, SeedPlan, seeds

# Store (for advanced usage)
from metalab.store import FileStore, Store

# Types (public)
from metalab.types import ArtifactDescriptor, Provenance, RunRecord, Status

__all__ = [
    # Version
    "__version__",
    # Types
    "RunRecord",
    "ArtifactDescriptor",
    "Status",
    "Provenance",
    # Events
    "Event",
    "EventKind",
    "EventCallback",
    # Seeds
    "SeedBundle",
    "SeedPlan",
    "seeds",
    # Capture / Runtime
    "Capture",
    "Runtime",
    "CancellationToken",
    "CancelledError",
    # Operation
    "operation",
    "OperationWrapper",
    # Params
    "ParamSource",
    "ParamCase",
    "ParamResolver",
    "GridSource",
    "RandomSource",
    "ManualSource",
    "ResolvedSource",
    "grid",
    "random",
    "manual",
    "with_resolver",
    "uniform",
    "loguniform",
    "loguniform_int",
    "randint",
    "choice",
    # Context
    "ContextSpec",
    "FrozenContext",
    "context_spec",
    "ContextBuilder",
    "ContextProvider",
    "DefaultContextBuilder",
    "DefaultContextProvider",
    # Experiment
    "Experiment",
    # Result
    "Results",
    "ResultHandle",  # Backward compatibility alias
    "Run",
    # Run
    "run",
    "load_results",
    # Store
    "Store",
    "FileStore",
    # Executor
    "Executor",
    "ThreadExecutor",
    "ProcessExecutor",
    "RunPayload",
    # Progress
    "MetricDisplay",
    "Progress",
    "ProgressTracker",
    "SimpleProgressTracker",
    "create_progress_tracker",
    "display_results",
]
