"""
metalab: A general experiment runner framework.

An Experiment is a reproducible mapping from (Context, Params, Seeds)
to (RunRecord + Artifacts), executed over a ParamSource using an Executor,
persisted by a Store.

Everything else is plug-ins.

Example:
    import metalab

    @metalab.operation
    def estimate_pi(params, seeds, capture):
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

    # Run and get results
    handle = metalab.run(exp)  # returns RunHandle
    results = handle.result()  # blocks until complete
    print(results.table())

    # Or for SLURM execution:
    handle = metalab.run(
        exp,
        store="/scratch/runs/pi_mc",
        executor=metalab.SlurmExecutor(metalab.SlurmConfig(partition="gpu")),
    )
    print(handle.status)  # check progress without blocking
"""

__version__ = "0.1.0"

# Capture / Runtime
from metalab.capture import Capture

# Context
from metalab.context import (
    ContextProvider,
    ContextSpec,
    DefaultContextProvider,
    FrozenContext,
    context_spec,
)

# Events
from metalab.events import Event, EventCallback, EventKind

# Executor (for advanced usage)
from metalab.executor import (
    Executor,
    LocalRunHandle,
    ProcessExecutor,
    RunHandle,
    RunPayload,
    RunStatus,
    ThreadExecutor,
)


# Lazy import for SLURM (requires submitit)
def __getattr__(name: str):
    if name in ("SlurmExecutor", "SlurmConfig", "SlurmRunHandle"):
        from metalab.executor.slurm import SlurmConfig, SlurmExecutor, SlurmRunHandle

        if name == "SlurmExecutor":
            return SlurmExecutor
        elif name == "SlurmConfig":
            return SlurmConfig
        elif name == "SlurmRunHandle":
            return SlurmRunHandle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Utilities / File hashing
from metalab._ids import DirPath, FilePath, Fingerprintable, dir_hash, file_hash

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
from metalab.result import ExperimentInfo, Results, Run

# High-level run facade
from metalab.runner import load_results, reconnect, run
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
    "ContextProvider",
    "DefaultContextProvider",
    # Experiment
    "Experiment",
    # Result
    "ExperimentInfo",
    "Results",
    "Run",
    # Run
    "run",
    "load_results",
    "reconnect",
    # Store
    "Store",
    "FileStore",
    # Executor
    "Executor",
    "ThreadExecutor",
    "ProcessExecutor",
    "RunPayload",
    "RunHandle",
    "RunStatus",
    "LocalRunHandle",
    # SLURM (lazy-loaded)
    "SlurmExecutor",
    "SlurmConfig",
    "SlurmRunHandle",
    # Progress
    "MetricDisplay",
    "Progress",
    "ProgressTracker",
    "SimpleProgressTracker",
    "create_progress_tracker",
    "display_results",
    # Utilities / File hashing
    "file_hash",
    "dir_hash",
    "FilePath",
    "DirPath",
    "Fingerprintable",
]
