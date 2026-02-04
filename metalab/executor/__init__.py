"""
Executor module: Platform-agnostic execution layer.

Provides:

- Executor: Protocol for execution backends
- ThreadExecutor: Thread-based parallel execution
- ProcessExecutor: Process-based parallel execution
- SlurmExecutor: SLURM cluster execution via submitit
- RunPayload: Serializable payload for workers
- RunHandle: Promise-like interface for tracking execution
- RunStatus: Status summary for a batch of runs
- HandleRegistry: Registry for reconnectable executor handles
"""

from metalab.executor.base import Executor
from metalab.executor.handle import LocalRunHandle, RunHandle, RunStatus
from metalab.executor.payload import RunPayload
from metalab.executor.process import ProcessExecutor
from metalab.executor.registry import HandleRegistry
from metalab.executor.thread import ThreadExecutor


# Lazy import for SlurmExecutor (requires submitit)
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


__all__ = [
    "Executor",
    "HandleRegistry",
    "RunPayload",
    "RunHandle",
    "RunStatus",
    "LocalRunHandle",
    "ThreadExecutor",
    "ProcessExecutor",
    "SlurmExecutor",
    "SlurmConfig",
    "SlurmRunHandle",
]
