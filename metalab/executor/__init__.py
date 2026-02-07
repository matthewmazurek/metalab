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
- ExecutorConfig: Base configuration class for executor backends
- ExecutorConfigRegistry: Registry mapping type names to config classes
- executor_from_config: Factory function for config-driven executor creation
- LocalExecutorConfig: Config for local (thread/process) execution
- SlurmExecutorConfig: Config for SLURM cluster execution
"""

from metalab.executor.base import Executor
from metalab.executor.config import (
    ExecutorConfig,
    ExecutorConfigRegistry,
    executor_from_config,
    resolve_executor,
)
from metalab.executor.handle import LocalRunHandle, RunHandle, RunStatus
from metalab.executor.local_config import (
    LocalExecutorConfig,
)  # triggers auto-registration
from metalab.executor.payload import RunPayload
from metalab.executor.process import ProcessExecutor
from metalab.executor.registry import HandleRegistry
from metalab.executor.thread import ThreadExecutor


# Lazy import for Slurm (requires submitit)
def __getattr__(name: str):
    if name in ("SlurmExecutor", "SlurmConfig", "SlurmRunHandle"):
        from metalab.executor.slurm import SlurmConfig, SlurmExecutor, SlurmRunHandle

        if name == "SlurmExecutor":
            return SlurmExecutor
        elif name == "SlurmConfig":
            return SlurmConfig
        elif name == "SlurmRunHandle":
            return SlurmRunHandle
    if name == "SlurmExecutorConfig":
        from metalab.executor.slurm_config import SlurmExecutorConfig

        return SlurmExecutorConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Executor",
    "ExecutorConfig",
    "ExecutorConfigRegistry",
    "executor_from_config",
    "resolve_executor",
    "HandleRegistry",
    "LocalExecutorConfig",
    "RunPayload",
    "RunHandle",
    "RunStatus",
    "LocalRunHandle",
    "ThreadExecutor",
    "ProcessExecutor",
    "SlurmExecutor",
    "SlurmExecutorConfig",
    "SlurmConfig",
    "SlurmRunHandle",
]
