"""
Executor module: Platform-agnostic execution layer.

Provides:
- Executor: Protocol for execution backends
- ThreadExecutor: Thread-based parallel execution
- ProcessExecutor: Process-based parallel execution
- RunPayload: Serializable payload for workers
"""

from metalab.executor.base import Executor
from metalab.executor.payload import RunPayload
from metalab.executor.process import ProcessExecutor
from metalab.executor.thread import ThreadExecutor

__all__ = [
    "Executor",
    "RunPayload",
    "ThreadExecutor",
    "ProcessExecutor",
]
