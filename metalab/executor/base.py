"""
Executor protocol: Platform-agnostic execution interface.

The Executor abstraction supports:
- Threads (same process)
- Processes (multiple processes)
- Future: HPC batch systems (ARC/Slurm)
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Protocol

from metalab.executor.payload import RunPayload
from metalab.types import RunRecord


class Executor(Protocol):
    """
    Protocol for execution backends.

    An Executor submits RunPayloads for execution and returns Futures
    that resolve to RunRecords.

    Key design decision: submit() takes only a RunPayload (no callables).
    Workers resolve operation and context_builder from string references.
    This ensures pickle-safety for ProcessExecutor and HPC backends.
    """

    def submit(self, payload: RunPayload) -> Future[RunRecord]:
        """
        Submit a run for execution.

        Args:
            payload: The serializable run payload.

        Returns:
            A Future that resolves to a RunRecord.
        """
        ...

    def gather(self, futures: list[Future[RunRecord]]) -> list[RunRecord]:
        """
        Wait for multiple futures and return results.

        Args:
            futures: List of futures from submit().

        Returns:
            List of RunRecords (in same order as futures).
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor.

        Args:
            wait: If True, wait for pending tasks to complete.
        """
        ...
