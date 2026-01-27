"""
Executor protocol: Platform-agnostic execution interface.

The Executor abstraction supports:
- Threads (same process)
- Processes (multiple processes)
- SLURM/HPC batch systems via submitit
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from metalab.executor.handle import RunHandle
    from metalab.executor.payload import RunPayload
    from metalab.operation import OperationWrapper
    from metalab.store.base import Store


class Executor(Protocol):
    """
    Protocol for execution backends.

    All executors implement submit() which takes a batch of payloads
    and returns a RunHandle for tracking/awaiting results.

    Key design decisions:
    - submit() takes all payloads at once (enables job arrays for SLURM)
    - Returns RunHandle, not individual futures (unified interface)
    - Workers resolve operation from string references (pickle-safe)
    """

    def submit(
        self,
        payloads: list[RunPayload],
        store: Store,
        operation: OperationWrapper,
        run_ids: list[str] | None = None,
    ) -> RunHandle:
        """
        Submit payloads for execution and return a handle.

        Args:
            payloads: List of run payloads to execute.
            store: Store for persisting results.
            operation: The operation to run.
            run_ids: All run IDs including skipped (for status tracking).

        Returns:
            A RunHandle for tracking and awaiting results.
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor.

        Args:
            wait: If True, wait for pending tasks to complete.
        """
        ...
