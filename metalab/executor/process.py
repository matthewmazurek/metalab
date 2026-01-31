"""
ProcessExecutor: Process-based parallel execution.

Runs operations in a ProcessPoolExecutor.
Workers import operation from string reference (no pickled callables).
"""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

from metalab.executor.handle import LocalRunHandle, RunHandle
from metalab.executor.payload import RunPayload
from metalab.types import RunRecord

if TYPE_CHECKING:
    from metalab.operation import OperationWrapper
    from metalab.store.base import Store


def _process_worker(payload_dict: dict[str, Any], worker_num: int) -> dict[str, Any]:
    """
    Top-level worker function (pickle-safe).

    This function is the entry point for ProcessPoolExecutor workers.
    It must be at module level to be picklable.

    Args:
        payload_dict: Serialized RunPayload.
        worker_num: Worker number for logging identification.

    Returns:
        Serialized RunRecord as dict.
    """
    from metalab.executor.core import execute_payload
    from metalab.executor.payload import RunPayload
    from metalab.operation import import_operation
    from metalab.schema import dump_run_record
    from metalab.store import create_store

    # Deserialize payload
    payload = RunPayload.from_dict(payload_dict)

    # Resolve operation from reference
    operation = import_operation(payload.operation_ref)

    # Create store from locator
    store = create_store(payload.store_locator)

    # Execute using shared logic
    result = execute_payload(
        run_id=payload.run_id,
        experiment_id=payload.experiment_id,
        context_spec=payload.context_spec,
        params_resolved=payload.params_resolved,
        seed_bundle=payload.seed_bundle,
        fingerprints=payload.fingerprints,
        metadata=payload.metadata,
        operation=operation,
        store=store,
        worker_id=f"process:{worker_num}",
        derived_metric_refs=payload.derived_metric_refs,
        capture_third_party_logs=True,
    )

    # Serialize result for return (avoid pickle issues with complex objects)
    return dump_run_record(result)


class ProcessExecutor:
    """
    Executor using ProcessPoolExecutor.

    Workers import operation from operation_ref string.
    """

    def __init__(self, max_workers: int = 4) -> None:
        """
        Initialize the process executor.

        Args:
            max_workers: Maximum number of worker processes.
        """
        self._max_workers = max_workers
        self._pool = ProcessPoolExecutor(max_workers=max_workers)
        self._worker_counter = 0

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
            operation: The operation to run (used to set operation_ref).
            run_ids: All run IDs including skipped (for status tracking).

        Returns:
            A LocalRunHandle for tracking and awaiting results.
        """
        # Use provided run_ids or extract from payloads
        all_run_ids = run_ids if run_ids is not None else [p.run_id for p in payloads]

        # Compute skipped run IDs (in all_run_ids but not in payloads)
        submitted_run_ids = {p.run_id for p in payloads}
        skipped_run_ids = [rid for rid in all_run_ids if rid not in submitted_run_ids]

        # Reset worker counter for this batch
        self._worker_counter = 0

        # Submit all payloads to the process pool
        futures: list[tuple[str, Future[RunRecord]]] = []
        for payload in payloads:
            # Increment worker counter
            self._worker_counter += 1
            worker_num = self._worker_counter

            # Convert payload to dict for pickling
            payload_dict = payload.to_dict()

            # Submit to pool with worker number
            future = self._pool.submit(_process_worker, payload_dict, worker_num)

            # Wrap future to deserialize result
            wrapped = _ResultWrapper(future)
            futures.append((payload.run_id, wrapped))

        return LocalRunHandle(
            futures=futures,
            store=store,
            run_ids=all_run_ids,
            skipped_run_ids=skipped_run_ids,
        )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the process pool."""
        self._pool.shutdown(wait=wait)


class _ResultWrapper:
    """Wrapper to deserialize results from worker."""

    def __init__(self, inner: Future) -> None:
        self._inner = inner

    def result(self, timeout: float | None = None) -> RunRecord:
        from metalab.schema import load_run_record

        data = self._inner.result(timeout=timeout)
        return load_run_record(data)

    def done(self) -> bool:
        return self._inner.done()

    def cancelled(self) -> bool:
        return self._inner.cancelled()

    def cancel(self) -> bool:
        return self._inner.cancel()
