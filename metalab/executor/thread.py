"""
ThreadExecutor: Thread-based parallel execution.

Runs operations in a ThreadPoolExecutor within the same process.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from metalab.executor.core import execute_payload
from metalab.executor.handle import LocalRunHandle, RunHandle
from metalab.executor.payload import RunPayload
from metalab.types import RunRecord

if TYPE_CHECKING:
    from metalab.executor.base import ExperimentPlan
    from metalab.operation import OperationWrapper
    from metalab.store.base import Store


class ThreadExecutor:
    """
    Executor using ThreadPoolExecutor.

    Since threads share memory, we can:
    - Pass operation directly (no need for string reference)
    - Share store instance
    """

    def __init__(self, max_workers: int = 4) -> None:
        """
        Initialize the thread executor.

        Args:
            max_workers: Maximum number of worker threads.
        """
        self._max_workers = max_workers
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

        # These are set per-submit call
        self._operation: OperationWrapper | None = None
        self._store: Store | None = None

        # Track worker numbers for logging
        self._worker_counter = 0
        self._worker_counter_lock = threading.Lock()
        self._worker_generation = 0
        self._worker_local = threading.local()

    def _get_worker_id(self) -> str:
        """Get a stable worker ID for the current thread within this submit."""
        if getattr(self._worker_local, "generation", None) == self._worker_generation:
            return self._worker_local.worker_id

        with self._worker_counter_lock:
            self._worker_counter += 1
            worker_id = f"thread:{self._worker_counter}"

        self._worker_local.generation = self._worker_generation
        self._worker_local.worker_id = worker_id
        return worker_id

    def submit_experiment(self, plan: "ExperimentPlan") -> RunHandle:
        """Submit an executor-agnostic experiment plan."""
        return self._submit_payloads(
            payloads=plan.to_payloads(),
            store=plan.store,
            operation=plan.experiment.operation,
            run_ids=plan.all_run_ids,
            job_id=plan.job_id,
        )

    def _submit_payloads(
        self,
        payloads: list[RunPayload],
        store: Store,
        operation: OperationWrapper,
        run_ids: list[str] | None = None,
        job_id: str | None = None,
    ) -> LocalRunHandle:
        """Submit materialized payloads to the thread pool."""
        # Store references for worker threads
        self._operation = operation
        self._store = store

        # Reset worker counter for this batch
        with self._worker_counter_lock:
            self._worker_counter = 0
            self._worker_generation += 1

        # Use provided run_ids or extract from payloads
        all_run_ids = run_ids if run_ids is not None else [p.run_id for p in payloads]

        # Compute skipped run IDs (in all_run_ids but not in payloads)
        submitted_run_ids = {p.run_id for p in payloads}
        skipped_run_ids = [rid for rid in all_run_ids if rid not in submitted_run_ids]

        # Submit all payloads to the thread pool
        futures: list[tuple[str, Future[RunRecord]]] = []
        for payload in payloads:
            future = self._pool.submit(self._execute_one, payload)
            futures.append((payload.run_id, future))

        return LocalRunHandle(
            futures=futures,
            store=store,
            run_ids=all_run_ids,
            job_id=job_id or (payloads[0].job_id if payloads else None),
            skipped_run_ids=skipped_run_ids,
        )

    def _execute_one(self, payload: RunPayload) -> RunRecord:
        """Execute a single run using shared execution logic."""
        store = self._store
        operation = self._operation

        if store is None or operation is None:
            raise RuntimeError("Executor not properly initialized")

        return execute_payload(
            run_id=payload.run_id,
            experiment_id=payload.experiment_id,
            context_spec=payload.context_spec,
            params_resolved=payload.params_resolved,
            seed_bundle=payload.seed_bundle,
            fingerprints=payload.fingerprints,
            metadata=payload.metadata,
            operation=operation,
            store=store,
            worker_id=self._get_worker_id(),
            job_id=payload.job_id,
            derived_metric_refs=payload.derived_metric_refs,
            capture_third_party_logs=False,
        )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        self._pool.shutdown(wait=wait)

    def __enter__(self) -> "ThreadExecutor":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager, ensuring shutdown is called."""
        self.shutdown(wait=True)
