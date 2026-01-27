"""
ThreadExecutor: Thread-based parallel execution.

Runs operations in a ThreadPoolExecutor within the same process.
"""

from __future__ import annotations

import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING

from metalab.capture import Capture
from metalab.executor.handle import LocalRunHandle, RunHandle
from metalab.executor.payload import RunPayload
from metalab.runtime import create_runtime
from metalab.types import Provenance, RunRecord

if TYPE_CHECKING:
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

    def _get_worker_id(self) -> str:
        """Get a unique worker ID for logging."""
        with self._worker_counter_lock:
            self._worker_counter += 1
            return f"thread:{self._worker_counter}"

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
            A LocalRunHandle for tracking and awaiting results.
        """
        # Store references for worker threads
        self._operation = operation
        self._store = store

        # Reset worker counter for this batch
        with self._worker_counter_lock:
            self._worker_counter = 0

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
            skipped_run_ids=skipped_run_ids,
        )

    def _execute_one(self, payload: RunPayload) -> RunRecord:
        """Execute a single run."""
        started_at = datetime.now()
        store = self._store
        operation = self._operation

        if store is None or operation is None:
            raise RuntimeError("Executor not properly initialized")

        # Get worker ID for this execution
        worker_id = self._get_worker_id()

        # Context is the spec itself - operations receive it directly
        context = payload.context_spec

        # Create runtime
        runtime = create_runtime(
            run_id=payload.run_id,
            resource_hints=payload.runtime_hints,
        )

        # Create capture interface with worker ID for logging
        capture = Capture(
            store=store,
            run_id=payload.run_id,
            artifact_dir=runtime.scratch_dir / "artifacts",
            worker_id=worker_id,
        )

        # Set log label from payload for human-readable filenames
        capture.set_log_label(payload.make_log_label())

        try:
            # Execute the operation
            record = operation.run(
                context=context,
                params=payload.params_resolved,
                seeds=payload.seed_bundle,
                runtime=runtime,
                capture=capture,
            )

            # Handle None return as success (no return needed from operations)
            if record is None:
                record = RunRecord.success()

            # Finalize capture (flushes logs)
            capture_data = capture.finalize()

            # Update record with capture data and timing
            finished_at = datetime.now()
            duration_ms = int((finished_at - started_at).total_seconds() * 1000)

            return RunRecord(
                run_id=payload.run_id,
                experiment_id=payload.experiment_id,
                status=record.status,
                context_fingerprint=payload.fingerprints.get("context", ""),
                params_fingerprint=payload.fingerprints.get("params", ""),
                seed_fingerprint=payload.fingerprints.get("seed", ""),
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                metrics={**record.metrics, **capture_data["metrics"]},
                provenance=Provenance(
                    code_hash=operation.code_hash,
                    executor_id="thread",
                ),
                params_resolved=payload.params_resolved,
                tags=record.tags,
                artifacts=capture_data["artifacts"],
            )

        except Exception as e:
            # Finalize capture even on failure (flushes logs)
            capture_data = capture.finalize()

            finished_at = datetime.now()
            duration_ms = int((finished_at - started_at).total_seconds() * 1000)

            return RunRecord.failed(
                run_id=payload.run_id,
                experiment_id=payload.experiment_id,
                context_fingerprint=payload.fingerprints.get("context", ""),
                params_fingerprint=payload.fingerprints.get("params", ""),
                seed_fingerprint=payload.fingerprints.get("seed", ""),
                started_at=started_at,
                finished_at=finished_at,
                error_type=type(e).__name__,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
                metrics=capture_data["metrics"],
                provenance=Provenance(
                    code_hash=operation.code_hash if operation else None,
                    executor_id="thread",
                ),
                params_resolved=payload.params_resolved,
                artifacts=capture_data["artifacts"],
            )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        self._pool.shutdown(wait=wait)
