"""
ProcessExecutor: Process-based parallel execution.

Runs operations in a ProcessPoolExecutor.
Workers import operation from string reference (no pickled callables).
"""

from __future__ import annotations

import os
import traceback
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING, Any

from metalab.executor.handle import LocalRunHandle, RunHandle
from metalab.executor.payload import RunPayload
from metalab.types import Provenance, RunRecord

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
    import io
    import logging

    from metalab.capture import Capture
    from metalab.executor.payload import RunPayload
    from metalab.operation import import_operation
    from metalab.runtime import create_runtime
    from metalab.schema import dump_run_record
    from metalab.store.file import FileStore

    # Deserialize payload
    payload = RunPayload.from_dict(payload_dict)

    started_at = datetime.now()

    # Resolve operation from reference
    operation = import_operation(payload.operation_ref)

    # Context is the spec itself - operations receive it directly
    context = payload.context_spec

    # Create store
    store = FileStore(payload.store_locator)

    # Create runtime
    runtime = create_runtime(
        run_id=payload.run_id,
        resource_hints=payload.runtime_hints,
    )

    # Create capture with worker ID for logging
    worker_id = f"process:{worker_num}"
    capture = Capture(
        store=store,
        run_id=payload.run_id,
        artifact_dir=runtime.scratch_dir / "artifacts",
        worker_id=worker_id,
    )

    # Set log label from payload for human-readable filenames
    capture.set_log_label(payload.make_log_label())

    # Set up logging capture for third-party library output
    log_buffer = io.StringIO()
    log_handler = logging.StreamHandler(log_buffer)
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger = logging.getLogger()
    old_handlers = root_logger.handlers[:]
    old_level = root_logger.level
    root_logger.handlers = [log_handler]
    root_logger.setLevel(logging.DEBUG)

    try:
        # Execute
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

        finished_at = datetime.now()
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        result = RunRecord(
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
                executor_id="process",
            ),
            params_resolved=payload.params_resolved,
            tags=record.tags,
            artifacts=capture_data["artifacts"],
        )

    except Exception as e:
        capture_data = capture.finalize()
        finished_at = datetime.now()
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        result = RunRecord.failed(
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
                code_hash=operation.code_hash,
                executor_id="process",
            ),
            params_resolved=payload.params_resolved,
            artifacts=capture_data["artifacts"],
        )

    finally:
        # Restore logging handlers
        root_logger.handlers = old_handlers
        root_logger.setLevel(old_level)

        # Save third-party logging output if any
        log_content = log_buffer.getvalue()
        if log_content:
            label = payload.make_log_label()
            store.put_log(payload.run_id, "logging", log_content, label=label)

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
