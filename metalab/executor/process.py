"""
ProcessExecutor: Process-based parallel execution.

Runs operations in a ProcessPoolExecutor.
Workers import operation from string reference (no pickled callables).
Each process has its own ContextProvider cache.
"""

from __future__ import annotations

import traceback
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from typing import Any

from metalab.executor.payload import RunPayload, import_ref
from metalab.types import Provenance, RunRecord


def _process_worker(payload_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Top-level worker function (pickle-safe).

    This function is the entry point for ProcessPoolExecutor workers.
    It must be at module level to be picklable.

    Args:
        payload_dict: Serialized RunPayload.

    Returns:
        Serialized RunRecord as dict.
    """
    from metalab.capture import Capture
    from metalab.context import DefaultContextBuilder, DefaultContextProvider
    from metalab.executor.payload import RunPayload
    from metalab.operation import import_operation
    from metalab.runtime import create_runtime
    from metalab.schema import dump_run_record, load_run_record
    from metalab.store.file import FileStore

    # Deserialize payload
    payload = RunPayload.from_dict(payload_dict)

    started_at = datetime.now()

    # Resolve operation from reference
    operation = import_operation(payload.operation_ref)

    # Resolve context builder
    if payload.context_builder_ref:
        builder = import_ref(payload.context_builder_ref)
    else:
        builder = DefaultContextBuilder()

    # Build context (per-process cache)
    provider = DefaultContextProvider(builder)
    context = provider.get(payload.context_spec)

    # Create store
    store = FileStore(payload.store_locator)

    # Create runtime
    runtime = create_runtime(
        run_id=payload.run_id,
        resource_hints=payload.runtime_hints,
    )

    # Create capture
    capture = Capture(
        store=store,
        run_id=payload.run_id,
        artifact_dir=runtime.scratch_dir / "artifacts",
    )

    try:
        # Execute
        record = operation.run(
            context=context,
            params=payload.params_resolved,
            seeds=payload.seed_bundle,
            runtime=runtime,
            capture=capture,
        )

        # Finalize capture
        capture_data = capture.finalize()

        finished_at = datetime.now()
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        result = RunRecord(
            run_id=payload.run_id,
            experiment_id=payload.experiment_id,
            status=record.status,
            context_fingerprint=record.context_fingerprint,
            params_fingerprint=record.params_fingerprint,
            seed_fingerprint=record.seed_fingerprint,
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
            context_fingerprint="",
            params_fingerprint="",
            seed_fingerprint="",
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

    # Serialize result for return (avoid pickle issues with complex objects)
    return dump_run_record(result)


class ProcessExecutor:
    """
    Executor using ProcessPoolExecutor.

    Workers import operation from operation_ref string.
    Each process has its own ContextProvider cache.
    """

    def __init__(self, max_workers: int = 4) -> None:
        """
        Initialize the process executor.

        Args:
            max_workers: Maximum number of worker processes.
        """
        self._max_workers = max_workers
        self._pool = ProcessPoolExecutor(max_workers=max_workers)

    def submit(self, payload: RunPayload) -> Future[RunRecord]:
        """Submit a run for execution."""
        # Convert payload to dict for pickling
        payload_dict = payload.to_dict()

        # Submit to pool
        future = self._pool.submit(_process_worker, payload_dict)

        # Wrap future to deserialize result
        return _ResultWrapper(future)

    def gather(self, futures: list[Future[RunRecord]]) -> list[RunRecord]:
        """Wait for futures and return results."""
        return [f.result() for f in futures]

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
