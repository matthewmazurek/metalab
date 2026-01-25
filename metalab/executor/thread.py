"""
ThreadExecutor: Thread-based parallel execution.

Runs operations in a ThreadPoolExecutor within the same process.
Context is cached and shared across runs via ContextProvider.
"""

from __future__ import annotations

import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING, Any

from metalab.capture import Capture
from metalab.capture.output import OutputCapture, OutputCaptureContext, OutputCaptureManager
from metalab.context import DefaultContextBuilder, DefaultContextProvider
from metalab.executor.payload import RunPayload
from metalab.runtime import create_runtime
from metalab.store.file import FileStore
from metalab.types import Provenance, RunRecord, Status

if TYPE_CHECKING:
    from metalab.context.builder import ContextBuilder
    from metalab.operation import OperationWrapper


class ThreadExecutor:
    """
    Executor using ThreadPoolExecutor.

    Since threads share memory, we can:
    - Pass operation directly (no need for string reference)
    - Share ContextProvider cache across runs
    - Share store instance
    """

    def __init__(
        self,
        max_workers: int = 4,
        operation: OperationWrapper | None = None,
        context_builder: ContextBuilder | None = None,
        output_capture: OutputCapture | None = None,
        console: Any | None = None,
    ) -> None:
        """
        Initialize the thread executor.

        Args:
            max_workers: Maximum number of worker threads.
            operation: The operation to run (can hold in-memory reference).
            context_builder: Context builder (default: passthrough).
            output_capture: Configuration for capturing stdout/stderr/logging.
            console: Rich console for output routing (from progress tracker).
        """
        self._max_workers = max_workers
        self._operation = operation
        self._context_builder = context_builder or DefaultContextBuilder()
        self._provider = DefaultContextProvider(self._context_builder, maxsize=1)
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._store_cache: dict[str, FileStore] = {}
        self._output_capture = output_capture
        self._console = console
        
        # Set up output capture manager with console if configured
        if output_capture is not None and console is not None:
            OutputCaptureManager.get_instance().set_console(console)

    def _get_store(self, locator: str) -> FileStore:
        """Get or create a store for the given locator."""
        if locator not in self._store_cache:
            self._store_cache[locator] = FileStore(locator)
        return self._store_cache[locator]

    def submit(self, payload: RunPayload) -> Future[RunRecord]:
        """Submit a run for execution."""
        return self._pool.submit(self._execute, payload)

    def _execute(self, payload: RunPayload) -> RunRecord:
        """Execute a single run."""
        started_at = datetime.now()
        store = self._get_store(payload.store_locator)

        # Get operation (use stored reference or import from payload)
        operation = self._operation
        if operation is None:
            from metalab.operation import import_operation

            operation = import_operation(payload.operation_ref)

        # Build context using provider (cached)
        context = self._provider.get(payload.context_spec)

        # Create runtime
        runtime = create_runtime(
            run_id=payload.run_id,
            resource_hints=payload.runtime_hints,
        )

        # Create capture interface
        capture = Capture(
            store=store,
            run_id=payload.run_id,
            artifact_dir=runtime.scratch_dir / "artifacts",
        )

        # Set up output capture context if configured
        output_ctx = None
        captured_output = None
        if self._output_capture is not None:
            output_ctx = OutputCaptureContext(
                self._output_capture,
                console=self._console,
            )

        try:
            # Execute the operation (with optional output capture)
            if output_ctx is not None:
                with output_ctx as captured:
                    captured_output = captured
                    record = operation.run(
                        context=context,
                        params=payload.params_resolved,
                        seeds=payload.seed_bundle,
                        runtime=runtime,
                        capture=capture,
                    )
            else:
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

            # Finalize capture (even on success)
            capture_data = capture.finalize()

            # Store captured output to logs
            self._store_captured_output(store, payload.run_id, captured_output)

            # Update record with capture data and timing
            finished_at = datetime.now()
            duration_ms = int((finished_at - started_at).total_seconds() * 1000)

            return RunRecord(
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
                    executor_id="thread",
                ),
                params_resolved=payload.params_resolved,
                tags=record.tags,
                artifacts=capture_data["artifacts"],
            )

        except Exception as e:
            # Finalize capture even on failure
            capture_data = capture.finalize()
            
            # Store captured output even on failure
            self._store_captured_output(store, payload.run_id, captured_output)
            
            finished_at = datetime.now()
            duration_ms = int((finished_at - started_at).total_seconds() * 1000)

            return RunRecord.failed(
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
                    code_hash=operation.code_hash if operation else None,
                    executor_id="thread",
                ),
                params_resolved=payload.params_resolved,
                artifacts=capture_data["artifacts"],
            )

    def _store_captured_output(
        self,
        store: FileStore,
        run_id: str,
        captured: Any | None,
    ) -> None:
        """Store captured output to the run's log directory."""
        if captured is None:
            return

        from metalab.capture.output import CapturedOutput
        
        if not isinstance(captured, CapturedOutput):
            return

        # Store stdout if captured
        if captured.stdout:
            store.put_log(run_id, "stdout", captured.stdout)

        # Store stderr if captured
        if captured.stderr:
            store.put_log(run_id, "stderr", captured.stderr)

        # Store logging output if captured
        if captured.logging_records:
            store.put_log(run_id, "logging", captured.format_logging())

    def gather(self, futures: list[Future[RunRecord]]) -> list[RunRecord]:
        """Wait for futures and return results."""
        return [f.result() for f in futures]

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        self._pool.shutdown(wait=wait)
        self._provider.close()
