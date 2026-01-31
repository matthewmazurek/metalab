"""
Core execution logic shared across all executors.

This module provides the common execution workflow used by ThreadExecutor,
ProcessExecutor, and SlurmExecutor to ensure consistent behavior.
"""

from __future__ import annotations

import io
import logging
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Any

from metalab.capture import Capture
from metalab.runtime import create_runtime
from metalab.types import Provenance, RunRecord

if TYPE_CHECKING:
    from metalab.operation import OperationWrapper
    from metalab.seeds.bundle import SeedBundle
    from metalab.store.base import Store

logger = logging.getLogger(__name__)


def execute_payload(
    *,
    run_id: str,
    experiment_id: str,
    context_spec: Any,
    params_resolved: dict[str, Any],
    seed_bundle: "SeedBundle",
    fingerprints: dict[str, str],
    metadata: dict[str, Any],
    operation: "OperationWrapper",
    store: "Store",
    worker_id: str,
    derived_metric_refs: list[str] | None = None,
    capture_third_party_logs: bool = False,
) -> RunRecord:
    """
    Execute a run payload and return the RunRecord.

    This is the shared execution workflow used by all executors:
    1. Create runtime and capture
    2. Write RUNNING record to store
    3. Execute the operation
    4. Handle success/failure
    5. Compute derived metrics (if configured)

    Args:
        run_id: Unique identifier for this run.
        experiment_id: The experiment identifier (name:version).
        context_spec: The serializable context specification.
        params_resolved: The resolved parameter dictionary.
        seed_bundle: The seed bundle for this run.
        fingerprints: Dict with context, params, seed fingerprints.
        metadata: Experiment-level metadata (passed to Runtime).
        operation: The operation to execute.
        store: Store for persisting results and artifacts.
        worker_id: Identifier for the worker (e.g., "thread:1", "slurm:123_0").
        derived_metric_refs: Optional list of derived metric function references.
        capture_third_party_logs: If True, capture root logger output.

    Returns:
        The completed RunRecord (success or failed).
    """
    started_at = datetime.now()

    # Create runtime
    runtime = create_runtime(
        run_id=run_id,
        metadata=metadata,
    )

    # Create capture interface
    capture = Capture(
        store=store,
        run_id=run_id,
        artifact_dir=runtime.scratch_dir / "artifacts",
        worker_id=worker_id,
    )

    # Set up third-party log capture if requested
    log_buffer: io.StringIO | None = None
    old_handlers: list[logging.Handler] = []
    old_level: int = logging.WARNING

    if capture_third_party_logs:
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

    # Write RUNNING record before execution
    running_record = RunRecord.running(
        run_id=run_id,
        experiment_id=experiment_id,
        context_fingerprint=fingerprints.get("context", ""),
        params_fingerprint=fingerprints.get("params", ""),
        seed_fingerprint=fingerprints.get("seed", ""),
        started_at=started_at,
        params_resolved=params_resolved,
        provenance=Provenance(
            code_hash=operation.code_hash,
            executor_id=worker_id.split(":")[0] if ":" in worker_id else worker_id,
        ),
    )
    store.put_run_record(running_record)

    try:
        # Execute the operation
        record = operation.run(
            context=context_spec,
            params=params_resolved,
            seeds=seed_bundle,
            runtime=runtime,
            capture=capture,
        )

        # Handle None return as success (no return needed from operations)
        if record is None:
            record = RunRecord.success()

        # Finalize capture (flushes logs)
        capture_data = capture.finalize()

        # Calculate timing
        finished_at = datetime.now()
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        # Build final record
        result = RunRecord(
            run_id=run_id,
            experiment_id=experiment_id,
            status=record.status,
            context_fingerprint=fingerprints.get("context", ""),
            params_fingerprint=fingerprints.get("params", ""),
            seed_fingerprint=fingerprints.get("seed", ""),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            metrics={**record.metrics, **capture_data["metrics"]},
            provenance=Provenance(
                code_hash=operation.code_hash,
                executor_id=worker_id.split(":")[0] if ":" in worker_id else worker_id,
            ),
            params_resolved=params_resolved,
            tags=record.tags,
            artifacts=capture_data["artifacts"],
        )

        # Compute derived metrics if configured
        if derived_metric_refs:
            _compute_derived_metrics(result, store, derived_metric_refs)

        return result

    except Exception as e:
        # Finalize capture even on failure (flushes logs)
        capture_data = capture.finalize()

        finished_at = datetime.now()
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        return RunRecord.failed(
            run_id=run_id,
            experiment_id=experiment_id,
            context_fingerprint=fingerprints.get("context", ""),
            params_fingerprint=fingerprints.get("params", ""),
            seed_fingerprint=fingerprints.get("seed", ""),
            started_at=started_at,
            finished_at=finished_at,
            error_type=type(e).__name__,
            error_message=str(e),
            error_traceback=traceback.format_exc(),
            metrics=capture_data["metrics"],
            provenance=Provenance(
                code_hash=operation.code_hash,
                executor_id=worker_id.split(":")[0] if ":" in worker_id else worker_id,
            ),
            params_resolved=params_resolved,
            artifacts=capture_data["artifacts"],
        )

    finally:
        # Restore logging handlers if we captured them
        if capture_third_party_logs and log_buffer is not None:
            root_logger = logging.getLogger()
            root_logger.handlers = old_handlers
            root_logger.setLevel(old_level)

            # Save third-party logging output if any
            log_content = log_buffer.getvalue()
            if log_content:
                store.put_log(run_id, "logging", log_content)


def _compute_derived_metrics(
    record: RunRecord,
    store: "Store",
    metric_refs: list[str],
) -> None:
    """
    Compute and store derived metrics for a completed run.

    Args:
        record: The completed run record.
        store: The store for persisting derived metrics.
        metric_refs: List of function references ('module:func' format).
    """
    from metalab.derived import compute_derived_for_run, import_derived_metric
    from metalab.result import Run

    # Create Run object from record
    run = Run(record, store)

    # Import and apply metric functions
    functions = []
    for ref in metric_refs:
        try:
            func = import_derived_metric(ref)
            functions.append(func)
        except Exception as e:
            logger.warning(f"Failed to import derived metric '{ref}': {e}")

    if functions:
        derived = compute_derived_for_run(run, functions)
        store.put_derived(record.run_id, derived)
