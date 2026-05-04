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
    job_id: str = "",
    capture_third_party_logs: bool = False,
) -> RunRecord:
    """
    Execute a run payload and return the RunRecord.

    This is the shared execution workflow used by all executors:
    1. Create runtime and capture
    2. Write RUNNING record to store
    3. Execute the operation
    4. Handle success/failure
    5. Persist the final run record

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
        capture_third_party_logs: If True, capture root logger output.

    Returns:
        The completed RunRecord (success or failed).
    """
    started_at = datetime.now()
    job_id = job_id or "unknown"
    event_sink = None
    if hasattr(store, "event_sink"):
        try:
            event_sink = store.event_sink(job_id, worker_id, experiment_id)
            event_sink.emit(
                "started",
                run_id=run_id,
                payload={"params": params_resolved},
            )
        except Exception as e:
            logger.debug(f"Failed to emit start event for {run_id}: {e}")
    if hasattr(store, "put_heartbeat"):
        try:
            store.put_heartbeat(
                job_id=job_id,
                worker_id=worker_id,
                experiment_id=experiment_id,
                state="running",
                current_run_id=run_id,
            )
        except Exception:
            pass

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

    # Set up third-party log capture if requested (additive, not replacing handlers)
    log_buffer: io.StringIO | None = None
    root_log_handler: logging.Handler | None = None

    if capture_third_party_logs:
        log_buffer = io.StringIO()
        root_log_handler = logging.StreamHandler(log_buffer)
        root_log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_log_handler.setLevel(logging.DEBUG)
        # Add handler additively - don't replace existing handlers
        logging.getLogger().addHandler(root_log_handler)

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

        # Persist final record for durability (survives crashes/disconnects)
        try:
            store.put_run_record(result)
        except Exception as e:
            logger.warning(f"Failed to persist final record for {run_id}: {e}")

        if event_sink is not None:
            try:
                event_sink.emit(
                    "finished",
                    run_id=run_id,
                    payload={
                        "duration_ms": duration_ms,
                        "params": params_resolved,
                        "metrics": result.metrics,
                    },
                )
            except Exception:
                pass
        if hasattr(store, "put_heartbeat"):
            try:
                store.put_heartbeat(
                    job_id=job_id,
                    worker_id=worker_id,
                    experiment_id=experiment_id,
                    state="idle",
                )
            except Exception:
                pass

        return result

    except Exception as e:
        # Finalize capture even on failure (flushes logs)
        capture_data = capture.finalize()

        finished_at = datetime.now()
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        result = RunRecord.failed(
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

        # Persist failed record for durability
        try:
            store.put_run_record(result)
        except Exception as persist_err:
            logger.warning(f"Failed to persist failed record for {run_id}: {persist_err}")

        if event_sink is not None:
            try:
                event_sink.emit(
                    "failed",
                    run_id=run_id,
                    payload={
                        "duration_ms": duration_ms,
                        "params": params_resolved,
                        "metrics": result.metrics,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )
            except Exception:
                pass
        if hasattr(store, "put_heartbeat"):
            try:
                store.put_heartbeat(
                    job_id=job_id,
                    worker_id=worker_id,
                    experiment_id=experiment_id,
                    state="failed",
                    current_run_id=run_id,
                )
            except Exception:
                pass

        return result

    finally:
        # Remove our handler from root logger (clean up additive handler)
        if root_log_handler is not None:
            logging.getLogger().removeHandler(root_log_handler)
            root_log_handler.close()

            # Save third-party logging output if any
            if log_buffer is not None:
                log_content = log_buffer.getvalue()
                if log_content:
                    store.put_log(run_id, "logging", log_content)
