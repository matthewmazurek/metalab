"""
SlurmExecutor: SLURM cluster execution via submitit.

Provides:
- SlurmConfig: Configuration for SLURM job parameters
- SlurmExecutor: Executor that submits jobs to SLURM via submitit
- SlurmRunHandle: Handle for tracking SLURM job execution

Job state tracking:
- Fresh submissions use submitit's job.state for accurate PENDING/RUNNING distinction
- Reconnections attempt to reconstruct Job objects from manifest job IDs
- Falls back to store-only polling if sacct is unavailable
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from metalab.executor.handle import RunHandle, RunStatus
from metalab.executor.payload import RunPayload
from metalab.result import Results
from metalab.types import Status

if TYPE_CHECKING:
    from metalab.events import EventCallback
    from metalab.operation import OperationWrapper
    from metalab.store.base import Store

logger = logging.getLogger(__name__)

# Check for submitit availability
try:
    import submitit

    SUBMITIT_AVAILABLE = True
except ImportError:
    SUBMITIT_AVAILABLE = False
    submitit = None  # type: ignore


@dataclass
class SlurmConfig:
    """
    Configuration for SLURM job submission.

    Attributes:
        partition: SLURM partition/queue name.
        time: Maximum walltime (e.g., "1:00:00" for 1 hour).
        cpus: Number of CPUs per task.
        memory: Memory per task (e.g., "4G", "16GB").
        gpus: Number of GPUs per task (0 for CPU-only).
        max_concurrent: Maximum concurrent jobs (maps to --array=%N).
        modules: Shell modules to load before execution.
        conda_env: Conda environment to activate.
        extra_sbatch: Additional sbatch directives as key-value pairs.
    """

    partition: str = "default"
    time: str = "1:00:00"
    cpus: int = 1
    memory: str = "4G"
    gpus: int = 0
    max_concurrent: int | None = None
    modules: list[str] = field(default_factory=list)
    conda_env: str | None = None
    extra_sbatch: dict[str, str] = field(default_factory=dict)

    def to_submitit_params(self) -> dict[str, Any]:
        """Convert to submitit parameter dict."""
        # Parse time string to minutes
        timeout_min = self._parse_time_to_minutes(self.time)

        # Parse memory to GB
        mem_gb = self._parse_memory_to_gb(self.memory)

        params: dict[str, Any] = {
            "slurm_partition": self.partition,
            "timeout_min": timeout_min,
            "cpus_per_task": self.cpus,
            "mem_gb": mem_gb,
        }

        if self.gpus > 0:
            params["gpus_per_node"] = self.gpus

        if self.max_concurrent is not None:
            params["slurm_array_parallelism"] = self.max_concurrent

        # Add extra sbatch directives
        for key, value in self.extra_sbatch.items():
            params[f"slurm_{key}"] = value

        return params

    @staticmethod
    def _parse_time_to_minutes(time_str: str) -> int:
        """Parse time string (HH:MM:SS or MM:SS) to minutes."""
        parts = time_str.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 60 + minutes + (1 if seconds > 0 else 0)
        elif len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes + (1 if seconds > 0 else 0)
        else:
            return int(parts[0])

    @staticmethod
    def _parse_memory_to_gb(mem_str: str) -> float:
        """Parse memory string (e.g., '4G', '16GB', '1024M') to GB."""
        mem_str = mem_str.upper().strip()
        if mem_str.endswith("GB"):
            return float(mem_str[:-2])
        elif mem_str.endswith("G"):
            return float(mem_str[:-1])
        elif mem_str.endswith("MB"):
            return float(mem_str[:-2]) / 1024
        elif mem_str.endswith("M"):
            return float(mem_str[:-1]) / 1024
        else:
            return float(mem_str)


def _slurm_worker(payload_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Worker function that runs on SLURM compute nodes.

    This function is submitted via submitit and executes on the cluster.
    It imports the operation, runs it, and persists results to the store.

    Args:
        payload_dict: Serialized RunPayload as dict.

    Returns:
        Dict with run_id and status for tracking.
    """
    import io
    import logging as _logging
    import os
    import traceback
    from datetime import datetime

    from metalab.capture import Capture
    from metalab.executor.payload import RunPayload
    from metalab.operation import import_operation
    from metalab.runtime import create_runtime
    from metalab.store.file import FileStore
    from metalab.types import Provenance, RunRecord

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

    # Build worker_id from SLURM environment
    job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    worker_id = f"slurm:{job_id}_{array_task_id}"

    # Create capture with worker ID for logging
    capture = Capture(
        store=store,
        run_id=payload.run_id,
        artifact_dir=runtime.scratch_dir / "artifacts",
        worker_id=worker_id,
    )

    # Set up logging capture for third-party library output
    log_buffer = io.StringIO()
    log_handler = _logging.StreamHandler(log_buffer)
    log_handler.setFormatter(
        _logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger = _logging.getLogger()
    old_handlers = root_logger.handlers[:]
    old_level = root_logger.level
    root_logger.handlers = [log_handler]
    root_logger.setLevel(_logging.DEBUG)

    try:
        # Execute
        record = operation.run(
            context=context,
            params=payload.params_resolved,
            seeds=payload.seed_bundle,
            runtime=runtime,
            capture=capture,
        )

        # Handle None return as success
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
                executor_id="slurm",
            ),
            params_resolved=payload.params_resolved,
            tags=record.tags,
            artifacts=capture_data["artifacts"],
        )

        # Persist to store immediately (source of truth)
        store.put_run_record(result)

        return {"run_id": payload.run_id, "status": result.status.value}

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
                executor_id="slurm",
            ),
            params_resolved=payload.params_resolved,
            artifacts=capture_data["artifacts"],
        )

        # Persist failed result
        store.put_run_record(result)

        return {"run_id": payload.run_id, "status": result.status.value}

    finally:
        # Restore logging handlers
        root_logger.handlers = old_handlers
        root_logger.setLevel(old_level)

        # Save third-party logging output if any
        log_content = log_buffer.getvalue()
        if log_content:
            store.put_log(payload.run_id, "logging", log_content)


class SlurmExecutor:
    """
    Executor that submits jobs to SLURM via submitit.

    Uses submitit's AutoExecutor for job submission, which handles:
    - Serialization of payloads
    - sbatch script generation
    - Job array submission
    - Result retrieval
    """

    def __init__(self, config: SlurmConfig | None = None) -> None:
        """
        Initialize the SLURM executor.

        Args:
            config: SLURM configuration. Uses defaults if not provided.

        Raises:
            ImportError: If submitit is not installed.
        """
        if not SUBMITIT_AVAILABLE:
            raise ImportError(
                "submitit is required for SlurmExecutor. "
                "Install with: pip install metalab[slurm]"
            )

        self._config = config or SlurmConfig()

    def submit(
        self,
        payloads: list[RunPayload],
        store: Store,
        operation: OperationWrapper,
        run_ids: list[str] | None = None,
    ) -> RunHandle:
        """
        Submit payloads to SLURM and return a handle.

        Args:
            payloads: List of run payloads to execute.
            store: Store for persisting results.
            operation: The operation to run.
            run_ids: All run IDs including skipped (for status tracking).

        Returns:
            A SlurmRunHandle for tracking and awaiting results.
        """
        # Use provided run_ids or extract from payloads
        all_run_ids = run_ids if run_ids is not None else [p.run_id for p in payloads]

        # Compute skipped run IDs (in all_run_ids but not in payloads)
        submitted_run_ids = {p.run_id for p in payloads}
        skipped_run_ids = [rid for rid in all_run_ids if rid not in submitted_run_ids]

        if not payloads:
            # Return empty handle if no payloads (all skipped)
            return SlurmRunHandle(
                jobs=[],
                store=store,
                run_ids=all_run_ids,
                job_array_id=None,
                skipped_run_ids=skipped_run_ids,
            )

        # Determine store path for submitit logs
        store_path = Path(store.root) if hasattr(store, "root") else Path(".")
        submitit_folder = store_path / ".submitit"

        # Create submitit executor
        executor = submitit.AutoExecutor(folder=str(submitit_folder))
        executor.update_parameters(**self._config.to_submitit_params())

        # Convert payloads to dicts for pickling
        payload_dicts = [p.to_dict() for p in payloads]

        # Submit as job array
        jobs = executor.map_array(_slurm_worker, payload_dicts)

        # Get job array ID from first job
        job_array_id = jobs[0].job_id.split("_")[0] if jobs else None

        # Build mapping of run_id -> slurm_job_id for reconnection
        run_id_to_job_id: dict[str, str] = {}
        for payload, job in zip(payloads, jobs):
            run_id_to_job_id[payload.run_id] = job.job_id

        # Write manifest for status tracking (after we have job_array_id)
        self._write_manifest(
            store,
            all_run_ids,
            payloads,
            skipped_run_ids,
            job_array_id,
            run_id_to_job_id,
            submitit_folder,
        )

        return SlurmRunHandle(
            jobs=jobs,
            store=store,
            run_ids=all_run_ids,
            job_array_id=job_array_id,
            skipped_run_ids=skipped_run_ids,
            run_id_to_job_id=run_id_to_job_id,
        )

    def _write_manifest(
        self,
        store: Store,
        run_ids: list[str],
        payloads: list[RunPayload],
        skipped_run_ids: list[str] | None = None,
        job_array_id: str | None = None,
        run_id_to_job_id: dict[str, str] | None = None,
        submitit_folder: Path | None = None,
    ) -> None:
        """Write manifest file for tracking expected runs."""
        store_path = Path(store.root) if hasattr(store, "root") else Path(".")
        manifest_path = store_path / "manifest.json"

        manifest = {
            "experiment_id": payloads[0].experiment_id if payloads else "",
            "executor_type": "slurm",
            "run_ids": run_ids,
            "submitted_run_ids": [p.run_id for p in payloads],
            "skipped_run_ids": skipped_run_ids or [],
            "job_array_id": job_array_id,
            "run_id_to_job_id": run_id_to_job_id or {},
            "submitit_folder": str(submitit_folder) if submitit_folder else None,
            "submitted_at": datetime.now().isoformat(),
            "total": len(run_ids),
            "submitted": len(payloads),
            "skipped": len(skipped_run_ids) if skipped_run_ids else 0,
        }

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def shutdown(self, wait: bool = True) -> None:
        """No-op for SLURM executor (jobs run independently)."""
        pass


class SlurmRunHandle:
    """
    RunHandle implementation for SLURM jobs via submitit.

    Supports two polling modes:
    1. Submitit mode (fresh submission or successful reconnection):
       Uses submitit's job.state via sacct for accurate PENDING/RUNNING distinction.
    2. Store-only mode (fallback):
       Polls store for completed records. Cannot distinguish PENDING from RUNNING.

    Emits events when state changes are detected for progress tracking.
    """

    def __init__(
        self,
        jobs: list[Any],  # list[submitit.Job]
        store: Store,
        run_ids: list[str],
        job_array_id: str | None,
        on_event: "EventCallback | None" = None,
        skipped_run_ids: list[str] | None = None,
        run_id_to_job_id: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the SLURM run handle.

        Args:
            jobs: List of submitit Job objects (empty for store-only polling).
            store: Store for reading results.
            run_ids: All run IDs (including skipped).
            job_array_id: SLURM job array ID.
            on_event: Optional callback for progress events.
            skipped_run_ids: Run IDs that were skipped due to resume.
            run_id_to_job_id: Mapping of run_id to SLURM job_id for reconnection.
        """
        self._jobs = jobs
        self._store = store
        self._run_ids = run_ids
        self._job_array_id = job_array_id
        self._on_event = on_event
        # Use list to preserve order for deterministic event emission
        self._skipped_run_ids: list[str] = list(skipped_run_ids or [])
        self._run_id_to_job_id = run_id_to_job_id or {}

        # Build mapping from run_id to job object for efficient lookup
        self._run_id_to_job: dict[str, Any] = {}
        if jobs:
            # For fresh submissions, jobs are in same order as submitted_run_ids
            submitted_run_ids = [
                rid for rid in run_ids if rid not in set(self._skipped_run_ids)
            ]
            for run_id, job in zip(submitted_run_ids, jobs):
                self._run_id_to_job[run_id] = job

        # Track seen status per run_id to avoid duplicate events
        # None = not seen, Status = last seen status
        self._seen_status: dict[str, Status | None] = {rid: None for rid in run_ids}

        # Mark skipped runs as already seen (they won't change)
        for rid in self._skipped_run_ids:
            self._seen_status[rid] = Status.SUCCESS  # Skipped = already successful

        # Track whether we're using submitit or store-only polling
        self._use_submitit_polling = bool(self._run_id_to_job)

        # Emit initial skip events
        self._emit_skip_events()

    def _emit_skip_events(self) -> None:
        """Emit skip events for runs that were skipped due to resume."""
        from metalab.events import Event, emit_event

        for run_id in self._skipped_run_ids:
            emit_event(
                self._on_event, Event.run_skipped(run_id, reason="already exists")
            )

    def _get_job_state(self, run_id: str) -> str | None:
        """
        Get SLURM job state for a run via submitit.

        Returns:
            State string ("PENDING", "RUNNING", "COMPLETED", "FAILED", etc.)
            or None if job is not tracked via submitit.
        """
        job = self._run_id_to_job.get(run_id)
        if job is None:
            return None

        try:
            return job.state
        except Exception:
            # sacct may not be available
            return None

    def _poll_via_submitit(self) -> RunStatus:
        """
        Poll using submitit's job.state for accurate PENDING/RUNNING distinction.

        Returns:
            Current RunStatus with accurate running/pending counts.
        """
        from metalab.events import Event, emit_event

        completed = 0
        failed = 0
        running = 0
        pending = 0

        for run_id in self._run_ids:
            # Skip already-skipped runs
            if run_id in self._skipped_run_ids:
                continue

            prev_status = self._seen_status[run_id]

            # First check store for completed records
            if self._store.run_exists(run_id):
                record = self._store.get_run_record(run_id)
                if record:
                    current_status = record.status

                    # Emit event if status changed from None (newly completed)
                    if prev_status is None and current_status is not None:
                        if current_status == Status.SUCCESS:
                            emit_event(
                                self._on_event,
                                Event.run_finished(
                                    run_id,
                                    duration_ms=record.duration_ms,
                                    metrics=record.metrics,
                                ),
                            )
                            completed += 1
                        elif current_status == Status.FAILED:
                            error_msg = ""
                            if record.error:
                                error_msg = record.error.message
                            emit_event(
                                self._on_event,
                                Event.run_failed(run_id, error=error_msg),
                            )
                            failed += 1

                        self._seen_status[run_id] = current_status
                    else:
                        # Already saw this status
                        if current_status == Status.SUCCESS:
                            completed += 1
                        elif current_status == Status.FAILED:
                            failed += 1
                    continue

            # No record yet - check SLURM job state
            job_state = self._get_job_state(run_id)

            if job_state in ("PENDING", "CONFIGURING", "REQUEUED"):
                pending += 1
            elif job_state in ("RUNNING", "COMPLETING"):
                running += 1
            elif job_state in (
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
            ):
                # Job finished but no record - might be writing or failed to write
                # Treat as running (will be resolved on next poll when record appears)
                running += 1
            else:
                # Unknown state or no state - treat as pending
                pending += 1

        skipped = len(self._skipped_run_ids)

        return RunStatus(
            total=len(self._run_ids),
            completed=completed,
            running=running,
            pending=pending,
            failed=failed,
            skipped=skipped,
        )

    def _poll_via_store(self) -> RunStatus:
        """
        Poll using store only (fallback when submitit is unavailable).

        Cannot distinguish PENDING from RUNNING - all in-flight jobs
        are reported as "running".

        Returns:
            Current RunStatus (running = all in-flight jobs).
        """
        from metalab.events import Event, emit_event

        completed = 0
        failed = 0
        in_flight = 0  # Can't distinguish running from pending

        for run_id in self._run_ids:
            # Skip already-skipped runs
            if run_id in self._skipped_run_ids:
                continue

            prev_status = self._seen_status[run_id]

            if self._store.run_exists(run_id):
                record = self._store.get_run_record(run_id)
                if record:
                    current_status = record.status

                    # Emit event if status changed from None (newly completed)
                    if prev_status is None and current_status is not None:
                        if current_status == Status.SUCCESS:
                            emit_event(
                                self._on_event,
                                Event.run_finished(
                                    run_id,
                                    duration_ms=record.duration_ms,
                                    metrics=record.metrics,
                                ),
                            )
                        elif current_status == Status.FAILED:
                            error_msg = ""
                            if record.error:
                                error_msg = record.error.message
                            emit_event(
                                self._on_event,
                                Event.run_failed(run_id, error=error_msg),
                            )

                        self._seen_status[run_id] = current_status

                    # Count for status
                    if current_status == Status.SUCCESS:
                        completed += 1
                    elif current_status == Status.FAILED:
                        failed += 1
                    else:
                        in_flight += 1
                else:
                    in_flight += 1
            else:
                in_flight += 1

        skipped = len(self._skipped_run_ids)

        # Report all in-flight as "running" (can't distinguish pending)
        return RunStatus(
            total=len(self._run_ids),
            completed=completed,
            running=in_flight,
            pending=0,
            failed=failed,
            skipped=skipped,
        )

    def _poll_and_emit(self) -> RunStatus:
        """
        Poll for state changes and emit events.

        Uses submitit polling if available, falls back to store-only.

        Returns:
            Current RunStatus.
        """
        if self._use_submitit_polling:
            return self._poll_via_submitit()
        else:
            return self._poll_via_store()

    def set_event_callback(self, callback: "EventCallback | None") -> None:
        """
        Set the event callback for progress tracking.

        Args:
            callback: Function to receive events, or None to disable.
        """
        self._on_event = callback

        # If setting a new callback, emit skip events that may have been missed
        if callback is not None:
            self._emit_skip_events()

    @property
    def job_id(self) -> str:
        """SLURM job array ID."""
        return self._job_array_id or "slurm-no-jobs"

    @property
    def status(self) -> RunStatus:
        """Current status by polling. Also emits events for state changes."""
        return self._poll_and_emit()

    @property
    def is_complete(self) -> bool:
        """True if all runs have finished."""
        status = self.status
        return status.done == status.total

    def result(self, timeout: float | None = None) -> Results:
        """
        Block until all jobs complete and return Results.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            Results object containing all completed runs.

        Raises:
            TimeoutError: If timeout is reached before completion.
        """
        start_time = time.time()

        # If we have submitit jobs, wait for them
        if self._jobs:
            for job in self._jobs:
                remaining = None
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        raise TimeoutError("Timeout waiting for SLURM jobs")

                try:
                    job.result(timeout=remaining)
                except Exception:
                    # Job failed, but result is in store
                    pass
        else:
            # Reconnection mode: poll until complete
            poll_interval = 2.0
            while not self.is_complete:
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        raise TimeoutError("Timeout waiting for SLURM jobs")
                time.sleep(poll_interval)

        # Final poll to emit any remaining events
        self._poll_and_emit()

        # Load all records from store
        records = []
        for run_id in self._run_ids:
            record = self._store.get_run_record(run_id)
            if record:
                records.append(record)

        return Results(store=self._store, records=records)

    def cancel(self) -> None:
        """Cancel all pending/running jobs."""
        for job in self._jobs:
            try:
                job.cancel()
            except Exception:
                pass

    @classmethod
    def from_store(
        cls,
        store: Store,
        on_event: "EventCallback | None" = None,
    ) -> "SlurmRunHandle":
        """
        Create a handle by loading manifest from store (reconnection).

        Attempts to reconstruct submitit Job objects from manifest job IDs
        to enable accurate PENDING/RUNNING distinction via sacct. Falls back
        to store-only polling with a warning if reconstruction fails.

        Args:
            store: Store containing the manifest and run records.
            on_event: Optional callback for progress events.

        Returns:
            A SlurmRunHandle for tracking and awaiting results.

        Raises:
            FileNotFoundError: If no manifest exists in the store.

        Example:
            # In a new session, reconnect to watch progress
            store = FileStore("./runs/my_exp")
            handle = SlurmRunHandle.from_store(store)

            with create_progress_tracker(total=handle.status.total) as tracker:
                handle.set_event_callback(tracker)
                results = handle.result()  # Blocks until complete
        """
        store_path = Path(store.root) if hasattr(store, "root") else Path(".")
        manifest_path = store_path / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest found at {manifest_path}. "
                "Cannot reconnect without a manifest."
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Check executor type
        executor_type = manifest.get("executor_type", "slurm")
        if executor_type != "slurm":
            raise ValueError(
                f"Cannot reconnect to executor type '{executor_type}' using SlurmRunHandle. "
                "Only 'slurm' executors support reconnection."
            )

        # Try to reconstruct submitit Job objects from manifest
        jobs: list[Any] = []
        run_id_to_job: dict[str, Any] = {}
        run_id_to_job_id = manifest.get("run_id_to_job_id", {})
        submitit_folder = manifest.get("submitit_folder")

        if run_id_to_job_id and submitit_folder and SUBMITIT_AVAILABLE:
            try:
                # Reconstruct Job objects using submitit
                submitit_folder_path = Path(submitit_folder)

                for run_id, job_id in run_id_to_job_id.items():
                    try:
                        # submitit Job objects can be reconstructed from folder + job_id
                        job = submitit.SlurmJob(
                            folder=submitit_folder_path, job_id=job_id
                        )
                        jobs.append(job)
                        run_id_to_job[run_id] = job
                    except Exception as e:
                        logger.debug(f"Failed to reconstruct job {job_id}: {e}")
                        # Continue without this job

                if run_id_to_job:
                    logger.info(
                        f"Reconnected to {len(run_id_to_job)} SLURM jobs via submitit. "
                        "Using sacct for accurate job state tracking."
                    )
                else:
                    warnings.warn(
                        "Failed to reconstruct submitit Job objects. "
                        "Falling back to store-only polling (cannot distinguish PENDING from RUNNING).",
                        stacklevel=2,
                    )
            except Exception as e:
                warnings.warn(
                    f"Failed to reconnect via submitit ({e}). "
                    "Falling back to store-only polling (cannot distinguish PENDING from RUNNING).",
                    stacklevel=2,
                )
        elif not SUBMITIT_AVAILABLE:
            warnings.warn(
                "submitit not available. "
                "Falling back to store-only polling (cannot distinguish PENDING from RUNNING).",
                stacklevel=2,
            )
        elif not run_id_to_job_id:
            warnings.warn(
                "Manifest missing job ID mapping (old format?). "
                "Falling back to store-only polling (cannot distinguish PENDING from RUNNING).",
                stacklevel=2,
            )

        handle = cls(
            jobs=jobs,
            store=store,
            run_ids=manifest["run_ids"],
            job_array_id=manifest.get("job_array_id"),
            on_event=on_event,
            skipped_run_ids=manifest.get("skipped_run_ids", []),
            run_id_to_job_id=run_id_to_job_id,
        )

        # Override the run_id_to_job mapping with reconstructed jobs
        handle._run_id_to_job = run_id_to_job
        handle._use_submitit_polling = bool(run_id_to_job)

        return handle
