"""
Runner: Orchestrates experiment execution with resume/dedupe.

The Runner:
1. Generates run payloads from experiment configuration
2. Checks for existing runs (resume)
3. Submits to executor
4. Returns a RunHandle for tracking/awaiting results

Progress tracking:
- Pass `progress=True` for automatic progress display (auto-detects rich)
- Pass `progress=Progress(...)` for customized progress display
- Pass `on_event=callback` for custom event handling
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metalab.derived import DerivedMetricFn
    from metalab.events import EventCallback
    from metalab.executor.base import Executor
    from metalab.executor.handle import RunHandle as RunHandleProtocol
    from metalab.executor.slurm import SlurmExecutor
    from metalab.experiment import Experiment
    from metalab.progress import Progress, ProgressTracker
    from metalab.store.base import Store

from metalab._canonical import fingerprint
from metalab._ids import (
    compute_run_id,
    fingerprint_params,
    fingerprint_seeds,
    resolve_context,
)
from metalab.executor.handle import RunHandle, RunStatus
from metalab.executor.payload import RunPayload
from metalab.executor.thread import ThreadExecutor
from metalab.result import Results
from metalab.store.file import FileStore
from metalab.types import Status

logger = logging.getLogger(__name__)


class ProgressRunHandle:
    """
    Wrapper handle that manages progress tracker lifecycle.

    Starts the progress tracker on creation and stops it when result() is called.
    Delegates all other operations to the underlying handle.
    """

    def __init__(
        self,
        handle: "RunHandleProtocol",
        tracker: "ProgressTracker",
    ) -> None:
        self._handle = handle
        self._tracker = tracker
        self._tracker_started = False
        self._tracker_stopped = False

        # Start the tracker and wire up events
        self._start_tracker()

    def _start_tracker(self) -> None:
        """Start the progress tracker."""
        if self._tracker_started:
            return
        self._tracker.__enter__()
        self._handle.set_event_callback(self._tracker)
        self._tracker_started = True

    def _stop_tracker(self) -> None:
        """Stop the progress tracker."""
        if self._tracker_stopped or not self._tracker_started:
            return
        self._tracker.__exit__(None, None, None)
        self._tracker_stopped = True

    @property
    def job_id(self) -> str:
        """Unique identifier for this execution batch."""
        return self._handle.job_id

    @property
    def status(self) -> RunStatus:
        """Current status of all runs (non-blocking)."""
        return self._handle.status

    @property
    def is_complete(self) -> bool:
        """True if all runs have finished (success or failure)."""
        return self._handle.is_complete

    def result(self, timeout: float | None = None) -> Results:
        """
        Block until all runs complete and return Results.

        Also stops the progress tracker display.
        """
        try:
            return self._handle.result(timeout=timeout)
        finally:
            self._stop_tracker()

    def cancel(self) -> None:
        """Cancel pending and running jobs."""
        self._handle.cancel()
        self._stop_tracker()

    def set_event_callback(self, callback: "EventCallback | None") -> None:
        """
        Set an additional event callback.

        Note: The progress tracker callback is always called first.
        """
        # Create a combined callback that calls both
        original_tracker = self._tracker

        def combined_callback(event: Any) -> None:
            original_tracker(event)
            if callback is not None:
                callback(event)

        self._handle.set_event_callback(combined_callback)

    def __del__(self) -> None:
        """Ensure tracker is stopped on garbage collection."""
        self._stop_tracker()


def generate_payloads(
    experiment: Experiment,
    store: Store,
    resume: bool = True,
    persist_manifest: bool = True,
    derived_metric_refs: list[str] | None = None,
) -> tuple[list[RunPayload], list[str]]:
    """
    Generate payloads for an experiment.

    Args:
        experiment: The experiment to run.
        store: Store for checking existing runs.
        resume: If True, skip already-completed runs.
        persist_manifest: If True, save resolved context manifest for auditability.
        derived_metric_refs: List of derived metric function references ('module:func').
            These are post-hoc computations that do NOT affect run fingerprints.

    Returns:
        Tuple of (payloads to execute, all run IDs including skipped).
    """
    # Resolve context - computes lazy hashes for FilePath/DirPath
    resolved_context, manifest = resolve_context(experiment.context)
    ctx_fp = fingerprint(resolved_context)

    # Optionally persist the resolved manifest for auditability
    if persist_manifest and manifest and hasattr(store, "root"):
        import json
        from pathlib import Path

        manifest_path = (
            Path(store.root) / f"{experiment.experiment_id}_context_manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w") as f:
            json.dump(
                {
                    "experiment_id": experiment.experiment_id,
                    "context_fingerprint": ctx_fp,
                    "resolved_fields": manifest,
                },
                f,
                indent=2,
            )
        logger.debug(f"Saved context manifest to {manifest_path}")

    # First pass: compute all run_ids and store resolved params/seeds
    # We need this before writing the manifest so we can include run_ids
    all_run_ids = []
    run_data: list[tuple[str, dict, Any, str, str]] = (
        []
    )  # (run_id, resolved_params, seed_bundle, params_fp, seed_fp)

    for param_case in experiment.params:
        # Resolve params if resolver is provided
        if experiment.param_resolver is not None:
            resolver = experiment.param_resolver
            if hasattr(resolver, "resolve"):
                resolved_params = resolver.resolve({}, param_case.params)
            else:
                resolved_params = resolver({}, param_case.params)
        else:
            resolved_params = param_case.params

        params_fp = fingerprint_params(resolved_params)

        for seed_bundle in experiment.seeds:
            seed_fp = fingerprint_seeds(seed_bundle)

            run_id = compute_run_id(
                experiment_id=experiment.experiment_id,
                context_fp=ctx_fp,
                params_fp=params_fp,
                seed_fp=seed_fp,
                code_fp=experiment.operation.code_hash,
            )

            all_run_ids.append(run_id)
            run_data.append((run_id, resolved_params, seed_bundle, params_fp, seed_fp))

    # Write experiment manifest (versioned by timestamp) - now includes run_ids
    if persist_manifest and hasattr(store, "root"):
        import json
        from datetime import datetime
        from pathlib import Path

        from metalab.manifest import build_experiment_manifest

        exp_manifest = build_experiment_manifest(
            experiment=experiment,
            context_fingerprint=ctx_fp,
            total_runs=len(all_run_ids),
            run_ids=all_run_ids,
        )

        exp_dir = Path(store.root) / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_id = experiment.experiment_id.replace(":", "_")
        exp_manifest_path = exp_dir / f"{safe_id}_{timestamp}.json"

        with exp_manifest_path.open("w") as f:
            json.dump(exp_manifest, f, indent=2, default=str)

        logger.debug(f"Saved experiment manifest to {exp_manifest_path}")

    # Second pass: create payloads, skipping successful runs if resume=True
    payloads = []

    for run_id, resolved_params, seed_bundle, params_fp, seed_fp in run_data:
        # Check for resume
        if resume and store.run_exists(run_id):
            existing = store.get_run_record(run_id)
            if existing and existing.status == Status.SUCCESS:
                continue

        payload = RunPayload(
            run_id=run_id,
            experiment_id=experiment.experiment_id,
            context_spec=experiment.context,
            params_resolved=resolved_params,
            seed_bundle=seed_bundle,
            store_locator=str(store.root) if hasattr(store, "root") else "",
            fingerprints={
                "context": ctx_fp,
                "params": params_fp,
                "seed": seed_fp,
            },
            metadata=experiment.metadata,
            operation_ref=experiment.operation.ref,
            derived_metric_refs=derived_metric_refs,
        )

        payloads.append(payload)

    return payloads, all_run_ids


def _run_slurm_indexed(
    experiment: "Experiment",
    store: "Store",
    executor: "SlurmExecutor",
    resume: bool = True,
    derived_metric_refs: list[str] | None = None,
) -> "RunHandle":
    """
    Submit experiment via index-addressed SLURM array.

    This avoids the per-task pickle overhead of submitit by:
    1. Writing a single array spec file with experiment configuration
    2. Submitting SLURM array job(s) via sbatch
    3. Each task reconstructs its parameters from SLURM_ARRAY_TASK_ID

    Args:
        experiment: The experiment to run.
        store: Store for persisting results.
        executor: The SLURM executor.
        resume: Skip existing successful runs.
        derived_metric_refs: Optional derived metric function references.

    Returns:
        SlurmRunHandle for tracking and awaiting results.
    """
    from metalab._canonical import fingerprint
    from metalab._ids import resolve_context

    # Resolve context - computes lazy hashes for FilePath/DirPath
    resolved_context, manifest = resolve_context(experiment.context)
    ctx_fp = fingerprint(resolved_context)

    # Compute total runs
    total_runs = len(experiment.params) * len(experiment.seeds)

    # Count existing successful runs if resuming
    skipped_count = 0
    if resume:
        # For SLURM indexed, we can't easily pre-scan all run_ids without
        # enumerating them (which defeats the purpose). Instead, we let
        # the worker skip at runtime. We could optionally do a store scan
        # here for existing runs.
        # For now, we set skipped_count to 0 and let workers skip.
        pass

    # Optionally persist the resolved manifest for auditability
    if manifest and hasattr(store, "root"):
        import json
        from pathlib import Path

        manifest_path = (
            Path(store.root) / f"{experiment.experiment_id}_context_manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w") as f:
            json.dump(
                {
                    "experiment_id": experiment.experiment_id,
                    "context_fingerprint": ctx_fp,
                    "resolved_fields": manifest,
                },
                f,
                indent=2,
            )
        logger.debug(f"Saved context manifest to {manifest_path}")

    # Submit via indexed array
    handle = executor.submit_indexed(
        experiment=experiment,
        store=store,
        context_fingerprint=ctx_fp,
        total_runs=total_runs,
        skipped_count=skipped_count,
        derived_metric_refs=derived_metric_refs,
    )

    return handle


def run(
    experiment: Experiment,
    store: str | Store | None = None,
    executor: "Executor | None" = None,
    resume: bool = True,
    progress: "bool | Progress | None" = None,
    on_event: "EventCallback | None" = None,
    derived_metrics: "list[str | DerivedMetricFn] | None" = None,
) -> RunHandle:
    """
    Run an experiment and return a handle for tracking/awaiting results.

    This is the main entry point for executing experiments. Returns a RunHandle
    which can be used to check status, wait for completion, or get results.

    By default, runs execute sequentially (one at a time). For parallel execution,
    pass an explicit executor.

    Args:
        experiment: The experiment to run.
        store: Store path or instance. Defaults to "./runs/{experiment.name}".
        executor: Executor instance. Defaults to sequential (single-threaded).
            For parallel execution, use:
            - ThreadExecutor(max_workers=N) for thread-based parallelism
            - ProcessExecutor(max_workers=N) for process-based parallelism
            - SlurmExecutor(...) for cluster execution
        resume: Skip existing successful runs (default: True).
        progress: Enable progress display. Options:
            - True: Auto-detect rich, use simple fallback if not available.
            - False/None: No progress display.
            - Progress(...): Customized progress display with title, metrics, etc.
        on_event: Optional event callback for custom event handling.
            Called in addition to any progress tracker.
        derived_metrics: List of derived metric functions or import references.
            These are post-hoc computations that do NOT affect run fingerprints.
            Functions must be importable (not lambdas). Can be specified as:
            - Function references: "myproject.metrics:final_loss"
            - Callable functions: final_loss (must have __module__ and __name__)

    Returns:
        RunHandle for tracking and awaiting results.

    Example:
        # Sequential execution (default)
        handle = metalab.run(exp)
        results = handle.result()

        # Parallel with threads
        handle = metalab.run(exp, executor=metalab.ThreadExecutor(max_workers=4))

        # Parallel with processes (bypasses GIL)
        handle = metalab.run(exp, executor=metalab.ProcessExecutor(max_workers=4))

        # SLURM cluster execution
        handle = metalab.run(
            exp,
            store="/scratch/runs/my_exp",
            executor=metalab.SlurmExecutor(
                metalab.SlurmConfig(partition="gpu", time="2:00:00")
            ),
        )

        # With progress display
        handle = metalab.run(exp, progress=True)
        results = handle.result()  # Shows live progress bar

        # Custom event handling
        def my_callback(event):
            print(f"Event: {event.kind}")
        handle = metalab.run(exp, on_event=my_callback)

        # With derived metrics
        handle = metalab.run(exp, derived_metrics=[final_loss, convergence_stats])
        results = handle.result()  # Derived metrics computed and stored per-run

        # Cancel if needed
        handle.cancel()
    """
    from metalab.derived import get_func_ref
    from metalab.progress import Progress as ProgressConfig
    from metalab.progress import create_progress_tracker

    # Resolve store
    if store is None:
        store = f"./runs/{experiment.name}"
    if isinstance(store, str):
        store = FileStore(store)

    # Resolve executor (default: sequential execution)
    if executor is None:
        executor = ThreadExecutor(max_workers=1)

    # Convert derived_metrics to references
    derived_metric_refs: list[str] | None = None
    if derived_metrics:
        derived_metric_refs = []
        for metric in derived_metrics:
            if isinstance(metric, str):
                derived_metric_refs.append(metric)
            else:
                # Convert callable to reference
                derived_metric_refs.append(get_func_ref(metric))

    # Check if using SLURM executor with index-addressed arrays
    from metalab.executor.slurm import SlurmExecutor

    if isinstance(executor, SlurmExecutor):
        # Use index-addressed SLURM array submission
        handle = _run_slurm_indexed(
            experiment=experiment,
            store=store,
            executor=executor,
            resume=resume,
            derived_metric_refs=derived_metric_refs,
        )
        all_run_ids_count = len(experiment.params) * len(experiment.seeds)
    else:
        # Standard payload-based submission for other executors
        payloads, all_run_ids = generate_payloads(
            experiment, store, resume, derived_metric_refs=derived_metric_refs
        )
        all_run_ids_count = len(all_run_ids)

        # Submit to executor
        handle = executor.submit(
            payloads=payloads,
            store=store,
            operation=experiment.operation,
            run_ids=all_run_ids,
        )

    # Wire up on_event callback if provided (and no progress)
    if on_event is not None and progress is None:
        handle.set_event_callback(on_event)

    # Set up progress tracking if requested
    if progress is not None and progress is not False:
        # Determine progress configuration
        if progress is True:
            progress_config = ProgressConfig(title=experiment.name)
        else:
            progress_config = progress
            # Use experiment name as default title if not specified
            if progress_config.title is None:
                progress_config = ProgressConfig(
                    title=experiment.name,
                    style=progress_config.style,
                    display_metrics=progress_config.display_metrics,
                )

        # Create progress tracker
        tracker = create_progress_tracker(
            total=all_run_ids_count,
            title=progress_config.title or experiment.name,
            style=progress_config.style,
            display_metrics=progress_config.display_metrics,
        )

        # Wrap handle with progress management
        handle = ProgressRunHandle(handle, tracker)

        # If there's also an on_event callback, wire it up
        if on_event is not None:
            handle.set_event_callback(on_event)

    return handle


def load_results(
    path: str,
    experiment_id: str | None = None,
) -> Results:
    """
    Load results from a store path.

    Use this to load results from a previous experiment run.

    Args:
        path: Path to the store directory (e.g., "./runs/my_experiment").
        experiment_id: Optional filter by experiment ID.

    Returns:
        Results containing the loaded runs.

    Example:
        # Load all results from a store
        results = metalab.load_results("./runs/gene_perturbation")

        # Access runs
        for run in results:
            print(run.metrics)

        # Load artifact from a specific run
        artifact = results[0].artifact("summary")

        # Filter and export
        results.successful.to_csv("./successful_runs.csv")
    """
    store = FileStore(path)
    return Results.from_store(store, experiment_id=experiment_id)


def reconnect(
    path: str,
    on_event: "EventCallback | None" = None,
    progress: "bool | Progress | None" = None,
) -> RunHandle:
    """
    Reconnect to an in-progress or completed experiment.

    Use this to resume watching progress of a SLURM experiment from a new session,
    or to check status of an experiment that was submitted earlier.

    Args:
        path: Path to the store directory (e.g., "./runs/my_experiment").
        on_event: Optional event callback for custom event handling.
        progress: Enable progress display. Options:
            - True: Auto-detect rich, use simple fallback if not available.
            - False/None: No progress display.
            - Progress(...): Customized progress display with title, metrics, etc.

    Returns:
        A RunHandle that can be used to check status and wait for results.

    Raises:
        FileNotFoundError: If no manifest exists at the path.

    Example:
        # Reconnect with progress display
        handle = metalab.reconnect("./runs/my_exp", progress=True)
        results = handle.result()  # Shows live progress

        # Check current status without blocking
        handle = metalab.reconnect("./runs/my_exp")
        print(handle.status)  # RunStatus(total=100, completed=45, ...)

        # Custom progress configuration
        handle = metalab.reconnect(
            "./runs/my_exp",
            progress=metalab.Progress(
                title="Resuming Experiment",
                display_metrics=["best_f:.2f"],
            ),
        )
        results = handle.result()
    """
    from metalab.executor.slurm import SlurmRunHandle
    from metalab.progress import Progress as ProgressConfig
    from metalab.progress import create_progress_tracker

    store = FileStore(path)
    handle: RunHandle = SlurmRunHandle.from_store(
        store, on_event=on_event if progress is None else None
    )

    # Wire up on_event callback if provided (and no progress)
    if on_event is not None and progress is None:
        handle.set_event_callback(on_event)

    # Set up progress tracking if requested
    if progress is not None and progress is not False:
        # Determine progress configuration
        if progress is True:
            progress_config = ProgressConfig(title="Experiment")
        else:
            progress_config = progress

        # Get total from handle status
        status = handle.status

        # Create progress tracker
        tracker = create_progress_tracker(
            total=status.total,
            title=progress_config.title or "Experiment",
            style=progress_config.style,
            display_metrics=progress_config.display_metrics,
        )

        # Wrap handle with progress management
        handle = ProgressRunHandle(handle, tracker)

        # If there's also an on_event callback, wire it up
        if on_event is not None:
            handle.set_event_callback(on_event)

    return handle
