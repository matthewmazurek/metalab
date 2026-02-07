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
    from metalab.store.config import StoreConfig

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
from metalab.store import (
    DEFAULT_STORE_ROOT,
    SupportsExperimentManifests,
    SupportsWorkingDirectory,
)
from metalab.types import Status

logger = logging.getLogger(__name__)


def _write_experiment_manifest(
    experiment: "Experiment",
    store: "Store",
    context_fingerprint: str,
    total_runs: int,
    run_ids: list[str] | None = None,
) -> None:
    """Write experiment manifest to store for Atlas."""
    from datetime import datetime

    from metalab.manifest import build_experiment_manifest

    exp_manifest = build_experiment_manifest(
        experiment=experiment,
        context_fingerprint=context_fingerprint,
        total_runs=total_runs,
        run_ids=run_ids,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if isinstance(store, SupportsExperimentManifests):
        store.put_experiment_manifest(
            experiment.experiment_id,
            exp_manifest,
            timestamp=timestamp,
        )
        logger.debug(f"Saved experiment manifest: {experiment.experiment_id}")


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
    def store(self) -> "Store":
        """The store used for this execution."""
        return self._handle.store

    @property
    def status(self) -> RunStatus:
        """Current status of all runs (non-blocking)."""
        return self._handle.status

    @property
    def is_complete(self) -> bool:
        """True if all runs have finished (success or failure)."""
        return self._handle.is_complete

    @property
    def can_reconnect(self) -> bool:
        """Delegate to inner handle."""
        return self._handle.can_reconnect

    @property
    def reconnect_locator(self) -> str | None:
        """Delegate to inner handle."""
        return self._handle.reconnect_locator

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

    def __enter__(self) -> "ProgressRunHandle":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager, ensuring tracker is stopped."""
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
    if persist_manifest and manifest and isinstance(store, SupportsWorkingDirectory):
        import json
        from pathlib import Path

        manifest_path = (
            store.get_working_directory()
            / f"{experiment.experiment_id}_context_manifest.json"
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
    if persist_manifest:
        _write_experiment_manifest(
            experiment, store, ctx_fp, len(all_run_ids), all_run_ids
        )

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
            store_locator=store.config.to_dict(),
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
    if manifest and isinstance(store, SupportsWorkingDirectory):
        import json
        from pathlib import Path

        manifest_path = (
            store.get_working_directory()
            / f"{experiment.experiment_id}_context_manifest.json"
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

    # Write experiment manifest for Atlas
    _write_experiment_manifest(experiment, store, ctx_fp, total_runs)

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
    experiment: "Experiment",
    store: "str | StoreConfig | None" = None,
    executor: "Executor | None" = None,
    file_root: str | None = None,
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
        store: Where to store results. Can be:
            - None: Default to "./experiments"
            - str: Parse as locator, auto-scope to experiment
            - StoreConfig: Scope to experiment and connect
        executor: Executor instance. Defaults to sequential (single-threaded).
            For parallel execution, use:
            - ThreadExecutor(max_workers=N) for thread-based parallelism
            - ProcessExecutor(max_workers=N) for process-based parallelism
            - SlurmExecutor(...) for cluster execution
        file_root: Optional filesystem root for artifact/log storage when
            *store* is a locator string (e.g. a Postgres DSN).  Passed through
            to ``parse_to_config(store, file_root=file_root)``.  Ignored when
            *store* is already a :class:`StoreConfig` or ``None``.
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
    ```python
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

    # With StoreConfig (pre-configured)
    config = FileStoreConfig(root="./experiments")
    handle = metalab.run(exp, store=config)  # auto-scopes to experiment

    # With PostgresStore (requires file_root for logs/artifacts)
    handle = metalab.run(
        exp,
        store="postgresql://localhost/db?file_root=/path/to/files",
    )

    # Cancel if needed
    handle.cancel()
    ```
    """
    from metalab.derived import get_func_ref
    from metalab.progress import Progress as ProgressConfig
    from metalab.progress import create_progress_tracker
    from metalab.store.config import StoreConfig
    from metalab.store.locator import parse_to_config

    # Resolve store to config, then scope and connect
    # Default: {DEFAULT_STORE_ROOT}/{safe_experiment_id}/ via collection-scoped storage
    if store is None:
        store = DEFAULT_STORE_ROOT

    if isinstance(store, str):
        config = (
            parse_to_config(store, file_root=file_root)
            if file_root
            else parse_to_config(store)
        )
    else:
        config = store

    # Scope to experiment and connect
    resolved_store: "Store" = config.scoped(experiment.experiment_id).connect()

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
            store=resolved_store,
            executor=executor,
            resume=resume,
            derived_metric_refs=derived_metric_refs,
        )
        all_run_ids_count = len(experiment.params) * len(experiment.seeds)
    else:
        # Standard payload-based submission for other executors
        payloads, all_run_ids = generate_payloads(
            experiment, resolved_store, resume, derived_metric_refs=derived_metric_refs
        )
        all_run_ids_count = len(all_run_ids)

        # Submit to executor
        handle = executor.submit(
            payloads=payloads,
            store=resolved_store,
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
    store: "str | StoreConfig",
    experiment_id: str | None = None,
) -> Results:
    """
    Load results from a store.

    Use this to load results from a previous experiment run.

    Args:
        store: Store path or StoreConfig.
        experiment_id: Optional filter by experiment ID.

    Returns:
        Results containing the loaded runs.

    Note:
        Does NOT auto-scope. Pass a scoped config or filter by experiment_id.

    Example:
    ```python
    # Load all results from a store path
    results = metalab.load_results("./runs/gene_perturbation")

    # Access runs
    for run in results:
        print(run.metrics)

    # Load artifact from a specific run
    artifact = results[0].artifact("summary")

    # Filter and export
    results.successful.to_csv("./successful_runs.csv")

    # Load with StoreConfig
    config = FileStoreConfig(root="./experiments", experiment_id="my_exp:1.0")
    results = metalab.load_results(config)
    ```
    """
    from metalab.store.config import StoreConfig
    from metalab.store.locator import parse_to_config

    if isinstance(store, str):
        config = parse_to_config(store)
    else:
        config = store

    resolved_store = config.connect()
    return Results.from_store(resolved_store, experiment_id=experiment_id)


# Local executor types that don't support reconnection
_LOCAL_EXECUTOR_TYPES = {"local", "thread", "process"}


def _load_manifest(store: "Store") -> dict:
    """
    Load the experiment manifest from a store.

    Args:
        store: Store instance with working directory support.

    Returns:
        The manifest dictionary.

    Raises:
        FileNotFoundError: If no manifest exists.
        TypeError: If the store doesn't support working directory.
    """
    import json

    if not isinstance(store, SupportsWorkingDirectory):
        raise TypeError(
            f"Cannot load manifest from store type {type(store).__name__}. "
            f"Store must support SupportsWorkingDirectory capability."
        )

    store_path = store.get_working_directory()
    manifest_path = store_path / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest found at {manifest_path}. "
            "Cannot reconnect without a manifest."
        )

    with open(manifest_path) as f:
        return json.load(f)


def reconnect(
    store: "str | StoreConfig",
    on_event: "EventCallback | None" = None,
    progress: "bool | Progress | None" = None,
    **kwargs,
) -> RunHandle:
    """
    Reconnect to an in-progress or completed experiment.

    Use this to resume watching progress of a SLURM experiment from a new session,
    or to check status of an experiment that was submitted earlier.

    Note: This function only supports async executors (SLURM, etc.) where jobs
    may still be running. For loading results from local executor experiments,
    use `load_results()` instead.

    Args:
        store: Store locator string or StoreConfig. Supports:
            - Path string: "./runs/my_experiment"
            - File URI: "file:///scratch/runs/my_exp"
            - Postgres URI: "postgresql://localhost/metalab" (requires file_root)
            - StoreConfig instance
        on_event: Optional event callback for custom event handling.
        progress: Enable progress display. Options:
            - True: Auto-detect rich, use simple fallback if not available.
            - False/None: No progress display.
            - Progress(...): Customized progress display with title, metrics, etc.
        **kwargs: Additional arguments passed to store config (e.g., file_root
            for Postgres stores).

    Returns:
        A RunHandle that can be used to check status and wait for results.

    Raises:
        FileNotFoundError: If no manifest exists at the store.
        ValueError: If the executor type doesn't support reconnection.

    Example:
    ```python
    # Reconnect with progress display (file store)
    handle = metalab.reconnect("./runs/my_exp", progress=True)
    results = handle.result()  # Shows live progress

    # Check current status without blocking
    handle = metalab.reconnect("./runs/my_exp")
    print(handle.status)  # RunStatus(total=100, completed=45, ...)

    # Reconnect with Postgres store
    handle = metalab.reconnect(
        "postgresql://localhost/metalab",
        file_root="/scratch/artifacts",
        progress=True,
    )

    # Custom progress configuration
    handle = metalab.reconnect(
        "./runs/my_exp",
        progress=metalab.Progress(
            title="Resuming Experiment",
            display_metrics=["best_f:.2f"],
        ),
    )
    results = handle.result()
    ```
    """
    from metalab.executor.registry import HandleRegistry
    from metalab.progress import Progress as ProgressConfig
    from metalab.progress import create_progress_tracker
    from metalab.store.config import StoreConfig
    from metalab.store.locator import parse_to_config

    # 1. Resolve store config
    if isinstance(store, str):
        config = parse_to_config(store, **kwargs)
    else:
        config = store

    store_instance = config.connect()

    # 2. Load manifest and get executor_type
    manifest = _load_manifest(store_instance)
    executor_type = manifest.get("executor_type")

    # 3. Reject local executors with helpful error
    if executor_type in _LOCAL_EXECUTOR_TYPES:
        raise ValueError(
            f"Cannot reconnect to '{executor_type}' executor - local runs are synchronous. "
            f"Use metalab.load_results() to retrieve completed results."
        )

    # 4. Dispatch to registered handle
    handle_class = HandleRegistry.get(executor_type)
    if handle_class is None:
        raise ValueError(
            f"No reconnectable handle registered for executor type '{executor_type}'. "
            f"Supported types: {HandleRegistry.types()}"
        )

    handle: RunHandle = handle_class.from_store(
        store_instance, on_event=on_event if progress is None else None
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
