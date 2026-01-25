"""
Runner: Orchestrates experiment execution with resume/dedupe.

The Runner:
1. Generates run payloads from experiment configuration
2. Checks for existing runs (resume)
3. Submits new runs to executor
4. Collects results
5. Emits events
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from metalab.capture.output import OutputCapture
    from metalab.progress import Progress

from metalab._ids import (
    compute_run_id,
    fingerprint_context,
    fingerprint_params,
    fingerprint_seeds,
)
from metalab.context import DefaultContextBuilder
from metalab.events import Event, EventEmitter, EventKind
from metalab.executor.payload import RunPayload
from metalab.executor.thread import ThreadExecutor
from metalab.result import Results
from metalab.store.file import FileStore
from metalab.types import Status

if TYPE_CHECKING:
    from metalab.events import EventCallback
    from metalab.executor.base import Executor
    from metalab.experiment import Experiment
    from metalab.store.base import Store

logger = logging.getLogger(__name__)


class Runner:
    """
    Orchestrates experiment execution.

    Handles:
    - Payload generation from experiment config
    - Resume/dedupe (skip existing successful runs)
    - Event emission for progress tracking
    - Result collection
    """

    def __init__(
        self,
        store: Store,
        executor: Executor,
        resume: bool = True,
        progress: bool = False,
        on_event: EventCallback | None = None,
    ) -> None:
        """
        Initialize the runner.

        Args:
            store: Storage backend.
            executor: Execution backend.
            resume: Skip existing successful runs.
            progress: Emit progress events.
            on_event: Event callback.
        """
        self._store = store
        self._executor = executor
        self._resume = resume
        self._progress = progress
        self._emitter = EventEmitter(on_event)

    def run(self, experiment: Experiment) -> Results:
        """
        Run an experiment.

        Args:
            experiment: The experiment to run.

        Returns:
            Results for accessing runs and artifacts.
        """
        # Get context fingerprint
        ctx_fp = fingerprint_context(experiment.context)

        # Generate all payloads
        payloads = []
        all_run_ids = []

        # Iterate over params x seeds
        for param_case in experiment.params:
            # Resolve params if resolver is provided
            if experiment.param_resolver is not None:
                resolver = experiment.param_resolver
                # Support both protocol (with .resolve method) and plain callable
                if hasattr(resolver, "resolve"):
                    resolved_params = resolver.resolve(
                        {},  # context_meta - could be enhanced
                        param_case.params,
                    )
                else:
                    # Plain callable: (context_meta, params_raw) -> params_resolved
                    resolved_params = resolver({}, param_case.params)
            else:
                resolved_params = param_case.params

            params_fp = fingerprint_params(resolved_params)

            for seed_bundle in experiment.seeds:
                seed_fp = fingerprint_seeds(seed_bundle)

                # Compute run_id
                run_id = compute_run_id(
                    experiment_id=experiment.experiment_id,
                    context_fp=ctx_fp,
                    params_fp=params_fp,
                    seed_fp=seed_fp,
                    code_fp=experiment.operation.code_hash,
                )

                all_run_ids.append(run_id)

                # Check for resume
                if self._resume and self._store.run_exists(run_id):
                    existing = self._store.get_run_record(run_id)
                    if existing and existing.status == Status.SUCCESS:
                        self._emitter.run_skipped(run_id, "already exists (success)")
                        continue

                # Create payload
                payload = RunPayload(
                    run_id=run_id,
                    experiment_id=experiment.experiment_id,
                    context_spec=experiment.context,
                    params_resolved=resolved_params,
                    seed_bundle=seed_bundle,
                    store_locator=str(self._store.root) if hasattr(self._store, "root") else "",
                    operation_ref=experiment.operation.ref,
                    context_builder_ref=None,  # Use default
                )

                payloads.append(payload)

        # Report progress
        total_runs = len(all_run_ids)
        runs_to_execute = len(payloads)
        skipped = total_runs - runs_to_execute

        if self._progress:
            self._emitter.progress(
                current=0,
                total=total_runs,
                message=f"Starting {runs_to_execute} runs ({skipped} skipped)",
            )

        # Submit all payloads
        futures = []
        for payload in payloads:
            self._emitter.run_started(payload.run_id)
            future = self._executor.submit(payload)
            futures.append((payload.run_id, future))

        # Gather results
        records = []
        completed = skipped

        for run_id, future in futures:
            try:
                record = future.result()
                records.append(record)

                # Persist record
                self._store.put_run_record(record)

                # Emit event
                if record.status == Status.SUCCESS:
                    self._emitter.run_finished(
                        run_id,
                        duration_ms=record.duration_ms,
                        metrics=record.metrics,
                    )
                else:
                    error_msg = record.error.get("message", "") if record.error else ""
                    self._emitter.run_failed(run_id, error=error_msg)

            except Exception as e:
                logger.error(f"Failed to get result for run {run_id}: {e}")
                self._emitter.run_failed(run_id, error=str(e))

            completed += 1
            if self._progress:
                self._emitter.progress(current=completed, total=total_runs)

        # Load any existing records for completeness
        if self._resume:
            for run_id in all_run_ids:
                if not any(r.run_id == run_id for r in records):
                    existing = self._store.get_run_record(run_id)
                    if existing:
                        records.append(existing)

        return Results(store=self._store, records=records)


def run(
    experiment: Experiment,
    store: str | Store | None = None,
    executor: str | Executor = "threads",
    max_workers: int = 4,
    resume: bool = True,
    progress: bool | Progress = False,
    capture_output: "bool | OutputCapture | None" = None,
    on_event: Callable[[Event], None] | None = None,
) -> Results:
    """
    Run an experiment.

    This is the main entry point for executing experiments.

    Args:
        experiment: The experiment to run.
        store: Store path or instance. If None, defaults to "./runs/{experiment.name}".
        executor: Executor type or instance (default: "threads").
        max_workers: Number of workers for built-in executors.
        resume: Skip existing successful runs (default: True).
        progress: Progress display configuration. Can be:
            - False: No progress display (default)
            - True: Auto-detect best progress display
            - Progress(...): Custom progress configuration
        capture_output: Output capture configuration. Can be:
            - None/False: No output capture (default)
            - True: Auto-detect based on progress setting
            - OutputCapture(...): Custom capture configuration
        on_event: Optional event callback (in addition to progress tracker).

    Returns:
        Results for accessing runs and artifacts.

    Example:
        # Simple progress
        result = metalab.run(exp, progress=True)

        # Custom progress display
        result = metalab.run(
            exp,
            progress=metalab.Progress(
                title="Gene Perturbation",
                display_metrics=["gene", "perturbation_value:>8.0f"],
            ),
        )

        # Capture and suppress output (clean progress bar)
        result = metalab.run(
            exp,
            progress=True,
            capture_output=metalab.OutputCapture.suppress(),
        )

        # Capture output and route through console
        result = metalab.run(
            exp,
            progress=True,
            capture_output=True,  # Auto-detects: console mode with progress
        )

        # Access individual runs
        run = result[0]
        print(run.metrics)
        artifact = run.artifact("summary")

        # Export results
        result.to_csv("./output/results.csv")
    """
    from metalab.capture.output import OutputCapture, normalize_output_capture
    from metalab.progress import Progress as ProgressConfig
    from metalab.progress import create_progress_tracker

    # Resolve store - default to ./runs/{experiment.name} for clean organization
    if store is None:
        store = f"./runs/{experiment.name}"
    if isinstance(store, str):
        store = FileStore(store)

    # Resolve progress configuration
    progress_tracker = None
    emit_progress = False

    if progress is True:
        # Auto-detect best progress tracker
        progress_tracker = create_progress_tracker(
            total=0,  # Will be updated when we know total
            title=experiment.name,
            style="auto",
        )
        emit_progress = True
    elif isinstance(progress, ProgressConfig):
        # Use provided configuration
        progress_tracker = create_progress_tracker(
            total=0,
            title=progress.title or experiment.name,
            style=progress.style,
            display_metrics=progress.display_metrics,
        )
        emit_progress = True

    # Normalize output capture configuration
    output_capture_config = normalize_output_capture(
        capture_output,
        has_progress=emit_progress,
    )

    # Get console from progress tracker for output routing
    console = progress_tracker.get_console() if progress_tracker else None

    # Resolve executor
    if isinstance(executor, str):
        if executor == "threads":
            executor = ThreadExecutor(
                max_workers=max_workers,
                operation=experiment.operation,
                context_builder=experiment.context_builder or DefaultContextBuilder(),
                output_capture=output_capture_config,
                console=console,
            )
        elif executor == "processes":
            from metalab.executor.process import ProcessExecutor

            executor = ProcessExecutor(max_workers=max_workers)
        else:
            raise ValueError(f"Unknown executor type: {executor}")

    # Combine event handlers
    def combined_event_handler(event: Event) -> None:
        if progress_tracker is not None:
            progress_tracker(event)
        if on_event is not None:
            on_event(event)

    effective_on_event = combined_event_handler if (progress_tracker or on_event) else None

    # Create runner and run
    runner = Runner(
        store=store,
        executor=executor,
        resume=resume,
        progress=emit_progress,
        on_event=effective_on_event,
    )

    try:
        if progress_tracker is not None:
            with progress_tracker:
                return runner.run(experiment)
        else:
            return runner.run(experiment)
    finally:
        executor.shutdown()


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
