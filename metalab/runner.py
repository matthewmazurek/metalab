"""
Runner: Orchestrates experiment execution with resume/dedupe.

The Runner:

1. Generates run payloads from experiment configuration
2. Checks for existing runs (resume)
3. Submits to executor
4. Returns a RunHandle for tracking/awaiting results

Use `metalab observe STORE` to monitor live status from another terminal.
Pass `on_event=callback` for custom in-process event handling.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metalab.derived import DerivedMetricFn
    from metalab.events import EventCallback
    from metalab.executor.base import Executor
    from metalab.experiment import Experiment
    from metalab.store.base import Store
    from metalab.store.config import StoreConfig

from metalab._canonical import fingerprint
from metalab._ids import (
    compute_run_id,
    fingerprint_params,
    fingerprint_seeds,
    resolve_context,
)
from metalab.executor.base import ExperimentPlan, RunPlanEntry
from metalab.executor.handle import RunHandle
from metalab.executor.thread import ThreadExecutor
from metalab.result import Results
from metalab.store import (
    DEFAULT_STORE_ROOT,
    SupportsExperimentManifests,
    SupportsWorkingDirectory,
)

logger = logging.getLogger(__name__)

# The package-level logger: all metalab.* loggers propagate here.
_pkg_logger = logging.getLogger("metalab")


def _ensure_verbose_logging() -> None:
    """Attach a stderr handler to the ``metalab`` logger if none exists.

    Called by :func:`run` and :func:`reconnect` when ``verbose=True``
    (the default).  If the application has already configured a handler
    on the ``metalab`` logger (or any ancestor), this is a no-op so we
    never duplicate output.
    """
    if _pkg_logger.handlers or _pkg_logger.parent and _pkg_logger.parent.handlers:
        return  # user or framework already configured logging
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    _pkg_logger.addHandler(handler)
    _pkg_logger.setLevel(logging.INFO)


def _write_experiment_manifest(
    experiment: "Experiment",
    store: "Store",
    context_fingerprint: str,
    total_runs: int,
    run_ids: list[str] | None = None,
) -> None:
    """Write versioned experiment metadata to the store."""
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


def prepare_experiment_plan(
    experiment: Experiment,
    store: Store,
    resume: bool = True,
    persist_manifest: bool = True,
    derived_metric_refs: list[str] | None = None,
    job_id: str | None = None,
    executor_type: str = "local",
) -> ExperimentPlan:
    """Prepare an executor-agnostic experiment submission plan.

    The runner owns experiment planning policy: context resolution, deterministic
    run IDs, resume decisions, root manifest writes, and runner-side events.
    Executors own only submission mechanics.
    """
    # Resolve context - computes lazy hashes for FilePath/DirPath
    resolved_context, manifest = resolve_context(experiment.context)
    ctx_fp = fingerprint(resolved_context)

    # Optionally persist the resolved manifest for auditability
    if persist_manifest and manifest and isinstance(store, SupportsWorkingDirectory):
        import json

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

    # First pass: compute all run_ids and store resolved params/seeds.
    run_entries: list[RunPlanEntry] = []

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

            run_entries.append(
                RunPlanEntry(
                    run_id=run_id,
                    params_resolved=resolved_params,
                    seed_bundle=seed_bundle,
                    params_fingerprint=params_fp,
                    seed_fingerprint=seed_fp,
                )
            )

    job_id = job_id or f"job-{uuid.uuid4().hex[:8]}"
    all_run_ids = [entry.run_id for entry in run_entries]
    if hasattr(store, "write_planned_run_ids"):
        try:
            store.write_planned_run_ids(all_run_ids)
        except Exception as e:
            logger.warning(f"Failed to write planned run ids: {e}")
    statuses = (
        store.get_run_statuses(all_run_ids)
        if resume and hasattr(store, "get_run_statuses")
        else {}
    )
    run_entries = [
        RunPlanEntry(
            run_id=entry.run_id,
            params_resolved=entry.params_resolved,
            seed_bundle=entry.seed_bundle,
            params_fingerprint=entry.params_fingerprint,
            seed_fingerprint=entry.seed_fingerprint,
            status=statuses.get(entry.run_id),
        )
        for entry in run_entries
    ]

    if hasattr(store, "write_root_manifest"):
        try:
            store.write_root_manifest(
                {
                    "experiment_id": experiment.experiment_id,
                    "expected_run_count": len(all_run_ids),
                    "expected_run_ids_inline": False,
                    "expected_run_ids_path": "index/planned-runs/{prefix}.ndjson",
                    "executor_type": executor_type,
                    "job_id": job_id,
                    "created_at": datetime.now().isoformat(),
                    "context_fingerprint": ctx_fp,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to write root manifest: {e}")

    # Write compact experiment manifest; run IDs live in planned-runs sidecars.
    if persist_manifest:
        _write_experiment_manifest(experiment, store, ctx_fp, len(all_run_ids))

    plan = ExperimentPlan(
        experiment=experiment,
        store=store,
        context_fingerprint=ctx_fp,
        run_entries=run_entries,
        job_id=job_id,
        executor_type=executor_type,
        derived_metric_refs=derived_metric_refs,
        resolved_context_manifest=manifest or None,
    )

    runner_sink = None
    if hasattr(store, "event_sink"):
        try:
            runner_sink = store.event_sink(job_id, "runner", experiment.experiment_id)
            runner_sink.emit(
                "planned_batch",
                payload={
                    "count": len(all_run_ids),
                    "executor_type": executor_type,
                },
            )
        except Exception:
            runner_sink = None

    if runner_sink is not None:
        for entry in plan.run_entries:
            if entry.should_skip:
                try:
                    runner_sink.emit(
                        "skipped",
                        run_id=entry.run_id,
                        payload={
                            "reason": "already success",
                            "params": entry.params_resolved,
                        },
                    )
                except Exception:
                    pass

    return plan


def run(
    experiment: "Experiment",
    store: "str | StoreConfig | None" = None,
    executor: "Executor | None" = None,
    resume: bool = True,
    on_event: "EventCallback | None" = None,
    derived_metrics: "list[str | DerivedMetricFn] | None" = None,
    verbose: bool = True,
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
            - str: Parse as filesystem run-store path/locator
            - StoreConfig: Connect directly; no experiment subdirectory is added
        executor: Executor instance. Defaults to sequential (single-threaded).
            For parallel execution, use:
            - ThreadExecutor(max_workers=N) for thread-based parallelism
            - ProcessExecutor(max_workers=N) for process-based parallelism
            - SlurmExecutor(...) for cluster execution
        resume: Skip existing successful runs (default: True).
        on_event: Optional event callback for custom event handling.
        derived_metrics: List of derived metric functions or import references.
            These are post-hoc computations that do NOT affect run fingerprints.
            Functions must be importable (not lambdas). Can be specified as:
            - Function references: "myproject.metrics:final_loss"
            - Callable functions: final_loss (must have __module__ and __name__)
        verbose: Log resolved store, executor, and run counts to stderr
            (default: True).  Automatically configures a handler on the
            ``metalab`` logger if none exists.  Set to ``False`` to silence.

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

    # Custom event handling
    def my_callback(event):
        print(f"Event: {event.kind}")
    handle = metalab.run(exp, on_event=my_callback)

    # With derived metrics
    handle = metalab.run(exp, derived_metrics=[final_loss, convergence_stats])
    results = handle.result()  # Derived metrics computed and stored per-run

    # With StoreConfig (pre-configured)
    config = FileStoreConfig(root="./experiments")
    handle = metalab.run(exp, store=config)

    # Cancel if needed
    handle.cancel()
    ```
    """
    if verbose:
        _ensure_verbose_logging()

    from metalab.derived import get_func_ref
    from metalab.store.locator import parse_to_config

    # Resolve store to config, then scope and connect
    # Default: {DEFAULT_STORE_ROOT}/{safe_experiment_id}/ via collection-scoped storage
    if store is None:
        store = DEFAULT_STORE_ROOT

    if isinstance(store, str):
        config = parse_to_config(store)
    else:
        config = store

    # The provided path is the run store. Clean-break v2 stores do not
    # auto-scope into experiment subdirectories.
    resolved_store: "Store" = config.connect()
    store_label = type(resolved_store).__name__
    if isinstance(resolved_store, SupportsWorkingDirectory):
        store_label = f"{store_label} at {resolved_store.get_working_directory()}"
    logger.info("Store: %s (experiment: %s)", store_label, experiment.experiment_id)

    # Resolve executor (default: sequential execution)
    if executor is None:
        executor = ThreadExecutor(max_workers=1)
        logger.info("Executor: sequential (default)")
    else:
        logger.info("Executor: %s", type(executor).__name__)

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

    plan = prepare_experiment_plan(
        experiment=experiment,
        store=resolved_store,
        resume=resume,
        derived_metric_refs=derived_metric_refs,
        job_id=f"job-{uuid.uuid4().hex[:8]}",
        executor_type=type(executor).__name__,
    )
    if plan.skipped_count > 0:
        logger.info(
            "Runs: %d total, %d already completed (resume), %d to execute",
            plan.total_runs,
            plan.skipped_count,
            len(plan.pending_entries),
        )
    else:
        logger.info("Runs: %d to execute", plan.total_runs)

    handle = executor.submit_experiment(plan)

    # Wire up on_event callback if provided.
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
    verbose: bool = True,
    **kwargs,
) -> RunHandle:
    """
    Reconnect to an in-flight or completed experiment.

    Use this to resume access to a SLURM experiment from a new session,
    or to check status of an experiment that was submitted earlier.

    Note: This function only supports async executors (SLURM, etc.) where jobs
    may still be running. For loading results from local executor experiments,
    use `load_results()` instead.

    Args:
        store: Store locator string or StoreConfig. Supports:
            - Path string: "./runs/my_experiment"
            - File URI: "file:///scratch/runs/my_exp"
            - StoreConfig instance
        on_event: Optional event callback for custom event handling.
        verbose: Log resolved store and executor info to stderr
            (default: True).  See :func:`run` for details.
        **kwargs: Additional arguments passed to store config.

    Returns:
        A RunHandle that can be used to check status and wait for results.

    Raises:
        FileNotFoundError: If no manifest exists at the store.
        ValueError: If the executor type doesn't support reconnection.

    Example:
    ```python
    # Check current status without blocking
    handle = metalab.reconnect("./runs/my_exp")
    print(handle.status)  # RunStatus(total=100, completed=45, ...)

    # For live monitoring, use: metalab observe ./runs/my_exp
    ```
    """
    if verbose:
        _ensure_verbose_logging()

    from metalab.executor.registry import HandleRegistry
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

    handle: RunHandle = handle_class.from_store(store_instance, on_event=on_event)

    # Wire up on_event callback if provided.
    if on_event is not None:
        handle.set_event_callback(on_event)

    return handle
