"""
SlurmExecutor: SLURM cluster execution via direct sbatch submission.

Provides:
- SlurmConfig: Configuration for SLURM job parameters
- SlurmExecutor: Executor that submits index-addressed SLURM arrays
- SlurmRunHandle: Handle for tracking SLURM job execution

This implementation uses index-addressed SLURM arrays where each array task
computes its parameters from SLURM_ARRAY_TASK_ID, avoiding per-task
serialization overhead.

Job state tracking:
- Uses squeue for active job counts (RUNNING, PENDING)
- Uses sacct for terminal job counts (COMPLETED, FAILED, etc.)
- Falls back to store-only polling if SLURM commands unavailable
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from metalab.executor.handle import RunStatus
from metalab.executor.payload import RunPayload
from metalab.result import Results

if TYPE_CHECKING:
    from metalab.events import EventCallback
    from metalab.experiment import Experiment
    from metalab.operation import OperationWrapper
    from metalab.store.base import Store

from metalab.store.capabilities import SupportsWorkingDirectory


def _get_store_path(store: "Store") -> Path:
    """
    Get the filesystem working directory from a store.

    Args:
        store: The store instance.

    Returns:
        Path to the store's working directory.

    Raises:
        TypeError: If the store doesn't support SupportsWorkingDirectory.
    """
    if isinstance(store, SupportsWorkingDirectory):
        return store.get_working_directory()
    raise TypeError(
        f"SLURM executor requires a store that supports filesystem access "
        f"(SupportsWorkingDirectory capability). Got {type(store).__name__}. "
        f"Use FileStore or a store with get_working_directory() method."
    )


logger = logging.getLogger(__name__)

# Default maximum array size (many clusters limit this)
DEFAULT_MAX_ARRAY_SIZE = 10000


def _serialize_context_spec(obj: Any) -> Any:
    """
    Serialize a context spec to a JSON-compatible structure with type preservation.

    Handles dataclasses (FilePath, DirPath, context_spec decorated classes),
    dicts, lists, and primitives. Type information is preserved via __type__ keys.

    Args:
        obj: The context spec object to serialize.

    Returns:
        A JSON-serializable structure.
    """
    import dataclasses

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialize_context_spec(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize_context_spec(v) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Preserve type information for reconstruction
        type_name = type(obj).__module__ + "." + type(obj).__qualname__
        fields = {
            f.name: _serialize_context_spec(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
        return {"__type__": type_name, **fields}

    # Fallback: convert to string
    return str(obj)


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
        max_array_size: Maximum tasks per array job (for sharding).
        chunk_size: Number of runs per array task. Higher values reduce
            scheduler load for large experiments (e.g., 100k runs with
            chunk_size=100 submits 1k array tasks instead of 100k).
        modules: Shell modules to load before execution.
        conda_env: Conda environment to activate.
        setup: List of bash commands to run before each task.
        extra_sbatch: Additional sbatch directives as key-value pairs.
    """

    partition: str = "default"
    time: str = "1:00:00"
    cpus: int = 1
    memory: str = "4G"
    gpus: int = 0
    max_concurrent: int | None = None
    max_array_size: int = DEFAULT_MAX_ARRAY_SIZE
    chunk_size: int = 1
    modules: list[str] = field(default_factory=list)
    conda_env: str | None = None
    setup: list[str] = field(default_factory=list)
    extra_sbatch: dict[str, str] = field(default_factory=dict)

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


# ---------------------------------------------------------------------------
# SLURM command utilities
# ---------------------------------------------------------------------------


def _run_cmd(cmd: list[str], timeout: float = 30.0) -> tuple[int, str, str]:
    """
    Run a command with timeout.

    Args:
        cmd: Command and arguments as list.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (exit_code, stdout, stderr).
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def _parse_slurm_job_id(output: str) -> str | None:
    """
    Parse job ID from sbatch output.

    Expected format: "Submitted batch job 12345"
    """
    for line in output.strip().split("\n"):
        if "Submitted batch job" in line:
            parts = line.split()
            if parts:
                return parts[-1]
    return None


def _normalize_slurm_state(state: str) -> str:
    """
    Normalize SLURM state string.

    - Strips trailing '+' (e.g., CANCELLED+ -> CANCELLED)
    - Uppercases for consistency
    """
    return state.upper().rstrip("+")


def _is_step_job(job_id: str) -> bool:
    """Check if job ID is a step (e.g., 12345.batch, 12345.extern)."""
    return "." in job_id and any(
        job_id.endswith(suffix) for suffix in [".batch", ".extern", ".0"]
    )


# ---------------------------------------------------------------------------
# SLURM polling helpers (can be used standalone for remote polling)
# ---------------------------------------------------------------------------


def squeue_state_counts(
    job_ids: list[str],
    timeout: float = 30.0,
) -> dict[str, int]:
    """
    Query squeue for active job state counts.

    Args:
        job_ids: List of SLURM job IDs to query.
        timeout: Command timeout in seconds.

    Returns:
        Dict mapping state to count (e.g., {"RUNNING": 5, "PENDING": 10}).
        Returns empty dict if squeue fails.

    Example:
        >>> counts = squeue_state_counts(["12345", "12346"])
        >>> counts
        {"RUNNING": 50, "PENDING": 100, "COMPLETING": 5}

    Notes:
        - Ignores .batch, .extern, and other step jobs
        - Returns empty dict if squeue fails (cluster unavailable, etc.)
    """
    if not job_ids:
        return {}

    # Query all job IDs at once
    job_list = ",".join(job_ids)
    exit_code, stdout, stderr = _run_cmd(
        ["squeue", "-h", "-j", job_list, "-o", "%i %T"],
        timeout=timeout,
    )

    if exit_code != 0:
        logger.debug(f"squeue failed: {stderr}")
        return {}

    counts: dict[str, int] = {}
    for line in stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            job_id, state = parts[0], parts[1]
            # Skip step jobs
            if _is_step_job(job_id):
                continue
            state = _normalize_slurm_state(state)
            counts[state] = counts.get(state, 0) + 1

    return counts


def sacct_state_counts(
    job_ids: list[str],
    timeout: float = 60.0,
) -> dict[str, int]:
    """
    Query sacct for terminal job state counts.

    Args:
        job_ids: List of SLURM job IDs to query.
        timeout: Command timeout in seconds.

    Returns:
        Dict mapping state to count (e.g., {"COMPLETED": 100, "FAILED": 5}).
        Returns empty dict if sacct fails.

    Example:
        >>> counts = sacct_state_counts(["12345", "12346"])
        >>> counts
        {"COMPLETED": 95, "FAILED": 3, "CANCELLED": 2}

    Notes:
        - Uses parsable output format for reliable parsing
        - Normalizes states by stripping '+' suffix
        - Ignores .batch, .extern, and other step jobs
        - Streams output to avoid memory issues with large arrays
    """
    if not job_ids:
        return {}

    job_list = ",".join(job_ids)
    exit_code, stdout, stderr = _run_cmd(
        ["sacct", "-n", "-P", "-j", job_list, "--format=JobIDRaw,State"],
        timeout=timeout,
    )

    if exit_code != 0:
        logger.debug(f"sacct failed: {stderr}")
        return {}

    counts: dict[str, int] = {}
    for line in stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) >= 2:
            job_id, state = parts[0], parts[1]
            # Skip step jobs
            if _is_step_job(job_id):
                continue
            state = _normalize_slurm_state(state)
            counts[state] = counts.get(state, 0) + 1

    return counts


def get_slurm_status_counts(
    job_ids: list[str],
    total_runs: int,
    sacct_cache: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    Get comprehensive SLURM status counts with explicit state buckets.

    This function combines squeue (for active jobs) and sacct (for terminal jobs)
    to provide a complete picture of job states.

    Args:
        job_ids: List of SLURM job IDs to query.
        total_runs: Total number of runs in the experiment.
        sacct_cache: Optional cached sacct counts to avoid repeated queries.

    Returns:
        Dict with explicit state buckets:
        - running: Tasks currently executing (RUNNING state)
        - pending: Tasks waiting to start (PENDING state)
        - completing: Tasks in transition (COMPLETING state)
        - completed: Successfully finished tasks (COMPLETED state)
        - failed: Tasks that failed (FAILED state)
        - cancelled: Cancelled tasks (CANCELLED state)
        - timeout: Tasks that timed out (TIMEOUT state)
        - oom: Out of memory tasks (OUT_OF_MEMORY state)
        - other: All other states (HELD, SUSPENDED, REQUEUED, etc.)
        - total: Total runs

    Notes:
        - Running/pending counts come from squeue (accurate for active jobs)
        - Terminal counts come from sacct (may have accounting lag)
        - "other" bucket catches transitional states that don't fit elsewhere
    """
    # Get active counts from squeue
    squeue_counts = squeue_state_counts(job_ids)

    # Get terminal counts from sacct (use cache if provided)
    if sacct_cache is not None:
        terminal_counts = sacct_cache
    else:
        terminal_counts = sacct_state_counts(job_ids)

    # Extract explicit buckets
    running = squeue_counts.get("RUNNING", 0)
    pending = squeue_counts.get("PENDING", 0)
    completing = squeue_counts.get("COMPLETING", 0)

    completed = terminal_counts.get("COMPLETED", 0)
    failed = terminal_counts.get("FAILED", 0)
    cancelled = terminal_counts.get("CANCELLED", 0)
    timeout = terminal_counts.get("TIMEOUT", 0)
    oom = terminal_counts.get("OUT_OF_MEMORY", 0)

    # Count other states
    known_states = {
        "RUNNING",
        "PENDING",
        "COMPLETING",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
    }

    other_from_squeue = sum(
        count for state, count in squeue_counts.items() if state not in known_states
    )
    other_from_sacct = sum(
        count for state, count in terminal_counts.items() if state not in known_states
    )

    # Calculate other as: total - accounted
    accounted = (
        running + pending + completing + completed + failed + cancelled + timeout + oom
    )

    # Add transitional states from squeue to 'other'
    other = max(0, total_runs - accounted) + other_from_squeue + other_from_sacct

    return {
        "running": running,
        "pending": pending,
        "completing": completing,
        "completed": completed,
        "failed": failed,
        "cancelled": cancelled,
        "timeout": timeout,
        "oom": oom,
        "other": other,
        "total": total_runs,
    }


# ---------------------------------------------------------------------------
# sbatch script generation
# ---------------------------------------------------------------------------


def _generate_sbatch_script(
    config: SlurmConfig,
    store_root: str,
    array_range: str,
    logs_dir: str,
    job_name: str,
    shard_offset: int = 0,
) -> str:
    """
    Generate sbatch script content.

    Args:
        config: SLURM configuration.
        store_root: Path to the store root.
        array_range: Array range string (e.g., "0-999" or "0-999%100").
        logs_dir: Directory for stdout/stderr logs.
        job_name: Job name for SLURM.
        shard_offset: Offset to add to SLURM_ARRAY_TASK_ID for global index.

    Returns:
        Complete sbatch script as string.
    """
    lines = ["#!/bin/bash"]

    # SBATCH directives
    lines.append(f"#SBATCH --job-name={job_name}")
    lines.append(f"#SBATCH --partition={config.partition}")
    lines.append(f"#SBATCH --time={config.time}")
    lines.append(f"#SBATCH --cpus-per-task={config.cpus}")
    lines.append(f"#SBATCH --mem={config.memory}")
    lines.append(f"#SBATCH --array={array_range}")
    lines.append(f"#SBATCH --output={logs_dir}/%A_%a.out")
    lines.append(f"#SBATCH --error={logs_dir}/%A_%a.err")

    if config.gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{config.gpus}")

    # Extra sbatch directives
    for key, value in config.extra_sbatch.items():
        # Convert underscores to hyphens for SLURM compatibility
        slurm_key = key.replace("_", "-")
        lines.append(f"#SBATCH --{slurm_key}={value}")

    lines.append("")

    # Set shard offset environment variable
    lines.append("# Shard offset for global task index computation")
    lines.append(f"export METALAB_SHARD_OFFSET={shard_offset}")
    lines.append("")

    # Setup commands
    if config.modules:
        for module in config.modules:
            lines.append(f"module load {module}")

    if config.conda_env:
        lines.append(f"conda activate {config.conda_env}")

    if config.setup:
        for cmd in config.setup:
            lines.append(cmd)

    if config.modules or config.conda_env or config.setup:
        lines.append("")

    # Worker invocation
    lines.append("# Run the metalab array worker")
    lines.append(
        f'python -m metalab.executor.slurm_array_worker --store "{store_root}"'
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Array spec file
# ---------------------------------------------------------------------------


def _write_array_spec(
    store_root: Path,
    experiment: "Experiment",
    context_fingerprint: str,
    shards: list[dict[str, Any]],
    chunk_size: int,
) -> None:
    """
    Write the array spec file that workers use to reconstruct runs.

    The context spec is serialized to JSON with type information preserved
    for dataclasses (FilePath, DirPath, context_spec decorated classes).

    Args:
        store_root: Path to the store root.
        experiment: The experiment being run.
        context_fingerprint: Precomputed context fingerprint.
        shards: List of shard info dicts with job_id, start_idx, end_idx.
        chunk_size: Number of runs per array task.
    """
    from metalab.manifest import serialize

    total_runs = len(experiment.params) * len(experiment.seeds)  # type: ignore[arg-type]

    # Serialize context spec to JSON with type preservation
    context_json_path = store_root / "context_spec.json"
    with open(context_json_path, "w") as f:
        json.dump(_serialize_context_spec(experiment.context), f, indent=2)

    spec = {
        "experiment_id": experiment.experiment_id,
        "operation_ref": experiment.operation.ref,
        "operation_code_hash": experiment.operation.code_hash,
        "context_fingerprint": context_fingerprint,
        "params": serialize(experiment.params),
        "seeds": serialize(experiment.seeds),
        "metadata": experiment.metadata,
        "param_cases": len(experiment.params),  # type: ignore[arg-type]
        "seed_replicates": len(experiment.seeds),  # type: ignore[arg-type]
        "total_runs": total_runs,
        "chunk_size": chunk_size,
        "total_chunks": (total_runs + chunk_size - 1) // chunk_size,
        "shards": shards,
        "derived_metric_refs": None,  # Set later if needed
    }

    spec_path = store_root / "slurm_array_spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# SlurmExecutor
# ---------------------------------------------------------------------------


class SlurmExecutor:
    """
    Executor that submits index-addressed SLURM arrays.

    Instead of serializing each task individually (like submitit's map_array),
    this executor:
    1. Writes a single array spec file with experiment configuration
    2. Submits one or more SLURM array jobs via sbatch
    3. Each array task reconstructs its parameters from SLURM_ARRAY_TASK_ID

    This approach scales to hundreds of thousands of tasks without the
    filesystem overhead of per-task serialization files.
    """

    def __init__(self, config: SlurmConfig | None = None) -> None:
        """
        Initialize the SLURM executor.

        Args:
            config: SLURM configuration. Uses defaults if not provided.
        """
        self._config = config or SlurmConfig()

    def submit_indexed(
        self,
        experiment: "Experiment",
        store: "Store",
        context_fingerprint: str,
        total_runs: int,
        skipped_count: int = 0,
        derived_metric_refs: list[str] | None = None,
    ) -> "SlurmRunHandle":
        """
        Submit an experiment as index-addressed SLURM arrays.

        Args:
            experiment: The experiment to run.
            store: Store for persisting results.
            context_fingerprint: Precomputed context fingerprint.
            total_runs: Total number of runs (P * R).
            skipped_count: Number of runs already completed (for resume).
            derived_metric_refs: Optional derived metric function references.

        Returns:
            A SlurmRunHandle for tracking and awaiting results.

        Raises:
            ValueError: If param source doesn't support indexing.
            RuntimeError: If sbatch submission fails.
        """
        # Validate that param source supports indexing
        if not hasattr(experiment.params, "__getitem__"):
            raise ValueError(
                f"SLURM array submission requires an indexable param source. "
                f"Got {type(experiment.params).__name__} which doesn't support __getitem__. "
                f"Use grid() or manual() instead of random()."
            )

        store_path = _get_store_path(store)

        # Create logs directory
        logs_dir = store_path / "slurm_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Compute chunking: each array task processes chunk_size runs
        chunk_size = self._config.chunk_size
        total_chunks = (total_runs + chunk_size - 1) // chunk_size

        # Shard by chunks (not runs)
        shards = self._compute_shards(total_chunks)

        # Get param/seed counts with type safety
        param_cases = len(experiment.params) if hasattr(experiment.params, "__len__") else 0  # type: ignore[arg-type]
        seed_replicates = len(experiment.seeds) if hasattr(experiment.seeds, "__len__") else 0  # type: ignore[arg-type]

        # Write array spec (before submission so workers can read it)
        _write_array_spec(
            store_root=store_path,
            experiment=experiment,
            context_fingerprint=context_fingerprint,
            shards=shards,
            chunk_size=chunk_size,
        )

        # Update spec with derived metrics if provided
        if derived_metric_refs:
            spec_path = store_path / "slurm_array_spec.json"
            with open(spec_path) as f:
                spec = json.load(f)
            spec["derived_metric_refs"] = derived_metric_refs
            with open(spec_path, "w") as f:
                json.dump(spec, f, indent=2)

        # Submit each shard
        job_ids: list[str] = []
        for shard in shards:
            job_id = self._submit_shard(
                config=self._config,
                store_root=str(store_path),
                shard=shard,
                logs_dir=str(logs_dir),
                job_name=f"metalab-{experiment.name}",
            )
            shard["job_id"] = job_id
            job_ids.append(job_id)

        # Update array spec with job IDs
        spec_path = store_path / "slurm_array_spec.json"
        with open(spec_path) as f:
            spec = json.load(f)
        spec["shards"] = shards
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2)

        # Write manifest
        self._write_manifest(
            store=store,
            experiment_id=experiment.experiment_id,
            job_ids=job_ids,
            shards=shards,
            total_runs=total_runs,
            total_chunks=total_chunks,
            chunk_size=chunk_size,
            param_cases=param_cases,
            seed_replicates=seed_replicates,
            skipped_count=skipped_count,
        )

        return SlurmRunHandle(
            store=store,
            job_ids=job_ids,
            shards=shards,
            total_runs=total_runs,
            chunk_size=chunk_size,
            skipped_count=skipped_count,
        )

    def _compute_shards(self, total_items: int) -> list[dict[str, Any]]:
        """
        Compute shard ranges for array submission.

        Args:
            total_items: Total number of array tasks (chunks) to submit.

        Returns:
            List of shard dicts with start_idx, end_idx, array_range.
        """
        max_size = self._config.max_array_size
        shards = []

        start_idx = 0
        while start_idx < total_items:
            end_idx = min(start_idx + max_size - 1, total_items - 1)
            shard_size = end_idx - start_idx + 1

            # Array range for sbatch (relative to shard, 0-based)
            array_end = shard_size - 1
            if self._config.max_concurrent:
                array_range = f"0-{array_end}%{self._config.max_concurrent}"
            else:
                array_range = f"0-{array_end}"

            shards.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "array_range": array_range,
                    "job_id": None,  # Set after submission
                }
            )

            start_idx = end_idx + 1

        return shards

    def _submit_shard(
        self,
        config: SlurmConfig,
        store_root: str,
        shard: dict[str, Any],
        logs_dir: str,
        job_name: str,
    ) -> str:
        """
        Submit a single shard as a SLURM array job.

        Args:
            config: SLURM configuration.
            store_root: Path to store root.
            shard: Shard info dict with start_idx, end_idx, array_range.
            logs_dir: Directory for logs.
            job_name: Job name.

        Returns:
            SLURM job ID.

        Raises:
            RuntimeError: If submission fails.
        """
        # Generate sbatch script with shard offset embedded
        script_content = _generate_sbatch_script(
            config=config,
            store_root=store_root,
            array_range=shard["array_range"],
            logs_dir=logs_dir,
            job_name=job_name,
            shard_offset=shard["start_idx"],
        )

        # Write script to temp file and submit
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sh",
            delete=False,
            dir=store_root,
        ) as f:
            f.write(script_content)
            script_path = f.name

        try:
            # Submit via sbatch
            exit_code, stdout, stderr = _run_cmd(
                ["sbatch", script_path],
                timeout=60.0,
            )

            if exit_code != 0:
                raise RuntimeError(
                    f"sbatch failed with exit code {exit_code}:\n{stderr}"
                )

            job_id = _parse_slurm_job_id(stdout)
            if not job_id:
                raise RuntimeError(
                    f"Failed to parse job ID from sbatch output:\n{stdout}"
                )

            logger.info(
                f"Submitted SLURM array job {job_id} "
                f"(tasks {shard['start_idx']}-{shard['end_idx']})"
            )

            return job_id

        finally:
            # Clean up script file
            try:
                os.unlink(script_path)
            except Exception:
                pass

    def _write_manifest(
        self,
        store: "Store",
        experiment_id: str,
        job_ids: list[str],
        shards: list[dict[str, Any]],
        total_runs: int,
        total_chunks: int,
        chunk_size: int,
        param_cases: int,
        seed_replicates: int,
        skipped_count: int = 0,
    ) -> None:
        """Write the lightweight manifest for reconnection and Atlas."""
        store_path = _get_store_path(store)
        manifest_path = store_path / "manifest.json"

        manifest = {
            "experiment_id": experiment_id,
            "executor_type": "slurm",
            "submission_mode": "array_indexed",
            "job_ids": job_ids,
            "shards": shards,
            "total_runs": total_runs,
            "total_chunks": total_chunks,
            "chunk_size": chunk_size,
            "param_cases": param_cases,
            "seed_replicates": seed_replicates,
            "skipped_count": skipped_count,
            "max_array_size": self._config.max_array_size,
            "store_root": str(store_path),
            "submitted_at": datetime.now().isoformat(),
        }

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # Legacy submit method for backward compatibility
    def submit(
        self,
        payloads: list[RunPayload],
        store: "Store",
        operation: "OperationWrapper",
        run_ids: list[str] | None = None,
    ) -> "SlurmRunHandle":
        """
        Legacy submit method - redirects to submit_indexed when possible.

        This method exists for backward compatibility. New code should use
        submit_indexed() directly via the runner.

        For large experiments, this will raise an error directing users
        to use the indexed submission path.
        """
        # For small experiments, we could theoretically still use the old approach,
        # but we'll encourage migration to the new approach
        if len(payloads) > 100:
            raise RuntimeError(
                f"Legacy submit() not supported for {len(payloads)} payloads. "
                "Use metalab.run() which now uses index-addressed SLURM arrays automatically."
            )

        warnings.warn(
            "SlurmExecutor.submit() is deprecated. "
            "Use metalab.run() for automatic index-addressed array submission.",
            DeprecationWarning,
            stacklevel=2,
        )

        # For very small experiments, create a minimal handle
        all_run_ids = run_ids if run_ids is not None else [p.run_id for p in payloads]
        submitted_run_ids = {p.run_id for p in payloads}
        skipped_run_ids = [rid for rid in all_run_ids if rid not in submitted_run_ids]

        return SlurmRunHandle(
            store=store,
            job_ids=[],
            shards=[],
            total_runs=len(all_run_ids),
            skipped_count=len(skipped_run_ids),
        )

    def shutdown(self, wait: bool = True) -> None:
        """No-op for SLURM executor (jobs run independently)."""
        pass

    def __enter__(self) -> "SlurmExecutor":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager."""
        self.shutdown(wait=True)


# ---------------------------------------------------------------------------
# SlurmRunHandle
# ---------------------------------------------------------------------------


class SlurmRunHandle:
    """
    RunHandle implementation for index-addressed SLURM arrays.

    Tracks job status via:
    - .done markers in store for completed run counts (accurate)
    - squeue for active chunk counts (RUNNING, PENDING)
    - sacct for terminal chunk counts (COMPLETED, FAILED, etc.)

    This handle does not depend on submitit.
    """

    def __init__(
        self,
        store: "Store",
        job_ids: list[str],
        shards: list[dict[str, Any]],
        total_runs: int,
        chunk_size: int = 1,
        skipped_count: int = 0,
        on_event: "EventCallback | None" = None,
    ) -> None:
        """
        Initialize the SLURM run handle.

        Args:
            store: Store for reading results.
            job_ids: List of SLURM job IDs (one per shard).
            shards: Shard info dicts with start_idx, end_idx, job_id.
            total_runs: Total number of runs.
            chunk_size: Number of runs per array task.
            skipped_count: Number of runs skipped (already complete).
            on_event: Optional callback for progress events.
        """
        self._store = store
        self._job_ids = job_ids
        self._shards = shards
        self._total_runs = total_runs
        self._chunk_size = chunk_size
        self._skipped_count = skipped_count
        self._on_event = on_event

        # Compute store path for .done marker counting
        self._store_path = _get_store_path(store)

        # Cache for sacct results (expensive to query)
        self._sacct_cache: dict[str, int] | None = None
        self._sacct_cache_time: datetime | None = None
        self._sacct_cache_ttl_seconds = 60  # Cache for 1 minute

        # Cache for .done count (moderate cost to scan)
        self._done_count_cache: int | None = None
        self._done_count_cache_time: datetime | None = None
        self._done_count_cache_ttl_seconds = 5  # Refresh every 5 seconds

    def set_event_callback(self, callback: "EventCallback | None") -> None:
        """Set the event callback for progress tracking."""
        self._on_event = callback

    @property
    def job_id(self) -> str:
        """Primary SLURM job ID (first shard)."""
        return self._job_ids[0] if self._job_ids else "slurm-no-jobs"

    def _count_done_markers(self) -> int:
        """
        Count completed runs via .done markers (cached).

        This is the source of truth for completion, as each run writes
        a .done marker atomically after success.
        """
        now = datetime.now()
        if (
            self._done_count_cache is not None
            and self._done_count_cache_time is not None
            and (now - self._done_count_cache_time).total_seconds()
            < self._done_count_cache_ttl_seconds
        ):
            return self._done_count_cache

        # Count .done files in runs directory
        runs_dir = self._store_path / "runs"
        if runs_dir.exists():
            count = sum(1 for _ in runs_dir.glob("*.done"))
        else:
            count = 0

        self._done_count_cache = count
        self._done_count_cache_time = now
        return count

    @property
    def status(self) -> RunStatus:
        """
        Current status using .done markers for completion counts.

        Completion is determined by counting .done marker files, which
        is accurate regardless of chunk_size. SLURM task counts are used
        for running/pending estimates.
        """
        # Count completed runs from .done markers (source of truth)
        completed = self._count_done_markers()

        if not self._job_ids:
            # No jobs submitted - completed count is all we have
            pending = max(0, self._total_runs - completed)
            return RunStatus(
                total=self._total_runs,
                completed=completed,
                running=0,
                pending=pending,
                failed=0,
                skipped=self._skipped_count,
            )

        # Get SLURM chunk counts for running/pending estimates
        squeue_counts = squeue_state_counts(self._job_ids)

        # Estimate running/pending runs from chunk counts
        running_chunks = squeue_counts.get("RUNNING", 0) + squeue_counts.get(
            "COMPLETING", 0
        )
        pending_chunks = squeue_counts.get("PENDING", 0)

        # Convert chunk counts to approximate run counts
        # (last chunk may have fewer runs, but this is a reasonable estimate)
        running_runs = min(
            running_chunks * self._chunk_size, self._total_runs - completed
        )
        pending_runs = min(
            pending_chunks * self._chunk_size,
            self._total_runs - completed - running_runs,
        )

        # Failed = total - completed - running - pending (when all chunks done)
        remaining = self._total_runs - completed - running_runs - pending_runs
        failed = max(0, remaining) if running_chunks == 0 and pending_chunks == 0 else 0

        return RunStatus(
            total=self._total_runs,
            completed=completed,
            running=running_runs,
            pending=pending_runs,
            failed=failed,
            skipped=self._skipped_count,
        )

    @property
    def detailed_status(self) -> dict[str, int]:
        """
        Get detailed status with explicit state buckets.

        Returns:
            Dict with state buckets:
            - completed: runs with .done markers
            - running, pending: estimated from SLURM chunk counts
            - failed: inferred when all chunks complete
            - total, skipped, chunk_size
        """
        completed = self._count_done_markers()

        if not self._job_ids:
            return {
                "running": 0,
                "pending": 0,
                "completed": completed,
                "failed": 0,
                "total": self._total_runs,
                "skipped": self._skipped_count,
                "chunk_size": self._chunk_size,
            }

        # Get SLURM chunk counts
        squeue_counts = squeue_state_counts(self._job_ids)
        sacct_cache = self._get_sacct_cache()

        running_chunks = squeue_counts.get("RUNNING", 0) + squeue_counts.get(
            "COMPLETING", 0
        )
        pending_chunks = squeue_counts.get("PENDING", 0)

        # Get terminal chunk counts for failure detection
        failed_chunks = (
            sacct_cache.get("FAILED", 0)
            + sacct_cache.get("CANCELLED", 0)
            + sacct_cache.get("TIMEOUT", 0)
            + sacct_cache.get("OUT_OF_MEMORY", 0)
        )

        # Estimate run counts
        running_runs = min(
            running_chunks * self._chunk_size, self._total_runs - completed
        )
        pending_runs = min(
            pending_chunks * self._chunk_size,
            self._total_runs - completed - running_runs,
        )

        # Failed runs: when all chunks done, remaining = failed
        remaining = self._total_runs - completed - running_runs - pending_runs
        failed_runs = (
            max(0, remaining) if running_chunks == 0 and pending_chunks == 0 else 0
        )

        return {
            "running": running_runs,
            "pending": pending_runs,
            "completed": completed,
            "failed": failed_runs,
            "failed_chunks": failed_chunks,
            "total": self._total_runs,
            "skipped": self._skipped_count,
            "chunk_size": self._chunk_size,
        }

    def _get_sacct_cache(self) -> dict[str, int] | None:
        """
        Get cached sacct counts, refreshing if stale.

        Returns:
            Cached sacct counts or None if should refresh.
        """
        now = datetime.now()
        if (
            self._sacct_cache is not None
            and self._sacct_cache_time is not None
            and (now - self._sacct_cache_time).total_seconds()
            < self._sacct_cache_ttl_seconds
        ):
            return self._sacct_cache

        # Refresh cache
        counts = sacct_state_counts(self._job_ids)
        self._sacct_cache = counts
        self._sacct_cache_time = now
        return counts

    def _count_failed_runs(self) -> int:
        """
        Count failed runs by scanning run records.

        This is more expensive than counting .done markers, so only
        called when we need accurate failure counts.
        """
        runs_dir = self._store_path / "runs"
        if not runs_dir.exists():
            return 0

        failed = 0
        for path in runs_dir.glob("*.json"):
            try:
                with path.open() as f:
                    data = json.load(f)
                if data.get("status") == "failed":
                    failed += 1
            except (json.JSONDecodeError, OSError):
                continue
        return failed

    @property
    def is_complete(self) -> bool:
        """True if all runs have finished."""
        status = self.status
        return status.done == status.total

    def result(self, timeout: float | None = None) -> Results:
        """
        Block until all jobs complete and return Results.

        Includes a settling loop to handle sacct accounting lag.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            Results object containing all completed runs.

        Raises:
            TimeoutError: If timeout is reached before completion.
        """
        start_time = time.time()
        poll_interval = 5.0
        settle_seconds = 10.0
        settled_at: float | None = None

        while True:
            status = self.status

            if status.done == status.total:
                # All done according to SLURM - start settling
                if settled_at is None:
                    settled_at = time.time()
                    logger.debug("All SLURM jobs complete, waiting for store writes...")

                # Check if we've settled long enough
                if time.time() - settled_at >= settle_seconds:
                    break
            else:
                settled_at = None  # Reset if not complete

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError("Timeout waiting for SLURM jobs")

            time.sleep(poll_interval)

        # Load all records from store
        records = self._store.list_run_records()

        return Results(store=self._store, records=records)

    def cancel(self) -> None:
        """Cancel all pending/running jobs."""
        for job_id in self._job_ids:
            exit_code, stdout, stderr = _run_cmd(
                ["scancel", job_id],
                timeout=30.0,
            )
            if exit_code != 0:
                logger.warning(f"Failed to cancel job {job_id}: {stderr}")

    @classmethod
    def from_store(
        cls,
        store: "Store",
        on_event: "EventCallback | None" = None,
    ) -> "SlurmRunHandle":
        """
        Create a handle by loading manifest from store (reconnection).

        Args:
            store: Store containing the manifest.
            on_event: Optional callback for progress events.

        Returns:
            A SlurmRunHandle for tracking and awaiting results.

        Raises:
            FileNotFoundError: If no manifest exists in the store.
        """
        store_path = _get_store_path(store)
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

        # Handle both old and new manifest formats
        if manifest.get("submission_mode") == "array_indexed":
            # New indexed format (chunk_size defaults to 1 for backward compat)
            return cls(
                store=store,
                job_ids=manifest.get("job_ids", []),
                shards=manifest.get("shards", []),
                total_runs=manifest.get("total_runs", 0),
                chunk_size=manifest.get("chunk_size", 1),
                skipped_count=manifest.get("skipped_count", 0),
                on_event=on_event,
            )
        else:
            # Old submitit format - limited support
            job_array_id = manifest.get("job_array_id")
            total = manifest.get("total", len(manifest.get("run_ids", [])))
            skipped = manifest.get("skipped", len(manifest.get("skipped_run_ids", [])))

            warnings.warn(
                "Reconnecting to old-format SLURM manifest. "
                "Status tracking may be limited.",
                stacklevel=2,
            )

            return cls(
                store=store,
                job_ids=[job_array_id] if job_array_id else [],
                shards=[],
                total_runs=total,
                chunk_size=1,
                skipped_count=skipped,
                on_event=on_event,
            )
