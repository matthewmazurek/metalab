#!/usr/bin/env python
"""
SLURM array worker: entrypoint for index-addressed SLURM array tasks.

This module is invoked by SLURM array jobs via:
    python -m metalab.executor.slurm_array_worker --store <store_root>

Each array task:
1. Reads the array spec from the store root
2. Computes (param_idx, seed_idx) from SLURM_ARRAY_TASK_ID + shard offset
3. Reconstructs ParamCase and SeedBundle using index-based access
4. Computes run_id deterministically
5. Checks for completion (successful canonical run record)
6. If not complete, executes the run via execute_payload()
7. Writes persistent events and heartbeats through the store

Environment contract:
- SLURM_ARRAY_TASK_ID: The array task index (0-based within shard)
- SLURM_ARRAY_JOB_ID: The job array ID
- SLURM_JOB_ID: The full job ID (includes array index suffix)
- METALAB_SHARD_OFFSET: (optional) Offset for sharded arrays
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def main() -> int:
    """
    Main entry point for the SLURM array worker.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="metalab SLURM array worker",
    )
    parser.add_argument(
        "--store",
        required=True,
        help="Path to the store root directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        return run_array_task(args.store)
    except Exception as e:
        logger.exception(f"Worker failed: {e}")
        return 1


def run_array_task(store_root: str) -> int:
    """
    Execute runs for this array task (one chunk of runs).

    Each array task processes chunk_size runs, identified by:
    chunk_id = shard_offset + SLURM_ARRAY_TASK_ID

    Args:
        store_root: Path to the store root directory.

    Returns:
        Exit code (0 for all success, 1 if any failed).
    """
    # Get SLURM environment
    array_task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "unknown")
    job_id = os.environ.get("SLURM_JOB_ID", "unknown")

    if array_task_id_str is None:
        logger.error("SLURM_ARRAY_TASK_ID not set - not running in SLURM context")
        return 1

    array_task_id = int(array_task_id_str)
    shard_offset = int(os.environ.get("METALAB_SHARD_OFFSET", "0"))
    chunk_id = shard_offset + array_task_id

    # Load array spec
    store_path = Path(store_root)
    spec_path = store_path / "slurm_array_spec.json"

    if not spec_path.exists():
        logger.error(f"Array spec not found: {spec_path}")
        return 1

    with open(spec_path) as f:
        spec = json.load(f)

    # Extract spec values
    seed_replicates = spec["seed_replicates"]
    total_runs = spec["total_runs"]
    chunk_size = spec.get("chunk_size", 1)

    # Compute run range for this chunk
    start_run = chunk_id * chunk_size
    end_run = min(start_run + chunk_size, total_runs)

    if start_run >= total_runs:
        logger.error(f"Chunk {chunk_id} out of range (total_runs={total_runs})")
        return 1

    logger.info(
        f"Starting chunk {chunk_id}: job={array_job_id}, "
        f"runs [{start_run}, {end_run}) of {total_runs}"
    )

    # Load shared resources once
    from metalab._ids import compute_run_id, fingerprint_params, fingerprint_seeds
    from metalab.executor.core import execute_payload
    from metalab.manifest import deserialize_param_source, deserialize_seed_plan
    from metalab.operation import import_operation
    from metalab.store.file import FileStoreConfig
    from metalab.types import Status

    store = FileStoreConfig(root=str(store_path)).connect()
    params_source = deserialize_param_source(spec["params"])
    seed_plan = deserialize_seed_plan(spec["seeds"])
    operation = import_operation(spec["operation_ref"])
    context_spec = _load_context_spec(store_path)
    ctx_fp = spec["context_fingerprint"]
    worker_id, heartbeat_payload = _worker_identity(
        array_job_id=array_job_id,
        job_id=job_id,
        chunk_id=chunk_id,
        start_run=start_run,
        end_run=end_run,
    )

    # Process each run in the chunk
    any_failed = False
    for global_run_idx in range(start_run, end_run):
        # Map global index to (param_idx, seed_idx)
        seed_idx = global_run_idx % seed_replicates
        param_idx = global_run_idx // seed_replicates

        param_case = params_source[param_idx]  # type: ignore[index]
        seed_bundle = seed_plan[seed_idx]

        # Compute deterministic run_id
        params_fp = fingerprint_params(param_case.params)
        seed_fp = fingerprint_seeds(seed_bundle)
        run_id = compute_run_id(
            experiment_id=spec["experiment_id"],
            context_fp=ctx_fp,
            params_fp=params_fp,
            seed_fp=seed_fp,
            code_fp=spec["operation_code_hash"],
        )

        # Skip if already complete
        if _is_run_complete(store, run_id, store_path):
            logger.info(f"Run {run_id} already complete, skipping")
            continue

        logger.info(f"Executing run {run_id} (idx={global_run_idx})")

        # Execute
        result = execute_payload(
            run_id=run_id,
            experiment_id=spec["experiment_id"],
            context_spec=context_spec,
            params_resolved=param_case.params,
            seed_bundle=seed_bundle,
            fingerprints={"context": ctx_fp, "params": params_fp, "seed": seed_fp},
            metadata=spec.get("metadata", {}),
            operation=operation,
            store=store,
            worker_id=worker_id,
            job_id=spec.get("job_id", array_job_id),
            heartbeat_payload=heartbeat_payload,
            capture_third_party_logs=True,
        )

        if result.status == Status.SUCCESS:
            logger.info(f"Run {run_id} completed successfully")
        else:
            logger.error(f"Run {run_id} failed: {result.error}")
            any_failed = True

    runs_in_chunk = end_run - start_run
    logger.info(f"Chunk {chunk_id} finished: {runs_in_chunk} runs processed")
    return 1 if any_failed else 0


def _is_run_complete(store: Any, run_id: str, work_dir: Path) -> bool:
    """
    Check if a run is complete using robust completion detection.

    A run is considered complete if its canonical run record exists and
    has Status.SUCCESS.

    Args:
        store: The Store instance.
        run_id: The run ID to check.
        work_dir: Path to the working directory.

    Returns:
        True if the run is complete, False otherwise.
    """
    from metalab.types import Status

    # Check for run record
    if not store.run_exists(run_id):
        return False

    record = store.get_run_record(run_id)
    if record is None:
        return False

    if record.status != Status.SUCCESS:
        return False

    return True


def _worker_identity(
    *,
    array_job_id: str,
    job_id: str,
    chunk_id: int,
    start_run: int,
    end_run: int,
) -> tuple[str, dict[str, Any]]:
    """Return stable SLURM worker id and heartbeat payload for one chunk attempt."""
    task_id = f"chunk:{chunk_id}"
    attempt_id = job_id
    return (
        f"slurm:{array_job_id}:chunk:{chunk_id}:attempt:{attempt_id}",
        {
            "task_id": task_id,
            "attempt_id": attempt_id,
            "chunk_id": chunk_id,
            "start_run": start_run,
            "end_run": end_run,
        },
    )


def _load_context_spec(store_path: Path) -> Any:
    """
    Load context spec from JSON file.

    The context is serialized to JSON during experiment submission with type
    information preserved for dataclasses (FilePath, DirPath, etc.).

    Args:
        store_path: Path to the store root directory.

    Returns:
        The deserialized context spec object, or None if not found.
    """
    from metalab.context.spec import deserialize_context_spec

    context_json_path = store_path / "context_spec.json"

    if not context_json_path.exists():
        logger.warning(f"Context spec JSON not found: {context_json_path}")
        return None

    try:
        with open(context_json_path, "r") as f:
            data = json.load(f)
        return deserialize_context_spec(data)
    except Exception as e:
        logger.error(f"Failed to load context spec: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
