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
5. Checks for completion (run record + .done marker)
6. If not complete, executes the run via execute_payload()
7. Writes .done marker after completion

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
from typing import Any

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
    from metalab.store.file import FileStore
    from metalab.types import Status

    store = FileStore(store_root)
    params_source = deserialize_param_source(spec["params"])
    seed_plan = deserialize_seed_plan(spec["seeds"])
    operation = import_operation(spec["operation_ref"])
    context_spec = _load_context_spec(store_path)
    ctx_fp = spec["context_fingerprint"]
    worker_id = f"slurm:{job_id}"

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
        if _is_run_complete(store, run_id):
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
            derived_metric_refs=spec.get("derived_metric_refs"),
            capture_third_party_logs=True,
        )

        # Persist result
        store.put_run_record(result)

        if result.status == Status.SUCCESS:
            _write_done_marker(store_path, run_id)
            logger.info(f"Run {run_id} completed successfully")
        else:
            logger.error(f"Run {run_id} failed: {result.error}")
            any_failed = True

    runs_in_chunk = end_run - start_run
    logger.info(f"Chunk {chunk_id} finished: {runs_in_chunk} runs processed")
    return 1 if any_failed else 0


def _is_run_complete(store: Any, run_id: str) -> bool:
    """
    Check if a run is complete using robust completion detection.

    A run is considered complete if:
    1. A run record exists with Status.SUCCESS
    2. AND a .done marker file exists

    This prevents skipping runs that have partial/corrupt records.

    Args:
        store: The FileStore instance.
        run_id: The run ID to check.

    Returns:
        True if the run is complete, False otherwise.
    """
    # Check for run record
    if not store.run_exists(run_id):
        return False

    record = store.get_run_record(run_id)
    if record is None:
        return False

    from metalab.types import Status

    if record.status != Status.SUCCESS:
        return False

    # Check for .done marker
    store_path = Path(store.root)
    done_marker = store_path / "runs" / f"{run_id}.done"

    return done_marker.exists()


def _write_done_marker(store_path: Path, run_id: str) -> None:
    """
    Write the .done marker file atomically.

    The .done marker signals that:
    1. The run record has been fully written
    2. All artifacts are persisted

    Args:
        store_path: Path to the store root.
        run_id: The run ID.
    """
    import tempfile
    from datetime import datetime

    runs_dir = store_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    done_marker = runs_dir / f"{run_id}.done"
    content = json.dumps({"completed_at": datetime.now().isoformat()})

    # Write atomically via temp file + rename
    fd, temp_path = tempfile.mkstemp(
        dir=str(runs_dir),
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.rename(temp_path, done_marker)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _load_context_spec(store_path: Path) -> Any:
    """
    Load context spec from pickled file.

    The context is pickled during experiment submission, so any Python object
    (dataclasses, custom types, etc.) is handled automatically without manual
    serialization logic.

    Args:
        store_path: Path to the store root directory.

    Returns:
        The unpickled context spec object, or None if not found.
    """
    import pickle

    context_pkl_path = store_path / "context_spec.pkl"

    if not context_pkl_path.exists():
        logger.warning(f"Context spec pickle not found: {context_pkl_path}")
        return None

    try:
        with open(context_pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load context spec: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
