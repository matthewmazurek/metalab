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
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

if TYPE_CHECKING:
    from metalab.store.base import Store

logger = logging.getLogger(__name__)


def _resolve_store_from_spec(spec: dict[str, Any], work_dir: Path) -> "Store":
    """
    Resolve the store backend from the array spec.

    For Postgres locators without embedded credentials, reads the DSN from
    `{experiments_root}/services/postgres/service.json`.

    Args:
        spec: The loaded array spec dictionary.
        work_dir: Path to the working directory (experiments_root).

    Returns:
        Configured Store instance.
    """
    from metalab.store import create_store

    # Check for store_locator in spec (new format)
    store_locator = spec.get("store_locator")
    experiments_root = spec.get("experiments_root", str(work_dir))
    experiment_id = spec.get("experiment_id")

    if store_locator is None:
        # Fallback for old spec format: use work_dir as FileStore
        logger.debug("No store_locator in spec, using FileStore at work_dir")
        return create_store(str(work_dir))

    # Parse the locator to determine scheme
    parsed = urlparse(store_locator)

    # File scheme or path: use directly (FileStore doesn't use experiment_id)
    if parsed.scheme in ("file", "") or (parsed.scheme and len(parsed.scheme) == 1):
        # len==1 catches Windows drive letters like "C:"
        logger.debug(f"Using FileStore from locator: {store_locator}")
        return create_store(store_locator)

    # PostgreSQL scheme: may need credential resolution
    if parsed.scheme in ("postgresql", "postgres"):
        resolved_locator = _resolve_postgres_locator(
            store_locator, experiments_root, experiment_id or "unknown"
        )
        logger.debug("Using PostgresStore with resolved locator")
        # PostgresStore uses experiment_id for nested FileStore directory
        return create_store(resolved_locator, experiment_id=experiment_id)

    # Unknown scheme: try directly
    logger.warning(f"Unknown store locator scheme '{parsed.scheme}', trying directly")
    return create_store(store_locator)


def _resolve_postgres_locator(
    locator: str, experiments_root: str, experiment_id: str
) -> str:
    """
    Resolve a Postgres locator by filling in credentials from service.json if needed.

    Args:
        locator: The Postgres connection URI.
        experiments_root: Path to experiments root directory.
        experiment_id: The experiment ID (for logging).

    Returns:
        Resolved Postgres connection URI with credentials.
    """
    parsed = urlparse(locator)

    # If password is already present, use as-is
    if parsed.password:
        logger.debug("Postgres locator already has password, using as-is")
        # Ensure experiments_root is in query params
        return _ensure_experiments_root_param(locator, experiments_root)

    # Try to load credentials from service.json
    service_json_path = Path(experiments_root) / "services" / "postgres" / "service.json"

    if not service_json_path.exists():
        logger.warning(
            f"Postgres service.json not found at {service_json_path}, "
            f"attempting connection without password"
        )
        return _ensure_experiments_root_param(locator, experiments_root)

    try:
        with open(service_json_path) as f:
            service_info = json.load(f)

        # Prefer connection_string from service.json if available
        if "connection_string" in service_info:
            base_dsn = service_info["connection_string"]
            logger.debug("Using connection_string from service.json")
            return _ensure_experiments_root_param(base_dsn, experiments_root)

        # Otherwise build from individual fields
        password = service_info.get("password")
        if password:
            # Rebuild the URL with password
            netloc = parsed.netloc
            if "@" in netloc:
                userinfo, hostinfo = netloc.rsplit("@", 1)
                if ":" in userinfo:
                    user = userinfo.split(":")[0]
                else:
                    user = userinfo
            else:
                user = service_info.get("user", "")
                hostinfo = netloc

            # URL-encode password in case it contains special chars
            from urllib.parse import quote

            new_netloc = f"{user}:{quote(password, safe='')}@{hostinfo}"
            resolved = urlunparse(
                (parsed.scheme, new_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
            )
            logger.debug("Resolved password from service.json fields")
            return _ensure_experiments_root_param(resolved, experiments_root)

    except Exception as e:
        logger.warning(f"Failed to load Postgres credentials from service.json: {e}")

    # Return original locator with experiments_root param
    return _ensure_experiments_root_param(locator, experiments_root)


def _ensure_experiments_root_param(locator: str, experiments_root: str) -> str:
    """
    Ensure the experiments_root query parameter is present in a Postgres locator.

    Args:
        locator: The Postgres connection URI.
        experiments_root: Path to experiments root directory.

    Returns:
        Locator with experiments_root query parameter.
    """
    parsed = urlparse(locator)
    query_params = parse_qs(parsed.query)

    # Only add if not already present
    if "experiments_root" not in query_params:
        query_params["experiments_root"] = [experiments_root]
        new_query = urlencode(query_params, doseq=True)
        locator = urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)
        )

    return locator


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
    from metalab.types import Status

    # Resolve store backend from spec (may be FileStore or PostgresStore)
    store = _resolve_store_from_spec(spec, store_path)
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


def _is_run_complete(store: Any, run_id: str, work_dir: Path) -> bool:
    """
    Check if a run is complete using robust completion detection.

    A run is considered complete if:
    1. A run record exists with Status.SUCCESS
    2. AND a .done marker file exists

    This prevents skipping runs that have partial/corrupt records.

    Args:
        store: The Store instance.
        run_id: The run ID to check.
        work_dir: Path to the working directory for .done markers.

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

    # Check for .done marker on filesystem (coordination marker)
    done_marker = work_dir / "runs" / f"{run_id}.done"

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
    Load context spec from JSON file.

    The context is serialized to JSON during experiment submission with type
    information preserved for dataclasses (FilePath, DirPath, etc.).

    Args:
        store_path: Path to the store root directory.

    Returns:
        The deserialized context spec object, or None if not found.
    """
    context_json_path = store_path / "context_spec.json"

    if not context_json_path.exists():
        logger.warning(f"Context spec JSON not found: {context_json_path}")
        return None

    try:
        with open(context_json_path, "r") as f:
            data = json.load(f)
        return _deserialize_context_spec(data)
    except Exception as e:
        logger.error(f"Failed to load context spec: {e}")
        raise


def _deserialize_context_spec(obj: Any) -> Any:
    """
    Deserialize a context spec from JSON structure with type reconstruction.

    Reconstructs dataclasses (FilePath, DirPath, context_spec decorated classes)
    from their serialized form using __type__ metadata.

    Args:
        obj: The JSON-loaded structure.

    Returns:
        The reconstructed context spec object.
    """
    import importlib

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, list):
        return [_deserialize_context_spec(item) for item in obj]
    if isinstance(obj, dict):
        if "__type__" in obj:
            # Reconstruct the dataclass
            type_path = obj["__type__"]
            module_path, class_name = type_path.rsplit(".", 1)
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                # Get field values, excluding __type__
                field_values = {
                    k: _deserialize_context_spec(v)
                    for k, v in obj.items()
                    if k != "__type__"
                }
                return cls(**field_values)
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not reconstruct type {type_path}: {e}")
                # Return as dict if reconstruction fails
                return {k: _deserialize_context_spec(v) for k, v in obj.items() if k != "__type__"}
        else:
            return {k: _deserialize_context_spec(v) for k, v in obj.items()}
    return obj


if __name__ == "__main__":
    sys.exit(main())
