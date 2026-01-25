"""
Placeholder for HPC batch executor.

This module provides a stub interface for future ARC/Slurm support.
The key requirement is that payloads must be fully serializable to JSON.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any

from metalab.executor.payload import RunPayload
from metalab.types import RunRecord


class BatchExecutor:
    """
    Placeholder for HPC batch system execution.

    This executor would:
    1. Serialize RunPayload to JSON
    2. Submit a job script that:
       - Loads the payload
       - Imports the operation from operation_ref
       - Executes the run
       - Writes results back to store
    3. Poll for job completion
    4. Return results

    Required payload fields for batch execution:
    - run_id: str
    - experiment_id: str
    - context_spec: JSON-serializable
    - params_resolved: dict
    - seed_bundle: {root_seed, replicate_index}
    - store_locator: path or URI accessible from compute nodes
    - operation_ref: "module:name" importable on compute nodes
    - context_builder_ref: "module:name" or None
    """

    def __init__(
        self,
        queue: str = "default",
        cpus: int = 1,
        memory: str = "4G",
        walltime: str = "1:00:00",
    ) -> None:
        """
        Initialize batch executor.

        Args:
            queue: Job queue/partition name.
            cpus: Number of CPUs per job.
            memory: Memory allocation (e.g., "4G", "16G").
            walltime: Maximum walltime (e.g., "1:00:00").
        """
        self._queue = queue
        self._cpus = cpus
        self._memory = memory
        self._walltime = walltime
        raise NotImplementedError(
            "BatchExecutor is a placeholder for future HPC support. "
            "Use ThreadExecutor or ProcessExecutor for now."
        )

    def submit(self, payload: RunPayload) -> Future[RunRecord]:
        """Submit a batch job."""
        raise NotImplementedError

    def gather(self, futures: list[Future[RunRecord]]) -> list[RunRecord]:
        """Wait for batch jobs to complete."""
        raise NotImplementedError

    def shutdown(self, wait: bool = True) -> None:
        """Cancel pending jobs."""
        raise NotImplementedError


# Example job script template for reference
JOB_SCRIPT_TEMPLATE = """
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={queue}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --time={walltime}
#SBATCH --output={log_dir}/{run_id}.out
#SBATCH --error={log_dir}/{run_id}.err

# Load environment (customize as needed)
source ~/.bashrc
conda activate metalab

# Run the job
python -m metalab.executor._batch_worker {payload_path}
"""
