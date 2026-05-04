"""
Executor protocol: Platform-agnostic execution interface.

The Executor abstraction supports:
- Threads (same process)
- Processes (multiple processes)
- SLURM/HPC batch systems via submitit
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from metalab.executor.handle import RunHandle
    from metalab.executor.payload import RunPayload
    from metalab.experiment import Experiment
    from metalab.seeds.bundle import SeedBundle
    from metalab.store.base import Store
    from metalab.types import Status


@dataclass(frozen=True)
class RunPlanEntry:
    """One deterministic run planned for an experiment submission."""

    run_id: str
    params_resolved: dict[str, Any]
    seed_bundle: SeedBundle
    params_fingerprint: str
    seed_fingerprint: str
    status: Status | None = None

    @property
    def should_skip(self) -> bool:
        """True when an existing canonical success should be skipped."""
        from metalab.types import Status

        return self.status == Status.SUCCESS


@dataclass(frozen=True)
class ExperimentPlan:
    """Executor-agnostic submission plan prepared by the runner."""

    experiment: Experiment
    store: Store
    context_fingerprint: str
    run_entries: list[RunPlanEntry]
    job_id: str
    executor_type: str
    resolved_context_manifest: dict[str, Any] | None = None

    @property
    def all_run_ids(self) -> list[str]:
        """All planned run IDs, including skipped successes."""
        return [entry.run_id for entry in self.run_entries]

    @property
    def total_runs(self) -> int:
        """Total planned run count."""
        return len(self.run_entries)

    @property
    def skipped_run_ids(self) -> list[str]:
        """Run IDs skipped because canonical successful records already exist."""
        return [entry.run_id for entry in self.run_entries if entry.should_skip]

    @property
    def skipped_count(self) -> int:
        """Number of skipped successful records."""
        return len(self.skipped_run_ids)

    @property
    def pending_entries(self) -> list[RunPlanEntry]:
        """Entries that should be submitted to workers."""
        return [entry for entry in self.run_entries if not entry.should_skip]

    def to_payloads(self) -> list[RunPayload]:
        """Materialize per-run payloads for payload-based executors."""
        from metalab.executor.payload import RunPayload

        return [
            RunPayload(
                run_id=entry.run_id,
                experiment_id=self.experiment.experiment_id,
                context_spec=self.experiment.context,
                params_resolved=entry.params_resolved,
                seed_bundle=entry.seed_bundle,
                store_locator=self.store.config.to_dict(),
                fingerprints={
                    "context": self.context_fingerprint,
                    "params": entry.params_fingerprint,
                    "seed": entry.seed_fingerprint,
                },
                metadata=self.experiment.metadata,
                operation_ref=self.experiment.operation.ref,
                job_id=self.job_id,
            )
            for entry in self.pending_entries
        ]


class Executor(Protocol):
    """Protocol for execution backends."""

    def submit_experiment(self, plan: ExperimentPlan) -> RunHandle:
        """Submit an executor-agnostic experiment plan."""
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        ...
