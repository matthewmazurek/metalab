"""SLURM executor configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from metalab.executor.config import ExecutorConfig


@dataclass
class SlurmExecutorConfig(ExecutorConfig):
    """
    Configuration for SLURM cluster execution.

    Accepts a flat dict (e.g., from YAML) and maps it to the existing
    SlurmConfig + SlurmExecutor. Handles convenience fields like
    mail_user/mail_type -> extra_sbatch mapping.
    """

    executor_type: ClassVar[str] = "slurm"

    partition: str = "default"
    time: str = "1:00:00"
    cpus: int = 1
    memory: str = "4G"
    gpus: int = 0
    max_concurrent: int | None = None
    max_array_size: int = 10000
    chunk_size: int = 1
    modules: list[str] = field(default_factory=list)
    conda_env: str | None = None
    setup: list[str] = field(default_factory=list)
    extra_sbatch: dict[str, str] = field(default_factory=dict)

    def create(self):
        """Create a SlurmExecutor from this config."""
        from metalab.executor.slurm import SlurmConfig, SlurmExecutor

        return SlurmExecutor(
            SlurmConfig(
                partition=self.partition,
                time=self.time,
                cpus=self.cpus,
                memory=self.memory,
                gpus=self.gpus,
                max_concurrent=self.max_concurrent,
                max_array_size=self.max_array_size,
                chunk_size=self.chunk_size,
                modules=self.modules,
                conda_env=self.conda_env,
                setup=self.setup,
                extra_sbatch=self.extra_sbatch,
            )
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SlurmExecutorConfig:
        """Parse from a config dict (e.g., YAML section).

        Handles convenience fields:
        - mail_user, mail_type are moved into extra_sbatch
        - String values for modules/setup are wrapped in a list

        Args:
            d: Configuration dictionary.

        Returns:
            A SlurmExecutorConfig instance.
        """
        d = d.copy()

        # Map convenience fields to extra_sbatch
        extra = dict(d.pop("extra_sbatch", {}))
        for key in ("mail_user", "mail_type"):
            if key in d:
                extra[key] = d.pop(key)

        # Handle list fields that might come as a single string
        for list_field in ("modules", "setup"):
            if list_field in d and isinstance(d[list_field], str):
                d[list_field] = [d[list_field]]

        # Filter to known fields only
        known = {
            "partition",
            "time",
            "cpus",
            "memory",
            "gpus",
            "max_concurrent",
            "max_array_size",
            "chunk_size",
            "modules",
            "conda_env",
            "setup",
        }
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered, extra_sbatch=extra)
