"""Local executor configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from metalab.executor.config import ExecutorConfig


@dataclass
class LocalExecutorConfig(ExecutorConfig):
    """
    Configuration for local (thread/process) execution.

    Creates a ProcessExecutor for multi-worker, or None for single-threaded.
    """

    executor_type: ClassVar[str] = "local"
    workers: int = 1

    def create(self):
        """Create executor: ProcessExecutor for workers>1, None for serial."""
        if self.workers == 1:
            return None
        from metalab.executor.process import ProcessExecutor

        return ProcessExecutor(max_workers=self.workers)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LocalExecutorConfig:
        """Parse from a config dict.

        Args:
            d: Configuration dictionary with optional 'workers' key.

        Returns:
            A LocalExecutorConfig instance.
        """
        return cls(workers=d.get("workers", 1))
