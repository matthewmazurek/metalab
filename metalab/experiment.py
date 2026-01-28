"""
Experiment: Container for experiment configuration.

An Experiment bundles together:
- Operation to run
- Context specification (lightweight manifest)
- Parameter source
- Seed plan
- Optional resolver
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metalab.context.spec import ContextSpec
    from metalab.operation import OperationWrapper
    from metalab.params.resolver import ParamResolver
    from metalab.params.source import ParamSource
    from metalab.seeds.plan import SeedPlan


@dataclass
class Experiment:
    """
    Container for experiment configuration.

    An Experiment defines what to run and how:
    - name/version: Identifies the experiment
    - description: Optional human-readable description
    - operation: The computation to perform
    - context: Shared configuration (lightweight spec)
    - params: Parameter sweep definition
    - seeds: Seed plan for replication
    - metadata: Experiment-level metadata (not fingerprinted)

    The metadata field is for arbitrary experiment-level information that should
    be persisted but does NOT affect reproducibility or run identity. Examples:
    - Resource hints: {"gpu": True, "memory_gb": 16}
    - Documentation: {"author": "name", "notes": "..."}
    - Data summaries: {"n_samples": 1000, "groups": ["A", "B"]}

    Example:
        exp = Experiment(
            name="pi_mc",
            version="0.1",
            description="Estimate pi using Monte Carlo sampling",
            context={},  # or a @context_spec decorated class
            operation=pi_monte_carlo,
            params=grid(n_samples=[1000, 10000]),
            seeds=seeds(base=42, replicates=3),
            tags=["example", "monte_carlo"],
            metadata={"author": "you", "resource_hints": {"gpu": False}},
        )
    """

    name: str
    version: str
    context: ContextSpec
    operation: OperationWrapper
    params: ParamSource
    seeds: SeedPlan
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    param_resolver: ParamResolver | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def experiment_id(self) -> str:
        """
        The experiment identifier.

        Format: "{name}:{version}"
        """
        return f"{self.name}:{self.version}"

    def __repr__(self) -> str:
        desc = f", description={self.description!r}" if self.description else ""
        return (
            f"Experiment(name={self.name!r}, version={self.version!r}{desc}, "
            f"params={self.params!r}, seeds={self.seeds!r})"
        )
