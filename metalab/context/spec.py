"""
ContextSpec protocol and FrozenContext type alias.

Invariants (codified here):
1. ContextSpec MUST be serializable (JSON-compatible or reconstructable from manifest)
2. ContextBuilder.build(spec) MUST be deterministic given the same environment/resources
3. FrozenContext MUST be treated as read-only by Operations (no mutation)
4. Runner MAY cache FrozenContext by context_fingerprint within a worker process
5. Context builders SHOULD avoid lazy mutation after build (thread safety)

What "Shared Context" Means:
- ThreadExecutor: Same FrozenContext instance reused across runs (in-memory cache)
- ProcessExecutor: Each process has its own cache (useful for batched runs)
- ARC/HPC: Each job has its own cache; cross-node sharing via external storage only

IMPORTANT: "Shared across ARC workers" means shared via external materialization
(dataset paths, cached artifacts on shared storage)—NOT in-memory sharing.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ContextSpec(Protocol):
    """
    Protocol for context specifications.

    A ContextSpec is a serializable manifest that describes how to construct
    the actual context (FrozenContext) on any worker. It should contain:
    - Dataset/resource identifiers (paths, URIs, IDs, checksums)
    - Configuration fragments
    - Versions of upstream processing steps (optional but recommended)

    The spec itself should be small and serializable. Heavy data loading
    happens in the ContextBuilder.

    Implementations can be:
    - A frozen dataclass
    - A plain dict
    - Any object that can be canonically serialized

    Example:
        @dataclass(frozen=True)
        class MyContextSpec:
            dataset_path: str
            dataset_checksum: str
            config: dict
    """

    # No required methods - any serializable object can be a ContextSpec
    # The protocol exists for type hints and documentation
    pass


# Type alias for built context
# User decides actual immutability; we just promise to treat it as read-only
FrozenContext = Any
