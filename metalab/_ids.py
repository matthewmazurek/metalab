"""
Consolidated identity module (internal).

Single source of truth for all fingerprinting and run_id computation.
This avoids circular imports and ensures consistent identity across the framework.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from metalab._canonical import canonical, fingerprint

if TYPE_CHECKING:
    from metalab.seeds.bundle import SeedBundle


def fingerprint_params(params: dict[str, Any]) -> str:
    """
    Compute a stable fingerprint for resolved parameters.

    Args:
        params: The resolved parameter dictionary.

    Returns:
        A 16-character hex fingerprint.
    """
    return fingerprint(params)


def fingerprint_context(spec: Any) -> str:
    """
    Compute a stable fingerprint for a context spec.

    The spec must be serializable (dataclass, dict, or object with __dict__).

    Args:
        spec: The context specification.

    Returns:
        A 16-character hex fingerprint.
    """
    return fingerprint(spec)


def fingerprint_seeds(bundle: SeedBundle) -> str:
    """
    Compute a stable fingerprint for a seed bundle.

    Args:
        bundle: The SeedBundle instance.

    Returns:
        A 16-character hex fingerprint.
    """
    # Create a canonical representation of the seed bundle
    seed_data = {
        "root_seed": bundle.root_seed,
        "replicate_index": bundle.replicate_index,
    }
    return fingerprint(seed_data)


def compute_run_id(
    experiment_id: str,
    context_fp: str,
    params_fp: str,
    seed_fp: str,
    code_fp: str | None = None,
) -> str:
    """
    Compute a stable run_id from component fingerprints.

    The run_id uniquely identifies a run based on:
    - experiment identity (name + version)
    - context (data/resources)
    - parameters (resolved values)
    - seeds (RNG state)
    - optionally, code hash

    Args:
        experiment_id: The experiment identifier (name:version).
        context_fp: Fingerprint of the context spec.
        params_fp: Fingerprint of resolved parameters.
        seed_fp: Fingerprint of the seed bundle.
        code_fp: Optional fingerprint of the operation code.

    Returns:
        A 16-character hex run_id.
    """
    # Combine all components into a canonical string
    components = {
        "experiment_id": experiment_id,
        "context_fp": context_fp,
        "params_fp": params_fp,
        "seed_fp": seed_fp,
    }
    if code_fp is not None:
        components["code_fp"] = code_fp

    # Use canonical() to ensure deterministic ordering
    canonical_str = canonical(components)
    hash_bytes = hashlib.sha256(canonical_str.encode("utf-8")).digest()
    return hash_bytes.hex()[:16]
