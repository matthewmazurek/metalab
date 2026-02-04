"""
Consolidated identity module (internal).

Single source of truth for all fingerprinting and run_id computation.
This avoids circular imports and ensures consistent identity across the framework.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from metalab._canonical import canonical, fingerprint

if TYPE_CHECKING:
    from metalab.seeds.bundle import SeedBundle


# ---------------------------------------------------------------------------
# Fingerprintable Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Fingerprintable(Protocol):
    """
    Protocol for types that compute their fingerprint payload at resolution time.

    Implement this protocol to create custom types that participate in
    context fingerprinting with lazy evaluation.

    Example:
    ```python
    @dataclass(frozen=True)
    class GitCommit:
        repo: str

        def fingerprint_payload(self) -> dict[str, Any]:
            # Compute hash at fingerprint time, not construction
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.repo
            ).decode().strip()
            return {"repo": self.repo, "commit": commit}
    ```
    """

    def fingerprint_payload(self) -> dict[str, Any]:
        """Return the data to include in the fingerprint."""
        ...


# ---------------------------------------------------------------------------
# FilePath / DirPath - Lazy file markers implementing Fingerprintable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilePath:
    """
    A file path marker that implements Fingerprintable.

    The hash is computed lazily at fingerprint time using file metadata
    (size, mtime, inode) for O(1) performance regardless of file size.

    When used in a context spec, the fingerprint will include the metadata
    hash, ensuring runs are deduplicated based on file identity.

    Args:
        path: Path to the file.
        hash_path: If True, include the path in the fingerprint (default: False).
                   When False, only file metadata is used - same file at different
                   paths produces the same fingerprint.

    Example:
    ```python
    @metalab.context_spec
    class DataSpec:
        data: FilePath

    spec = DataSpec(data=FilePath("./cache/adata.h5ad"))
    preprocess(spec)  # File created here
    metalab.run(...)  # Hash computed here
    ```
    """

    path: str
    hash_path: bool = False

    def __post_init__(self) -> None:
        # Convert Path objects to strings
        if isinstance(self.path, Path):
            object.__setattr__(self, "path", str(self.path))

    def fingerprint_payload(self) -> dict[str, Any]:
        """Compute metadata-based hash and return payload for fingerprinting."""
        payload: dict[str, Any] = {"hash": _file_metadata_hash(self.path)}
        if self.hash_path:
            payload["path"] = self.path
        return payload

    def __str__(self) -> str:
        """Return the path (for use in operations)."""
        return self.path

    def __fspath__(self) -> str:
        """Support os.fspath() and Path() construction."""
        return self.path

    def __repr__(self) -> str:
        if self.hash_path:
            return f"FilePath({self.path!r}, hash_path=True)"
        return f"FilePath({self.path!r})"


@dataclass(frozen=True)
class DirPath:
    """
    A directory path marker that implements Fingerprintable.

    The hash is computed lazily at fingerprint time using file metadata
    (size, mtime, inode) for O(1) performance per file.

    When used in a context spec, the fingerprint will include the aggregated
    metadata hash of all matching files.

    Args:
        path: Path to the directory.
        pattern: Glob pattern for files to include (default: "*").
        hash_path: If True, include the path in the fingerprint (default: False).

    Example:
    ```python
    @metalab.context_spec
    class DataSpec:
        raw_data: DirPath

    spec = DataSpec(raw_data=DirPath("./data/raw/", pattern="*.csv"))
    ```
    """

    path: str
    pattern: str = "*"
    hash_path: bool = False

    def __post_init__(self) -> None:
        # Convert Path objects to strings
        if isinstance(self.path, Path):
            object.__setattr__(self, "path", str(self.path))

    def fingerprint_payload(self) -> dict[str, Any]:
        """Compute metadata-based hash and return payload for fingerprinting."""
        payload: dict[str, Any] = {"hash": _dir_metadata_hash(self.path, self.pattern)}
        if self.hash_path:
            payload["path"] = self.path
            payload["pattern"] = self.pattern
        return payload

    def __str__(self) -> str:
        """Return the path (for use in operations)."""
        return self.path

    def __fspath__(self) -> str:
        """Support os.fspath() and Path() construction."""
        return self.path

    def __repr__(self) -> str:
        parts = [repr(self.path)]
        if self.pattern != "*":
            parts.append(f"pattern={self.pattern!r}")
        if self.hash_path:
            parts.append("hash_path=True")
        return f"DirPath({', '.join(parts)})"


# Backwards compatibility aliases (deprecated)
FileHash = FilePath
DirHash = DirPath


# ---------------------------------------------------------------------------
# Metadata-based hashing (O(1) regardless of file size)
# ---------------------------------------------------------------------------


def _file_metadata_hash(path: str) -> str:
    """
    Hash file metadata (size, mtime_ns, inode) - O(1) regardless of file size.

    This is much faster than content hashing for large files (e.g., 10GB AnnData
    takes <1ms vs ~30s for content hashing).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {path}")
    stat = p.stat()
    # Combine size + mtime_ns + inode for change detection
    metadata = f"{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ino}"
    return hashlib.sha256(metadata.encode()).hexdigest()[:16]


def _dir_metadata_hash(path: str, pattern: str = "*") -> str:
    """
    Hash directory by aggregating metadata of all matching files.

    Files are sorted by relative path for deterministic ordering.
    """
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"directory not found: {path}")

    entries = []
    for f in sorted(p.glob(pattern)):
        if f.is_file():
            stat = f.stat()
            # Include relative path for ordering stability
            rel_path = f.relative_to(p)
            entries.append(
                f"{rel_path}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ino}"
            )

    combined = "\n".join(entries)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Content-based hashing (kept for manual use / backwards compatibility)
# ---------------------------------------------------------------------------


def _compute_file_hash(path: str, algorithm: str = "sha256") -> str:
    """Internal: compute file content hash (slower, but content-based)."""
    p = Path(path)
    h = hashlib.new(algorithm)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _compute_dir_hash(path: str, pattern: str = "*", algorithm: str = "sha256") -> str:
    """Internal: compute directory content hash (slower, but content-based)."""
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    h = hashlib.new(algorithm)
    for file_path in sorted(p.glob(pattern)):
        if file_path.is_file():
            h.update(file_path.name.encode())
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Context Resolution
# ---------------------------------------------------------------------------


def resolve_context(spec: Any, _path: str = "context") -> tuple[Any, dict[str, Any]]:
    """
    Resolve a context spec, calling fingerprint_payload() on Fingerprintable objects.

    This function recursively walks the spec structure and:
    - For Fingerprintable objects: calls fingerprint_payload() to compute lazy values
    - For dataclasses, dicts, lists: recursively resolves children
    - For primitives: passes through unchanged

    Args:
        spec: The context specification to resolve.
        _path: Internal path tracking for error messages.

    Returns:
        Tuple of (resolved_spec_for_fingerprinting, manifest_for_storage)

    Raises:
        FileNotFoundError: If a FilePath/DirPath references a missing file,
                          with the full path in the error message.

    Example:
    ```python
    resolved, manifest = resolve_context(spec)
    # manifest contains {"context.data": {"type": "FilePath", "hash": "abc123"}}
    ```
    """
    manifest: dict[str, Any] = {}

    # Check for Fingerprintable protocol (duck typing via hasattr)
    if hasattr(spec, "fingerprint_payload"):
        try:
            payload = spec.fingerprint_payload()
            manifest[_path] = {"type": type(spec).__name__, **payload}
            return payload, manifest
        except FileNotFoundError as e:
            # Re-raise with full path context
            raise FileNotFoundError(f"{_path}: {e}") from e

    # Recursively walk dataclasses
    if dataclasses.is_dataclass(spec) and not isinstance(spec, type):
        resolved: dict[str, Any] = {}
        for field in dataclasses.fields(spec):
            field_path = f"{_path}.{field.name}"
            value = getattr(spec, field.name)
            resolved_value, field_manifest = resolve_context(value, field_path)
            resolved[field.name] = resolved_value
            manifest.update(field_manifest)
        return resolved, manifest

    # Recursively walk dicts
    if isinstance(spec, dict):
        resolved_dict: dict[str, Any] = {}
        for k, v in spec.items():
            key_path = f"{_path}.{k}"
            resolved_value, field_manifest = resolve_context(v, key_path)
            resolved_dict[k] = resolved_value
            manifest.update(field_manifest)
        return resolved_dict, manifest

    # Recursively walk lists/tuples
    if isinstance(spec, (list, tuple)):
        resolved_list: list[Any] = []
        for i, item in enumerate(spec):
            item_path = f"{_path}[{i}]"
            resolved_value, field_manifest = resolve_context(item, item_path)
            resolved_list.append(resolved_value)
            manifest.update(field_manifest)
        return resolved_list, manifest

    # Primitives pass through unchanged
    return spec, manifest


def file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    """
    Compute a hash of a file's contents.

    Useful for including file checksums in context specs to ensure
    fingerprint changes when source data changes.

    Args:
        path: Path to the file to hash.
        algorithm: Hash algorithm to use (default: "sha256").

    Returns:
        A 16-character hex hash of the file contents.

    Raises:
        FileNotFoundError: If the file does not exist.

    Example:
    ```python
    @metalab.context_spec
    class MyDataSpec:
        raw_path: str
        raw_hash: str  # Include so fingerprint changes if data changes

    spec = MyDataSpec(
        raw_path="data/matrix.mtx.gz",
        raw_hash=metalab.file_hash("data/matrix.mtx.gz"),
    )
    ```
    """
    path = Path(path)
    h = hashlib.new(algorithm)

    with path.open("rb") as f:
        # Read in chunks for memory efficiency with large files
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()[:16]


def dir_hash(path: str | Path, pattern: str = "*", algorithm: str = "sha256") -> str:
    """
    Compute a hash of a directory's contents.

    Hashes all files matching the pattern, sorted by name for stability.
    Useful for fingerprinting a directory of input files.

    Args:
        path: Path to the directory.
        pattern: Glob pattern for files to include (default: "*" for all files).
        algorithm: Hash algorithm to use (default: "sha256").

    Returns:
        A 16-character hex hash of the combined file contents.

    Raises:
        FileNotFoundError: If the directory does not exist.

    Example:
    ```python
    spec = MyDataSpec(
        raw_dir="data/raw/",
        raw_hash=metalab.dir_hash("data/raw/", pattern="*.mtx.gz"),
    )
    ```
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    h = hashlib.new(algorithm)

    # Sort files for deterministic ordering
    for file_path in sorted(path.glob(pattern)):
        if file_path.is_file():
            # Include filename in hash for stability
            h.update(file_path.name.encode())
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)

    return h.hexdigest()[:16]


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

    This function always resolves the spec first, computing lazy values
    (like FilePath/DirPath metadata hashes) before fingerprinting.
    This ensures that file content changes are reflected in the fingerprint.

    Args:
        spec: The context specification.

    Returns:
        A 16-character hex fingerprint.

    Raises:
        FileNotFoundError: If any FilePath/DirPath references a missing file.
    """
    resolved, _manifest = resolve_context(spec)
    return fingerprint(resolved)


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
