"""
Unit tests for SLURM executor chunking and sharding logic.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from metalab._ids import DirPath, FilePath
from metalab.context.spec import serialize_context_spec
from metalab.executor.slurm import SlurmConfig, SlurmExecutor
from metalab.executor.slurm_array_worker import _load_context_spec


# Module-level context specs for serialization tests
@dataclasses.dataclass(frozen=True)
class _TestContextSpec:
    """Test context spec for serialization roundtrip."""

    adata_file: FilePath
    n_neighbors: int = 30


@dataclasses.dataclass(frozen=True)
class _InnerContext:
    """Inner context for nested tests."""

    output_dir: DirPath
    format: str = "csv"


@dataclasses.dataclass(frozen=True)
class _OuterContext:
    """Outer context for nested tests."""

    name: str
    inner: _InnerContext


@dataclasses.dataclass
class _CustomContextWithMethod:
    """Custom context with methods for testing pickle."""

    value: int

    def double(self) -> int:
        return self.value * 2


class TestSlurmImports:
    """Smoke tests to ensure SLURM code paths have valid imports."""

    def test_runner_slurm_indexed_imports(self):
        """Ensure _run_slurm_indexed has valid imports."""
        from metalab.runner import _run_slurm_indexed  # noqa: F401

    def test_slurm_array_worker_imports(self):
        """Ensure slurm_array_worker module has valid imports."""
        from metalab.executor import slurm_array_worker  # noqa: F401


class TestSlurmConfigDefaults:
    """Test SlurmConfig default values."""

    def test_default_chunk_size_is_one(self):
        """Default chunk_size should be 1 (no chunking)."""
        config = SlurmConfig()
        assert config.chunk_size == 1

    def test_custom_chunk_size(self):
        """chunk_size can be customized."""
        config = SlurmConfig(chunk_size=100)
        assert config.chunk_size == 100


class TestComputeShards:
    """Test the _compute_shards method."""

    def test_single_shard_small_count(self):
        """Small item count fits in one shard."""
        config = SlurmConfig(max_array_size=10000)
        executor = SlurmExecutor(config)

        shards = executor._compute_shards(total_items=100)

        assert len(shards) == 1
        assert shards[0]["start_idx"] == 0
        assert shards[0]["end_idx"] == 99
        assert shards[0]["array_range"] == "0-99"

    def test_multiple_shards_large_count(self):
        """Large item count requires multiple shards."""
        config = SlurmConfig(max_array_size=100)
        executor = SlurmExecutor(config)

        shards = executor._compute_shards(total_items=250)

        assert len(shards) == 3
        # First shard: 0-99
        assert shards[0]["start_idx"] == 0
        assert shards[0]["end_idx"] == 99
        assert shards[0]["array_range"] == "0-99"
        # Second shard: 100-199
        assert shards[1]["start_idx"] == 100
        assert shards[1]["end_idx"] == 199
        assert shards[1]["array_range"] == "0-99"
        # Third shard: 200-249
        assert shards[2]["start_idx"] == 200
        assert shards[2]["end_idx"] == 249
        assert shards[2]["array_range"] == "0-49"

    def test_max_concurrent_in_array_range(self):
        """max_concurrent adds throttle to array range."""
        config = SlurmConfig(max_array_size=10000, max_concurrent=50)
        executor = SlurmExecutor(config)

        shards = executor._compute_shards(total_items=100)

        assert len(shards) == 1
        assert shards[0]["array_range"] == "0-99%50"

    def test_exact_boundary_no_extra_shard(self):
        """Exact boundary doesn't create empty shard."""
        config = SlurmConfig(max_array_size=100)
        executor = SlurmExecutor(config)

        shards = executor._compute_shards(total_items=200)

        assert len(shards) == 2
        assert shards[0]["end_idx"] == 99
        assert shards[1]["start_idx"] == 100
        assert shards[1]["end_idx"] == 199


class TestChunkingMath:
    """Test chunking calculations."""

    def test_chunk_size_reduces_array_tasks(self):
        """chunk_size > 1 reduces total array tasks."""
        config = SlurmConfig(max_array_size=10000, chunk_size=10)
        executor = SlurmExecutor(config)

        # 1000 runs with chunk_size=10 = 100 chunks
        total_runs = 1000
        chunk_size = config.chunk_size
        total_chunks = (total_runs + chunk_size - 1) // chunk_size

        assert total_chunks == 100

        shards = executor._compute_shards(total_items=total_chunks)
        assert len(shards) == 1
        assert shards[0]["end_idx"] == 99

    def test_chunk_size_with_remainder(self):
        """Remainder runs handled correctly."""
        config = SlurmConfig(max_array_size=10000, chunk_size=7)
        executor = SlurmExecutor(config)

        # 100 runs with chunk_size=7 = ceil(100/7) = 15 chunks
        total_runs = 100
        chunk_size = config.chunk_size
        total_chunks = (total_runs + chunk_size - 1) // chunk_size

        assert total_chunks == 15

        shards = executor._compute_shards(total_items=total_chunks)
        assert len(shards) == 1
        assert shards[0]["end_idx"] == 14

    def test_large_experiment_with_chunking(self):
        """100k runs with chunk_size=100 = 1000 array tasks."""
        config = SlurmConfig(max_array_size=10000, chunk_size=100)
        executor = SlurmExecutor(config)

        total_runs = 100_000
        chunk_size = config.chunk_size
        total_chunks = (total_runs + chunk_size - 1) // chunk_size

        assert total_chunks == 1000

        shards = executor._compute_shards(total_items=total_chunks)
        # 1000 chunks fits in one shard (max_array_size=10000)
        assert len(shards) == 1
        assert shards[0]["end_idx"] == 999

    def test_very_large_chunked_experiment_shards(self):
        """1M runs with chunk_size=100, max_array_size=5000 = 2 shards."""
        config = SlurmConfig(max_array_size=5000, chunk_size=100)
        executor = SlurmExecutor(config)

        total_runs = 1_000_000
        chunk_size = config.chunk_size
        total_chunks = (total_runs + chunk_size - 1) // chunk_size

        assert total_chunks == 10_000

        shards = executor._compute_shards(total_items=total_chunks)
        # 10k chunks with max_array_size=5000 = 2 shards
        assert len(shards) == 2
        assert shards[0]["start_idx"] == 0
        assert shards[0]["end_idx"] == 4999
        assert shards[1]["start_idx"] == 5000
        assert shards[1]["end_idx"] == 9999


class TestWorkerChunkRange:
    """Test the chunk range calculation used in the worker."""

    def test_chunk_range_first_chunk(self):
        """First chunk covers [0, chunk_size)."""
        chunk_size = 10
        total_runs = 100
        chunk_id = 0

        start_run = chunk_id * chunk_size
        end_run = min(start_run + chunk_size, total_runs)

        assert start_run == 0
        assert end_run == 10

    def test_chunk_range_middle_chunk(self):
        """Middle chunk covers correct range."""
        chunk_size = 10
        total_runs = 100
        chunk_id = 5

        start_run = chunk_id * chunk_size
        end_run = min(start_run + chunk_size, total_runs)

        assert start_run == 50
        assert end_run == 60

    def test_chunk_range_last_chunk_partial(self):
        """Last chunk may have fewer runs."""
        chunk_size = 10
        total_runs = 95
        chunk_id = 9  # Last chunk

        start_run = chunk_id * chunk_size
        end_run = min(start_run + chunk_size, total_runs)

        assert start_run == 90
        assert end_run == 95  # Only 5 runs in last chunk

    def test_global_index_to_param_seed_mapping(self):
        """Global run index maps to (param_idx, seed_idx) correctly."""
        seed_replicates = 5
        param_cases = 10

        # Test a few indices
        test_cases = [
            (0, 0, 0),  # First run
            (4, 0, 4),  # Last seed of first param
            (5, 1, 0),  # First seed of second param
            (49, 9, 4),  # Last run
        ]

        for global_idx, expected_param, expected_seed in test_cases:
            seed_idx = global_idx % seed_replicates
            param_idx = global_idx // seed_replicates

            assert param_idx == expected_param, f"global_idx={global_idx}"
            assert seed_idx == expected_seed, f"global_idx={global_idx}"


class TestContextSpecSerialization:
    """Test context spec JSON serialization and loading."""

    def test_dataclass_context_roundtrip(self, tmp_path: Path):
        """Dataclass context specs should survive JSON roundtrip."""
        original = _TestContextSpec(
            adata_file=FilePath("/path/to/data.h5ad"),
            n_neighbors=15,
        )

        # Write JSON (simulating what _write_array_spec does)
        json_path = tmp_path / "context_spec.json"
        with open(json_path, "w") as f:
            json.dump(serialize_context_spec(original), f)

        # Load (simulating what worker does)
        loaded = _load_context_spec(tmp_path)

        # Verify loaded object is identical
        assert type(loaded) == type(original)
        assert hasattr(loaded, "adata_file")
        assert hasattr(loaded, "n_neighbors")
        assert loaded.adata_file.path == "/path/to/data.h5ad"
        assert loaded.n_neighbors == 15

    def test_dict_context_roundtrip(self, tmp_path: Path):
        """Plain dict context specs should survive JSON roundtrip."""
        original = {"dataset": "/path/to/data.csv", "version": "1.0"}

        json_path = tmp_path / "context_spec.json"
        with open(json_path, "w") as f:
            json.dump(serialize_context_spec(original), f)

        loaded = _load_context_spec(tmp_path)

        assert isinstance(loaded, dict)
        assert loaded["dataset"] == "/path/to/data.csv"
        assert loaded["version"] == "1.0"

    def test_none_context_roundtrip(self, tmp_path: Path):
        """None context should survive JSON roundtrip."""
        json_path = tmp_path / "context_spec.json"
        with open(json_path, "w") as f:
            json.dump(serialize_context_spec(None), f)

        loaded = _load_context_spec(tmp_path)
        assert loaded is None

    def test_missing_json_returns_none(self, tmp_path: Path):
        """Missing JSON file should return None with warning."""
        loaded = _load_context_spec(tmp_path)
        assert loaded is None

    def test_nested_dataclass_context_roundtrip(self, tmp_path: Path):
        """Nested dataclass context specs should survive JSON roundtrip."""
        original = _OuterContext(
            name="test",
            inner=_InnerContext(output_dir=DirPath("/output"), format="parquet"),
        )

        json_path = tmp_path / "context_spec.json"
        with open(json_path, "w") as f:
            json.dump(serialize_context_spec(original), f)

        loaded = _load_context_spec(tmp_path)

        assert type(loaded) == type(original)
        assert hasattr(loaded, "name")
        assert hasattr(loaded, "inner")
        assert loaded.name == "test"
        assert type(loaded.inner) == _InnerContext
        assert loaded.inner.output_dir.path == "/output"
        assert loaded.inner.format == "parquet"

    def test_custom_class_with_methods(self, tmp_path: Path):
        """Custom dataclasses with methods should serialize via JSON."""
        # Note: Methods are preserved because we reconstruct the class
        original = _CustomContextWithMethod(value=21)

        json_path = tmp_path / "context_spec.json"
        with open(json_path, "w") as f:
            json.dump(serialize_context_spec(original), f)

        loaded = _load_context_spec(tmp_path)

        assert loaded.value == 21
        assert loaded.double() == 42  # Methods work after reconstruction!


class TestWorkerStoreResolution:
    """Test worker store locator resolution logic."""

    def test_resolve_from_dict_config(self, tmp_path: Path):
        """When spec has dict locator (from StoreConfig.to_dict), returns FileStore."""
        from metalab.store import FileStore
        from metalab.store.config import StoreConfig
        from metalab.store.file import FileStoreConfig

        # Create a dict locator like StoreConfig.to_dict() produces
        config = FileStoreConfig(root=str(tmp_path))
        store_locator = config.to_dict()

        # Resolve using the same logic as the worker
        store = StoreConfig.from_dict(store_locator).connect()

        assert isinstance(store, FileStore)
        assert store.root.resolve() == tmp_path.resolve()

    def test_resolve_from_string_locator(self, tmp_path: Path):
        """When spec has string locator, returns FileStore."""
        from metalab.store import FileStore
        from metalab.store.locator import create_store

        store = create_store(str(tmp_path))
        assert isinstance(store, FileStore)

    def test_default_filestore_when_no_locator(self, tmp_path: Path):
        """When spec lacks store_locator, falls back to work_dir as FileStore."""
        from metalab.store import FileStore
        from metalab.store.file import FileStoreConfig

        store = FileStoreConfig(root=str(tmp_path)).connect()
        assert isinstance(store, FileStore)
        assert store.root.resolve() == tmp_path.resolve()
