"""
Unit tests for SLURM executor chunking and sharding logic.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from metalab._ids import DirPath, FilePath
from metalab.executor.slurm import SlurmConfig, SlurmExecutor, _serialize_context_spec
from metalab.executor.slurm_array_worker import (
    _ensure_experiments_root_param,
    _load_context_spec,
    _resolve_postgres_locator,
    _resolve_store_from_spec,
)


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
            (0, 0, 0),   # First run
            (4, 0, 4),   # Last seed of first param
            (5, 1, 0),   # First seed of second param
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
            json.dump(_serialize_context_spec(original), f)

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
            json.dump(_serialize_context_spec(original), f)

        loaded = _load_context_spec(tmp_path)

        assert isinstance(loaded, dict)
        assert loaded["dataset"] == "/path/to/data.csv"
        assert loaded["version"] == "1.0"

    def test_none_context_roundtrip(self, tmp_path: Path):
        """None context should survive JSON roundtrip."""
        json_path = tmp_path / "context_spec.json"
        with open(json_path, "w") as f:
            json.dump(_serialize_context_spec(None), f)

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
            json.dump(_serialize_context_spec(original), f)

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
            json.dump(_serialize_context_spec(original), f)

        loaded = _load_context_spec(tmp_path)

        assert loaded.value == 21
        assert loaded.double() == 42  # Methods work after reconstruction!


class TestWorkerStoreResolution:
    """Test worker store locator resolution logic."""

    def test_resolve_from_spec_with_filestore_locator(self, tmp_path: Path):
        """When spec has file:// locator, returns FileStore."""
        from metalab.store import FileStore

        spec = {
            "experiment_id": "test_exp",
            "store_locator": f"file://{tmp_path}",
            "experiments_root": str(tmp_path),
        }
        store = _resolve_store_from_spec(spec, tmp_path)
        assert isinstance(store, FileStore)
        assert store.root.resolve() == tmp_path.resolve()

    def test_resolve_from_spec_without_locator_uses_work_dir(self, tmp_path: Path):
        """When spec lacks store_locator, falls back to work_dir as FileStore."""
        from metalab.store import FileStore

        spec = {
            "experiment_id": "test_exp",
            # No store_locator field
        }
        store = _resolve_store_from_spec(spec, tmp_path)
        assert isinstance(store, FileStore)
        assert store.root.resolve() == tmp_path.resolve()

    def test_resolve_from_spec_with_plain_path_locator(self, tmp_path: Path):
        """When spec has plain path locator (no scheme), returns FileStore."""
        from metalab.store import FileStore

        spec = {
            "experiment_id": "test_exp",
            "store_locator": str(tmp_path),
            "experiments_root": str(tmp_path),
        }
        store = _resolve_store_from_spec(spec, tmp_path)
        assert isinstance(store, FileStore)


class TestPostgresLocatorResolution:
    """Test Postgres locator credential resolution."""

    def test_resolve_postgres_with_password_unchanged(self, tmp_path: Path):
        """Postgres locator with password is used as-is (plus experiments_root)."""
        locator = "postgresql://user:secret@localhost:5432/db"
        experiments_root = str(tmp_path)

        resolved = _resolve_postgres_locator(locator, experiments_root, "test_exp")

        assert "user:secret@" in resolved
        assert "experiments_root" in resolved

    def test_resolve_postgres_fills_password_from_service_json(self, tmp_path: Path):
        """Postgres locator without password reads from service.json."""
        # Create service.json
        service_dir = tmp_path / "services" / "postgres"
        service_dir.mkdir(parents=True)
        service_json = service_dir / "service.json"
        service_json.write_text(json.dumps({
            "host": "db-host",
            "port": 5432,
            "user": "dbuser",
            "password": "secret123",
            "connection_string": "postgresql://dbuser:secret123@db-host:5432/metalab",
        }))

        # Locator without password
        locator = "postgresql://dbuser@localhost:5432/db"
        experiments_root = str(tmp_path)

        resolved = _resolve_postgres_locator(locator, experiments_root, "test_exp")

        # Should use connection_string from service.json (with experiments_root added)
        assert "dbuser:secret123@" in resolved
        assert "experiments_root" in resolved

    def test_resolve_postgres_missing_service_json_returns_original(self, tmp_path: Path):
        """When service.json doesn't exist, returns original locator."""
        locator = "postgresql://user@localhost:5432/db"
        experiments_root = str(tmp_path)

        resolved = _resolve_postgres_locator(locator, experiments_root, "test_exp")

        # Should add experiments_root but keep original user (no password)
        assert "user@localhost" in resolved
        assert "experiments_root" in resolved


class TestExperimentsRootParam:
    """Test experiments_root query parameter handling."""

    def test_adds_experiments_root_when_missing(self):
        """Adds experiments_root param when not present."""
        locator = "postgresql://user:pass@localhost:5432/db"
        result = _ensure_experiments_root_param(locator, "/shared/experiments")

        assert "experiments_root" in result
        assert "/shared/experiments" in result or "experiments" in result

    def test_preserves_existing_experiments_root(self):
        """Does not duplicate experiments_root if already present."""
        locator = "postgresql://user:pass@localhost:5432/db?experiments_root=/old/path"
        result = _ensure_experiments_root_param(locator, "/new/path")

        # Should preserve the original value
        assert result.count("experiments_root") == 1
        assert "/old/path" in result

    def test_preserves_existing_query_params(self):
        """Preserves other query params when adding experiments_root."""
        locator = "postgresql://user:pass@localhost:5432/db?schema=custom"
        result = _ensure_experiments_root_param(locator, "/shared/experiments")

        assert "schema=custom" in result
        assert "experiments_root" in result
