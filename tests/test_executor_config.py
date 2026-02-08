"""Tests for metalab.executor config, local_config, and slurm_config modules."""

from __future__ import annotations

import pytest

from metalab.executor.config import ExecutorConfigRegistry, executor_from_config

# ---------------------------------------------------------------------------
# ExecutorConfigRegistry
# ---------------------------------------------------------------------------


class TestExecutorConfigRegistry:
    def test_local_registered(self):
        # Discovered via metalab.executors entry point
        assert "local" in ExecutorConfigRegistry.types()

    def test_slurm_registered(self):
        # Discovered via metalab.executors entry point (no manual import needed)
        assert "slurm" in ExecutorConfigRegistry.types()


# ---------------------------------------------------------------------------
# executor_from_config
# ---------------------------------------------------------------------------


class TestExecutorFromConfig:
    def test_local_serial(self):
        result = executor_from_config("local", {"workers": 1})
        assert result is None  # None means serial execution

    def test_local_parallel(self):
        result = executor_from_config("local", {"workers": 4})
        assert result is not None
        result.shutdown()

    def test_local_default(self):
        result = executor_from_config("local", {})
        assert result is None

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown executor type"):
            executor_from_config("nonexistent", {})


# ---------------------------------------------------------------------------
# LocalExecutorConfig
# ---------------------------------------------------------------------------


class TestLocalExecutorConfig:
    def test_from_dict_defaults(self):
        from metalab.executor.local_config import LocalExecutorConfig

        config = LocalExecutorConfig.from_dict({})
        assert config.workers == 1

    def test_from_dict_custom_workers(self):
        from metalab.executor.local_config import LocalExecutorConfig

        config = LocalExecutorConfig.from_dict({"workers": 8})
        assert config.workers == 8

    def test_create_serial(self):
        from metalab.executor.local_config import LocalExecutorConfig

        config = LocalExecutorConfig.from_dict({})
        assert config.create() is None

    def test_create_parallel(self):
        from metalab.executor.local_config import LocalExecutorConfig

        config = LocalExecutorConfig.from_dict({"workers": 2})
        executor = config.create()
        assert executor is not None
        executor.shutdown()


# ---------------------------------------------------------------------------
# SlurmExecutorConfig
# ---------------------------------------------------------------------------


class TestSlurmExecutorConfig:
    def test_from_dict_basic(self):
        from metalab.executor.slurm_config import SlurmExecutorConfig

        config = SlurmExecutorConfig.from_dict(
            {
                "partition": "gpu",
                "time": "2:00:00",
                "memory": "16G",
            }
        )
        assert config.partition == "gpu"
        assert config.time == "2:00:00"
        assert config.memory == "16G"

    def test_convenience_fields_to_extra_sbatch(self):
        from metalab.executor.slurm_config import SlurmExecutorConfig

        config = SlurmExecutorConfig.from_dict(
            {
                "partition": "gpu",
                "mail_user": "test@example.com",
                "mail_type": "end,fail",
            }
        )
        assert config.extra_sbatch == {
            "mail_user": "test@example.com",
            "mail_type": "end,fail",
        }
        assert config.partition == "gpu"

    def test_string_to_list_modules(self):
        from metalab.executor.slurm_config import SlurmExecutorConfig

        config = SlurmExecutorConfig.from_dict({"modules": "python/3.11"})
        assert config.modules == ["python/3.11"]

    def test_string_to_list_setup(self):
        from metalab.executor.slurm_config import SlurmExecutorConfig

        config = SlurmExecutorConfig.from_dict({"setup": "source setup.sh"})
        assert config.setup == ["source setup.sh"]

    def test_list_modules_unchanged(self):
        from metalab.executor.slurm_config import SlurmExecutorConfig

        config = SlurmExecutorConfig.from_dict(
            {
                "modules": ["python/3.11", "cuda/12.0"],
            }
        )
        assert config.modules == ["python/3.11", "cuda/12.0"]

    def test_defaults(self):
        from metalab.executor.slurm_config import SlurmExecutorConfig

        config = SlurmExecutorConfig.from_dict({})
        assert config.partition == "default"
        assert config.time == "1:00:00"
        assert config.cpus == 1
        assert config.memory == "4G"
        assert config.gpus == 0
        assert config.modules == []
        assert config.setup == []
        assert config.extra_sbatch == {}

    def test_unknown_fields_ignored(self):
        from metalab.executor.slurm_config import SlurmExecutorConfig

        # Fields not in 'known' set are silently dropped
        config = SlurmExecutorConfig.from_dict(
            {
                "partition": "gpu",
                "unknown_field": "value",
            }
        )
        assert config.partition == "gpu"
        assert not hasattr(config, "unknown_field")
