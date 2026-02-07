"""Tests for metalab.config module."""

from __future__ import annotations

import pytest

from metalab.config import ProjectConfig, ResolvedConfig, deep_merge, find_config_file

SAMPLE_CONFIG = {
    "project": {"name": "test-project", "default_env": "local"},
    "services": {
        "postgres": {"auth_method": "scram-sha-256", "database": "metalab"},
        "atlas": {"port": 8000},
    },
    "environments": {
        "local": {"type": "local", "file_root": "./runs"},
        "slurm": {
            "type": "slurm",
            "gateway": "hpc.example.com",
            "user": "testuser",
            "partition": "gpu",
            "file_root": "/shared/experiments",
        },
    },
}


# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_basic(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99, "e": 5}}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": {"c": 99, "d": 3, "e": 5}}

    def test_does_not_mutate_base(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}}
        deep_merge(base, override)
        assert base["b"]["c"] == 2

    def test_does_not_mutate_override(self):
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        deep_merge(base, override)
        assert override == {"a": {"y": 2}}

    def test_override_wins_for_leaf(self):
        result = deep_merge({"k": "old"}, {"k": "new"})
        assert result["k"] == "new"

    def test_disjoint_keys(self):
        result = deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_empty_base(self):
        result = deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override(self):
        result = deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_nested_three_levels(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = deep_merge(base, override)
        assert result == {"a": {"b": {"c": 99, "d": 2}}}

    def test_override_dict_with_scalar(self):
        """Override replaces a dict with a scalar when types differ."""
        result = deep_merge({"a": {"nested": 1}}, {"a": "scalar"})
        assert result["a"] == "scalar"


# ---------------------------------------------------------------------------
# ProjectConfig.from_dict
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_basic(self):
        config = ProjectConfig.from_dict(SAMPLE_CONFIG)
        assert config.project.name == "test-project"
        assert config.project.default_env == "local"

    def test_environments_parsed(self):
        config = ProjectConfig.from_dict(SAMPLE_CONFIG)
        assert "local" in config.environments
        assert "slurm" in config.environments
        assert config.environments["slurm"].type == "slurm"
        assert config.environments["local"].type == "local"

    def test_environment_config_excludes_type(self):
        config = ProjectConfig.from_dict(SAMPLE_CONFIG)
        # 'type' is popped from the config dict
        assert "type" not in config.environments["slurm"].config
        assert config.environments["slurm"].config["gateway"] == "hpc.example.com"

    def test_services_parsed(self):
        config = ProjectConfig.from_dict(SAMPLE_CONFIG)
        assert "postgres" in config.services
        assert config.services["postgres"]["database"] == "metalab"

    def test_missing_project_section(self):
        config = ProjectConfig.from_dict({"environments": {"dev": {"type": "local"}}})
        assert config.project.name == ""
        assert config.project.default_env is None

    def test_empty_dict(self):
        config = ProjectConfig.from_dict({})
        assert config.project.name == ""
        assert config.environments == {}
        assert config.services == {}


# ---------------------------------------------------------------------------
# ProjectConfig.resolve
# ---------------------------------------------------------------------------


class TestResolve:
    def test_default_env(self):
        config = ProjectConfig.from_dict(SAMPLE_CONFIG)
        resolved = config.resolve()  # should use default_env "local"
        assert resolved.env_name == "local"
        assert resolved.env_type == "local"
        assert resolved.file_root == "./runs"

    def test_explicit_env(self):
        config = ProjectConfig.from_dict(SAMPLE_CONFIG)
        resolved = config.resolve("slurm")
        assert resolved.env_name == "slurm"
        assert resolved.env_type == "slurm"
        assert resolved.file_root == "/shared/experiments"
        assert resolved.env_config["gateway"] == "hpc.example.com"

    def test_unknown_env_raises(self):
        config = ProjectConfig.from_dict(SAMPLE_CONFIG)
        with pytest.raises(ValueError, match="Unknown environment"):
            config.resolve("nonexistent")

    def test_no_default_raises(self):
        data = {**SAMPLE_CONFIG, "project": {"name": "test"}}  # no default_env
        config = ProjectConfig.from_dict(data)
        with pytest.raises(ValueError, match="No environment specified"):
            config.resolve()

    def test_services_merged_into_resolved(self):
        config = ProjectConfig.from_dict(SAMPLE_CONFIG)
        resolved = config.resolve("local")
        assert resolved.services["postgres"]["database"] == "metalab"
        assert resolved.services["atlas"]["port"] == 8000

    def test_file_root_popped_from_env_config(self):
        config = ProjectConfig.from_dict(SAMPLE_CONFIG)
        resolved = config.resolve("slurm")
        # file_root is extracted to a top-level field, not in env_config
        assert "file_root" not in resolved.env_config
        assert resolved.file_root == "/shared/experiments"


# ---------------------------------------------------------------------------
# Local overrides
# ---------------------------------------------------------------------------


class TestLocalOverrides:
    def test_environment_override(self):
        local = {"environments": {"slurm": {"user": "override-user"}}}
        config = ProjectConfig.from_dict(SAMPLE_CONFIG, local_overrides=local)
        resolved = config.resolve("slurm")
        assert resolved.env_config["user"] == "override-user"
        assert resolved.env_config["gateway"] == "hpc.example.com"  # preserved

    def test_service_override(self):
        local = {"services": {"postgres": {"port": 5433}}}
        config = ProjectConfig.from_dict(SAMPLE_CONFIG, local_overrides=local)
        resolved = config.resolve("local")
        assert resolved.services["postgres"]["port"] == 5433
        assert resolved.services["postgres"]["auth_method"] == "scram-sha-256"

    def test_local_adds_new_service(self):
        local = {"services": {"redis": {"port": 6379}}}
        config = ProjectConfig.from_dict(SAMPLE_CONFIG, local_overrides=local)
        resolved = config.resolve("local")
        assert resolved.services["redis"]["port"] == 6379

    def test_local_project_override(self):
        local = {"project": {"name": "overridden-name"}}
        config = ProjectConfig.from_dict(SAMPLE_CONFIG, local_overrides=local)
        resolved = config.resolve("local")
        assert resolved.project.name == "overridden-name"


# ---------------------------------------------------------------------------
# find_config_file
# ---------------------------------------------------------------------------


class TestFindConfigFile:
    def test_found_in_parent(self, tmp_path):
        config_file = tmp_path / ".metalab.toml"
        config_file.write_text('[project]\nname = "test"\n')
        subdir = tmp_path / "a" / "b" / "c"
        subdir.mkdir(parents=True)
        found = find_config_file(subdir)
        assert found == config_file

    def test_found_in_start_dir(self, tmp_path):
        config_file = tmp_path / ".metalab.toml"
        config_file.write_text('[project]\nname = "test"\n')
        found = find_config_file(tmp_path)
        assert found == config_file

    def test_not_found(self, tmp_path):
        found = find_config_file(tmp_path)
        assert found is None


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


class TestQueries:
    def test_list_environments(self):
        config = ProjectConfig.from_dict(SAMPLE_CONFIG)
        envs = config.list_environments()
        assert envs == ["local", "slurm"]

    def test_list_environments_empty(self):
        config = ProjectConfig.from_dict({})
        assert config.list_environments() == []


# ---------------------------------------------------------------------------
# ResolvedConfig helpers
# ---------------------------------------------------------------------------


class TestResolvedConfigHelpers:
    def test_has_service(self):
        config = ProjectConfig.from_dict(SAMPLE_CONFIG)
        resolved = config.resolve("local")
        assert resolved.has_service("postgres")
        assert resolved.has_service("atlas")
        assert not resolved.has_service("redis")

    def test_get_service(self):
        config = ProjectConfig.from_dict(SAMPLE_CONFIG)
        resolved = config.resolve("local")
        pg = resolved.get_service("postgres")
        assert pg["database"] == "metalab"

    def test_get_service_missing_raises(self):
        config = ProjectConfig.from_dict(SAMPLE_CONFIG)
        resolved = config.resolve("local")
        with pytest.raises(KeyError, match="No service 'redis'"):
            resolved.get_service("redis")
