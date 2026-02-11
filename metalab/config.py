"""
ProjectConfig: Project-level configuration loader for metalab.

This module provides:

- find_config_file: Walk up directories to locate .metalab.toml
- deep_merge: Recursively merge two dicts (override wins for leaf values)
- ProjectInfo: Typed project metadata
- EnvironmentProfile: A named environment profile (local, slurm, etc.)
- ResolvedConfig: Fully resolved config for a specific environment
- ProjectConfig: Main config object with load/resolve interface

Configuration is loaded from `.metalab.toml` with optional `.metalab.local.toml`
overrides. The resolution order is:

    base sections → named environment profile → local overrides

Example:
    >>> config = ProjectConfig.load()
    >>> resolved = config.resolve("slurm")
    >>> resolved.env_type
    'slurm'
    >>> resolved.get_service("postgres")
    {'auth_method': 'scram-sha-256', 'database': 'metalab'}
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_FILENAME = ".metalab.toml"
LOCAL_CONFIG_FILENAME = ".metalab.local.toml"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def find_config_file(start_dir: Path | None = None) -> Path | None:
    """
    Walk up from *start_dir* to find `.metalab.toml`.

    Starts at *start_dir* (default: current working directory) and checks each
    ancestor directory until the filesystem root is reached.

    Args:
        start_dir: Directory to start searching from. Defaults to ``Path.cwd()``.

    Returns:
        Path to the config file, or ``None`` if not found.
    """
    current = (start_dir or Path.cwd()).resolve()

    while True:
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            return candidate

        parent = current.parent
        if parent == current:
            # Reached filesystem root
            return None
        current = parent


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep-merge two dicts. *override* wins for leaf values.

    For keys present in both dicts:
    - If both values are dicts, recurse.
    - Otherwise the *override* value wins.

    Neither input is mutated; a new dict is returned.

    Args:
        base: The base dictionary.
        override: The override dictionary whose values take precedence.

    Returns:
        A new merged dictionary.
    """
    merged: dict[str, Any] = {}

    for key in base.keys() | override.keys():
        if key in base and key in override:
            base_val = base[key]
            over_val = override[key]
            if isinstance(base_val, dict) and isinstance(over_val, dict):
                merged[key] = deep_merge(base_val, over_val)
            else:
                merged[key] = over_val
        elif key in base:
            merged[key] = base[key]
        else:
            merged[key] = override[key]

    return merged


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectInfo:
    """
    Typed project metadata from the ``[project]`` table.

    Attributes:
        name: Human-readable project name.
        default_env: Default environment profile to resolve when none is
            specified explicitly.
    """

    name: str
    default_env: str | None = None


@dataclass(frozen=True)
class EnvironmentProfile:
    """
    A named environment profile from ``[environments.NAME]``.

    The ``type`` field identifies the execution backend (``"local"``,
    ``"slurm"``, ``"kubernetes"``, etc.).  All other keys from the TOML
    section are stored in ``config``.

    Attributes:
        name: Profile name (the TOML key under ``[environments]``).
        type: Backend type identifier.
        config: All remaining key/value pairs from the profile section.
    """

    name: str
    type: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedConfig:
    """
    Fully resolved config for a specific environment.

    Produced by :meth:`ProjectConfig.resolve`.  All base, profile, and local
    override layers have been merged.

    Attributes:
        project: Project metadata.
        env_name: The resolved environment profile name.
        env_type: Backend type (``"local"``, ``"slurm"``, etc.).
        env_config: Merged environment-specific configuration.
        services: Merged service configurations keyed by service name.
        file_root: Filesystem root for run artifacts, if specified.
    """

    project: ProjectInfo
    env_name: str
    env_type: str
    env_config: dict[str, Any]
    services: dict[str, dict[str, Any]]
    file_root: str | None

    @property
    def services_resources(self) -> dict[str, Any]:
        """Scheduler-specific resources for the services node.

        Returns the ``services`` sub-table from the environment config
        (e.g., ``[environments.slurm.services]``).  Returns an empty dict
        when no sub-table is present (typical for local environments).
        """
        return self.env_config.get("services", {})

    @property
    def executor_resources(self) -> dict[str, Any]:
        """Scheduler-specific resources for experiment jobs.

        Returns the ``executor`` sub-table from the environment config
        (e.g., ``[environments.slurm.executor]``).  Returns an empty dict
        when no sub-table is present (typical for local environments).
        """
        return self.env_config.get("executor", {})

    def has_service(self, name: str) -> bool:
        """Return ``True`` if a service named *name* is configured."""
        return name in self.services

    def get_service(self, name: str) -> dict[str, Any]:
        """
        Return the config dict for the service named *name*.

        Args:
            name: Service name (e.g. ``"postgres"``).

        Returns:
            The service configuration dict.

        Raises:
            KeyError: If no service with that name exists.
        """
        try:
            return self.services[name]
        except KeyError:
            available = ", ".join(sorted(self.services)) or "(none)"
            raise KeyError(
                f"No service {name!r} in config. Available services: {available}"
            ) from None


@dataclass
class ProjectConfig:
    """
    Main project configuration loaded from ``.metalab.toml``.

    Holds all parsed sections (project metadata, environments, services) and
    any local overrides from ``.metalab.local.toml``.  Use :meth:`resolve` to
    produce a :class:`ResolvedConfig` for a specific environment.

    Typical usage::

        config = ProjectConfig.load()
        resolved = config.resolve()          # uses default_env
        resolved = config.resolve("slurm")   # explicit environment
    """

    project: ProjectInfo
    environments: dict[str, EnvironmentProfile]
    services: dict[str, dict[str, Any]]
    _local_overrides: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, start_dir: Path | None = None) -> ProjectConfig:
        """
        Find and load project configuration.

        Walks up from *start_dir* (default: cwd) to locate ``.metalab.toml``,
        parses it, and optionally deep-merges ``.metalab.local.toml`` from the
        same directory.

        Args:
            start_dir: Directory to start searching from.

        Returns:
            A fully-constructed :class:`ProjectConfig`.

        Raises:
            FileNotFoundError: If no ``.metalab.toml`` is found.
        """
        config_path = find_config_file(start_dir)
        if config_path is None:
            raise FileNotFoundError(
                f"Could not find {CONFIG_FILENAME} in {start_dir or Path.cwd()} "
                f"or any parent directory"
            )

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        local_overrides: dict[str, Any] = {}
        local_path = config_path.parent / LOCAL_CONFIG_FILENAME
        if local_path.is_file():
            with open(local_path, "rb") as f:
                local_overrides = tomllib.load(f)

        return cls.from_dict(data, local_overrides=local_overrides)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        local_overrides: dict[str, Any] | None = None,
    ) -> ProjectConfig:
        """
        Create a :class:`ProjectConfig` from a parsed TOML dict.

        This is the canonical constructor used by :meth:`load` and useful for
        testing without touching the filesystem.

        Args:
            data: Parsed TOML data (from the base config file).
            local_overrides: Optional parsed TOML data from the local override
                file. These are stored and applied during :meth:`resolve`.

        Returns:
            A fully-constructed :class:`ProjectConfig`.
        """
        local_overrides = local_overrides or {}

        # -- project --
        project_raw = data.get("project", {})
        project = ProjectInfo(
            name=project_raw.get("name", ""),
            default_env=project_raw.get("default_env"),
        )

        # -- environments --
        environments: dict[str, EnvironmentProfile] = {}
        for env_name, env_raw in data.get("environments", {}).items():
            env_raw = dict(env_raw)  # shallow copy so we can pop
            env_type = env_raw.pop("type", "unknown")
            environments[env_name] = EnvironmentProfile(
                name=env_name,
                type=env_type,
                config=env_raw,
            )

        # -- services --
        services: dict[str, dict[str, Any]] = {
            name: dict(svc) for name, svc in data.get("services", {}).items()
        }

        return cls(
            project=project,
            environments=environments,
            services=services,
            _local_overrides=local_overrides,
        )

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, env_name: str | None = None) -> ResolvedConfig:
        """
        Resolve a named environment profile into a flat config.

        Merging order:

        1. Base ``[services]`` section
        2. Named environment profile
        3. Local overrides for that environment (and services)

        If *env_name* is ``None``, uses ``project.default_env`` (including
        local overrides from ``.metalab.local.toml``).

        Args:
            env_name: Environment profile name, or ``None`` for default.

        Returns:
            A :class:`ResolvedConfig` with all layers merged.

        Raises:
            ValueError: If no environment name can be determined, or the
                requested environment does not exist.
        """
        # Determine which environment to resolve.
        # Use merged project.default_env (base + local overrides) so that
        # .metalab.local.toml can override default_env for local development.
        local_project = self._local_overrides.get("project", {})
        effective_default_env = local_project.get(
            "default_env", self.project.default_env
        )
        env_name = env_name or effective_default_env
        if env_name is None:
            available = ", ".join(sorted(self.environments)) or "(none)"
            raise ValueError(
                "No environment specified and no default_env set in [project]. "
                f"Available environments: {available}"
            )

        if env_name not in self.environments:
            available = ", ".join(sorted(self.environments)) or "(none)"
            raise ValueError(
                f"Unknown environment {env_name!r}. "
                f"Available environments: {available}"
            )

        profile = self.environments[env_name]

        # Start with the base environment config from the profile
        env_config = dict(profile.config)

        # Apply local overrides for this environment
        local_env_overrides = self._local_overrides.get("environments", {}).get(
            env_name, {}
        )
        if local_env_overrides:
            local_env = dict(local_env_overrides)
            # "type" override goes to env_type below
            local_env.pop("type", None)
            env_config = deep_merge(env_config, local_env)

        # Resolve env_type (local override can change it)
        env_type = local_env_overrides.get("type", profile.type)

        # Merge services: base services + local service overrides
        merged_services = {name: dict(svc) for name, svc in self.services.items()}
        local_services = self._local_overrides.get("services", {})
        for svc_name, svc_override in local_services.items():
            if svc_name in merged_services:
                merged_services[svc_name] = deep_merge(
                    merged_services[svc_name], dict(svc_override)
                )
            else:
                merged_services[svc_name] = dict(svc_override)

        # Extract file_root from the (merged) environment config
        file_root = env_config.pop("file_root", None)

        # Apply local project overrides to ProjectInfo
        local_project = self._local_overrides.get("project", {})
        project = self.project
        if local_project:
            project = ProjectInfo(
                name=local_project.get("name", project.name),
                default_env=local_project.get("default_env", project.default_env),
            )

        return ResolvedConfig(
            project=project,
            env_name=env_name,
            env_type=env_type,
            env_config=env_config,
            services=merged_services,
            file_root=file_root,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_environments(self) -> list[str]:
        """
        List all available environment profile names.

        Returns:
            Sorted list of environment names.
        """
        return sorted(self.environments)
