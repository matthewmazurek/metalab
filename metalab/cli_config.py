"""Configuration helpers for the ``metalab run`` CLI."""

from __future__ import annotations

import json
import string
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

_METALAB_KEYS = {"store", "executor", "resume", "workers", "executor_config"}


@dataclass(frozen=True)
class RunConfig:
    """Resolved execution options for ``metalab run``."""

    store: str | None = None
    executor: str = "local"
    resume: bool = True
    workers: int = 1
    executor_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedRunConfig:
    """Config file split into application and MetaLab-owned sections."""

    app_config: dict[str, Any]
    run_config: RunConfig


def load_run_config(path: str | None) -> LoadedRunConfig:
    """Load and split a run config file."""
    if path is None:
        return LoadedRunConfig(app_config={}, run_config=RunConfig())

    raw = _load_mapping(Path(path))
    metalab_raw = raw.get("metalab", {})
    if not isinstance(metalab_raw, dict):
        raise ValueError("Config key `metalab` must be a mapping when present.")

    app_config = {k: v for k, v in raw.items() if k != "metalab"}
    metalab_config = _interpolate_metalab_config(metalab_raw, app_config)
    run_config = run_config_from_mapping(metalab_config)
    return LoadedRunConfig(app_config=app_config, run_config=run_config)


def run_config_from_mapping(config: dict[str, Any]) -> RunConfig:
    """Build a shallowly validated ``RunConfig`` from a mapping."""
    unknown = sorted(set(config) - _METALAB_KEYS)
    if unknown:
        keys = ", ".join(unknown)
        allowed = ", ".join(sorted(_METALAB_KEYS))
        raise ValueError(f"Unknown `metalab` config key(s): {keys}. Allowed: {allowed}.")

    executor_config = config.get("executor_config", {})
    if executor_config is None:
        executor_config = {}
    if not isinstance(executor_config, dict):
        raise ValueError("Config key `metalab.executor_config` must be a mapping.")

    return RunConfig(
        store=config.get("store"),
        executor=config.get("executor", "local"),
        resume=config.get("resume", True),
        workers=config.get("workers", 1),
        executor_config=dict(executor_config),
    )


def merge_cli_overrides(
    base: RunConfig,
    *,
    store: str | None = None,
    executor: str | None = None,
    resume: bool | None = None,
    workers: int | None = None,
) -> RunConfig:
    """Apply explicit CLI values on top of config-file values."""
    updates: dict[str, Any] = {}
    if store is not None:
        updates["store"] = store
    if executor is not None:
        updates["executor"] = executor
    if resume is not None:
        updates["resume"] = resume
    if workers is not None:
        updates["workers"] = workers
    return replace(base, **updates)


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".toml":
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        elif suffix in {".yaml", ".yml"}:
            data = _load_yaml(path)
        else:
            raise ValueError(
                f"Unsupported config format {suffix!r}. Use .json, .toml, .yaml, or .yml."
            )
    except FileNotFoundError as e:
        raise ValueError(f"Config file not found: {path}") from e

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping/object.")
    return data


def _load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError(
            "YAML config requires PyYAML. Install with `uv sync --extra config` "
            "or install `metalab[config]`."
        ) from e
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _interpolate_metalab_config(value: Any, app_config: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _interpolate_string(value, app_config)
    if isinstance(value, dict):
        return {k: _interpolate_metalab_config(v, app_config) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_metalab_config(v, app_config) for v in value]
    return value


def _interpolate_string(value: str, app_config: dict[str, Any]) -> str:
    formatter = string.Formatter()
    fields = [field for _, field, _, _ in formatter.parse(value) if field]
    if not fields:
        return value

    replacements: dict[str, str] = {}
    for field_name in fields:
        if not field_name.isidentifier():
            raise ValueError(
                f"Unsupported interpolation field {{{field_name}}}. "
                "Only top-level names like {experiment_name} are supported."
            )
        if field_name not in app_config:
            raise ValueError(f"Missing interpolation value for {{{field_name}}}.")
        replacement = app_config[field_name]
        if not isinstance(replacement, str | int | float | bool):
            raise ValueError(
                f"Interpolation value {{{field_name}}} must be a top-level scalar."
            )
        replacements[field_name] = str(replacement)
    return value.format_map(replacements)
